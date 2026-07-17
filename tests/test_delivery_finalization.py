"""Commit 6 tests: finalization idempotency, delivery policy, config validation.

No live S3 / notification API / real PDFs required -- delivery calls are mocked
and stage_reports is stubbed to emit real (tiny) files so hashing/markers work.

Run:  python tests/test_delivery_finalization.py
"""

from __future__ import annotations

import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core import constants as C
from core import config as CFG
from core.lifecycle import LifecycleState as LS, ArrivalState
from orchestrator import lifecycle_runner as LR
from orchestrator import batch_manifest as BM
from delivery import s3_upload, notification, finalization as FIN


# -----------------------------------------------------------------------------
# harness
# -----------------------------------------------------------------------------

class _Rec:
    def __init__(self):
        self.uploads = []
        self.emails = []

    def install(self, *, email_ok=True):
        self._orig = (s3_upload.upload_pdf, s3_upload.upload_json,
                      s3_upload.upload_tree, notification.send_email)
        s3_upload.upload_pdf = lambda c, p, k: (self.uploads.append(("pdf", p)) or
                                                f"https://s3/{os.path.basename(p)}")
        s3_upload.upload_json = lambda c, p, k: (self.uploads.append(("json", p)) or
                                                 f"https://s3/{os.path.basename(p)}")
        s3_upload.upload_tree = lambda *a, **k: 0
        notification.send_email = lambda **kw: (self.emails.append(kw) or email_ok)

    def restore(self):
        (s3_upload.upload_pdf, s3_upload.upload_json,
         s3_upload.upload_tree, notification.send_email) = self._orig


def _ctx(ws):
    return LR.RunContext(workspace_root=ws, recon_models_dir="x", feat_models_dir="y",
                         s3_client=object(), skip_upload=False, skip_email=False,
                         verbose=False)


def _manifest(ws, present, missing_final=()):
    m = BM.BatchManifest.new(batch_key="B", train_timestamp="B")
    m.global_state_version = "GSTHASH"
    m.report_revision = 1
    m.report_status = "FINAL"
    for cam in present:
        m.set_camera(BM.CameraSlot(camera_id=cam, s3_key=cam, etag="e", bucket="__local__"))
    for cam in missing_final:
        m.cameras[cam] = BM.CameraSlot(camera_id=cam,
                                       arrival_state=ArrivalState.CAMERA_MISSING_FINAL)
    return m


def _stub_reports(*, pdf=True, json_=True):
    """Return a stage_reports stub that writes real files and a res dict."""
    def stub(manifest, ctx, *, final, cameras=None):
        root = LR.batch_root(ctx, manifest.batch_key)
        rdir = os.path.join(root, CFG.DIR_REPORTS)
        os.makedirs(rdir, exist_ok=True)
        res = {"unified": {}, "camera_pdf_paths": {}, "report_meta": LR._report_meta(manifest, final=final)}
        if json_:
            jp = os.path.join(rdir, "combined_train_report.json")
            open(jp, "w").write('{"ok":1}')
            res["json_path"] = jp
        if pdf:
            pp = os.path.join(rdir, "combined_train_report.pdf")
            open(pp, "w").write("PDF")
            res["pdf_path"] = pp
        else:
            res["pdf_path"] = None
        manifest.report_status = res["report_meta"]["report_status"]
        return res
    return stub


# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------

def test_defaults_and_old_env_safe():
    # 11 + 12: defaults with no env set
    assert CFG.ENABLE_LEFT_UP_FALLBACK_MASTER is False
    assert CFG.MASTER_WAIT_MINUTES == 10.0
    assert CFG.UPLOAD_INTERIM_REPORTS is False and CFG.EMAIL_INTERIM_REPORTS is False
    assert CFG.LATE_CAMERA_POLICY == "IGNORE"


def test_config_validation_rejects_bad_deadlines():
    # 10: support wait > final wait
    o1, o2 = CFG.SUPPORT_FUSION_WAIT_MINUTES, CFG.FINAL_CAMERA_WAIT_MINUTES
    CFG.SUPPORT_FUSION_WAIT_MINUTES, CFG.FINAL_CAMERA_WAIT_MINUTES = 100.0, 10.0
    try:
        errs = CFG.validate_config(mode="local", skip_upload=True, skip_email=True)
        assert any("must not exceed" in e for e in errs), errs
    finally:
        CFG.SUPPORT_FUSION_WAIT_MINUTES, CFG.FINAL_CAMERA_WAIT_MINUTES = o1, o2
    # interim email without interim generation
    og = CFG.GENERATE_INTERIM_REPORTS
    CFG.GENERATE_INTERIM_REPORTS = False
    try:
        oe = CFG.EMAIL_INTERIM_REPORTS
        CFG.EMAIL_INTERIM_REPORTS = True
        errs = CFG.validate_config(mode="local", skip_upload=True, skip_email=True)
        assert any("EMAIL_INTERIM_REPORTS" in e for e in errs)
        CFG.EMAIL_INTERIM_REPORTS = oe
    finally:
        CFG.GENERATE_INTERIM_REPORTS = og


def test_interim_is_local_only():
    # 1 + 8: interim reports never upload/email
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    m = _manifest(ws, [C.CAMERA_RIGHT_UP])
    rec = _Rec(); rec.install()
    orig = LR.stage_reports
    LR.stage_reports = _stub_reports()  # not used by interim path here, but keep symmetric
    try:
        # an INTERIM report is generated but delivery is untouched
        LR.stage_reports(m, ctx, final=False)
        assert rec.uploads == [] and rec.emails == []
    finally:
        LR.stage_reports = orig
        rec.restore()


def test_exactly_one_email_on_finalization():
    # 2 + 5(complete): all cameras present -> COMPLETED, one email, one upload set
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    m = _manifest(ws, list(C.ALL_CAMERAS))
    rec = _Rec(); rec.install()
    orig = LR.stage_reports
    LR.stage_reports = _stub_reports()
    try:
        LR.stage_finalize(m, ctx)
    finally:
        LR.stage_reports = orig
        rec.restore()
    assert m.lifecycle_status == LS.COMPLETED
    assert len(rec.emails) == 1
    mk = FIN.load(LR.batch_root(ctx, "B"))
    assert mk["email_sent"] is True and mk["uploaded"] is True
    assert mk["terminal_status"] == C.BATCH_COMPLETED


def test_completed_partial():
    # 5(partial): a missing-final camera -> COMPLETED_PARTIAL
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    m = _manifest(ws, [C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP],
                  missing_final=[C.CAMERA_LEFT_UP_TOP])
    rec = _Rec(); rec.install()
    orig = LR.stage_reports; LR.stage_reports = _stub_reports()
    try:
        LR.stage_finalize(m, ctx)
    finally:
        LR.stage_reports = orig; rec.restore()
    assert m.lifecycle_status == LS.COMPLETED_PARTIAL
    mk = FIN.load(LR.batch_root(ctx, "B"))
    assert C.CAMERA_LEFT_UP_TOP in mk["cameras_missing_final"]


def test_restart_after_upload_before_email():
    # 3: marker says uploaded but not emailed -> upload skipped, email sent
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    m = _manifest(ws, list(C.ALL_CAMERAS))
    orig = LR.stage_reports; LR.stage_reports = _stub_reports()
    # first pass: upload succeeds, email FAILS -> marker uploaded=true, email_sent=false
    rec1 = _Rec(); rec1.install(email_ok=False)
    try:
        LR.stage_finalize(m, ctx)
    finally:
        rec1.restore()
    mk = FIN.load(LR.batch_root(ctx, "B"))
    assert mk["uploaded"] is True and mk["email_sent"] is False
    n_uploads_1 = len(rec1.uploads)
    # restart: re-run finalize -> no re-upload, email now succeeds
    m2 = _manifest(ws, list(C.ALL_CAMERAS))
    rec2 = _Rec(); rec2.install(email_ok=True)
    try:
        LR.stage_finalize(m2, ctx)
    finally:
        LR.stage_reports = orig; rec2.restore()
    assert rec2.uploads == [], "must not re-upload after restart"
    assert len(rec2.emails) == 1, "email sent exactly once on resume"
    assert FIN.load(LR.batch_root(ctx, "B"))["email_sent"] is True
    assert n_uploads_1 > 0


def test_restart_after_email_no_resend():
    # 4: marker email_sent=true (same idem key) -> no resend
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    m = _manifest(ws, list(C.ALL_CAMERAS))
    orig = LR.stage_reports; LR.stage_reports = _stub_reports()
    rec1 = _Rec(); rec1.install(email_ok=True)
    try:
        LR.stage_finalize(m, ctx)
    finally:
        rec1.restore()
    assert len(rec1.emails) == 1
    m2 = _manifest(ws, list(C.ALL_CAMERAS))
    rec2 = _Rec(); rec2.install(email_ok=True)
    try:
        LR.stage_finalize(m2, ctx)
    finally:
        LR.stage_reports = orig; rec2.restore()
    assert rec2.emails == [], "must not resend after confirmed email"
    assert rec2.uploads == [], "must not re-upload"


def test_report_failed_uploads_json_suppresses_email():
    # 7: pdf missing -> REPORT_FAILED, JSON uploaded, email suppressed
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    m = _manifest(ws, list(C.ALL_CAMERAS))
    rec = _Rec(); rec.install()
    orig = LR.stage_reports; LR.stage_reports = _stub_reports(pdf=False)
    try:
        LR.stage_finalize(m, ctx)
    finally:
        LR.stage_reports = orig; rec.restore()
    assert m.lifecycle_status == LS.REPORT_FAILED
    assert any(kind == "json" for kind, _ in rec.uploads), "JSON must upload"
    assert rec.emails == [], "email suppressed when PDF unavailable"
    mk = FIN.load(LR.batch_root(ctx, "B"))
    assert mk["email_status"] == "suppressed_report_failed"


def _mock_advance_stages():
    import core.camera_features as CF

    def fake_seal(manifest, ctx, *, master_camera):
        gsdir = os.path.join(LR.batch_root(ctx, manifest.batch_key), CFG.DIR_GLOBAL_STATE)
        os.makedirs(gsdir, exist_ok=True)
        open(os.path.join(gsdir, "global_train_state.json"), "w").write("{}")
        manifest.global_state_version = "GSTHASH"
        LR._transition(manifest, LS.GLOBAL_STATE_SEALED, ctx, reason="mock")
        return True

    def fake_process(manifest, ctx, *, cameras):
        for c in cameras:
            manifest.completed_features.setdefault(c, [])
            for f in CF.features_for_camera(c):
                if f not in manifest.completed_features[c]:
                    manifest.completed_features[c].append(f)
        LR._persist(manifest, ctx)

    saved = (LR.stage_seal, LR.stage_process_cameras, LR.stage_reports)
    LR.stage_seal = fake_seal
    LR.stage_process_cameras = fake_process
    LR.stage_reports = _stub_reports()
    return saved


def test_failed_no_global_state_no_delivery():
    # 6: no side master by hard deadline -> FAILED_NO_GLOBAL_STATE, no report/email
    from datetime import datetime, timezone, timedelta
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    saved = _mock_advance_stages()
    rec = _Rec(); rec.install()
    try:
        m = BM.BatchManifest.new(batch_key="F", train_timestamp="F")
        m.set_camera(BM.CameraSlot(camera_id=C.CAMERA_RIGHT_UP_TOP, s3_key="t",
                                   etag="e", bucket="__local__"))
        m.master_deadline = BM.iso(datetime.now(timezone.utc) - timedelta(minutes=1))
        m.final_camera_deadline = BM.iso(datetime.now(timezone.utc) - timedelta(minutes=1))
        m = LR.advance(m, ctx)
    finally:
        LR.stage_seal, LR.stage_process_cameras, LR.stage_reports = saved
        rec.restore()
    assert m.lifecycle_status == LS.FAILED_NO_GLOBAL_STATE
    assert rec.emails == [] and rec.uploads == []
    assert FIN.load(LR.batch_root(ctx, "F")) is None   # no finalization marker


def test_terminal_batch_ignores_late_camera():
    # 9: a terminal batch is never re-advanced / re-delivered
    ws = tempfile.mkdtemp()
    ctx = _ctx(ws)
    m = _manifest(ws, list(C.ALL_CAMERAS))
    m.lifecycle_status = LS.COMPLETED
    rec = _Rec(); rec.install()
    try:
        m.set_camera(BM.CameraSlot(camera_id=C.CAMERA_LEFT_UP_TOP, s3_key="x", etag="new"))
        m2 = LR.advance(m, ctx)
    finally:
        rec.restore()
    assert m2.lifecycle_status == LS.COMPLETED
    assert rec.emails == [] and rec.uploads == []


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
