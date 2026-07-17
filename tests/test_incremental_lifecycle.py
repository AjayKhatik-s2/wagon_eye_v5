"""Unit tests for the resumable, incremental batch lifecycle (Commits 1-3).

Runnable with pytest, or directly:  python tests/test_incremental_lifecycle.py

These tests use synthetic tracks + mocked heavy stages, so they need NO .pt
models and assert the LIFECYCLE / MANIFEST / RECONSTRUCTION-PROVENANCE /
MATERIALIZER behavior only -- never inference quality.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_WC = os.path.join(_REPO, "wagon_count")
if _WC not in sys.path:
    sys.path.insert(0, _WC)


# -----------------------------------------------------------------------------
# [C1] fusion consensus + [C10] master-id stamping  (wagon_count/global_alignment)
# -----------------------------------------------------------------------------

def test_master_id_and_fusion_consensus():
    from global_train_state import (LocalCameraTracks, GapEvent, CAMERA_RIGHT_UP,
                                     CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP)
    import global_alignment as ga

    def gap(cam, fr, fps=10.0):
        return GapEvent(track_id=fr, camera_id=cam, start_frame=fr - 1,
                        end_frame=fr + 1, confidence=0.6, hit_count=5, fps=fps)

    fps, N = 10.0, 100

    # [C10] master-only build stamps the ACTUAL master (LEFT_UP), not the constant
    master = LocalCameraTracks(camera_id=CAMERA_LEFT_UP, video_path="x", fps=fps,
                               total_frames=N, gaps=[gap(CAMERA_LEFT_UP, 30),
                                                     gap(CAMERA_LEFT_UP, 60)])
    st = ga.assemble_global_train_state(master_tracks=master, support_tracks=[],
                                        initial_classifications=[], verbose=False)
    assert st.master_camera == CAMERA_LEFT_UP
    assert st.wagons[0].supporting_cameras[0] == CAMERA_LEFT_UP

    # [C1] one support camera can NEVER recover a gap
    m2 = LocalCameraTracks(camera_id=CAMERA_RIGHT_UP, video_path="x", fps=fps,
                           total_frames=N, gaps=[gap(CAMERA_RIGHT_UP, 10),
                                                 gap(CAMERA_RIGHT_UP, 30)])
    s1 = LocalCameraTracks(camera_id=CAMERA_LEFT_UP, video_path="x", fps=fps,
                           total_frames=N, gaps=[gap(CAMERA_LEFT_UP, 50)])
    st2 = ga.assemble_global_train_state(master_tracks=m2, support_tracks=[s1],
                                         initial_classifications=[], verbose=False)
    assert len(st2.corrections_applied) == 0

    # [C1] two agreeing support cameras recover exactly one gap
    s_lt = LocalCameraTracks(camera_id=CAMERA_LEFT_UP, video_path="x", fps=fps,
                             total_frames=N, gaps=[gap(CAMERA_LEFT_UP, 50)])
    s_rt = LocalCameraTracks(camera_id=CAMERA_RIGHT_UP_TOP, video_path="x", fps=fps,
                             total_frames=N, gaps=[gap(CAMERA_RIGHT_UP_TOP, 50)])
    st3 = ga.assemble_global_train_state(master_tracks=m2, support_tracks=[s_lt, s_rt],
                                         initial_classifications=[], verbose=False)
    assert len(st3.corrections_applied) == 1


# -----------------------------------------------------------------------------
# [C12] manifest schema fail-safe + arming the support window
# -----------------------------------------------------------------------------

def test_manifest_schema_and_arming():
    from orchestrator import batch_manifest as BM
    from core import constants as C

    m = BM.BatchManifest.new(batch_key="k", train_timestamp="k")
    assert m.master_deadline and m.final_camera_deadline
    assert m.support_fusion_deadline is None            # not armed until master
    m.set_camera(BM.CameraSlot(camera_id=C.MASTER_CAMERA, s3_key="x", etag="e1"))
    assert m.support_fusion_deadline is not None         # armed on RIGHT_UP arrival

    d = m.to_dict(); d["manifest_schema_version"] = 999
    try:
        BM.BatchManifest.from_dict(d)
        assert False, "should refuse a newer schema"
    except BM.ManifestSchemaError:
        pass


# -----------------------------------------------------------------------------
# [C5][C6] materializer idempotency + safe replacement
# -----------------------------------------------------------------------------

def test_materializer_idempotency_and_safe_replace():
    from core.global_state_loader import GlobalTrainState, GlobalWagon
    from core import constants as C
    from materializer import wagon_cache_builder as WCB

    state = GlobalTrainState(total_wagons=1,
                             wagons=[GlobalWagon("GW_1", 1, 0, 10, 0.0, 1.0, "WAGON")],
                             master_fps=10.0, master_total_frames=10)
    cache = tempfile.mkdtemp()
    cam = C.CAMERA_RIGHT_UP
    folder = C.CAMERA_FOLDER[cam]
    calls = {"n": 0}

    def fake_extract(*, camera_id, video_path, state, local_fps, cache_root,
                     jpeg_quality, verbose):
        calls["n"] += 1
        d = os.path.join(cache_root, "GW_1", C.CAMERA_FOLDER[camera_id])
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "frame_000000.jpg"), "w").close()
        return camera_id, {"GW_1": 1}

    orig = WCB._extract_one_camera
    WCB._extract_one_camera = fake_extract
    try:
        vp = {cam: "dummy.mp4"}
        common = dict(state=state, video_paths=vp, per_camera_fps={cam: 10.0},
                      cache_root=cache, verbose=False)
        WCB.build_cameras(global_state_version="v1", camera_etags={cam: "e1"}, **common)
        made = os.path.join(cache, "GW_1", folder, "frame_000000.jpg")
        assert os.path.isfile(made) and calls["n"] == 1
        WCB.build_cameras(global_state_version="v1", camera_etags={cam: "e1"}, **common)
        assert calls["n"] == 1                          # same etag+version -> skip
        WCB.build_cameras(global_state_version="v1", camera_etags={cam: "e2"}, **common)
        assert calls["n"] == 2                          # etag change -> rebuild

        def boom(**k):
            raise RuntimeError("extract failed")
        WCB._extract_one_camera = boom
        WCB.build_cameras(global_state_version="v1", camera_etags={cam: "e3"}, **common)
        assert os.path.isfile(made)                     # failed rebuild preserves cache
    finally:
        WCB._extract_one_camera = orig


# -----------------------------------------------------------------------------
# state machine: seal-once, resumable restart, late-attach, finalize, partial
# -----------------------------------------------------------------------------

def _mock_stages(LR, LS, CFG):
    import core.camera_features as CF

    def fake_seal(manifest, ctx, *, master_camera):
        gsdir = os.path.join(LR.batch_root(ctx, manifest.batch_key), CFG.DIR_GLOBAL_STATE)
        os.makedirs(gsdir, exist_ok=True)
        open(os.path.join(gsdir, "global_train_state.json"), "w").write("{}")
        manifest.global_state_version = "v1"
        manifest.master_camera = master_camera
        LR._transition(manifest, LS.GLOBAL_STATE_SEALED, ctx, reason="mock")
        return True

    def fake_process(manifest, ctx, *, cameras):
        for c in cameras:
            manifest.completed_features.setdefault(c, [])
            for f in CF.features_for_camera(c):
                if f not in manifest.completed_features[c]:
                    manifest.completed_features[c].append(f)
        LR._persist(manifest, ctx)

    def fake_reports(manifest, ctx, *, final, cameras=None):
        manifest.report_revision += 1
        manifest.report_status = "FINAL" if final else "INTERIM"
        LR._persist(manifest, ctx)
        return {}

    def fake_finalize(manifest, ctx):
        partial = bool(manifest.missing_cameras())
        manifest.mark_missing_final()
        LR._finish(manifest, ctx,
                   LS.COMPLETED_PARTIAL if partial else LS.COMPLETED)

    LR.stage_seal = fake_seal
    LR.stage_process_cameras = fake_process
    LR.stage_reports = fake_reports
    LR.stage_finalize = fake_finalize


def test_state_machine_end_to_end():
    from orchestrator import lifecycle_runner as LR
    from orchestrator import batch_manifest as BM
    from core.lifecycle import LifecycleState as LS
    from core import constants as C
    from core import config as CFG

    ws = tempfile.mkdtemp()
    ctx = LR.RunContext(workspace_root=ws, recon_models_dir="x", feat_models_dir="y",
                        s3_client=None, skip_upload=True, skip_email=True, verbose=False)
    _mock_stages(LR, LS, CFG)
    past = lambda: BM.iso(datetime.now(timezone.utc) - timedelta(minutes=1))
    future = lambda: BM.iso(datetime.now(timezone.utc) + timedelta(minutes=30))

    # RIGHT_UP arrives, support window still open -> WAITING_FOR_SUPPORT
    m = BM.BatchManifest.new(batch_key="B1", train_timestamp="B1")
    m.set_camera(BM.CameraSlot(camera_id=C.CAMERA_RIGHT_UP, s3_key="r", etag="e",
                               bucket="__local__"))
    m.support_fusion_deadline = future()
    m = LR.advance(m, ctx)
    assert m.lifecycle_status == LS.WAITING_FOR_SUPPORT

    # support window expires -> seal from RIGHT_UP alone, process, wait for late
    m.support_fusion_deadline = past()
    m = LR.advance(m, ctx)
    assert m.lifecycle_status == LS.WAITING_FOR_LATE_CAMERAS
    assert m.global_state_version == "v1"
    assert m.completed_features.get("RIGHT_UP") == ["door", "ocr"]

    # restart: reload from disk, resume exactly
    m2 = BM.load_local(LR.batch_root(ctx, "B1"))
    assert m2.lifecycle_status == LS.WAITING_FOR_LATE_CAMERAS
    assert m2.global_state_version == "v1"

    # late cameras arrive -> process late -> complete -> finalize
    for cam in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
        m2.set_camera(BM.CameraSlot(camera_id=cam, s3_key=cam, etag="e",
                                    bucket="__local__"))
    m2 = LR.advance(m2, ctx)
    assert m2.lifecycle_status == LS.COMPLETED
    assert m2.terminal_status == C.BATCH_COMPLETED

    # partial closure: RIGHT_UP only, final deadline passed -> completed_partial
    mp = BM.BatchManifest.new(batch_key="B2", train_timestamp="B2")
    mp.set_camera(BM.CameraSlot(camera_id=C.CAMERA_RIGHT_UP, s3_key="r", etag="e",
                                bucket="__local__"))
    mp.support_fusion_deadline = past()
    mp.final_camera_deadline = past()
    mp = LR.advance(mp, ctx)
    assert mp.lifecycle_status == LS.COMPLETED_PARTIAL


def test_no_master_fails_safe():
    """Top-only / no-master before hard deadline waits; after it, fails safe."""
    from orchestrator import lifecycle_runner as LR
    from orchestrator import batch_manifest as BM
    from core.lifecycle import LifecycleState as LS
    from core import constants as C
    from core import config as CFG

    ws = tempfile.mkdtemp()
    ctx = LR.RunContext(workspace_root=ws, recon_models_dir="x", feat_models_dir="y",
                        s3_client=None, skip_upload=True, skip_email=True, verbose=False)
    _mock_stages(LR, LS, CFG)
    past = lambda: BM.iso(datetime.now(timezone.utc) - timedelta(minutes=1))

    m = BM.BatchManifest.new(batch_key="B3", train_timestamp="B3")
    m.set_camera(BM.CameraSlot(camera_id=C.CAMERA_RIGHT_UP_TOP, s3_key="t", etag="e",
                               bucket="__local__"))
    m.master_deadline = past()
    m.final_camera_deadline = past()      # hard close, still no side master
    m = LR.advance(m, ctx)
    assert m.lifecycle_status == LS.FAILED_NO_GLOBAL_STATE
    assert m.global_state_version is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
