"""Resumable, incremental batch lifecycle driver.

`advance(manifest, ctx)` is a pure state machine over a BatchManifest: each call
makes as much forward progress as it safely can, persisting the manifest
atomically after every transition, and returns when the batch is either waiting
on a deadline / late camera, or terminal.  Every action is guarded by persisted
artifacts + markers so a restart resumes exactly where it left off and a
duplicate poll repeats no work.

The heavy stages (reconstruction / materialize / features / fusion / render /
report / deliver) are thin wrappers around the existing stage modules; the
per-camera idempotency + subset behavior is layered in by later commits (the
call sites here already pass the present-camera subset).

Key invariants enforced here:
  * GlobalTrainState is sealed exactly once.  Once `global_state_version` is
    set, RECONSTRUCTING is never re-entered -- late cameras only attach.
  * A late RIGHT_UP arriving after a LEFT_UP fallback seal contributes ONLY its
    right-door + OCR features; its classification + gaps are ignored (the sealed
    GST cannot change).  [C4]
  * processed_batches.json is written elsewhere and holds ONLY terminal batches.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from core import config as CFG
from core import constants as C
from core import camera_features as CF
from core.feature_config import FeatureConfig
from core.lifecycle import ArrivalState, LifecycleState, is_terminal
from core.logging_setup import get_logger

from orchestrator import batch_manifest as BM
from orchestrator.batch_manifest import BatchManifest, CameraSlot

log = get_logger("lifecycle")


# -----------------------------------------------------------------------------
# Run context (shared config for one advance() invocation)
# -----------------------------------------------------------------------------

@dataclass
class RunContext:
    workspace_root: str
    recon_models_dir: str
    feat_models_dir: str
    s3_client: object = None
    feature_config: Optional[FeatureConfig] = None
    skip_upload: bool = False
    skip_email: bool = False
    verbose: bool = True
    repo_root: str = ""

    def __post_init__(self):
        if self.feature_config is None:
            self.feature_config = FeatureConfig.all_on()
        if not self.repo_root:
            self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def batch_root(ctx: RunContext, batch_key: str) -> str:
    return os.path.join(ctx.workspace_root, batch_key)


def _persist(manifest: BatchManifest, ctx: RunContext) -> None:
    root = batch_root(ctx, manifest.batch_key)
    BM.write_local(manifest, root)
    if ctx.s3_client is not None and not ctx.skip_upload:
        BM.save_s3(ctx.s3_client, manifest)


def _transition(manifest: BatchManifest, new_state: str, ctx: RunContext,
                reason: str = "") -> None:
    old = manifest.lifecycle_status
    if old == new_state:
        return
    manifest.lifecycle_status = new_state
    log.info("[LIFECYCLE %s] %s -> %s%s", manifest.batch_key, old, new_state,
             f"  ({reason})" if reason else "")
    _persist(manifest, ctx)


# -----------------------------------------------------------------------------
# Downloads (present cameras only)
# -----------------------------------------------------------------------------

def _download_present(manifest: BatchManifest, ctx: RunContext) -> Dict[str, str]:
    """Resolve local paths for every PRESENT camera, downloading from S3 if
    needed.  Idempotent: an already-downloaded file is reused."""
    root = batch_root(ctx, manifest.batch_key)
    dl_root = os.path.join(root, CFG.DIR_DOWNLOADS)
    os.makedirs(dl_root, exist_ok=True)
    paths: Dict[str, str] = {}
    for cam in manifest.present_cameras():
        slot = manifest.cameras[cam]
        if slot.local_path and os.path.exists(slot.local_path):
            paths[cam] = slot.local_path
            continue
        if slot.bucket == "__local__":
            paths[cam] = slot.s3_key
            slot.local_path = slot.s3_key
            continue
        local = os.path.join(dl_root, f"{cam}_{slot.filename or os.path.basename(slot.s3_key)}")
        if not os.path.exists(local):
            ctx.s3_client.download_file(slot.bucket, slot.s3_key, local)
        slot.local_path = local
        paths[cam] = local
    return paths


# -----------------------------------------------------------------------------
# Stage: seal (Stage 1 -- reconstruction)  [subset support arrives in Commit 2]
# -----------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_seal(manifest: BatchManifest, ctx: RunContext, *, master_camera: str) -> bool:
    """Run Stage 1 with the present cameras + chosen master.  Returns True on a
    successful seal.  Idempotent: if already sealed, returns True immediately."""
    from reconstruction import runner as reconstruction_runner

    if manifest.global_state_version:
        return True  # already sealed -- never reseal

    root = batch_root(ctx, manifest.batch_key)
    stage0_root = os.path.join(root, CFG.DIR_GLOBAL_STATE)
    os.makedirs(stage0_root, exist_ok=True)

    video_paths = _download_present(manifest, ctx)
    present = list(video_paths.keys())
    fallback = (master_camera != C.MASTER_CAMERA)
    log.info("[SEAL %s] reconstructing: master=%s present=%s fallback=%s",
             manifest.batch_key, master_camera, present, fallback)

    try:
        recon = reconstruction_runner.run(
            video_paths=video_paths,
            reconstruction_models_dir=ctx.recon_models_dir,
            output_dir=stage0_root,
            repo_root=ctx.repo_root,
            master_camera=master_camera,
            allow_fallback_master=fallback,
            verbose=ctx.verbose,
        )
    except reconstruction_runner.ReconstructionError as e:
        log.error("[SEAL %s] reconstruction failed: %s", manifest.batch_key, e)
        manifest.sealing_reason = f"reconstruction_failed: {e}"
        _transition(manifest, LifecycleState.FAILED_NO_GLOBAL_STATE, ctx, reason=str(e))
        return False

    # Seal: compute an immutable version hash + capture provenance.
    version = _sha256_file(recon.state_json_path)
    manifest.global_state_version = version
    manifest.global_state_status = "SEALED"
    manifest.master_camera = getattr(recon, "master_camera", master_camera)
    manifest.fallback_master_used = fallback
    manifest.reconstruction_mode = getattr(recon, "reconstruction_mode", None)
    manifest.support_cameras_present = getattr(recon, "support_cameras_present", [])
    manifest.support_fusion_used = getattr(recon, "support_fusion_used", False)
    manifest.support_gap_recoveries = getattr(recon, "support_gap_recoveries", 0)
    manifest.sealed_at = BM.iso(BM._now())
    manifest.sealing_reason = getattr(recon, "sealing_reason", "sealed")
    log.info("[SEAL %s] SEALED v=%s master=%s mode=%s support_present=%s "
             "fusion_used=%s recoveries=%d",
             manifest.batch_key, version[:12], manifest.master_camera,
             manifest.reconstruction_mode, manifest.support_cameras_present,
             manifest.support_fusion_used, manifest.support_gap_recoveries)
    _transition(manifest, LifecycleState.GLOBAL_STATE_SEALED, ctx, reason="sealed")
    return True


# -----------------------------------------------------------------------------
# Stage: process available cameras (materialize + features + fuse + render +
# reports).  Commit 1 reuses the complete-set builders over the present subset;
# per-camera idempotency + incremental fusion/reports land in Commits 3-5.
# -----------------------------------------------------------------------------

def _load_sealed_state(manifest: BatchManifest, ctx: RunContext):
    from core.global_state_loader import load_global_train_state
    root = batch_root(ctx, manifest.batch_key)
    p = os.path.join(root, CFG.DIR_GLOBAL_STATE, "global_train_state.json")
    return load_global_train_state(p)


def _late_master_note(manifest: BatchManifest, present: List[str]) -> None:
    """[C4] If GST was sealed via LEFT_UP fallback and RIGHT_UP is now present,
    log that its classification + gaps are ignored (sealed GST is immutable)."""
    if (manifest.fallback_master_used
            and manifest.master_camera != C.MASTER_CAMERA
            and C.MASTER_CAMERA in present):
        msg = (f"late RIGHT_UP attached after {manifest.master_camera} fallback seal: "
               f"contributes right-door+OCR only; classification/gaps IGNORED "
               f"(sealed GST v{(manifest.global_state_version or '')[:12]} immutable)")
        if msg not in manifest.notes:
            manifest.notes.append(msg)
        log.warning("[LATE-MASTER %s] %s", manifest.batch_key, msg)


def stage_process_cameras(manifest: BatchManifest, ctx: RunContext,
                          *, cameras: List[str]) -> None:
    """Materialize + run features for the given cameras, then re-fuse + render +
    interim reports.  `cameras` is the set to (re)process this pass."""
    from materializer import wagon_cache_builder
    from fusion import wagon_state_builder

    root = batch_root(ctx, manifest.batch_key)
    stage0_root   = os.path.join(root, CFG.DIR_GLOBAL_STATE)
    cache_root    = os.path.join(root, CFG.DIR_WAGON_CACHE)
    states_root   = os.path.join(root, CFG.DIR_WAGON_STATES)
    evidence_root = os.path.join(root, CFG.DIR_EVIDENCE)
    for d in (cache_root, states_root, evidence_root):
        os.makedirs(d, exist_ok=True)

    state = _load_sealed_state(manifest, ctx)
    video_paths = _download_present(manifest, ctx)
    present = list(video_paths.keys())
    _late_master_note(manifest, present)

    # only cameras we were asked to process AND that are present
    todo = [c for c in cameras if c in present]
    if not todo:
        return

    from core.global_state_loader import load_per_camera_fps
    pcf_path = os.path.join(stage0_root, "per_camera_tracking.json")
    per_camera_fps = load_per_camera_fps(pcf_path) if os.path.exists(pcf_path) else {}

    # --- Stage 2: materialize just the requested cameras (idempotent) ---
    subset_paths = {c: video_paths[c] for c in todo}
    camera_etags = {c: manifest.cameras[c].etag for c in todo if c in manifest.cameras}
    wagon_cache_builder.build_cameras(
        state=state, video_paths=subset_paths,
        per_camera_fps=per_camera_fps, cache_root=cache_root,
        global_state_version=manifest.global_state_version or "",
        camera_etags=camera_etags, verbose=ctx.verbose,
    )
    for c in todo:
        if c not in manifest.materialized_cameras:
            manifest.materialized_cameras.append(c)

    # --- Stage 3: run each requested camera's registered features ---
    _run_camera_features(manifest, ctx, cameras=todo, state=state,
                         cache_root=cache_root, states_root=states_root,
                         evidence_root=evidence_root)

    # --- Stage 4: re-fuse from all currently available per-camera results ---
    manifest.fusion_revision += 1
    wagon_state_builder.build(
        state=state, wagon_states_root=states_root,
        camera_arrival=_camera_arrival(manifest),
        disabled_features=_disabled_features(ctx),
        global_state_version=manifest.global_state_version or "",
        fusion_revision=manifest.fusion_revision,
        verbose=ctx.verbose,
    )
    _persist(manifest, ctx)


def _camera_arrival(manifest: BatchManifest) -> Dict[str, str]:
    """Map each camera to its arrival state for fusion: PRESENT / PENDING /
    CAMERA_MISSING_FINAL (only after final closure marks it)."""
    out: Dict[str, str] = {}
    for cam in C.ALL_CAMERAS:
        slot = manifest.cameras.get(cam)
        out[cam] = (slot.arrival_state if slot else ArrivalState.PENDING_CAMERA)
    return out


def _disabled_features(ctx: RunContext) -> set:
    return set(ctx.feature_config.disabled_keys())


def _run_camera_features(manifest: BatchManifest, ctx: RunContext, *,
                         cameras: List[str], state, cache_root: str,
                         states_root: str, evidence_root: str) -> None:
    """Run each camera's registered features (load-before-damage per camera),
    writing ONLY that camera's per-camera namespace.  A (camera, feature) is
    skipped when its completion marker matches the current identity (ETag +
    GST version + model hash + processor schema + threshold hash), so a late
    camera never re-runs, and never overwrites, another camera's results."""
    from features.door   import processor as door_proc
    from features.load   import processor as load_proc
    from features.damage import processor as damage_proc
    from features.ocr    import processor as ocr_proc
    from orchestrator import feature_markers as FM

    fmap = {
        CF.FEATURE_DOOR:   door_proc.run,
        CF.FEATURE_OCR:    ocr_proc.run,
        CF.FEATURE_LOAD:   load_proc.run,
        CF.FEATURE_DAMAGE: damage_proc.run,
    }
    gst_version = manifest.global_state_version or ""
    done = manifest.completed_features
    for cam in cameras:
        slot = manifest.cameras.get(cam)
        etag = slot.etag if slot else None
        source_key = slot.s3_key if slot else None
        feats = [f for f in CF.features_for_camera(cam)   # load before damage
                 if ctx.feature_config.is_enabled(f)]
        for feat in feats:
            identity = FM.compute_identity(
                camera_id=cam, feature=feat, source_key=source_key, etag=etag,
                global_state_version=gst_version, feat_models_dir=ctx.feat_models_dir,
            )
            if FM.is_up_to_date(states_root, identity):
                log.info("[FEATURE %s/%s/%s] up-to-date -- skip",
                         manifest.batch_key, cam, feat)
            else:
                status = "OK"
                try:
                    fmap[feat](
                        state=state, cache_root=cache_root,
                        feature_models_dir=ctx.feat_models_dir,
                        output_dir=states_root, evidence_root=evidence_root,
                        cameras=[cam], verbose=ctx.verbose,
                    )
                except Exception as e:
                    status = "FAILED"
                    log.error("[FEATURE %s/%s/%s] crashed: %s",
                              manifest.batch_key, cam, feat, e, exc_info=True)
                FM.write_marker(states_root, identity, status=status,
                                wagons_completed=len(state.wagons))
            done.setdefault(cam, [])
            if feat not in done[cam]:
                done[cam].append(feat)
    manifest.completed_features = done


# -----------------------------------------------------------------------------
# Stage: reports (interim / final) + delivery
# -----------------------------------------------------------------------------

def _report_meta(manifest: BatchManifest, *, final: bool) -> dict:
    present = manifest.present_cameras()
    missing_final = [c for c in C.ALL_CAMERAS
                     if manifest.cameras.get(c)
                     and manifest.cameras[c].arrival_state == ArrivalState.CAMERA_MISSING_FINAL]
    pending = [c for c in C.ALL_CAMERAS if c not in present and c not in missing_final]
    if final:
        status = "FINAL" if manifest.is_complete() and not missing_final else "FINAL_PARTIAL"
    else:
        status = "INTERIM"
    partial_reason = ""
    if status == "FINAL_PARTIAL":
        partial_reason = f"cameras absent at closure: {missing_final or pending}"
    gst = manifest.global_state_version or ""
    return {
        "report_revision": manifest.report_revision,
        "report_status": status,
        "cameras_present": present,
        "cameras_pending": pending,
        "cameras_missing_final": missing_final,
        "generated_from_global_state_version": gst,
        "generated_from_global_state_hash": gst,
        "fusion_revision": manifest.fusion_revision,
        "partial_reason": partial_reason,
    }


def stage_reports(manifest: BatchManifest, ctx: RunContext, *, final: bool,
                  cameras: Optional[List[str]] = None) -> Optional[dict]:
    """Regenerate overlays + camera PDFs for `cameras` (a late camera regenerates
    only its own artifacts) and always regenerate the aggregating combined
    report/JSON.  Never loads a model."""
    from reporting import combined_train_report, camera_reports
    from rendering import feature_overlay_renderer
    from fusion import wagon_state_builder

    root = batch_root(ctx, manifest.batch_key)
    stage0_root    = os.path.join(root, CFG.DIR_GLOBAL_STATE)
    cache_root     = os.path.join(root, CFG.DIR_WAGON_CACHE)
    states_root    = os.path.join(root, CFG.DIR_WAGON_STATES)
    evidence_root  = os.path.join(root, CFG.DIR_EVIDENCE)
    processed_root = os.path.join(root, CFG.DIR_PROCESSED_VIDEOS)
    reports_root   = os.path.join(root, CFG.DIR_REPORTS)
    for d in (processed_root, reports_root):
        os.makedirs(d, exist_ok=True)

    if not CFG.GENERATE_INTERIM_REPORTS and not final:
        return None

    state = _load_sealed_state(manifest, ctx)
    unified = wagon_state_builder.build(
        state=state, wagon_states_root=states_root,
        camera_arrival=_camera_arrival(manifest),
        disabled_features=_disabled_features(ctx),
        global_state_version=manifest.global_state_version or "",
        fusion_revision=manifest.fusion_revision, verbose=False)

    video_paths = _download_present(manifest, ctx)
    pcf_path = os.path.join(stage0_root, "per_camera_tracking.json")
    # Regenerate overlays ONLY for the affected cameras (default: all present).
    render_cams = cameras if cameras is not None else manifest.present_cameras()

    try:
        feature_overlay_renderer.render_all_cameras(
            state=state, unified=unified, evidence_root=evidence_root,
            video_paths=video_paths, per_camera_tracking_path=pcf_path,
            output_dir=processed_root,
            enabled_features=set(ctx.feature_config.enabled_keys()),
            cameras=render_cams, verbose=ctx.verbose,
        )
    except Exception as e:
        log.error("[RENDER %s] overlay rendering failed: %s", manifest.batch_key, e)

    # 5a: camera reports -- only the affected cameras (default: all present)
    try:
        camera_reports.build_all(
            state=state, unified=unified, evidence_root=evidence_root,
            wagon_states_root=states_root, cache_root=cache_root,
            per_camera_tracking_path=pcf_path, output_dir=reports_root,
            batch_key=manifest.batch_key, logo_path=CFG.LOGO_PATH,
            cameras=render_cams, verbose=ctx.verbose,
        )
    except Exception as e:
        log.error("[REPORT %s] camera reports failed: %s", manifest.batch_key, e)

    # All camera PDFs that exist on disk (sibling links in the combined report).
    from reporting.camera_reports import CAMERA_FILE
    camera_pdf_paths = {cam: os.path.join(reports_root, CAMERA_FILE[cam])
                        for cam in C.ALL_CAMERAS
                        if os.path.isfile(os.path.join(reports_root, CAMERA_FILE[cam]))}
    camera_pdf_urls = {cam: os.path.basename(p) for cam, p in camera_pdf_paths.items()}

    manifest.report_revision += 1
    meta = _report_meta(manifest, final=final)
    manifest.report_status = meta["report_status"]

    # 5b: combined report always regenerates (aggregates all cameras)
    result = combined_train_report.build(
        state=state, unified=unified, output_dir=reports_root,
        batch_key=manifest.batch_key,
        source_video_urls={c: manifest.cameras[c].s3_url
                           for c in manifest.present_cameras()
                           if manifest.cameras[c].bucket != "__local__"},
        processed_video_urls={},
        evidence_root=evidence_root, wagon_states_root=states_root,
        cache_root=cache_root,
        missing_cameras=(meta["cameras_missing_final"] + meta["cameras_pending"]),
        camera_pdf_urls=camera_pdf_urls,
        logo_path=CFG.LOGO_PATH, report_meta=meta, verbose=ctx.verbose,
    )
    _persist(manifest, ctx)
    return {"unified": unified, "camera_pdf_paths": camera_pdf_paths,
            "report_meta": meta, **(result or {})}


def stage_finalize(manifest: BatchManifest, ctx: RunContext) -> None:
    """One upload + one email per FINAL revision, idempotent across restarts.

    A finalization marker (delivery/finalization.json) records what was
    delivered; on re-entry each step already done for the current report
    revision is skipped -- no duplicate uploads, no duplicate email."""
    from core.unified_wagon_state import summarize_wagons
    from delivery import s3_upload, notification, finalization as FIN

    manifest.mark_missing_final()
    res = stage_reports(manifest, ctx, final=True) or {}
    unified = res.get("unified") or {}
    report_pdf_path = res.get("pdf_path")
    report_json_path = res.get("json_path")
    camera_pdf_paths = res.get("camera_pdf_paths") or {}
    meta = res.get("report_meta") or {}

    partial = bool(manifest.missing_cameras()) or (report_pdf_path is None)
    terminal = (LifecycleState.COMPLETED_PARTIAL if partial
                else LifecycleState.COMPLETED)
    if report_pdf_path is None:
        terminal = LifecycleState.REPORT_FAILED

    root = batch_root(ctx, manifest.batch_key)
    combined_pdf_hash = FIN.sha256_file(report_pdf_path)
    combined_json_hash = FIN.sha256_file(report_json_path)
    final_report_hash = combined_pdf_hash or combined_json_hash
    idem_key = FIN.email_idempotency_key(
        manifest.batch_key, manifest.report_revision, final_report_hash)

    prior = FIN.load(root) or {}
    same_rev = (prior.get("report_revision") == manifest.report_revision
                and prior.get("combined_pdf_hash") == combined_pdf_hash
                and prior.get("combined_json_hash") == combined_json_hash)
    already_uploaded = bool(same_rev and prior.get("uploaded"))
    already_emailed = bool(prior.get("email_sent")
                           and prior.get("email_idempotency_key") == idem_key)

    upload_urls: Dict[str, Any] = dict(prior.get("upload_urls") or {}) if already_uploaded else {}
    camera_report_hashes = dict(prior.get("camera_report_hashes") or {})
    processed_video_hashes = dict(prior.get("processed_video_hashes") or {})

    def _mk(uploaded: bool, email_sent: bool, email_status: str) -> Dict[str, Any]:
        return {
            "batch_key": manifest.batch_key,
            "terminal_status": _terminal_status_name(terminal),
            "report_revision": manifest.report_revision,
            "report_status": manifest.report_status,
            "global_state_version": manifest.global_state_version,
            "combined_pdf_hash": combined_pdf_hash,
            "combined_json_hash": combined_json_hash,
            "camera_report_hashes": camera_report_hashes,
            "processed_video_hashes": processed_video_hashes,
            "upload_urls": upload_urls,
            "uploaded": uploaded,
            "email_status": email_status,
            "email_sent": email_sent,
            "email_idempotency_key": idem_key,
            "cameras_present": meta.get("cameras_present") or manifest.present_cameras(),
            "cameras_missing_final": meta.get("cameras_missing_final") or [],
        }

    # ---- upload (skipped if already done for this revision) ----
    uploaded = already_uploaded
    # Interim delivery is opt-in; final always delivers (unless skip/failure).
    can_deliver = (not ctx.skip_upload and ctx.s3_client is not None
                   and terminal != LifecycleState.FAILED)
    if can_deliver and not already_uploaded:
        try:
            if report_pdf_path:                      # suppressed for REPORT_FAILED
                upload_urls["pdf"] = s3_upload.upload_pdf(
                    ctx.s3_client, report_pdf_path, manifest.batch_key)
            if report_json_path:                     # ALWAYS upload the canonical JSON
                upload_urls["json"] = s3_upload.upload_json(
                    ctx.s3_client, report_json_path, manifest.batch_key)
            for cam, path in camera_pdf_paths.items():
                u = s3_upload.upload_pdf(ctx.s3_client, path, manifest.batch_key)
                if u:
                    upload_urls[f"camera_{cam}"] = u
                camera_report_hashes[cam] = FIN.sha256_file(path)
            for cam in manifest.present_cameras():
                mp4 = os.path.join(root, CFG.DIR_PROCESSED_VIDEOS, f"{cam}_processed.mp4")
                h = FIN.sha256_file(mp4)
                if h:
                    processed_video_hashes[cam] = h
            for sub in (CFG.DIR_GLOBAL_STATE, CFG.DIR_WAGON_STATES, CFG.DIR_REPORTS,
                        CFG.DIR_EVIDENCE, CFG.DIR_PROCESSED_VIDEOS):
                s3_upload.upload_tree(
                    ctx.s3_client, os.path.join(root, sub), manifest.batch_key,
                    sub_prefix=sub,
                    skip_extensions=({".jpg", ".jpeg"} if sub == CFG.DIR_GLOBAL_STATE else None))
            uploaded = True
        except Exception as e:
            log.error("[FINALIZE %s] delivery failed: %s", manifest.batch_key, e)
    # Persist the marker AFTER upload, BEFORE email, so a crash here resumes
    # without re-uploading and still sends the email.
    FIN.write(root, _mk(uploaded, already_emailed, prior.get("email_status") or "pending"))

    report_pdf_url = upload_urls.get("pdf")
    report_json_url = upload_urls.get("json")

    # ---- email (exactly once; suppressed when the final PDF is unavailable) ----
    email_sent = already_emailed
    if terminal == LifecycleState.REPORT_FAILED:
        email_status = "suppressed_report_failed"
        log.warning("[FINALIZE %s] REPORT_FAILED -- JSON uploaded, email suppressed",
                    manifest.batch_key)
    elif already_emailed:
        email_status = "already_sent"
        log.info("[FINALIZE %s] email already sent (idem=%s) -- skip",
                 manifest.batch_key, idem_key[:12])
    elif ctx.skip_email or not report_pdf_url:
        email_status = "skipped"
    else:
        ok = False
        try:
            ok = notification.send_email(
                batch_key=manifest.batch_key,
                report_pdf_url=report_pdf_url, report_json_url=report_json_url,
                summary=summarize_wagons(list(unified.values())),
                cameras_present=manifest.present_cameras(),
                cameras_missing=manifest.missing_cameras(),
                final_status=manifest.report_status or "FINAL",
                idempotency_key=idem_key)
        except Exception as e:
            log.error("[FINALIZE %s] email failed: %s", manifest.batch_key, e)
        email_sent = bool(ok)
        email_status = "sent" if ok else "failed"
        if ok:
            log.info("[FINALIZE %s] email sent once (best-effort exactly-once; a crash "
                     "between API-200 and marker write may resend)", manifest.batch_key)

    FIN.write(root, _mk(uploaded, email_sent, email_status))

    # ---- legacy dashboard ingest (Stage-6, read-only, ON by default -> V1) ----
    # Re-derives the old per-camera *_inspection.json feed from finalized
    # artifacts and POSTs it to the dashboard ingest API.  Fully isolated: it
    # never mutates the manifest/report/sealed state, and any failure here is
    # swallowed so it cannot change the batch's terminal outcome.
    try:
        from delivery import dashboard_ingest
        if dashboard_ingest.is_enabled():
            dashboard_ingest.run(batch_root=root, s3_client=ctx.s3_client,
                                 skip_upload=ctx.skip_upload)
    except Exception as e:
        log.error("[FINALIZE %s] dashboard ingest error (non-fatal): %s",
                  manifest.batch_key, e)

    _finish(manifest, ctx, terminal)


def _terminal_status_name(terminal_state: str) -> str:
    from core.lifecycle import terminal_batch_status
    return terminal_batch_status(terminal_state)


def _finish(manifest: BatchManifest, ctx: RunContext, terminal_state: str) -> None:
    from core.lifecycle import terminal_batch_status
    manifest.terminal_status = terminal_batch_status(terminal_state)
    _transition(manifest, terminal_state, ctx, reason="finalized")


# -----------------------------------------------------------------------------
# State machine
# -----------------------------------------------------------------------------

def _choose_master(manifest: BatchManifest) -> Optional[str]:
    """Which camera can serve as master right now?  RIGHT_UP if present; else
    LEFT_UP only if the fallback is enabled."""
    present = manifest.present_cameras()
    if C.MASTER_CAMERA in present:
        return C.MASTER_CAMERA
    if CFG.ENABLE_LEFT_UP_FALLBACK_MASTER and C.CAMERA_LEFT_UP in present:
        return C.CAMERA_LEFT_UP
    return None


def advance(manifest: BatchManifest, ctx: RunContext) -> BatchManifest:
    """Drive the batch forward as far as it can go this tick."""
    guard = 0
    while guard < 32:
        guard += 1
        st = manifest.lifecycle_status
        if is_terminal(st):
            return manifest

        # ---- pre-seal ----
        if st in (LifecycleState.DISCOVERED, LifecycleState.COLLECTING_CAMERAS,
                  LifecycleState.WAITING_FOR_MASTER, LifecycleState.WAITING_FOR_SUPPORT):
            master = _choose_master(manifest)
            log.info("[PRESEAL %s] state=%s present=%s master=%s complete=%s "
                     "past_support=%s past_master=%s past_final=%s",
                     manifest.batch_key, st, manifest.present_cameras(), master,
                     manifest.is_complete(), manifest.past_support_window(),
                     manifest.past_master_deadline(), manifest.past_final_deadline())
            if master == C.MASTER_CAMERA:
                # RIGHT_UP present: wait for the short support window (unless all
                # support already here), then seal -- never wait for final. [C3]
                if manifest.is_complete() or manifest.past_support_window():
                    _transition(manifest, LifecycleState.RECONSTRUCTING, ctx)
                    continue
                _transition(manifest, LifecycleState.WAITING_FOR_SUPPORT, ctx)
                return manifest
            # RIGHT_UP absent
            if manifest.past_master_deadline():
                if master == C.CAMERA_LEFT_UP:      # fallback enabled + LEFT_UP present
                    _transition(manifest, LifecycleState.RECONSTRUCTING, ctx,
                                reason="left_up_fallback_master")
                    continue
                if manifest.past_final_deadline():
                    manifest.sealing_reason = "no_master_by_final_deadline"
                    _finish(manifest, ctx, LifecycleState.FAILED_NO_GLOBAL_STATE)
                    return manifest
                _transition(manifest, LifecycleState.WAITING_FOR_MASTER, ctx)
                return manifest
            _transition(manifest, LifecycleState.COLLECTING_CAMERAS, ctx)
            return manifest

        # ---- seal ----
        if st == LifecycleState.RECONSTRUCTING:
            master = _choose_master(manifest) or C.MASTER_CAMERA
            ok = stage_seal(manifest, ctx, master_camera=master)
            if not ok:
                return manifest  # transitioned to FAILED_NO_GLOBAL_STATE
            continue

        # ---- post-seal: process everything present ----
        if st == LifecycleState.GLOBAL_STATE_SEALED:
            _transition(manifest, LifecycleState.PROCESSING_AVAILABLE, ctx)
            continue

        if st == LifecycleState.PROCESSING_AVAILABLE:
            stage_process_cameras(manifest, ctx, cameras=manifest.present_cameras())
            stage_reports(manifest, ctx, final=False)
            log.info("[POSTSEAL %s] processed present=%s missing=%s complete=%s "
                     "past_final=%s -> %s", manifest.batch_key,
                     manifest.present_cameras(), manifest.missing_cameras(),
                     manifest.is_complete(), manifest.past_final_deadline(),
                     ("FINALIZING" if (manifest.is_complete()
                                       or manifest.past_final_deadline())
                      else "WAITING_FOR_LATE_CAMERAS"))
            if manifest.is_complete() or manifest.past_final_deadline():
                _transition(manifest, LifecycleState.FINALIZING, ctx,
                            reason=("complete" if manifest.is_complete()
                                    else "final_deadline"))
                continue
            _transition(manifest, LifecycleState.WAITING_FOR_LATE_CAMERAS, ctx)
            return manifest

        if st == LifecycleState.WAITING_FOR_LATE_CAMERAS:
            # any present camera whose features aren't done yet?
            pending = [c for c in manifest.present_cameras()
                       if set(CF.features_for_camera(c)) - set(manifest.completed_features.get(c, []))]
            if pending:
                _transition(manifest, LifecycleState.PROCESSING_LATE_CAMERA, ctx,
                            reason=f"late:{pending}")
                continue
            if manifest.is_complete() or manifest.past_final_deadline():
                _transition(manifest, LifecycleState.FINALIZING, ctx,
                            reason=("complete" if manifest.is_complete() else "final_deadline"))
                continue
            return manifest  # keep waiting

        if st == LifecycleState.PROCESSING_LATE_CAMERA:
            pending = [c for c in manifest.present_cameras()
                       if set(CF.features_for_camera(c)) - set(manifest.completed_features.get(c, []))]
            stage_process_cameras(manifest, ctx, cameras=pending)
            # regenerate ONLY the late camera(s) overlay + PDF; combined always
            stage_reports(manifest, ctx, final=False, cameras=pending)
            _transition(manifest, LifecycleState.WAITING_FOR_LATE_CAMERAS, ctx)
            continue

        if st == LifecycleState.FINALIZING:
            stage_finalize(manifest, ctx)
            return manifest

        log.error("[LIFECYCLE %s] unknown state %s -- marking FAILED",
                  manifest.batch_key, st)
        _finish(manifest, ctx, LifecycleState.FAILED)
        return manifest

    log.warning("[LIFECYCLE %s] advance() hit guard limit at %s",
                manifest.batch_key, manifest.lifecycle_status)
    return manifest
