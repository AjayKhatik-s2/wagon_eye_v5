"""WagonEye v4 Master Orchestrator -- train-state-native.

Run modes:
    python -m orchestrator.master_runner --auto         # continuous S3 polling
    python -m orchestrator.master_runner --once         # one batch, exit
    python -m orchestrator.master_runner --batch <key>  # replay a specific batch
    python -m orchestrator.master_runner --local-only --local-inputs DIR

Pipeline (per batch):
    Stage 1  reconstruction.runner.run     -> GlobalTrainState
    Stage 2  materializer.wagon_cache_builder.build  -> wagon_cache/
    Stage 3  features.{door,load,damage,ocr}.processor.run (parallel)
    Stage 4  fusion.wagon_state_builder.build
    Stage 5  reporting.combined_train_report.build
    Stage 6  delivery.{s3_upload, notification}

There is NO legacy v3 fallback.  Stage-1 failure -> batch is marked
failed_no_global_state and abandoned.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Make sibling packages importable when running this file directly.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Local packages
from core import constants as C
from core import config as CFG
from core.logging_setup import setup_logging, get_logger
from core.feature_config import (
    FeatureConfig, FEATURE_REGISTRY, parse_disable_arg,
)
from core.batch import (
    CameraVideo, TrainBatch,
    build_local_batch, scan_local_video_dir,
)
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons

from reconstruction import runner as reconstruction_runner
from materializer import wagon_cache_builder
from features.door   import processor as door_proc
from features.load   import processor as load_proc
from features.damage import processor as damage_proc
from features.ocr    import processor as ocr_proc
from fusion import wagon_state_builder
from reporting import combined_train_report, camera_reports
from rendering import feature_overlay_renderer
from delivery import s3_upload, notification

log = get_logger("orchestrator")

# Default per-batch paths -- centralized in core.config (env-overridable, all
# defaulting to the pre-migration <repo>/... locations).
DEFAULT_WORKSPACE_PARENT = CFG.WORKSPACE_ROOT
DEFAULT_MODELS_DIR        = CFG.MODELS_DIR
DEFAULT_RECON_MODELS_DIR  = CFG.RECON_MODELS_DIR
DEFAULT_FEAT_MODELS_DIR   = CFG.FEAT_MODELS_DIR

# Resolve the inference device once for the whole process (CUDA if available,
# else CPU; WAGONEYE_DEVICE overrides).  Used only for post-batch GPU cache
# hygiene here -- feature/reconstruction modules resolve it independently.
_DEVICE = CFG.resolve_device()


def _free_gpu_cache() -> None:
    """Release fragmented CUDA allocator blocks between batches.

    Does NOT evict the per-process YOLO model cache (model reuse across
    batches is intended) -- it only trims allocator fragmentation, which
    matters over many-hour continuous runs.  No-op on CPU.
    """
    if _DEVICE != "cuda":
        return
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Outcome
# -----------------------------------------------------------------------------

@dataclass
class BatchOutcome:
    batch: TrainBatch
    state: Optional[GlobalTrainState] = None
    unified: Dict[str, UnifiedWagonState] = field(default_factory=dict)
    feature_summary: Dict[str, Dict[str, str]] = field(default_factory=dict)
    cache_summary: Optional[Any] = None
    report_pdf_path: Optional[str] = None
    report_pdf_url: Optional[str] = None
    report_json_path: Optional[str] = None
    report_json_url: Optional[str] = None
    camera_pdf_paths: Dict[str, str] = field(default_factory=dict)
    camera_pdf_urls:  Dict[str, str] = field(default_factory=dict)
    processed_video_paths: Dict[str, str] = field(default_factory=dict)
    processed_video_urls:  Dict[str, str] = field(default_factory=dict)
    final_status: str = "unknown"
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


# -----------------------------------------------------------------------------
# Per-batch flow
# -----------------------------------------------------------------------------

def _print_feature_config(cfg: FeatureConfig, *, header: str) -> None:
    print(header)
    for spec in FEATURE_REGISTRY:
        flag = "[ON] " if cfg.is_enabled(spec.key) else "[OFF]"
        print(f"  {flag} {spec.display_name}")


def resolve_feature_config(
    *,
    disable_features: str = "",
    interactive: Optional[bool] = None,
) -> FeatureConfig:
    """Decide which Stage-3 features run this session.

    Precedence:
        1. --disable-features CLI value (explicit, never prompts).
        2. Interactive TTY prompt (only when stdin is a real terminal AND
           the caller allows it).
        3. Default: every feature ON (auto / cron / piped runs).

    Safe for non-interactive/auto/cron: when stdin is not a TTY we NEVER block
    on input -- we return all-ON (or honour the CLI list).
    """
    cli_disabled = parse_disable_arg(disable_features)
    if cli_disabled:
        cfg = FeatureConfig.from_disabled(cli_disabled)
        _print_feature_config(
            cfg, header="Feature Configuration (from --disable-features):")
        return cfg

    try:
        is_tty = sys.stdin.isatty()
    except Exception:
        is_tty = False
    interactive = bool(is_tty if interactive is None else (interactive and is_tty))

    cfg = FeatureConfig.all_on()
    if not interactive:
        return cfg

    _print_feature_config(cfg, header="Current Feature Configuration:")
    try:
        ans = input("Turn OFF any feature? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return cfg
    if ans not in ("y", "yes"):
        return cfg

    print("\nSelect feature(s) to turn OFF (comma-separated numbers, e.g. 2,4):")
    for i, spec in enumerate(FEATURE_REGISTRY, start=1):
        print(f"  {i}. {spec.display_name}")
    try:
        sel = input("Disable: ").strip()
    except (EOFError, KeyboardInterrupt):
        return cfg
    for tok in sel.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError:
            continue
        if 1 <= n <= len(FEATURE_REGISTRY):
            cfg.disable(FEATURE_REGISTRY[n - 1].key)

    _print_feature_config(cfg, header="\nFinal Feature Configuration:")
    return cfg


def process_batch(
    *,
    batch: TrainBatch,
    workspace_root: str,
    recon_models_dir: str,
    feat_models_dir: str,
    s3_client=None,
    skip_upload: bool = False,
    skip_email: bool = False,
    verbose: bool = True,
    feature_config: Optional[FeatureConfig] = None,
) -> BatchOutcome:
    if feature_config is None:
        feature_config = FeatureConfig.all_on()
    t_batch = time.time()
    out = BatchOutcome(batch=batch)
    batch_root  = os.path.join(workspace_root, batch.batch_key)
    download_root  = os.path.join(batch_root, CFG.DIR_DOWNLOADS)
    stage0_root    = os.path.join(batch_root, CFG.DIR_GLOBAL_STATE)
    cache_root     = os.path.join(batch_root, CFG.DIR_WAGON_CACHE)
    states_root    = os.path.join(batch_root, CFG.DIR_WAGON_STATES)
    evidence_root  = os.path.join(batch_root, CFG.DIR_EVIDENCE)
    processed_root = os.path.join(batch_root, CFG.DIR_PROCESSED_VIDEOS)
    reports_root   = os.path.join(batch_root, CFG.DIR_REPORTS)
    archive_root   = os.path.join(batch_root, CFG.DIR_ARCHIVE)
    for d in (download_root, stage0_root, cache_root, states_root,
              evidence_root, processed_root, reports_root, archive_root):
        os.makedirs(d, exist_ok=True)

    log.info("%s", "=" * 78)
    log.info("BATCH %s", batch.batch_key)
    log.info("cameras present : %s", batch.present_cameras())
    log.info("cameras missing : %s", batch.missing_cameras() or "—")

    # ---- Download (or pass through local paths) ----
    video_paths: Dict[str, str] = {}
    try:
        for cam in C.ALL_CAMERAS:
            cv = batch.videos.get(cam)
            if cv is None:
                continue
            if cv.bucket == "__local__":
                video_paths[cam] = cv.s3_key
            else:
                local_path = os.path.join(download_root, f"{cam}_{cv.filename}")
                s3_client.download_file(cv.bucket, cv.s3_key, local_path)
                video_paths[cam] = local_path
    except Exception as e:
        out.error = f"download: {e}"
        out.final_status = C.BATCH_FAILED
        out.elapsed_seconds = time.time() - t_batch
        log.error("download failed: %s", e, exc_info=True)
        return out

    # ---- Stage 1: reconstruction ----
    log.info("--- STAGE 1  Global train reconstruction ---")
    _t = time.time()
    try:
        recon = reconstruction_runner.run(
            video_paths=video_paths,
            reconstruction_models_dir=recon_models_dir,
            output_dir=stage0_root,
            repo_root=_REPO_ROOT,
            verbose=verbose,
        )
        out.state = recon.state
    except reconstruction_runner.ReconstructionError as e:
        out.error = f"stage1: {e}"
        out.final_status = C.BATCH_FAILED_NO_GLOBAL
        out.elapsed_seconds = time.time() - t_batch
        log.error("[BATCH] aborted (stage1): %s", e)
        return out
    log.info("STAGE 1 complete (%.1fs)", time.time() - _t)

    # ---- Stage 2: materializer ----
    log.info("--- STAGE 2  Wagon cache materialization ---")
    _t = time.time()
    try:
        out.cache_summary = wagon_cache_builder.build(
            state=recon.state,
            video_paths=video_paths,
            per_camera_fps=recon.per_camera_fps,
            cache_root=cache_root,
            verbose=verbose,
        )
    except Exception as e:
        out.error = f"stage2: {e}"
        out.final_status = C.BATCH_FAILED
        out.elapsed_seconds = time.time() - t_batch
        log.error("stage2 materializer failed: %s", e, exc_info=True)
        return out
    log.info("STAGE 2 complete (%.1fs)", time.time() - _t)

    # ---- Stage 3: feature processors ----
    # The damage processor reads the sibling `load` JSON to drop floor_damage
    # tracks on LOADED wagons.  Under full 4-way parallelism that read raced the
    # load writer (handled fail-open, but nondeterministic).  We therefore run
    # the LOAD feature to completion FIRST, then door / ocr / damage in parallel
    # -- so the loaded-wagon floor-damage filter always sees a fully-written
    # wagon_states/load/<gw>.json.  Feature-wise execution + per-model reuse are
    # preserved (each YOLO/easyocr model still loads once and is reused across
    # all wagons within its processor).
    log.info("--- STAGE 3  Feature inference ---")
    _t = time.time()
    log.info("feature config: enabled=%s disabled=%s",
             feature_config.enabled_keys() or "—",
             feature_config.disabled_keys() or "—")
    _print_feature_config(feature_config, header="  feature config:")
    feature_kwargs = dict(
        state=recon.state,
        cache_root=cache_root,
        feature_models_dir=feat_models_dir,
        output_dir=states_root,
        evidence_root=evidence_root,
        verbose=verbose,
    )

    def _run_feature(name, fn):
        try:
            return fn(**feature_kwargs)
        except Exception as e:
            log.error("[STAGE3/%s] CRASHED: %s", name, e, exc_info=True)
            return {}

    def _mark_disabled(name):
        """Write a DISABLED_BY_USER sentinel JSON for every wagon of a
        toggled-off feature so fusion + reports show 'DISABLED BY USER'
        instead of silently treating the field as NO_DATA."""
        from features._common import write_per_wagon_json, empty_payload
        feature_out = os.path.join(states_root, name)
        summary: Dict[str, str] = {}
        for gw in recon.state.wagons:
            payload = empty_payload(
                gw.global_id, name, C.STATUS_DISABLED,
                disabled_by_user=True,
            )
            write_per_wagon_json(feature_out, gw.global_id, payload)
            summary[gw.global_id] = C.STATUS_DISABLED
        log.info("[STAGE3/%s] DISABLED BY USER -- wrote sentinel for %d wagons",
                 name, len(summary))
        return summary

    # 1) Load first (deterministic input for damage's load-aware filter).
    if feature_config.is_enabled("load"):
        out.feature_summary["load"] = _run_feature("load", load_proc.run)
    else:
        out.feature_summary["load"] = _mark_disabled("load")

    # 2) Then door / ocr / damage -- only the enabled ones run (in parallel).
    all_parallel = {
        "door":   door_proc.run,
        "ocr":    ocr_proc.run,
        "damage": damage_proc.run,
    }
    parallel_targets = {n: fn for n, fn in all_parallel.items()
                        if feature_config.is_enabled(n)}
    for name in all_parallel:
        if name not in parallel_targets:
            out.feature_summary[name] = _mark_disabled(name)

    if parallel_targets:
        with ThreadPoolExecutor(max_workers=len(parallel_targets)) as ex:
            futs = {ex.submit(_run_feature, name, fn): name
                    for name, fn in parallel_targets.items()}
            for f in as_completed(futs):
                out.feature_summary[futs[f]] = f.result()

    log.info("STAGE 3 complete (%.1fs)", time.time() - _t)

    # ---- Stage 4: fusion ----
    log.info("--- STAGE 4  Wagon state fusion ---")
    _t = time.time()
    try:
        out.unified = wagon_state_builder.build(
            state=recon.state,
            wagon_states_root=states_root,
            verbose=verbose,
        )
    except Exception as e:
        out.error = f"stage4: {e}"
        out.final_status = C.BATCH_FAILED
        out.elapsed_seconds = time.time() - t_batch
        log.error("stage4 fusion failed: %s", e, exc_info=True)
        return out
    log.info("STAGE 4 complete (%.1fs)", time.time() - _t)

    # ---- Stage 4b: feature overlay video rendering (visualization only) ----
    log.info("--- STAGE 4b  Feature overlay rendering ---")
    try:
        out.processed_video_paths = feature_overlay_renderer.render_all_cameras(
            state=recon.state,
            unified=out.unified,
            evidence_root=evidence_root,
            video_paths=video_paths,
            per_camera_tracking_path=recon.per_camera_tracking_path,
            output_dir=processed_root,
            enabled_features=set(feature_config.enabled_keys()),
            verbose=verbose,
        )
    except Exception as e:
        log.error("[STAGE4b] feature overlay rendering FAILED: %s", e, exc_info=True)
        out.processed_video_paths = {}

    # Deterministic S3 URLs for processed videos so Stage 5 can embed them
    # before Stage 6 actually uploads (mirrors `s3_upload.upload_tree`'s key
    # construction: <archive_prefix>/<batch_key>/processed_videos/<file>).
    def _processed_video_url(cam: str, local_path: str) -> str:
        if not local_path or skip_upload:
            return local_path or ""
        key = (f"{C.S3_ARCHIVE_PREFIX}/{batch.batch_key}/"
               f"processed_videos/{os.path.basename(local_path)}")
        return f"https://{C.S3_OUTPUT_BUCKET}.s3.{C.S3_REGION}.amazonaws.com/{key}"

    out.processed_video_urls = {
        cam: _processed_video_url(cam, p)
        for cam, p in out.processed_video_paths.items()
    }

    # Resolve logo asset (lives at <repo>/reporting/assets/Logo.jpeg).
    # NOTE: this was previously built from _PKG_DIR (the orchestrator/ dir),
    # which resolved to a nonexistent path so every report silently rendered
    # without a logo.  core.config.LOGO_PATH is anchored at PROJECT_ROOT.
    _logo_path = CFG.LOGO_PATH
    _per_camera_tracking_path = recon.per_camera_tracking_path

    # ---- Stage 5a: camera-wise reports (legacy hierarchy; built first so
    # the combined report's DETAILED CAMERA REPORTS table can link them) ----
    log.info("--- STAGE 5a  Camera-wise reports ---")
    try:
        out.camera_pdf_paths = {
            cam: v for cam, v in camera_reports.build_all(
                state=recon.state,
                unified=out.unified,
                evidence_root=evidence_root,
                wagon_states_root=states_root,
                cache_root=cache_root,
                per_camera_tracking_path=_per_camera_tracking_path,
                output_dir=reports_root,
                batch_key=batch.batch_key,
                logo_path=_logo_path,
                verbose=verbose,
            ).items() if v
        }
    except Exception as e:
        log.error("[STAGE5a] camera reports FAILED: %s", e, exc_info=True)
        out.camera_pdf_paths = {}

    # Relative basenames are linkable both locally (sibling file:// in the
    # reports/ dir) and on S3 (sibling object under reports/<batch>/).
    camera_pdf_urls: Dict[str, str] = {
        cam: os.path.basename(p) for cam, p in out.camera_pdf_paths.items()
    }

    # ---- Stage 5b: combined report (aggregates the 4 camera reports) ----
    log.info("--- STAGE 5b  Combined report ---")
    try:
        result = combined_train_report.build(
            state=recon.state,
            unified=out.unified,
            output_dir=reports_root,
            batch_key=batch.batch_key,
            source_video_urls={
                cam: (batch.videos[cam].s3_url
                      if cam in batch.videos
                      and batch.videos[cam].bucket != "__local__"
                      else "")
                for cam in C.ALL_CAMERAS if cam in batch.videos
            },
            processed_video_urls=out.processed_video_urls,
            evidence_root=evidence_root,
            wagon_states_root=states_root,
            cache_root=cache_root,
            missing_cameras=list(batch.missing_cameras()),
            camera_pdf_urls=camera_pdf_urls,
            logo_path=_logo_path,
            verbose=verbose,
        )
        out.report_json_path = result.get("json_path")
        out.report_pdf_path  = result.get("pdf_path")
    except Exception as e:
        out.error = f"stage5: {e}"
        out.final_status = C.BATCH_REPORT_FAILED
        out.elapsed_seconds = time.time() - t_batch
        log.error("stage5 combined report failed: %s", e, exc_info=True)
        _free_gpu_cache()
        return out

    # ---- decide completion class ----
    partial = any(
        v in (C.STATUS_NO_FRAMES, C.STATUS_FAILED, C.NO_DATA)
        for d in out.feature_summary.values() for v in d.values()
    )
    if out.report_pdf_path is None:
        out.final_status = C.BATCH_REPORT_FAILED
    else:
        out.final_status = (
            C.BATCH_COMPLETED_PARTIAL if partial else C.BATCH_COMPLETED
        )

    # ---- Stage 6: delivery ----
    if skip_upload:
        out.elapsed_seconds = time.time() - t_batch
        log.info("[BATCH %s] %s (%.1fs) -- upload skipped",
                 batch.batch_key, out.final_status, out.elapsed_seconds)
        _free_gpu_cache()
        return out

    log.info("--- STAGE 6  Delivery ---")
    if out.report_pdf_path:
        out.report_pdf_url = s3_upload.upload_pdf(
            s3_client, out.report_pdf_path, batch.batch_key,
        )
    if out.report_json_path:
        out.report_json_url = s3_upload.upload_json(
            s3_client, out.report_json_path, batch.batch_key,
        )
    # Camera-wise PDFs go through the same microservice flow.
    for cam, path in out.camera_pdf_paths.items():
        url = s3_upload.upload_pdf(s3_client, path, batch.batch_key)
        if url:
            out.camera_pdf_urls[cam] = url

    # Archive everything per-feature + the cache (skip huge JPEGs in cache to S3
    # by default; keep wagon_states + global_state + reports which are small)
    n_state  = s3_upload.upload_tree(s3_client, stage0_root, batch.batch_key,
                                     sub_prefix="global_state",
                                     skip_extensions={".jpg", ".jpeg"})
    n_states = s3_upload.upload_tree(s3_client, states_root, batch.batch_key,
                                     sub_prefix="wagon_states")
    n_reports = s3_upload.upload_tree(s3_client, reports_root, batch.batch_key,
                                      sub_prefix="reports")
    n_evidence = s3_upload.upload_tree(s3_client, evidence_root, batch.batch_key,
                                       sub_prefix="evidence")
    n_videos = s3_upload.upload_tree(s3_client, processed_root, batch.batch_key,
                                     sub_prefix="processed_videos")
    log.info("[STAGE6] archived: global_state=%d files, wagon_states=%d files, "
             "reports=%d files, evidence=%d files, processed_videos=%d files",
             n_state, n_states, n_reports, n_evidence, n_videos)

    if not skip_email:
        summary = summarize_wagons(list(out.unified.values()))
        notification.send_email(
            batch_key=batch.batch_key,
            report_pdf_url=out.report_pdf_url,
            report_json_url=out.report_json_url,
            summary=summary,
            cameras_present=batch.present_cameras(),
            cameras_missing=batch.missing_cameras(),
            final_status=out.final_status,
        )

    out.elapsed_seconds = time.time() - t_batch
    log.info("[BATCH %s] %s (%.1fs)",
             batch.batch_key, out.final_status, out.elapsed_seconds)
    _free_gpu_cache()
    return out


# -----------------------------------------------------------------------------
# Graceful-shutdown flag for the continuous service.  A SIGTERM/SIGINT sets
# this flag rather than raising, so an in-flight batch is NEVER interrupted
# mid-processing -- run_auto() only checks the flag between batches and after
# each idle sleep, then exits cleanly.  This is what makes `systemctl stop`
# (which sends SIGTERM) finish the current batch before shutting down.
# -----------------------------------------------------------------------------

_SHUTDOWN_REQUESTED = False


def _request_shutdown(signum, _frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    try:
        name = signal.Signals(signum).name
    except Exception:
        name = str(signum)
    log.info("[ORCH] %s received -- will stop after the current batch", name)


def _install_signal_handlers() -> None:
    """Register SIGTERM/SIGINT handlers for graceful shutdown.

    signal.signal only works on the main thread; guarded so importing / calling
    from a worker thread (or a platform without a given signal) never crashes.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _request_shutdown)
        except (ValueError, OSError, AttributeError):
            pass


# -----------------------------------------------------------------------------
# Continuous mode (S3 polling).  Batch acquisition is delegated to the
# in-package train_batch_manager (Stage-0); this loop owns scheduling +
# graceful shutdown only.
# -----------------------------------------------------------------------------

# Ambiguity guard: when a candidate is within tolerance of two active batches,
# attach to the nearest; if the two nearest are closer together than this many
# seconds we cannot decide safely -> hold the video for review, don't attach.
_ATTACH_AMBIGUITY_SEC = 5.0


def _canonical_dt(ts: str):
    from datetime import datetime, timezone
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _attach_candidate(cv, actives, processed, ctx, tolerance_sec):
    """Attach a discovered CameraVideo to an active manifest (or create one).

    Idempotent: re-seeing the same (camera, ETag) is a no-op; a changed ETag
    refreshes that camera's slot (its derived artifacts get rebuilt later).
    Never reopens a terminal batch; ambiguous matches are held for review.
    """
    from orchestrator import batch_manifest as BM
    from core.lifecycle import ArrivalState

    cand_dt = _canonical_dt(cv.train_timestamp)
    if cand_dt is None:
        return

    # rank active manifests by temporal distance
    scored = []
    for m in actives.values():
        mdt = _canonical_dt(m.canonical_train_timestamp)
        if mdt is None:
            continue
        dist = abs((cand_dt - mdt).total_seconds())
        if dist <= tolerance_sec:
            scored.append((dist, m))
    scored.sort(key=lambda x: (x[0], x[1].batch_key))

    # Full grouping-decision trace: which active batches are within the match
    # window for this candidate, and whether a TERMINAL batch is also in range
    # (the latter is what produces "matches terminal batch -- ignored").
    terminal_in_range = [
        key for key in processed
        if _canonical_dt(key) is not None
        and abs((cand_dt - _canonical_dt(key)).total_seconds()) <= tolerance_sec
    ]
    log.info("[GROUPING] camera=%s ts=%s key=%s tolerance=%ds "
             "active_in_range=%s terminal_in_range=%s",
             cv.camera_id, cv.train_timestamp, cv.s3_key, int(tolerance_sec),
             [(m.batch_key, round(d, 1)) for d, m in scored], terminal_in_range)

    target = None
    if scored:
        if len(scored) >= 2 and (scored[1][0] - scored[0][0]) < _ATTACH_AMBIGUITY_SEC:
            best = scored[0][1]
            best.videos_for_review.append({
                "camera_id": cv.camera_id, "s3_key": cv.s3_key, "etag": cv.etag,
                "reason": "ambiguous_match",
                "candidates": [m.batch_key for _, m in scored[:2]],
            })
            log.warning("[ATTACH] %s ambiguous between %s -- held for review",
                        cv.s3_key, [m.batch_key for _, m in scored[:2]])
            BM.write_local(best, os.path.join(ctx.workspace_root, best.batch_key))
            BM.save_s3(ctx.s3_client, best)
            return
        target = scored[0][1]

    if target is None:
        # terminal batch within tolerance? -> ignore (no silent reopen)
        for key in processed:
            kdt = _canonical_dt(key)
            if kdt is not None and abs((cand_dt - kdt).total_seconds()) <= tolerance_sec:
                log.info("[ATTACH] %s matches terminal batch %s -- ignored (no reopen)",
                         cv.s3_key, key)
                return
        # brand new batch
        target = BM.BatchManifest.new(batch_key=cv.train_timestamp,
                                      train_timestamp=cv.train_timestamp,
                                      match_window_sec=int(tolerance_sec))
        actives[target.batch_key] = target
        log.info("[ATTACH] new batch %s from %s (%s)",
                 target.batch_key, cv.camera_id, cv.s3_key)

    slot = target.cameras.get(cv.camera_id)
    if slot is not None and slot.arrival_state == ArrivalState.PRESENT and slot.etag == cv.etag:
        return  # already have this exact source version
    if slot is not None and slot.etag and slot.etag != cv.etag:
        log.info("[ATTACH] %s/%s ETag changed %s -> %s: rebuild that camera",
                 target.batch_key, cv.camera_id, slot.etag, cv.etag)
        # drop that camera from completed markers so it re-materializes/re-runs
        target.materialized_cameras = [c for c in target.materialized_cameras
                                       if c != cv.camera_id]
        target.completed_features.pop(cv.camera_id, None)
    new_slot = BM.CameraSlot(
        camera_id=cv.camera_id, bucket=cv.bucket, s3_key=cv.s3_key, etag=cv.etag,
        filename=cv.filename, s3_url=cv.s3_url,
        last_modified=(cv.last_modified.isoformat() if cv.last_modified
                       and hasattr(cv.last_modified, "isoformat") else None),
    )
    target.set_camera(new_slot)
    attached = sorted(target.cameras.keys())
    log.info("[ATTACH] batch=%s camera=%s ts=%s decision=ATTACHED cameras=%d/%d %s",
             target.batch_key, cv.camera_id, cv.train_timestamp,
             len(attached), len(C.ALL_CAMERAS), attached)
    BM.write_local(target, os.path.join(ctx.workspace_root, target.batch_key))
    BM.save_s3(ctx.s3_client, target)


def _extraction_sweep() -> Dict[str, Dict[str, int]]:
    """Stage A (in-process): run ONE raw->trimmed extraction sweep of all four
    cameras, then return.

    This reuses the EXISTING extraction producer verbatim -- the same
    `train_extraction.run_extraction_service.sweep_camera` the standalone
    service runs -- so no extraction logic is duplicated or changed.  Calling it
    at the top of each `--auto` poll tick (before complete-train discovery) is
    what makes `master_runner --auto` a true single-process pipeline:

        RAW bucket -> extraction/trim -> complete-train  (this function)
        complete-train -> discovery -> lifecycle         (the rest of run_auto)

    Fault tolerance: a per-camera failure (missing extraction model, S3 error,
    a bad clip) is logged and the sweep continues to the next camera; the tick
    then proceeds to discovery with whatever was produced, and the NEXT tick
    retries.  The extractor's own per-camera ledger + S3 ongoing-state make
    re-sweeps cheap and idempotent (an already-handled raw key is skipped), so
    retrying never re-extracts or double-uploads a clip.
    """
    try:
        from train_extraction import run_extraction_service as EXT
        from train_extraction import driver as EXT_D
    except Exception as e:
        log.error("[EXTRACT] extraction package unavailable -- skipping raw sweep: %s",
                  e, exc_info=True)
        return {}

    # Ensure the extraction classifier models are present locally (pull from the
    # models bucket on a fresh host) BEFORE building any extractor -- done here at
    # the orchestrator boundary so the standalone train_extraction package needs
    # no change and no core import.  A present model is an instant no-op.
    try:
        from core import model_sync
        model_sync.ensure_extraction_models()
    except Exception as e:
        log.error("[MODEL] extraction model sync error (continuing to sweep): %s", e)

    log.info("[RAW] scanning raw bucket for new clips (all 4 cameras) ...")
    totals: Dict[str, Dict[str, int]] = {}
    for camera in EXT_D.ALL_CAMERAS:
        if _SHUTDOWN_REQUESTED:
            log.info("[EXTRACT] shutdown requested -- stopping raw sweep")
            break
        log.info("[EXTRACT] %s sweep start", camera)
        try:
            r = EXT.sweep_camera(camera)          # list raw -> trim -> upload trimmed
        except Exception as e:
            log.error("[EXTRACT] %s sweep crashed (continuing): %s",
                      camera, e, exc_info=True)
            continue
        totals[camera] = r
        if r.get("trains"):
            log.info("[UPLOAD] complete-train <- %s : %d trimmed clip(s) uploaded",
                     camera, r["trains"])
        log.info("[EXTRACT] %s done: listed=%d new=%d trains=%d errors=%d",
                 camera, r.get("listed", 0), r.get("new", 0),
                 r.get("trains", 0), r.get("errors", 0))
    n_trains = sum(t.get("trains", 0) for t in totals.values())
    log.info("[RAW] sweep complete: %d new trimmed clip(s) across %d camera(s)",
             n_trains, len(totals))
    return totals


def run_auto(*args, **kwargs):
    """Continuous S3 polling loop -- manifest-driven, resumable, multi-batch.

    Each tick: discover input videos, attach them to active BatchManifests (or
    create new ones), then advance() every active batch's state machine.  Only
    TERMINAL batches are written to processed_batches.json; every waiting batch
    persists as a manifest and is revisited on the next poll.
    """
    try:
        from orchestrator.train_batch_manager import (
            list_candidate_videos, load_batch_state, save_batch_state,
            DEFAULT_BATCH_TOLERANCE_SEC,
        )
        from orchestrator import batch_manifest as BM
        from orchestrator import lifecycle_runner as LR
        from core.lifecycle import is_terminal, terminal_batch_status
    except Exception as e:
        log.error("[ORCH] continuous polling unavailable: %s", e, exc_info=True)
        return 3

    import boto3

    _install_signal_handlers()
    s3 = boto3.client("s3", region_name=C.S3_REGION)
    state_loc = f"{C.S3_OUTPUT_BUCKET}/{C.S3_STATE_KEY}"

    workspace_root = kwargs.get("workspace") or DEFAULT_WORKSPACE_PARENT
    os.makedirs(workspace_root, exist_ok=True)
    poll_interval = kwargs.get("poll_interval", 60)
    run_once      = kwargs.get("run_once", False)
    force_key     = kwargs.get("force_batch_key")
    run_extraction = kwargs.get("run_extraction", CFG.AUTO_RUN_EXTRACTION)

    if run_extraction and not force_key:
        log.info("[ORCH] in-process extraction ENABLED -- this single process "
                 "runs RAW -> trimmed -> complete-train -> reports (no separate "
                 "extraction service needed). Disable with --skip-extraction or "
                 "WAGONEYE_AUTO_RUN_EXTRACTION=false.")
    else:
        log.info("[ORCH] in-process extraction DISABLED -- INSPECTION-ONLY: polling "
                 "input bucket %s (a separate instance performs extraction; this "
                 "process never reads the raw bucket).", C.S3_INPUT_BUCKET)

    ctx = LR.RunContext(
        workspace_root=workspace_root,
        recon_models_dir=kwargs.get("recon_models_dir") or DEFAULT_RECON_MODELS_DIR,
        feat_models_dir=kwargs.get("feat_models_dir") or DEFAULT_FEAT_MODELS_DIR,
        s3_client=s3,
        feature_config=kwargs.get("feature_config") or FeatureConfig.all_on(),
        skip_upload=kwargs.get("skip_upload", False),
        skip_email=kwargs.get("skip_email", False),
        repo_root=_REPO_ROOT,
    )

    processed = load_batch_state(s3, state_loc)
    log.info("[ORCH] workspace: %s | terminal batches so far: %d",
             workspace_root, len(processed))

    while not _SHUTDOWN_REQUESTED:
        try:
            # ---- Stage A: RAW -> trimmed extraction FIRST (never bypass it) ----
            # Runs before complete-train discovery so the correct order holds:
            #   RAW bucket -> extraction -> complete-train -> discovery -> lifecycle.
            # Skipped for --batch replay (force_key) and when extraction is off.
            if run_extraction and not force_key:
                _extraction_sweep()

            actives = {m.batch_key: m for m in
                       BM.list_active_manifests(s3, processed_batches=processed)}

            for cv in list_candidate_videos(s3):
                _attach_candidate(cv, actives, processed, ctx,
                                  DEFAULT_BATCH_TOLERANCE_SEC)

            keys = ([force_key] if force_key
                    else sorted(actives.keys()))
            any_terminal = False
            for key in keys:
                m = actives.get(key)
                if m is None:
                    continue
                m = LR.advance(m, ctx)
                if is_terminal(m.lifecycle_status):
                    processed[key] = (m.terminal_status
                                      or terminal_batch_status(m.lifecycle_status))
                    save_batch_state(s3, state_loc, processed)
                    any_terminal = True
                    log.info("[ORCH] batch %s TERMINAL: %s", key, processed[key])

            if run_once:
                return 0
            if not actives:
                log.info("[ORCH] no active batches; sleeping %ds", poll_interval)
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            log.info("[ORCH] interrupted")
            return 0
        except Exception as e:
            log.error("[ORCH] unhandled tick error: %s", e, exc_info=True)
            if run_once:
                return 3
            time.sleep(poll_interval)

    log.info("[ORCH] shutdown complete")
    return 0


# -----------------------------------------------------------------------------
# Local mode
# -----------------------------------------------------------------------------

def run_local(
    *,
    local_inputs: str,
    batch_key: Optional[str],
    workspace: Optional[str],
    recon_models_dir: str,
    feat_models_dir: str,
    feature_config: Optional[FeatureConfig] = None,
) -> int:
    if not os.path.isdir(local_inputs):
        log.error("ERROR: %s does not exist", local_inputs)
        return 2
    video_paths = scan_local_video_dir(local_inputs)
    missing = [c for c in C.ALL_CAMERAS if c not in video_paths]
    if missing:
        log.error("ERROR: missing videos for %s in %s.", missing, local_inputs)
        return 2
    batch = build_local_batch(video_paths, batch_key=batch_key)
    workspace = workspace or DEFAULT_WORKSPACE_PARENT

    # No s3 client needed; skip_upload=True
    class _NoopS3:
        def download_file(self, *a, **kw):
            raise RuntimeError("s3 download invoked in --local-only mode")
        def upload_file(self, *a, **kw):
            return None

    outcome = process_batch(
        batch=batch, workspace_root=workspace,
        recon_models_dir=recon_models_dir,
        feat_models_dir=feat_models_dir,
        s3_client=_NoopS3(),
        skip_upload=True, skip_email=True,
        feature_config=feature_config or FeatureConfig.all_on(),
    )
    if outcome.report_pdf_path:
        log.info("[LOCAL] PDF : %s", outcome.report_pdf_path)
    if outcome.report_json_path:
        log.info("[LOCAL] JSON: %s", outcome.report_json_path)
    for cam, path in outcome.camera_pdf_paths.items():
        log.info("[LOCAL] %-13s PDF: %s", cam, path)
    for cam, path in outcome.processed_video_paths.items():
        log.info("[LOCAL] VIDEO  %s: %s", cam, path)
    return 0 if outcome.final_status in (C.BATCH_COMPLETED,
                                          C.BATCH_COMPLETED_PARTIAL) else 3


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orchestrator.master_runner",
        description="WagonEye v4 train-state-native orchestrator.",
    )
    p.add_argument("--auto",  action="store_true", help="continuous S3 polling")
    p.add_argument("--once",  action="store_true", help="one batch then exit")
    p.add_argument("--batch", default=None,
                   help="force a specific batch_key (replay / debug)")
    p.add_argument("--local-only",   action="store_true",
                   help="skip S3 entirely; videos come from --local-inputs")
    p.add_argument("--local-inputs", default=CFG.LOCAL_INPUTS_DIR,
                   help="folder to scan in --local-only mode "
                        "(default: <repo>/local_inputs; "
                        "override with WAGONEYE_LOCAL_INPUTS_DIR)")
    p.add_argument("--workspace",    default=None,
                   help="workspace root (default: <repo>/batch_outputs; "
                        "override with WAGONEYE_WORKSPACE_ROOT)")
    p.add_argument("--recon-models-dir", default=DEFAULT_RECON_MODELS_DIR)
    p.add_argument("--feat-models-dir",  default=DEFAULT_FEAT_MODELS_DIR)
    p.add_argument("--poll-interval",    type=int,   default=60)
    p.add_argument("--partial-wait",     type=float, default=30.0)
    p.add_argument("--skip-upload",      action="store_true")
    p.add_argument("--skip-email",       action="store_true")
    p.add_argument("--skip-extraction",  action="store_true",
                   help="do NOT run the in-process raw->trimmed extraction sweep "
                        "in --auto (poll complete-train only; use this if a "
                        "separate train_extraction producer service is running)")
    p.add_argument("--disable-features", default="",
                   help="comma-separated feature keys to turn OFF "
                        "(door,ocr,load,damage); skips the interactive prompt")
    p.add_argument("--no-interactive",   action="store_true",
                   help="never prompt for feature config (force all-ON unless "
                        "--disable-features given)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Initialize logging before any stage runs.  A rotating file handler under
    # WAGONEYE_LOG_DIR (default <repo>/logs) plus stdout so foreground runs and
    # `journalctl`/`tail -f` both see the same structured, timestamped lines.
    setup_logging()
    log.info("WagonEye v4 orchestrator starting (device=%s)", _DEVICE)

    # ---- Fail-fast configuration validation + redacted startup summary ----
    if args.local_only:
        mode = "local"
    elif args.batch:
        mode = "batch"
    elif args.once:
        mode = "once"
    else:
        mode = "auto"
    log.info("%s", CFG.startup_summary(mode=mode))
    cfg_errors = CFG.validate_config(
        mode=mode,
        skip_upload=args.skip_upload or args.local_only,
        skip_email=args.skip_email or args.local_only,
    )
    if cfg_errors:
        for e in cfg_errors:
            log.error("[CONFIG] %s", e)
        log.error("[CONFIG] %d configuration error(s) -- refusing to start.",
                  len(cfg_errors))
        return 2

    # Continuous --auto polling is a daemon: never prompt there.  Interactive
    # toggling is only offered for --local-only / --once / --batch foreground
    # runs, and only when stdin is a real TTY (resolve_feature_config gates it).
    interactive = (not args.no_interactive) and (not args.auto)
    feature_config = resolve_feature_config(
        disable_features=args.disable_features,
        interactive=interactive,
    )

    if args.local_only:
        return run_local(
            local_inputs=args.local_inputs,
            batch_key=args.batch,
            workspace=args.workspace,
            recon_models_dir=args.recon_models_dir,
            feat_models_dir=args.feat_models_dir,
            feature_config=feature_config,
        )

    if not (args.auto or args.once or args.batch):
        log.error("ERROR: pass --auto, --once, --batch <key>, or --local-only")
        return 2

    return run_auto(
        workspace=args.workspace,
        recon_models_dir=args.recon_models_dir,
        feat_models_dir=args.feat_models_dir,
        poll_interval=args.poll_interval,
        partial_wait_minutes=args.partial_wait,
        run_once=(args.once or bool(args.batch)),
        force_batch_key=args.batch,
        skip_upload=args.skip_upload,
        skip_email=args.skip_email,
        run_extraction=(not args.skip_extraction) and CFG.AUTO_RUN_EXTRACTION,
        feature_config=feature_config,
    )


if __name__ == "__main__":
    sys.exit(main())
