"""Stage 1 -- subprocess wrapper around wagon_count/run_global_count.py.

The wagon_count package owns:
    * gap detection per camera
    * cross-camera gap fusion
    * RIGHT_UP master classification
    * deterministic GW_n id assignment

We invoke it as a subprocess with `--no-frames` so we get
`global_train_state.json` + `per_camera_tracking.json` plus the per-camera
tracking-overlay mp4s under `<output_dir>/processed_videos/` (kept as debug
artifacts; the rich feature-overlay videos are produced separately by
`rendering.feature_overlay_renderer`).  The new materializer/ owns frame
extraction so the wagon_count step does not duplicate it.

Returns the parsed GlobalTrainState (lightweight dataclass from
core.global_state_loader) or raises on failure.  Caller is responsible
for marking the batch as `failed_no_global_state` when this raises.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

from core import constants as C
from core.global_state_loader import (
    GlobalTrainState, load_global_train_state, load_per_camera_fps,
)
from core.logging_setup import get_logger

log = get_logger("reconstruction")


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Outcome of one Stage-1 invocation."""
    state: GlobalTrainState
    per_camera_fps: Dict[str, float]
    state_json_path: str
    per_camera_tracking_path: str
    output_dir: str
    elapsed_seconds: float
    # Master-first reconstruction provenance (mirrored from the sealed state)
    master_camera: str = C.MASTER_CAMERA
    reconstruction_mode: str = ""
    participating_cameras: Optional[list] = None
    missing_at_reconstruction: Optional[list] = None
    support_cameras_present: Optional[list] = None
    support_fusion_used: bool = False
    support_gap_recoveries: int = 0
    reconstruction_confidence: float = 1.0
    fallback_master_used: bool = False
    sealing_reason: str = ""


class ReconstructionError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Subprocess driver
# -----------------------------------------------------------------------------

def _find_wagon_count_dir(repo_root: str) -> str:
    """Locate the wagon_count subpackage shipped next to this file."""
    candidate = os.path.join(repo_root, "wagon_count")
    if os.path.isfile(os.path.join(candidate, "run_global_count.py")):
        return candidate
    raise ReconstructionError(
        f"wagon_count/ not found under {repo_root}. "
        f"Expected {candidate}/run_global_count.py."
    )


_CAM_FLAG = {
    C.CAMERA_RIGHT_UP:     "--right_up",
    C.CAMERA_LEFT_UP:      "--left_up",
    C.CAMERA_RIGHT_UP_TOP: "--right_up_top",
    C.CAMERA_LEFT_UP_TOP:  "--left_up_top",
}


def run(
    *,
    video_paths: Dict[str, str],
    reconstruction_models_dir: str,
    output_dir: str,
    repo_root: str,
    master_camera: str = C.MASTER_CAMERA,
    allow_fallback_master: bool = False,
    python_executable: Optional[str] = None,
    timeout_seconds: int = 7200,
    verbose: bool = True,
) -> ReconstructionResult:
    """Run Stage 1 over the PRESENT cameras (master-first, subset-capable).

    Args:
        video_paths: {camera_id -> local path} for the cameras present NOW.
            Must include `master_camera`.  Absent cameras are simply not
            reconstructed -- their features attach later without a reseal.
        master_camera: which present camera drives the master timeline
            (default RIGHT_UP).  A non-RIGHT_UP master requires
            allow_fallback_master.
        allow_fallback_master: opt-in for a non-RIGHT_UP (LEFT_UP) master.
        reconstruction_models_dir: path to models/reconstruction/.
        output_dir: where wagon_count writes its outputs.
        repo_root: parent that contains the wagon_count/ subpackage.

    Raises:
        ReconstructionError on any failure (master absent, subprocess exit
        != 0, no JSON produced, zero wagons).
    """
    if master_camera not in video_paths:
        raise ReconstructionError(
            f"Stage 1 master camera {master_camera} is not present "
            f"(present: {sorted(video_paths)})")
    if master_camera != C.MASTER_CAMERA and not allow_fallback_master:
        raise ReconstructionError(
            f"master {master_camera} != {C.MASTER_CAMERA} requires "
            f"allow_fallback_master=True")
    for cam, p in video_paths.items():
        if not os.path.exists(p):
            raise ReconstructionError(f"Video for {cam} does not exist: {p}")

    # Ensure the reconstruction weights are present locally, pulling any missing
    # ones from the models bucket (wagon-eye-models) BEFORE the standalone
    # wagon_count subprocess -- which cannot import core -- resolves them by name.
    # A present model is a no-op; a failed sync leaves the dir as-is so the
    # existing "models dir does not exist / model not found" errors still fire.
    try:
        from core import model_sync
        model_sync.ensure_reconstruction_models(reconstruction_models_dir)
    except Exception as e:  # never let sync bookkeeping fail the stage
        log.warning("[STAGE1] reconstruction model sync error (continuing): %s", e)

    if not os.path.isdir(reconstruction_models_dir):
        raise ReconstructionError(
            f"reconstruction_models_dir does not exist: "
            f"{reconstruction_models_dir}")

    wagon_count_dir = _find_wagon_count_dir(repo_root)
    script = os.path.join(wagon_count_dir, "run_global_count.py")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [python_executable or sys.executable, script]
    for cam in C.ALL_CAMERAS:            # deterministic flag order
        if cam in video_paths:
            cmd += [_CAM_FLAG[cam], video_paths[cam]]
    cmd += ["--master-camera", master_camera]
    if allow_fallback_master:
        cmd += ["--allow-fallback-master"]
    cmd += [
        "--models-dir", reconstruction_models_dir,
        "--output",     output_dir,
        "--no-frames",      # materializer owns frame extraction
        # wagon_count's tracking overlay videos are kept (no --no-videos).
    ]

    if verbose:
        log.info("[STAGE1] launching wagon_count: %s", " ".join(cmd))

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=wagon_count_dir,
            capture_output=True, text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        # Convert to ReconstructionError so process_batch's Stage-1 handler
        # marks the batch failed_no_global_state and moves on, instead of an
        # uncaught exception that would leave the batch un-checkpointed and
        # retried forever on the next poll.
        elapsed = time.time() - t0
        log.error("[STAGE1] wagon_count timed out after %.0fs (limit %ds)",
                  elapsed, timeout_seconds)
        raise ReconstructionError(
            f"wagon_count subprocess timed out after {timeout_seconds}s"
        ) from e
    elapsed = time.time() - t0

    # Persist the FULL subprocess trace to a per-batch file (wagon_count stays
    # standalone -- it must not import core.logging_setup -- so its complete
    # output is captured here rather than by the subprocess itself).
    try:
        trace_path = os.path.join(output_dir, "stage1_wagon_count.log")
        with open(trace_path, "w", encoding="utf-8") as fh:
            fh.write(f"# cmd: {' '.join(cmd)}\n# exit={proc.returncode} "
                     f"elapsed={elapsed:.1f}s\n\n--- STDOUT ---\n")
            fh.write(proc.stdout or "")
            fh.write("\n\n--- STDERR ---\n")
            fh.write(proc.stderr or "")
    except Exception as e:  # never let logging bookkeeping fail the stage
        log.warning("[STAGE1] could not write stage1 trace file: %s", e)

    if verbose:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        log.info("[STAGE1] subprocess exit=%d (%.1fs)", proc.returncode, elapsed)
        if tail:
            log.info("[STAGE1] --- stdout tail ---\n%s\n[STAGE1] ----------------------",
                     tail)
        if proc.returncode != 0 and proc.stderr:
            log.error("[STAGE1] --- stderr tail ---\n%s\n[STAGE1] ----------------------",
                      proc.stderr[-2000:])

    if proc.returncode != 0:
        raise ReconstructionError(
            f"wagon_count subprocess exited {proc.returncode}"
        )

    state_path = os.path.join(output_dir, "global_train_state.json")
    if not os.path.isfile(state_path):
        raise ReconstructionError(
            f"wagon_count did not produce {state_path}"
        )

    state = load_global_train_state(state_path)
    if state.total_wagons <= 0:
        raise ReconstructionError(
            f"wagon_count returned total_wagons={state.total_wagons}; "
            f"aborting batch"
        )

    pcf_path = os.path.join(output_dir, "per_camera_tracking.json")
    per_camera_fps = load_per_camera_fps(pcf_path) if os.path.exists(pcf_path) else {}

    if verbose:
        log.info("[STAGE1] OK  total_wagons=%d  (E:%d  W:%d  B:%d)  master_fps=%.2f",
                 state.total_wagons, state.engine_count,
                 state.regular_wagon_count, state.brake_van_count,
                 state.master_fps)

    return ReconstructionResult(
        state=state,
        per_camera_fps=per_camera_fps,
        state_json_path=state_path,
        per_camera_tracking_path=pcf_path,
        output_dir=output_dir,
        elapsed_seconds=elapsed,
        master_camera=getattr(state, "master_camera", master_camera),
        reconstruction_mode=getattr(state, "reconstruction_mode", ""),
        participating_cameras=list(getattr(state, "participating_cameras", []) or []),
        missing_at_reconstruction=list(getattr(state, "missing_at_reconstruction", []) or []),
        support_cameras_present=list(getattr(state, "support_cameras_present", []) or []),
        support_fusion_used=bool(getattr(state, "support_fusion_used", False)),
        support_gap_recoveries=int(getattr(state, "support_gap_recoveries", 0) or 0),
        reconstruction_confidence=float(getattr(state, "reconstruction_confidence", 1.0) or 1.0),
        fallback_master_used=bool(getattr(state, "fallback_master_used", False)),
        sealing_reason=getattr(state, "sealing_reason", "") or "",
    )
