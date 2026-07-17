"""Stage 2 -- single-pass per-video frame extraction.

Takes the authoritative GlobalTrainState (Stage 1 output) plus the 4
source videos and writes per-wagon, per-camera JPEG folders:

    wagon_cache/
        GW_1/
            right_up/
                frame_000023.jpg
                frame_000024.jpg
                ...
            left_up/...
            right_up_top/...
            left_up_top/...
        GW_2/...

For each camera we open `cv2.VideoCapture` ONCE, walk it linearly, and
write JPEGs to whichever GW_n bucket the current local frame falls into.
No video decoding happens downstream of this stage.

Mapping master_time → local_frame is direct:

    local_start = round(GW.start_time * local_fps)
    local_end   = round(GW.end_time   * local_fps) - 1

clipped into [0, total_frames - 1].  This is the same convention used
by the wagon_count package itself, so the cache stays bit-equivalent to
what wagon_count's --no-frames=False would have produced.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.logging_setup import get_logger

log = get_logger("materializer")

# Bump when the on-disk cache layout / extraction semantics change so stale
# caches from an older materializer are rebuilt rather than reused.
MATERIALIZER_SCHEMA_VERSION = 1
_MARKER_DIR = ".materialized"


# -----------------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------------

@dataclass
class CacheBuildResult:
    cache_root: str
    frames_written: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # {gw_id -> {camera_id -> n_frames}}

    per_camera_total: Dict[str, int] = field(default_factory=dict)
    missing_cameras: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def total_frames(self) -> int:
        return sum(self.per_camera_total.values())


# -----------------------------------------------------------------------------
# Master-time -> local-frame mapping
# -----------------------------------------------------------------------------

def _wagon_local_range(
    wagon: GlobalWagon, local_fps: float, local_total_frames: int,
) -> Tuple[int, int]:
    """Convert a GlobalWagon's master-clock time window to inclusive
    [start, end] frame indices in the camera's local timeline."""
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round(wagon.start_time * local_fps))
    ef = int(round(wagon.end_time   * local_fps)) - 1
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


# -----------------------------------------------------------------------------
# Per-camera worker
# -----------------------------------------------------------------------------

def _extract_one_camera(
    *,
    camera_id: str,
    video_path: str,
    state: GlobalTrainState,
    local_fps: float,
    cache_root: str,
    jpeg_quality: int,
    verbose: bool,
) -> Tuple[str, Dict[str, int]]:
    """Open the video once, walk frames sequentially, dispatch by GW range."""

    if not os.path.exists(video_path):
        if verbose:
            log.warning("[STAGE2/%s] video missing: %s", camera_id, video_path)
        return camera_id, {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            log.warning("[STAGE2/%s] cv2 could not open %s", camera_id, video_path)
        return camera_id, {}

    # Reported total; some containers lie -- we just use it for clipping.
    total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Build a frame_idx -> (gw_id, dst_dir) map
    frame_to_target: Dict[int, Tuple[str, str]] = {}
    counts: Dict[str, int] = {}
    cam_folder = C.CAMERA_FOLDER[camera_id]
    for gw in state.wagons:
        sf, ef = _wagon_local_range(gw, local_fps, total_meta or 10**7)
        if ef < sf:
            continue
        dst = os.path.join(cache_root, gw.global_id, cam_folder)
        os.makedirs(dst, exist_ok=True)
        counts[gw.global_id] = 0
        for f in range(sf, ef + 1):
            # Wagons must not overlap; last-write-wins is a harmless
            # one-frame seam if they happen to touch.
            frame_to_target[f] = (gw.global_id, dst)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    frame_idx = 0
    written = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        target = frame_to_target.get(frame_idx)
        if target is not None:
            gw_id, dst = target
            out_path = os.path.join(dst, f"frame_{frame_idx:06d}.jpg")
            if cv2.imwrite(out_path, frame, encode_params):
                counts[gw_id] = counts.get(gw_id, 0) + 1
                written += 1
        frame_idx += 1
        if verbose and frame_idx % 1000 == 0:
            log.info("  [STAGE2/%s] scanned %d frames, wrote %d",
                     camera_id, frame_idx, written)
    cap.release()
    elapsed = time.time() - t0

    if verbose:
        log.info("[STAGE2/%s] done in %.1fs  frames_written=%d",
                 camera_id, elapsed, written)

    return camera_id, counts


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build(
    *,
    state: GlobalTrainState,
    video_paths: Dict[str, str],
    per_camera_fps: Dict[str, float],
    cache_root: str,
    jpeg_quality: int = C.JPEG_QUALITY,
    parallel: bool = True,
    verbose: bool = True,
) -> CacheBuildResult:
    """Extract per-wagon JPEG folders for ALL cameras.

    Cameras with a missing source video produce an empty cache subtree
    -- they end up in `result.missing_cameras` but the stage still
    succeeds. Each remaining camera that successfully extracts > 0
    frames counts as "present".
    """
    os.makedirs(cache_root, exist_ok=True)

    if verbose:
        log.info("[STAGE2] building wagon_cache at %s", cache_root)
        log.info("[STAGE2] wagons=%d  cameras=%s",
                 len(state.wagons), list(video_paths.keys()))

    result = CacheBuildResult(cache_root=cache_root)

    workload = []
    for cam in C.ALL_CAMERAS:
        if cam not in video_paths:
            result.missing_cameras.append(cam)
            continue
        local_fps = per_camera_fps.get(cam) or state.master_fps or 25.0
        workload.append((cam, video_paths[cam], local_fps))

    t_start = time.time()

    if parallel:
        with ThreadPoolExecutor(max_workers=min(4, len(workload) or 1)) as ex:
            futs = {
                ex.submit(
                    _extract_one_camera,
                    camera_id=cam, video_path=path,
                    state=state, local_fps=fps,
                    cache_root=cache_root,
                    jpeg_quality=jpeg_quality,
                    verbose=verbose,
                ): cam
                for (cam, path, fps) in workload
            }
            for f in as_completed(futs):
                cam_id, counts = f.result()
                for gw_id, n in counts.items():
                    result.frames_written.setdefault(gw_id, {})[cam_id] = n
                result.per_camera_total[cam_id] = sum(counts.values())
    else:
        for (cam, path, fps) in workload:
            cam_id, counts = _extract_one_camera(
                camera_id=cam, video_path=path,
                state=state, local_fps=fps,
                cache_root=cache_root,
                jpeg_quality=jpeg_quality,
                verbose=verbose,
            )
            for gw_id, n in counts.items():
                result.frames_written.setdefault(gw_id, {})[cam_id] = n
            result.per_camera_total[cam_id] = sum(counts.values())

    result.elapsed_seconds = time.time() - t_start

    if verbose:
        log.info("[STAGE2] done in %.1fs  total_frames=%d",
                 result.elapsed_seconds, result.total_frames())
        for cam in C.ALL_CAMERAS:
            n = result.per_camera_total.get(cam, 0)
            status = "missing" if cam in result.missing_cameras else f"{n} frames"
            log.info("  %-14s %s", cam, status)

    return result


# -----------------------------------------------------------------------------
# Idempotent, per-camera build (incremental lifecycle)  [Commit 3]
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marker_path(cache_root: str, camera_id: str) -> str:
    return os.path.join(cache_root, _MARKER_DIR, f"{camera_id}.json")


def _read_marker(cache_root: str, camera_id: str) -> Optional[dict]:
    p = _marker_path(cache_root, camera_id)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_marker(cache_root: str, camera_id: str, payload: dict) -> None:
    d = os.path.join(cache_root, _MARKER_DIR)
    os.makedirs(d, exist_ok=True)
    path = _marker_path(cache_root, camera_id)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{camera_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _camera_cache_nonempty(cache_root: str, state: GlobalTrainState, cam_folder: str) -> bool:
    for gw in state.wagons:
        d = os.path.join(cache_root, gw.global_id, cam_folder)
        if os.path.isdir(d) and any(f.endswith(".jpg") for f in os.listdir(d)):
            return True
    return False


def _swap_camera_cache(temp_root: str, cache_root: str, cam_folder: str,
                       state: GlobalTrainState) -> None:
    """Atomically replace one camera's per-wagon folders from a temp build.

    Only the freshly-built camera folders are moved in; the previous version is
    removed just before its replacement lands, so a crash mid-swap leaves at
    worst one wagon's folder for one camera in a rebuildable state -- and a
    FAILED build never reaches here at all (previous cache preserved)."""
    for gw in state.wagons:
        src = os.path.join(temp_root, gw.global_id, cam_folder)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(cache_root, gw.global_id, cam_folder)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)


def build_cameras(
    *,
    state: GlobalTrainState,
    video_paths: Dict[str, str],
    per_camera_fps: Dict[str, float],
    cache_root: str,
    global_state_version: str,
    camera_etags: Optional[Dict[str, str]] = None,
    jpeg_quality: int = C.JPEG_QUALITY,
    force: bool = False,
    verbose: bool = True,
) -> CacheBuildResult:
    """Materialize ONLY the given cameras, idempotently.

    A camera is SKIPPED when its marker matches the current
    (etag, global_state_version, materializer_schema_version) AND its cache is
    non-empty.  Otherwise it is (re)built into a temp camera-scoped directory
    and atomically swapped into place -- so a changed ETag rebuilds just that
    camera and a failed rebuild never destroys the previous valid cache.
    """
    os.makedirs(cache_root, exist_ok=True)
    camera_etags = camera_etags or {}
    result = CacheBuildResult(cache_root=cache_root)

    for cam in C.ALL_CAMERAS:
        if cam not in video_paths:
            result.missing_cameras.append(cam)
            continue
        cam_folder = C.CAMERA_FOLDER[cam]
        etag = camera_etags.get(cam)
        marker = _read_marker(cache_root, cam)
        up_to_date = (
            not force
            and marker is not None
            and marker.get("etag") == etag
            and marker.get("global_state_version") == global_state_version
            and marker.get("materializer_schema_version") == MATERIALIZER_SCHEMA_VERSION
            and marker.get("status") == "OK"
            and _camera_cache_nonempty(cache_root, state, cam_folder)
        )
        if up_to_date:
            if verbose:
                log.info("[STAGE2/%s] up-to-date (etag=%s v=%s) -- skip",
                         cam, str(etag)[:12], str(global_state_version)[:12])
            n = int(marker.get("frames_written", 0))
            result.per_camera_total[cam] = n
            continue

        local_fps = per_camera_fps.get(cam) or state.master_fps or 25.0
        temp_root = tempfile.mkdtemp(prefix=f".tmp_{cam}_", dir=cache_root)
        try:
            _cam, counts = _extract_one_camera(
                camera_id=cam, video_path=video_paths[cam], state=state,
                local_fps=local_fps, cache_root=temp_root,
                jpeg_quality=jpeg_quality, verbose=verbose,
            )
            frames_written = sum(counts.values())
            # atomic swap into the real cache
            _swap_camera_cache(temp_root, cache_root, cam_folder, state)
            for gw_id, n in counts.items():
                result.frames_written.setdefault(gw_id, {})[cam] = n
            result.per_camera_total[cam] = frames_written
            _write_marker(cache_root, cam, {
                "camera": cam,
                "source_key": video_paths[cam],
                "etag": etag,
                "local_fps": local_fps,
                "total_frames": state.master_total_frames,
                "global_state_version": global_state_version,
                "materializer_schema_version": MATERIALIZER_SCHEMA_VERSION,
                "status": "OK" if frames_written > 0 else "NO_FRAMES",
                "frames_written": frames_written,
                "completed_at": _now_iso(),
            })
            if verbose:
                log.info("[STAGE2/%s] materialized %d frames (etag=%s)",
                         cam, frames_written, str(etag)[:12])
        except Exception as e:
            log.error("[STAGE2/%s] build failed, previous cache preserved: %s",
                      cam, e, exc_info=True)
        finally:
            if os.path.isdir(temp_root):
                shutil.rmtree(temp_root, ignore_errors=True)

    result.elapsed_seconds = 0.0
    return result
