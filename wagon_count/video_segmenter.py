"""
video_segmenter.py
==================

Phase-1 visualization + frame extraction.

Two responsibilities:

1) render_processed_video(...)
     Take a camera's raw video + its LocalCameraTracks + the final
     GlobalTrainState, and produce a processed MP4 with overlay:
       - tracked gap bounding boxes (cyan)
       - vertical wagon-boundary lines at fused gap centers (magenta)
       - the current global wagon id and classification
       - per-frame confidence of the current gap (if any)

2) extract_wagon_frames(...)
     For one camera, write every frame of each global wagon to disk
     as JPEGs under  output/{camera_id}/{GW_n}/frame_NNNNNN.jpg .

Frame range derivation for non-master cameras:
    The four input videos are assumed to share a t=0 alignment (they
    were trimmed to the same train pass by the upstream system).
    Each global wagon's master time window [t_start, t_end] is mapped
    to the local camera by   local_frame = round(t * local_fps)  ,
    clipped to that camera's total_frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

from global_train_state import (
    GapEvent,
    LocalCameraTracks,
    GlobalWagon,
    GlobalTrainState,
    SegmentClass,
    MASTER_CAMERA,
)


# -----------------------------------------------------------------------------
# Colors (BGR)
# -----------------------------------------------------------------------------
_GAP_COLOR = (255, 255, 0)         # cyan      - raw YOLO detection (single frame)
_TRACKED_GAP_COLOR = (0, 255, 255) # yellow    - tracked gap (interpolated bbox)
_BOUNDARY_COLOR = (255, 0, 255)    # magenta   - fused wagon boundary flash
_WAGON_TEXT_COLOR = (0, 255, 0)    # green
_INFO_TEXT_COLOR = (255, 255, 255) # white
_PANEL_BG = (0, 0, 0)

_CLASS_COLORS = {
    SegmentClass.ENGINE: (0, 165, 255),     # orange
    SegmentClass.WAGON: (0, 255, 0),         # green
    SegmentClass.BRAKE_VAN: (0, 0, 255),     # red
    SegmentClass.UNKNOWN: (128, 128, 128),   # gray
}


# =============================================================================
# Helpers: master-time <-> local-frame mapping
# =============================================================================

def map_global_wagon_to_local_frames(
    wagon: GlobalWagon,
    local_fps: float,
    local_total_frames: int,
) -> Tuple[int, int]:
    """Convert one global wagon's master time window to the camera's frame range.

    Returns inclusive [start, end] indices clipped into the camera.
    """
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round(wagon.start_time * local_fps))
    ef = int(round(wagon.end_time * local_fps)) - 1
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


def build_camera_wagon_frame_map(
    state: GlobalTrainState,
    local_tracks: LocalCameraTracks,
) -> Dict[str, Tuple[int, int]]:
    """For a given camera, map each global_id -> (start_frame, end_frame)."""
    out: Dict[str, Tuple[int, int]] = {}
    for w in state.wagons:
        out[w.global_id] = map_global_wagon_to_local_frames(
            w, local_tracks.fps, local_tracks.total_frames,
        )
    return out


# =============================================================================
# OVERLAY RENDERER
# =============================================================================

def _interp_gap_bbox(gap: GapEvent, frame_idx: int) -> Optional[List[float]]:
    """Interpolate a tracked gap's bounding box at an arbitrary frame.

    Uses the gap's per-hit ``hit_frames`` + ``bbox_history`` (recorded by
    the tracker).  Returns ``None`` if the gap has no usable bbox history,
    or if ``frame_idx`` is outside the gap's [start_frame, end_frame] span.

    Strategy:
        - Before the first hit  -> clamp to first bbox
        - After the last hit    -> clamp to last bbox
        - Between two hits      -> linear interpolation (component-wise)
    """
    if not gap.bbox_history or not gap.hit_frames:
        return None
    if frame_idx < gap.start_frame or frame_idx > gap.end_frame:
        return None

    hf = gap.hit_frames
    bh = gap.bbox_history
    # bh and hf are parallel lists.  hf is monotonically non-decreasing
    # because hits are appended in chronological order.
    if frame_idx <= hf[0]:
        return list(bh[0])
    if frame_idx >= hf[-1]:
        return list(bh[-1])
    # Find the bracketing pair (hf[i], hf[i+1])
    for i in range(len(hf) - 1):
        f0, f1 = hf[i], hf[i + 1]
        if f0 <= frame_idx <= f1:
            if f1 == f0:
                return list(bh[i])
            t = (frame_idx - f0) / (f1 - f0)
            b0, b1 = bh[i], bh[i + 1]
            return [b0[j] + t * (b1[j] - b0[j]) for j in range(4)]
    return list(bh[-1])


def _draw_info_panel(
    frame: np.ndarray,
    lines: List[Tuple[str, Tuple[int, int, int]]],
) -> None:
    """Draw a translucent panel on the top-left with `lines` of text."""
    h, w = frame.shape[:2]
    panel_w = 380
    panel_h = 22 * len(lines) + 16
    panel_h = max(panel_h, 30)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def render_processed_video(
    *,
    local_tracks: LocalCameraTracks,
    state: GlobalTrainState,
    output_path: str,
    draw_raw_detections: bool = True,
    verbose: bool = True,
) -> str:
    """Render the overlay video for ONE camera.

    Returns the output path.  Creates parent directory if needed.
    """
    if local_tracks.fps <= 0:
        raise ValueError(f"Cannot render: invalid fps for {local_tracks.camera_id}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    cap = cv2.VideoCapture(local_tracks.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for rendering: {local_tracks.video_path}")

    fps = local_tracks.fps
    width = local_tracks.width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = local_tracks.height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # mp4v is broadly compatible; the legacy code uses 'avc1' which is
    # less portable when openh264 isn't installed.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer at {output_path}")

    # Precompute frame -> active GapEvent (camera's own tracked gaps)
    frame_to_active_gap: Dict[int, GapEvent] = {}
    for g in local_tracks.gaps:
        for f in range(g.start_frame, g.end_frame + 1):
            frame_to_active_gap[f] = g

    # Per-camera global_id -> [start_local_frame, end_local_frame]
    wagon_ranges = build_camera_wagon_frame_map(state, local_tracks)

    # Build a frame_idx -> GlobalWagon map for fast lookup, AND a list of
    # boundary frames at which to draw the magenta line.
    # NOTE: this is the camera's *projected* wagon boundary, derived from
    # master timing -- so even cameras that locally missed a gap will draw
    # the corrected boundary thanks to fusion.
    frame_to_wagon: Dict[int, GlobalWagon] = {}
    boundary_frames: List[int] = []
    for w in state.wagons:
        sf, ef = wagon_ranges[w.global_id]
        for f in range(sf, ef + 1):
            frame_to_wagon[f] = w
        boundary_frames.append(sf)
    # The very first boundary at frame 0 is implicit; skip drawing it
    boundary_frames = sorted(set(b for b in boundary_frames if b > 0))

    if verbose:
        print(f"[RENDER/{local_tracks.camera_id}] writing -> {output_path}")
        print(f"  {local_tracks.total_frames} frames, {len(state.wagons)} wagons projected")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MAGENTA wagon-boundary flash + centered "GW_BOUNDARY" label
        # whenever the frame is within +/-3 of any projected boundary frame.
        for b in boundary_frames:
            if abs(b - frame_idx) <= 3:
                cv2.line(frame, (0, 0), (frame.shape[1], 0), _BOUNDARY_COLOR, 4)
                cv2.line(frame, (0, frame.shape[0] - 1),
                         (frame.shape[1], frame.shape[0] - 1),
                         _BOUNDARY_COLOR, 4)
                # Centered banner: "GW_BOUNDARY"
                label = "GW_BOUNDARY"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = max(0, (frame.shape[1] - tw) // 2)
                ty = th + 16
                # filled magenta background tab for legibility
                cv2.rectangle(frame, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8),
                              _BOUNDARY_COLOR, -1)
                cv2.putText(frame, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                break

        # Active local gap (for track id + confidence in the info panel
        # AND for tracked-bbox interpolation below)
        active_gap = frame_to_active_gap.get(frame_idx)

        # YELLOW tracked-gap bbox -- drawn on EVERY frame inside any
        # confirmed gap's span, interpolated between hit frames.  Even when
        # YOLO does not fire on a given frame, the tracker still knows the
        # gap is there.  This is what guarantees a continuous bbox on every
        # camera, end-to-end through the gap's lifespan.
        # Label uses an explicit two-line "TRACKED_GAP #N / conf=X" format.
        if active_gap is not None:
            interp_bbox = _interp_gap_bbox(active_gap, frame_idx)
            if interp_bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in interp_bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), _TRACKED_GAP_COLOR, 2)
                # Two-line label above the bbox
                cv2.putText(frame, f"TRACKED_GAP #{active_gap.track_id}",
                            (x1, max(0, y1 - 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TRACKED_GAP_COLOR, 2, cv2.LINE_AA)
                cv2.putText(frame, f"conf={active_gap.confidence:.2f}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TRACKED_GAP_COLOR, 2, cv2.LINE_AA)

        # CYAN raw YOLO per-frame detections, drawn ON TOP of the tracked
        # box.  Label uses "RAW / conf=X" so it is visually distinguishable
        # from the tracked box even if both colors are present.
        if draw_raw_detections:
            for det in local_tracks.raw_frame_detections.get(frame_idx, []):
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), _GAP_COLOR, 2)
                # Two-line label below the top edge so it doesn't collide
                # with the tracked-gap label drawn above the box.
                cv2.putText(frame, "RAW", (x1 + 4, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GAP_COLOR, 2, cv2.LINE_AA)
                cv2.putText(frame, f"conf={det['confidence']:.2f}",
                            (x1 + 4, y1 + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GAP_COLOR, 2, cv2.LINE_AA)

        # Current global wagon
        current_wagon = frame_to_wagon.get(frame_idx)

        # Info panel -- explicit "label: value" layout for debugging.
        info_lines: List[Tuple[str, Tuple[int, int, int]]] = []
        info_lines.append((f"Camera:               {local_tracks.camera_id}",
                           _INFO_TEXT_COLOR))
        info_lines.append((f"Frame:                {frame_idx} / {local_tracks.total_frames}",
                           _INFO_TEXT_COLOR))
        if current_wagon is not None:
            cls = current_wagon.classification
            color = _CLASS_COLORS.get(cls, _WAGON_TEXT_COLOR)
            info_lines.append((f"Current Wagon:        {current_wagon.global_id}",
                               _WAGON_TEXT_COLOR))
            info_lines.append((f"Classification:       {cls}",
                               color))
        else:
            info_lines.append(("Current Wagon:        —", _INFO_TEXT_COLOR))
            info_lines.append(("Classification:       —", _INFO_TEXT_COLOR))
        info_lines.append((f"Total Global Wagons:  {state.total_wagons}",
                           _INFO_TEXT_COLOR))
        if state.fallback_used:
            info_lines.append(("FALLBACK MODE (pure RIGHT_UP)", (0, 0, 255)))

        _draw_info_panel(frame, info_lines)

        writer.write(frame)
        frame_idx += 1
        if verbose and frame_idx % 500 == 0:
            print(f"  ... rendered {frame_idx} frames")

    cap.release()
    writer.release()
    if verbose:
        print(f"[RENDER/{local_tracks.camera_id}] done ({frame_idx} frames)")
    return output_path


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def extract_wagon_frames(
    *,
    local_tracks: LocalCameraTracks,
    state: GlobalTrainState,
    output_root: str,
    jpeg_quality: int = 90,
    every_nth_frame: int = 1,
    verbose: bool = True,
) -> Dict[str, int]:
    """Write per-wagon frame folders for ONE camera.

    Output layout:
        {output_root}/{camera_id}/{global_id}/frame_NNNNNN.jpg

    Parameters
    ----------
    every_nth_frame : keep 1 frame out of every N (1 = all frames).
        Useful for trimming disk usage; defaults to keeping everything.

    Returns
    -------
    dict  global_id -> number of frames written
    """
    if every_nth_frame < 1:
        raise ValueError(f"every_nth_frame must be >= 1 (got {every_nth_frame})")

    cap = cv2.VideoCapture(local_tracks.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for frame extraction: {local_tracks.video_path}")

    cam_root = os.path.join(output_root, local_tracks.camera_id)
    os.makedirs(cam_root, exist_ok=True)

    wagon_ranges = build_camera_wagon_frame_map(state, local_tracks)
    # frame_idx -> (global_id, wagon_dir)
    frame_to_target: Dict[int, Tuple[str, str]] = {}
    counts: Dict[str, int] = {}
    for w in state.wagons:
        sf, ef = wagon_ranges[w.global_id]
        wdir = os.path.join(cam_root, w.global_id)
        os.makedirs(wdir, exist_ok=True)
        counts[w.global_id] = 0
        for f in range(sf, ef + 1):
            # Last assignment wins; wagons must not overlap so this is
            # only ever a one-frame seam issue.
            frame_to_target[f] = (w.global_id, wdir)

    if verbose:
        print(f"[EXTRACT/{local_tracks.camera_id}] writing into {cam_root} "
              f"({len(state.wagons)} wagon folders)")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        target = frame_to_target.get(frame_idx)
        if target is not None and (frame_idx % every_nth_frame == 0):
            gid, wdir = target
            out_path = os.path.join(wdir, f"frame_{frame_idx:06d}.jpg")
            ok = cv2.imwrite(out_path, frame, encode_params)
            if ok:
                counts[gid] = counts.get(gid, 0) + 1
                written += 1
        frame_idx += 1
        if verbose and frame_idx % 1000 == 0:
            print(f"  ... scanned {frame_idx} frames, written {written}")
    cap.release()

    if verbose:
        print(f"[EXTRACT/{local_tracks.camera_id}] done. "
              f"Total frames written: {written}")
        for gid in sorted(counts.keys(), key=_gw_sort_key):
            print(f"   {gid:>6}: {counts[gid]} frames")
    return counts


def _gw_sort_key(gid: str):
    # 'GW_12' -> 12 ; preserves natural ordering
    try:
        return int(gid.split("_", 1)[1])
    except Exception:
        return 0
