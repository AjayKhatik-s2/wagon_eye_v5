"""Feature overlay renderer  --  visualization-only, LEGACY-PARITY.

Produces one overlay mp4 per camera (4 total) whose visual appearance clones
the legacy WagonEye `_tracked.mp4` output as closely as possible:

    Side cameras (RIGHT_UP / LEFT_UP) reproduce the legacy DOOR annotation
    (old_system/RIGHT_UP/door_processor.py `_annotate_frame`):
        * per confirmed track: a state-coloured box (3px stroke when OPEN,
          else 2px), a filled label bar with black "Door {id}: {STATE}"
          text, the raw last-frame confidence printed below the box, and a
          velocity arrow when the door is moving.
        * a single-frame red "EVENT: ... - Track N" banner top-left on the
          exact frame a door-level event fired.

    Top cameras (RIGHT_UP_TOP / LEFT_UP_TOP) reproduce the legacy DAMAGE
    annotation (old_system/RIGHT_UP_TOP/damage_processor.py `_annotate_frame`):
        * per raw per-frame detection: a class-coloured 2px box with a filled
          label bar and white "{class}: {conf}" text.
        * a green top-left info block: "Frame: N", "Damages: K", "Type: CLASS".

There are NO v4-only HUD additions here: no info side panel, no magenta
wagon-boundary flash, no bottom anomaly/gap banners, no OCR overlay -- none of
those existed in old_system's processed videos.

This module NEVER invokes any detector / YOLO / OCR model.  Every box is
replayed from artifacts Stage 3 already persisted:

    * evidence/<gw>/door/overlay.json    per-frame Kalman-smoothed door track
                                         trajectories + door-level events
    * evidence/<gw>/damage/overlay.json  raw per-frame damage detections
    * per_camera_tracking.json           per-camera fps / total_frames / dims

The four cameras render in parallel threads (OpenCV decode releases the GIL).

Output layout (one mp4 per camera):

    <output_dir>/<CAMERA_ID>_processed.mp4
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import cv2

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState


# -----------------------------------------------------------------------------
# Legacy colour maps (BGR) -- verbatim from old_system
# -----------------------------------------------------------------------------

# old_system/RIGHT_UP/door_processor.py STATE_COLORS_BGR
_DOOR_STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "open_door":        (0, 0, 255),     # Red
    "open":             (0, 0, 255),     # Red
    "closed_door":      (0, 255, 0),     # Green
    "closed":           (0, 255, 0),     # Green
    "closed_with_wire": (0, 255, 255),   # Yellow
    "partial_closed":   (0, 255, 255),   # Yellow
    "partially_closed": (0, 255, 255),   # Yellow
    "damage":           (0, 0, 255),     # Red
    "other":            (255, 165, 0),   # Orange
    "unknown":          (128, 128, 128), # Gray
}

# old_system/RIGHT_UP_TOP/damage_processor.py DAMAGE_COLORS
_DAMAGE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "floor_damage":      (0, 0, 255),     # Red
    "inner_wall_damage": (0, 165, 255),   # Orange
    "outer_wall_damage": (0, 255, 255),   # Yellow
    "no_damage":         (0, 255, 0),     # Green
    "unknown":           (128, 128, 128), # Gray
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


# -----------------------------------------------------------------------------
# Per-camera overlay registry (replays persisted trajectories)
# -----------------------------------------------------------------------------

class _OverlayRegistry:
    """Per-frame draw items + events for ONE camera.

    Builds two O(1) lookups:
        boxes_by_frame[frame_idx]  -> list of door/damage draw items
        events_by_frame[frame_idx] -> list of door events (side cameras)
    """

    def __init__(
        self, *, camera_id: str, evidence_root: str, wagons: List[GlobalWagon],
        enabled_features: Optional[set] = None,
    ) -> None:
        self.camera_id = camera_id
        self.boxes_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        self.events_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        # Explicit gate: a feature NOT in enabled_features is never ingested, so
        # a DISABLED feature can never render -- even if a stale overlay.json
        # from a previous run still sits on disk.  When None (legacy default)
        # all features are eligible and the "no overlay.json -> no boxes"
        # fallback applies.
        self._door_enabled = enabled_features is None or "door" in enabled_features
        self._damage_enabled = enabled_features is None or "damage" in enabled_features
        if not evidence_root or not os.path.isdir(evidence_root):
            return
        for gw in wagons:
            ev_gw_dir = os.path.join(evidence_root, gw.global_id)
            if not os.path.isdir(ev_gw_dir):
                continue
            if self._door_enabled:
                self._ingest_door(ev_gw_dir)
            if self._damage_enabled:
                self._ingest_damage(ev_gw_dir)

    # --- internal -------------------------------------------------------

    @staticmethod
    def _load_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _push_box(self, frame_idx: Any, item: Dict[str, Any]) -> None:
        try:
            fi = int(frame_idx)
        except (TypeError, ValueError):
            return
        if fi < 0:
            return
        self.boxes_by_frame.setdefault(fi, []).append(item)

    def _load_overlay(self, ev_gw_dir: str, feature: str) -> Optional[Dict[str, Any]]:
        """Prefer the per-camera overlay (evidence/<gw>/<feature>/<CAMERA>/
        overlay.json); fall back to the legacy flat path for old batches."""
        per_cam = os.path.join(ev_gw_dir, feature, self.camera_id, "overlay.json")
        data = self._load_json(per_cam)
        if data is not None:
            return data
        return self._load_json(os.path.join(ev_gw_dir, feature, "overlay.json"))

    def _ingest_door(self, ev_gw_dir: str) -> None:
        data = self._load_overlay(ev_gw_dir, "door")
        if not data:
            return
        for tr in data.get("tracks") or []:
            if not isinstance(tr, dict) or tr.get("camera_id") != self.camera_id:
                continue
            tid = tr.get("track_id")
            for fr in tr.get("frames") or []:
                bbox = fr.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                self._push_box(fr.get("frame_idx"), {
                    "kind":       "door",
                    "bbox":       list(bbox),
                    "track_id":   tid,
                    "state_raw":  str(fr.get("state_raw") or ""),
                    "last_class": str(fr.get("last_class") or ""),
                    "confidence": float(fr.get("confidence") or 0.0),
                    "velocity":   fr.get("velocity") or [0.0, 0.0],
                })
        for ev in data.get("events") or []:
            if not isinstance(ev, dict) or ev.get("camera_id") != self.camera_id:
                continue
            try:
                fi = int(ev.get("frame_idx", -1))
            except (TypeError, ValueError):
                continue
            if fi < 0:
                continue
            self.events_by_frame.setdefault(fi, []).append({
                "event":    str(ev.get("event", "")),
                "track_id": ev.get("track_id"),
            })

    def _ingest_damage(self, ev_gw_dir: str) -> None:
        data = self._load_overlay(ev_gw_dir, "damage")
        if not data:
            return
        for det in data.get("detections") or []:
            if not isinstance(det, dict) or det.get("camera_id") != self.camera_id:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            self._push_box(det.get("frame_idx"), {
                "kind":       "damage",
                "bbox":       list(bbox),
                "class_name": str(det.get("class_name") or ""),
                "confidence": float(det.get("confidence") or 0.0),
            })


# -----------------------------------------------------------------------------
# Legacy draw primitives (cloned pixel-for-pixel from old_system)
# -----------------------------------------------------------------------------

def _draw_door_track(frame, item: Dict[str, Any]) -> None:
    """Clone of old_system door_processor `_annotate_frame` per-track block."""
    try:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
    except (TypeError, ValueError, KeyError):
        return

    state_raw = str(item.get("state_raw") or "")
    last_class = str(item.get("last_class") or "")
    state_name = state_raw.lower()

    color = _DOOR_STATE_COLORS.get(state_name, (255, 255, 255))
    if state_raw == "UNKNOWN":
        color = _DOOR_STATE_COLORS.get(last_class.lower(), (128, 128, 128))

    thickness = 3 if state_raw == "OPEN" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if state_raw == "UNKNOWN":
        display_state = last_class.upper() if last_class else state_raw
    else:
        display_state = state_raw
    label = f"Door {item.get('track_id')}: {display_state}"
    conf_label = f"{float(item.get('confidence') or 0.0):.2f}"

    font_scale = 0.6
    text_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, _FONT, font_scale, text_thickness)
    label_y = max(y1 - 5, text_h + 5)
    cv2.rectangle(frame, (x1, label_y - text_h - 5),
                  (x1 + text_w + 5, label_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y - 2), _FONT, font_scale,
                (0, 0, 0), text_thickness)

    cv2.putText(frame, conf_label, (x1, y2 + 20), _FONT, 0.5, color, 1)

    vel = item.get("velocity") or [0.0, 0.0]
    try:
        vx, vy = float(vel[0]), float(vel[1])
    except (TypeError, ValueError, IndexError):
        vx, vy = 0.0, 0.0
    if abs(vx) > 1 or abs(vy) > 1:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        end_x, end_y = int(cx + vx * 3), int(cy + vy * 3)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2)


def _draw_damage_det(frame, item: Dict[str, Any]) -> None:
    """Clone of old_system damage_processor `_annotate_frame` per-detection block."""
    try:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
    except (TypeError, ValueError, KeyError):
        return
    class_name = str(item.get("class_name") or "")
    conf = float(item.get("confidence") or 0.0)
    color = _DAMAGE_COLORS.get(class_name, _DAMAGE_COLORS["unknown"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name}: {conf:.2f}"
    (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), _FONT, 0.6, (255, 255, 255), 2)


def _draw_event_banner(frame, events: List[Dict[str, Any]]) -> None:
    """Clone of the legacy single-frame red event banner (top-left)."""
    y_offset = 30
    for ev in events:
        text = f"EVENT: {ev.get('event','')} - Track {ev.get('track_id')}"
        cv2.putText(frame, text, (10, y_offset), _FONT, 0.8, (0, 0, 255), 2)
        y_offset += 30


def _draw_damage_info(frame, frame_idx: int, n_damages: int, frame_class: str) -> None:
    """Clone of the legacy green top-left damage info block."""
    info_lines = [
        f"Frame: {frame_idx}",
        f"Damages: {n_damages}",
        f"Type: {frame_class}",
    ]
    for i, line in enumerate(info_lines):
        y = 30 + i * 25
        cv2.putText(frame, line, (10, y), _FONT, 0.7, (0, 255, 0), 2)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _map_wagon_to_local_frames(
    wagon: GlobalWagon, local_fps: float, local_total_frames: int,
) -> Tuple[int, int]:
    """Mirror of wagon_count/video_segmenter.map_global_wagon_to_local_frames."""
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round(wagon.start_time * local_fps))
    ef = int(round(wagon.end_time * local_fps)) - 1
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


def _load_camera_tracking(per_camera_tracking_path: str) -> Dict[str, Any]:
    if not per_camera_tracking_path or not os.path.isfile(per_camera_tracking_path):
        return {}
    try:
        with open(per_camera_tracking_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Single-camera render
# -----------------------------------------------------------------------------

def _render_one_camera(
    *,
    camera_id: str,
    video_path: str,
    output_path: str,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],   # kept for API symmetry; not drawn
    evidence_root: str,
    camera_meta: Dict[str, Any],
    enabled_features: Optional[set] = None,
    verbose: bool = True,
) -> str:
    del unified  # legacy videos carry no fused-state HUD

    if not os.path.isfile(video_path):
        raise RuntimeError(f"raw video missing for {camera_id}: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for {camera_id}: {video_path}")

    src_fps = float(camera_meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total   = int(camera_meta.get("total_frames")
                  or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width   = int(camera_meta.get("width")
                  or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height  = int(camera_meta.get("height")
                  or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc,
                             src_fps if src_fps > 0 else 25.0, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer for {output_path}")

    is_side = camera_id in C.SIDE_CAMERAS
    is_top  = camera_id in C.TOP_CAMERAS

    # The legacy top-camera info block prints the wagon TYPE; map frame -> wagon
    # class for that line only (side cameras don't draw it).
    frame_to_wagon: Dict[int, GlobalWagon] = {}
    if is_top:
        for w in state.wagons:
            sf, ef = _map_wagon_to_local_frames(w, src_fps, total)
            for f in range(sf, ef + 1):
                frame_to_wagon[f] = w

    overlay = _OverlayRegistry(
        camera_id=camera_id, evidence_root=evidence_root, wagons=state.wagons,
        enabled_features=enabled_features,
    )

    if verbose:
        n_boxes = sum(len(v) for v in overlay.boxes_by_frame.values())
        print(f"[RENDER/{camera_id}] writing -> {output_path}  "
              f"({total} frames, {n_boxes} box-instances)")

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        n_damages = 0
        for item in overlay.boxes_by_frame.get(frame_idx, []):
            if item.get("kind") == "door":
                _draw_door_track(frame, item)
            else:
                _draw_damage_det(frame, item)
                n_damages += 1

        if is_side:
            evs = overlay.events_by_frame.get(frame_idx)
            if evs:
                _draw_event_banner(frame, evs)
        elif is_top:
            w = frame_to_wagon.get(frame_idx)
            frame_class = str(w.classification).upper() if w else "WAGON"
            _draw_damage_info(frame, frame_idx, n_damages, frame_class)

        writer.write(frame)
        written += 1
        frame_idx += 1
        if verbose and frame_idx % 500 == 0:
            print(f"  [{camera_id}] {frame_idx} frames")

    cap.release()
    writer.release()
    if verbose:
        print(f"[RENDER/{camera_id}] done ({written} frames)")
    return output_path


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def render_all_cameras(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: str,
    video_paths: Dict[str, str],
    per_camera_tracking_path: str,
    output_dir: str,
    enabled_features: Optional[set] = None,
    cameras: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Render camera overlay videos in parallel (visualization only -- never
    loads any model).

    ``cameras`` restricts rendering to a subset so a late camera regenerates
    ONLY its own <CAMERA>_processed.mp4; existing overlays for other cameras
    are left untouched.  Returns ``{camera_id -> output_mp4_path}`` for every
    camera that rendered successfully.
    """
    os.makedirs(output_dir, exist_ok=True)
    tracking = _load_camera_tracking(per_camera_tracking_path)

    target = set(cameras) if cameras is not None else set(C.ALL_CAMERAS)
    jobs: Dict[str, Dict[str, Any]] = {}
    for cam in C.ALL_CAMERAS:
        if cam not in target:
            continue
        vp = video_paths.get(cam)
        if not vp:
            if verbose:
                print(f"[RENDER/{cam}] SKIP -- no raw video path")
            continue
        out_mp4 = os.path.join(output_dir, f"{cam}_processed.mp4")
        jobs[cam] = {
            "video_path":  vp,
            "output_path": out_mp4,
            "camera_meta": tracking.get(cam, {}) or {},
        }

    if not jobs:
        return {}

    t0 = time.time()
    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        futs = {
            ex.submit(
                _render_one_camera,
                camera_id=cam,
                video_path=cfg["video_path"],
                output_path=cfg["output_path"],
                state=state,
                unified=unified,
                evidence_root=evidence_root,
                camera_meta=cfg["camera_meta"],
                enabled_features=enabled_features,
                verbose=verbose,
            ): cam
            for cam, cfg in jobs.items()
        }
        for f in as_completed(futs):
            cam = futs[f]
            try:
                results[cam] = f.result()
            except Exception as e:
                print(f"[RENDER/{cam}] FAILED: {type(e).__name__}: {e}")
                if verbose:
                    traceback.print_exc(limit=3)

    if verbose:
        print(f"[RENDER] done {len(results)}/{len(jobs)} cameras  "
              f"({time.time() - t0:.1f}s)")
    return results
