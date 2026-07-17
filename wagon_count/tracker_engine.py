"""
tracker_engine.py
=================

Phase-1 per-camera tracking engine.

Responsibilities (per camera):
  1. Open the input video.
  2. Run the appropriate gap-detection YOLO model on every frame.
  3. Track gap detections temporally with a constant-velocity Kalman
     filter on the bounding-box horizontal center, plus a hit/miss
     persistence rule.
  4. Emit one GapEvent per stable track.
  5. (RIGHT_UP only) Classify each pre-fusion master segment using
     side_classification.pt by majority voting across sampled frames.

This module deliberately reuses YOLO via `ultralytics` -- the same
pattern as RIGHT_UP/gap_validation.py and RIGHT_UP/wagon_classifier.py.

Determinism notes:
  * YOLO inference is deterministic given the same weights, device, and
    pre-processing.  We never use stochastic augmentation.
  * Track id assignment is monotonic in detection arrival order, which
    is itself deterministic (frame index then ymin then xmin).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

# OpenCV used only for reading frames + measuring video metadata.
# All model inference goes through ultralytics.YOLO.
import cv2

from global_train_state import (
    GapEvent,
    LocalCameraTracks,
    SegmentClass,
    _MasterClassification,
    CAMERA_RIGHT_UP,
    SIDE_CAMERAS,
    TOP_CAMERAS,
)


# =============================================================================
# DEVICE RESOLUTION
# =============================================================================
#
# wagon_count/ is a self-contained package (it must zip and run on its own,
# see its README), so it does NOT import core.config.  This mirrors the same
# logic locally: honour WAGONEYE_DEVICE, else auto-detect CUDA, else CPU.
# The Stage-1 subprocess inherits WAGONEYE_DEVICE from the orchestrator's
# environment, so the whole pipeline agrees on the device without any extra
# flag plumbing.  Passing the resolved device explicitly into inference keeps
# GPU behaviour identical (cuda == ultralytics' own default there) while
# giving a clean CPU fallback on a CPU-only host.

def _resolve_device(force: Optional[str] = None) -> str:
    choice = (force or os.environ.get("WAGONEYE_DEVICE") or "auto").strip().lower()
    if choice in ("cuda", "gpu"):
        return "cuda"
    if choice == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# =============================================================================
# CONSTANT-VELOCITY KALMAN FILTER (1-D on bbox center_x)
# =============================================================================
#
# State vector  : [x, vx]
# Measurement   : [x]
# Transition    : x_{t+1} = x_t + vx_t,   vx_{t+1} = vx_t
# We keep this hand-rolled rather than pulling in filterpy so the module
# has no extra runtime dependency beyond numpy/cv2/ultralytics.

class _KF1D:
    __slots__ = ("x", "P", "F", "H", "Q", "R")

    def __init__(self, init_x: float, process_var: float = 4.0, meas_var: float = 9.0):
        self.x = np.array([[init_x], [0.0]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 100.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.eye(2, dtype=np.float64) * process_var
        self.R = np.array([[meas_var]], dtype=np.float64)

    def predict(self) -> float:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0])

    def update(self, z: float) -> None:
        y = np.array([[z]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def cx(self) -> float:
        return float(self.x[0, 0])


# =============================================================================
# INTERNAL TRACK STATE
# =============================================================================

@dataclass
class _Track:
    track_id: int
    first_frame: int
    last_seen_frame: int
    confidences: List[float] = field(default_factory=list)
    centers: List[float] = field(default_factory=list)
    hit_frames: List[int] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    hit_count: int = 0
    miss_count: int = 0
    kf: Optional[_KF1D] = None
    confirmed: bool = False

    def predicted_center(self) -> float:
        if self.kf is None:
            return self.centers[-1] if self.centers else 0.0
        return self.kf.predict()

    def update(self, frame_idx: int, center_x: float, conf: float,
               bbox: Optional[List[float]] = None) -> None:
        if self.kf is None:
            self.kf = _KF1D(center_x)
        else:
            self.kf.update(center_x)
        self.centers.append(self.kf.cx)
        self.confidences.append(conf)
        self.hit_frames.append(frame_idx)
        if bbox is not None:
            self.bboxes.append([float(v) for v in bbox])
        self.hit_count += 1
        self.miss_count = 0
        self.last_seen_frame = frame_idx

    def mark_miss(self) -> None:
        self.miss_count += 1


# =============================================================================
# GAP TRACKER
# =============================================================================

class GapTracker:
    """Per-camera gap detection + tracking + GapEvent emission.

    Parameters
    ----------
    camera_id        : 'RIGHT_UP' | 'LEFT_UP' | 'RIGHT_UP_TOP' | 'LEFT_UP_TOP'
    model_path       : path to the YOLO weights for this camera --
                       right_up_wagon_gap.pt for RIGHT_UP,
                       left_up_wagon_gap.pt  for LEFT_UP,
                       top_gap.pt            for either top camera.
    confidence       : YOLO confidence threshold
    min_height_ratio : min bbox height as a fraction of frame height
                       (rejects floor-strip and tiny detections)
    match_distance_px: max horizontal distance to associate a detection with
                       an existing track
    min_hits         : a track must accumulate at least this many hits to be
                       emitted as a confirmed GapEvent
    max_miss         : a track is closed after this many consecutive misses
    """

    def __init__(
        self,
        camera_id: str,
        model_path: str,
        confidence: float = 0.4,
        min_height_ratio: float = 0.35,
        match_distance_px: float = 80.0,
        min_hits: int = 3,
        max_miss: int = 30,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gap model not found for {camera_id}: {model_path}")
        # Defer ultralytics import so the file can be parsed without it.
        # Mirror the torch.load monkey-patch used by RIGHT_UP/wagon_classifier.py
        # so .pt weights load on torch >= 2.6.
        import torch
        _orig_load = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        torch.load = _patched
        from ultralytics import YOLO

        self.camera_id = camera_id
        self.model_path = model_path
        self.confidence = float(confidence)
        self.min_height_ratio = float(min_height_ratio)
        self.match_distance_px = float(match_distance_px)
        self.min_hits = int(min_hits)
        self.max_miss = int(max_miss)
        # Was previously stored but never used (ultralytics guessed the device).
        # Resolve it now and actually pass it into inference below.
        self.device = device or _resolve_device()
        self.verbose = verbose

        if verbose:
            print(f"[GapTracker/{camera_id}] Loading {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        # Single-class models (very common for gap detectors) get a permissive
        # class filter -- whatever the one class is named, we accept it.
        # Multi-class models still require the class name to contain "gap".
        self._is_single_class_model = (len(self.class_names) == 1)
        if verbose:
            print(f"[GapTracker/{camera_id}] Classes: {self.class_names}  "
                  f"(single_class={self._is_single_class_model})")
            print(f"[GapTracker/{camera_id}] Filters: conf>={self.confidence}  "
                  f"min_height_ratio={self.min_height_ratio}")

        # Per-process diagnostic counters, reset at the start of each
        # process_video() call.  Help diagnose "no bbox shown" cases by
        # revealing whether YOLO never returned boxes vs. the filters
        # ate them.
        self._diag_total_yolo_boxes = 0
        self._diag_after_class = 0
        self._diag_after_conf = 0
        self._diag_kept = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def process_video(
        self,
        video_path: str,
        frame_limit: int = 0,
        keep_raw_detections: bool = True,
    ) -> LocalCameraTracks:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Video reports non-positive fps: {video_path}")

        if self.verbose:
            print(f"\n[GapTracker/{self.camera_id}] {os.path.basename(video_path)}")
            print(f"  fps={fps:.3f}  frames={total_frames_meta}  size={width}x{height}")

        active_tracks: List[_Track] = []
        completed_tracks: List[_Track] = []
        next_track_id = 1
        raw_detections: Dict[int, List[Dict[str, Any]]] = {}

        # Reset diagnostic counters for this video
        self._diag_total_yolo_boxes = 0
        self._diag_after_class = 0
        self._diag_after_conf = 0
        self._diag_kept = 0

        frame_idx = 0
        t0 = time.time()

        while True:
            if frame_limit and frame_idx >= frame_limit:
                break
            ret, frame = cap.read()
            if not ret:
                break

            detections = self._detect_gaps(frame, height)

            if keep_raw_detections and detections:
                # Lightweight payload for the overlay renderer (bbox + conf)
                raw_detections[frame_idx] = [
                    {
                        "bbox": [float(x) for x in d["bbox"]],
                        "confidence": d["confidence"],
                        "center_x": d["center_x"],
                    }
                    for d in detections
                ]

            # Predict step for all active tracks
            for tr in active_tracks:
                tr.predicted_center()

            # Greedy nearest-neighbor association on x distance.
            # Sort detections by center_x so association is deterministic.
            detections_sorted = sorted(detections, key=lambda d: d["center_x"])
            used_track_idx: set = set()

            for det in detections_sorted:
                best_i, best_d = -1, float("inf")
                cx = det["center_x"]
                for i, tr in enumerate(active_tracks):
                    if i in used_track_idx:
                        continue
                    d = abs(cx - tr.kf.cx) if tr.kf is not None else abs(cx - tr.centers[-1])
                    if d < best_d and d <= self.match_distance_px:
                        best_d = d
                        best_i = i
                if best_i >= 0:
                    used_track_idx.add(best_i)
                    active_tracks[best_i].update(frame_idx, cx, det["confidence"], det["bbox"])
                    if active_tracks[best_i].hit_count >= self.min_hits:
                        active_tracks[best_i].confirmed = True
                else:
                    tr = _Track(
                        track_id=next_track_id,
                        first_frame=frame_idx,
                        last_seen_frame=frame_idx,
                    )
                    tr.update(frame_idx, cx, det["confidence"], det["bbox"])
                    active_tracks.append(tr)
                    next_track_id += 1

            # Increment miss for tracks not matched this frame
            for i, tr in enumerate(active_tracks):
                if i not in used_track_idx:
                    tr.mark_miss()

            # Close tracks that exceeded max_miss
            still_active: List[_Track] = []
            for tr in active_tracks:
                if tr.miss_count >= self.max_miss:
                    if tr.confirmed:
                        completed_tracks.append(tr)
                else:
                    still_active.append(tr)
            active_tracks = still_active

            frame_idx += 1
            if self.verbose and frame_idx % 200 == 0:
                print(f"  ... frame {frame_idx}  active_tracks={len(active_tracks)}  "
                      f"completed={len(completed_tracks)}")

        cap.release()
        # Flush surviving confirmed tracks
        for tr in active_tracks:
            if tr.confirmed:
                completed_tracks.append(tr)

        # Sort by first_frame so GapEvents are temporally ordered, then
        # rewrite track_ids 1..N for determinism
        completed_tracks.sort(key=lambda t: (t.first_frame, t.last_seen_frame))

        events: List[GapEvent] = []
        for new_id, tr in enumerate(completed_tracks, start=1):
            mean_conf = float(np.mean(tr.confidences)) if tr.confidences else 0.0
            span = max(1, tr.last_seen_frame - tr.first_frame + 1)
            tcs = float(min(1.0, tr.hit_count / span))
            events.append(GapEvent(
                track_id=new_id,
                camera_id=self.camera_id,
                start_frame=tr.first_frame,
                end_frame=tr.last_seen_frame,
                confidence=mean_conf,
                hit_count=tr.hit_count,
                center_x_trajectory=list(tr.centers),
                fps=fps,
                temporal_consistency_score=tcs,
                hit_frames=list(tr.hit_frames),
                bbox_history=[list(b) for b in tr.bboxes],
            ))

        # Effective frame count = whatever we actually iterated through
        effective_frames = frame_idx
        # Some containers misreport CAP_PROP_FRAME_COUNT; trust what we read.
        total_frames = max(effective_frames, total_frames_meta if total_frames_meta > 0 else 0)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"[GapTracker/{self.camera_id}] done in {elapsed:.1f}s  "
                  f"emitted {len(events)} confirmed gaps  "
                  f"({frame_idx} frames processed)")
            # Filter-stage diagnostics -- helps spot "no bbox shown" cases.
            print(f"  YOLO boxes: total={self._diag_total_yolo_boxes}  "
                  f"after_class={self._diag_after_class}  "
                  f"after_conf={self._diag_after_conf}  "
                  f"kept={self._diag_kept}")
            if self._diag_total_yolo_boxes > 0 and self._diag_kept == 0:
                print(f"  ⚠ All {self._diag_total_yolo_boxes} YOLO boxes were "
                      f"rejected by filters.  Lower --side/top-confidence or "
                      f"--side/top-min-height-ratio for this camera.")

        return LocalCameraTracks(
            camera_id=self.camera_id,
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            gaps=events,
            raw_frame_detections=raw_detections if keep_raw_detections else {},
        )

    # ------------------------------------------------------------------
    # Per-frame YOLO inference
    # ------------------------------------------------------------------
    def _detect_gaps(self, frame: np.ndarray, frame_h: int) -> List[Dict[str, Any]]:
        results = self.model(frame, verbose=False, device=self.device)[0]
        dets: List[Dict[str, Any]] = []
        if results.boxes is None or len(results.boxes) == 0:
            return dets

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, clss):
            self._diag_total_yolo_boxes += 1

            name = self.class_names.get(int(cls_id), "unknown").lower()
            # Permissive class filter:
            #   - single-class models: accept any (model is gap-only by design)
            #   - multi-class models: require "gap" as a substring of the name
            if not self._is_single_class_model and "gap" not in name:
                continue
            self._diag_after_class += 1

            if float(conf) < self.confidence:
                continue
            self._diag_after_conf += 1

            # Height-ratio filter rejects tiny noise detections.  For top
            # cameras the gap is a thin horizontal strip and the caller
            # should configure min_height_ratio to a low value (~0.05).
            h = float(box[3] - box[1])
            if h < frame_h * self.min_height_ratio:
                continue
            self._diag_kept += 1

            cx = float((box[0] + box[2]) / 2.0)
            dets.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "confidence": float(conf),
                "center_x": cx,
                "height": h,
            })
        return dets


# =============================================================================
# MASTER CLASSIFIER (RIGHT_UP only)
# =============================================================================

# Reuse the label mapping from the existing wagon_classifier so behavior
# is consistent with the legacy pipeline.
_ENGINE_LABELS = {"engine", "loco", "engine_head", "locono", "locomotive"}
_BRAKEVAN_LABELS = {"tail", "brake_van", "brakevan", "guard_van", "wagon_tail"}
_TRACK_LABELS = {"track", "background", "empty_track", "rail", "tracks"}

# Minimum mean vote confidence required before a segment may be called ENGINE.
# Below this, an 'engine'/'loco' vote is treated as UNCERTAIN (UNKNOWN), not
# promoted to ENGINE.  This is the single guard that stops a low-confidence
# loco-front / track-strip read at train entry from creating a phantom leading
# engine.  NEVER assume "first detected wagon == ENGINE".
ENGINE_MIN_CONFIDENCE = 0.55


class MasterClassifier:
    """Classify master segments using side_classification.pt.

    A *master segment* is the span between two consecutive RIGHT_UP gaps
    (or between video start and the first gap / between the last gap and
    video end).  For each segment we sample N frames evenly, run the
    classifier, and majority-vote.
    """

    def __init__(
        self,
        model_path: str,
        num_samples: int = 5,
        verbose: bool = True,
        device: Optional[str] = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classification model not found: {model_path}")
        import torch
        _orig_load = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        torch.load = _patched
        from ultralytics import YOLO

        self.model_path = model_path
        self.num_samples = int(num_samples)
        self.verbose = verbose
        self.device = device or _resolve_device()
        if verbose:
            print(f"[MasterClassifier] Loading {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        if verbose:
            print(f"[MasterClassifier] Classes: {self.class_names}")

    def classify_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        results = self.model(frame, verbose=False, device=self.device)[0]
        if getattr(results, "probs", None) is not None:
            top1 = int(results.probs.top1)
            conf = float(results.probs.top1conf)
            return self.class_names.get(top1, "unknown").lower(), conf
        if results.boxes is not None and len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy().astype(int)
            best = int(np.argmax(confs))
            return self.class_names.get(int(cls[best]), "unknown").lower(), float(confs[best])
        return "wagon", 0.0

    def classify_segments(
        self,
        video_path: str,
        segments: List[Tuple[int, int]],
    ) -> List[_MasterClassification]:
        """Classify each (start_frame, end_frame) segment of `video_path`."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for classification: {video_path}")

        out: List[_MasterClassification] = []
        try:
            for idx, (sf, ef) in enumerate(segments):
                label, conf = self._classify_one(cap, sf, ef)
                seg_class = self._label_to_class(label, conf)
                out.append(_MasterClassification(
                    segment_index=idx,
                    start_frame=sf,
                    end_frame=ef,
                    label=seg_class,
                    confidence=conf,
                ))
                if self.verbose:
                    print(f"  [seg {idx}] frames {sf}-{ef} -> {seg_class} "
                          f"(raw='{label}', conf={conf:.2f})")
        finally:
            cap.release()
        return out

    def _classify_one(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
    ) -> Tuple[str, float]:
        span = max(1, end_frame - start_frame + 1)
        margin = max(1, int(span * 0.1))
        safe_s = start_frame + margin
        safe_e = end_frame - margin
        if safe_e <= safe_s:
            sample_idxs = [start_frame + span // 2]
        elif self.num_samples == 1:
            sample_idxs = [safe_s + (safe_e - safe_s) // 2]
        else:
            step = (safe_e - safe_s) / max(self.num_samples - 1, 1)
            sample_idxs = [int(round(safe_s + i * step)) for i in range(self.num_samples)]

        labels: List[str] = []
        confs: List[float] = []
        for fi in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            lbl, c = self.classify_frame(frame)
            labels.append(lbl)
            confs.append(c)

        if not labels:
            return "wagon", 0.0

        # Majority vote, deterministic tiebreak by alphabetical order
        from collections import Counter
        counts = Counter(labels)
        top = max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0][0])))[0]
        kept_confs = [c for l, c in zip(labels, confs) if l == top]
        return top, float(np.mean(kept_confs)) if kept_confs else 0.0

    @staticmethod
    def _label_to_class(label: str, confidence: float = 1.0) -> str:
        lbl = (label or "").lower()
        if lbl in _ENGINE_LABELS:
            # NEVER assume the first wagon == ENGINE.  Promote to ENGINE only
            # when the model is confident; otherwise leave it UNCERTAIN
            # (UNKNOWN) so a phantom leading segment can be dropped rather than
            # emitted as a false engine.
            if confidence >= ENGINE_MIN_CONFIDENCE:
                return SegmentClass.ENGINE
            return SegmentClass.UNKNOWN
        if lbl in _BRAKEVAN_LABELS:
            return SegmentClass.BRAKE_VAN
        if lbl in _TRACK_LABELS:
            return SegmentClass.UNKNOWN
        return SegmentClass.WAGON


# =============================================================================
# CONVENIENCE: split a camera's frame range into segments using its gaps
# =============================================================================

def segments_from_gaps(
    gaps: List[GapEvent],
    total_frames: int,
) -> List[Tuple[int, int]]:
    """Convert a list of GapEvents into (start_frame, end_frame) segments.

    Gaps must come from one camera.  The function uses each gap's
    midpoint frame as the inter-wagon boundary, in temporal order.

    Returns a list of [start, end] inclusive segment ranges covering
    [0, total_frames - 1].
    """
    if total_frames <= 0:
        return []
    if not gaps:
        return [(0, total_frames - 1)]

    boundaries: List[int] = []
    for g in sorted(gaps, key=lambda x: x.center_frame):
        b = int(round(g.center_frame))
        b = max(0, min(total_frames - 1, b))
        boundaries.append(b)

    segments: List[Tuple[int, int]] = []
    prev = 0
    for b in boundaries:
        if b <= prev:
            continue
        segments.append((prev, b - 1))
        prev = b
    if prev <= total_frames - 1:
        segments.append((prev, total_frames - 1))
    return segments
