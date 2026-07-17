"""
global_train_state.py
=====================

Phase-1 data contracts for the Wagon Eye global synchronization engine.

This module defines the *only* canonical entities used by the Phase-1
pipeline:

    GapEvent           - one tracked inter-wagon gap, in a single camera
    LocalCameraTracks  - everything one camera produced (fps, frames, gaps,
                         optional per-segment classifications for RIGHT_UP)
    SegmentClass       - classification label for a master segment
    GlobalWagon        - one physical wagon after cross-camera fusion
    GlobalTrainState   - the final globally synchronized train understanding

These types are deliberately decoupled from the legacy v3 TrainSession /
CameraEvidence types in RIGHT_UP/train_session.py.  Phase 1 does NOT
attach doors / OCR / damages -- those are explicitly out of scope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any


# Canonical camera ids used across the system. RIGHT_UP is the master.
CAMERA_RIGHT_UP = "RIGHT_UP"
CAMERA_LEFT_UP = "LEFT_UP"
CAMERA_RIGHT_UP_TOP = "RIGHT_UP_TOP"
CAMERA_LEFT_UP_TOP = "LEFT_UP_TOP"

MASTER_CAMERA = CAMERA_RIGHT_UP
SIDE_CAMERAS = (CAMERA_RIGHT_UP, CAMERA_LEFT_UP)
TOP_CAMERAS = (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)
ALL_CAMERAS = (CAMERA_RIGHT_UP, CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)


# Segment classification labels. RIGHT_UP is the only authority.
class SegmentClass:
    ENGINE = "ENGINE"
    WAGON = "WAGON"
    BRAKE_VAN = "BRAKE_VAN"
    UNKNOWN = "UNKNOWN"

    ALL = (ENGINE, WAGON, BRAKE_VAN, UNKNOWN)


# =============================================================================
# GAP EVENT
# =============================================================================

@dataclass
class GapEvent:
    """One tracked inter-wagon gap in a single camera.

    A GapEvent represents a temporally persistent gap track: it survived
    through `hit_count` frames, was active from `start_frame` to `end_frame`,
    and its image-plane horizontal center traversed `center_x_trajectory`.

    The `temporal_consistency_score` is a 0..1 value capturing how stable
    the track was -- (hit_count / span_frames) with span_frames clamped.
    """

    track_id: int
    camera_id: str
    start_frame: int
    end_frame: int
    confidence: float                  # mean detection confidence over the track
    hit_count: int                     # number of frames the gap was detected
    center_x_trajectory: List[float] = field(default_factory=list)
    fps: float = 0.0
    temporal_consistency_score: float = 0.0

    # Per-hit data, parallel arrays to center_x_trajectory.  hit_frames[i] is
    # the video frame index at which the i-th hit was observed; bbox_history[i]
    # is the YOLO bounding box [x1, y1, x2, y2] at that hit.  Used by the
    # overlay renderer to draw a continuous bbox for the full duration of
    # the gap, interpolating across miss frames between hits.
    hit_frames: List[int] = field(default_factory=list)
    bbox_history: List[List[float]] = field(default_factory=list)

    # Optional model class label, for debug (typically just 'gap')
    class_label: str = "gap"

    @property
    def span_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame + 1)

    @property
    def center_frame(self) -> float:
        return (self.start_frame + self.end_frame) / 2.0

    @property
    def center_time(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.center_frame / self.fps

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps if self.fps > 0 else 0.0

    @property
    def end_time(self) -> float:
        return self.end_frame / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "center_time": round(self.center_time, 4),
            "confidence": round(self.confidence, 4),
            "hit_count": self.hit_count,
            "temporal_consistency_score": round(self.temporal_consistency_score, 4),
            "class_label": self.class_label,
        }


# =============================================================================
# LOCAL CAMERA TRACKS
# =============================================================================

@dataclass
class _MasterClassification:
    """Initial per-master-segment classification produced by RIGHT_UP.

    A master segment is the span between two consecutive RIGHT_UP gaps
    (or video start/end). Indexed by zero-based segment ordinal in RIGHT_UP's
    pre-fusion timeline.
    """
    segment_index: int
    start_frame: int
    end_frame: int
    label: str                      # SegmentClass.*
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "label": self.label,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class LocalCameraTracks:
    """Everything one camera produced during Phase-1 tracking.

    The master camera (RIGHT_UP) also fills `classifications` with the
    initial pre-fusion segment labels.  Other cameras leave it empty.
    """
    camera_id: str
    video_path: str
    fps: float
    total_frames: int
    width: int = 0
    height: int = 0
    gaps: List[GapEvent] = field(default_factory=list)
    classifications: List[_MasterClassification] = field(default_factory=list)
    # Raw per-frame gap detections (frame_idx -> list of bbox dicts) kept
    # ONLY for the overlay renderer to draw raw detection boxes.  Not
    # serialized into JSON output.
    raw_frame_detections: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def local_wagon_count(self) -> int:
        """Number of locally-detected wagon segments = gaps + 1."""
        # An empty video produces 0 wagons, not 1.
        if self.total_frames <= 0:
            return 0
        return len(self.gaps) + 1

    def to_dict(self, include_classifications: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "video_path": self.video_path,
            "fps": round(self.fps, 4),
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "gap_count": len(self.gaps),
            "local_wagon_count": self.local_wagon_count,
            "gaps": [g.to_dict() for g in self.gaps],
        }
        if include_classifications and self.classifications:
            out["classifications"] = [c.to_dict() for c in self.classifications]
        return out


# =============================================================================
# GLOBAL WAGON
# =============================================================================

@dataclass
class GlobalWagon:
    """One physical wagon, after cross-camera fusion.

    Time coordinates are in MASTER (RIGHT_UP) seconds.  Frame coordinates
    are in MASTER frames.  Per-camera local frame ranges are derivable from
    `time_window` * each camera's fps (Phase-1 assumes the four trimmed
    videos share a t=0 alignment -- see README at the head of run_global_count.py).
    """
    global_id: str                          # 'GW_1', 'GW_2', ...
    wagon_index: int                        # 1-based positional index
    start_frame_master: int
    end_frame_master: int
    start_time: float                       # seconds (master clock)
    end_time: float
    classification: str = SegmentClass.UNKNOWN
    classification_confidence: float = 0.0
    supporting_cameras: List[str] = field(default_factory=list)
    # Provenance: was this wagon created by inserting a recovered gap?
    split_from_global_id: Optional[str] = None
    # The two boundary gap track ids in the master (None at video edges)
    leading_gap: Optional[Dict[str, Any]] = None   # serialized GapEvent or {'source':'edge'}
    trailing_gap: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id": self.global_id,
            "wagon_index": self.wagon_index,
            "start_frame_master": self.start_frame_master,
            "end_frame_master": self.end_frame_master,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4),
            "classification": self.classification,
            "classification_confidence": round(self.classification_confidence, 4),
            "supporting_cameras": list(self.supporting_cameras),
            "split_from_global_id": self.split_from_global_id,
            "leading_gap": self.leading_gap,
            "trailing_gap": self.trailing_gap,
        }


# =============================================================================
# CORRECTION RECORD
# =============================================================================

@dataclass
class GapCorrection:
    """Audit record for one fused-in gap that RIGHT_UP missed."""
    inserted_at_master_time: float
    inserted_at_master_frame: int
    supporting_cameras: List[str]
    mean_confidence: float
    time_spread_sec: float            # max - min of contributing center_times
    contributing_track_ids: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inserted_at_master_time": round(self.inserted_at_master_time, 4),
            "inserted_at_master_frame": self.inserted_at_master_frame,
            "supporting_cameras": list(self.supporting_cameras),
            "mean_confidence": round(self.mean_confidence, 4),
            "time_spread_sec": round(self.time_spread_sec, 4),
            "contributing_track_ids": dict(self.contributing_track_ids),
        }


# =============================================================================
# GLOBAL TRAIN STATE
# =============================================================================

@dataclass
class GlobalTrainState:
    """The final globally synchronized train state.

    This is the ONLY required Phase-1 output.  Serialized as JSON via
    `to_dict()` / `to_json()`.
    """
    total_wagons: int
    wagons: List[GlobalWagon] = field(default_factory=list)

    # Master-clock info
    master_camera: str = MASTER_CAMERA
    master_fps: float = 0.0
    master_total_frames: int = 0

    # Per-camera bookkeeping
    per_camera_local_counts: Dict[str, int] = field(default_factory=dict)
    per_camera_gap_counts: Dict[str, int] = field(default_factory=dict)
    per_camera_status: Dict[str, str] = field(default_factory=dict)

    # Fusion audit
    corrections_applied: List[GapCorrection] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""

    # ---- Master-first incremental reconstruction provenance (v4 lifecycle) ----
    # Which cameras actually participated when this state was SEALED, and which
    # were missing at that moment.  A late camera enriches features later but
    # NEVER changes any of these.
    participating_cameras: List[str] = field(default_factory=list)
    missing_at_reconstruction: List[str] = field(default_factory=list)
    # MASTER_ONLY | MASTER_WITH_SUPPORT_AVAILABLE | MASTER_WITH_FUSED_SUPPORT
    reconstruction_mode: str = ""
    # Support present is NOT the same as support used: a gap is only recovered
    # when >=2 support cameras agree (insert_min_support), recorded as a
    # GapCorrection.  support_fusion_used == (support_gap_recoveries > 0).
    support_cameras_present: List[str] = field(default_factory=list)
    support_fusion_used: bool = False
    support_gap_recoveries: int = 0
    reconstruction_confidence: float = 1.0
    fallback_master_used: bool = False
    sealed_at: str = ""
    sealing_reason: str = ""

    # Free-form notes from the runner
    notes: List[str] = field(default_factory=list)

    def add_note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def regular_wagon_count(self) -> int:
        """Wagons whose classification is WAGON (excludes ENGINE / BRAKE_VAN)."""
        return sum(1 for w in self.wagons if w.classification == SegmentClass.WAGON)

    @property
    def engine_count(self) -> int:
        return sum(1 for w in self.wagons if w.classification == SegmentClass.ENGINE)

    @property
    def brake_van_count(self) -> int:
        return sum(1 for w in self.wagons if w.classification == SegmentClass.BRAKE_VAN)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "wagon_eye.global_train_state.v1",
            "master_camera": self.master_camera,
            "master_fps": round(self.master_fps, 4),
            "master_total_frames": self.master_total_frames,
            "total_wagons": self.total_wagons,
            "regular_wagon_count": self.regular_wagon_count,
            "engine_count": self.engine_count,
            "brake_van_count": self.brake_van_count,
            "wagons": [w.to_dict() for w in self.wagons],
            "per_camera_local_counts": dict(self.per_camera_local_counts),
            "per_camera_gap_counts": dict(self.per_camera_gap_counts),
            "per_camera_status": dict(self.per_camera_status),
            "corrections_applied": [c.to_dict() for c in self.corrections_applied],
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "participating_cameras": list(self.participating_cameras),
            "missing_at_reconstruction": list(self.missing_at_reconstruction),
            "reconstruction_mode": self.reconstruction_mode,
            "support_cameras_present": list(self.support_cameras_present),
            "support_fusion_used": self.support_fusion_used,
            "support_gap_recoveries": self.support_gap_recoveries,
            "reconstruction_confidence": round(self.reconstruction_confidence, 4),
            "fallback_master_used": self.fallback_master_used,
            "sealed_at": self.sealed_at,
            "sealing_reason": self.sealing_reason,
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# =============================================================================
# UTILITY -- compact text summary for stdout
# =============================================================================

def summarize_state(state: GlobalTrainState) -> str:
    """Return a multi-line text summary suitable for logging at end of run."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("  GLOBAL TRAIN STATE SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Master camera        : {state.master_camera}")
    lines.append(f"  Master fps           : {state.master_fps:.3f}")
    lines.append(f"  Master total frames  : {state.master_total_frames}")
    lines.append("")
    lines.append("  Per-camera local counts (wagons / gaps / status):")
    for cam in ALL_CAMERAS:
        wc = state.per_camera_local_counts.get(cam, "—")
        gc = state.per_camera_gap_counts.get(cam, "—")
        st = state.per_camera_status.get(cam, "—")
        lines.append(f"    {cam:<14}  wagons={wc}   gaps={gc}   [{st}]")
    lines.append("")
    lines.append(f"  Corrections applied  : {len(state.corrections_applied)}")
    for c in state.corrections_applied:
        lines.append(
            f"    + insert @ t={c.inserted_at_master_time:.2f}s  "
            f"frame={c.inserted_at_master_frame}  "
            f"supports={'/'.join(c.supporting_cameras)}  "
            f"conf={c.mean_confidence:.2f}  spread={c.time_spread_sec:.2f}s"
        )
    lines.append("")
    lines.append(f"  FINAL FUSED WAGON COUNT : {state.total_wagons}")
    lines.append(f"     regular wagons       : {state.regular_wagon_count}")
    lines.append(f"     engines              : {state.engine_count}")
    lines.append(f"     brake vans           : {state.brake_van_count}")
    if state.fallback_used:
        lines.append(f"  ⚠ FALLBACK USED       : {state.fallback_reason}")
    lines.append("=" * 70)
    return "\n".join(lines)
