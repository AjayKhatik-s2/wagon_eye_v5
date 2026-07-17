"""Load `global_train_state.json` into a lightweight in-memory record.

This deliberately does NOT import from `wagon_count.global_train_state`:
we want `wagon_eye_v4/` to remain importable even when the wagon_count
subpackage is missing (e.g. for inspecting an already-computed state).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GlobalWagon:
    global_id: str
    wagon_index: int
    start_frame_master: int
    end_frame_master: int
    start_time: float
    end_time: float
    classification: str
    classification_confidence: float = 0.0
    supporting_cameras: List[str] = field(default_factory=list)
    split_from_global_id: Optional[str] = None
    leading_gap: Optional[Dict[str, Any]] = None
    trailing_gap: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class GlobalTrainState:
    total_wagons: int
    wagons: List[GlobalWagon] = field(default_factory=list)
    master_camera: str = "RIGHT_UP"
    master_fps: float = 0.0
    master_total_frames: int = 0

    per_camera_local_counts: Dict[str, int] = field(default_factory=dict)
    per_camera_gap_counts:   Dict[str, int] = field(default_factory=dict)
    per_camera_status:       Dict[str, str] = field(default_factory=dict)

    corrections_applied: List[Dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""

    # Master-first incremental reconstruction provenance (mirrors
    # wagon_count.global_train_state.GlobalTrainState; the JSON schema is the
    # contract between the two definitions).
    participating_cameras: List[str] = field(default_factory=list)
    missing_at_reconstruction: List[str] = field(default_factory=list)
    reconstruction_mode: str = ""
    support_cameras_present: List[str] = field(default_factory=list)
    support_fusion_used: bool = False
    support_gap_recoveries: int = 0
    reconstruction_confidence: float = 1.0
    fallback_master_used: bool = False
    sealed_at: str = ""
    sealing_reason: str = ""

    notes: List[str] = field(default_factory=list)

    @property
    def regular_wagon_count(self) -> int:
        return sum(1 for w in self.wagons if w.classification == "WAGON")

    @property
    def engine_count(self) -> int:
        return sum(1 for w in self.wagons if w.classification == "ENGINE")

    @property
    def brake_van_count(self) -> int:
        return sum(1 for w in self.wagons if w.classification == "BRAKE_VAN")


def load_global_train_state(path: str) -> GlobalTrainState:
    """Parse `global_train_state.json` (as emitted by wagon_count)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"global_train_state.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    wagons: List[GlobalWagon] = []
    for w in doc.get("wagons", []):
        wagons.append(GlobalWagon(
            global_id=w["global_id"],
            wagon_index=int(w.get("wagon_index", 0)),
            start_frame_master=int(w.get("start_frame_master", 0)),
            end_frame_master=int(w.get("end_frame_master", 0)),
            start_time=float(w.get("start_time", 0.0)),
            end_time=float(w.get("end_time", 0.0)),
            classification=w.get("classification", "UNKNOWN"),
            classification_confidence=float(w.get("classification_confidence", 0.0)),
            supporting_cameras=list(w.get("supporting_cameras") or []),
            split_from_global_id=w.get("split_from_global_id"),
            leading_gap=w.get("leading_gap"),
            trailing_gap=w.get("trailing_gap"),
        ))

    return GlobalTrainState(
        total_wagons=int(doc.get("total_wagons", 0)),
        wagons=wagons,
        master_camera=doc.get("master_camera", "RIGHT_UP"),
        master_fps=float(doc.get("master_fps", 0.0)),
        master_total_frames=int(doc.get("master_total_frames", 0)),
        per_camera_local_counts=dict(doc.get("per_camera_local_counts") or {}),
        per_camera_gap_counts=dict(doc.get("per_camera_gap_counts") or {}),
        per_camera_status=dict(doc.get("per_camera_status") or {}),
        corrections_applied=list(doc.get("corrections_applied") or []),
        fallback_used=bool(doc.get("fallback_used", False)),
        fallback_reason=doc.get("fallback_reason", "") or "",
        participating_cameras=list(doc.get("participating_cameras") or []),
        missing_at_reconstruction=list(doc.get("missing_at_reconstruction") or []),
        reconstruction_mode=doc.get("reconstruction_mode", "") or "",
        support_cameras_present=list(doc.get("support_cameras_present") or []),
        support_fusion_used=bool(doc.get("support_fusion_used", False)),
        support_gap_recoveries=int(doc.get("support_gap_recoveries", 0) or 0),
        reconstruction_confidence=float(doc.get("reconstruction_confidence", 1.0) or 1.0),
        fallback_master_used=bool(doc.get("fallback_master_used", False)),
        sealed_at=doc.get("sealed_at", "") or "",
        sealing_reason=doc.get("sealing_reason", "") or "",
        notes=list(doc.get("notes") or []),
    )


def load_per_camera_fps(per_camera_tracking_json: str) -> Dict[str, float]:
    """Read each camera's source fps from per_camera_tracking.json."""
    out: Dict[str, float] = {}
    try:
        with open(per_camera_tracking_json, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    for cam, meta in doc.items():
        if isinstance(meta, dict):
            try:
                out[cam] = float(meta.get("fps") or 0.0) or 0.0
            except (TypeError, ValueError):
                pass
    return out
