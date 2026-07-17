"""UnifiedWagonState -- one physical wagon, fully fused across cameras.

This is the canonical record consumed by reporting/.  It carries:
    - identity         (global_id, classification, OCR)
    - per-side door state
    - load status
    - damage status
    - provenance (which cameras contributed) + an overall confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from . import constants as C


@dataclass
class UnifiedWagonState:
    global_id: str
    wagon_index: int

    # Stage-0 authoritative
    classification: str = C.CLASS_UNKNOWN
    classification_confidence: float = 0.0

    # Identity
    wagon_identifier: str = C.NO_DATA
    wagon_identifier_confidence: float = 0.0

    # Doors (side cameras)
    left_door: str = C.NO_DATA
    left_door_confidence: float = 0.0
    right_door: str = C.NO_DATA
    right_door_confidence: float = 0.0

    # Load (top cameras)
    load_status: str = C.NO_DATA
    load_confidence: float = 0.0

    # Damage
    top_damage: str = C.NO_DATA
    top_damage_details: List[Dict[str, Any]] = field(default_factory=list)
    side_damage: str = C.NO_DATA
    side_damage_details: List[Dict[str, Any]] = field(default_factory=list)

    # Provenance
    supporting_cameras: List[str] = field(default_factory=list)
    missing_cameras: List[str] = field(default_factory=list)
    confidence: float = 0.0          # 0..1 combined
    anomalies: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ---- Incremental-lifecycle provenance (v4; additive -- old readers that
    #      key on the fields above are unaffected) ----
    global_state_version: str = ""            # sealed GST hash this fusion came from
    fusion_revision: int = 0
    fused_at: str = ""                        # ISO timestamp of this fusion
    # {field_name -> source camera id} for every fused field
    field_sources: Dict[str, str] = field(default_factory=dict)
    # {field_name -> result state} distinguishing PENDING_CAMERA /
    # CAMERA_MISSING_FINAL / NO_FRAMES / FAILED / DISABLED_BY_USER / OK
    field_status: Dict[str, str] = field(default_factory=dict)
    # {camera_id -> arrival/result status}
    camera_status: Dict[str, str] = field(default_factory=dict)
    # wagon-level rollup: PENDING / COMPLETE_NO_ANOMALY / COMPLETE_WITH_ANOMALY
    result_state: str = "PENDING"

    # ----------------------------------------------------------------
    # convenience predicates
    # ----------------------------------------------------------------

    @property
    def has_open_door(self) -> bool:
        return self.left_door == C.DOOR_OPEN or self.right_door == C.DOOR_OPEN

    @property
    def has_damage(self) -> bool:
        return (self.top_damage == C.DAMAGE_PRESENT
                or self.side_damage == C.DAMAGE_PRESENT)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_wagons(wagons: List[UnifiedWagonState]) -> Dict[str, Any]:
    """Train-level summary: counts of every flagged condition."""
    return {
        "total_wagons":   len(wagons),
        "engine_count":   sum(1 for w in wagons if w.classification == C.CLASS_ENGINE),
        "wagon_count":    sum(1 for w in wagons if w.classification == C.CLASS_WAGON),
        "brake_van_count":sum(1 for w in wagons if w.classification == C.CLASS_BRAKE_VAN),
        "left_doors_open":  sum(1 for w in wagons if w.left_door == C.DOOR_OPEN),
        "right_doors_open": sum(1 for w in wagons if w.right_door == C.DOOR_OPEN),
        "loaded":           sum(1 for w in wagons if w.load_status == C.LOAD_LOADED),
        "empty":            sum(1 for w in wagons if w.load_status == C.LOAD_EMPTY),
        "top_damaged":      sum(1 for w in wagons if w.top_damage == C.DAMAGE_PRESENT),
        "side_damaged":     sum(1 for w in wagons if w.side_damage == C.DAMAGE_PRESENT),
        "ocr_captured":     sum(1 for w in wagons if w.wagon_identifier != C.NO_DATA),
    }
