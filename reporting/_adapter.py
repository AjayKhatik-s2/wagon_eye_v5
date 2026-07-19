"""v4 backend  ->  legacy report view-model.

The legacy WagonEye report generators consumed a tabular `merged_wagons`
list and per-camera "doors" lists (one entry per detected anomaly per
camera-side, each with a `snapshot_path`).  This module composes that
view-model from the v4 train-state-native artifacts the orchestrator
already produces, with NO inference / model loads / video decoding.

Inputs (all already on disk by the time the orchestrator reaches Stage 5):
    * GlobalTrainState                          -- in-memory dataclass
    * UnifiedWagonState per wagon               -- in-memory dict
    * wagon_states/{door,damage,ocr,load}/<gw>.json  -- per-feature raw
    * evidence/<gw>/<feature>/{*.jpg, metadata.json}  -- snapshots
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState

from . import _brand
from . import _evidence_lookup as ev


# -----------------------------------------------------------------------------
# View model
# -----------------------------------------------------------------------------

@dataclass
class LegacyViewModel:
    """Direct shape of the legacy report inputs."""
    merged_wagons:   List[Dict[str, Any]] = field(default_factory=list)
    left_doors:      List[Dict[str, Any]] = field(default_factory=list)
    right_doors:     List[Dict[str, Any]] = field(default_factory=list)
    top_doors:       List[Dict[str, Any]] = field(default_factory=list)
    left_top_doors:  List[Dict[str, Any]] = field(default_factory=list)
    state_counts:    Dict[str, int]       = field(default_factory=dict)
    summary_kpis:    Dict[str, Any]       = field(default_factory=dict)
    missing_cameras: List[str]            = field(default_factory=list)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ocr_display(u: UnifiedWagonState) -> str:
    if u.wagon_identifier in (None, "", C.NO_DATA):
        return ""
    return str(u.wagon_identifier)


def _ocr_for_table(u: UnifiedWagonState) -> str:
    s = _ocr_display(u)
    return s if s else "-"


def _wagon_type_for_table(u: UnifiedWagonState) -> str:
    if u.load_status == C.LOAD_LOADED:
        return "LOADED"
    if u.load_status == C.LOAD_EMPTY:
        return "EMPTY"
    return "-"


def _rake_type(wagons: Sequence[UnifiedWagonState]) -> str:
    loaded = sum(1 for w in wagons if w.load_status == C.LOAD_LOADED)
    empty  = sum(1 for w in wagons if w.load_status == C.LOAD_EMPTY)
    if loaded == 0 and empty == 0:
        return "N/A"
    return "LOADED RAKE" if loaded >= empty else "EMPTY RAKE"


# -----------------------------------------------------------------------------
# Build doors view-model entries from a single wagon
# -----------------------------------------------------------------------------

def _door_entries(
    *, evidence_root: Optional[str], gw_id: str, wagon_number: int,
    u: UnifiedWagonState,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {'left': [...], 'right': [...]} -- legacy-style door dicts.

    Only anomalous sides are emitted (the legacy "Damaged Wagon Report"
    only iterates over open/damage entries).
    """
    out = {"left": [], "right": []}

    for side_key, state_val, conf in (
        ("left",  u.left_door,  u.left_door_confidence),
        ("right", u.right_door, u.right_door_confidence),
    ):
        if not _brand.is_side_anomaly(state_val):
            continue
        snap = ev.evidence_snapshot(evidence_root, gw_id, "door", f"{side_key}_best")
        out[side_key].append({
            "wagon_number":         wagon_number,
            "door_number":          1,
            "state":                state_val or "",
            "confidence":           float(conf or 0.0),
            "local_snapshot_path":  snap,
        })
    return out


def _damage_entries(
    *, evidence_root: Optional[str], wagon_states_root: Optional[str],
    gw_id: str, wagon_number: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {'right_top': [...], 'left_top': [...]} -- legacy damage dicts
    sourced from evidence/<gw>/damage/metadata.json::tracks plus the
    per-track snapshot path."""
    out = {"right_top": [], "left_top": []}
    md = ev.evidence_metadata(evidence_root, gw_id, "damage")
    tracks = md.get("tracks") or []
    if not tracks:
        return out

    sorted_tracks = sorted(
        (t for t in tracks if isinstance(t, dict)),
        key=lambda t: float(t.get("best_confidence") or 0.0),
        reverse=True,
    )

    for tr in sorted_tracks:
        cam = tr.get("camera_id")
        if cam not in (C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
            continue
        idx = tr.get("track_idx")
        snap = ev.evidence_snapshot(evidence_root, gw_id, "damage", f"track_{int(idx)}") if idx else None
        bucket = "right_top" if cam == C.CAMERA_RIGHT_UP_TOP else "left_top"
        out[bucket].append({
            "wagon_number":        wagon_number,
            "damage_number":       int(idx or 1),
            "state":               tr.get("class_name") or "damage",
            "confidence":          float(tr.get("best_confidence") or 0.0),
            "local_snapshot_path": snap,
            "frame_idx":           tr.get("best_frame_idx"),
            "bbox":                tr.get("bbox"),
        })
    return out


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build_legacy_view_model(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    wagon_states_root: Optional[str] = None,
    evidence_root: Optional[str] = None,
    missing_cameras: Sequence[str] = (),
) -> LegacyViewModel:
    """Compose the legacy report inputs from v4 backend state."""
    vm = LegacyViewModel(missing_cameras=list(missing_cameras))

    wagons_in_order = [u for u in (unified.get(w.global_id) for w in state.wagons) if u]

    # State counts -- used by per-feature Detection Summary tables
    sc: Dict[str, int] = {
        "OPEN":           0,
        "CLOSED":         0,
        "PARTIAL CLOSED": 0,
        "DAMAGE":         0,
    }
    for u in wagons_in_order:
        for s in (u.left_door, u.right_door):
            norm = _brand.format_door_status(s)
            if norm in sc:
                sc[norm] += 1
            elif s and s != C.NO_DATA:
                sc[norm] = sc.get(norm, 0) + 1
    vm.state_counts = sc

    # KPIs (10-column summary table)
    left_open       = sum(1 for u in wagons_in_order if u.left_door  == C.DOOR_OPEN)
    right_open      = sum(1 for u in wagons_in_order if u.right_door == C.DOOR_OPEN)
    left_partial    = sum(1 for u in wagons_in_order if u.left_door  == C.DOOR_PARTIAL)
    right_partial   = sum(1 for u in wagons_in_order if u.right_door == C.DOOR_PARTIAL)
    top_damages     = sum(1 for u in wagons_in_order if u.top_damage == C.DAMAGE_PRESENT)
    side_damages    = sum(1 for u in wagons_in_order if u.side_damage == C.DAMAGE_PRESENT)
    loaded_count    = sum(1 for u in wagons_in_order if u.load_status == C.LOAD_LOADED)
    empty_count     = sum(1 for u in wagons_in_order if u.load_status == C.LOAD_EMPTY)
    ocr_captured    = sum(1 for u in wagons_in_order if u.wagon_identifier not in (None, "", C.NO_DATA))
    # Loco numbers (5-digit, from ENGINE wagons' RIGHT_UP loco OCR), in rake order,
    # deduped -- surfaced in the KPI summary, PDF filename, and email subject.
    loco_numbers = list(dict.fromkeys(
        u.loco_number for u in wagons_in_order
        if getattr(u, "loco_number", C.NO_DATA) not in (None, "", C.NO_DATA)))

    any_anomaly = any(u.anomalies for u in wagons_in_order)
    vm.summary_kpis = {
        "total_wagons":    state.total_wagons,
        "engine_count":    state.engine_count,
        "brake_van_count": state.brake_van_count,
        "wagon_count":     state.regular_wagon_count,
        "left_open":       left_open,
        "right_open":      right_open,
        "top_damages":     top_damages,
        "left_top_damages":side_damages,
        "left_partial":    left_partial,
        "right_partial":   right_partial,
        "loaded_count":    loaded_count,
        "empty_count":     empty_count,
        "ocr_captured":    ocr_captured,
        "rake_type":       _rake_type(wagons_in_order),
        "status":          "NOT OK" if any_anomaly else "OK",
        "loco_numbers":    loco_numbers,
    }

    # merged_wagons + per-camera doors
    for idx, gw in enumerate(state.wagons, start=1):
        u = unified.get(gw.global_id)
        if u is None:
            continue
        is_non_wagon = u.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN)
        is_loaded    = (u.load_status == C.LOAD_LOADED)

        left_text  = (_brand.format_door_status(u.left_door)
                      if u.left_door  not in (None, "", C.NO_DATA)
                      else "NO DOOR DETECTED")
        right_text = (_brand.format_door_status(u.right_door)
                      if u.right_door not in (None, "", C.NO_DATA)
                      else "NO DOOR DETECTED")

        # Top-camera columns: the per-camera damage_status lives in the damage
        # FEATURE JSON (wagon_states/damage/<gw>.json), NOT the evidence
        # metadata (which only carries top_damage + tracks).  Reading the
        # feature JSON gives an accurate per-camera R-TOP / L-TOP split; we
        # fall back to the fused top_damage only when the feature JSON is
        # absent.  Note these are the v4 sentinels "DAMAGE"/"OK"/"NO_DATA".
        raw_damage = ev.read_wagon_feature_json(wagon_states_root, "damage", gw.global_id)
        per_cam = (raw_damage.get("per_camera") or {}) if isinstance(raw_damage, dict) else {}
        right_top_state = (per_cam.get(C.CAMERA_RIGHT_UP_TOP, {}) or {}).get("damage_status", "")
        left_top_state  = (per_cam.get(C.CAMERA_LEFT_UP_TOP,  {}) or {}).get("damage_status", "")
        if not right_top_state:
            right_top_state = u.top_damage or ""
        if not left_top_state:
            left_top_state = u.top_damage or ""
        top_text       = _brand.format_damage_status(right_top_state)
        left_top_text  = _brand.format_damage_status(left_top_state)

        has_open_left      = _brand.is_side_anomaly(u.left_door)
        has_open_right     = _brand.is_side_anomaly(u.right_door)
        has_open_top       = _brand.is_top_anomaly(right_top_state)
        has_open_left_top  = _brand.is_top_anomaly(left_top_state)

        vm.merged_wagons.append({
            "wagon_sr_no":         idx,
            "global_id":           gw.global_id,
            "classification":      u.classification,
            "is_loaded":           is_loaded,
            "is_non_wagon":        is_non_wagon,
            "ocr_wagon_number":    _ocr_for_table(u),
            "left_doors_text":     left_text,
            "right_doors_text":    right_text,
            "top_doors_text":      top_text,
            "left_top_doors_text": left_top_text,
            "has_open_left":       has_open_left,
            "has_open_right":      has_open_right,
            "has_open_top":        has_open_top,
            "has_open_left_top":   has_open_left_top,
            "top_damage":          u.top_damage,
            "side_damage":         u.side_damage,
            "confidence":          u.confidence,
            "anomalies":           list(u.anomalies),
            "wagon_type":          _wagon_type_for_table(u),
            "left_door":           u.left_door,
            "right_door":          u.right_door,
            "left_door_confidence":  u.left_door_confidence,
            "right_door_confidence": u.right_door_confidence,
            "load_status":         u.load_status,
            "load_confidence":     u.load_confidence,
            "start_time":          gw.start_time,
            "end_time":             gw.end_time,
            "start_frame_master":   gw.start_frame_master,
            "end_frame_master":     gw.end_frame_master,
        })

        door_entries = _door_entries(
            evidence_root=evidence_root, gw_id=gw.global_id,
            wagon_number=idx, u=u,
        )
        vm.left_doors.extend(door_entries["left"])
        vm.right_doors.extend(door_entries["right"])

        dmg_entries = _damage_entries(
            evidence_root=evidence_root,
            wagon_states_root=wagon_states_root,
            gw_id=gw.global_id, wagon_number=idx,
        )
        vm.top_doors.extend(dmg_entries["right_top"])
        vm.left_top_doors.extend(dmg_entries["left_top"])

    return vm
