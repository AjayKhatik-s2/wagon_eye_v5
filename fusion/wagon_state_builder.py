"""Stage 4 -- fuse per-feature, per-wagon results into UnifiedWagonStates.

Two on-disk layouts are supported, chosen deterministically PER BATCH:

  * camera-scoped (new, default for any new batch):
        wagon_states/door/RIGHT_UP/<gw>.json    -> right_door
        wagon_states/door/LEFT_UP/<gw>.json     -> left_door
        wagon_states/ocr/RIGHT_UP/<gw>.json     -> wagon_identifier
        wagon_states/load/RIGHT_UP_TOP/<gw>.json (primary)
        wagon_states/load/LEFT_UP_TOP/<gw>.json  (fallback)
        wagon_states/damage/{RIGHT_UP_TOP,LEFT_UP_TOP}/<gw>.json

  * legacy flat (read-only, ONLY for old pre-existing batches):
        wagon_states/{door,ocr,load,damage}/<gw>.json

A batch is identified as camera-scoped if any per-camera feature directory (or
the .features marker dir) exists; only then are camera-scoped files read.  A
camera-scoped batch NEVER falls back to stale flat files -- a missing
camera-scoped file stays PENDING / NO_DATA, never silently satisfied by a flat
file in the same directory.

Authority rules (identical to the pre-split fusion):
    classification    <- sealed GlobalTrainState only
    wagon_identifier  <- RIGHT_UP OCR
    right_door        <- RIGHT_UP door
    left_door         <- LEFT_UP door
    load_status       <- RIGHT_UP_TOP if valid, else LEFT_UP_TOP fallback
    top_damage        <- DAMAGE if either top camera reports confirmed damage
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from core import constants as C
from core import camera_features as CF
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.lifecycle import ArrivalState, ResultState
from core.unified_wagon_state import UnifiedWagonState
from core.feature_config import get_spec
from core.logging_setup import get_logger

log = get_logger("fusion")

_ALL_FEATURES = (CF.FEATURE_DOOR, CF.FEATURE_OCR, CF.FEATURE_LOAD, CF.FEATURE_DAMAGE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# JSON I/O (forgiving reads; atomic writes)
# -----------------------------------------------------------------------------

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _signature(obj: Dict[str, Any]) -> str:
    """Content signature EXCLUDING the volatile fusion timestamp, so an
    unchanged re-fuse is recognised as identical."""
    d = dict(obj)
    d.pop("fused_at", None)
    return json.dumps(d, sort_keys=True, default=str)


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Atomic write that is IDEMPOTENT: if the file already holds the same
    content (ignoring the fused_at timestamp), it is left byte-for-byte
    untouched so a re-run with unchanged inputs produces equivalent JSON."""
    existing = _read_json(path)
    if existing is not None and _signature(existing) == _signature(obj):
        return
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".unified.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _is_ok(p):        return bool(p) and p.get("status") == C.STATUS_OK
def _is_failed(p):    return bool(p) and p.get("status") == C.STATUS_FAILED
def _is_no_frames(p): return bool(p) and p.get("status") == C.STATUS_NO_FRAMES
def _is_disabled(p):  return bool(p) and p.get("status") == C.STATUS_DISABLED


# -----------------------------------------------------------------------------
# Layout detection (deterministic, per batch)
# -----------------------------------------------------------------------------

def detect_layout(wagon_states_root: str) -> str:
    """Return 'camera' or 'flat'.  Empty / new batches default to 'camera'."""
    if os.path.isdir(os.path.join(wagon_states_root, ".features")):
        return "camera"
    for feat in _ALL_FEATURES:
        fdir = os.path.join(wagon_states_root, feat)
        if not os.path.isdir(fdir):
            continue
        for cam in C.ALL_CAMERAS:
            if os.path.isdir(os.path.join(fdir, cam)):
                return "camera"
    # No per-camera dirs: flat only if a flat <feature>/<gw>.json exists.
    for feat in _ALL_FEATURES:
        fdir = os.path.join(wagon_states_root, feat)
        if os.path.isdir(fdir):
            for fn in os.listdir(fdir):
                if fn.endswith(".json"):
                    return "flat"
    return "camera"


# -----------------------------------------------------------------------------
# Camera-scoped fusion
# -----------------------------------------------------------------------------

def _cam_json(states_root: str, feature: str, camera: str, gw_id: str):
    return _read_json(os.path.join(states_root, feature, camera, f"{gw_id}.json"))


def _result_state(present: bool, missing_final: bool, payload) -> str:
    """Map (arrival, payload) -> a ResultState for one owned field."""
    if missing_final:
        return ResultState.CAMERA_MISSING_FINAL
    if not present:
        return ResultState.PENDING_CAMERA
    if payload is None:
        return ResultState.PENDING_CAMERA      # camera present, feature not run yet
    if _is_disabled(payload):
        return ResultState.DISABLED_BY_USER
    if _is_failed(payload):
        return ResultState.FAILED
    if _is_no_frames(payload):
        return ResultState.NO_FRAMES
    if _is_ok(payload):
        return ResultState.OK
    return C.NO_DATA


def _fuse_camera_scoped(
    gw: GlobalWagon, states_root: str, *,
    camera_arrival: Dict[str, str], disabled_features: Set[str],
    gst_version: str, fusion_revision: int,
) -> UnifiedWagonState:
    gw_id = gw.global_id
    u = UnifiedWagonState(
        global_id=gw_id, wagon_index=gw.wagon_index,
        classification=gw.classification,
        classification_confidence=gw.classification_confidence,
        global_state_version=gst_version, fusion_revision=fusion_revision,
        fused_at=_now_iso(),
    )

    def present(cam):       return camera_arrival.get(cam) == ArrivalState.PRESENT
    def missing_final(cam): return camera_arrival.get(cam) == ArrivalState.CAMERA_MISSING_FINAL

    supporting: Set[str] = set()

    def _apply(field, camera, feature, payload, value_key, conf_key, conf_field):
        """Set one owned field + its per-field status/source."""
        u.field_sources[field] = camera
        if feature in disabled_features:
            u.field_status[field] = ResultState.DISABLED_BY_USER
            setattr(u, field, C.DISABLED_DISPLAY)
            return
        rs = _result_state(present(camera), missing_final(camera), payload)
        u.field_status[field] = rs
        if rs == ResultState.OK:
            val = payload.get(value_key)
            if val and val != C.NO_DATA:
                setattr(u, field, str(val))
                if conf_field:
                    setattr(u, conf_field, float(payload.get(conf_key, 0.0) or 0.0))
                supporting.add(camera)
            else:
                # present + ran but no value (e.g. ENGINE/BRAKE_VAN skip)
                supporting.add(camera)

    # ---- OCR: wagon_identifier <- RIGHT_UP ----
    ocr = _cam_json(states_root, CF.FEATURE_OCR, C.CAMERA_RIGHT_UP, gw_id)
    _apply("wagon_identifier", C.CAMERA_RIGHT_UP, CF.FEATURE_OCR, ocr,
           "wagon_identifier", "wagon_identifier_confidence",
           "wagon_identifier_confidence")
    # loco number (5-digit, ENGINE wagons) rides along with RIGHT_UP OCR.
    if _is_ok(ocr) and ocr.get("loco_number") not in (None, C.NO_DATA):
        u.loco_number = str(ocr["loco_number"])
        u.loco_number_confidence = float(ocr.get("loco_number_confidence", 0.0) or 0.0)
        u.field_sources["loco_number"] = C.CAMERA_RIGHT_UP
        u.field_status["loco_number"] = ResultState.OK

    # ---- Doors: right <- RIGHT_UP, left <- LEFT_UP ----
    dr = _cam_json(states_root, CF.FEATURE_DOOR, C.CAMERA_RIGHT_UP, gw_id)
    _apply("right_door", C.CAMERA_RIGHT_UP, CF.FEATURE_DOOR, dr,
           "right_door", "right_door_confidence", "right_door_confidence")
    dl = _cam_json(states_root, CF.FEATURE_DOOR, C.CAMERA_LEFT_UP, gw_id)
    _apply("left_door", C.CAMERA_LEFT_UP, CF.FEATURE_DOOR, dl,
           "left_door", "left_door_confidence", "left_door_confidence")

    # ---- Load: RIGHT_UP_TOP primary, LEFT_UP_TOP fallback ----
    if CF.FEATURE_LOAD in disabled_features:
        u.field_sources["load_status"] = C.CAMERA_RIGHT_UP_TOP
        u.field_status["load_status"] = ResultState.DISABLED_BY_USER
        u.load_status = C.DISABLED_DISPLAY
    else:
        lr = _cam_json(states_root, CF.FEATURE_LOAD, C.CAMERA_RIGHT_UP_TOP, gw_id)
        ll = _cam_json(states_root, CF.FEATURE_LOAD, C.CAMERA_LEFT_UP_TOP, gw_id)
        chosen_cam, chosen = None, None
        if _is_ok(lr) and lr.get("load_status") not in (None, C.NO_DATA):
            chosen_cam, chosen = C.CAMERA_RIGHT_UP_TOP, lr
        elif _is_ok(ll) and ll.get("load_status") not in (None, C.NO_DATA):
            chosen_cam, chosen = C.CAMERA_LEFT_UP_TOP, ll
        if chosen is not None:
            u.load_status = str(chosen["load_status"])
            u.load_confidence = float(chosen.get("load_confidence", 0.0) or 0.0)
            u.field_sources["load_status"] = chosen_cam
            u.field_status["load_status"] = ResultState.OK
            supporting.add(chosen_cam)
        else:
            # no valid load: report the PRIMARY camera's arrival/result state
            u.field_sources["load_status"] = C.CAMERA_RIGHT_UP_TOP
            u.field_status["load_status"] = _result_state(
                present(C.CAMERA_RIGHT_UP_TOP), missing_final(C.CAMERA_RIGHT_UP_TOP), lr)

    # ---- Damage: any top camera confirmed DAMAGE wins ----
    if CF.FEATURE_DAMAGE in disabled_features:
        u.field_sources["top_damage"] = C.CAMERA_RIGHT_UP_TOP
        u.field_status["top_damage"] = ResultState.DISABLED_BY_USER
        u.top_damage = C.DISABLED_DISPLAY
    else:
        details: List[Dict[str, Any]] = []
        damaged_cam = None
        any_ok = False
        first_source = None
        for cam in C.TOP_CAMERAS:
            dj = _cam_json(states_root, CF.FEATURE_DAMAGE, cam, gw_id)
            if first_source is None:
                first_source = cam
            if _is_ok(dj):
                any_ok = True
                if dj.get("damage_status") == C.DAMAGE_PRESENT:
                    damaged_cam = damaged_cam or cam
                    details.extend(dj.get("top_damage_details") or [])
                if present(cam):
                    supporting.add(cam)
        if damaged_cam is not None:
            u.top_damage = C.DAMAGE_PRESENT
            u.top_damage_details = details
            u.field_sources["top_damage"] = damaged_cam
            u.field_status["top_damage"] = ResultState.OK
        elif any_ok:
            u.top_damage = C.DAMAGE_OK
            u.field_sources["top_damage"] = C.CAMERA_RIGHT_UP_TOP
            u.field_status["top_damage"] = ResultState.OK
        else:
            u.field_sources["top_damage"] = C.CAMERA_RIGHT_UP_TOP
            u.field_status["top_damage"] = _result_state(
                present(C.CAMERA_RIGHT_UP_TOP), missing_final(C.CAMERA_RIGHT_UP_TOP),
                _cam_json(states_root, CF.FEATURE_DAMAGE, C.CAMERA_RIGHT_UP_TOP, gw_id))

    # ---- Side damage: DAMAGE if EITHER side camera reports confirmed damage ----
    # (production side_damage.pt `damage` class; the `damage` feature run on the
    # side cameras). Mirrors the top-damage authority pattern.
    if CF.FEATURE_DAMAGE in disabled_features:
        u.side_damage = C.DISABLED_DISPLAY
        u.field_sources["side_damage"] = C.CAMERA_RIGHT_UP
        u.field_status["side_damage"] = ResultState.DISABLED_BY_USER
    else:
        side_details: List[Dict[str, Any]] = []
        side_damaged_cam = None
        side_any_ok = False
        for cam in C.SIDE_CAMERAS:
            sj = _cam_json(states_root, CF.FEATURE_DAMAGE, cam, gw_id)
            if _is_ok(sj):
                side_any_ok = True
                if sj.get("damage_status") == C.DAMAGE_PRESENT:
                    side_damaged_cam = side_damaged_cam or cam
                    side_details.extend(sj.get("side_damage_details") or [])
                if present(cam):
                    supporting.add(cam)
        if side_damaged_cam is not None:
            u.side_damage = C.DAMAGE_PRESENT
            u.side_damage_details = side_details
            u.field_sources["side_damage"] = side_damaged_cam
            u.field_status["side_damage"] = ResultState.OK
        elif side_any_ok:
            u.side_damage = C.DAMAGE_OK
            u.field_sources["side_damage"] = C.CAMERA_RIGHT_UP
            u.field_status["side_damage"] = ResultState.OK
        else:
            u.field_sources["side_damage"] = C.CAMERA_RIGHT_UP
            u.field_status["side_damage"] = _result_state(
                present(C.CAMERA_RIGHT_UP), missing_final(C.CAMERA_RIGHT_UP),
                _cam_json(states_root, CF.FEATURE_DAMAGE, C.CAMERA_RIGHT_UP, gw_id))

    # ---- Provenance + camera status ----
    u.supporting_cameras = sorted(supporting)
    u.missing_cameras = sorted(set(C.ALL_CAMERAS) - supporting)
    u.camera_status = {cam: camera_arrival.get(cam, ArrivalState.PENDING_CAMERA)
                       for cam in C.ALL_CAMERAS}

    _finish_anomalies_and_confidence(u, disabled_features)
    return u


# -----------------------------------------------------------------------------
# Legacy flat fusion (read-only; preserves the pre-split behaviour exactly)
# -----------------------------------------------------------------------------

def _fuse_flat(gw: GlobalWagon, states_root: str, *,
               gst_version: str, fusion_revision: int) -> UnifiedWagonState:
    door   = _read_json(os.path.join(states_root, "door",   f"{gw.global_id}.json"))
    ocr    = _read_json(os.path.join(states_root, "ocr",    f"{gw.global_id}.json"))
    load   = _read_json(os.path.join(states_root, "load",   f"{gw.global_id}.json"))
    damage = _read_json(os.path.join(states_root, "damage", f"{gw.global_id}.json"))

    u = UnifiedWagonState(
        global_id=gw.global_id, wagon_index=gw.wagon_index,
        classification=gw.classification,
        classification_confidence=gw.classification_confidence,
        global_state_version=gst_version, fusion_revision=fusion_revision,
        fused_at=_now_iso(),
    )
    if _is_ok(ocr) and ocr.get("wagon_identifier") not in (None, C.NO_DATA):
        u.wagon_identifier = str(ocr["wagon_identifier"])
        u.wagon_identifier_confidence = float(ocr.get("wagon_identifier_confidence", 0.0) or 0.0)
    if _is_ok(ocr) and ocr.get("loco_number") not in (None, C.NO_DATA):
        u.loco_number = str(ocr["loco_number"])
        u.loco_number_confidence = float(ocr.get("loco_number_confidence", 0.0) or 0.0)
    if _is_ok(door):
        if door.get("left_door") not in (None, C.NO_DATA):
            u.left_door = str(door["left_door"])
            u.left_door_confidence = float(door.get("left_door_confidence", 0.0) or 0.0)
        if door.get("right_door") not in (None, C.NO_DATA):
            u.right_door = str(door["right_door"])
            u.right_door_confidence = float(door.get("right_door_confidence", 0.0) or 0.0)
    if _is_ok(load) and load.get("load_status") not in (None, C.NO_DATA):
        u.load_status = str(load["load_status"])
        u.load_confidence = float(load.get("load_confidence", 0.0) or 0.0)
    if _is_ok(damage) and damage.get("top_damage") not in (None, C.NO_DATA):
        u.top_damage = str(damage["top_damage"])
        u.top_damage_details = list(damage.get("top_damage_details") or [])

    disabled: Set[str] = set()
    for key, payload in (("door", door), ("ocr", ocr), ("load", load), ("damage", damage)):
        if _is_disabled(payload):
            disabled.add(key)
            spec = get_spec(key)
            if spec:
                for fld in spec.owned_fields:
                    if hasattr(u, fld):
                        setattr(u, fld, C.DISABLED_DISPLAY)

    supporting: Set[str] = set()
    for payload in (door, ocr, load, damage):
        if _is_ok(payload):
            for c in payload.get("supporting_cameras") or []:
                supporting.add(c)
    u.supporting_cameras = sorted(supporting)
    u.missing_cameras = sorted(set(C.ALL_CAMERAS) - supporting)
    u.camera_status = {cam: (ArrivalState.PRESENT if cam in supporting
                             else ArrivalState.PENDING_CAMERA)
                       for cam in C.ALL_CAMERAS}
    _finish_anomalies_and_confidence(u, disabled)
    return u


# -----------------------------------------------------------------------------
# Shared: anomalies + combined confidence (identical rule to the old fusion)
# -----------------------------------------------------------------------------

def _finish_anomalies_and_confidence(u: UnifiedWagonState, disabled: Set[str]) -> None:
    anomalies: List[str] = []
    if u.left_door == C.DOOR_OPEN and CF.FEATURE_DOOR not in disabled:
        anomalies.append("LEFT_DOOR_OPEN")
    if u.right_door == C.DOOR_OPEN and CF.FEATURE_DOOR not in disabled:
        anomalies.append("RIGHT_DOOR_OPEN")
    if u.top_damage == C.DAMAGE_PRESENT and CF.FEATURE_DAMAGE not in disabled:
        anomalies.append("TOP_DAMAGE")
    if u.side_damage == C.DAMAGE_PRESENT:
        anomalies.append("SIDE_DAMAGE")
    # OCR_MISSING only once RIGHT_UP has actually produced a result (not while
    # pending / missing / disabled).
    ocr_state = u.field_status.get("wagon_identifier")
    ocr_ran = ocr_state in (ResultState.OK, ResultState.NO_FRAMES) or ocr_state is None
    if (u.wagon_identifier == C.NO_DATA and u.classification == C.CLASS_WAGON
            and CF.FEATURE_OCR not in disabled and ocr_ran
            and u.field_status.get("wagon_identifier") != ResultState.PENDING_CAMERA):
        anomalies.append("OCR_MISSING")
    u.anomalies = sorted(anomalies)

    confs = [c for c in (u.wagon_identifier_confidence, u.left_door_confidence,
                         u.right_door_confidence, u.load_confidence) if c > 0]
    u.confidence = round(sum(confs) / len(confs), 4) if confs else 0.0

    # wagon-level rollup
    pending = any(v == ResultState.PENDING_CAMERA for v in u.field_status.values())
    if u.anomalies:
        u.result_state = ResultState.COMPLETE_WITH_ANOMALY
    elif pending:
        u.result_state = "PENDING"
    else:
        u.result_state = ResultState.COMPLETE_NO_ANOMALY

    for cam in u.missing_cameras:
        u.notes.append(f"missing:{cam}")
    u.notes = sorted(set(u.notes))


# -----------------------------------------------------------------------------
# Camera arrival inference (when the caller doesn't pass explicit context)
# -----------------------------------------------------------------------------

def _infer_camera_arrival(states_root: str, layout: str) -> Dict[str, str]:
    """A camera is PRESENT if any per-camera feature dir it owns exists; else
    PENDING.  (CAMERA_MISSING_FINAL requires the closure signal from the caller.)"""
    if layout == "flat":
        return {cam: ArrivalState.PRESENT for cam in C.ALL_CAMERAS}
    arrival: Dict[str, str] = {}
    for cam in C.ALL_CAMERAS:
        present = any(
            os.path.isdir(os.path.join(states_root, u.feature, cam))
            for u in CF.units_for_camera(cam)
        )
        arrival[cam] = ArrivalState.PRESENT if present else ArrivalState.PENDING_CAMERA
    return arrival


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build(
    *,
    state: GlobalTrainState,
    wagon_states_root: str,
    camera_arrival: Optional[Dict[str, str]] = None,
    disabled_features: Optional[Set[str]] = None,
    global_state_version: str = "",
    fusion_revision: int = 0,
    write_per_wagon_json: bool = True,
    verbose: bool = True,
) -> Dict[str, UnifiedWagonState]:
    """Fuse every wagon into a UnifiedWagonState, atomically writing
    wagon_states/unified/<gw>.json.  Deterministic + idempotent."""
    layout = detect_layout(wagon_states_root)
    if camera_arrival is None:
        camera_arrival = _infer_camera_arrival(wagon_states_root, layout)
    disabled_features = set(disabled_features or ())

    unified_dir = os.path.join(wagon_states_root, "unified")
    if write_per_wagon_json:
        os.makedirs(unified_dir, exist_ok=True)

    if verbose:
        log.info("[STAGE4] fusing %d wagons  layout=%s  arrival=%s  disabled=%s",
                 len(state.wagons), layout, camera_arrival, sorted(disabled_features))

    unified: Dict[str, UnifiedWagonState] = {}
    t0 = time.time()
    for gw in state.wagons:
        try:
            if layout == "camera":
                u = _fuse_camera_scoped(
                    gw, wagon_states_root, camera_arrival=camera_arrival,
                    disabled_features=disabled_features,
                    gst_version=global_state_version, fusion_revision=fusion_revision)
            else:
                u = _fuse_flat(gw, wagon_states_root,
                               gst_version=global_state_version,
                               fusion_revision=fusion_revision)
            unified[gw.global_id] = u
            if write_per_wagon_json:
                # atomic: a failed write leaves the previous unified JSON intact
                _atomic_write_json(os.path.join(unified_dir, f"{gw.global_id}.json"),
                                   u.to_dict())
        except Exception as e:
            log.error("[STAGE4] fusion failed for %s: %s (previous unified JSON kept)",
                      gw.global_id, e, exc_info=True)

    if verbose:
        log.info("[STAGE4] done in %.1fs -> %s", time.time() - t0, unified_dir)
    return unified
