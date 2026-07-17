"""Camera -> feature dependency registry.

Which camera drives which feature is fixed by the physical rig geometry and
must be explicit so late-arriving cameras trigger ONLY their own work and
never rerun another camera's models:

    RIGHT_UP      -> door (right side), ocr
    LEFT_UP       -> door (left side)
    RIGHT_UP_TOP  -> load (primary authority), damage
    LEFT_UP_TOP   -> load (support), damage

Classification is NOT a feature here -- it is owned by the sealed
GlobalTrainState (RIGHT_UP master) and is never rerun downstream.

`load` must be finalized before `damage` for a given top camera (the
loaded-wagon floor-damage filter reads that camera's completed load result),
so `ordered_features_for_camera` always yields load before damage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import constants as C


# Feature keys (match core.feature_config.FEATURE_REGISTRY)
FEATURE_DOOR   = "door"
FEATURE_OCR    = "ocr"
FEATURE_LOAD   = "load"
FEATURE_DAMAGE = "damage"

# The model file each feature loads (under models/features/).  Used to identity
# a completion marker -- a model swap must invalidate prior results.
FEATURE_MODEL_FILENAME: Dict[str, str] = {
    FEATURE_DOOR:   C.MODEL_DOOR_STATE,
    FEATURE_OCR:    C.MODEL_WAGON_ID_COUNTING,
    FEATURE_LOAD:   C.MODEL_LOADED,
    FEATURE_DAMAGE: C.MODEL_DAMAGE,
}

# Processor output-schema version per feature.  Bump when a processor's output
# shape or write layout changes so stale results are re-run rather than reused.
FEATURE_SCHEMA_VERSION: Dict[str, int] = {
    FEATURE_DOOR:   1,
    FEATURE_OCR:    1,
    FEATURE_LOAD:   1,
    FEATURE_DAMAGE: 1,
}

# The relevant thresholds per feature; hashed into the completion marker so a
# threshold change invalidates that camera-feature's cached results.
FEATURE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    FEATURE_DOOR:   {"conf_door": C.CONF_DOOR},
    FEATURE_OCR:    {"conf_ocr_box": C.CONF_OCR_BOX,
                     "wagon_number_length": float(C.WAGON_NUMBER_LENGTH)},
    FEATURE_LOAD:   {"loaded_ratio": 0.35},
    FEATURE_DAMAGE: {"conf_damage": C.CONF_DAMAGE,
                     "area_min": 0.005, "area_max": 0.40,
                     "edge_bypass_conf": 0.70},
}


@dataclass(frozen=True)
class CameraFeatureUnit:
    """One unit of feature work a single camera contributes."""
    camera_id: str
    feature: str
    role: str  # 'right' | 'left' | 'primary' | 'support' | 'sole'


# The full registry, in a deterministic order.
WORK_UNITS: Tuple[CameraFeatureUnit, ...] = (
    CameraFeatureUnit(C.CAMERA_RIGHT_UP,     FEATURE_DOOR,   "right"),
    CameraFeatureUnit(C.CAMERA_RIGHT_UP,     FEATURE_OCR,    "sole"),
    CameraFeatureUnit(C.CAMERA_LEFT_UP,      FEATURE_DOOR,   "left"),
    CameraFeatureUnit(C.CAMERA_RIGHT_UP_TOP, FEATURE_LOAD,   "primary"),
    CameraFeatureUnit(C.CAMERA_RIGHT_UP_TOP, FEATURE_DAMAGE, "primary"),
    CameraFeatureUnit(C.CAMERA_LEFT_UP_TOP,  FEATURE_LOAD,   "support"),
    CameraFeatureUnit(C.CAMERA_LEFT_UP_TOP,  FEATURE_DAMAGE, "support"),
)

# Deterministic finalization priority within a single camera: load before
# damage (floor-damage filter dependency); everything else is order-free.
_FEATURE_ORDER = {
    FEATURE_LOAD:   0,
    FEATURE_DAMAGE: 1,
    FEATURE_DOOR:   0,
    FEATURE_OCR:    1,
}


def units_for_camera(camera_id: str) -> List[CameraFeatureUnit]:
    return [u for u in WORK_UNITS if u.camera_id == camera_id]


def features_for_camera(camera_id: str) -> List[str]:
    """Distinct feature keys this camera drives, load-before-damage ordered."""
    feats = [u.feature for u in units_for_camera(camera_id)]
    # de-dup preserving order, then stable-sort by the finalization priority
    seen: Dict[str, None] = {}
    for f in feats:
        seen.setdefault(f, None)
    return sorted(seen.keys(), key=lambda f: _FEATURE_ORDER.get(f, 9))


def cameras_for_feature(feature: str) -> List[str]:
    out: List[str] = []
    for cam in C.ALL_CAMERAS:
        if any(u.feature == feature for u in units_for_camera(cam)):
            out.append(cam)
    return out


def units_for_cameras(camera_ids) -> List[CameraFeatureUnit]:
    """All work units triggered by a set of arrived cameras (registry order)."""
    s = set(camera_ids)
    return [u for u in WORK_UNITS if u.camera_id in s]
