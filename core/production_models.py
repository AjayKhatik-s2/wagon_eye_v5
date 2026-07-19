"""Production feature-model registry + graceful loader (milestone 1).

Central, config-driven reference for the PRODUCTION ``.pt`` weights that the
Stage-3 feature processors load. Milestone 1 reproduces production behaviour
*exactly*, so these are the authoritative models; the v4-native models under
``models/features/`` are shelved until the post-milestone model-swap phase.

Contract (per the milestone-1 directive):
  * No dummy inference, no fake predictions, no bypass of model loading.
  * If a model file is absent, loading raises a clear ``MissingProductionModel``
    ("Production model not found: models/production/<name>.pt"). A processor
    catches it and marks that feature ``NO_DATA`` for every wagon (with the
    reason), so the batch still runs, seals, fuses and reports.
  * When the real ``.pt`` files are copied into ``models/production/`` on EC2,
    the pipeline runs with ZERO code changes.

Paths come from ``core.config.PROD_MODELS_DIR`` (env ``WAGONEYE_PROD_MODELS_DIR``,
default ``<repo>/models/production``). See ``models/production/README.md`` for
the per-model documentation.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

from core import config as CFG
from core import constants as C

# -----------------------------------------------------------------------------
# Production model filenames (exactly as stored in s3://wagon-eye-models/).
# -----------------------------------------------------------------------------

SIDE_DAMAGE      = "side_damage.pt"        # side doors (open/close) + side damage
TOP_LEFT_DAMAGE  = "top_left_damage.pt"    # 4-class top damage, LEFT top camera
RIGHT_TOP_DAMAGE = "right_top_damage.pt"   # 4-class top damage, RIGHT top camera
WAGON_NUMBER     = "wagon_number.pt"       # wagon-number plate bbox (RIGHT_UP OCR)
LTOP_CLASSIFY    = "ltop.pt"               # top classification (LEFT) -> load
TOP_CLASSIFY     = "top_classification.pt" # top classification (RIGHT) -> load

ALL_PRODUCTION_MODELS = (
    SIDE_DAMAGE, TOP_LEFT_DAMAGE, RIGHT_TOP_DAMAGE,
    WAGON_NUMBER, LTOP_CLASSIFY, TOP_CLASSIFY,
)

# Loco-number OCR does NOT need a production-only file: the reconstruction gap
# model already staged at models/reconstruction/right_up_gap.pt emits the
# `locono` class used to locate loco-number bands. Documented here so it is not
# mistaken for a missing production model.
LOCO_BAND_SOURCE = os.path.join("models", "reconstruction", "right_up_gap.pt")


# -----------------------------------------------------------------------------
# (feature, camera) -> production model filename
# -----------------------------------------------------------------------------
# Mirrors production camera->feature authority:
#   side cameras  (RIGHT_UP, LEFT_UP)      -> doors + side damage  (side_damage.pt)
#   top cameras   (RIGHT_UP_TOP, LEFT_TOP) -> 4-class top damage   (per-camera)
#   top cameras                            -> load via classification (per-camera)
#   RIGHT_UP                               -> wagon-number bbox     (wagon_number.pt)

_MODEL_FOR: Dict[tuple, str] = {
    ("door",        C.CAMERA_RIGHT_UP):     SIDE_DAMAGE,
    ("door",        C.CAMERA_LEFT_UP):      SIDE_DAMAGE,
    ("side_damage", C.CAMERA_RIGHT_UP):     SIDE_DAMAGE,
    ("side_damage", C.CAMERA_LEFT_UP):      SIDE_DAMAGE,
    ("damage",      C.CAMERA_RIGHT_UP_TOP): RIGHT_TOP_DAMAGE,
    ("damage",      C.CAMERA_LEFT_UP_TOP):  TOP_LEFT_DAMAGE,
    # Side damage is part of the `damage` feature run on the SIDE cameras with
    # the side model's `damage` class (production side_damage.pt).
    ("damage",      C.CAMERA_RIGHT_UP):     SIDE_DAMAGE,
    ("damage",      C.CAMERA_LEFT_UP):      SIDE_DAMAGE,
    ("load",        C.CAMERA_RIGHT_UP_TOP): TOP_CLASSIFY,
    ("load",        C.CAMERA_LEFT_UP_TOP):  LTOP_CLASSIFY,
    ("ocr",         C.CAMERA_RIGHT_UP):     WAGON_NUMBER,
}


class MissingProductionModel(FileNotFoundError):
    """Raised when a required production ``.pt`` is not present on disk."""


def _rel(filename: str) -> str:
    """Repo-relative path used in the clear error message."""
    return os.path.join("models", "production", filename).replace("\\", "/")


def resolve(filename: str) -> str:
    """Absolute path a production model *should* live at (no existence check)."""
    return os.path.join(CFG.PROD_MODELS_DIR, filename)


def require(filename: str) -> str:
    """Absolute path to an existing production model, else a clear error."""
    path = resolve(filename)
    if not os.path.isfile(path):
        raise MissingProductionModel(f"Production model not found: {_rel(filename)}")
    return path


def model_for(feature: str, camera: str) -> str:
    """Production model filename for a (feature, camera) pair."""
    try:
        return _MODEL_FOR[(feature, camera)]
    except KeyError:
        raise KeyError(
            f"No production model mapped for feature={feature!r} camera={camera!r}"
        )


# -----------------------------------------------------------------------------
# Cached loader (one YOLO per .pt per process). Loads only real weights; never
# fabricates a model. Raises MissingProductionModel when the file is absent.
# -----------------------------------------------------------------------------

_CACHE: Dict[str, object] = {}
_LOCK = threading.Lock()


def load(filename: str, *, task: Optional[str] = None):
    """Load (and cache) a production YOLO model, or raise MissingProductionModel.

    ``task`` is passed through to ultralytics (use ``"classify"`` for the top
    classification / load models). For ``.pt`` weights the task is inferred
    correctly even when omitted.
    """
    path = require(filename)  # clear error if the file is missing
    with _LOCK:
        cached = _CACHE.get(path)
        if cached is not None:
            return cached
        # torch.load shim for torch >= 2.6 (same approach as features/_common
        # and the wagon_count subpackage).
        import torch
        _orig_load = torch.load

        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)

        torch.load = _patched
        from ultralytics import YOLO
        model = YOLO(path, task=task) if task else YOLO(path)
        _CACHE[path] = model
        return model


def load_for(feature: str, camera: str, *, task: Optional[str] = None):
    """Load the production model mapped to a (feature, camera) pair."""
    return load(model_for(feature, camera), task=task)


def status() -> Dict[str, bool]:
    """{filename -> present_on_disk} for every production model (diagnostics)."""
    return {f: os.path.isfile(resolve(f)) for f in ALL_PRODUCTION_MODELS}
