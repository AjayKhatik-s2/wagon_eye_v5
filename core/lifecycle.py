"""Lifecycle, arrival, and result-state vocabularies for the resumable,
incremental batch lifecycle.

Three DISTINCT state spaces are modelled here and must never be conflated
(this separation is a hard requirement -- a camera that has simply not
arrived yet is NOT the same as a feature that produced no data):

    LifecycleState  -- where a whole batch is in the seal -> process ->
                       finalize flow.  Persisted as `lifecycle_status` in
                       the BatchManifest.
    ArrivalState    -- per-camera: has this camera's video shown up yet?
                       Persisted per camera in the manifest.
    ResultState     -- per (camera, feature) or per unified field: the
                       outcome of inference / fusion.  `PENDING_CAMERA`
                       here means "the owning camera has not arrived" and
                       must never be collapsed to a generic NO_DATA before
                       final closure.

Only the terminal LifecycleStates are written into
`master_runner/processed_batches.json`; every non-terminal batch lives on
as an active BatchManifest and is revisited on later polls.
"""

from __future__ import annotations

from typing import Set

from . import constants as C


# -----------------------------------------------------------------------------
# Lifecycle states (batch-level)
# -----------------------------------------------------------------------------

class LifecycleState:
    # -- pre-seal (collecting inputs / waiting on deadlines) --
    DISCOVERED               = "DISCOVERED"
    COLLECTING_CAMERAS       = "COLLECTING_CAMERAS"
    WAITING_FOR_MASTER       = "WAITING_FOR_MASTER"
    WAITING_FOR_SUPPORT      = "WAITING_FOR_SUPPORT"
    RECONSTRUCTING           = "RECONSTRUCTING"
    # -- post-seal (GlobalTrainState is immutable from here on) --
    GLOBAL_STATE_SEALED      = "GLOBAL_STATE_SEALED"
    PROCESSING_AVAILABLE     = "PROCESSING_AVAILABLE_CAMERAS"
    WAITING_FOR_LATE_CAMERAS = "WAITING_FOR_LATE_CAMERAS"
    PROCESSING_LATE_CAMERA   = "PROCESSING_LATE_CAMERA"
    FINALIZING               = "FINALIZING"
    # -- terminal --
    COMPLETED                = "COMPLETED"
    COMPLETED_PARTIAL        = "COMPLETED_PARTIAL"
    FAILED_NO_GLOBAL_STATE   = "FAILED_NO_GLOBAL_STATE"
    REPORT_FAILED            = "REPORT_FAILED"
    FAILED                   = "FAILED"


TERMINAL_STATES: Set[str] = {
    LifecycleState.COMPLETED,
    LifecycleState.COMPLETED_PARTIAL,
    LifecycleState.FAILED_NO_GLOBAL_STATE,
    LifecycleState.REPORT_FAILED,
    LifecycleState.FAILED,
}

# A GlobalTrainState exists (sealed) in any of these states.
SEALED_STATES: Set[str] = {
    LifecycleState.GLOBAL_STATE_SEALED,
    LifecycleState.PROCESSING_AVAILABLE,
    LifecycleState.WAITING_FOR_LATE_CAMERAS,
    LifecycleState.PROCESSING_LATE_CAMERA,
    LifecycleState.FINALIZING,
    LifecycleState.COMPLETED,
    LifecycleState.COMPLETED_PARTIAL,
}


# Map a terminal LifecycleState -> the legacy `processed_batches.json` value
# (constants.BATCH_*) so the persisted terminal vocabulary is unchanged.
_TERMINAL_TO_BATCH = {
    LifecycleState.COMPLETED:              C.BATCH_COMPLETED,
    LifecycleState.COMPLETED_PARTIAL:      C.BATCH_COMPLETED_PARTIAL,
    LifecycleState.FAILED_NO_GLOBAL_STATE: C.BATCH_FAILED_NO_GLOBAL,
    LifecycleState.REPORT_FAILED:          C.BATCH_REPORT_FAILED,
    LifecycleState.FAILED:                 C.BATCH_FAILED,
}


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def is_sealed(state: str) -> bool:
    return state in SEALED_STATES


def terminal_batch_status(state: str) -> str:
    """Return the constants.BATCH_* string persisted in processed_batches.json
    for a terminal LifecycleState.  Non-terminal states raise (they are never
    persisted as terminal)."""
    try:
        return _TERMINAL_TO_BATCH[state]
    except KeyError:
        raise ValueError(f"{state!r} is not a terminal lifecycle state")


# -----------------------------------------------------------------------------
# Arrival state (per camera)
# -----------------------------------------------------------------------------

class ArrivalState:
    PENDING_CAMERA       = "PENDING_CAMERA"        # video not seen yet
    PRESENT              = "PRESENT"               # video discovered/downloaded
    CAMERA_MISSING_FINAL = "CAMERA_MISSING_FINAL"  # never arrived by final deadline


# -----------------------------------------------------------------------------
# Result state (per camera-feature / per unified field)
# -----------------------------------------------------------------------------

class ResultState:
    PENDING_CAMERA       = "PENDING_CAMERA"        # owning camera not arrived (pre-closure)
    CAMERA_MISSING_FINAL = "CAMERA_MISSING_FINAL"  # owning camera never arrived (post-closure)
    NO_FRAMES            = C.STATUS_NO_FRAMES      # camera present but no cache frames
    FAILED               = C.STATUS_FAILED         # inference raised for this wagon/feature
    DISABLED_BY_USER     = C.STATUS_DISABLED       # feature toggled off
    COMPLETE_NO_ANOMALY  = "COMPLETE_NO_ANOMALY"   # valid inference, nothing flagged
    COMPLETE_WITH_ANOMALY = "COMPLETE_WITH_ANOMALY"  # valid inference, anomaly flagged
    OK                   = C.STATUS_OK             # valid inference (value carried elsewhere)


# States that mean "no usable value yet, but NOT a permanent absence"; these
# must be shown distinctly in reports and never coerced to generic NO_DATA
# before final closure.
_PENDING_RESULT_STATES: Set[str] = {
    ResultState.PENDING_CAMERA,
}


def is_pending(result_state: str) -> bool:
    return result_state in _PENDING_RESULT_STATES
