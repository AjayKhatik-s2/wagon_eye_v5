"""Canonical constants shared across the wagon_eye_v4 pipeline."""

from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    """Read a WAGONEYE_* override, falling back to the pre-migration default."""
    val = os.getenv(name)
    return val if val else default


def _env_list(name: str, default: list) -> list:
    """Comma/semicolon separated env override for a recipient list."""
    raw = os.getenv(name)
    if not raw:
        return default
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]

# -----------------------------------------------------------------------------
# Cameras
# -----------------------------------------------------------------------------

CAMERA_RIGHT_UP     = "RIGHT_UP"
CAMERA_LEFT_UP      = "LEFT_UP"
CAMERA_RIGHT_UP_TOP = "RIGHT_UP_TOP"
CAMERA_LEFT_UP_TOP  = "LEFT_UP_TOP"

ALL_CAMERAS = (
    CAMERA_RIGHT_UP, CAMERA_LEFT_UP,
    CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP,
)
SIDE_CAMERAS = (CAMERA_RIGHT_UP, CAMERA_LEFT_UP)
TOP_CAMERAS  = (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)
MASTER_CAMERA = CAMERA_RIGHT_UP

# Canonical lowercase cache folder name per camera
CAMERA_FOLDER = {
    CAMERA_RIGHT_UP:     "right_up",
    CAMERA_LEFT_UP:      "left_up",
    CAMERA_RIGHT_UP_TOP: "right_up_top",
    CAMERA_LEFT_UP_TOP:  "left_up_top",
}

# Reverse lookup
CAMERA_FROM_FOLDER = {v: k for k, v in CAMERA_FOLDER.items()}


# -----------------------------------------------------------------------------
# Status sentinel values
# -----------------------------------------------------------------------------

NO_DATA       = "NO_DATA"
STATUS_OK     = "OK"
STATUS_FAILED = "FAILED"
STATUS_NO_FRAMES = "NO_FRAMES"
STATUS_DISABLED  = "DISABLED_BY_USER"   # feature-JSON status when a user toggled it OFF

# Display string carried in UnifiedWagonState fields owned by a disabled
# feature, and rendered verbatim in reports in place of NO_DATA / OK.
DISABLED_DISPLAY = "DISABLED BY USER"

# Batch outcome statuses persisted in processed_batches.json
BATCH_COMPLETED          = "completed"
BATCH_COMPLETED_PARTIAL  = "completed_partial"
BATCH_REPORT_FAILED      = "report_failed"
BATCH_FAILED_NO_GLOBAL   = "failed_no_global_state"
BATCH_FAILED             = "failed"


# -----------------------------------------------------------------------------
# Classification labels (matching wagon_count.global_train_state.SegmentClass)
# -----------------------------------------------------------------------------

CLASS_ENGINE    = "ENGINE"
CLASS_WAGON     = "WAGON"
CLASS_BRAKE_VAN = "BRAKE_VAN"
CLASS_UNKNOWN   = "UNKNOWN"


# -----------------------------------------------------------------------------
# Reconstruction model filenames (in models/reconstruction/)
# -----------------------------------------------------------------------------

# Short names (preferred); the wagon_count package now also accepts these.
MODEL_RIGHT_UP_GAP        = "right_up_gap.pt"
MODEL_LEFT_UP_GAP         = "left_up_gap.pt"
MODEL_TOP_GAP             = "top_gap.pt"
MODEL_SIDE_CLASSIFICATION = "side_classification.pt"


# -----------------------------------------------------------------------------
# Feature model filenames (in models/features/)
# -----------------------------------------------------------------------------

MODEL_DOOR_STATE        = "door_state.pt"
MODEL_LOADED            = "loaded.pt"
MODEL_DAMAGE            = "damage.pt"
MODEL_WAGON_ID_COUNTING = "wagon_id_counting.pt"


# -----------------------------------------------------------------------------
# Door state vocabulary (from the trained door_state.pt model)
# -----------------------------------------------------------------------------

DOOR_CLOSED  = "CLOSED"
DOOR_OPEN    = "OPEN"
DOOR_PARTIAL = "PARTIAL"
DOOR_DAMAGED = "DAMAGED"

# Map raw YOLO class names to canonical door states. Anything not in the
# dict is preserved verbatim (uppercased) so downstream can still see it.
DOOR_LABEL_TO_STATE = {
    "open":               DOOR_OPEN,
    "open_door":          DOOR_OPEN,
    "closed":             DOOR_CLOSED,
    "closed_door":        DOOR_CLOSED,
    "closed_with_wire":   DOOR_PARTIAL,
    "partial_closed":     DOOR_PARTIAL,
    "partially_closed":   DOOR_PARTIAL,
    "partial":            DOOR_PARTIAL,
    "damage":             DOOR_DAMAGED,
}


# -----------------------------------------------------------------------------
# Load status vocabulary
# -----------------------------------------------------------------------------

LOAD_LOADED = "LOADED"
LOAD_EMPTY  = "EMPTY"

LOAD_LABEL_TO_STATE = {
    "loaded": LOAD_LOADED,
    "load":   LOAD_LOADED,
    "full":   LOAD_LOADED,
    "empty":  LOAD_EMPTY,
    "unload": LOAD_EMPTY,
}


# -----------------------------------------------------------------------------
# Damage vocabulary (top cameras)
# -----------------------------------------------------------------------------

DAMAGE_PRESENT = "DAMAGE"
DAMAGE_OK      = "OK"

# Top-camera damage classes we COUNT as damage.  Outer-wall damage is
# skipped on top cameras because it is the side cameras' responsibility.
DAMAGE_CLASSES_TOP = {"floor_damage", "inner_wall_damage"}
DAMAGE_CLASSES_NEGATIVE = {"no_damage"}


# -----------------------------------------------------------------------------
# S3 storage architecture (2026 migration).  Every value is overridable via a
# WAGONEYE_* environment variable.  Three buckets, strictly separated:
#
#   RAW      biro-wagon-raw-video-copy   raw camera recordings only (extraction IN)
#   TRIMMED  complete-train              trimmed complete-train clips
#                                        (extraction OUT == inspection IN);
#                                        laid out per camera folder.
#   OUTPUT   end-results                 ALL inspection outputs, under three
#                                        purpose prefixes + the processed-batches
#                                        state file:
#                                          reports/<batch_key>/...
#                                          dashboard/<camera_folder>/<date>/...
#                                          archive/<batch_key>/...   (batch tree,
#                                            manifest, evidence, processed_videos,
#                                            global_state, wagon_states, metadata)
#                                          processed_batches.json    (duplicate guard)
#
# master_runner --auto reads ONLY the TRIMMED bucket; it never reads RAW.
# -----------------------------------------------------------------------------

S3_REGION = _env("WAGONEYE_S3_REGION", "ap-south-1")

# --- buckets ---
S3_RAW_BUCKET     = _env("WAGONEYE_S3_RAW_BUCKET", "biro-wagon-raw-video-copy")
S3_TRIMMED_BUCKET = _env("WAGONEYE_S3_TRIMMED_BUCKET", "complete-train")
S3_OUTPUT_BUCKET  = _env("WAGONEYE_S3_OUTPUT_BUCKET", "end-results")

# Inspection input == the trimmed bucket (master_runner --auto polls THIS, never RAW).
S3_INPUT_BUCKET = _env("WAGONEYE_S3_INPUT_BUCKET", S3_TRIMMED_BUCKET)
# Comma-separated camera folders under the trimmed bucket the poller scans.
S3_INPUT_PREFIXES = _env_list("WAGONEYE_S3_INPUT_PREFIXES", [
    "camera_CCTV_HZBN_DHN_1_LEFT_UP/",
    "camera_CCTV_HZBN_DHN_2_RIGHT_UP/",
    "camera_CCTV_HZBN_DHN_5_RIGHT_TOP/",
    "camera_CCTV_HZBN_DHN_6_LEFT_TOP/",
])

# --- output-bucket purpose prefixes (under S3_OUTPUT_BUCKET = end-results) ---
S3_REPORTS_PREFIX   = _env("WAGONEYE_S3_REPORTS_PREFIX", "reports")
S3_DASHBOARD_PREFIX = _env("WAGONEYE_S3_DASHBOARD_PREFIX", "dashboard")
S3_ARCHIVE_PREFIX   = _env("WAGONEYE_S3_ARCHIVE_PREFIX", "archive")

# Processed-batches state file (duplicate-processing guard), at the output-bucket
# root so it lives alongside reports/ dashboard/ archive/.
S3_STATE_KEY = _env("WAGONEYE_S3_STATE_KEY", "processed_batches.json")

# --- model registry bucket ---
# Weights (.pt) are pulled from here on demand by core.model_sync whenever a
# required model is missing locally (extraction / reconstruction / production
# feature models).  Objects are stored flat by basename
# (s3://wagon-eye-models/<name>.pt); S3_MODELS_PREFIX prepends a key prefix if set.
S3_MODELS_BUCKET = _env("WAGONEYE_MODELS_S3_BUCKET", "wagon-eye-models")
S3_MODELS_PREFIX = _env("WAGONEYE_MODELS_S3_PREFIX", "")

# DEPRECATED (pre-migration single-prefix layout `train_batch/<key>/...`).
# Retained only so an env override that still sets it does not error; NO code
# path uses it after the migration -- reports/dashboard/archive prefixes replace
# it. Remove once no deployment references WAGONEYE_S3_TRAIN_BATCH_PREFIX.
S3_TRAIN_BATCH_PREFIX = _env("WAGONEYE_S3_TRAIN_BATCH_PREFIX", "train_batch")

UPLOAD_API_URL = _env("WAGONEYE_UPLOAD_API_URL",
                      "https://reports-api.suvidhaen.com/api/upload-pdf")
EMAIL_API_URL = _env(
    "WAGONEYE_EMAIL_API_URL",
    "https://ms-pnr-location-notification-api.suvidhaen.com/"
    "notification_microservice/send-email",
)
PRODUCT_NAME = _env("WAGONEYE_PRODUCT_NAME", "CCTV-WagonEye-CombinedReports")

EMAIL_RECEIVER = _env_list("WAGONEYE_EMAIL_RECEIVER", ["atul.nitt.cse@gmail.com"])
EMAIL_RECEIVER_CC = _env_list("WAGONEYE_EMAIL_RECEIVER_CC", [
    "Shivank.kumar.s2.s2@gmail.com",
    "rithish.sheru.s2@gmail.com",
    "omarbil01.s2@gmail.com",
    "kumarankitiitps2@gmail.com",
    "ajaykhatik6367s2@gmail.com",
    "priyankagp51.s2@gmail.com",
    "aman.freelancer.s2@gmail.com",
    "rajchaudhary01.official@gmail.com",
    "shyambabugupt.s2@gmail.com",
    "contact@suvidhaen.com",
])


# -----------------------------------------------------------------------------
# Misc tunables
# -----------------------------------------------------------------------------

# Confidence floors (inference)
CONF_DOOR    = 0.40
CONF_DAMAGE  = 0.55
CONF_OCR_BOX = 0.40

# JPEG quality for materializer. q95 matches the production wagon-segment frames
# (cv2.IMWRITE_JPEG_QUALITY, 95) so the cached frames fed to the damage/OCR
# models are pixel-comparable to production during validation.
JPEG_QUALITY = 95

# OCR
WAGON_NUMBER_LENGTH = 11
