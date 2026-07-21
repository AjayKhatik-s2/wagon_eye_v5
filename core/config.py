"""Central configuration layer for the wagon_eye_v4 pipeline.

Single source of truth for every filesystem path and every runtime knob
that used to be scattered across modules or hardcoded for a SageMaker
notebook.  Every value is env-override-capable and defaults to *exactly*
the path/behaviour the pipeline used before the EC2 migration, so a
deployment that sets no environment variables behaves identically to the
pre-migration code.

Design rules:
    * No module should hardcode an absolute path -- import from here.
    * PROJECT_ROOT is discovered dynamically from this file's location, so
      the project works no matter where it is cloned on the EC2 host.
    * Nothing here loads a model, reads a frame, or touches GlobalTrainState.
      It is pure configuration (mirrors core/feature_config.py's discipline).

Environment variables (all optional):
    WAGONEYE_WORKSPACE_ROOT     output root (default <root>/batch_outputs)
    WAGONEYE_MODELS_DIR         models root (default <root>/models)
    WAGONEYE_RECON_MODELS_DIR   reconstruction .pt dir
    WAGONEYE_FEAT_MODELS_DIR    feature .pt dir (v4-native models; shelved for milestone 1)
    WAGONEYE_PROD_MODELS_DIR    PRODUCTION .pt dir (milestone-1 authoritative models)
    WAGONEYE_LOCAL_INPUTS_DIR   default --local-inputs folder
    WAGONEYE_LOG_DIR            log directory (default <root>/logs)
    WAGONEYE_LOG_LEVEL          root log level (default INFO)
    WAGONEYE_DEVICE             force 'cuda' / 'cpu' (default: auto-detect)
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# Project root -- discovered from this file, never hardcoded.
# core/config.py  ->  <PROJECT_ROOT>/core/config.py, so PROJECT_ROOT is two
# levels up.  Works regardless of where the repo is cloned on the EC2 box.
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(var: str, default: str) -> str:
    """Return an absolute path from an env var, falling back to `default`.

    A relative env value is resolved against PROJECT_ROOT so the project
    still works no matter what the process working directory is.
    """
    raw = os.getenv(var)
    if not raw:
        return default
    return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)


def _env_str(var: str, default: str) -> str:
    val = os.getenv(var)
    return val if val else default


def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(var: str, default: bool) -> bool:
    raw = os.getenv(var)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# -----------------------------------------------------------------------------
# Filesystem paths (all overridable; all default to the pre-migration values)
# -----------------------------------------------------------------------------

MODELS_DIR       = _env_path("WAGONEYE_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
RECON_MODELS_DIR = _env_path("WAGONEYE_RECON_MODELS_DIR",
                             os.path.join(MODELS_DIR, "reconstruction"))
FEAT_MODELS_DIR  = _env_path("WAGONEYE_FEAT_MODELS_DIR",
                             os.path.join(MODELS_DIR, "features"))
# Milestone-1 authoritative models: the PRODUCTION .pt weights (side_damage,
# top_left_damage, right_top_damage, wagon_number, ltop, top_classification).
# These reproduce production feature behaviour exactly; the v4-native models in
# FEAT_MODELS_DIR are shelved until the post-milestone model-swap phase. On EC2,
# copy the production .pt files here and the pipeline runs with no code change.
# See models/production/README.md and core/production_models.py.
PROD_MODELS_DIR  = _env_path("WAGONEYE_PROD_MODELS_DIR",
                             os.path.join(MODELS_DIR, "production"))

WORKSPACE_ROOT   = _env_path("WAGONEYE_WORKSPACE_ROOT",
                             os.path.join(PROJECT_ROOT, "batch_outputs"))
LOCAL_INPUTS_DIR = _env_path("WAGONEYE_LOCAL_INPUTS_DIR",
                             os.path.join(PROJECT_ROOT, "local_inputs"))
LOG_DIR          = _env_path("WAGONEYE_LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))

# Per-page report logo (was previously built from the wrong base dir in
# orchestrator/master_runner.py, so it silently never resolved).
LOGO_PATH = os.path.join(PROJECT_ROOT, "reporting", "assets", "Logo.jpeg")

LOG_LEVEL = _env_str("WAGONEYE_LOG_LEVEL", "INFO")


# -----------------------------------------------------------------------------
# Per-batch output subfolder names (were inline string literals in
# orchestrator/master_runner.py; centralized so renaming is a one-line edit).
# -----------------------------------------------------------------------------

DIR_DOWNLOADS        = "downloads"
DIR_GLOBAL_STATE     = "global_state"
DIR_WAGON_CACHE      = "wagon_cache"
DIR_WAGON_STATES     = "wagon_states"
DIR_EVIDENCE         = "evidence"
DIR_PROCESSED_VIDEOS = "processed_videos"
DIR_REPORTS          = "reports"
DIR_ARCHIVE          = "archive"

BATCH_SUBDIRS = (
    DIR_DOWNLOADS, DIR_GLOBAL_STATE, DIR_WAGON_CACHE, DIR_WAGON_STATES,
    DIR_EVIDENCE, DIR_PROCESSED_VIDEOS, DIR_REPORTS, DIR_ARCHIVE,
)


# -----------------------------------------------------------------------------
# Device resolution (CPU / CUDA) -- centralized so every model load and every
# inference call selects the same device deterministically.  Before the
# migration nothing branched on torch.cuda.is_available(); ultralytics guessed
# and easyocr was hard-forced to GPU.  These helpers make the choice explicit
# and give a clean CPU fallback on a CPU-only EC2 instance while preserving
# the exact GPU behaviour on GPU hosts.
# -----------------------------------------------------------------------------

def resolve_device(force: str | None = None) -> str:
    """Return 'cuda' or 'cpu'.

    Precedence:
        1. `force` argument (explicit caller override).
        2. WAGONEYE_DEVICE env var ('cuda' / 'cpu' / 'auto').
        3. Auto-detect via torch.cuda.is_available().

    Any torch import/detection failure degrades safely to 'cpu' so the
    pipeline never crashes on a box without a working CUDA stack.
    """
    choice = (force or os.getenv("WAGONEYE_DEVICE") or "auto").strip().lower()
    if choice in ("cuda", "gpu"):
        return "cuda"
    if choice == "cpu":
        return "cpu"
    # auto-detect
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def use_half_precision(device: str | None = None) -> bool:
    """FP16 inference is only safe/beneficial on CUDA.

    Returns True only for a CUDA device -- this preserves the pre-migration
    `half=True` behaviour on GPU exactly, while avoiding the
    half-precision-on-CPU footgun on a CPU-only host.
    """
    dev = device if device is not None else resolve_device()
    return dev == "cuda"


# -----------------------------------------------------------------------------
# Incremental batch-lifecycle deadlines + policy flags (all env-overridable).
#
# These three deadlines are SEMANTICALLY SEPARATE and must not be conflated:
#   MASTER_WAIT_MINUTES        -- how long, from first-seen, we wait for the
#                                 RIGHT_UP master before it is "late".
#   SUPPORT_FUSION_WAIT_MINUTES -- short window AFTER RIGHT_UP arrives to let
#                                 support cameras show up and improve Stage-1
#                                 cross-camera gap recovery.  When it expires we
#                                 seal from RIGHT_UP + whatever support exists --
#                                 we do NOT wait for the final deadline.
#   FINAL_CAMERA_WAIT_MINUTES  -- from first-seen, the hard close: still-missing
#                                 cameras become CAMERA_MISSING_FINAL and the
#                                 batch finalizes (completed_partial).  This is
#                                 NOT an implicit master deadline.
# -----------------------------------------------------------------------------

MASTER_WAIT_MINUTES         = _env_float("WAGONEYE_MASTER_WAIT_MINUTES", 10.0)
SUPPORT_FUSION_WAIT_MINUTES = _env_float("WAGONEYE_SUPPORT_FUSION_WAIT_MINUTES", 3.0)
FINAL_CAMERA_WAIT_MINUTES   = _env_float("WAGONEYE_FINAL_CAMERA_WAIT_MINUTES", 30.0)

# LEFT_UP fallback master is UNVALIDATED (side_classification.pt is a RIGHT_UP
# model) -- OFF by default.  When enabled, the reconstruction path still guards
# against sealing if classification support for the chosen master is absent.
ENABLE_LEFT_UP_FALLBACK_MASTER = _env_bool("WAGONEYE_ENABLE_LEFT_UP_FALLBACK_MASTER", False)

# Interim (pre-closure) reports are regenerated on disk as cameras arrive, but
# by default they are LOCAL-ONLY: only ONE upload+email happens at closure.
# Uploading / emailing interim revisions must be explicitly opted into.
GENERATE_INTERIM_REPORTS = _env_bool("WAGONEYE_GENERATE_INTERIM_REPORTS", True)
UPLOAD_INTERIM_REPORTS   = _env_bool("WAGONEYE_UPLOAD_INTERIM_REPORTS", False)
EMAIL_INTERIM_REPORTS    = _env_bool("WAGONEYE_EMAIL_INTERIM_REPORTS", False)

# S3 prefix under which per-batch manifests are mirrored.  Empty -> the manifest
# lives at <archive_prefix>/<key>/manifest.json (default: archive/<key>/manifest.json
# in the end-results bucket).
MANIFEST_S3_PREFIX = _env_str("WAGONEYE_MANIFEST_S3_PREFIX", "")

# Poll cadence for the active-batch scheduler (seconds).
ACTIVE_BATCH_POLL_INTERVAL = int(_env_float("WAGONEYE_ACTIVE_BATCH_POLL_INTERVAL", 60))

# What to do with a camera that arrives AFTER a batch is terminal.
# IGNORE (default) -> log + drop; the sealed report is never reopened.
LATE_CAMERA_POLICY = _env_str("WAGONEYE_LATE_CAMERA_POLICY", "IGNORE").upper()

# Extraction topology.  In the FINALIZED production architecture train
# extraction runs on a SEPARATE instance and uploads complete-train clips into
# the input bucket; THIS instance is INSPECTION-ONLY and must NOT extract.  So
# the default is FALSE: `master_runner --auto` polls the input bucket and never
# touches the raw bucket, never initializes an extractor, and never downloads
# extraction models (the whole extraction sweep is skipped -- see
# master_runner._extraction_sweep, which is the only place any of that happens).
# Set true (single-instance topology only) to have ONE `--auto` process also run
# raw->trimmed extraction in-process.  `--skip-extraction` forces it off.
AUTO_RUN_EXTRACTION = _env_bool("WAGONEYE_AUTO_RUN_EXTRACTION", False)

# Automatic model synchronization: when a required model (.pt) is missing
# locally, core.model_sync downloads it from the models bucket (constants.
# S3_MODELS_BUCKET, default wagon-eye-models) before it is loaded, so a fresh
# host needs no manual model copy.  A present model is always an instant no-op.
# Set false to require models to be pre-staged on disk (the pre-sync behaviour).
MODEL_SYNC_ENABLED = _env_bool("WAGONEYE_MODEL_SYNC_ENABLED", True)


# -----------------------------------------------------------------------------
# Startup configuration validation + redacted summary
# -----------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised (collected) when the effective configuration is invalid."""


def validate_config(*, mode: str, skip_upload: bool = False,
                    skip_email: bool = False) -> list:
    """Return a list of human-readable configuration errors (empty = OK).

    `mode` is 'auto' | 'local' | 'once' | 'batch'.  The caller fails fast and
    refuses to poll when this is non-empty.
    """
    from core import constants as C
    errors: list = []

    # deadlines non-negative
    for name, val in (("MASTER_WAIT_MINUTES", MASTER_WAIT_MINUTES),
                      ("SUPPORT_FUSION_WAIT_MINUTES", SUPPORT_FUSION_WAIT_MINUTES),
                      ("FINAL_CAMERA_WAIT_MINUTES", FINAL_CAMERA_WAIT_MINUTES)):
        if val < 0:
            errors.append(f"{name} must be >= 0 (got {val})")
    # support window must not exceed the final deadline
    if SUPPORT_FUSION_WAIT_MINUTES > FINAL_CAMERA_WAIT_MINUTES:
        errors.append(
            f"SUPPORT_FUSION_WAIT_MINUTES ({SUPPORT_FUSION_WAIT_MINUTES}) must not "
            f"exceed FINAL_CAMERA_WAIT_MINUTES ({FINAL_CAMERA_WAIT_MINUTES})")
    if ACTIVE_BATCH_POLL_INTERVAL <= 0:
        errors.append("ACTIVE_BATCH_POLL_INTERVAL must be > 0")
    if LATE_CAMERA_POLICY not in ("IGNORE",):
        errors.append(f"LATE_CAMERA_POLICY={LATE_CAMERA_POLICY!r} unsupported "
                      f"(only IGNORE is implemented)")
    # interim upload/email require interim generation
    if UPLOAD_INTERIM_REPORTS and not GENERATE_INTERIM_REPORTS:
        errors.append("UPLOAD_INTERIM_REPORTS=true requires GENERATE_INTERIM_REPORTS=true")
    if EMAIL_INTERIM_REPORTS and not GENERATE_INTERIM_REPORTS:
        errors.append("EMAIL_INTERIM_REPORTS=true requires GENERATE_INTERIM_REPORTS=true")

    # S3 discovery for continuous polling
    if mode in ("auto", "once", "batch"):
        if not C.S3_OUTPUT_BUCKET:
            errors.append("WAGONEYE_S3_OUTPUT_BUCKET is required for --auto/--once/--batch")
        if not C.S3_INPUT_PREFIXES:
            errors.append("WAGONEYE_S3_INPUT_PREFIXES is empty -- --auto would discover "
                          "nothing.  Set it to the camera-video prefix(es).")
    # delivery endpoints required unless explicitly skipped
    if mode in ("auto", "once", "batch"):
        if not skip_email and (not C.EMAIL_API_URL or not C.EMAIL_RECEIVER):
            errors.append("email enabled but EMAIL_API_URL / EMAIL_RECEIVER missing "
                          "(or pass --skip-email)")

    # writable dirs
    import tempfile as _tf
    for name, d in (("WORKSPACE_ROOT", WORKSPACE_ROOT), ("LOG_DIR", LOG_DIR),
                    ("TMPDIR", _tf.gettempdir())):
        try:
            os.makedirs(d, exist_ok=True)
            if not os.access(d, os.W_OK):
                errors.append(f"{name} is not writable: {d}")
        except OSError as e:
            errors.append(f"{name} could not be created ({d}): {e}")
    return errors


def startup_summary(*, mode: str) -> str:
    """A single multi-line summary of effective incremental settings.
    Secrets and recipient addresses are REDACTED (counts only)."""
    from core import constants as C
    n_to = len(C.EMAIL_RECEIVER or [])
    n_cc = len(C.EMAIL_RECEIVER_CC or [])
    lines = [
        "WagonEye v4 effective configuration:",
        f"  mode                     : {mode}",
        f"  device                   : {resolve_device()}",
        f"  workspace                : {WORKSPACE_ROOT}",
        f"  log_dir                  : {LOG_DIR}",
        f"  master_wait_min          : {MASTER_WAIT_MINUTES}",
        f"  support_fusion_wait_min  : {SUPPORT_FUSION_WAIT_MINUTES}",
        f"  final_camera_wait_min    : {FINAL_CAMERA_WAIT_MINUTES}",
        f"  left_up_fallback_master  : {ENABLE_LEFT_UP_FALLBACK_MASTER} (experimental)",
        f"  generate_interim_reports : {GENERATE_INTERIM_REPORTS}",
        f"  upload_interim_reports   : {UPLOAD_INTERIM_REPORTS}",
        f"  email_interim_reports    : {EMAIL_INTERIM_REPORTS}",
        f"  late_camera_policy       : {LATE_CAMERA_POLICY}",
        f"  auto_run_extraction      : {AUTO_RUN_EXTRACTION} (in-process RAW->trimmed)",
        f"  model_sync_enabled       : {MODEL_SYNC_ENABLED} (bucket={C.S3_MODELS_BUCKET})",
        f"  poll_interval_s          : {ACTIVE_BATCH_POLL_INTERVAL}",
        f"  s3_raw_bucket            : {C.S3_RAW_BUCKET}",
        f"  s3_trimmed_bucket        : {C.S3_TRIMMED_BUCKET}",
        f"  s3_output_bucket         : {C.S3_OUTPUT_BUCKET}",
        f"  s3_input_bucket          : {C.S3_INPUT_BUCKET}",
        f"  s3_input_prefixes        : {len(C.S3_INPUT_PREFIXES)} configured",
        f"  s3_output_prefixes       : reports={C.S3_REPORTS_PREFIX} "
        f"dashboard={C.S3_DASHBOARD_PREFIX} archive={C.S3_ARCHIVE_PREFIX}",
        f"  s3_state_key             : {C.S3_STATE_KEY}",
        f"  email_recipients         : to={n_to} cc={n_cc} (redacted)",
    ]
    return "\n".join(lines)
