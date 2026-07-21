"""Centralized, transparent model synchronization from S3 (wagon-eye-models).

ONE place that guarantees a required model file exists locally before it is
loaded.  Every model-load boundary in the pipeline calls `ensure_local(path)`:

    * production feature models      core.production_models.resolve()
    * direct-path YOLO loads         features._common.load_yolo()
    * reconstruction models          reconstruction.runner.run()  (before the
                                     wagon_count subprocess launches)
    * extraction classifiers         orchestrator.master_runner._extraction_sweep()

If the file is already on disk it returns immediately (Task 4: never
re-download); otherwise it downloads the matching object from the models bucket
ATOMICALLY (temp file + os.replace, Task 6) with retries and returns the local
path.

Design (per the "reuse existing utilities" directive):
  * Reuses the SAME boto3 + region + IAM-role mechanism the rest of the
    inspection pipeline already uses (delivery/*, orchestrator/*): a lazily
    created `boto3.client("s3", region_name=C.S3_REGION)`.  No new AWS
    abstraction, no new credential path, no parallel download system.
  * NEVER raises to the caller.  On any failure it logs and returns the (still
    missing) path, so each stage's EXISTING missing-model handling runs exactly
    as before -- only that stage is affected (Task 6: abort only that stage):
        production_models.require()  -> MissingProductionModel -> NO_DATA
        train_extraction driver      -> FileNotFoundError -> that camera skipped
        wagon_count subprocess       -> "Model not found" -> Stage-1 fails safe
  * Idempotent + partial (Tasks 4/5): a present, non-empty file is an instant
    no-op; only missing files are fetched, each independently.  A per-process
    negative cache avoids re-hitting S3 for a name that is not in the bucket.

Inference logic, model contents, and every stage's request pattern are
UNCHANGED -- this layer only ensures the file is on disk before the existing
loader opens it.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Dict, Iterable, Optional

from core import constants as C
from core import config as CFG
from core.logging_setup import get_logger

log = get_logger("model_sync")

# Retry policy (env-overridable; defaults reproduce sane production behaviour).
_RETRIES = max(1, int(os.getenv("WAGONEYE_MODEL_SYNC_RETRIES", "3")))
_BACKOFF_SEC = float(os.getenv("WAGONEYE_MODEL_SYNC_BACKOFF", "3"))

# Lazily created, process-wide boto3 S3 client (same mechanism as delivery/*).
_client = None
_client_lock = threading.Lock()

# Names we tried this process and could NOT fetch -- so a genuinely-absent model
# is not re-requested from S3 on every resolve() call within one long run.
_failed: set = set()
_state_lock = threading.Lock()


# -----------------------------------------------------------------------------
# S3 client + key mapping
# -----------------------------------------------------------------------------

def _s3():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            import boto3
            _client = boto3.client("s3", region_name=C.S3_REGION)
        return _client


def _s3_key_for(filename: str) -> str:
    """Object key inside the models bucket.  Models are stored flat by basename
    (s3://wagon-eye-models/<name>.pt); an optional prefix is honoured."""
    prefix = (C.S3_MODELS_PREFIX or "").strip("/")
    return f"{prefix}/{filename}" if prefix else filename


# -----------------------------------------------------------------------------
# Public: ensure a single model exists locally
# -----------------------------------------------------------------------------

def ensure_local(local_path: str, *, s3_key: Optional[str] = None,
                 bucket: Optional[str] = None) -> str:
    """Ensure `local_path` exists locally, downloading it from the models bucket
    if missing.  Returns `local_path` unconditionally (present, freshly
    downloaded, or -- on failure -- still missing so the caller's existing
    missing-model handling takes over)."""
    if not local_path:
        return local_path
    abspath = os.path.abspath(local_path)
    name = os.path.basename(abspath)

    # Present + non-empty -> instant no-op (Task 4).  This is the common path on
    # every run after the first, and on any host where models are pre-staged.
    if os.path.isfile(abspath) and os.path.getsize(abspath) > 0:
        return local_path

    if not CFG.MODEL_SYNC_ENABLED:
        log.info("[MODEL] %s missing and sync DISABLED -- leaving to stage handling",
                 name)
        return local_path

    with _state_lock:
        if abspath in _failed:
            return local_path  # already attempted this process; do not storm S3

    log.info("[MODEL] checking %s -- not present locally, syncing", name)
    key = s3_key or _s3_key_for(name)
    bkt = bucket or C.S3_MODELS_BUCKET
    if _download_atomic(bkt, key, abspath, name):
        return local_path

    with _state_lock:
        _failed.add(abspath)
    log.error("[MODEL] %s could NOT be synced from s3://%s/%s -- stage will handle "
              "the missing model", name, bkt, key)
    return local_path


def ensure_many(paths: Iterable[str]) -> Dict[str, bool]:
    """Ensure several models (Task 5: partial -- only the missing ones download).
    Returns {path -> present_after}."""
    out: Dict[str, bool] = {}
    for p in paths:
        ensure_local(p)
        out[p] = os.path.isfile(p) and os.path.getsize(p) > 0
    return out


# -----------------------------------------------------------------------------
# Atomic download (temp file + os.replace) with retries -- Task 6
# -----------------------------------------------------------------------------

def _is_permanent(e: Exception) -> bool:
    """True for errors that retrying cannot fix (missing creds, wrong endpoint,
    object/bucket absent, access denied) -- so we fail fast instead of backing
    off pointlessly (also keeps hosts without S3 configured fast)."""
    name = type(e).__name__
    if name in ("NoCredentialsError", "PartialCredentialsError",
                "EndpointConnectionError", "NoSuchKey", "NoSuchBucket"):
        return True
    try:  # botocore ClientError carries an HTTP/AWS error code
        code = str(e.response["Error"]["Code"])  # type: ignore[attr-defined]
        if code in ("403", "404", "AccessDenied", "NoSuchKey", "NoSuchBucket"):
            return True
    except Exception:
        pass
    return False


def _download_atomic(bucket: str, key: str, dest: str, name: str) -> bool:
    """Download s3://bucket/key to `dest` atomically.  A partial/failed download
    NEVER becomes the destination file (temp file is verified non-empty, then
    os.replace publishes it in one atomic step)."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    for attempt in range(1, _RETRIES + 1):
        tmp = None
        try:
            log.info("[MODEL] downloading %s from s3://%s/%s (attempt %d/%d)",
                     name, bucket, key, attempt, _RETRIES)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".",
                                       prefix=f".{name}.", suffix=".part")
            os.close(fd)
            _s3().download_file(bucket, key, tmp)
            if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                raise IOError("downloaded file is empty")
            os.replace(tmp, dest)                       # atomic publish
            tmp = None
            size = os.path.getsize(dest)
            log.info("[MODEL] download complete: %s (%d bytes)", name, size)
            if os.path.isfile(dest) and size > 0:
                log.info("[MODEL] verified %s", name)
                return True
            raise IOError("post-move verification failed")
        except Exception as e:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if _is_permanent(e):
                log.warning("[MODEL] %s: %s (%s) -- not retryable, giving up",
                            name, e, type(e).__name__)
                return False
            log.warning("[MODEL] %s attempt %d/%d failed: %s",
                        name, attempt, _RETRIES, e)
            if attempt < _RETRIES:
                time.sleep(_BACKOFF_SEC * attempt)
    return False


# -----------------------------------------------------------------------------
# Convenience bundles for stage boundaries that resolve models by NAME
# -----------------------------------------------------------------------------

# Extraction classifiers (side_classification.pt for the side cameras,
# top_classification.pt for the top cameras).  Synced from the orchestrator so
# the standalone train_extraction package stays free of any core import.
_EXTRACTION_MODEL_FILES = ("side_classification.pt", "top_classification.pt")

# Reconstruction models the wagon_count subprocess resolves by name (it accepts
# the short S3 names directly, and maps the *_wagon_gap.pt aliases to them).
_RECON_MODEL_FILES = ("right_up_gap.pt", "left_up_gap.pt",
                      "top_gap.pt", "side_classification.pt")


def extraction_models_dir() -> str:
    """Resolve the extraction classify-model dir EXACTLY as train_extraction does
    (WAGONEYE_EXTRACTION_MODELS_DIR, else <repo>/models/extraction)."""
    return os.environ.get(
        "WAGONEYE_EXTRACTION_MODELS_DIR",
        os.path.join(CFG.PROJECT_ROOT, "models", "extraction"))


def ensure_extraction_models() -> Dict[str, bool]:
    d = extraction_models_dir()
    return ensure_many(os.path.join(d, f) for f in _EXTRACTION_MODEL_FILES)


def ensure_reconstruction_models(models_dir: str) -> Dict[str, bool]:
    return ensure_many(os.path.join(models_dir, f) for f in _RECON_MODEL_FILES)
