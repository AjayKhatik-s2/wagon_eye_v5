"""Per-(camera, feature) completion markers for the incremental lifecycle.

A marker records the full identity of the inputs a camera-feature result was
produced from, so feature work is skipped ONLY when every relevant identity
matches.  Any change to the source ETag, the sealed GlobalTrainState version,
the model file (SHA-256), the processor schema version, or the feature
thresholds invalidates that one camera-feature marker (and only that one).

Marker path:  wagon_states/.features/<CAMERA>/<feature>.json
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core import camera_features as CF

_MARKER_ROOT = ".features"
MARKER_SCHEMA_VERSION = 1

# Cache model hashes by (abspath, mtime_ns, size) so we don't re-hash a large,
# unchanged .pt on every poll tick.
_HASH_CACHE: Dict[tuple, str] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    """SHA-256 of a file, cached on (path, mtime, size).  'MISSING' if absent."""
    if not path or not os.path.isfile(path):
        return "MISSING"
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _HASH_CACHE[key] = digest
    return digest


def feature_config_hash(feature: str, enabled: bool) -> str:
    payload = {
        "feature": feature,
        "enabled": bool(enabled),
        "schema": CF.FEATURE_SCHEMA_VERSION.get(feature),
        "thresholds": CF.FEATURE_THRESHOLDS.get(feature, {}),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compute_identity(
    *, camera_id: str, feature: str, source_key: Optional[str], etag: Optional[str],
    global_state_version: str, feat_models_dir: str, enabled: bool = True,
) -> Dict[str, Any]:
    model_filename = CF.FEATURE_MODEL_FILENAME.get(feature, "")
    model_path = os.path.join(feat_models_dir, model_filename) if model_filename else ""
    return {
        "camera_id": camera_id,
        "feature": feature,
        "source_key": source_key,
        "etag": etag,
        "global_state_version": global_state_version,
        "model_filename": model_filename,
        "model_sha256": sha256_file(model_path),
        "processor_schema_version": CF.FEATURE_SCHEMA_VERSION.get(feature),
        "feature_config_hash": feature_config_hash(feature, enabled),
    }


# The identity fields that must all match for a skip (source_key excluded --
# ETag is the authoritative version signal; a moved-but-identical object should
# still skip).
_IDENTITY_KEYS = (
    "camera_id", "feature", "etag", "global_state_version",
    "model_sha256", "processor_schema_version", "feature_config_hash",
)


def _marker_path(states_root: str, camera_id: str, feature: str) -> str:
    return os.path.join(states_root, _MARKER_ROOT, camera_id, f"{feature}.json")


def read_marker(states_root: str, camera_id: str, feature: str) -> Optional[Dict[str, Any]]:
    p = _marker_path(states_root, camera_id, feature)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def is_up_to_date(states_root: str, identity: Dict[str, Any]) -> bool:
    """True iff a matching OK marker exists for this identity."""
    marker = read_marker(states_root, identity["camera_id"], identity["feature"])
    if not marker or marker.get("status") != "OK":
        return False
    return all(marker.get(k) == identity.get(k) for k in _IDENTITY_KEYS)


def write_marker(states_root: str, identity: Dict[str, Any], *,
                 status: str, wagons_completed: int) -> str:
    d = os.path.join(states_root, _MARKER_ROOT, identity["camera_id"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{identity['feature']}.json")
    payload = dict(identity)
    payload.update({
        "marker_schema_version": MARKER_SCHEMA_VERSION,
        "status": status,
        "wagons_completed": wagons_completed,
        "completed_at": _now_iso(),
    })
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{identity['feature']}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path
