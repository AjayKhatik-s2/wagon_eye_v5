"""BatchManifest -- the persisted, resumable source of truth for one train
batch's incremental lifecycle.

Design guarantees:
  * Schema-versioned.  An unknown / newer `manifest_schema_version` is REFUSED
    (fails safe): the loader returns None and logs, so a future format is never
    misread as the current one.
  * Atomic local writes (temp file + os.replace) so a crash mid-write never
    corrupts the manifest.
  * Concurrency-safe S3 strategy: each batch's manifest is its OWN S3 object
    (`train_batch/<key>/manifest.json`) -- a single writer per batch.  There is
    NO shared read-modify-write index; active batches are discovered by LISTING
    the `train_batch/` prefix and skipping terminal manifests.

Only terminal LifecycleStates get mirrored into
`master_runner/processed_batches.json` (handled by train_batch_manager); the
manifest carries every non-terminal batch across polls and restarts.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core import config as CFG
from core import constants as C
from core.lifecycle import ArrivalState, LifecycleState
from core.logging_setup import get_logger

log = get_logger("manifest")

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_BASENAME = "manifest.json"


class ManifestSchemaError(RuntimeError):
    """Raised internally when a manifest carries an unsupported schema version."""


# -----------------------------------------------------------------------------
# time helpers (UTC ISO-8601, stable + sortable)
# -----------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# CameraSlot
# -----------------------------------------------------------------------------

@dataclass
class CameraSlot:
    camera_id: str
    bucket: str = ""
    s3_key: str = ""
    etag: Optional[str] = None
    filename: str = ""
    s3_url: str = ""
    local_path: Optional[str] = None
    file_size: int = 0
    last_modified: Optional[str] = None      # ISO
    arrived_at: Optional[str] = None         # ISO
    arrival_state: str = ArrivalState.PENDING_CAMERA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraSlot":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# BatchManifest
# -----------------------------------------------------------------------------

@dataclass
class BatchManifest:
    batch_key: str
    canonical_train_timestamp: str
    match_window_sec: int = 120
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION
    first_seen_at: Optional[str] = None                       # ISO

    cameras: Dict[str, CameraSlot] = field(default_factory=dict)

    # Semantically separate deadlines (ISO). master/final are anchored at
    # first_seen_at; support_fusion is ARMED only when RIGHT_UP arrives.
    master_deadline: Optional[str] = None
    support_fusion_deadline: Optional[str] = None
    final_camera_deadline: Optional[str] = None

    lifecycle_status: str = LifecycleState.DISCOVERED

    # Sealed GlobalTrainState provenance
    global_state_status: str = "NONE"                         # NONE | SEALED
    global_state_version: Optional[str] = None                # sha256 of sealed JSON
    master_camera: Optional[str] = None
    fallback_master_used: bool = False
    reconstruction_mode: Optional[str] = None
    support_cameras_present: List[str] = field(default_factory=list)
    support_fusion_used: bool = False
    support_gap_recoveries: int = 0
    sealed_at: Optional[str] = None
    sealing_reason: Optional[str] = None

    # Downstream progress (mirrors on-disk markers; the markers remain the
    # authoritative skip/rebuild signal, this is a convenience view)
    materialized_cameras: List[str] = field(default_factory=list)
    completed_features: Dict[str, List[str]] = field(default_factory=dict)

    fusion_revision: int = 0
    report_revision: int = 0
    report_status: Optional[str] = None                       # INTERIM | FINAL | FINAL_PARTIAL
    terminal_status: Optional[str] = None                     # constants.BATCH_*

    videos_for_review: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ---- construction ----
    @classmethod
    def new(cls, *, batch_key: str, train_timestamp: str,
            match_window_sec: int = 120) -> "BatchManifest":
        now = iso(_now())
        m = cls(
            batch_key=batch_key,
            canonical_train_timestamp=train_timestamp,
            match_window_sec=match_window_sec,
            first_seen_at=now,
        )
        m._recompute_anchor_deadlines()
        return m

    # ---- camera arrival ----
    def has_master(self, master_camera: str = C.MASTER_CAMERA) -> bool:
        s = self.cameras.get(master_camera)
        return bool(s and s.arrival_state == ArrivalState.PRESENT)

    def present_cameras(self) -> List[str]:
        return [c for c in C.ALL_CAMERAS
                if c in self.cameras
                and self.cameras[c].arrival_state == ArrivalState.PRESENT]

    def missing_cameras(self) -> List[str]:
        return [c for c in C.ALL_CAMERAS if c not in self.present_cameras()]

    def is_complete(self) -> bool:
        return not self.missing_cameras()

    def set_camera(self, slot: CameraSlot) -> None:
        """Record an arrived camera.  Arms the support-fusion window the first
        time the master (RIGHT_UP) arrives (deadline is relative to arrival,
        NOT to first_seen)."""
        slot.arrival_state = ArrivalState.PRESENT
        if not slot.arrived_at:
            slot.arrived_at = iso(_now())
        self.cameras[slot.camera_id] = slot
        if slot.camera_id == C.MASTER_CAMERA and self.support_fusion_deadline is None:
            self.support_fusion_deadline = iso(
                _now() + timedelta(minutes=CFG.SUPPORT_FUSION_WAIT_MINUTES)
            )

    def mark_missing_final(self) -> None:
        for cam in C.ALL_CAMERAS:
            s = self.cameras.get(cam)
            if s is None:
                self.cameras[cam] = CameraSlot(
                    camera_id=cam, arrival_state=ArrivalState.CAMERA_MISSING_FINAL)
            elif s.arrival_state != ArrivalState.PRESENT:
                s.arrival_state = ArrivalState.CAMERA_MISSING_FINAL

    # ---- deadlines ----
    def _recompute_anchor_deadlines(self) -> None:
        base = parse_iso(self.first_seen_at) or _now()
        self.master_deadline = iso(base + timedelta(minutes=CFG.MASTER_WAIT_MINUTES))
        self.final_camera_deadline = iso(
            base + timedelta(minutes=CFG.FINAL_CAMERA_WAIT_MINUTES))

    def past(self, deadline_iso: Optional[str]) -> bool:
        dt = parse_iso(deadline_iso)
        return dt is not None and _now() >= dt

    def past_master_deadline(self) -> bool:
        return self.past(self.master_deadline)

    def past_support_window(self) -> bool:
        # Only meaningful once armed (RIGHT_UP arrived).
        return self.support_fusion_deadline is not None and self.past(self.support_fusion_deadline)

    def past_final_deadline(self) -> bool:
        return self.past(self.final_camera_deadline)

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cameras"] = {k: v.to_dict() if isinstance(v, CameraSlot) else v
                        for k, v in self.cameras.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BatchManifest":
        ver = int(d.get("manifest_schema_version", 0))
        if ver > MANIFEST_SCHEMA_VERSION:
            raise ManifestSchemaError(
                f"manifest schema v{ver} is newer than supported "
                f"v{MANIFEST_SCHEMA_VERSION}; refusing to interpret")
        cams_raw = d.get("cameras") or {}
        cams = {k: CameraSlot.from_dict(v) for k, v in cams_raw.items()}
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k != "cameras"}  # type: ignore[attr-defined]
        m = cls(**known)  # type: ignore[arg-type]
        m.cameras = cams
        # coerce older manifests up to current schema version
        m.manifest_schema_version = MANIFEST_SCHEMA_VERSION
        return m


# -----------------------------------------------------------------------------
# Local persistence (atomic)
# -----------------------------------------------------------------------------

def _manifest_local_path(batch_root: str) -> str:
    return os.path.join(batch_root, MANIFEST_BASENAME)


def write_local(manifest: BatchManifest, batch_root: str) -> None:
    """Atomically write the manifest to <batch_root>/manifest.json."""
    os.makedirs(batch_root, exist_ok=True)
    path = _manifest_local_path(batch_root)
    body = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    fd, tmp = tempfile.mkstemp(dir=batch_root, prefix=".manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_local(batch_root: str) -> Optional[BatchManifest]:
    path = _manifest_local_path(batch_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return BatchManifest.from_dict(data)
    except ManifestSchemaError as e:
        log.error("[MANIFEST] %s (%s) -- leaving batch untouched", path, e)
        return None
    except (OSError, json.JSONDecodeError) as e:
        log.error("[MANIFEST] could not read %s: %s", path, e)
        return None


# -----------------------------------------------------------------------------
# S3 mirror (idempotent; one object per batch = single writer per batch)
# -----------------------------------------------------------------------------

def manifest_s3_key(batch_key: str) -> str:
    prefix = (CFG.MANIFEST_S3_PREFIX or C.S3_TRAIN_BATCH_PREFIX).strip("/")
    return f"{prefix}/{batch_key}/{MANIFEST_BASENAME}"


def save_s3(s3_client, manifest: BatchManifest, bucket: Optional[str] = None) -> None:
    bucket = bucket or C.S3_OUTPUT_BUCKET
    key = manifest_s3_key(manifest.batch_key)
    try:
        body = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
        s3_client.put_object(Bucket=bucket, Key=key, Body=body,
                             ContentType="application/json")
    except Exception as e:  # non-fatal: local manifest remains authoritative
        log.error("[MANIFEST] failed to mirror %s to S3: %s", manifest.batch_key, e)


def load_s3(s3_client, batch_key: str, bucket: Optional[str] = None) -> Optional[BatchManifest]:
    bucket = bucket or C.S3_OUTPUT_BUCKET
    key = manifest_s3_key(batch_key)
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return BatchManifest.from_dict(data)
    except ManifestSchemaError as e:
        log.error("[MANIFEST] s3 %s: %s -- skipping", batch_key, e)
        return None
    except Exception as e:
        name = type(e).__name__
        if name not in ("NoSuchKey", "ClientError"):
            log.warning("[MANIFEST] could not read s3 manifest %s: %s", batch_key, e)
        return None


def list_active_manifests(
    s3_client,
    *,
    processed_batches: Dict[str, str],
    bucket: Optional[str] = None,
) -> List[BatchManifest]:
    """Discover non-terminal manifests by LISTING the train_batch/ prefix.

    Terminal batches (present in `processed_batches`) are skipped without a
    fetch.  This is the concurrency-safe alternative to a shared active-index
    file: each manifest object is single-writer, and the active set is
    reconciled by listing rather than mutating shared state.
    """
    from core.lifecycle import is_terminal

    bucket = bucket or C.S3_OUTPUT_BUCKET
    prefix = f"{C.S3_TRAIN_BATCH_PREFIX}/"
    out: List[BatchManifest] = []
    token = None
    seen_keys: set = set()
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        try:
            resp = s3_client.list_objects_v2(**kwargs)
        except Exception as e:
            log.error("[MANIFEST] list_objects_v2 failed (prefix=%s): %s", prefix, e)
            break
        for cp in resp.get("CommonPrefixes", []):
            p = cp.get("Prefix", "")            # train_batch/<key>/
            key = p[len(prefix):].strip("/")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            if key in processed_batches:
                continue                        # terminal -- skip
            m = load_s3(s3_client, key, bucket=bucket)
            if m is None:
                continue
            if is_terminal(m.lifecycle_status):
                continue
            out.append(m)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
            if not token:
                break
        else:
            break
    return out
