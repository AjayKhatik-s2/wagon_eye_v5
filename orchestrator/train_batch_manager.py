"""Stage-0 batch acquisition for continuous (`--auto`) mode.

This module is the S3-facing half of the orchestrator: it discovers source
videos in S3, clusters them into per-train batches by filename timestamp,
decides which batch is runnable, and persists a small processed-batches state
file so a restarted service never reprocesses a batch.

It owns NO detection / fusion / reporting logic -- once a `TrainBatch` is
handed back to `orchestrator.master_runner.process_batch`, the batch is
processed exactly as before.  Nothing here changes how a batch is analysed.

Call contract (consumed verbatim by master_runner.run_auto):

    poll_for_batches(s3_client, processed_batches, start_time, tolerance_sec)
        -> List[TrainBatch]
    select_runnable_batch(batches, partial_wait_minutes) -> Optional[TrainBatch]
    load_batch_state(s3_client, state_loc) -> Dict[str, str]
    save_batch_state(s3_client, state_loc, processed) -> None
    DEFAULT_BATCH_TOLERANCE_SEC : int

Configuration (all via core.constants, i.e. WAGONEYE_* env overrides):
    S3_INPUT_BUCKET        bucket holding the source videos
    S3_INPUT_PREFIXES      comma-separated key prefixes to scan (one per
                           camera rig, or a single shared prefix).  If empty,
                           polling finds nothing and --auto idles (safe
                           default: the operator must point it at real data).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core import constants as C
from core.batch import CameraVideo, TrainBatch, parse_train_timestamp
from core.logging_setup import get_logger

log = get_logger("batch_manager")

# Two source videos of the same train pass may carry filename timestamps that
# differ by a few seconds (each camera's trimmer stamps independently).  Videos
# whose timestamps fall within this window are clustered into one batch.
DEFAULT_BATCH_TOLERANCE_SEC = 120

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# Warn only once if the operator hasn't configured any input prefixes.
_WARNED_NO_PREFIXES = False


# -----------------------------------------------------------------------------
# processed_batches state file (JSON on S3)
# -----------------------------------------------------------------------------

def _split_state_loc(state_loc: str) -> Tuple[str, str]:
    """`"bucket/key/with/slashes.json"` -> ("bucket", "key/with/slashes.json")."""
    parts = state_loc.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid state_loc (expected 'bucket/key'): {state_loc!r}")
    return parts[0], parts[1]


def load_batch_state(s3_client, state_loc: str) -> Dict[str, str]:
    """Read the {batch_key -> final_status} map from S3.  Missing -> {}."""
    bucket, key = _split_state_loc(state_loc)
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        log.warning("[BATCH] state file %s is not a dict; starting empty", state_loc)
        return {}
    except Exception as e:
        # NoSuchKey on first run is normal; anything else we log and start fresh
        # rather than crash the service.
        name = type(e).__name__
        if name in ("NoSuchKey", "ClientError"):
            log.info("[BATCH] no existing state file at %s (%s) -- starting empty",
                     state_loc, name)
        else:
            log.warning("[BATCH] could not read state file %s: %s", state_loc, e)
        return {}


def save_batch_state(s3_client, state_loc: str, processed: Dict[str, str]) -> None:
    """Persist the {batch_key -> final_status} map back to S3 (best-effort)."""
    bucket, key = _split_state_loc(state_loc)
    try:
        body = json.dumps(processed, indent=2, sort_keys=True).encode("utf-8")
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=body,
            ContentType="application/json",
        )
    except Exception as e:
        # Non-fatal: the batch was still processed; we just failed to checkpoint.
        # Worst case the batch is re-evaluated next poll and skipped as already
        # present in the in-memory `processed` dict for this process lifetime.
        log.error("[BATCH] failed to persist state file %s: %s", state_loc, e)


# -----------------------------------------------------------------------------
# S3 listing + camera / timestamp resolution
# -----------------------------------------------------------------------------

def _camera_for_key(key: str) -> Optional[str]:
    """Match an S3 key's basename to a camera id by substring.

    Longest camera name first so RIGHT_UP_TOP wins over RIGHT_UP -- identical
    disambiguation to core.batch.scan_local_video_dir.
    """
    base = key.rsplit("/", 1)[-1].lower()
    if not base.endswith(_VIDEO_EXTS):
        return None
    for cam in sorted(C.ALL_CAMERAS, key=len, reverse=True):
        if cam.lower() in base:
            return cam
    return None


def _clean_etag(raw) -> Optional[str]:
    """S3 ETags come wrapped in quotes; normalize to a bare hex string."""
    if not raw:
        return None
    return str(raw).strip().strip('"')


def _list_input_objects(s3_client) -> List[Tuple[str, str, object, Optional[str]]]:
    """Return [(bucket, key, last_modified, etag), ...] for every video under
    the configured input prefixes.  Handles pagination."""
    global _WARNED_NO_PREFIXES
    prefixes = C.S3_INPUT_PREFIXES
    bucket = C.S3_INPUT_BUCKET
    if not prefixes:
        if not _WARNED_NO_PREFIXES:
            log.warning("[BATCH] WAGONEYE_S3_INPUT_PREFIXES is empty -- no source "
                        "videos will be discovered.  Set it to the S3 prefix(es) "
                        "holding the camera videos.")
            _WARNED_NO_PREFIXES = True
        return []

    out: List[Tuple[str, str, object, Optional[str]]] = []
    for prefix in prefixes:
        token = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix.strip("/")}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                resp = s3_client.list_objects_v2(**kwargs)
            except Exception as e:
                log.error("[BATCH] list_objects_v2 failed (bucket=%s prefix=%s): %s",
                          bucket, prefix, e)
                break
            for item in resp.get("Contents", []):
                key = item["Key"]
                if key.lower().endswith(_VIDEO_EXTS):
                    out.append((bucket, key, item.get("LastModified"),
                                _clean_etag(item.get("ETag"))))
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    break
            else:
                break
    return out


def list_candidate_videos(s3_client) -> List[CameraVideo]:
    """Classify every discoverable input video into a CameraVideo (camera id +
    train timestamp + ETag).  No clustering -- the manifest scheduler attaches
    each candidate to an active batch (or creates one).  Unclassifiable objects
    are dropped."""
    out: List[CameraVideo] = []
    for bucket, key, last_modified, etag in _list_input_objects(s3_client):
        cam = _camera_for_key(key)
        ts = parse_train_timestamp(key)
        if not cam or not ts:
            continue
        out.append(CameraVideo(
            camera_id=cam, bucket=bucket, s3_key=key,
            filename=key.rsplit("/", 1)[-1],
            s3_url=_https_url(bucket, key),
            train_timestamp=ts, last_modified=last_modified, etag=etag,
        ))
    # deterministic order: timestamp, camera, key
    out.sort(key=lambda cv: (cv.train_timestamp, cv.camera_id, cv.s3_key))
    return out


def _ts_to_dt(ts: str) -> Optional[datetime]:
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# Batch discovery
# -----------------------------------------------------------------------------

def poll_for_batches(
    *,
    s3_client,
    processed_batches: Dict[str, str],
    start_time: Optional[datetime] = None,
    tolerance_sec: int = DEFAULT_BATCH_TOLERANCE_SEC,
) -> List[TrainBatch]:
    """Discover candidate TrainBatches from the S3 input prefixes.

    Videos are grouped into batches by filename timestamp: each new video
    either joins an existing open cluster (if its timestamp is within
    `tolerance_sec` of that cluster's anchor and that camera slot is free) or
    starts a new cluster.  Batches already present in `processed_batches`
    (the persisted state map) are excluded.

    `start_time` is accepted for API compatibility and used only for logging;
    de-duplication is done via `processed_batches`, which lets `--batch <key>`
    replay an older batch and lets a restarted service resume cleanly.
    """
    objects = _list_input_objects(s3_client)
    if not objects:
        return []

    # Build (camera, timestamp) candidates, dropping anything we can't classify.
    candidates = []
    for bucket, key, last_modified, etag in objects:
        cam = _camera_for_key(key)
        ts = parse_train_timestamp(key)
        if not cam or not ts:
            continue
        dt = _ts_to_dt(ts)
        if dt is None:
            continue
        candidates.append((dt, ts, cam, bucket, key, last_modified, etag))

    # Deterministic order: by timestamp, then camera, then key.
    candidates.sort(key=lambda c: (c[0], c[2], c[4]))

    # Greedy temporal clustering.
    clusters: List[Dict] = []
    for dt, ts, cam, bucket, key, last_modified, etag in candidates:
        placed = False
        for cl in clusters:
            if cam in cl["videos"]:
                continue  # camera slot already filled for this cluster
            if abs((dt - cl["anchor"]).total_seconds()) <= tolerance_sec:
                cl["videos"][cam] = CameraVideo(
                    camera_id=cam, bucket=bucket, s3_key=key,
                    filename=key.rsplit("/", 1)[-1],
                    s3_url=_https_url(bucket, key),
                    train_timestamp=cl["batch_key"],
                    last_modified=last_modified,
                )
                placed = True
                break
        if not placed:
            clusters.append({
                "anchor": dt,
                "batch_key": ts,
                "videos": {cam: CameraVideo(
                    camera_id=cam, bucket=bucket, s3_key=key,
                    filename=key.rsplit("/", 1)[-1],
                    s3_url=_https_url(bucket, key),
                    train_timestamp=ts,
                    last_modified=last_modified,
                )},
            })

    batches: List[TrainBatch] = []
    for cl in clusters:
        if cl["batch_key"] in processed_batches:
            continue
        batches.append(TrainBatch(
            batch_key=cl["batch_key"],
            train_timestamp=cl["batch_key"],
            videos=cl["videos"],
        ))

    if batches:
        log.info("[BATCH] discovered %d unprocessed batch(es): %s",
                 len(batches), [b.batch_key for b in batches])
    return batches


def _https_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"


# -----------------------------------------------------------------------------
# Batch selection
# -----------------------------------------------------------------------------

def select_runnable_batch(
    batches: List[TrainBatch],
    partial_wait_minutes: float = 30.0,
) -> Optional[TrainBatch]:
    """Pick the batch to run next.

    Priority:
        1. The OLDEST complete batch (all 4 cameras present) -- run immediately.
        2. Otherwise the oldest PARTIAL batch, but only once it has aged past
           `partial_wait_minutes` (giving stragglers time to upload).  A
           younger partial batch is held back (returns None) so we don't run a
           3-camera batch that would have been complete 30 s later.
    """
    if not batches:
        return None

    # Oldest first == earliest train_timestamp.
    ordered = sorted(batches, key=lambda b: b.train_timestamp)

    complete = [b for b in ordered if b.is_complete()]
    if complete:
        return complete[0]

    wait_sec = partial_wait_minutes * 60.0
    for b in ordered:
        if b.age_seconds() >= wait_sec:
            log.info("[BATCH] %s partial (cameras=%s), aged %.0fs >= %.0fs wait "
                     "-- running partial", b.batch_key, b.present_cameras(),
                     b.age_seconds(), wait_sec)
            return b
    return None
