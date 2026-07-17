"""Resolve model paths — local files pass through, ``s3://...`` URIs are downloaded
to a local cache directory on first use (and re-downloaded only when their S3
ETag changes).

Design goals:

* **Backward compatible** — any existing local-path YAML keeps working.
* **Atomic** — downloads go to ``<path>.part`` and are renamed only after the
  full file is written, so a half-downloaded model can never be loaded.
* **Etag-aware** — subsequent pipeline restarts use the cached copy and skip
  the download unless the object in S3 has changed.
* **Graceful** — if S3 is briefly unreachable but a cached copy exists, the
  cached copy is used and a warning is logged.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "train-inspection-engine", "models"
)


_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def is_remote_uri(path_or_uri: str) -> bool:
    """Return True if ``path_or_uri`` looks like an ``s3://...`` URI."""
    return bool(path_or_uri) and path_or_uri.startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI_RE.match(uri)
    if not match:
        raise ValueError(f"Invalid s3 URI: {uri!r}")
    return match.group(1), match.group(2)


def resolve_path(
    path_or_uri: str,
    *,
    s3_client=None,
    cache_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Resolve ``path_or_uri`` to a local filesystem path.

    * **Local path** — returned unchanged after an ``os.path.exists`` check.
    * **``s3://bucket/key``** — downloaded to ``<cache_dir>/<bucket>/<key>``
      and returned. Re-uses the cached file when its ETag matches the
      current S3 object's ETag.

    Parameters
    ----------
    path_or_uri:
        Either a local filesystem path or an ``s3://bucket/key`` URI.
    s3_client:
        An :class:`~train_inspection_engine.core.s3.S3Client` instance.
        Required only when ``path_or_uri`` is an S3 URI.
    cache_dir:
        Local directory for caching downloaded files. Falls back to
        :data:`DEFAULT_CACHE_DIR` if not provided.
    logger:
        Optional logger for download progress messages.

    Raises
    ------
    FileNotFoundError
        If the local path doesn't exist, or if a remote object is missing
        and no cached copy is available.
    ValueError
        If a malformed ``s3://`` URI is supplied.
    """
    log = logger or logging.getLogger(__name__)

    if not path_or_uri:
        raise FileNotFoundError("Empty model path")

    if not is_remote_uri(path_or_uri):
        if not os.path.exists(path_or_uri):
            raise FileNotFoundError(f"Model not found: {path_or_uri!r}")
        return path_or_uri

    if s3_client is None:
        raise RuntimeError(
            f"Cannot resolve {path_or_uri!r}: no S3 client supplied."
        )

    bucket, key = _parse_s3_uri(path_or_uri)
    base = cache_dir or DEFAULT_CACHE_DIR
    local_path = os.path.join(base, bucket, key)
    etag_path = local_path + ".etag"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    remote_etag = None
    try:
        head = s3_client.client.head_object(Bucket=bucket, Key=key)
        remote_etag = head["ETag"].strip('"')
    except Exception as e:  # noqa: BLE001 — graceful fallback
        if os.path.exists(local_path):
            log.warning(
                "Could not reach S3 for %s (%s) — using cached copy at %s",
                path_or_uri, e, local_path,
            )
            return local_path
        raise FileNotFoundError(
            f"S3 object unreachable and no local cache: {path_or_uri!r} ({e})"
        ) from e

    cached_etag = None
    if os.path.exists(local_path) and os.path.exists(etag_path):
        try:
            with open(etag_path, "r", encoding="utf-8") as f:
                cached_etag = f.read().strip()
        except OSError:
            cached_etag = None

    if cached_etag == remote_etag and os.path.exists(local_path):
        log.info(
            "Model up-to-date in cache: %s (etag=%s)", local_path, remote_etag,
        )
        return local_path

    if cached_etag is not None:
        log.info(
            "Model %s changed in S3 (etag %s → %s) — refreshing cache",
            path_or_uri, cached_etag, remote_etag,
        )
    else:
        log.info("Downloading model %s → %s", path_or_uri, local_path)

    part_path = local_path + ".part"
    try:
        s3_client.client.download_file(bucket, key, part_path)
        os.replace(part_path, local_path)
        with open(etag_path, "w", encoding="utf-8") as f:
            f.write(remote_etag)
    finally:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log.info(
        "Cached model %s (%.1f MB, etag=%s)", local_path, size_mb, remote_etag,
    )
    return local_path
