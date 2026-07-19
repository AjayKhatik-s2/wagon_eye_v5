"""Stage 6 -- upload `batch_outputs/<key>/` to S3.

Strategy (end-results bucket, 2026 storage architecture):
    * PDF/JSON reports go under  reports/<batch_key>/<file>  (microservice first
      for the PDF, S3 direct fallback).
    * The batch tree (global_state + wagon_states + evidence + processed_videos +
      metadata) is recursively uploaded under  archive/<batch_key>/<sub>/...
    * Dashboard payloads go under  dashboard/...  (see delivery/dashboard_ingest).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.s3")


# -----------------------------------------------------------------------------
# Content-type per extension (very small mapping)
# -----------------------------------------------------------------------------

_CONTENT_TYPES = {
    ".pdf":  "application/pdf",
    ".json": "application/json",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".mp4":  "video/mp4",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
}


def _content_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


# -----------------------------------------------------------------------------
# Microservice PDF upload (proven helper preserved from the legacy
# master_runner; same API and product name).
# -----------------------------------------------------------------------------

def _upload_pdf_microservice(pdf_path: str) -> Optional[str]:
    import requests
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d-%m-%Y")
    for attempt in range(1, 4):
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
                data  = {"product_name": C.PRODUCT_NAME, "folder_name": today}
                resp = requests.post(C.UPLOAD_API_URL, data=data, files=files,
                                     timeout=120)
            if resp.status_code == 200:
                url = resp.json().get("url")
                if url:
                    log.info("[DELIVERY] PDF microservice URL: %s", url)
                    return url
        except Exception as e:
            log.warning("[DELIVERY] PDF microservice attempt %d/3 failed: %s",
                        attempt, e)
        time.sleep(10)
    return None


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def upload_pdf(s3_client, pdf_path: str, batch_key: str) -> Optional[str]:
    """Microservice first; S3 direct fallback."""
    if not os.path.exists(pdf_path):
        return None
    url = _upload_pdf_microservice(pdf_path)
    if url:
        return url
    bucket = C.S3_OUTPUT_BUCKET
    key = f"{C.S3_REPORTS_PREFIX}/{batch_key}/{os.path.basename(pdf_path)}"
    try:
        s3_client.upload_file(
            pdf_path, bucket, key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
        return f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"
    except Exception as e:
        log.error("[DELIVERY] S3 PDF fallback failed: %s", e)
        return None


def upload_json(s3_client, json_path: str, batch_key: str) -> Optional[str]:
    if not os.path.exists(json_path):
        return None
    bucket = C.S3_OUTPUT_BUCKET
    key = f"{C.S3_REPORTS_PREFIX}/{batch_key}/{os.path.basename(json_path)}"
    try:
        s3_client.upload_file(
            json_path, bucket, key,
            ExtraArgs={"ContentType": "application/json"},
        )
        url = f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"
        log.info("[DELIVERY] JSON URL: %s", url)
        return url
    except Exception as e:
        log.error("[DELIVERY] JSON upload failed: %s", e)
        return None


def upload_tree(
    s3_client, local_dir: str, batch_key: str,
    *, sub_prefix: str = "",
    skip_extensions: Optional[set] = None,
) -> int:
    """Recursively upload everything under `local_dir` to the ARCHIVE prefix
    s3://<output_bucket>/<archive_prefix>/<batch_key>/<sub_prefix>/...
    (the complete processed-batch tree: global_state, wagon_states, evidence,
    processed_videos, metadata).

    Returns the number of files uploaded.
    """
    if not os.path.isdir(local_dir):
        return 0
    bucket = C.S3_OUTPUT_BUCKET
    base   = f"{C.S3_ARCHIVE_PREFIX}/{batch_key}"
    if sub_prefix:
        base = f"{base}/{sub_prefix.strip('/')}"
    skip = skip_extensions or set()

    count = 0
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if any(fn.lower().endswith(ext) for ext in skip):
                continue
            local = os.path.join(root, fn)
            rel   = os.path.relpath(local, local_dir).replace(os.sep, "/")
            key   = f"{base}/{rel}"
            try:
                s3_client.upload_file(
                    local, bucket, key,
                    ExtraArgs={"ContentType": _content_type_for(fn)},
                )
                count += 1
            except Exception as e:
                log.warning("[DELIVERY] upload failed %s -> s3://%s/%s: %s",
                            local, bucket, key, e)
    return count
