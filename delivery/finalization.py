"""Idempotent finalization marker for Stage 6 delivery.

A batch's finalization (upload + one email) must survive a restart WITHOUT
duplicate uploads or duplicate email.  This module persists a per-batch marker

    <batch_root>/delivery/finalization.json

recording exactly what was delivered.  On re-entry, finalization consults the
marker and skips any step already completed for the current report revision.

Exactly-once email is best-effort across a crash in the narrow window between
the notification API returning success and this marker being persisted: we
persist `email_sent=true` ONLY after a confirmed 200, and log that a crash in
that window may cause one resend.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

FINALIZATION_SCHEMA_VERSION = 1
_SUBDIR = "delivery"
_BASENAME = "finalization.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def marker_path(batch_root: str) -> str:
    return os.path.join(batch_root, _SUBDIR, _BASENAME)


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def email_idempotency_key(batch_key: str, report_revision: int,
                          final_report_hash: Optional[str]) -> str:
    raw = f"{batch_key}|{report_revision}|{final_report_hash or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load(batch_root: str) -> Optional[Dict[str, Any]]:
    p = marker_path(batch_root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write(batch_root: str, payload: Dict[str, Any]) -> None:
    """Atomically persist the finalization marker."""
    d = os.path.join(batch_root, _SUBDIR)
    os.makedirs(d, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("finalization_schema_version", FINALIZATION_SCHEMA_VERSION)
    payload.setdefault("finalized_at", _now_iso())
    path = os.path.join(d, _BASENAME)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".finalization.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
