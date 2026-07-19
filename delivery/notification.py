"""Stage 6 -- ONE email per batch."""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.email")


def send_email(
    *,
    batch_key: str,
    report_pdf_url: Optional[str],
    report_json_url: Optional[str],
    summary: Dict[str, Any],
    cameras_present: List[str],
    cameras_missing: List[str],
    final_status: str,
    idempotency_key: Optional[str] = None,
) -> bool:
    """POST to the notification microservice.  Returns True on HTTP 200.

    `idempotency_key` (derived from batch_key + report_revision + final report
    hash) is sent both as a payload field and an `Idempotency-Key` header so a
    notification service that honours either can de-duplicate a resend."""
    import requests
    if not report_pdf_url:
        log.warning("[DELIVERY] no PDF URL -- skipping email")
        return False

    ist = timezone(timedelta(hours=5, minutes=30))
    date_time_str = datetime.now(ist).strftime("%d-%m-%Y - %H:%M")

    _locos = summary.get("loco_numbers") or []
    _loco_str = ",".join(str(x) for x in _locos) if _locos else "—"
    subject = (
        f"WagonEye Combined Report | v4 | "
        f"{batch_key} | wagons={summary.get('total_wagons', 0)} | "
        f"loco={_loco_str} | {date_time_str}"
    )
    context = {
        "report_date":   date_time_str,
        "report_url":    report_pdf_url,
        "json_url":      report_json_url or "",
        "generated_by":  "WagonEye v4 (train-state-native)",
        "status":        final_status,
        "cameras_present": ", ".join(cameras_present) or "—",
        "cameras_missing": ", ".join(cameras_missing) or "—",
        "total_wagons":    summary.get("total_wagons", 0),
        "loaded_wagons":   summary.get("loaded", 0),
        "empty_wagons":    summary.get("empty", 0),
        "open_doors":      summary.get("left_doors_open", 0)
                           + summary.get("right_doors_open", 0),
        "top_damaged":     summary.get("top_damaged", 0),
        "mode": "TRAIN-STATE-NATIVE",
    }
    payload = {
        "to": C.EMAIL_RECEIVER,
        "cc": C.EMAIL_RECEIVER_CC,
        "subject": subject,
        "context": context,
        "attachment_url": report_pdf_url,
        "mail_from_name": "WagonEye v4",
        "template_name": "rake_inspection_report_v1.txt",
    }
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None

    for attempt in range(1, 4):
        try:
            resp = requests.post(C.EMAIL_API_URL, json=payload, headers=headers,
                                 timeout=60)
            if resp.status_code == 200:
                log.info("[DELIVERY] email sent (%s)", final_status)
                return True
            log.warning("[DELIVERY] email attempt %d/3 -> HTTP %d",
                        attempt, resp.status_code)
        except Exception as e:
            log.warning("[DELIVERY] email attempt %d/3 failed: %s", attempt, e)
        time.sleep(15 * attempt)
    return False
