"""IST/UTC time helpers."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist_str() -> str:
    return now_ist().strftime("%Y-%m-%d")


def today_utc_iso() -> str:
    return now_utc().date().isoformat()


def utc_to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Parse IST timestamp from a video filename matching ..._YYYYMMDD_HHMMSS[_suffix].

    The numbers in the filename are already IST — we attach the tzinfo label without shifting.
    Returns None if no match.
    """
    basename = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    match = re.search(r"_(\d{8})_(\d{6})(?:_|$)", basename)
    if not match:
        return None
    date_str, time_str = match.group(1), match.group(2)
    try:
        parsed = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        return parsed.replace(tzinfo=IST)
    except ValueError:
        return None
