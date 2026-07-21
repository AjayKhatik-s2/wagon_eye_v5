"""Continuous train-extraction producer service.

Polls each camera's RAW CCTV bucket, cuts out the train pass with the vendored
V4 extractor, and uploads the trimmed clip(s) to that camera's TRIMMED bucket.
It does NO inspection -- point the trimmed buckets at the prefixes that
`wagon_eye_v4_new`'s `--auto` orchestrator polls and the two run independently.

    python -m train_extraction.run_extraction_service                 # all 4 cameras
    python -m train_extraction.run_extraction_service --camera RIGHT_UP
    python -m train_extraction.run_extraction_service --once          # one sweep, exit
    python -m train_extraction.run_extraction_service --dry-run       # list only, no extract

Processed raw keys are remembered in a small local JSON ledger under
WAGONEYE_EXTRACTION_STATE_DIR (default <root>/logs/extraction_state) so a
restart never re-extracts an already-handled raw clip.  (The extractor's own
S3 state store additionally preserves cross-clip ongoing-train continuity.)

Graceful shutdown: SIGTERM/SIGINT finish the current key, then exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Set

from . import driver as D

log = logging.getLogger("extraction.service")

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts")

_STOP = False


def _handle_stop(signum, _frame):
    global _STOP
    _STOP = True
    log.info("signal %s received -- finishing current key then exiting", signum)


# -----------------------------------------------------------------------------
# local processed-key ledger (per camera)
# -----------------------------------------------------------------------------

def _state_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("WAGONEYE_EXTRACTION_STATE_DIR",
                       os.path.join(root, "logs", "extraction_state"))
    os.makedirs(d, exist_ok=True)
    return d


def _ledger_path(camera: str) -> str:
    return os.path.join(_state_dir(), f"processed_{camera.lower()}.json")


def _load_ledger(camera: str) -> Set[str]:
    p = _ledger_path(camera)
    if not os.path.isfile(p):
        return set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return set(json.load(f).get("processed", []))
    except (OSError, json.JSONDecodeError):
        return set()


def _save_ledger(camera: str, processed: Set[str]) -> None:
    p = _ledger_path(camera)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed)}, f, indent=2)
    os.replace(tmp, p)


# -----------------------------------------------------------------------------
# one sweep of one camera
# -----------------------------------------------------------------------------

def _list_raw_keys(ex, raw_bucket: str) -> List[str]:
    """Sorted video keys under the camera's raw bucket/prefix.

    `raw_bucket` is a "<bucket>/<camera_folder>" string; we pass the embedded
    camera folder as the S3 list Prefix so each camera lists ONLY its own raw
    folder.  Without this the list would scan the whole raw bucket and every
    camera would (mis)process every other camera's raw clips.
    """
    from .url_utils import split_bucket_prefix
    _bucket, prefix = split_bucket_prefix(raw_bucket)
    objs = ex.s3.list_objects(raw_bucket, prefix=prefix)
    keys = [o["Key"] for o in objs
            if str(o.get("Key", "")).lower().endswith(_VIDEO_EXTS)]
    return sorted(keys)


def sweep_camera(camera: str, *, dry_run: bool = False) -> Dict[str, int]:
    """Extract every not-yet-processed raw clip for one camera."""
    result = {"listed": 0, "new": 0, "trains": 0, "errors": 0}
    try:
        ex = D.get_extractor(camera)
    except FileNotFoundError as e:
        log.error("[%s] cannot start (missing model): %s", camera, e)
        result["errors"] += 1
        return result
    raw_bucket = D.raw_bucket_for(camera)

    processed = _load_ledger(camera)
    keys = _list_raw_keys(ex, raw_bucket)
    result["listed"] = len(keys)

    for key in keys:
        if _STOP:
            break
        if key in processed:
            continue
        result["new"] += 1
        if dry_run:
            log.info("[%s] DRY-RUN would extract: %s", camera, key)
            continue
        try:
            trains = D.extract_trains(camera, key)
            result["trains"] += len(trains)
            for t in trains:
                log.info("[%s] trimmed -> %s", camera,
                         getattr(t, "trimmed_video_url", "?"))
            # mark processed only after a successful extract call
            processed.add(key)
            _save_ledger(camera, processed)
        except Exception as e:
            result["errors"] += 1
            log.error("[%s] extract failed for %s: %s", camera, key, e,
                      exc_info=True)
    return result


# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="train_extraction.run_extraction_service",
        description="Continuous train-extraction producer (raw -> trimmed).")
    p.add_argument("--camera", action="append", choices=list(D.ALL_CAMERAS),
                   help="limit to one or more cameras (default: all four)")
    p.add_argument("--once", action="store_true",
                   help="run a single sweep of all selected cameras, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="list what WOULD be extracted; upload/extract nothing")
    p.add_argument("--poll-interval", type=int,
                   default=int(os.environ.get("WAGONEYE_EXTRACTION_POLL_INTERVAL", "60")),
                   help="seconds between sweeps in continuous mode (default 60)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("WAGONEYE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cameras = tuple(args.camera) if args.camera else D.ALL_CAMERAS

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    log.info("train-extraction producer starting: cameras=%s once=%s dry_run=%s "
             "poll=%ss", cameras, args.once, args.dry_run, args.poll_interval)

    while not _STOP:
        for camera in cameras:
            if _STOP:
                break
            r = sweep_camera(camera, dry_run=args.dry_run)
            log.info("[%s] sweep: listed=%d new=%d trains=%d errors=%d",
                     camera, r["listed"], r["new"], r["trains"], r["errors"])
        if args.once or _STOP:
            break
        # interruptible sleep between sweeps
        for _ in range(max(1, args.poll_interval)):
            if _STOP:
                break
            time.sleep(1)

    log.info("train-extraction producer exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
