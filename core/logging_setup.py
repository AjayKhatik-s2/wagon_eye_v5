"""Centralized logging for the wagon_eye_v4 pipeline.

Before the EC2 migration the pipeline emitted only bare `print()` calls to
stdout -- fine for a notebook, useless for a long-running background service.
This module configures the standard `logging` framework once at each process
entry point:

    * a RotatingFileHandler writing timestamped, level-tagged lines to
      <LOG_DIR>/wagon_eye.log (rotated at 50 MB, 10 backups kept), and
    * a StreamHandler to stdout so foreground / interactive runs still show
      progress exactly as before.

Stage-boundary code (the orchestrator, reconstruction runner, materializer,
delivery) logs through `get_logger(...)`.  Deep inference modules keep their
existing `print()` calls -- those are still captured into the same log file
because a systemd service redirects the process's stdout/stderr into it, and
in foreground runs they interleave with the stream handler as they always did.

`setup_logging()` is idempotent: calling it twice (e.g. orchestrator + a
re-import) will not attach duplicate handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from core import config as CFG

_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 50 * 1024 * 1024   # 50 MB per file
_BACKUP_COUNT = 10

_CONFIGURED = False


def setup_logging(
    *,
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    log_filename: str = "wagon_eye.log",
    stream: bool = True,
) -> logging.Logger:
    """Configure the root logger once. Safe to call more than once.

    Args:
        log_dir: directory for the rotating log file (default CFG.LOG_DIR).
        level:   root level name, e.g. 'INFO' / 'DEBUG' (default CFG.LOG_LEVEL).
        log_filename: base filename inside `log_dir`.
        stream:  also mirror to stdout (True for foreground/service visibility).

    Returns:
        The configured root 'wagon_eye' logger namespace parent.
    """
    global _CONFIGURED

    log_dir = log_dir or CFG.LOG_DIR
    level_name = (level or CFG.LOG_LEVEL or "INFO").upper()
    level_value = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level_value)

    if _CONFIGURED:
        # Already set up in this process; just (re)assert the level.
        return logging.getLogger("wagon_eye")

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # File handler (best-effort: if the dir can't be created we still get stdout).
    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, log_filename),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(level_value)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[logging_setup] could not open log file in {log_dir}: {e}",
              file=sys.stderr)

    if stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level_value)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # Quiet a couple of chatty third-party loggers so the pipeline log stays
    # readable (unchanged inference behaviour; purely log-noise control).
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger("wagon_eye").info(
        "logging initialized: dir=%s level=%s", log_dir, level_name)
    return logging.getLogger("wagon_eye")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. get_logger('orchestrator')."""
    return logging.getLogger(f"wagon_eye.{name}")
