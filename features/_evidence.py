"""Evidence persistence helpers used by all four feature processors.

Output layout (one per batch):

    evidence/
        GW_1/
            door/
                left_best.jpg          full frame, LEFT_UP camera
                left_crop.jpg          door bbox crop (LEFT_UP)
                right_best.jpg
                right_crop.jpg
                metadata.json
            ocr/
                best_frame.jpg
                number_crop.jpg
                metadata.json
            damage/
                track_1.jpg            damage track snapshot
                track_1_crop.jpg
                track_2.jpg
                track_2_crop.jpg
                metadata.json
            load/
                best_frame.jpg
                metadata.json
        GW_2/
            ...

The point of saving evidence in a per-wagon folder (rather than per-feature)
is so the rich combined PDF can pull all 4 camera snapshots for one wagon
from a single directory.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def wagon_evidence_dir(evidence_root: str, gw_id: str, feature: str) -> str:
    """Return (and create) `evidence/<gw_id>/<feature>/` (legacy flat layout)."""
    p = os.path.join(evidence_root, gw_id, feature)
    os.makedirs(p, exist_ok=True)
    return p


def camera_evidence_dir(evidence_root: str, gw_id: str, feature: str,
                        camera_id: str) -> str:
    """Return (and create) `evidence/<gw_id>/<feature>/<CAMERA>/`.

    Camera-scoped so one camera's evidence is never removed/overwritten by
    another camera processing the same wagon+feature."""
    p = os.path.join(evidence_root, gw_id, feature, camera_id)
    os.makedirs(p, exist_ok=True)
    return p


@contextmanager
def atomic_camera_evidence(evidence_root: str, gw_id: str, feature: str,
                           camera_id: str):
    """Build one camera's evidence into a temp dir, then atomically swap it into
    `evidence/<gw>/<feature>/<CAMERA>/`.

    A failed build (exception in the block) discards the temp dir and leaves the
    previous camera evidence intact.  Only THIS camera's directory is swapped --
    sibling cameras under the same feature are never touched.  Yields the temp
    directory path the caller writes into."""
    final = os.path.join(evidence_root, gw_id, feature, camera_id)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final + ".tmp_build"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        yield tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # swap: move the old aside first so we never delete it before the new lands
    backup = final + ".old"
    if os.path.isdir(backup):
        shutil.rmtree(backup, ignore_errors=True)
    if os.path.isdir(final):
        os.replace(final, backup)
    os.replace(tmp, final)
    if os.path.isdir(backup):
        shutil.rmtree(backup, ignore_errors=True)


# -----------------------------------------------------------------------------
# I/O primitives
# -----------------------------------------------------------------------------

def save_jpeg(path: str, frame: np.ndarray, quality: int = 92) -> bool:
    """Write a BGR frame to JPEG.  Returns True on success."""
    if frame is None or frame.size == 0:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return bool(cv2.imwrite(
        path, frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    ))


def safe_crop(
    frame: np.ndarray,
    bbox: List[float],
    *, pad: int = 8,
) -> Optional[np.ndarray]:
    """Crop with bounds clipping + optional padding.  None if degenerate."""
    if frame is None or bbox is None or len(bbox) != 4:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def draw_annotated_bbox(
    frame: np.ndarray,
    bbox: List[float],
    *,
    label: str,
    color: tuple = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a labelled rectangle on a copy of the frame."""
    if frame is None or bbox is None or len(bbox) != 4:
        return frame
    out = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            out, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    return out


def write_metadata(path: str, payload: Dict[str, Any]) -> None:
    """Atomically write a JSON metadata file (temp + os.replace)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".meta.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Frame lookup (read a specific cached JPEG by frame index)
# -----------------------------------------------------------------------------

def read_cached_frame(
    cache_root: str,
    gw_id: str,
    camera_id_lower: str,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Read one frame from wagon_cache/<gw>/<cam>/frame_NNNNNN.jpg."""
    p = os.path.join(cache_root, gw_id, camera_id_lower,
                     f"frame_{int(frame_idx):06d}.jpg")
    if not os.path.exists(p):
        return None
    return cv2.imread(p)


# -----------------------------------------------------------------------------
# Best-frame tracker (used inside each feature processor)
# -----------------------------------------------------------------------------

class BestFrameTracker:
    """Keep the single best (frame, bbox, score) seen so far.

    Score is the caller's choice -- typically `confidence` for damage and
    OCR, and a custom priority for doors (where DAMAGE > OPEN > PARTIAL >
    CLOSED).
    """

    def __init__(self) -> None:
        self.score: float = -1.0
        self.frame: Optional[np.ndarray] = None
        self.bbox: Optional[List[float]] = None
        self.frame_idx: int = -1
        self.meta: Dict[str, Any] = {}

    def update(
        self,
        *,
        score: float,
        frame: Optional[np.ndarray],
        bbox: Optional[List[float]] = None,
        frame_idx: int = -1,
        **meta: Any,
    ) -> bool:
        """Returns True if this is now the new best."""
        if frame is None:
            return False
        if score <= self.score:
            return False
        # store a COPY so subsequent mutation of `frame` doesn't change us
        self.frame = frame.copy()
        self.bbox  = list(bbox) if bbox is not None else None
        self.score = float(score)
        self.frame_idx = int(frame_idx)
        self.meta = dict(meta)
        return True

    def has_data(self) -> bool:
        return self.frame is not None
