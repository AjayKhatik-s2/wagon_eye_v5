"""Evidence-frame quality scoring.

Faithful port of the legacy WagonEye snapshot heuristics that lived in
``old_system/RIGHT_UP/door_processor.py``:

    _compute_detection_quality   -> detection_quality()
    _is_edge_detection           -> is_edge_detection()
    _score_detection             -> snapshot_score()

These are the heuristics the legacy side/door pipeline actually used to pick
the report snapshot for each track (the multi-metric ``SnapshotSelector`` was
dead code in both the old and the new system and is deliberately NOT revived).

The scorer blends bbox area, horizontal centre proximity, model confidence and
a crop-quality term (brightness + Laplacian texture), then applies a hard
multiplicative penalty for boxes hugging a frame edge.  There is NO hard
blur/quality REJECTION gate -- legacy never had one; quality is a soft
down-weight only.

Imported directly by the feature processors (``from core.frame_quality import
...``); intentionally NOT re-exported from ``core/__init__`` so the
reportlab-only reporting layer never transitively imports cv2.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


# --- legacy constants (verbatim from door_processor.py) ----------------------

_EDGE_MARGIN_RATIO  = 0.08    # _is_edge_detection: within 8% of any edge
_AREA_OPTIMAL_RATIO = 0.28    # full-door area target (raised from legacy 0.15 so
                              # a larger door keeps scoring up to ~28% of frame)
_EDGE_PENALTY       = 0.3     # 70% score reduction for edge-hugging boxes

# _score_detection term weights.  ITEM 5: bias evidence selection toward the
# LARGEST visible (full) door -- area weight raised 2.0 -> 3.5 so a maximal-area
# door dominates the 2.5 centring weight, plus a small un-saturated tie-break so
# the physically largest door wins among equally-full candidates.  The edge
# penalty (0.3) and crop-quality term are UNCHANGED, so blurry / edge frames are
# still suppressed; there is still NO hard rejection gate.
_W_AREA          = 3.5
_W_CENTER        = 2.5
_W_CONF          = 1.0
_W_QUALITY       = 0.5
_W_AREA_TIEBREAK = 0.5

# ITEM 4: total growth applied to the chosen door box (7.5% per side, clipped to
# frame) so the persisted overlay box + evidence crop visually contain the WHOLE
# door.  Still a clean axis-aligned rectangle (legacy draw parity preserved).
_DOOR_BBOX_EXPAND_FRAC = 0.15

Bbox = Sequence[float]


def detection_quality(frame: np.ndarray, bbox: Bbox, *, pad: int = 5) -> float:
    """Crop brightness + Laplacian-texture quality in ``[0.1, 1.0]``.

    Faithful port of ``_compute_detection_quality`` (the quality scalar only;
    the legacy glare/reason flags are not needed downstream).  Brighter-than-200
    crops and low-texture (blurry / featureless) crops are penalised; the result
    is clamped to ``[0.1, 1.0]`` so a poor frame is down-weighted, never excluded.
    """
    if frame is None or bbox is None or len(bbox) != 4:
        return 1.0
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    brightness = float(np.mean(gray))
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_penalty = max(0.0, (brightness - 200.0) / 55.0)
    texture_penalty = max(0.0, (100.0 - texture) / 100.0)
    quality = 1.0 - 0.5 * brightness_penalty - 0.5 * texture_penalty
    return max(0.1, min(1.0, quality))


def is_edge_detection(
    bbox: Bbox, frame_w: int, frame_h: int,
    *, margin_ratio: float = _EDGE_MARGIN_RATIO,
) -> bool:
    """True if the bbox hugs any frame edge (door entering / leaving view)."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))
    mx = frame_w * margin_ratio
    my = frame_h * margin_ratio
    return (x1 < mx or x2 > frame_w - mx or y1 < my or y2 > frame_h - my)


def snapshot_score(
    bbox: Bbox, confidence: float, quality: float,
    frame_w: int, frame_h: int,
) -> float:
    """Snapshot scorer (legacy ``_score_detection`` + ITEM 5 max-area bias).

    ``(area*_W_AREA + raw_area*_W_AREA_TIEBREAK + center_h*_W_CENTER
       + conf*_W_CONF + quality*_W_QUALITY) * edge_penalty``

    Centre proximity is horizontal-only (the door crosses the frame
    horizontally as the train passes).  DIVERGES from the legacy port: the area
    term now peaks at ~28% of the frame (was 15%) and carries weight 3.5 (was
    2.0) plus a small un-saturated max-area tie-break, so the LARGEST visible
    full-door frame dominates.  The edge-hugging penalty (0.3) and the
    crop-quality term are UNCHANGED, so blurry / edge frames are still
    suppressed; there is still NO hard rejection gate.
    """
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))
    frame_area = max(1.0, float(frame_w) * float(frame_h))
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_score = min(1.0, bbox_area / (frame_area * _AREA_OPTIMAL_RATIO))
    # Un-saturated raw-area fraction so that, among boxes that all hit the
    # optimal-area cap, the physically largest door still wins the tie.
    raw_area_frac = min(1.0, bbox_area / frame_area)

    frame_cx = max(1.0, frame_w / 2.0)
    cx = (x1 + x2) / 2.0
    center_score = 1.0 - abs(cx - frame_cx) / frame_cx   # legacy: unclamped

    edge_penalty = (_EDGE_PENALTY
                    if is_edge_detection(bbox, frame_w, frame_h) else 1.0)

    score = (
        area_score * _W_AREA
        + raw_area_frac * _W_AREA_TIEBREAK
        + center_score * _W_CENTER
        + float(confidence) * _W_CONF
        + float(quality) * _W_QUALITY
    ) * edge_penalty
    return float(score)


def expand_bbox(bbox: Bbox, frac: float, frame_w: int, frame_h: int) -> list:
    """Grow an axis-aligned ``[x1,y1,x2,y2]`` box by ``frac`` (total, split
    evenly per side) about its centre, clipped to ``[0, frame_w] x [0, frame_h]``.

    Returns a plain 4-float list -- still a clean rectangle (no shape change, so
    legacy draw parity holds).  Degenerate / missing boxes are returned as-is.
    Used to make the chosen door box visually contain the WHOLE door in both the
    processed-video overlay and the evidence crop.
    """
    if bbox is None or len(bbox) != 4:
        return list(bbox) if bbox is not None else bbox
    if not frame_w or not frame_h:
        return [float(v) for v in bbox]
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))
    dx = max(0.0, x2 - x1) * frac / 2.0
    dy = max(0.0, y2 - y1) * frac / 2.0
    nx1 = max(0.0, x1 - dx)
    ny1 = max(0.0, y1 - dy)
    nx2 = min(float(frame_w), x2 + dx)
    ny2 = min(float(frame_h), y2 + dy)
    if nx2 <= nx1 or ny2 <= ny1:
        return [x1, y1, x2, y2]
    return [nx1, ny1, nx2, ny2]
