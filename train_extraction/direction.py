"""Train direction detection via motion tracking + (optional) zone analysis."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np


def detect_direction_zones(
    video_path: str,
    start_frame: int,
    fps: float,
    classifier=None,
    check_duration: float = 10.0,
    flow_downscale: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Determine train direction using optical-flow + optional zone analysis.

    Returns ``'left-to-right'``, ``'right-to-left'`` or ``'unknown'``.

    ``flow_downscale`` (0 < f <= 1) shrinks frames before the dense Farneback
    optical-flow computation — the single most expensive op here. The decision
    only uses the *sign* of mean horizontal motion, which survives downscaling,
    so the label is unchanged while cost drops roughly with the square of f.
    ``1.0`` keeps the original full-resolution behaviour.
    """
    logger = logger or logging.getLogger(__name__)
    scale = flow_downscale if 0 < flow_downscale <= 1.0 else 1.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "unknown"
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    check_frames = int(check_duration * fps)

    zones_sequence: list[str] = []
    motion_vectors: list[tuple[float, float]] = []
    prev_gray = None

    def _gray(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return g

    ret, first_frame = cap.read()
    if ret:
        prev_gray = _gray(first_frame)
        if classifier is not None:
            cls = classifier.classify(first_frame)
            if cls.zone is not None:
                zones_sequence.append(cls.zone)

    for _ in range(1, check_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if classifier is not None:
            cls = classifier.classify(frame)
            if cls.zone is not None:
                zones_sequence.append(cls.zone)

        gray = _gray(frame)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            h, w = flow.shape[:2]
            mid = flow[int(h * 0.3):int(h * 0.7), int(w * 0.2):int(w * 0.8), :]
            avg_dx = float(np.mean(mid[:, :, 0]))
            avg_dy = float(np.mean(mid[:, :, 1]))
            motion_vectors.append((avg_dx, avg_dy))
        prev_gray = gray

    cap.release()

    # Zone-based decision
    zone_direction = "unknown"
    if len(zones_sequence) >= 5:
        first_zone, last_zone = zones_sequence[0], zones_sequence[-1]
        if first_zone == "left" and last_zone in {"center", "right"}:
            zone_direction = "left-to-right"
        elif first_zone == "right" and last_zone in {"center", "left"}:
            zone_direction = "right-to-left"
        elif first_zone == "center":
            right_count = zones_sequence.count("right")
            left_count = zones_sequence.count("left")
            if right_count > left_count * 1.5:
                zone_direction = "left-to-right"
            elif left_count > right_count * 1.5:
                zone_direction = "right-to-left"

    # Motion-based decision
    motion_direction = "unknown"
    if len(motion_vectors) >= 5:
        avg_motion_x = float(np.mean([dx for dx, _ in motion_vectors]))
        if avg_motion_x > 0.2:
            motion_direction = "left-to-right"
        elif avg_motion_x < -0.2:
            motion_direction = "right-to-left"

    if motion_direction != "unknown":
        final = motion_direction
        conf = "high" if zone_direction == motion_direction else "medium"
    elif zone_direction != "unknown":
        final = zone_direction
        conf = "medium"
    else:
        final = "unknown"
        conf = "low"

    logger.info(
        "Direction: zones=%s motion=%s final=%s (%s)",
        zone_direction, motion_direction, final, conf,
    )
    return final
