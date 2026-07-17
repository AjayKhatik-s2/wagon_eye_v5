"""Identify complete train segments in a video using the frame classifier."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2

from .classifier import FrameClassifier
from .direction import detect_direction_zones


@dataclass
class TrainSegment:
    start_frame: int
    end_frame: int
    fps: float
    duration: float
    direction: str = "unknown"
    is_complete: bool = True
    segments: list[dict] = field(default_factory=list)
    segment_count: int = 1


@dataclass
class IncompleteTrain:
    start_frame: int
    end_frame: int
    fps: float
    duration_so_far: float


class TrainSegmentFinder:
    """Scans a video for one or more train segments and an optional incomplete tail.

    Implements the 3-phase algorithm from the original notebooks:
        1. Collect raw segments (track-streak end + start/end buffers).
        2. Merge nearby segments (gap < 30s).
        3. Validate direction per merged train.
    """

    def __init__(
        self,
        classifier: FrameClassifier,
        min_train_duration: float = 40.0,
        start_buffer_seconds: float = 5.0,
        track_end_seconds: float = 5.0,
        end_extra_buffer: float = 5.0,
        merge_gap_seconds: float = 30.0,
        frame_stride: int = 1,
        direction_flow_downscale: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.classifier = classifier
        self.min_train_duration = min_train_duration
        self.start_buffer_seconds = start_buffer_seconds
        self.track_end_seconds = track_end_seconds
        self.end_extra_buffer = end_extra_buffer
        self.merge_gap_seconds = merge_gap_seconds
        # Classify every Nth frame in the raw-video scan. >1 trades sub-second
        # boundary precision (absorbed by the multi-second buffers) for speed.
        self.frame_stride = max(1, int(frame_stride))
        self.direction_flow_downscale = direction_flow_downscale
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------

    def check_train_at_start(self, video_path: str, check_duration_sec: int = 5) -> bool:
        """Return True if a train is visible near the start of the video.

        Used to decide whether a new clip continues an ongoing-train sequence.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ret, frame = cap.read()
        if ret and self.classifier.classify(frame).is_train:
            cap.release()
            return True

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        max_frames = int(check_duration_sec * fps)
        stride = self.frame_stride
        detections = 0
        frame_idx = -1
        while frame_idx + 1 < max_frames:
            if not cap.grab():
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue  # cheap skip — decode only every Nth frame
            ret, frame = cap.retrieve()
            if not ret:
                break
            if self.classifier.classify(frame).is_train:
                detections += 1
                if detections >= 2:
                    cap.release()
                    return True
        cap.release()
        return False

    # ------------------------------------------------------------------

    def analyze(
        self, video_path: str, min_incomplete_duration: float = 30.0
    ) -> Tuple[List[TrainSegment], Optional[IncompleteTrain]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(
            "Analyzing %s | FPS=%.2f frames=%d duration=%.1fs",
            video_path, fps, total_frames, total_frames / fps,
        )

        track_end_threshold = int(self.track_end_seconds * fps)
        start_buffer_frames = int(self.start_buffer_seconds * fps)
        end_extra_frames = int(self.end_extra_buffer * fps)
        stride = self.frame_stride

        raw_segments: list[dict] = []
        in_train = False
        raw_train_start_frame: Optional[int] = None
        consecutive_track_frames = 0
        pending_end_frame: Optional[int] = None

        # Iterate real frame indices, but only classify every ``stride`` frames.
        # Counters stay in REAL-frame units (each observed no-train frame stands
        # for ``stride`` real frames) so ``track_end_threshold`` keeps its
        # seconds-based meaning and emitted boundaries are unchanged at stride=1.
        frame_idx = -1
        while True:
            if not cap.grab():
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue  # cheap skip — avoid decoding frames we won't classify
            ret, frame = cap.retrieve()
            if not ret:
                break

            cls = self.classifier.classify(frame)
            if cls.ignored:
                # e.g. a parallel train on the second track — skip this frame
                # entirely so it neither starts/extends our train nor counts
                # toward the empty-track "train ended" streak.
                continue
            is_train = cls.is_train

            if not in_train:
                if is_train:
                    in_train = True
                    raw_train_start_frame = frame_idx
                    consecutive_track_frames = 0
                    pending_end_frame = None
            else:
                if is_train:
                    consecutive_track_frames = 0
                    pending_end_frame = None
                else:
                    if consecutive_track_frames == 0:
                        pending_end_frame = frame_idx
                    consecutive_track_frames += stride

                    if consecutive_track_frames >= track_end_threshold:
                        start_frame = max(0, raw_train_start_frame - start_buffer_frames)
                        end_frame = min(
                            total_frames - 1,
                            pending_end_frame + end_extra_frames,
                        )
                        duration = (end_frame - start_frame) / fps
                        raw_segments.append({
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "start_time": start_frame / fps,
                            "end_time": end_frame / fps,
                            "duration": duration,
                            "fps": fps,
                        })

                        in_train = False
                        raw_train_start_frame = None
                        consecutive_track_frames = 0
                        pending_end_frame = None

        cap.release()

        # Handle incomplete tail
        incomplete: Optional[IncompleteTrain] = None
        if in_train and raw_train_start_frame is not None:
            trimmed_start = max(0, raw_train_start_frame - start_buffer_frames)
            duration_so_far = (total_frames - trimmed_start) / fps
            incomplete = IncompleteTrain(
                start_frame=trimmed_start,
                end_frame=total_frames - 1,
                fps=fps,
                duration_so_far=duration_so_far,
            )
            self.logger.info(
                "Train INCOMPLETE: start=%d duration_so_far=%.1fs",
                trimmed_start, duration_so_far,
            )
            return [], incomplete

        if not raw_segments:
            return [], None

        # Phase 2: merge nearby segments
        raw_segments.sort(key=lambda x: x["start_frame"])
        merged: list[dict] = []
        current: Optional[dict] = None
        for seg in raw_segments:
            if seg["duration"] < self.min_train_duration:
                continue
            if current is None:
                current = dict(seg)
                current["segments"] = [seg]
                current["segment_count"] = 1
                continue
            gap_seconds = seg["start_time"] - current["end_time"]
            if gap_seconds <= self.merge_gap_seconds:
                current["end_frame"] = seg["end_frame"]
                current["end_time"] = seg["end_time"]
                current["duration"] = current["end_time"] - current["start_time"]
                current["segments"].append(seg)
                current["segment_count"] += 1
            else:
                if current["duration"] >= self.min_train_duration:
                    merged.append(current)
                current = dict(seg)
                current["segments"] = [seg]
                current["segment_count"] = 1

        if current and current["duration"] >= self.min_train_duration:
            merged.append(current)

        if not merged:
            return [], None

        # Phase 3: validate direction
        train_segments: list[TrainSegment] = []
        for train in merged:
            direction = detect_direction_zones(
                video_path,
                train["start_frame"],
                fps,
                classifier=self.classifier,
                flow_downscale=self.direction_flow_downscale,
                logger=self.logger,
            )
            train_segments.append(TrainSegment(
                start_frame=train["start_frame"],
                end_frame=train["end_frame"],
                fps=fps,
                duration=train["duration"],
                direction=direction,
                segments=train["segments"],
                segment_count=train["segment_count"],
            ))

        return train_segments, None
