"""Multi-video train extractor — handles ongoing-train sequences across clips."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2

from .s3 import S3Client
from .state import PipelineState, PipelineStateStore
from .video_io import merge_videos, trim_video
from .time_utils import parse_timestamp_from_filename, utc_to_ist
from .url_utils import s3_object_url, split_bucket_prefix
from .direction import detect_direction_zones
from .segment_finder import TrainSegmentFinder


@dataclass
class ExtractedTrain:
    local_path: str
    s3_key: str
    raw_video_basename: str
    upload_timestamp: datetime
    direction: str
    trimmed_video_url: str
    raw_video_urls: List[str]


class TrainExtractor:
    """Extracts trimmed train clips from raw S3 videos.

    Owns multi-video sequence tracking — if a clip ends mid-train, the next clip
    is checked for continuation and the two (or more) are merged before trimming.
    """

    def __init__(
        self,
        s3: S3Client,
        segment_finder: TrainSegmentFinder,
        raw_video_bucket: str,
        trimmed_video_bucket: str,
        state: PipelineState,
        state_store: PipelineStateStore,
        region: str,
        temp_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.s3 = s3
        self.segment_finder = segment_finder
        self.raw_video_bucket = raw_video_bucket
        self.trimmed_video_bucket = trimmed_video_bucket
        self.state = state
        self.state_store = state_store
        self.region = region
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="train_extractor_")
        self.logger = logger or logging.getLogger(__name__)

        self.ongoing_videos: list[Tuple[str, str, str]] = []  # (local, s3_key, basename)
        self.ongoing_direction: Optional[str] = None
        self.train_in_progress: bool = False

    # ------------------------------------------------------------------

    def _download(self, s3_key: str) -> str:
        bucket, _ = split_bucket_prefix(self.raw_video_bucket)
        local_path = os.path.join(self.temp_dir, os.path.basename(s3_key))
        self.s3.download_file(bucket, s3_key, local_path)
        return local_path

    def _upload(self, local_path: str, s3_key: str) -> str:
        return self.s3.upload_file(local_path, self.trimmed_video_bucket, s3_key)

    def _public_url(self, key: str) -> str:
        return s3_object_url(self.trimmed_video_bucket, key, self.region)

    def _raw_url(self, key: str) -> str:
        return s3_object_url(self.raw_video_bucket, key, self.region)

    def _video_upload_ist(self, s3_key: str) -> datetime:
        bucket, _ = split_bucket_prefix(self.raw_video_bucket)
        head = self.s3.head_object(bucket, s3_key)
        return utc_to_ist(head["LastModified"])

    def _parse_timestamp(self, video_key: str) -> datetime:
        parsed = parse_timestamp_from_filename(os.path.basename(video_key))
        if parsed is not None:
            return parsed
        return self._video_upload_ist(video_key)

    # ------------------------------------------------------------------

    def extract(
        self, video_key: str, min_incomplete_duration: float = 30.0
    ) -> List[ExtractedTrain]:
        """Process a single raw video, returning any complete train clips emitted."""
        upload_ts = self._parse_timestamp(video_key)
        basename = os.path.splitext(os.path.basename(video_key))[0]
        local_path = self._download(video_key)
        emitted: list[ExtractedTrain] = []

        if self.train_in_progress:
            emitted = self._handle_ongoing(
                video_key, basename, upload_ts, local_path, min_incomplete_duration
            )
            if emitted is not None:
                return emitted

        return self._handle_standalone(
            video_key, basename, upload_ts, local_path, min_incomplete_duration
        )

    # --- branch helpers -----------------------------------------------

    def _emit_segments(
        self,
        merged_path: str,
        segments,
        first_video_name: str,
        first_video_ts: datetime,
        raw_video_urls: List[str],
        suffix_complete: str = "train",
    ) -> list[ExtractedTrain]:
        emitted: list[ExtractedTrain] = []
        single = len(segments) == 1
        for idx, seg in enumerate(segments, start=1):
            out_name = (
                f"{first_video_name}_{suffix_complete}.mp4"
                if single
                else f"{first_video_name}_{suffix_complete}_part{idx}.mp4"
            )
            out_path = os.path.join(self.temp_dir, out_name)
            trim_video(
                merged_path, seg.start_frame, seg.end_frame, seg.fps, out_path, self.logger
            )
            s3_key = self._upload(out_path, out_name)
            url = self._public_url(s3_key)
            emitted.append(ExtractedTrain(
                local_path=out_path,
                s3_key=s3_key,
                raw_video_basename=first_video_name,
                upload_timestamp=first_video_ts,
                direction=seg.direction,
                trimmed_video_url=url,
                raw_video_urls=raw_video_urls,
            ))
        return emitted

    def _handle_ongoing(
        self, video_key, basename, upload_ts, local_path, min_incomplete_duration
    ):
        self.logger.info(
            "Ongoing train with %d prior clip(s) — checking continuation",
            len(self.ongoing_videos),
        )
        if self.segment_finder.check_train_at_start(local_path, 5):
            self.logger.info("New clip continues the train — merging")
            self.ongoing_videos.append((local_path, video_key, basename))

            all_paths = [vp for vp, _, _ in self.ongoing_videos]
            merged_path = os.path.join(self.temp_dir, "merged_ongoing_train.mp4")
            merge_videos(all_paths, merged_path, self.logger, tmp_dir=self.temp_dir)
            segments, incomplete = self.segment_finder.analyze(
                merged_path, min_incomplete_duration
            )

            first_name = self.ongoing_videos[0][2]
            first_ts = self._parse_timestamp(self.ongoing_videos[0][1])
            raw_urls = [self._raw_url(vkey) for _, vkey, _ in self.ongoing_videos]

            emitted: list[ExtractedTrain] = []
            if incomplete is not None:
                # Still incomplete — emit any complete segments, keep current video.
                if segments:
                    emitted = self._emit_segments(merged_path, segments, first_name, first_ts, raw_urls)
                    videos_to_keep = [(local_path, video_key, basename)]
                    for vpath, vkey, _ in self.ongoing_videos:
                        if vpath != local_path and os.path.exists(vpath):
                            try:
                                os.remove(vpath)
                            except OSError:
                                pass
                        self.state.processed_videos.add(vkey)
                    self.ongoing_videos = videos_to_keep
                if os.path.exists(merged_path):
                    os.remove(merged_path)
                self.state.processed_videos.add(video_key)
                self.state_store.save(self.state)
                return emitted

            # Sequence completed.
            if not segments:
                cap = cv2.VideoCapture(merged_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                direction = detect_direction_zones(
                    merged_path, 0, fps,
                    classifier=self.segment_finder.classifier, logger=self.logger,
                )
            else:
                direction = segments[0].direction

            if segments:
                emitted = self._emit_segments(merged_path, segments, first_name, first_ts, raw_urls)

            if os.path.exists(merged_path):
                os.remove(merged_path)
            for vpath, vkey, _ in self.ongoing_videos:
                if os.path.exists(vpath):
                    try:
                        os.remove(vpath)
                    except OSError:
                        pass
                self.state.processed_videos.add(vkey)
            self.ongoing_videos = []
            self.ongoing_direction = None
            self.train_in_progress = False
            self.state_store.save(self.state)
            return emitted

        # Current clip does NOT continue — flush accumulated.
        self.logger.info("New clip does not continue the train — flushing accumulated")
        all_paths = [vp for vp, _, _ in self.ongoing_videos]
        merged_path = os.path.join(self.temp_dir, "merged_incomplete_train.mp4")
        merge_videos(all_paths, merged_path, self.logger, tmp_dir=self.temp_dir)

        first_name = self.ongoing_videos[0][2]
        first_ts = self._parse_timestamp(self.ongoing_videos[0][1])
        raw_urls = [self._raw_url(vkey) for _, vkey, _ in self.ongoing_videos]

        cap = cv2.VideoCapture(merged_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        segments, incomplete = self.segment_finder.analyze(merged_path, min_incomplete_duration)
        direction = (
            segments[0].direction if segments
            else detect_direction_zones(
                merged_path, 0, fps,
                classifier=self.segment_finder.classifier, logger=self.logger,
            )
        )

        emitted: list[ExtractedTrain] = []
        if segments:
            emitted.extend(self._emit_segments(merged_path, segments, first_name, first_ts, raw_urls))

        # Incomplete tail — save only if ≥ min train duration.
        if incomplete is not None and incomplete.duration_so_far >= self.segment_finder.min_train_duration:
            out_name = f"{first_name}_train_incomplete.mp4"
            out_path = os.path.join(self.temp_dir, out_name)
            trim_video(
                merged_path,
                incomplete.start_frame,
                incomplete.end_frame,
                fps,
                out_path,
                self.logger,
            )
            s3_key = self._upload(out_path, out_name)
            emitted.append(ExtractedTrain(
                local_path=out_path,
                s3_key=s3_key,
                raw_video_basename=first_name,
                upload_timestamp=first_ts,
                direction=direction,
                trimmed_video_url=self._public_url(s3_key),
                raw_video_urls=raw_urls,
            ))
        elif not segments and not incomplete:
            total_dur = total_frames / fps
            if total_dur >= self.segment_finder.min_train_duration:
                out_name = f"{first_name}_train_incomplete.mp4"
                out_path = os.path.join(self.temp_dir, out_name)
                shutil.copy(merged_path, out_path)
                s3_key = self._upload(out_path, out_name)
                emitted.append(ExtractedTrain(
                    local_path=out_path,
                    s3_key=s3_key,
                    raw_video_basename=first_name,
                    upload_timestamp=first_ts,
                    direction=direction,
                    trimmed_video_url=self._public_url(s3_key),
                    raw_video_urls=raw_urls,
                ))

        if os.path.exists(merged_path):
            os.remove(merged_path)
        for vpath, vkey, _ in self.ongoing_videos:
            if os.path.exists(vpath):
                try:
                    os.remove(vpath)
                except OSError:
                    pass
            self.state.processed_videos.add(vkey)

        self.ongoing_videos = []
        self.ongoing_direction = None
        self.train_in_progress = False
        self.state_store.save(self.state)

        # Bug fix: previously this returned None and let extract() call
        # _handle_standalone on the current clip — but that discarded the
        # `emitted` trains we just trimmed + uploaded above (the flushed
        # incomplete tail of the prior train). Process the current clip
        # inline so its result is concatenated with the flush emissions.
        current_emitted = self._handle_standalone(
            video_key, basename, upload_ts, local_path, min_incomplete_duration,
        )
        return emitted + (current_emitted or [])

    def _handle_standalone(
        self, video_key, basename, upload_ts, local_path, min_incomplete_duration
    ) -> list[ExtractedTrain]:
        segments, incomplete = self.segment_finder.analyze(local_path, min_incomplete_duration)
        emitted: list[ExtractedTrain] = []
        raw_urls = [self._raw_url(video_key)]

        if incomplete is not None:
            self.ongoing_videos = [(local_path, video_key, basename)]
            self.train_in_progress = True
            if segments:
                self.ongoing_direction = segments[0].direction
            else:
                cap = cv2.VideoCapture(local_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                self.ongoing_direction = detect_direction_zones(
                    local_path, 0, fps,
                    classifier=self.segment_finder.classifier, logger=self.logger,
                )

            if segments:
                emitted.extend(self._emit_segments(local_path, segments, basename, upload_ts, raw_urls))

            self.state.processed_videos.add(video_key)
            self.state_store.save(self.state)
            return emitted

        if segments:
            emitted.extend(self._emit_segments(local_path, segments, basename, upload_ts, raw_urls))

        if not self.train_in_progress:
            try:
                os.remove(local_path)
            except OSError:
                pass

        self.state.processed_videos.add(video_key)
        self.state_store.save(self.state)
        return emitted
