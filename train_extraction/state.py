"""Pipeline state persistence — processed videos + baseline set."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Iterable, Optional, Set

from .time_utils import today_ist_str
from .url_utils import split_bucket_prefix


@dataclass
class PipelineState:
    """In-memory state with S3-backed persistence."""

    processed_videos: Set[str] = field(default_factory=set)
    baseline_videos: Set[str] = field(default_factory=set)


class PipelineStateStore:
    """Loads and saves :class:`PipelineState` from an S3 location."""

    def __init__(self, s3_client, trimmed_video_bucket: str, logger: Optional[logging.Logger] = None):
        self.s3 = s3_client
        self.trimmed_video_bucket = trimmed_video_bucket
        self.logger = logger or logging.getLogger(__name__)

    @property
    def state_key(self) -> str:
        bucket, prefix = split_bucket_prefix(self.trimmed_video_bucket)
        return f"{prefix}/.pipeline_state.json" if prefix else ".pipeline_state.json"

    def load(self) -> PipelineState:
        bucket, _ = split_bucket_prefix(self.trimmed_video_bucket)
        try:
            tmp = tempfile.mktemp(suffix=".json")
            self.s3.client.download_file(bucket, self.state_key, tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = PipelineState(
                processed_videos=set(data.get("processed_videos", [])),
                baseline_videos=set(data.get("baseline_videos", [])),
            )
            self.logger.info(
                "Loaded state: %d processed video(s)", len(state.processed_videos),
            )
            try:
                os.remove(tmp)
            except OSError:
                pass
            return state
        except Exception:  # noqa: BLE001 — first-run is expected
            self.logger.info("No previous state — starting fresh")
            return PipelineState()

    def save(self, state: PipelineState) -> None:
        bucket, _ = split_bucket_prefix(self.trimmed_video_bucket)
        tmp = tempfile.mktemp(suffix=".json")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "processed_videos": sorted(state.processed_videos),
                        "baseline_videos": sorted(state.baseline_videos),
                        "last_updated": today_ist_str(),
                    },
                    f,
                    indent=2,
                )
            self.s3.client.upload_file(tmp, bucket, self.state_key)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("State save failed: %s", e)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def add_processed(self, state: PipelineState, keys: Iterable[str]) -> None:
        state.processed_videos.update(keys)
        self.save(state)
