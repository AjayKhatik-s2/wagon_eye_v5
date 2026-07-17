"""Generalized train-extraction driver (all four cameras, one code path).

Vendored from the V1 `v4-pipeline` train_extraction package (itself vendored
from the V4 Train-Inspection-Engine).  The four per-camera drivers there were
byte-identical except for a small config table -- this module folds them into
ONE `build_extractor(camera)` / `extract_trains(camera, key)` pair.

Role in the new architecture
----------------------------
This is a PRODUCER only.  It reads a RAW CCTV bucket, cuts out the train pass
with the vendored V4 extractor, and UPLOADS the trimmed clip(s) to a "trimmed"
bucket.  It performs NO inspection.  Point the trimmed bucket at the prefixes
that `wagon_eye_v4_new`'s `--auto` orchestrator polls
(WAGONEYE_S3_INPUT_BUCKET / WAGONEYE_S3_INPUT_PREFIXES) and the two systems
connect through S3 with zero code coupling:

    RAW bucket --> [extraction.driver: cut train] --> TRIMMED bucket
                                                          |
                                        WAGONEYE_S3_INPUT_BUCKET / PREFIXES
                                                          v
                                 [orchestrator.master_runner --auto: inspect]

The trimmed clip is named "<raw_basename>_train.mp4"; the raw basename carries
the "YYYYMMDD_HHMMSS" stamp, so the inspection side's `parse_train_timestamp`
clusters the four cameras' clips into one TrainBatch exactly as before.

Every value below is env-overridable; the defaults reproduce the production
per-camera config (raw/trimmed buckets, classify model, ignore classes).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List, Optional

from ultralytics import YOLO

from .classifier import DEFAULT_CLASS_ALIASES, FrameClassifier
from .extractor import ExtractedTrain, TrainExtractor
from .s3 import S3Client
from .segment_finder import TrainSegmentFinder
from .state import PipelineStateStore

_LOGGER = logging.getLogger("extraction.driver")

# train_extraction/ -> <PROJECT_ROOT>/train_extraction/driver.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Canonical camera ids (kept local so this producer never imports the inspection
# package -- they must match core.constants.ALL_CAMERAS).
RIGHT_UP = "RIGHT_UP"
LEFT_UP = "LEFT_UP"
RIGHT_UP_TOP = "RIGHT_UP_TOP"
LEFT_UP_TOP = "LEFT_UP_TOP"
ALL_CAMERAS = (RIGHT_UP, LEFT_UP, RIGHT_UP_TOP, LEFT_UP_TOP)

# -----------------------------------------------------------------------------
# Per-camera extraction config (the ONLY thing that differed between the four
# vendored drivers).  raw/trimmed are "<bucket>/<prefix>" strings; `model` is
# the classify-model filename expected under the extraction models dir.
# -----------------------------------------------------------------------------

_CAMERA_CONFIG: Dict[str, Dict[str, object]] = {
    RIGHT_UP: {
        "raw":     "biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_2_RIGHT_UP",
        "trimmed": "biro-wagon-pre-processed-video-copy/camera_CCTV_HZBN_DHN_2_RIGHT_UP",
        "model":   "side_classification.pt",
        "ignore":  ["second_track"],   # a parallel train is ignored (V4 right_up.yaml)
    },
    LEFT_UP: {
        "raw":     "biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_1_LEFT_UP",
        "trimmed": "biro-wagon-pre-processed-video-copy/camera_CCTV_HZBN_DHN_1_LEFT_UP",
        "model":   "side_classification.pt",
        "ignore":  [],
    },
    RIGHT_UP_TOP: {
        "raw":     "biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_5_RIGHT_TOP",
        "trimmed": "biro-wagon-pre-processed-video-copy/camera_CCTV_HZBN_DHN_5_RIGHT_TOP",
        "model":   "top_classification.pt",
        "ignore":  [],
    },
    LEFT_UP_TOP: {
        "raw":     "biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_6_LEFT_TOP",
        "trimmed": "biro-wagon-pre-processed-video-copy/camera_CCTV_HZBN_DHN_6_LEFT_TOP",
        "model":   "top_classification.pt",
        "ignore":  [],
    },
}

# Extraction classify models live here by default.  IMPORTANT: these are the
# EXTRACTION classifiers (empty_track / wagon / engine / second_track) -- they
# are NOT the inspection models.  In particular side_classification.pt here may
# differ from models/reconstruction/side_classification.pt used by Stage 1, so
# keep them in a separate dir to avoid a filename collision.
_DEFAULT_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "extraction")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _cam_env(camera: str, name: str, default: str) -> str:
    """Per-camera override wins: WAGONEYE_EXTRACTION_<CAM>_<NAME> then <NAME>."""
    return os.environ.get(f"WAGONEYE_EXTRACTION_{camera}_{name}",
                          os.environ.get(f"WAGONEYE_EXTRACTION_{name}", default))


# Process-wide extractor cache (one per camera) so the extractor's in-memory
# ongoing-train state survives across polling-loop iterations.
_EXTRACTORS: Dict[str, TrainExtractor] = {}


def _build_extractor(camera: str) -> TrainExtractor:
    if camera not in _CAMERA_CONFIG:
        raise ValueError(f"unknown camera {camera!r}; expected one of {ALL_CAMERAS}")
    cfg = _CAMERA_CONFIG[camera]
    log = logging.getLogger(f"extraction.{camera.lower()}")

    region = _env("AWS_REGION", os.environ.get("WAGONEYE_S3_REGION", "ap-south-1"))
    raw_bucket = _cam_env(camera, "RAW_BUCKET", str(cfg["raw"]))
    trimmed_bucket = _cam_env(camera, "TRIMMED_BUCKET", str(cfg["trimmed"]))

    models_dir = _env("WAGONEYE_EXTRACTION_MODELS_DIR", _DEFAULT_MODELS_DIR)
    cls_path = _cam_env(camera, "CLASSIFICATION_MODEL",
                        os.path.join(models_dir, str(cfg["model"])))
    if not os.path.exists(cls_path):
        raise FileNotFoundError(
            f"[{camera}] extraction classify model not found: {cls_path} "
            f"(set WAGONEYE_EXTRACTION_MODELS_DIR or "
            f"WAGONEYE_EXTRACTION_{camera}_CLASSIFICATION_MODEL)")

    log.info("Loading extraction classifier: %s", cls_path)
    ignore = [c for c in _cam_env(camera, "IGNORE_CLASSES",
                                  ",".join(cfg["ignore"])).split(",") if c]  # type: ignore[arg-type]
    classifier = FrameClassifier(
        YOLO(cls_path, task="classify"),
        track_class_name=_cam_env(camera, "TRACK_CLASS", "empty_track"),
        class_aliases=DEFAULT_CLASS_ALIASES,
        ignore_class_names=ignore,
    )
    log.info("classifier classes=%s track=%r ignore=%s",
             classifier.model.names, classifier.track_class_name,
             classifier.ignore_class_names)

    finder = TrainSegmentFinder(
        classifier,
        min_train_duration=float(_cam_env(camera, "MIN_TRAIN_DURATION", "40")),
        start_buffer_seconds=float(_cam_env(camera, "START_BUFFER_SECONDS", "5")),
        track_end_seconds=float(_cam_env(camera, "TRACK_END_SECONDS", "5")),
        end_extra_buffer=float(_cam_env(camera, "END_EXTRA_BUFFER", "5")),
        merge_gap_seconds=float(_cam_env(camera, "MERGE_GAP_SECONDS", "30")),
        frame_stride=int(_cam_env(camera, "ANALYSIS_FRAME_STRIDE", "15")),
        direction_flow_downscale=float(_cam_env(camera, "DIRECTION_FLOW_DOWNSCALE", "1.0")),
        logger=log,
    )

    s3 = S3Client(region=region, logger=log)   # IAM-role creds on EC2
    state_store = PipelineStateStore(s3, trimmed_bucket, logger=log)
    state = state_store.load()

    temp_dir = _cam_env(camera, "TMPDIR",
                        os.path.join(tempfile.gettempdir(),
                                     f"train_extraction_{camera.lower()}"))
    os.makedirs(temp_dir, exist_ok=True)

    log.info("extractor ready: raw=%s trimmed=%s", raw_bucket, trimmed_bucket)
    return TrainExtractor(
        s3=s3, segment_finder=finder,
        raw_video_bucket=raw_bucket, trimmed_video_bucket=trimmed_bucket,
        state=state, state_store=state_store, region=region,
        temp_dir=temp_dir, logger=log,
    )


def get_extractor(camera: str) -> TrainExtractor:
    """Return the process-wide extractor for `camera`, building it on first use."""
    if camera not in _EXTRACTORS:
        _EXTRACTORS[camera] = _build_extractor(camera)
    return _EXTRACTORS[camera]


def raw_bucket_for(camera: str) -> str:
    return _cam_env(camera, "RAW_BUCKET", str(_CAMERA_CONFIG[camera]["raw"]))


def extract_trains(camera: str, raw_s3_key: str,
                   min_incomplete_duration: Optional[float] = None) -> List[ExtractedTrain]:
    """Cut trimmed train clip(s) out of ONE raw video key for `camera`.

    Downloads the raw video, finds + trims the train segment(s), uploads the
    trimmed clip(s) to the camera's trimmed bucket, and returns them.  Returns
    [] when the raw clip holds no complete train yet (leading part of a train
    that continues into the next clip -- held in the extractor's in-memory
    ongoing state until the continuation arrives)."""
    ex = get_extractor(camera)
    if min_incomplete_duration is None:
        min_incomplete_duration = ex.segment_finder.min_train_duration
    return ex.extract(raw_s3_key, min_incomplete_duration=min_incomplete_duration)
