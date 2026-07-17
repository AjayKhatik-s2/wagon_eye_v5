"""Train-extraction producer package (vendored from the V4 Train-Inspection-Engine).

Reads RAW CCTV video from S3, cuts out the train pass, and uploads trimmed
per-camera clips to a "trimmed" bucket -- the bucket `wagon_eye_v4_new`'s
`--auto` orchestrator polls.  It performs NO inspection.

Core algorithm modules (classifier, direction, extractor, s3, segment_finder,
state, model_store, time_utils, url_utils, video_io) are copied VERBATIM from
the V1 `v4-pipeline` train_extraction package; only `driver.py` (generalized to
all four cameras) and `run_extraction_service.py` (the continuous runner) are
new.  See README.md for how it wires to the inspection pipeline.
"""

from .driver import extract_trains, get_extractor, ALL_CAMERAS  # noqa: F401
