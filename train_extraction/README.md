# train_extraction/ — train-extraction producer

Reads **raw** continuous CCTV video from S3, cuts out the train pass, and
uploads the **trimmed** per-camera clips to the bucket the inspection pipeline
polls. It does **no inspection** — all inspection is done by `wagon_eye_v4_new`
(`orchestrator.master_runner --auto`).

```
RAW CCTV bucket ──▶ [train_extraction.run_extraction_service] ──▶ TRIMMED bucket
                                                                 │
                                            WAGONEYE_S3_INPUT_BUCKET / PREFIXES
                                                                 ▼
                                    [orchestrator.master_runner --auto: inspect]
```

The two systems connect **only through S3** — no shared process, no code
coupling. Extraction writes `"<raw_basename>_train.mp4"`; the raw basename
carries the `YYYYMMDD_HHMMSS` stamp, so the inspection side clusters the four
cameras' clips into one train exactly as before.

## What is vendored vs new

- **Verbatim** (from the V1 `v4-pipeline` `train_extraction/`, itself vendored
  from the V4 Train-Inspection-Engine): `classifier.py`, `direction.py`,
  `extractor.py`, `s3.py`, `segment_finder.py`, `state.py`, `model_store.py`,
  `time_utils.py`, `url_utils.py`, `video_io.py`. Extraction behaviour is
  unchanged.
- **New**: `driver.py` (the four per-camera drivers folded into one
  `build_extractor(camera)` config table) and `run_extraction_service.py`
  (the continuous poll→extract→upload runner).

## Models

Drop the **extraction classify** models here (default
`models/extraction/`, override with `WAGONEYE_EXTRACTION_MODELS_DIR`):

| File                     | Cameras                    |
|--------------------------|----------------------------|
| `side_classification.pt` | RIGHT_UP, LEFT_UP          |
| `top_classification.pt`  | RIGHT_UP_TOP, LEFT_UP_TOP  |

> ⚠️ These are the **extraction** classifiers (classes `empty_track` /
> `wagon` / `engine` / `second_track` …). `side_classification.pt` here may be
> a **different** model than `models/reconstruction/side_classification.pt`
> used by Stage-1 counting — keep them in separate dirs. `top_classification.pt`
> is not part of the inspection model set and must be supplied for the top
> cameras.

## Run

```bash
# all four cameras, continuous:
python -m train_extraction.run_extraction_service

# one camera:
python -m train_extraction.run_extraction_service --camera RIGHT_UP

# single sweep then exit (cron-style):
python -m train_extraction.run_extraction_service --once

# see what WOULD be extracted, upload nothing:
python -m train_extraction.run_extraction_service --dry-run --once
```

Production: `deploy/wagon-eye-extraction.service` (runs alongside
`wagon-eye.service`). Copy `deploy/wagon-eye-extraction.env.example` →
`deploy/wagon-eye-extraction.env` and edit.

## Wiring to inspection

Set the inspection side to poll the **trimmed** bucket/prefixes this producer
writes to (in `deploy/wagon-eye.env`):

```bash
WAGONEYE_S3_INPUT_BUCKET=biro-wagon-pre-processed-video-copy
WAGONEYE_S3_INPUT_PREFIXES=camera_CCTV_HZBN_DHN_2_RIGHT_UP/,camera_CCTV_HZBN_DHN_1_LEFT_UP/,camera_CCTV_HZBN_DHN_5_RIGHT_TOP/,camera_CCTV_HZBN_DHN_6_LEFT_TOP/
```

Then run both services: `wagon-eye-extraction` (producer) and `wagon-eye`
(`--auto` consumer).

## State / restart safety

Processed raw keys are recorded per camera under
`WAGONEYE_EXTRACTION_STATE_DIR` (default `logs/extraction_state/`), so a restart
never re-extracts a handled raw clip. The extractor additionally keeps
cross-clip **ongoing-train** continuity in its own S3 state store (a train that
spans two raw clips is stitched when the continuation arrives).
