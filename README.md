# WagonEye v4 — Train-State-Native Production Pipeline

Single source of truth for wagon counting, numbering, and classification
is the **GlobalTrainState** produced by `wagon_count/`. Everything
downstream — frame extraction, feature inference, fusion, reporting —
consumes that state and never recounts wagons, never re-segments video,
never re-runs gap detection.

```
4 source videos
    │
    ▼
[Stage 1] reconstruction.runner  →  global_state/global_train_state.json
            + wagon_count tracking overlay mp4s (debug artifacts)
            → global_state/processed_videos/<CAM>_processed.mp4
    │
    ▼
[Stage 2] materializer.wagon_cache_builder
            single-pass per video → wagon_cache/<GW_n>/<camera>/*.jpg
    │
    ▼
[Stage 3] features.{door,load,damage,ocr} (parallel)
            pure YOLO inference on cached frames
            persists evidence snapshots + metadata under evidence/
            → wagon_states/<feature>/<GW_n>.json
            → evidence/<GW_n>/<feature>/{*.jpg,metadata.json}
    │
    ▼
[Stage 4] fusion.wagon_state_builder
            authority-rule merge → wagon_states/unified/<GW_n>.json
    │
    ▼
[Stage 4b] rendering.feature_overlay_renderer  (visualization-only,
            consumes state + evidence + tracking JSON; never reruns
            any detector model)
            → processed_videos/<CAM>_processed.mp4
    │
    ▼
[Stage 5a] reporting.camera_reports   (legacy camera-wise hierarchy)
            4 camera-wise PDFs, each showing ONLY what that camera is
            authoritative for:
              RIGHT_UP     → right door, OCR, classification
              LEFT_UP      → left door
              RIGHT_UP_TOP → load, top damage
              LEFT_UP_TOP  → top-damage support / validation
            Each report: Camera Summary page (visible wagons, anomalies,
            processing confidence, coverage %), per-wagon pages (2×2
            quartile overview + per-detection detail pages, snapshots
            from THAT camera only), Anomaly Summary grouped by severity,
            and a Camera Evidence section.
            → reports/{right_up,left_up,right_up_top,left_up_top}_report.pdf
    │
    ▼
[Stage 5b] reporting.combined_train_report
            Aggregates the 4 camera reports into the unified train view
            (legacy CombinedReportGenerator visual identity, rebuilt
            against v4 state via _adapter.LegacyViewModel): navy title
            banner, VIDEO EVIDENCE table, PARTIAL REPORT banner,
            DETAILED CAMERA REPORTS links (LEFT/RIGHT/R-TOP/L-TOP),
            10-col INSPECTION SUMMARY KPI row (LOCO, RAKE TYPE, STATUS),
            7-col wagon inspection table with issue-row highlighting,
            and the "Damaged Wagon Report" per-anomaly evidence grid
            (camera-priority ordered).  Schema v4 JSON includes
            legacy_view_model.
            → reports/combined_train_report.{pdf,json}
    │
    ▼
[Stage 6] delivery.s3_upload + delivery.notification
            S3 archive (incl. evidence + processed_videos) +
            one email per batch
```

## Package layout

```
wagon_eye_v4/
├── README.md                          (this file)
├── requirements.txt
├── orchestrator/master_runner.py      ★ entry point
├── reconstruction/runner.py           Stage 1 (subprocess wagon_count)
├── materializer/wagon_cache_builder.py Stage 2
├── features/
│   ├── _common.py                     YOLO loader cache + helpers
│   ├── _evidence.py                   evidence persistence helpers
│   │                                   (per-wagon JPEG + metadata)
│   ├── door/processor.py              door_state.pt
│   ├── load/processor.py              loaded.pt
│   ├── damage/processor.py            damage.pt
│   └── ocr/processor.py               wagon_id_counting.pt + easyocr
├── fusion/wagon_state_builder.py      Stage 4
├── rendering/
│   └── feature_overlay_renderer.py    Stage 4b (visualization-only;
│                                       never reruns any detector)
├── reporting/
│   ├── _brand.py                      Legacy WagonEye palette, paragraph
│   │                                   styles, anomaly + state helpers,
│   │                                   page widgets (logo, warning banner,
│   │                                   camera links).
│   ├── _adapter.py                    v4 backend -> legacy report view-
│   │                                   model (merged_wagons + per-camera
│   │                                   doors + KPIs).
│   ├── _evidence_lookup.py            Quartile + midpoint cache frame
│   │                                   resolution + evidence snapshot
│   │                                   path helpers.
│   ├── _pages.py                      Shared reportlab page widgets
│   │                                   (doc maker, bordered image,
│   │                                   detection-summary table, wagon
│   │                                   quartile overview, detail page,
│   │                                   simple-state page).
│   ├── camera_reports.py              Stage 5a (4 camera-wise PDFs by
│   │                                   camera authority: RIGHT_UP /
│   │                                   LEFT_UP / RIGHT_UP_TOP /
│   │                                   LEFT_UP_TOP).
│   ├── combined_train_report.py       Stage 5b (aggregates the 4 camera
│   │                                   reports; legacy visual identity:
│   │                                   navy title banner, 10-col KPI
│   │                                   summary, 7-col wagon table,
│   │                                   Damaged Wagon Report evidence
│   │                                   grid; schema v4).
│   └── assets/Logo.jpeg               Per-page logo (copied from the
│                                       legacy product).
├── delivery/
│   ├── s3_upload.py                   Stage 6: PDF/JSON + tree upload
│   └── notification.py                Stage 6: one email per batch
├── core/
│   ├── constants.py                   camera ids, classes, statuses
│   ├── unified_wagon_state.py         UnifiedWagonState dataclass
│   ├── global_state_loader.py         GlobalTrainState (in-memory)
│   └── batch.py                       CameraVideo / TrainBatch
├── models/
│   ├── reconstruction/                drop your 4 Stage-1 .pt files
│   │                                   (right_up_gap.pt, left_up_gap.pt,
│   │                                    top_gap.pt, side_classification.pt)
│   └── features/                      drop your 4 Stage-3 .pt files
│                                       (door_state.pt, loaded.pt,
│                                        damage.pt, wagon_id_counting.pt)
└── wagon_count/                       Phase-1 backend (copied verbatim
                                       + short-name alias shim)
```

There are NO `RIGHT_UP/`, `LEFT_UP/`, `RIGHT_UP_TOP/`, `LEFT_UP_TOP/`
folders inside this package. Camera-centric assumptions, legacy
`DoorProcessor` / `DamageProcessor` wrappers, `cv2.VideoCapture` calls
downstream of Stage 1, and mini-mp4 reconstruction are all removed.

## Output per batch

```
batch_outputs/<batch_key>/
├── downloads/                         raw videos (downloaded from S3, or
│                                       local-passthrough)
├── global_state/
│   ├── global_train_state.json
│   ├── per_camera_tracking.json
│   └── processed_videos/              wagon_count's debug tracking videos
│       └── <CAM>_processed.mp4         (4 cameras)
├── wagon_cache/
│   ├── GW_1/{right_up,left_up,right_up_top,left_up_top}/frame_*.jpg
│   ├── GW_2/...
│   └── ...
├── wagon_states/
│   ├── door/GW_*.json
│   ├── load/GW_*.json
│   ├── damage/GW_*.json
│   ├── ocr/GW_*.json
│   └── unified/GW_*.json
├── evidence/                          best-frame snapshots + metadata
│   ├── GW_1/door/{left_best,left_crop,right_best,right_crop}.jpg
│   │       + metadata.json
│   ├── GW_1/damage/{track_1,track_1_crop,...}.jpg + metadata.json
│   ├── GW_1/ocr/{best_frame,number_crop}.jpg + metadata.json
│   ├── GW_1/load/best_frame.jpg + metadata.json
│   └── GW_2/...
├── processed_videos/                  rich feature-overlay videos
│   └── <CAM>_processed.mp4             (4 cameras; rendered from
│                                        state+evidence — no detector rerun)
├── reports/
│   ├── combined_train_report.json     schema v4 (legacy_view_model +
│   │                                   evidence_pages)
│   ├── combined_train_report.pdf      legacy WagonEye visual identity:
│   │                                   title banner + KPIs + wagon table +
│   │                                   Damaged Wagon Report
│   ├── right_up_report.pdf            camera-wise reports (Stage 5a):
│   ├── left_up_report.pdf              one per camera, each scoped to
│   ├── right_up_top_report.pdf         that camera's authority
│   └── left_up_top_report.pdf
└── archive/                           run logs (future)
```

## Deployment

For a production **EC2** install (one-command setup script, systemd service,
continuous S3 polling, monitoring, reboot-safe restart), see
[DEPLOYMENT.md](DEPLOYMENT.md). The pipeline is EC2-native: it auto-detects its
own project root and needs no SageMaker/Jupyter runtime. Every path and
operational setting is configurable via `WAGONEYE_*` environment variables
(see [Configuration](#configuration-environment-variables) below and
`deploy/wagon-eye.env.example`).

## Quick start

```bash
# 1) Install dependencies (or run scripts/setup_ec2.sh on a fresh EC2 box)
pip install -r requirements.txt

# 2) Drop the 8 .pt model files into:
#       wagon_eye_v4/models/reconstruction/
#           right_up_gap.pt   (or right_up_wagon_gap.pt — both work)
#           left_up_gap.pt    (or left_up_wagon_gap.pt — both work)
#           top_gap.pt
#           side_classification.pt
#       wagon_eye_v4/models/features/
#           door_state.pt
#           loaded.pt
#           damage.pt
#           wagon_id_counting.pt

# 3) Local single-batch (no S3):
mkdir -p wagon_eye_v4/local_inputs
# copy 4 trimmed train videos in -- filenames must contain
# 'right_up' / 'left_up' / 'right_up_top' / 'left_up_top'
cd wagon_eye_v4
python -m orchestrator.master_runner --local-only --local-inputs ./local_inputs

# 4) Continuous S3 polling (production):
python -m orchestrator.master_runner --auto

# 5) Single-batch replay:
python -m orchestrator.master_runner --batch 20260408_032134
```

## Configuration (environment variables)

Every filesystem path and operational setting is centralized in
`core/config.py` and `core/constants.py`, and each is overridable via an
environment variable. **All defaults reproduce the original behaviour**, so an
unset environment behaves exactly as before. No module hardcodes an absolute
path — `PROJECT_ROOT` is auto-detected from the source tree.

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAGONEYE_WORKSPACE_ROOT`     | `<repo>/batch_outputs`        | Per-batch output root. |
| `WAGONEYE_MODELS_DIR`         | `<repo>/models`               | Models root. |
| `WAGONEYE_RECON_MODELS_DIR`   | `<models>/reconstruction`     | Stage-1 `.pt` files. |
| `WAGONEYE_FEAT_MODELS_DIR`    | `<models>/features`           | Stage-3 `.pt` files. |
| `WAGONEYE_LOCAL_INPUTS_DIR`   | `<repo>/local_inputs`         | `--local-only` scan folder. |
| `WAGONEYE_LOG_DIR`            | `<repo>/logs`                 | Rotating log directory. |
| `WAGONEYE_LOG_LEVEL`         | `INFO`                        | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |
| `WAGONEYE_DEVICE`            | auto (CUDA if available)      | Force `cuda` / `cpu`. |
| `WAGONEYE_S3_REGION`         | `ap-south-1`                  | AWS region. |
| `WAGONEYE_S3_OUTPUT_BUCKET`  | `biro-wagon-report-biro-copy` | Output/archive bucket. |
| `WAGONEYE_S3_INPUT_BUCKET`   | = output bucket               | Source-video bucket (`--auto`). |
| `WAGONEYE_S3_INPUT_PREFIXES` | *(empty)*                     | Comma-sep prefixes to poll (`--auto`). |
| `WAGONEYE_S3_TRAIN_BATCH_PREFIX` | `train_batch`             | S3 key prefix for outputs. |
| `WAGONEYE_S3_STATE_KEY`      | `master_runner/processed_batches.json` | Processed-batch state file. |
| `WAGONEYE_UPLOAD_API_URL`    | *(prod URL)*                  | PDF microservice endpoint. |
| `WAGONEYE_EMAIL_API_URL`     | *(prod URL)*                  | Email microservice endpoint. |
| `WAGONEYE_EMAIL_RECEIVER`    | *(prod list)*                 | Comma-sep TO recipients. |
| `WAGONEYE_EMAIL_RECEIVER_CC` | *(prod list)*                 | Comma-sep CC recipients. |

AWS credentials use boto3's default chain — an **EC2 IAM instance role** is
picked up automatically; no keys in the repo.

### Logging

All stage-boundary output goes through the `logging` module (configured once in
`core/logging_setup.py`): a rotating file at `<LOG_DIR>/wagon_eye.log`
(50 MB × 10) plus stdout, with timestamps, levels, and per-stage
start/duration/status lines. `--auto` traps `SIGTERM`/`SIGINT` and finishes the
current batch before exiting (graceful `systemctl stop`).

## Run-mode flags

| Flag                        | Effect                                                |
|-----------------------------|-------------------------------------------------------|
| `--auto`                    | Continuous S3 polling.                                |
| `--once`                    | Process one batch then exit.                          |
| `--batch <key>`             | Replay / debug a specific batch_key.                  |
| `--local-only`              | Skip S3; use `--local-inputs` instead.                |
| `--local-inputs DIR`        | Folder to scan for the 4 videos.                      |
| `--workspace DIR`           | Output root (default: `./batch_outputs`).             |
| `--recon-models-dir DIR`    | Override `models/reconstruction/`.                    |
| `--feat-models-dir DIR`     | Override `models/features/`.                          |
| `--skip-upload`             | Don't upload PDF/JSON, don't archive to S3.           |
| `--skip-email`              | Don't send the combined email.                        |
| `--poll-interval N`         | Continuous-mode S3 poll interval (seconds).           |
| `--partial-wait N`          | Wait this many minutes for missing cameras.           |
| `--disable-features LIST`   | Comma-separated feature keys to turn OFF (`door,ocr,load,damage`); skips the interactive prompt. |
| `--no-interactive`          | Never prompt for feature config (force all features ON unless `--disable-features` is given). |

## Feature configuration (enable/disable)

Before Stage 3 runs, you can choose which feature processors execute. The
feature set is **registry-driven** (`core/feature_config.py`) — adding a
feature there makes it appear here automatically.

Three ways to configure, in precedence order:

1. **CLI (non-interactive, scriptable):**
   ```bash
   python -m orchestrator.master_runner --local-only --local-inputs ./local_inputs \
          --disable-features ocr,damage
   ```
2. **Interactive prompt** — shown automatically on foreground runs
   (`--local-only` / `--once` / `--batch`) **only when stdin is a real
   terminal**:
   ```
   Current Feature Configuration:
     [ON]  Door
     [ON]  OCR
     [ON]  Load
     [ON]  Damage
   Turn OFF any feature? (y/n): y

   Select feature(s) to turn OFF (comma-separated numbers, e.g. 2,4):
     1. Door
     2. OCR
     3. Load
     4. Damage
   Disable: 2,4

   Final Feature Configuration:
     [ON]  Door
     [OFF] OCR
     [ON]  Load
     [OFF] Damage
   ```
3. **Default** — every feature ON. Continuous `--auto` polling, piped, and
   cron runs **never prompt** (safe for unattended operation); pass
   `--disable-features` to change the set there.

A disabled feature is skipped in Stage 3 and its per-wagon state is marked
`DISABLED_BY_USER`. In the camera-wise and combined reports its fields read
**`DISABLED BY USER`** (never flagged as an anomaly), and its overlay is
never drawn in the processed videos.

## Failure handling

| Failure                                  | Outcome                                            |
|------------------------------------------|----------------------------------------------------|
| Stage 1 reconstruction errors / 0 wagons | Batch marked `failed_no_global_state`. Abort.      |
| Stage 2 cannot open one video            | That camera's wagon_cache subtree empty. Continue. Batch is `completed_partial`. |
| Stage 3 one feature processor crashes    | Its per-wagon JSONs marked `FAILED`. Other features continue. |
| Stage 3 one wagon fails in one feature   | That `(feature, GW_n)` JSON marked `FAILED`. Rest unaffected. |
| Stage 4 fusion error per wagon           | Unified state for that wagon partial. Report still generated. |
| Stage 4b overlay renderer fails per cam  | That camera's mp4 missing; other cameras continue. Combined PDF still generated. |
| Stage 5a one camera report crashes       | That camera's PDF missing; other 3 camera reports + combined report unaffected. |
| Stage 5b combined PDF crashes            | Batch marked `report_failed`. JSON still written.  |
| PDF microservice down                    | S3 direct-upload fallback URL used.                |
| Email API down                           | Logged; batch outcome still persisted.             |

## Authority rules (fusion)

| Field                  | Authority                       |
|------------------------|---------------------------------|
| classification         | GlobalTrainState (RIGHT_UP)     |
| wagon_identifier (OCR) | RIGHT_UP only                   |
| right_door             | RIGHT_UP                        |
| left_door              | LEFT_UP                         |
| load_status            | RIGHT_UP_TOP (LEFT_UP_TOP fall) |
| top_damage             | any TOP camera reporting DAMAGE |

## Constraints honored

- ❌ No `DoorProcessor.process_video()` or `DamageProcessor.process_video()`
  wrappers anywhere.
- ❌ No mini-mp4 reconstruction.
- ❌ No `cv2.VideoCapture` for **inference** outside Stage 2 (materializer)
  and Stage 1 (wagon_count subprocess).  Stage 4b's overlay renderer uses
  `cv2.VideoCapture` strictly for **visualization** — it never invokes any
  detector / YOLO / OCR / tracking model; everything it draws comes from
  already-persisted `GlobalTrainState`, `UnifiedWagonState`, evidence
  metadata, and `per_camera_tracking.json`.
- ❌ No per-camera folders (`RIGHT_UP/` etc.) in this package.
- ❌ No `wagon_gap.pt` recompute downstream of Stage 1.
- ✅ All models in a single centralized tree (`models/{reconstruction,features}/`).
- ✅ Frames extracted exactly once.
- ✅ GlobalTrainState is the immutable backbone.

---

## Asynchronous camera arrival (incremental lifecycle)

The four camera videos for one train do **not** arrive in S3 together. Rather than
wait for all four (and abandon partial batches), the `--auto` scheduler manages a
persistent **BatchManifest** per train and processes cameras incrementally:

```
asynchronous camera arrival
  → persistent manifest (batch_outputs/<key>/manifest.json, mirrored to S3)
  → wait for RIGHT_UP master
  → short support-fusion window (armed when RIGHT_UP arrives)
  → SEAL immutable GlobalTrainState (RIGHT_UP + present support)
  → process available cameras (per-camera features + fusion + interim reports)
  → attach late cameras incrementally (no reseal, no wagon renumber)
  → finalize when complete OR final-camera deadline expires
  → ONE upload + ONE email
```

**The sealed GlobalTrainState is immutable.** Once sealed, total wagon count,
`GW_n` ids, time boundaries, ordering, and classification never change. Late
cameras only attach door/OCR/load/damage features to the existing wagons.

### Lifecycle states
`DISCOVERED → COLLECTING_CAMERAS → WAITING_FOR_MASTER → WAITING_FOR_SUPPORT →
RECONSTRUCTING → GLOBAL_STATE_SEALED → PROCESSING_AVAILABLE_CAMERAS →
WAITING_FOR_LATE_CAMERAS → PROCESSING_LATE_CAMERA → FINALIZING →`
terminal: `COMPLETED | COMPLETED_PARTIAL | FAILED_NO_GLOBAL_STATE | REPORT_FAILED | FAILED`.
Only terminal states are written to `master_runner/processed_batches.json`; every
non-terminal batch lives on as an active manifest and is revisited each poll.

### Persistence & markers
| Artifact | Location | Purpose |
|---|---|---|
| Batch manifest | `batch_outputs/<key>/manifest.json` (+ `s3://<out>/train_batch/<key>/manifest.json`) | resumable lifecycle state; schema-versioned, atomic writes |
| Materialization marker | `wagon_cache/.materialized/<CAMERA>.json` | skip re-extraction; keyed on ETag + GST version + materializer schema |
| Feature marker | `wagon_states/.features/<CAMERA>/<feature>.json` | skip re-inference; keyed on ETag + GST version + model SHA-256 + processor schema + threshold hash |
| Finalization marker | `delivery/finalization.json` | idempotent one-upload/one-email; records report hashes, upload URLs, email status + idempotency key |

### Camera-scoped layout
```
wagon_states/{door/RIGHT_UP,door/LEFT_UP,ocr/RIGHT_UP,
              load/RIGHT_UP_TOP,load/LEFT_UP_TOP,
              damage/RIGHT_UP_TOP,damage/LEFT_UP_TOP}/GW_n.json
wagon_states/unified/GW_n.json          (Stage 4 fusion, atomic + idempotent)
evidence/<GW_n>/<feature>/<CAMERA_ID>/...
```
A camera only ever writes inside its own namespace, so a late camera can never
overwrite another camera's results or evidence. New batches write **only** this
layout; legacy flat batches (`wagon_states/<feature>/GW_n.json`) remain readable
read-only (a new-schema batch never falls back to a stale flat file).

### Authority rules (unchanged)
classification ← sealed GlobalTrainState · wagon_identifier ← RIGHT_UP OCR ·
right_door ← RIGHT_UP · left_door ← LEFT_UP · load_status ← RIGHT_UP_TOP (else
LEFT_UP_TOP fallback) · top_damage ← any top camera with confirmed DAMAGE.
Missing/pending camera fields are reported as `PENDING_CAMERA` /
`CAMERA_MISSING_FINAL`, never a false OK. Disabled features stay `DISABLED_BY_USER`
and never raise an anomaly.

### Report revisions, complete vs partial
Reports carry `report_meta`: `report_revision`, `report_status`
(`INTERIM | FINAL | FINAL_PARTIAL`), `cameras_present/pending/missing_final`,
`generated_from_global_state_version/_hash`, `fusion_revision`, `partial_reason`.
All four cameras present at closure → `COMPLETED` / `FINAL`. One or more absent at
the final deadline → `COMPLETED_PARTIAL` / `FINAL_PARTIAL` (the PDF shows the
existing partial-report banner and `NO DATA` for permanently-missing-camera fields).

### Restart recovery, duplicates, ETag
- **Restart** at any point resumes from the manifest + on-disk markers without
  repeating completed reconstruction, materialization, feature inference, upload,
  or email.
- **Duplicate poll / repeat S3 notification** → no repeated work (markers).
- **ETag change** on a camera object → only that camera's cache/features/evidence
  are rebuilt (temp-build + atomic swap; a failed rebuild preserves the previous).
- **Ambiguous timestamp match** (a video within tolerance of two active batches
  and too close to decide) is held in `videos_for_review`, not silently attached.

### Terminal late-camera behaviour
A camera that arrives after a batch is terminal is logged and **ignored**
(`WAGONEYE_LATE_CAMERA_POLICY=IGNORE`, the only supported policy). Terminal
batches are never reopened.

### LEFT_UP fallback limitation
`WAGONEYE_ENABLE_LEFT_UP_FALLBACK_MASTER` is **false by default and experimental**:
`side_classification.pt` is a RIGHT_UP-trained model and is unvalidated on LEFT_UP.
With it off, a train that never gets a RIGHT_UP master fails safely as
`FAILED_NO_GLOBAL_STATE` rather than sealing an unvalidated timeline.

### Model-unavailable behaviour
If a feature `.pt` model is missing, that feature emits `NO_DATA` per wagon (the
pipeline still runs, seals, fuses, and reports); the lifecycle, layout, markers,
and delivery are unaffected.

## Troubleshooting

| Symptom | Cause / action |
|---|---|
| Batch stuck in `WAITING_FOR_MASTER` | RIGHT_UP hasn't been discovered. Check `WAGONEYE_S3_INPUT_PREFIXES` and that the RIGHT_UP filename contains `right_up` + a `YYYYMMDD_HHMMSS` timestamp. Closes `FAILED_NO_GLOBAL_STATE` after `FINAL_CAMERA_WAIT_MINUTES`. |
| Batch stuck in `WAITING_FOR_SUPPORT` | RIGHT_UP present, support window still open. It seals automatically after `SUPPORT_FUSION_WAIT_MINUTES` even with no support camera. |
| Camera in S3 but not attached | Filename lacks the camera substring or the `YYYYMMDD_HHMMSS` timestamp, or its timestamp is outside the 120 s clustering tolerance of the batch. |
| Ambiguous timestamp held for review | Video within tolerance of two active batches and closer than the ambiguity threshold — see `videos_for_review` in the manifest; attach manually or re-upload with a clearer timestamp. |
| ETag replacement detected | A camera object was replaced; only that camera's derived artifacts rebuild (log line `ETag changed ... rebuild that camera`). |
| Feature marker invalidated | A model file or threshold changed → the feature re-runs for that camera (marker keys include model SHA-256 + threshold hash). Expected after a model swap. |
| Final partial report missing a camera | That camera never arrived by the final deadline → `cameras_missing_final` in `report_meta`; its fields show `NO DATA`. |
| Duplicate-email concern after restart | Email is guarded by `delivery/finalization.json` (`email_sent` + idempotency key). Exactly-once is best-effort across a crash in the narrow window between the API returning 200 and the marker being persisted — that window may cause one resend. |
| Terminal batch received a late camera | Logged and ignored by design; the sealed report is never reopened. |
