# WagonEye — S3 Storage Architecture Migration Report

**Scope:** storage flow only. No ML / reconstruction / feature / fusion /
reporting logic changed. Every bucket name and prefix is now config-driven; the
automatic pipeline reads from `complete-train` and writes everything to
`end-results`.

## Target architecture

| Concern | Bucket | Layout |
|---|---|---|
| Raw CCTV (extraction input) | `biro-wagon-raw-video-copy` | `<camera_folder>/…` (unchanged) |
| Trimmed complete trains (extraction output = inspection input) | **`complete-train`** | `camera_CCTV_HZBN_DHN_{1_LEFT_UP,2_RIGHT_UP,5_RIGHT_TOP,6_LEFT_TOP}/` |
| Inspection outputs (everything) | **`end-results`** | `reports/` · `dashboard/` · `archive/` · `processed_batches.json` |

`master_runner --auto` reads **only** `complete-train`; it never reads the raw bucket.

---

## 1. Every modified file

**Code (8):**
| File | What changed |
|---|---|
| `core/constants.py` | New buckets `S3_RAW_BUCKET`/`S3_TRIMMED_BUCKET`; `S3_OUTPUT_BUCKET`→`end-results`; `S3_INPUT_BUCKET`→trimmed; default `S3_INPUT_PREFIXES`=4 camera folders; new `S3_REPORTS_PREFIX`/`S3_DASHBOARD_PREFIX`/`S3_ARCHIVE_PREFIX`; `S3_STATE_KEY`→`processed_batches.json`; `S3_TRAIN_BATCH_PREFIX` deprecated (unused). |
| `core/config.py` | Manifest-prefix comment → archive; startup summary prints raw/trimmed/output buckets + the three output prefixes + state key. |
| `delivery/s3_upload.py` | Report PDF/JSON keys → `reports/<batch_key>/<file>`; batch-tree upload base → `archive/<batch_key>/…`; docstring. |
| `delivery/dashboard_ingest.py` | Dashboard payload bucket default → `S3_OUTPUT_BUCKET`; per-camera JSON key → `dashboard/<folder>/<date>/…`; evidence URL builder → `archive/<key>/evidence/…`; reuse=False evidence-copy key → `dashboard/…`. |
| `orchestrator/batch_manifest.py` | Manifest S3 key → `archive/<key>/manifest.json` (via `MANIFEST_S3_PREFIX or S3_ARCHIVE_PREFIX`); active-manifest listing prefix → `archive/`; docstring. |
| `orchestrator/master_runner.py` | Deterministic processed-video URL → `archive/<key>/processed_videos/…`; state file is `S3_OUTPUT_BUCKET/S3_STATE_KEY` = `end-results/processed_batches.json`. |
| `train_extraction/driver.py` | Per-camera trimmed bucket default → `complete-train/<camera_folder>` (raw bucket unchanged). |
| `tests/test_dashboard_ingest.py` | Bucket assertion updated to the new default (`C.S3_OUTPUT_BUCKET`). |

**Deploy + docs (12):** `deploy/wagon-eye.env.example`, `deploy/wagon-eye-extraction.env.example`, `README.md`, `DEPLOYMENT.md`, `complate_workflow.txt`, `train_extraction/README.md`, `docs/AUTO_PIPELINE_ARCHITECTURE.md`, `docs/WAGONEYE_V5_ARCHITECTURE.md`, `docs/EC2_DEPLOYMENT_CHECKLIST.md`, `docs/WAGONEYE_V5_VALIDATION_CHECKLIST.md`, `docs/WAGONEYE_V5_PRODUCTION_SIGNOFF.md`, and this report.

**Not changed (confirmed clean):** `deploy/*.service`, `scripts/setup_ec2.sh` (no hardcoded buckets — they use `EnvironmentFile`), and all ML/feature/fusion/reporting/reconstruction/materializer modules.

---

## 2. Every bucket path changed

| Old | New | Where |
|---|---|---|
| `biro-wagon-report-biro-copy` (output) | `end-results` | `S3_OUTPUT_BUCKET` default (constants) |
| `biro-wagon-pre-processed-video-copy` (trimmed/input) | `complete-train` | `S3_TRIMMED_BUCKET`/`S3_INPUT_BUCKET` (constants), `train_extraction/driver.py`, extraction env example |
| `ankit-version-1-prod` (dashboard JSON) | `end-results` (`dashboard/` prefix) | `dashboard_ingest._inspection_bucket()` default |
| `biro-wagon-raw-video-copy` (raw) | **unchanged** (now explicit `S3_RAW_BUCKET`) | constants + driver |

---

## 3. Every S3 prefix changed

| Old prefix | New prefix |
|---|---|
| `train_batch/<key>/reports/` | `reports/<key>/` |
| `train_batch/<key>/<sub>/` (tree/evidence/processed_videos/global_state/wagon_states) | `archive/<key>/<sub>/` |
| `train_batch/<key>/manifest.json` | `archive/<key>/manifest.json` |
| `<camera_folder>/<date>/…_inspection.json` (dashboard) | `dashboard/<camera_folder>/<date>/…_inspection.json` |
| `master_runner/processed_batches.json` (state) | `processed_batches.json` (output-bucket root) |

---

## 4. Every upload destination (after migration)

| Artifact | Destination |
|---|---|
| Combined PDF | `end-results/reports/<batch_key>/combined_train_report.pdf` (microservice first, S3 fallback) |
| Camera PDFs | `end-results/reports/<batch_key>/<camera>_report.pdf` |
| Combined JSON | `end-results/reports/<batch_key>/combined_train_report.json` |
| Batch tree (global_state, wagon_states, evidence, processed_videos, reports copy, metadata) | `end-results/archive/<batch_key>/<sub>/…` |
| Batch manifest | `end-results/archive/<batch_key>/manifest.json` |
| Per-camera dashboard JSON | `end-results/dashboard/<camera_folder>/<date>/<raw>_inspection.json` (+ POST to ingest API) |
| Processed-batches state | `end-results/processed_batches.json` |
| Trimmed train clip (extraction) | `complete-train/<camera_folder>/<raw>_train.mp4` |

> Note (storage correctness, not a logic change): report PDFs now key by
> **basename** under `reports/<key>/`, so the combined and the four camera PDFs
> land at distinct keys (previously the S3-fallback path used one fixed key for
> all). Report **content** is unchanged.

> `wagon_cache` JPEGs are **not** uploaded (regenerable, hundreds of MB/train) —
> the archive relocates the *existing* archived set. To retain the full cache,
> add `CFG.DIR_WAGON_CACHE` to the `upload_tree` list in
> `orchestrator/lifecycle_runner.py` (one line) — call this out to ops before
> enabling due to volume.

---

## 5. Every download source (after migration)

| Read | Source |
|---|---|
| `--auto` source-video discovery | `complete-train` under the 4 camera prefixes (`train_batch_manager`, `S3_INPUT_BUCKET`/`S3_INPUT_PREFIXES`) |
| Trimmed clip download | `complete-train/<camera_folder>/<clip>` |
| Active-manifest resume (list) | `end-results/archive/*/manifest.json` |
| Per-batch manifest load | `end-results/archive/<key>/manifest.json` |
| Processed-batches state | `end-results/processed_batches.json` |
| Raw clip (extraction only) | `biro-wagon-raw-video-copy/<camera_folder>/…` |
| Models | `s3://wagon-eye-models/…` (unchanged) |

---

## 6. Every environment variable

**New:**
| Var | Default |
|---|---|
| `WAGONEYE_S3_RAW_BUCKET` | `biro-wagon-raw-video-copy` |
| `WAGONEYE_S3_TRIMMED_BUCKET` | `complete-train` |
| `WAGONEYE_S3_REPORTS_PREFIX` | `reports` |
| `WAGONEYE_S3_DASHBOARD_PREFIX` | `dashboard` |
| `WAGONEYE_S3_ARCHIVE_PREFIX` | `archive` |

**Changed default:**
| Var | Old default | New default |
|---|---|---|
| `WAGONEYE_S3_OUTPUT_BUCKET` | `biro-wagon-report-biro-copy` | `end-results` |
| `WAGONEYE_S3_INPUT_BUCKET` | = output bucket | `complete-train` (= trimmed) |
| `WAGONEYE_S3_INPUT_PREFIXES` | *(empty)* | the 4 camera folders |
| `WAGONEYE_S3_STATE_KEY` | `master_runner/processed_batches.json` | `processed_batches.json` |
| `WAGONEYE_INSPECTION_JSON_BUCKET` | `ankit-version-1-prod` | `end-results` |

**Unchanged / reference:** `WAGONEYE_S3_REGION`, `WAGONEYE_MANIFEST_S3_PREFIX`
(empty → falls back to `archive`), `WAGONEYE_INSPECTION_VERSION`, the extraction
per-camera `WAGONEYE_EXTRACTION_<CAM>_{RAW,TRIMMED}_BUCKET` overrides.

**Deprecated (no code path uses it):** `WAGONEYE_S3_TRAIN_BATCH_PREFIX`.

---

## 7. Complete end-to-end data flow (after migration)

```
biro-wagon-raw-video-copy/<camera_folder>/*.mp4
        │  train_extraction.run_extraction_service  (classify → trim → upload)
        ▼
complete-train/<camera_folder>/<raw>_train.mp4
        │  orchestrator.master_runner --auto  (polls complete-train ONLY)
        │    ├─ processed_batches.json guard  ←→  end-results/processed_batches.json
        │    └─ manifest resume/list          ←→  end-results/archive/<key>/manifest.json
        ▼
   GlobalTrain pipeline (reconstruct → materialize → features → fuse → render → report)
        │
        ├─ end-results/reports/<batch_key>/{combined_train_report.pdf,.json, <cam>_report.pdf}
        ├─ end-results/archive/<batch_key>/{global_state,wagon_states,evidence,processed_videos,reports,manifest.json}
        ├─ end-results/dashboard/<camera_folder>/<date>/<raw>_inspection.json  → POST ingest API
        ├─ email (one per batch; subject includes loco numbers)
        └─ end-results/processed_batches.json  (batch marked terminal → never reprocessed)
```

---

## 8. Verification performed
- All 8 changed code modules byte-compile.
- Config resolves: `RAW=biro-wagon-raw-video-copy`, `TRIMMED=INPUT=complete-train`,
  `OUTPUT=end-results`, prefixes `reports/dashboard/archive`, `state=processed_batches.json`,
  4 input prefixes; sample keys — manifest `archive/<key>/manifest.json`, evidence
  `https://end-results.s3…/archive/<key>/evidence/…`.
- **No retired strings remain** anywhere (`biro-wagon-pre-processed-video-copy`,
  `biro-wagon-report-biro-copy`, `ankit-version-1-prod`), and no stale
  `train_batch/…` or `master_runner/processed_batches.json` layout references in
  code or docs (the only surviving `train_batch` mention is the intentional
  DEPRECATED note in `constants.py`).
- `tests/test_dashboard_ingest.py` (storage feed): **17/17 pass** with the new layout.

## 9. Out-of-scope note (transparency)
Running the full suite surfaces **4 failures in `tests/test_camera_isolation.py`
and `tests/test_incremental_lifecycle.py`** that are **unrelated to this storage
migration.** They assert the *pre-side-damage* feature registry
(`features_for_camera(RIGHT_UP) == ["door","ocr"]`) and the removed
loaded-wagon floor-damage filter — i.e. they encode v4 behaviour that the
earlier production-behaviour phase intentionally changed (damage now also runs on
side cameras; the floor filter was dropped to match production). They fail on
this code regardless of storage. They should be updated to the intended
behaviour as a separate, clearly-scoped test-alignment task.
