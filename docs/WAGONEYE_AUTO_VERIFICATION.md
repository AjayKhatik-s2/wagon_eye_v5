# WagonEye v5 — AUTO Pipeline Production Verification

**Objective:** verify the complete AUTO flow (raw video → end-results) works with no manual intervention.
**Constraint honoured:** no optimization, no refactor, no model/reconstruction/feature/report logic changes. No code was modified.
**Method:** full code trace grounded against the on-disk repo, the prior run artifacts, and the environment. Every claim cites `file:line`.

---

## 0. What could and could not be executed here (honesty statement)

This box **cannot run AUTO end-to-end**, so every runtime PASS/FAIL below is a **code-trace verdict**, not an observed execution, unless explicitly marked "observed."

Environment facts (verified this session):

| Fact | Evidence |
|---|---|
| `boto3`, `ultralytics`, `cv2` importable | ran `python -c import` — all OK |
| `models/reconstruction/` complete: `left_up_gap.pt`, `right_up_gap.pt`, `side_classification.pt`, **`top_gap.pt`** | `ls models/reconstruction/` |
| **`models/production/` is EMPTY (README.md only)** | `ls models/production/` |
| `models/features/` has v4-native `.pt` (unused by production processors) | `ls models/features/` |
| **AUTO has never run**: log is 3× `mode:local`, 0× auto; no `[ORCH]`/`[LIFECYCLE]` lines | `grep` on `logs/wagon_eye.log` |
| **No AUTO artifacts anywhere**: no `manifest.json`, no `wagon_states/.features`, no `wagon_cache/.materialized`, no `delivery/` in either prior batch | `ls` on both `batch_outputs/*` |
| A prior LOCAL run completed fully: 62 wagons, 4 camera PDFs + combined PDF/JSON, 4 processed videos | `find batch_outputs/20260719_113220` |
| That run's features were all NO_DATA (production models absent), `total_wagons=62`, 58 WAGON / 0 ENGINE / 0 BRAKE / **4 UNKNOWN**, last wagon `GW_62` class=UNKNOWN | parsed `global_train_state.json` + `unified/GW_62.json` |
| wagon_states are **camera-scoped** (`door/RIGHT_UP`, `door/LEFT_UP`) even in LOCAL | `ls wagon_states/door/` |

**Two consequences for this verification:**
1. Even if AUTO could run on this box, every Stage-3 feature would emit NO_DATA because `models/production/` is empty (`production_models.require()` raises `MissingProductionModel`, `core/production_models.py:94-99`). The pipeline would still seal, fuse, report — but with empty findings.
2. The AUTO orchestration layer (manifest lifecycle, per-camera markers, finalization marker, dashboard ingest) has produced **no artifact ever** on this machine. Its correctness is asserted only from code reading.

---

## TASK 1 — Complete AUTO execution flow

Two independent OS processes (`docs/AUTO_PIPELINE_ARCHITECTURE.md §2.1`), communicating only through S3.

```
┌─────────────────────────── SERVICE 1: EXTRACTION PRODUCER ───────────────────────────┐
 python -m train_extraction.run_extraction_service            run_extraction_service.py:138
   main() → loop every 60s → sweep_camera(cam) for cam in ALL_CAMERAS                    :165
     └ D.get_extractor(cam)                                    driver.py:177 → _build_extractor:115
         builds FrameClassifier(YOLO side/top_classification.pt) + TrainSegmentFinder
     └ keys = _list_raw_keys(ex, raw_bucket)                   run_extraction_service.py:86
         └ ex.s3.list_objects(raw_bucket)   ◄── ***BLOCKER E-1***  s3.py:53
     └ for key not in local ledger:  D.extract_trains(cam,key) driver.py:188
         └ TrainExtractor.extract()                            extractor.py:97
             ├ _download raw → temp                            :69
             ├ train_in_progress? _handle_ongoing (merge across clips) :153
             └ _handle_standalone → segment_finder.analyze()   :317,320
                 └ _emit_segments → trim_video → _upload        :119,137,140
                     └ s3.upload_file(path, "complete-train/<folder>", "<raw>_train.mp4")  s3.py:86
   raw basename carries YYYYMMDD_HHMMSS → survives into "<raw>_train.mp4"                 :131
└──────────────────────────────────────────────────────────────────────────────────────┘
                                          │ writes
                                          ▼
        complete-train/camera_CCTV_HZBN_DHN_{1_LEFT_UP,2_RIGHT_UP,5_RIGHT_TOP,6_LEFT_TOP}/<raw>_train.mp4
                                          │ polled by
                                          ▼
┌─────────────────────────── SERVICE 2: INSPECTION (--auto) ───────────────────────────┐
 python -m orchestrator.master_runner --auto                  master_runner.py:877 main()
   ├ setup_logging / startup_summary(mode=auto)                :883,895
   ├ validate_config(mode=auto) → return 2 if any error        :896  (needs S3_INPUT_PREFIXES, endpoints)
   ├ resolve_feature_config(interactive=False)                 :911
   └ run_auto()                                                :700
       ├ boto3.client("s3", ap-south-1)                        :723
       ├ state_loc = end-results/processed_batches.json        :724
       ├ processed = load_batch_state(s3, state_loc)           train_batch_manager.py:64
       └ WHILE not _SHUTDOWN_REQUESTED:                         :747
           (a) actives = list_active_manifests(s3, processed)  batch_manifest.py:318
                 LIST end-results/archive/ Delimiter=/ → GET each archive/<k>/manifest.json
           (b) for cv in list_candidate_videos(s3):            train_batch_manager.py:196
                 _list_input_objects → _camera_for_key + parse_train_timestamp  :130 / batch.py:75
                 _attach_candidate(cv, actives, processed)      master_runner.py:605
                   rank actives |Δt|≤120s; ambiguous(<5s)→review; terminal-in-range→ignore
                   else BatchManifest.new(); set_camera(); write_local+save_s3   :668,691,696
           (c) for key in sorted(actives): LR.advance(m, ctx)  lifecycle_runner.py:638
                   DISCOVERED→…→RECONSTRUCTING→GLOBAL_STATE_SEALED→PROCESSING_AVAILABLE
                   →(WAITING/PROCESSING_LATE)*→FINALIZING→terminal
           (d) if is_terminal: processed[key]=status; save_batch_state()  master_runner.py:764
           (e) time.sleep(poll_interval=60)                    :775
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Per-batch stage chain inside `advance()` (all in `lifecycle_runner.py`):

```
stage_seal (Stage 1)        :129 → reconstruction_runner.run → wagon_count subprocess
  seal: global_state_version = sha256(global_train_state.json)  :162
stage_process_cameras       :211
  Stage 2  wagon_cache_builder.build_cameras (per-camera, marker-gated)  :243
  Stage 3  _run_camera_features (door/ocr/load/damage, marker-gated)     :285
  Stage 4  wagon_state_builder.build (fusion)                            :260
stage_reports (interim+final):373
  Stage 4b feature_overlay_renderer.render_all_cameras   :409
  Stage 5a camera_reports.build_all                      :421
  Stage 5b combined_train_report.build                   :443
stage_finalize              :461
  upload_pdf/json + upload_tree (archive)                :531-551
  notification.send_email                                :577
  dashboard_ingest.run                                   :603
  _finish → terminal                                     :609
```

---

## TASK 2 — Raw video ingestion

| Question | Answer | Ref |
|---|---|---|
| Raw bucket | `biro-wagon-raw-video-copy` (`S3_RAW_BUCKET`) | `constants.py:180`; per-camera in `driver.py:66-89` |
| Watcher | `run_extraction_service.sweep_camera()` polling loop | `:94,165` |
| Extractor | `TrainExtractor.extract()` via `driver.extract_trains()` | `extractor.py:97`, `driver.py:188` |
| Uploads trimmed | `extractor._upload → S3Client.upload_file` | `extractor.py:75`, `s3.py:86` |
| Filename | `f"{first_video_name}_train.mp4"` (single); `_train_part{idx}`; `_train_incomplete` | `extractor.py:131,134,255` |
| Timestamp | inherited from raw basename via the `_train.mp4` suffix; parsed downstream by `parse_train_timestamp` regex `(\d{8}_\d{6})` | `batch.py:72` |
| Camera name | not in filename; encoded by the **trimmed folder** the clip is uploaded into (`_CAMERA_CONFIG[cam]["trimmed"]`) | `driver.py:66-89` |
| Duplicate prevention | per-camera local ledger `logs/extraction_state/processed_<cam>.json`; key added only after successful extract | `run_extraction_service.py:63,124-126` |
| Failure handling | model missing → sweep logs, `errors+=1`, camera skipped; per-key exception → logged, **not** added to ledger → retried every sweep | `:99,127-130` |

### ⛔ BLOCKER E-1 — extraction lists the entire raw bucket
```python
# run_extraction_service.py:88
objs = ex.s3.list_objects(raw_bucket)          # raw_bucket = "biro-...-copy/camera_CCTV_HZBN_DHN_2_RIGHT_UP"
# s3.py:53-61
def list_objects(self, bucket_string, prefix=""):
    bucket, _ = split_bucket_prefix(bucket_string)   # prefix parsed then DISCARDED
    params = {"Bucket": bucket, "Prefix": prefix}    # prefix == "" → whole bucket
```
The camera folder embedded in `raw_bucket` is dropped; `prefix` defaults to `""` and no caller passes it. `upload_file` (`s3.py:91`) *does* honour the embedded prefix — the read/write asymmetry is the defect. **Effect:** every one of the four cameras lists **all four cameras' raw clips**, extracts each with its own (wrong) classifier, and uploads into its own trimmed folder. Every raw clip is processed 4× and each trimmed folder is cross-contaminated with clips derived from other cameras' raw footage. This corrupts the four-camera grouping before inspection even begins. **This is the first functional defect in the raw→trimmed leg.**

---

## TASK 3 — complete-train ingestion

| Question | Answer | Ref |
|---|---|---|
| Discovery | `list_candidate_videos(s3)` → `_list_input_objects` paginates the 4 input prefixes | `train_batch_manager.py:196,155` |
| Camera parse | `_camera_for_key(FULL key, lowercased)`, TOP tokens first (`right_up_top`/`right_top`, `left_up_top`/`left_top`), then side | `:130,115,124` |
| Timestamp parse | `parse_train_timestamp(basename)` regex `(\d{8}_\d{6})` | `batch.py:75` |
| Batch key | first arriving camera's timestamp → `BatchManifest.new(batch_key=cv.train_timestamp)` | `master_runner.py:668` |
| Four-camera attach | `_attach_candidate` clusters by `|Δt|≤120s` and calls `target.set_camera(slot)`; RIGHT_UP arrival arms the support window | `master_runner.py:605,691`; `manifest.py:177` |
| Duplicate batch prevention | active manifests discovered by LISTING `archive/`; keys in `processed_batches` skipped; terminal manifests skipped | `batch_manifest.py:318,353-359` |
| processed_batches.json | `{batch_key → terminal status}` at `end-results/processed_batches.json`; written only for terminal batches | `master_runner.py:724,764`; `train_batch_manager.py:87` |

Decision points, all traced:
- **Ambiguous** (two batches within 120 s and their distances differ by <5 s) → `videos_for_review`, **not attached, not alerted** (`master_runner.py:645-656`).
- **Terminal in range** → ignored, no reopen (`:659-666`).
- **Same ETag** → no-op; **changed ETag** → drop that camera's materialize + feature markers, rebuild (`:676-684`).
- **Grouping fragility:** each camera's clip is named after its own first raw clip (`extractor.py:131`); if a train spans raw clips longer than the 120 s tolerance (`train_batch_manager.py:44`), the four clips cannot cluster and one train fragments into single-camera batches.

---

## TASK 4 — Complete processing (does every stage run automatically?)

| Stage | Function (AUTO) | Ref | Auto-runs? |
|---|---|---|---|
| 1 Reconstruction | `stage_seal → reconstruction_runner.run → wagon_count subprocess` | `lifecycle_runner.py:129`, `runner.py:93` | code-path YES |
| 2 Materialization | `stage_process_cameras → wagon_cache_builder.build_cameras` | `:243` | YES |
| 3 Door/OCR/Load/Damage | `_run_camera_features` → `{door,ocr,load,damage}.processor.run(cameras=[cam])` | `:285,324` | YES (NO_DATA if model absent) |
| 4 Fusion | `wagon_state_builder.build(camera_arrival, disabled_features, gst_version, fusion_revision)` | `:260` | YES |
| 4b Overlay | `feature_overlay_renderer.render_all_cameras` | `:409` | YES (caught on fail) |
| 5a Camera reports | `camera_reports.build_all` | `:421` | YES (caught on fail) |
| 5b Combined report | `combined_train_report.build` | `:443` | YES (**uncaught on fail — see Task 7**) |
| Dashboard | `dashboard_ingest.run` | `:603` | YES (fully swallowed) |
| Archive | `s3_upload.upload_tree(...)` for 5 subtrees | `:546-551` | YES (conditional on `can_deliver`) |
| Upload | `upload_pdf/json` | `:531-535` | YES (conditional) |

Every stage is wired to run automatically. The conditionality and failure isolation are detailed in Tasks 6–7.

---

## TASK 5 — Exact output locations (LOCAL → upload fn → S3)

All local paths are `batch_outputs/<batch_key>/…` (`config.py:94`, `DIR_*` `:112-119`). S3 bucket is `end-results` (`constants.py:182`). Prefixes: `reports`/`dashboard`/`archive` (`:195-197`). **Observed** = seen in the prior LOCAL batch tree.

```
Combined PDF
  LOCAL   batch_outputs/<key>/reports/combined_train_report.pdf        [observed]
  UPLOAD  s3_upload.upload_pdf   (microservice first, S3 PUT fallback)  s3_upload.py:77
  S3      end-results/reports/<key>/combined_train_report.pdf           :85

Combined JSON
  LOCAL   batch_outputs/<key>/reports/combined_train_report.json        [observed]
  UPLOAD  s3_upload.upload_json                                          s3_upload.py:97
  S3      end-results/reports/<key>/combined_train_report.json          :101

Camera PDFs ×4 (right_up, left_up, right_up_top, left_up_top _report.pdf)
  LOCAL   batch_outputs/<key>/reports/<cam>_report.pdf                  [observed]
  UPLOAD  s3_upload.upload_pdf (per camera)                             lifecycle_runner.py:536
  S3      end-results/reports/<key>/<cam>_report.pdf

Dashboard JSON (per present camera)
  LOCAL   batch_outputs/<key>/delivery/dashboard/<raw>_inspection.json  dashboard_ingest.py:739
  UPLOAD  s3_client.upload_file (inside dashboard_ingest)               :763
  S3      end-results/dashboard/<Folder>/<YYYY-MM-DD>/<raw>_inspection.json  :744

Evidence (per wagon/feature/camera)
  LOCAL   batch_outputs/<key>/evidence/<GW>/<feature>/<CAMERA>/{*.jpg,metadata.json}  [dir observed, empty—models absent]
  UPLOAD  s3_upload.upload_tree(sub_prefix="evidence")                  lifecycle_runner.py:546
  S3      end-results/archive/<key>/evidence/<GW>/<feature>/<CAMERA>/…  s3_upload.py:130

Processed videos ×4
  LOCAL   batch_outputs/<key>/processed_videos/<CAM>_processed.mp4      [observed]
  UPLOAD  s3_upload.upload_tree(sub_prefix="processed_videos")          lifecycle_runner.py:546
  S3      end-results/archive/<key>/processed_videos/<CAM>_processed.mp4

Manifest
  LOCAL   batch_outputs/<key>/manifest.json                             batch_manifest.py:244 [NOT produced by LOCAL]
  UPLOAD  batch_manifest.save_s3                                        :290
  S3      end-results/archive/<key>/manifest.json                       :286

Global state
  LOCAL   batch_outputs/<key>/global_state/global_train_state.json      [observed]
          + per_camera_tracking.json + stage1_wagon_count.log
  UPLOAD  s3_upload.upload_tree(sub_prefix="global_state", skip .jpg)   lifecycle_runner.py:546
  S3      end-results/archive/<key>/global_state/…

Unified wagon states
  LOCAL   batch_outputs/<key>/wagon_states/unified/<GW>.json            [observed: 62 files]
          + wagon_states/<feature>/<CAMERA>/<GW>.json + .features/ markers
  UPLOAD  s3_upload.upload_tree(sub_prefix="wagon_states")              lifecycle_runner.py:546
  S3      end-results/archive/<key>/wagon_states/…

Archive (whole tree, the 5 subtrees above)
  UPLOAD  s3_upload.upload_tree ×5                                      lifecycle_runner.py:546-551
  S3      end-results/archive/<key>/{global_state,wagon_states,reports,evidence,processed_videos}/…

processed_batches.json
  LOCAL   (none — S3-only)
  UPLOAD  train_batch_manager.save_batch_state                         :87
  S3      end-results/processed_batches.json  (bucket root)            master_runner.py:724
```
Note: `wagon_cache/` JPEGs are intentionally **not** uploaded (`docs/STORAGE_MIGRATION_REPORT.md §4`).

---

## TASK 6 — end-results bucket audit

| Path | Writer module | Upload fn | Conditional? | Order | Retried? |
|---|---|---|---|---|---|
| `reports/<key>/…` | `delivery/s3_upload` | `upload_pdf`/`upload_json` | yes: `can_deliver = not skip_upload and s3 is not None and terminal != FAILED` (`lifecycle_runner.py:526`) + `not already_uploaded` | in FINALIZING, **after** report build | microservice 3× internal; **not** re-tried once marked terminal (Task 11) |
| `archive/<key>/…` | `delivery/s3_upload.upload_tree` | `upload_tree` ×5 | same `can_deliver` gate | after reports, same block | per-file exceptions swallowed (`s3_upload.py:149`) |
| `archive/<key>/manifest.json` | `batch_manifest.save_s3` | `put_object` | every transition when `not skip_upload` (`lifecycle_runner.py:74`) | throughout | best-effort, non-fatal |
| `dashboard/…` | `delivery/dashboard_ingest` | `s3_client.upload_file` | `is_enabled()` (default true) + not already ingested | **after** archive tree (so evidence URLs resolve) | internal 3× on ≥500; failure recorded |
| `processed_batches.json` | `train_batch_manager.save_batch_state` | `put_object` | only when a batch reaches terminal | after finalize | failure logged; in-memory only → **re-emails on restart** |

Upload happens strictly **after** report generation within `stage_finalize` (`:471` builds reports, `:528` uploads). Dashboard runs **after** the archive tree upload so its `reuse=True` evidence URLs (`dashboard_ingest.py:290`) point at already-uploaded objects.

---

## TASK 7 — Why reports may not reach S3 (every path)

| # | Condition | Exact path | Outcome |
|---|---|---|---|
| 1 | `--skip-upload` set (EC2 guide Step 4 uses it) | `can_deliver` false (`lifecycle_runner.py:526`) | reports built locally, **never uploaded** |
| 2 | Combined report raises | `combined_train_report.build` bare in `stage_reports` (`:443`) → unwinds `stage_finalize`→`advance`→tick handler (`master_runner.py:779`) | no upload; batch stuck at `FINALIZING`, retried every tick |
| 3 | `report_pdf_path is None` | `terminal = REPORT_FAILED` (`:481`); PDF upload skipped, **JSON still uploaded** (`:533`), email suppressed (`:564`) | partial: JSON only |
| 4 | `s3_client is None` | `can_deliver` false | no upload |
| 5 | `terminal == FAILED` | `can_deliver` false (`:527`) | no upload |
| 6 | microservice down AND S3 PUT fallback throws | `upload_pdf` returns None (`s3_upload.py:92`) | PDF URL empty → **email skipped** (`notification.py:32`); archive tree may still upload |
| 7 | `upload_tree` per-file error | caught, logged, counted only (`s3_upload.py:149`) | **partial archive indistinguishable from complete** |
| 8 | Overlay / camera-report failure | caught (`lifecycle_runner.py:416,428`) | combined report + upload still proceed |
| 9 | `_SHUTDOWN_REQUESTED` before FINALIZING | loop exits between batches (`master_runner.py:747`) | batch resumes next start; not lost |
| 10 | `advance()` guard exhaustion (32) | returns non-terminal (`:742`) | no finalize this tick |
| 11 | AUTO never invoked / `validate_config` fails | `main` returns 2 (`master_runner.py:906`) | nothing runs |

---

## TASK 8 — Success-criteria checklist

Legend: **PASS** = code path correct and (where noted) observed; **FAIL** = defect proven in code; **NOT VERIFIED** = requires an execution environment absent here.

| Milestone | Verdict | Basis |
|---|---|---|
| ✔ Raw bucket | NOT VERIFIED | no S3 access; bucket wiring correct (`constants.py:180`) |
| ✔ Extraction | **FAIL** | E-1: whole-bucket listing, cross-contamination (`s3.py:54`) |
| ✔ complete-train upload | PARTIAL | upload keys correct (`s3.py:91`), but inputs wrong due to E-1 |
| ✔ Batch detection | NOT VERIFIED (code PASS) | `list_candidate_videos` correct given prefixes set + models present |
| ✔ Four-camera batch | NOT VERIFIED | grouping fragile (D-9); camera-token misattribution risk (`train_batch_manager.py:130`) |
| ✔ Reconstruction | NOT VERIFIED (models present here) | Stage 1 wiring PASS; `top_gap.pt` present |
| ✔ Materialization | NOT VERIFIED (code PASS) | `build_cameras` correct; **M-1** hides decode failure (`wagon_cache_builder.py:401`) |
| ✔ Feature inference | **degraded** | `models/production/` empty → all NO_DATA (`production_models.py:97`); **F-3** whole-camera failure isolation |
| ✔ Fusion | PASS (observed in LOCAL) | camera-scoped fusion produced 62 unified states |
| ✔ Overlay | NOT VERIFIED (code PASS, isolated) | `:409` |
| ✔ Camera reports | PASS (observed in LOCAL) | 4 PDFs on disk |
| ✔ Combined report | PASS (observed in LOCAL) | PDF+JSON on disk; **but uncaught in AUTO** (Task 7 #2) |
| ✔ Dashboard | NOT VERIFIED | never produced any artifact; posts to LIVE API by default (`dashboard_ingest.py:140`) |
| ✔ Archive upload | NOT VERIFIED | `upload_tree` wiring correct; **L-1** livelock if `--skip-upload` |
| ✔ Upload to end-results | NOT VERIFIED | conditional gate correct; **E-2** never retried after terminal |
| ✔ Terminal state | NOT VERIFIED | reachable on happy path; **L-2/L-3** duplicate-delivery on re-entry |

No milestone from **Batch detection onward has ever executed on this box** — those verdicts are code-trace only.

---

## TASK 11 — Where AUTO can terminate early / silently

- `validate_config` non-empty → exit 2 before polling (`master_runner.py:906`) — e.g. `S3_INPUT_PREFIXES` empty, or email endpoint missing without `--skip-email`.
- Manifest with a newer `manifest_schema_version` → `load_s3` returns None → batch invisible forever (`batch_manifest.py:222,307`).
- Ambiguous attach → `videos_for_review`, never processed, never alerted (`master_runner.py:645`).
- `--batch <key>` on a non-active key → `m is None: continue` → returns 0 (`master_runner.py:756-762`).
- `stage_seal` fail (master absent / exit≠0 / 0 wagons / 7200 s timeout) → `FAILED_NO_GLOBAL_STATE` terminal (`lifecycle_runner.py:160`, `runner.py:205-221`).
- Combined-report exception in AUTO → pinned at `FINALIZING`, infinite retry (Task 7 #2).
- Email/upload failure → **still marked terminal**, never retried (`lifecycle_runner.py:553,588,609`) → report notification permanently lost.
- `save_batch_state` failure → in-memory only; on restart the terminal batch re-runs and **re-emails**.

---

## Cross-mode defects that specifically break AUTO (not LOCAL)

| ID | Defect | Ref | AUTO effect |
|---|---|---|---|
| E-1 | extraction lists whole raw bucket | `s3.py:54` | 4× processing, cross-contaminated trimmed clips |
| L-1 | `_persist` mirrors to S3 only when `not skip_upload`, but `run_auto` rebuilds actives from S3 only | `lifecycle_runner.py:74` + `master_runner.py:749` | `--skip-upload` → Stage 1 re-runs every tick forever (livelock) |
| L-2 | `report_revision` incremented unconditionally → new idempotency key each finalize | `lifecycle_runner.py:438,488` | any FINALIZING re-entry → **duplicate email** to 10-recipient CC + duplicate upload |
| L-3 | combined-report exception uncaught in AUTO (caught in LOCAL) | `:443` vs `master_runner.py:472` | infinite finalize loop, compounding L-2 |
| L-4 | 5 PDFs × microservice 3× × (120 s timeout + 10 s sleep) inside one `advance()` | `s3_upload.py:54-69` | up to ~32 min blocking the single-threaded poll loop; other batches miss deadlines |
| M-1 | `build_cameras` swallows decode failure, caller ignores return, camera still marked materialized | `wagon_cache_builder.py:401` + `lifecycle_runner.py:249` | corrupt video → silent empty-cache partial report |
| F-3 | whole-camera feature try/except → one wagon's crash fails all wagons + re-runs forever | `lifecycle_runner.py:323-333` | batch never reaches terminal via that camera |
| E-2 | delivery failure marks batch terminal anyway, never retried | `:553,588,609` | transient outage permanently loses report/email |
| R-1 | trailing phantom segment counted as wagon (no trailing guard) | `global_alignment.py:333,348` | +1 wagon; **observed**: `GW_62`=UNKNOWN, 4 UNKNOWN of 62 |

---

## FINAL CONCLUSION

**Q1 — Does the AUTO pipeline currently work completely from raw video to end-results?**
**No — not as a hands-off flow, and it cannot be proven to on this box.** Two layers:

- *Raw → complete-train (Service 1):* **broken.** E-1 (`s3.py:54` discards the camera prefix) makes every camera list and re-extract the entire raw bucket, producing 4× duplicated, cross-contaminated trimmed clips. The four-camera grouping that everything downstream depends on is built on wrong inputs.
- *complete-train → end-results (Service 2):* on a strict happy path (all 4 clips correctly present within tolerance, `models/production/` staged, delivery endpoints reachable, **no `--skip-upload`**, and FINALIZING entered exactly once) the code path **can** reach `COMPLETED`. But none of those conditions hold by default here: `models/production/` is empty (all features NO_DATA), the validation guide invokes `--skip-upload` (→ L-1 livelock), and any re-entry duplicates email/upload (L-2/L-3). It has never executed once.

**Q2 — If not, at exactly which stage does it stop?**
Walking the flow top-down, the **first stage that fails is Extraction (Stage A / Service 1)** — it does not halt, it silently produces wrong output (E-1, `run_extraction_service.py:88` → `s3.py:54`). If Service 1 is bypassed by placing correct clips in complete-train, the next hard stop is invocation-dependent: with `--skip-upload` the batch **never reaches terminal** (L-1 livelock, never uploads to end-results); without it, the pipeline can complete but with duplicate-delivery (L-2/L-3) and a +1 wagon miscount (R-1). On this specific box a prerequisite also blocks meaningful output: **`models/production/` is empty**, so Stage 3 yields all-NO_DATA reports.

**Q3 — First blocking issue that must be fixed:**
**E-1 — `S3Client.list_objects` discards the camera prefix (`train_extraction/s3.py:54-58`, called at `run_extraction_service.py:88`).** It is first in the flow, it corrupts every downstream input, and no amount of correctness later can recover from cross-contaminated batches. Fix by threading the embedded prefix into the `list_objects` call (the write path `upload_file` already does this at `s3.py:91`).
Immediately behind it, before a genuine end-to-end AUTO run can be declared working: stage the production models into `models/production/` (else all findings are NO_DATA), and resolve the `--auto` delivery defects **L-1** (skip-upload livelock) and **L-2/L-3** (duplicate/looping finalization).

*No code was modified. Verification only.*
