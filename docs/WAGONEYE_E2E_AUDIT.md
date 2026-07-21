# WagonEye v5 — End-to-End Production Audit

**Scope:** `wagon_eye_v4_new/` (~24,840 LOC, 80 modules). Static execution trace from raw CCTV in S3 to reports/email/archive.
**Method:** followed actual call graphs. Every claim below cites a file and line. No code was modified.

## 0. Audit coverage & honesty statement

**Read in full and traced:** `orchestrator/*` (all 5), `core/{config,constants,batch,lifecycle,camera_features}.py`, `reconstruction/runner.py`, `wagon_count/{run_global_count,global_alignment}.py`, `materializer/wagon_cache_builder.py`, `fusion/wagon_state_builder.py`, `features/_common.py`, `delivery/{s3_upload,notification,finalization}.py`, `train_extraction/{driver,run_extraction_service,s3,extractor,url_utils}.py`.

**Signature-verified only (bodies not fully read):** the four feature processor bodies, `reporting/*`, `rendering/*`, `delivery/dashboard_ingest.py`, `wagon_count/{tracker_engine,video_segmenter,global_train_state}.py`, `features/inference_lib/*`. Findings about these are limited to their call contracts, which I did verify.

**Critical correction to the premise.** The task states the automatic pipeline fails while local succeeds. `logs/wagon_eye.log` contains **only three runs, all `mode : local`** (`grep "mode  *:"` → 3×local, 0×auto). There is **no `[ORCH]`, `[DISCOVERY]`, `[GROUPING]`, `[ATTACH]`, or `[LIFECYCLE]` line anywhere in the log.** `--auto` has never been executed on this machine. Therefore I cannot attribute an observed AUTO failure to a root cause from evidence. What follows in §14 is a list of defects that **provably would** break AUTO, each traced in code — not a diagnosis of a failure I witnessed. The one recorded failure in the log is a **local** run (§14.6).

---

## 1. Raw video entry

| Question | Answer | Reference |
|---|---|---|
| Where raw video enters | `biro-wagon-raw-video-copy/<camera_folder>/` | `core/constants.py:180` |
| Which module watches raw | `train_extraction/run_extraction_service.py` `sweep_camera()` | `:94` |
| Which module trims | `train_extraction/extractor.py` `TrainExtractor.extract()` | `:97` |
| Which module uploads trimmed | `extractor._upload` → `s3.S3Client.upload_file` | `:75`, `s3.py:86` |
| Raw bucket | `S3_RAW_BUCKET` | `constants.py:180` |
| Trimmed bucket | `complete-train` = `S3_TRIMMED_BUCKET` = `S3_INPUT_BUCKET` | `constants.py:181,185` |
| Decides trimming starts | `TrainSegmentFinder.analyze()` via `_handle_standalone` | `extractor.py:320` |
| Decides trimming complete | `incomplete is None` → segments emitted | `extractor.py:324,345` |

```
biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_{1_LEFT_UP,2_RIGHT_UP,5_RIGHT_TOP,6_LEFT_TOP}/*.mp4
   │
   ├─ run_extraction_service.main()          loop, poll=60s              :165
   │   └─ sweep_camera(camera)                                            :94
   │       ├─ D.get_extractor(camera) ── builds YOLO classifier          driver.py:115
   │       ├─ _list_raw_keys(ex, raw_bucket) ◄── ***DEFECT E-1***        :86
   │       ├─ skip if key in local ledger  logs/extraction_state/         :63
   │       └─ D.extract_trains(camera, key)                              driver.py:188
   │           └─ TrainExtractor.extract()                               extractor.py:97
   │               ├─ _download → temp                                   :69
   │               ├─ train_in_progress? → _handle_ongoing               :153
   │               │     ├─ check_train_at_start → merge_videos          :160
   │               │     └─ else flush + inline standalone               :312
   │               └─ _handle_standalone → segment_finder.analyze()      :317
   │                   └─ _emit_segments → trim_video → _upload          :119
   ▼
complete-train/<camera_folder>/<raw_basename>_train.mp4
```

Naming: `f"{first_video_name}_train.mp4"` (`extractor.py:131`), multi-segment → `_train_part{idx}.mp4`, flushed tail → `_train_incomplete.mp4` (`:255`). The raw basename carries `YYYYMMDD_HHMMSS`, which is the only batch-grouping signal downstream.

---

## 2. Automatic pipeline entry — `master_runner --auto`

```
main()                                                        master_runner.py:877
 ├─ setup_logging()                                                       :883
 ├─ CFG.startup_summary(mode="auto")                                      :895
 ├─ CFG.validate_config(mode="auto") ── returns 2 on any error            :896
 ├─ resolve_feature_config(interactive=False for --auto)                  :911
 └─ run_auto()                                                            :931
     ├─ boto3.client("s3", region=ap-south-1)                             :723
     ├─ state_loc = "end-results/processed_batches.json"                  :724
     ├─ processed = load_batch_state(s3, state_loc)          train_batch_manager.py:64
     └─ WHILE not _SHUTDOWN_REQUESTED:                                    :747
         │
         │ (a) S3 POLL / ACTIVE SET
         ├─ BM.list_active_manifests(s3, processed)          batch_manifest.py:318
         │     LIST end-results/archive/ Delimiter="/" → CommonPrefixes
         │     skip key in processed; GET archive/<k>/manifest.json
         │     skip is_terminal(m.lifecycle_status)
         │
         │ (b) CANDIDATE DISCOVERY
         ├─ list_candidate_videos(s3)                  train_batch_manager.py:196
         │     └─ _list_input_objects(): paginated LIST over the 4 prefixes  :155
         │         └─ _camera_for_key(key)  ── camera parsing               :130
         │         └─ parse_train_timestamp(key) ── timestamp regex     core/batch.py:75
         │         drop → log "[DISCOVERY] ... decision=DROPPED"            :214
         │
         │ (c) ATTACHMENT
         ├─ for cv in candidates: _attach_candidate(...)     master_runner.py:605
         │     rank actives by |Δt| ≤ 120s                                  :620
         │     2 within 5s of each other → videos_for_review, no attach     :645
         │     none active, one terminal in range → ignore, no reopen       :659
         │     else BatchManifest.new(batch_key=cv.train_timestamp)         :668
         │     same ETag → no-op; changed ETag → drop markers, rebuild      :676
         │     target.set_camera(slot) → arms support window on RIGHT_UP  manifest:177
         │     BM.write_local + BM.save_s3                                  :696
         │
         │ (d) LIFECYCLE
         ├─ for key in sorted(actives): LR.advance(m, ctx)  lifecycle_runner.py:638
         │
         │ (e) TERMINAL
         └─ if is_terminal: processed[key]=...; save_batch_state()          :764
```

**`poll_for_batches()` and `select_runnable_batch()` (`train_batch_manager.py:242,334`) are dead code** — `run_auto` never calls them. Consequently `--partial-wait` (`master_runner.py:865`) is parsed, passed as `partial_wait_minutes` (`:936`), and **never read by `run_auto`** (which reads only `workspace`, `poll_interval`, `run_once`, `force_batch_key`, models dirs, feature_config, skip_*). Silent no-op flag.

**`--once` does not mean "one batch."** `run_once=True` returns after **one poll tick** (`:771`), typically before any batch reaches terminal.

**`--batch <key>` silently no-ops** when the key isn't an active manifest: `keys=[force_key]`, `m = actives.get(key)`, `if m is None: continue` (`:756-762`) → returns 0. Replaying a completed batch does nothing and reports success.

---

## 3. Local pipeline — `master_runner --local-only`

```
main() → run_local()                                          master_runner.py:793
 ├─ scan_local_video_dir(local_inputs)                          core/batch.py:112
 ├─ HARD REQUIREMENT: all 4 cameras present, else return 2               :806
 ├─ build_local_batch() → bucket="__local__", batch_key=now()  core/batch.py:84
 ├─ _NoopS3 (download raises, upload returns None)                       :814
 └─ process_batch(skip_upload=True, skip_email=True)                     :193
```

### The central architectural fact

**LOCAL and AUTO are two independent pipeline drivers.** They share the *stage modules* but not the *orchestration*:

| | LOCAL | AUTO |
|---|---|---|
| Driver | `process_batch()` `master_runner.py:193` | `LR.advance()` `lifecycle_runner.py:638` |
| Stage 2 | `wagon_cache_builder.build()` `:271` | `.build_cameras()` `lifecycle_runner.py:243` |
| Stage 3 | all-cams, `load` serial then 3 parallel `:336-358` | per-camera, marker-gated `_run_camera_features` `:285` |
| Stage 3 skip logic | **none** — always re-runs | `FM.is_up_to_date()` `:318` |
| Stage 4 | `build(state, root, verbose)` `:366` | `+camera_arrival, disabled_features, gst_version, fusion_revision` `:260` |
| Disabled features | `_mark_disabled()` writes **flat** `:322` | `disabled_features` set passed to fusion `:281` |
| Stage 5b failure | caught → `BATCH_REPORT_FAILED` `:472` | **uncaught** → propagates `:443` |
| Cameras required | 4/4 or abort `:806` | master only; partial OK |
| Persistence | none | BatchManifest + 4 marker families |
| Delivery | skipped | `stage_finalize()` `:461` |

I verified every shared stage module accepts **both** call shapes (`cameras=`, `report_meta=`, `camera_arrival=`, `idempotency_key=` are all optional params with defaults). **There is no signature mismatch.** The divergences are behavioural, listed in §14.

---

## 4. Camera discovery

```
S3 key: camera_CCTV_HZBN_DHN_5_RIGHT_TOP/CH05_20260719_060012_train.mp4
   │
   ├─ ext filter (.mp4/.avi/.mov/.mkv)      train_batch_manager.py:140,184
   ├─ _camera_for_key(FULL key, lowercased)                          :130
   │    match order: RIGHT_UP_TOP, LEFT_UP_TOP, RIGHT_UP, LEFT_UP    :124
   │    tokens: right_up_top|right_top / left_up_top|left_top
   │            / right_up / left_up                                 :115
   │    → RIGHT_UP_TOP (folder token "right_top")
   ├─ parse_train_timestamp(basename) regex (\d{8}_\d{6})       core/batch.py:72
   │    → "20260719_060012"
   ├─ batch_key = that timestamp (first camera to arrive wins)  master_runner.py:668
   └─ _attach_candidate → CameraSlot, arrival_state=PRESENT     manifest.py:169
```

**Failure modes, all traced:**

| # | Failure | Behaviour | Ref |
|---|---|---|---|
| D-1 | No camera token in key | `DROPPED reason=no_camera_match`, logged INFO, never retried | `:208-217` |
| D-2 | No `YYYYMMDD_HHMMSS` in basename | `DROPPED reason=no_timestamp` | same |
| D-3 | **Token matched anywhere in key** — a raw basename containing `left_top` inside the `2_RIGHT_UP/` folder is classified **LEFT_UP_TOP**, because match order puts TOP first and matching is substring-over-whole-key. Folder is not authoritative. | silent misattribution | `:130-145` |
| D-4 | Two batches within 120s and their distances differ by <5s | held in `videos_for_review`, **never processed, never alerted** | `:645-656` |
| D-5 | Timestamp within 120s of a terminal key | ignored, no reopen | `:659-666` |
| D-6 | Camera arrives after terminal | ignored (`LATE_CAMERA_POLICY=IGNORE`, only supported value) | `config.py:214,248` |
| D-7 | Missing camera at final deadline | `mark_missing_final()` → `CAMERA_MISSING_FINAL` → `COMPLETED_PARTIAL` | `manifest.py:182` |
| D-8 | ETag change on an attached camera | materialized + feature markers dropped for that camera only | `master_runner.py:678-684` |

**Grouping fragility (D-9).** `batch_key` is the *first-arriving* camera's timestamp, and each camera's trimmed clip is named after **its own first raw clip** (`extractor.py:131`). When a train spans two raw clips, cameras that began the train in different raw clips get basenames separated by the raw-clip duration. If raw clips exceed ~120 s, the four clips **cannot cluster** (`DEFAULT_BATCH_TOLERANCE_SEC = 120`, `train_batch_manager.py:44`) and one train becomes 2–4 single-camera batches, each of which fails `_choose_master` or seals master-only.

---

## 5. Stage 1 — Global reconstruction

Driver `reconstruction/runner.py:93` builds an argv and runs `wagon_count/run_global_count.py` as a **subprocess with `cwd=wagon_count/`** (`:164`), 7200 s timeout (`:102`), `--no-frames` (`:154`).

Guards before launch: master in `video_paths` (`:123`), fallback opt-in (`:127`), **`os.path.exists(p)` for each video (`:131`)**, models dir exists (`:135`). After: exit≠0 (`:205`), JSON present (`:210`), `total_wagons > 0` (`:217`). Full stdout/stderr → `global_state/stage1_wagon_count.log` (`:185`).

Inside `run_global_count.main()`:

1. **Input resolution** `_resolve_optional_input` (`:142`) — absent camera → `None`, not an error. Master absent → **exit 4** (`:341`).
2. **Model resolution** `_resolve_model` (`:166`) with aliases `right_up_wagon_gap.pt→right_up_gap.pt`, `left_up_wagon_gap.pt→left_up_gap.pt` (`:160`). **`top_gap.pt` has no alias** — the production name `top_gap_2.pt` documented in `docs/EC2_DEPLOYMENT_CHECKLIST.md` will **not** resolve → exit 2.
3. **STEP 1** per-camera `GapTracker.process_video` (`:188/:201`). Side: conf 0.4, min_height_ratio **0.35**; Top: conf 0.4, ratio **0.05** (`:286-292`).
4. **STEP 2** `MasterClassifier` votes ENGINE/WAGON/BRAKE_VAN over 5 samples/segment on the master only (`:214`). Failure is **caught and downgraded to a warning**, `initial_classifications=[]` (`:437`) → everything downstream classifies UNKNOWN but the run still succeeds.
5. **STEP 3** fusion `global_alignment.assemble_global_train_state` (`:440`):
   - `match_support_to_master` (`:98`) — per support gap, best master gap by `max(iou, time_score)`, gated on `iou ≥ 0.2 or dt ≤ 1.0s`.
   - `cluster_unmatched_supports` (`:143`) — running-mean sweep over leftovers, spread ≤ 1.5 s.
   - `decide_inserted_gaps` (`:175`) — quorum **≥2 distinct cameras**, spread ≤1.5 s, mean conf ≥0.4, ≥1.0 s from any master gap.
   - `fuse_master_timeline` (`:223`) — inserts become synthetic `GapEvent`s with **negative `track_id`** (provenance marker).
   - `build_global_wagons` (`:296`) — boundaries = gap centers; segments between them; `GW_{i}` assigned in rake order; classification by frame containment else nearest.
   - **Double fallback:** any fusion exception → `build_wagons_pure_master` (`:482`); still empty → same again (`:490`). A total fusion failure is invisible except via `fallback_used` in the JSON.
6. **Outputs:** `global_train_state.json` (written twice, `:494` and `:570`), `per_camera_tracking.json` (`:507`), `processed_videos/<CAM>_processed.mp4` (`:515`), `stage1_wagon_count.log`.
7. **Seal** back in `lifecycle_runner.stage_seal` (`:129`): `global_state_version = sha256(global_train_state.json)` (`:162`), `_transition → GLOBAL_STATE_SEALED`. Guarded by `if manifest.global_state_version: return True` (`:134`) — never reseals.

### Stage-1 finding R-1 — asymmetric phantom-segment guard

`build_global_wagons` (`global_alignment.py:326-352`) builds segments as `[0 … b₁-1], [b₁ … b₂-1], …, [bₙ … total-1]`. The **leading** segment (pre-train footage in the trim buffer) is dropped only if it classifies UNKNOWN:

```python
if len(segs) > 1 and initial_classifications:
    if label_for_frame(...) == SegmentClass.UNKNOWN:
        segs = segs[1:]
```

There is **no equivalent guard for the trailing segment** (`prev … master_total_frames-1`, `:333`), which is the post-train empty track inside the end buffer (`END_EXTRA_BUFFER=5s`, `TRACK_END_SECONDS=5s`, `driver.py:151-152`). That trailing phantom is emitted as a real `GW_n`, inflating `total_wagons` by one and producing a final wagon whose features will all be NO_DATA/OK-empty. This is a strong candidate for systematic off-by-one count error vs. production.

---

## 6. Stage 2 — Materialization

```
GlobalTrainState.wagons[].{start_time,end_time}   (master clock)
   × per_camera_fps[cam]  (from per_camera_tracking.json, else master_fps, else 25.0)
   ▼
_wagon_local_range(): sf=round(start*fps), ef=round(end*fps)-1, clipped   :81
   ▼
_extract_one_camera(): cap.read() linear walk, dispatch by frame_to_target :101
   ▼
wagon_cache/GW_n/<lowercase_folder>/frame_NNNNNN.jpg   (q95, constants.py:245)
```

`build()` (`:177`, LOCAL) runs all cameras through a 4-worker pool, **no markers, no skip, always re-extracts**.
`build_cameras()` (`:324`, AUTO) is per-camera, **serial**, and idempotent: skip iff marker `etag` + `global_state_version` + `materializer_schema_version==2` + `status=="OK"` + cache non-empty (`:355-363`). Rebuild goes to `tempfile.mkdtemp(dir=cache_root)` then `_swap_camera_cache` (`:305`) — `rmtree` then `move` per wagon folder.

**Finding M-1 (silent data loss).** `build_cameras` wraps the build in `try/except Exception` that logs and **continues** (`:401`). No marker is written, `per_camera_total[cam]` is unset, and the function returns success. The caller `stage_process_cameras` (`:243`) ignores the return value entirely and unconditionally appends the camera to `manifest.materialized_cameras` (`:249-251`). A camera whose video is corrupt is recorded as materialized with an empty cache; its features then produce NO_FRAMES and the batch reports `COMPLETED_PARTIAL` with no indication the cause was a decode failure.

**Finding M-2.** `_extract_one_camera` returns `(camera_id, {})` on missing file / `cap.isOpened()` false (`:113,119`) — a warning only. Combined with M-1, three distinct failure classes (missing, unopenable, mid-decode exception) are indistinguishable downstream.

---

## 7. Feature pipeline

Registry `core/camera_features.py:77` — 9 work units:

| Camera | Features | Output dir |
|---|---|---|
| RIGHT_UP | door(right), ocr(sole), damage(side) | `wagon_states/{door,ocr,damage}/RIGHT_UP/GW_n.json` |
| LEFT_UP | door(left), damage(side) | `wagon_states/{door,damage}/LEFT_UP/` |
| RIGHT_UP_TOP | load(primary), damage(primary) | `wagon_states/{load,damage}/RIGHT_UP_TOP/` |
| LEFT_UP_TOP | load(support), damage(support) | `wagon_states/{load,damage}/LEFT_UP_TOP/` |

Ordering: `features_for_camera` sorts by `_FEATURE_ORDER` = load(0) before damage(1), door(0) before ocr(1) (`:93-112`).

Marker identity (`feature_markers.compute_identity:65`): camera, feature, **etag**, **global_state_version**, **model_sha256** (production model via `PM.model_for(feature, camera)`), **processor_schema_version**, **feature_config_hash** (thresholds). Skip iff marker `status=="OK"` and all 7 keys match (`:123`). Absent `.pt` → `sha256_file` returns `"MISSING"` (`:37`) — a stable identity, so staging the real model later invalidates and re-runs.

Shared substrate `features/_common.py`: cached YOLO loader with `torch.load` monkey-patch (`:56`), `DEVICE`/`HALF` resolved once (`:44`), **adaptive stable-interior trim of 5% per side clamped [3,12] frames** applied to inference only (`:114-144`).

### Finding F-1 — threshold registry is stale
`FEATURE_THRESHOLDS` (`camera_features.py:57`) hashes `CONF_DOOR=0.40`, `CONF_DAMAGE=0.55`, `loaded_ratio=0.35`. The documented production values are door 0.85/0.88, top damage 0.70, load majority-vote ≥0.80. The marker therefore **does not key on the thresholds the processors actually use**. A real threshold change will not invalidate any marker, and cached results from the old thresholds will be silently reused.

### Finding F-2 — disabled features are broken in LOCAL
`process_batch._mark_disabled` writes to `os.path.join(states_root, name)` → `wagon_states/load/GW_1.json` — the **flat** layout (`master_runner.py:322,329`). Fusion's `detect_layout` returns `"camera"` whenever any per-camera dir exists (`wagon_state_builder.py:111-129`), and camera-scoped fusion never reads flat files (module docstring `:18-20`). Additionally `process_batch` calls `wagon_state_builder.build()` **without `disabled_features`** (`:366`), so it defaults to `set()`. Net effect: with `--disable-features` in LOCAL mode, `DISABLED BY USER` never reaches the report; fields show `PENDING_CAMERA`/`NO_DATA` instead. AUTO handles this correctly (`lifecycle_runner.py:263,281`).

### Finding F-3 — per-wagon exception isolation is not per-wagon in AUTO
`_run_camera_features` wraps the **entire feature run for a camera** in one try (`lifecycle_runner.py:323-333`) and writes a single marker `status="FAILED"`. A crash on wagon 40 of 62 loses wagons 40–62 and marks the whole (camera,feature) failed. `is_up_to_date` requires `status=="OK"` (`:126`), so it will re-run wholesale next tick — and crash again at the same wagon, forever, without ever reaching a terminal state via that path.

---

## 8. Stage 4 — Fusion

`fusion/wagon_state_builder.build:451`. Layout auto-detect (`:111`), per-wagon try/except keeping the previous JSON (`:495`), idempotent atomic write that skips rewriting when the content signature minus `fused_at` is unchanged (`:79-98`).

Authority (`_fuse_camera_scoped:159`):

| Field | Rule | Ref |
|---|---|---|
| `classification` | sealed GST only | `:168` |
| `wagon_identifier` | RIGHT_UP OCR | `:199` |
| `loco_number` | rides RIGHT_UP OCR | `:204` |
| `right_door` / `left_door` | RIGHT_UP / LEFT_UP | `:211,214` |
| `load_status` | RIGHT_UP_TOP if OK+valued, **else** LEFT_UP_TOP | `:227-230` |
| `top_damage` | DAMAGE if **either** top cam confirms; details merged | `:253-268` |
| `side_damage` | DAMAGE if **either** side cam confirms | `:290-303` |

`_result_state` (`:140`) preserves the three-way distinction: `CAMERA_MISSING_FINAL` > `PENDING_CAMERA` (incl. camera present but feature not yet run, `:147`) > `DISABLED_BY_USER`/`FAILED`/`NO_FRAMES`/`OK`/`NO_DATA`. A missing feature is **never** a false OK.

Late cameras: `stage_process_cameras` bumps `fusion_revision` and re-runs full fusion over all currently-available per-camera JSON (`lifecycle_runner.py:259-267`). No reseal, no renumber.

**Finding X-1.** `confidence` averages only 4 fields — `wagon_identifier, left_door, right_door, load` (`:410`). Damage confidences are excluded, so a wagon whose only finding is damage has `confidence=0.0`, which is the report's sort key.

**Finding X-2.** `_finish_anomalies_and_confidence` gates `TOP_DAMAGE` on `FEATURE_DAMAGE not in disabled` (`:396`) but `SIDE_DAMAGE` has **no disabled guard** (`:398`). With damage disabled, `u.side_damage` is set to `DISABLED_DISPLAY` (`:283`) so the comparison is false in practice — latent, not currently firing, but the asymmetry is a trap.

---

## 9. Reports

`stage_reports` (AUTO, `lifecycle_runner.py:373`) / inline Stage 5a+5b (LOCAL, `master_runner.py:420,447`).

| Output | Producer | Failure handling |
|---|---|---|
| `reports/<CAM>_report.pdf` ×4 | `camera_reports.build_all` | caught+logged both paths (`LR:428`, `MR:436`) |
| `reports/combined_train_report.pdf` | `combined_train_report.build` | **LOCAL caught → REPORT_FAILED (`MR:472`); AUTO UNCAUGHT (`LR:443`)** |
| `reports/combined_train_report.json` | same call | same |
| `processed_videos/<CAM>_processed.mp4` | `feature_overlay_renderer.render_all_cameras` | caught (`LR:416`, `MR:392`) |
| `delivery/dashboard/<CAM>_inspection.json` | `dashboard_ingest.run` | fully swallowed (`LR:605`) |
| `delivery/finalization.json` | `finalization.write` | atomic |

`report_meta` (`:346`) carries `report_revision`, `report_status ∈ {INTERIM, FINAL, FINAL_PARTIAL}`, camera present/pending/missing_final, `generated_from_global_state_version`, `fusion_revision`, `partial_reason`. LOCAL passes **no** `report_meta` — local PDFs lack all provenance.

Neither path loads a model in reporting; the invariant holds.

---

## 10. S3 output

| Artifact | Key | Module | Env |
|---|---|---|---|
| Combined PDF | `reports/<key>/combined_train_report.pdf` | `s3_upload.upload_pdf:77` (microservice first, S3 PUT fallback `:85`) | `WAGONEYE_UPLOAD_API_URL`, `_S3_REPORTS_PREFIX`, `_S3_OUTPUT_BUCKET` |
| Camera PDFs ×4 | `reports/<key>/<CAM>_report.pdf` | same | same |
| Combined JSON | `reports/<key>/combined_train_report.json` | `upload_json:97` | same |
| Batch tree | `archive/<key>/{global_state,wagon_states,reports,evidence,processed_videos}/…` | `upload_tree:115` | `_S3_ARCHIVE_PREFIX` |
| Manifest | `archive/<key>/manifest.json` | `batch_manifest.save_s3:290` | `_MANIFEST_S3_PREFIX` (empty→archive) |
| Dashboard JSON | `dashboard/<camera_folder>/<date>/…_inspection.json` + POST | `dashboard_ingest` | `_S3_DASHBOARD_PREFIX`, `_DASHBOARD_INGEST_ENABLED`, `_INSPECTION_VERSION` |
| Terminal state | `processed_batches.json` (bucket root) | `save_batch_state:87` | `_S3_STATE_KEY` |
| Trimmed clip | `complete-train/<camera_folder>/<raw>_train.mp4` | `extractor._upload:75` | `_EXTRACTION_<CAM>_TRIMMED_BUCKET` |

`wagon_cache` JPEGs are deliberately not uploaded. `upload_tree` swallows per-file exceptions and only returns a count (`:149`) — a partial archive is indistinguishable from a complete one.

**Finding S-1.** `list_active_manifests` (`batch_manifest.py:318`) lists `archive/` with `Delimiter="/"`, so **every batch ever archived remains a CommonPrefix forever**. Each poll tick enumerates all of them; every key not in `processed_batches` triggers a `GET`. Cost and latency grow linearly and without bound with lifetime batch count, and if `processed_batches.json` is ever lost, every historical archive prefix is re-fetched and any non-terminal one is resurrected.

---

## 11. Email

`delivery/notification.send_email:15`, called once from `stage_finalize:577` (AUTO) and once from `process_batch:534` (LOCAL, only when `not skip_upload and not skip_email`).

Sent when: `terminal != REPORT_FAILED` **and** not `already_emailed` **and** not `ctx.skip_email` **and** `report_pdf_url` is truthy (`:564-574`). Subject carries batch key, wagon count, loco numbers, IST timestamp (`:41`). Payload references the PDF by URL (`attachment_url`), plus `json_url`; no file attachment. `idempotency_key` goes in both body and `Idempotency-Key` header (`:71-73`). Retries 3× with 15/30/45 s backoff (`:75-86`).

**Finding E-2 — email is not retried after final failure.** If all three attempts fail, `send_email` returns False → `email_status="failed"`, `email_sent=False` (`:588`) → `FIN.write` → **`_finish(...)` marks the batch terminal anyway** (`:609`) → `processed_batches[key]` set → `list_active_manifests` skips it forever. The email is permanently lost. `docs/AUTO_PIPELINE_ARCHITECTURE.md §10.3` claims "only the not-yet-succeeded deliverables are retried" — the code does not implement this. The identical argument applies to `uploaded=False` (`:553`).

---

## 12. State machine

```
DISCOVERED ──┐
COLLECTING_CAMERAS ──┤ (pre-seal block, lifecycle_runner.py:648)
WAITING_FOR_MASTER ──┤
WAITING_FOR_SUPPORT ─┘
   │ master==RIGHT_UP and (is_complete or past_support_window)      :659
   ├───────────────────────────────► RECONSTRUCTING
   │ master==RIGHT_UP, support window open → WAITING_FOR_SUPPORT, return  :662
   │ no master and past_master_deadline:
   │     ├ LEFT_UP + fallback enabled → RECONSTRUCTING              :666
   │     ├ past_final_deadline → FAILED_NO_GLOBAL_STATE (terminal)  :671
   │     └ else → WAITING_FOR_MASTER, return                        :674
   │ else → COLLECTING_CAMERAS, return                              :676
   ▼
RECONSTRUCTING ── stage_seal fails → FAILED_NO_GLOBAL_STATE         :160
   │ ok
   ▼
GLOBAL_STATE_SEALED ──(unconditional)──► PROCESSING_AVAILABLE_CAMERAS :688
   │ stage_process_cameras(present) + stage_reports(final=False)     :693
   ├ is_complete or past_final_deadline → FINALIZING                 :702
   └ else → WAITING_FOR_LATE_CAMERAS, return                         :707
   ▼
WAITING_FOR_LATE_CAMERAS                                             :710
   ├ any present camera with unfinished features → PROCESSING_LATE_CAMERA :715
   ├ is_complete or past_final_deadline → FINALIZING                 :718
   └ else return (wait)                                              :722
   ▼
PROCESSING_LATE_CAMERA → process + report → WAITING_FOR_LATE_CAMERAS :724-730
   ▼
FINALIZING → stage_finalize()                                        :733
   ▼
COMPLETED | COMPLETED_PARTIAL | REPORT_FAILED | FAILED | FAILED_NO_GLOBAL_STATE
```

`advance()` loops with `guard < 32` (`:641`); exhaustion logs a warning and returns non-terminal (`:742`). Unknown state → `FAILED` (`:737`).

**Timers** (all `core/config.py`, minutes):

| Timer | Default | Anchor | Set | Read |
|---|---|---|---|---|
| `MASTER_WAIT_MINUTES` | 10 | `first_seen_at` | `manifest.py:194` | `:665` |
| `SUPPORT_FUSION_WAIT_MINUTES` | 3 | **RIGHT_UP arrival** | `manifest.py:177` | `:659` |
| `FINAL_CAMERA_WAIT_MINUTES` | 30 | `first_seen_at` | `manifest.py:195` | `:702,718` |
| `ACTIVE_BATCH_POLL_INTERVAL` | 60 | — | `config.py:210` | **unused — `run_auto` uses its own `poll_interval` kwarg default 60 (`:728`)** |
| Stage-1 subprocess timeout | 7200 s | — | `runner.py:102` | `:166` |
| PDF microservice | 120 s × 3, 10 s sleep | — | `s3_upload.py:59,69` | |
| Email | 60 s × 3, 15/30/45 s | — | `notification.py:77,86` | |

`past_support_window` returns False until RIGHT_UP arrives (`manifest.py:205`) — correct arming.

---

## 13. Failure analysis

| Stage | Failure | Handling | Continue? |
|---|---|---|---|
| A extract | model missing | `sweep_camera` logs, `errors+=1`, camera skipped this sweep | yes |
| A extract | per-key exception | logged, key **not** added to ledger → retried forever each sweep | yes, retries |
| 0 discovery | `list_objects_v2` throws | `break` out of that prefix, other prefixes continue | yes |
| 0 discovery | prefixes empty | one-time warning, discovers nothing | idles |
| 0 manifest | newer schema | `ManifestSchemaError` → `load_s3` returns None → batch invisible | **batch silently dropped** |
| 0 attach | ambiguous | `videos_for_review`, no attach, no alert | **video silently dropped** |
| 1 seal | master absent | `ReconstructionError` → `FAILED_NO_GLOBAL_STATE` | terminal |
| 1 seal | subprocess exit≠0 / no JSON / 0 wagons | same | terminal |
| 1 seal | timeout 7200 s | converted to `ReconstructionError` | terminal |
| 1 internal | classification throws | **warning only**, all wagons UNKNOWN | yes, silently degraded |
| 1 internal | fusion throws | **double fallback to pure-master** | yes, silently degraded |
| 2 materialize | corrupt video | logged, no marker, camera still marked materialized (**M-1**) | yes, silently empty |
| 3 features | model missing | `NO_DATA` per wagon | yes |
| 3 features | exception | whole (camera,feature) FAILED (**F-3**) | yes, re-runs forever |
| 4 fusion | per-wagon exception | previous unified JSON kept | yes |
| 4b render | exception | caught both paths | yes |
| 5a camera PDFs | exception | caught both paths | yes |
| 5b combined | exception | LOCAL → REPORT_FAILED; **AUTO → propagates to tick handler** | AUTO: infinite retry |
| 6 upload | exception | `uploaded=False`, marker written, **terminal anyway** | never retried |
| 6 email | 3× failure | `email_sent=False`, **terminal anyway** | never retried |
| 6 dashboard | any exception | fully swallowed | yes |
| loop | any tick exception | logged, `sleep(poll_interval)` | yes |

**Silent-termination points:** manifest schema refusal; ambiguous-attach hold; `--batch` on a non-active key; `_SHUTDOWN_REQUESTED` between batches; `advance()` guard exhaustion; `save_batch_state` failure (logged, in-memory only — on restart the batch re-runs and **re-emails**).

---

## 14. LOCAL vs AUTO — traced differences

Restating §0: no AUTO run exists in the logs, so this is "what would break," proven from code, not a post-mortem.

### 14.1 E-1 — extraction lists the entire raw bucket *(most severe)*

```python
# run_extraction_service.py:88
objs = ex.s3.list_objects(raw_bucket)     # raw_bucket = "bucket/camera_folder"

# s3.py:53
def list_objects(self, bucket_string, prefix=""):
    bucket, _ = split_bucket_prefix(bucket_string)   # ← prefix DISCARDED
    params = {"Bucket": bucket, "Prefix": prefix}    # ← "" — whole bucket
```

The camera prefix embedded in `bucket_string` is parsed and thrown away; the `prefix` parameter defaults to `""` and no caller passes it. `upload_file` (`s3.py:91`) *does* honour the prefix — the asymmetry is the bug.

Consequence: each of the four cameras' sweeps lists **every object in `biro-wagon-raw-video-copy`**, and since ledgers are per-camera (`:60`), each camera extracts **all four cameras' raw clips** using its own classifier and uploads them into its own trimmed folder. Every raw clip is processed 4× and lands under 4 wrong camera folders. At inspection, `_camera_for_key` matches tokens anywhere in the key with TOP priority (D-3), so a `left_top` raw basename sitting in the `2_RIGHT_UP/` folder is classified `LEFT_UP_TOP`. Batches are then assembled from cross-contaminated, misattributed clips. This alone is sufficient to make AUTO produce garbage while LOCAL (which reads four hand-placed files) is unaffected.

### 14.2 L-1 — `--skip-upload` breaks AUTO's resume loop

`_persist` mirrors to S3 only when `not ctx.skip_upload` (`lifecycle_runner.py:74`), but `run_auto` rebuilds `actives` **exclusively from S3** each tick (`master_runner.py:749`) and never calls `BM.load_local`. `_attach_candidate` does call `save_s3` unconditionally (`:697`), so the manifest exists at `DISCOVERED` — but no lifecycle transition is ever mirrored. Every tick reloads the `DISCOVERED` manifest, `global_state_version` is None, and Stage 1 re-runs from scratch, forever. `--skip-upload` is safe in LOCAL and a livelock in AUTO.

### 14.3 L-2 — exactly-once email/upload breaks on any FINALIZING re-entry

`stage_reports` increments `manifest.report_revision` **unconditionally on every call** (`:438`), and `stage_finalize` calls it (`:471`). The idempotency key is derived from `report_revision` (`:488-489`), and the skip conditions compare against it (`:492-497`):

```python
same_rev = (prior["report_revision"] == manifest.report_revision and …)
already_emailed = prior["email_sent"] and prior["email_idempotency_key"] == idem_key
```

Any re-entry into `FINALIZING` — crash mid-finalize, a `stage_reports` exception, an S3 hiccup — produces a **new** `report_revision`, hence a **new** `idem_key`, hence `already_emailed=False` and `same_rev=False`. The batch re-uploads everything and **sends a second email**. The documented guarantee ("best-effort exactly-once, only the API-200↔marker-write window") holds only if `FINALIZING` is entered exactly once. LOCAL never has this problem because it has no finalization marker and no re-entry.

### 14.4 L-3 — combined-report failure is terminal in LOCAL, infinite in AUTO

`combined_train_report.build` is inside `try/except` in `process_batch` (`master_runner.py:449-478` → `BATCH_REPORT_FAILED`) but **bare** in `stage_reports` (`lifecycle_runner.py:443`). In AUTO the exception unwinds `stage_finalize` → `advance` → `run_auto`'s tick handler (`:779`), which logs and sleeps. The manifest is left at `FINALIZING` (non-terminal), so the next tick re-enters, re-increments `report_revision`, re-uploads, and re-emails (via L-2). A single malformed report becomes an unbounded upload/email loop.

### 14.5 L-4 — delivery blocks the single-threaded poll loop

`upload_pdf` tries the microservice 3× with a 120 s timeout and a **10 s unconditional sleep after every attempt including the last** (`s3_upload.py:54-69`), and is called for the combined PDF plus each camera PDF (`lifecycle_runner.py:531-537`) — 5 PDFs. Worst case ≈ 5 × 3 × 130 s ≈ **32 minutes** inside one `advance()`, during which no other batch is polled or advanced and `_SHUTDOWN_REQUESTED` is not checked. At a train every few minutes this guarantees backlog and can push other batches past `FINAL_CAMERA_WAIT_MINUTES` into spurious `COMPLETED_PARTIAL`.

### 14.6 The one recorded failure is LOCAL

```
ERROR: master camera RIGHT_UP video is not present; cannot reconstruct (present: [])
ERROR [orchestrator] [BATCH] aborted (stage1): wagon_count subprocess exited 4
```

`reconstruction/runner.py:131` validates `os.path.exists(p)` in the **parent's** cwd, where a relative `--local-inputs` path resolves. It then launches the subprocess with `cwd=wagon_count/` (`:164`) passing those same relative paths, where `_resolve_optional_input` (`run_global_count.py:142`) returns `None` for all four → `present_videos == {}` → exit 4. The docs call this a "usage note"; it is a genuine defect — `runner.run` should `os.path.abspath()` every path before building argv. **AUTO is immune**: `_download_present` builds paths under `CFG.WORKSPACE_ROOT`, which is absolute (`config.py:94`).

---

## 15. TOP 10 RISKS (ranked by likelihood × blast radius)

| # | Risk | Where | Effect |
|---|---|---|---|
| **1** | **Extraction lists the whole raw bucket — camera prefix silently discarded** | `train_extraction/s3.py:54-58` ← `run_extraction_service.py:88` | Every raw clip processed by all 4 cameras with the wrong classifier and uploaded to 4 wrong folders. 4× cost, cross-contaminated batches, misattributed cameras. **Fires on the first AUTO sweep.** |
| **2** | **Exactly-once email/upload defeated by unconditional `report_revision` increment** | `lifecycle_runner.py:438,488-497` | Any FINALIZING re-entry → duplicate email to the full production CC list (10 recipients, `constants.py:219`) + duplicate S3 upload. |
| **3** | **Combined-report exception uncaught in AUTO** | `lifecycle_runner.py:443` vs `master_runner.py:472` | Batch pinned in `FINALIZING`; unbounded retry loop; combines with #2 to spam email every poll tick. |
| **4** | **Email/upload failure still marks the batch terminal — never retried** | `lifecycle_runner.py:553,588,609` | Transient API outage permanently loses the report notification. Contradicts `AUTO_PIPELINE_ARCHITECTURE.md §10.3`. |
| **5** | **Trailing phantom segment counted as a wagon** | `global_alignment.py:326-352` (leading guard, no trailing guard) | Systematic +1 wagon count vs production on every train; final `GW_n` is post-train empty track. |
| **6** | **`--skip-upload` livelocks AUTO** | `lifecycle_runner.py:74` + `master_runner.py:749` | Stage 1 (the most expensive stage) re-runs every poll tick forever. The natural first-safe-run flag is the one that breaks it. |
| **7** | **Delivery blocks the single-threaded loop up to ~32 min** | `s3_upload.py:54-69` × 5 PDFs | Backlog; other batches pushed past their final deadline into false `COMPLETED_PARTIAL`; SIGTERM unresponsive. |
| **8** | **Camera identified by substring anywhere in the key, TOP-first** | `train_batch_manager.py:130-145` | Folder is not authoritative; a basename token overrides it. Amplified by #1 into systematic misattribution. |
| **9** | **Materialization failure recorded as success** | `wagon_cache_builder.py:401` + `lifecycle_runner.py:249` (return value ignored) | Corrupt video → empty cache → camera marked materialized → silent all-NO_DATA partial report with no diagnosable cause. |
| **10** | **Feature markers key on stale thresholds** | `camera_features.py:57` (0.40/0.55/0.35 vs production 0.85/0.88/0.70/0.80) | A threshold change does not invalidate any marker; stale results silently reused. Undermines the whole skip-logic correctness argument. |

**Runners-up:** unbounded `archive/` CommonPrefix enumeration per tick (`batch_manifest.py:318`); `--batch <key>` silently no-ops and returns 0 (`master_runner.py:756-762`); ambiguous-attach videos held with no alerting (`master_runner.py:645`); `top_gap_2.pt` unresolvable — no alias (`run_global_count.py:160`); Stage-1 classification failure degrades to all-UNKNOWN with only a warning (`run_global_count.py:437`); `--disable-features` non-functional in LOCAL (F-2); dead `poll_for_batches`/`select_runnable_batch`/`--partial-wait`/`ACTIVE_BATCH_POLL_INTERVAL`.

---

*No code was modified. No fixes applied. Audit only.*
