# WagonEye v5 — Complete End-to-End Validation Checklist

Per-stage acceptance checklist for a full production-equivalent run. For each
stage: **input · output · success · failure · files to inspect · logs to
inspect · benchmark to capture.** Companion docs: `WAGONEYE_V5_EC2_VALIDATION_GUIDE.md`
(how to run), `WAGONEYE_V5_ARCHITECTURE.md` (diagram), `WAGONEYE_V5_SCHEMA_PARITY.md`
(payload), `WAGONEYE_V5_PRODUCTION_SIGNOFF.md` (final gate).

Conventions: `$BK` = batch key (`YYYYMMDD_HHMMSS`); `$B` = `batch_outputs/$BK`;
main log = `logs/wagon_eye.log`. All per-stage timings appear as `STAGE <name>`
lines in the log; per-feature lines as `[FEAT/<feature>] done in …`.

---

## Stage A — Train extraction (raw → trimmed)  *(separate producer service)*
- **Input:** raw CCTV clips in `s3://biro-wagon-raw-video-copy/<camera_folder>/…` (per camera).
- **Output:** trimmed `"<raw_basename>_train.mp4"` in `s3://complete-train/<camera_folder>/`.
- **Success:** for a raw clip containing a train pass, ≥1 trimmed clip uploaded; raw key recorded in the local ledger; cross-clip trains stitched.
- **Failure:** no trimmed clip for a clip that clearly contains a train; repeated re-extraction of an already-processed key (ledger not honoured); extractor exception.
- **Files:** `logs/extraction_state/processed_<camera>.json` (ledger); the trimmed S3 objects.
- **Logs:** `extraction.service` / `extraction.<camera>` lines: `sweep: listed=… new=… trains=… errors=…`; `trimmed -> <url>`.
- **Benchmark:** extraction wall-time per raw clip; #trains/clip. (Only when validating the producer; the inspection validation below starts from trimmed clips.)

## Stage 1 — Global reconstruction → GlobalTrainState
- **Input:** 4 trimmed clips (RIGHT_UP master + 3 support) in `$B/downloads/`.
- **Output:** `$B/global_state/global_train_state.json` + `per_camera_tracking.json` + Stage-1 debug overlays `$B/global_state/processed_videos/<CAM>_processed.mp4`.
- **Success:** subprocess exit 0; `total_wagons > 0`; `master_camera=RIGHT_UP`; per-camera gap counts present; cross-camera gap recoveries logged where support agrees.
- **Failure:** subprocess exit ≠ 0; missing JSON; `total_wagons == 0`; `master camera RIGHT_UP video is not present` (batch → `failed_no_global_state`).
- **Files:** `global_train_state.json` (`total_wagons`, `wagons[]` with `global_id`/`classification`/`start_time`/`end_time`), `stage1_wagon_count.log`.
- **Logs:** `[STAGE1] launching wagon_count …`; `[STAGE1] OK total_wagons=… (E:… W:… B:…)`.
- **Benchmark:** Stage-1 wall time (dev ref: **~47 min CPU** for a 62-wagon, ~264 s, 15 fps train); #wagons; #gap recoveries.

## Stage 2 — Wagon-cache materialization
- **Input:** `global_train_state.json` + the 4 trimmed clips.
- **Output:** `$B/wagon_cache/GW_n/<camera>/frame_NNNNNN.jpg` (q95) + `$B/wagon_cache/.materialized/<CAMERA>.json` markers.
- **Success:** every present camera populated for every `GW_n`; frames contiguous within each wagon window; on-disk JPEG count == reported `total_frames`; JPEG quality **95**.
- **Failure:** empty subtree for a present camera; frame count mismatch; a q90 cache reused after the q95 change (marker/schema not bumped).
- **Files:** `wagon_cache/GW_1/right_up/` (spot-check first/last frame ids align with the wagon's time window); `.materialized/*.json`.
- **Logs:** `[STAGE2] done in …s total_frames=…`; per-camera `frames_written`.
- **Benchmark:** Stage-2 wall time (dev ref: **~22 s** for 15,533 JPEGs); frames written; disk used (`du -sh $B/wagon_cache`).

## Stage 3.Door — production side_damage.pt doors
- **Input:** `wagon_cache/GW_n/{right_up,left_up}/` + `models/production/side_damage.pt`.
- **Output:** `$B/wagon_states/door/{RIGHT_UP,LEFT_UP}/GW_n.json` + `$B/evidence/GW_n/door/<CAMERA>/{side}_best.jpg`,`{side}_crop.jpg`,`metadata.json`.
- **Success:** one file per wagon per side camera; `status=OK`; `right_door`/`left_door ∈ {OPEN,CLOSED}`; `door_close_detected` bool; evidence present for OPEN wagons; conf 0.85 (RIGHT_UP)/0.88 (LEFT_UP).
- **Failure:** all `NO_DATA` with the model present (class-name mismatch → check `YOLO(side_damage.pt).names` has `door_open`/`door_close`); `FAILED` per wagon.
- **Files:** `wagon_states/door/RIGHT_UP/GW_1.json` (schema per `EC2_STAGE3_DOOR_VALIDATION.md` §6).
- **Logs:** `[FEAT/door] done in …s ok=n/total`; per-wagon `[door/<CAM>/<GW>] <side>=<STATE> …`.
- **Benchmark:** door stage wall time; per-wagon avg; #OPEN/#CLOSED. (Template: `docs/benchmarks/door.md`.)

## Stage 3.OCR — production wagon_number.pt + loco OCR
- **Input:** `wagon_cache/GW_n/right_up/` + `models/production/wagon_number.pt` + `models/reconstruction/right_up_gap.pt` (loco `locono`) + `easyocr`.
- **Output:** `$B/wagon_states/ocr/RIGHT_UP/GW_n.json` + `$B/evidence/GW_n/ocr/RIGHT_UP/{best_frame,number_crop}.jpg` (wagon) / `{loco_best,loco_crop}.jpg` (engine).
- **Success:** WAGON wagons → `wagon_identifier` (11-digit) where readable, prefix-corrected; ENGINE wagons → `loco_number` (5-digit) + `is_valid_5_digit`; BRAKE_VAN → skipped `NO_DATA`.
- **Failure:** all `NO_DATA` with models + easyocr present; loco numbers empty on a rake with a visible loco (check `right_up_gap.pt` present + emits `locono`); `FAILED`.
- **Files:** `wagon_states/ocr/RIGHT_UP/GW_*.json` (`wagon_identifier`, `loco_number`, `candidates`).
- **Logs:** `[FEAT/ocr] done in …`; `[ocr/<GW>] <number> …`; `[ocr/<GW>] ENGINE loco=<n>`.
- **Benchmark:** OCR stage wall time (slowest feature); per-wagon avg; #valid wagon numbers; #valid loco numbers; easyocr init time.

## Stage 3.Load — production top classification
- **Input:** `wagon_cache/GW_n/{right_up_top,left_up_top}/` + `models/production/{top_classification.pt,ltop.pt}`.
- **Output:** `$B/wagon_states/load/{RIGHT_UP_TOP,LEFT_UP_TOP}/GW_n.json` + `evidence/GW_n/load/<CAMERA>/best_frame.jpg`.
- **Success:** WAGON wagons → `load_status ∈ {LOADED,EMPTY}` by majority vote (conf ≥ 0.80); ENGINE/BRAKE_VAN → `NO_DATA` skipped; RIGHT_UP_TOP authoritative.
- **Failure:** all `NO_DATA` with models present (check `top_classification.pt` emits `wagon_loaded`/`wagon_empty`); `FAILED`.
- **Files:** `wagon_states/load/RIGHT_UP_TOP/GW_*.json` (`load_status`, `loaded_count`, `empty_count`, `loaded_ratio`).
- **Logs:** `[FEAT/load] done in …`.
- **Benchmark:** load stage wall time; per-wagon avg; #LOADED/#EMPTY.

## Stage 3.Damage — production top 4-class + side damage
- **Input:** top: `wagon_cache/GW_n/{right_up_top,left_up_top}/` + `models/production/{right_top_damage.pt,top_left_damage.pt}`; side: `wagon_cache/GW_n/{right_up,left_up}/` + `side_damage.pt`.
- **Output:** `$B/wagon_states/damage/<CAMERA>/GW_n.json` (all 4 cameras) + `evidence/GW_n/damage/<CAMERA>/track_*.jpg`.
- **Success:** top files carry `damage_status` + 4-class booleans (`body_dmg_detected`,`floor_dmg_detected`,`*_probable_detected`); side files carry `damage_status` + `side_damage_details`; conf top 0.70, side 0.85/0.88; band gap-tol 5; **no** loaded-wagon floor filter.
- **Failure:** all `NO_DATA` with models present (class-name mismatch); `FAILED`; side-camera damage files missing (registry not wired).
- **Files:** `wagon_states/damage/RIGHT_UP_TOP/GW_*.json` (top), `wagon_states/damage/RIGHT_UP/GW_*.json` (side, `side=true`).
- **Logs:** `[FEAT/damage] done in …`; `[damage/<CAM>/<GW>] <status> body=… floor=… …` / `SIDE <status>`.
- **Benchmark:** damage stage wall time (top + side); per-wagon avg; #top-damaged; #side-damaged.

## Stage 4 — Fusion → UnifiedWagonState
- **Input:** `$B/wagon_states/{door,ocr,load,damage}/<CAMERA>/GW_n.json` + GlobalTrainState.
- **Output:** `$B/wagon_states/unified/GW_n.json` (one per wagon).
- **Success:** every wagon has a unified state; authority rules honoured (classification←GST, wagon_identifier←RIGHT_UP OCR, right/left_door←side cams, load←RIGHT_UP_TOP else LEFT_UP_TOP, top_damage←any top, **side_damage←either side**, **loco_number←RIGHT_UP OCR**); anomalies set (`LEFT/RIGHT_DOOR_OPEN`, `TOP_DAMAGE`, `SIDE_DAMAGE`, `OCR_MISSING`); missing/pending fields never a false OK.
- **Failure:** fusion exception per wagon (previous unified kept); a field showing OK when its feature was `NO_DATA`/pending.
- **Files:** `wagon_states/unified/GW_1.json` (all fields + `anomalies` + `field_sources`/`field_status` + `result_state`).
- **Logs:** `[STAGE4] fusing N wagons layout=camera …`; `[STAGE4] done`.
- **Benchmark:** fusion wall time (fast, no inference).

## Stage 4b — Overlay rendering (visualization only)
- **Input:** GlobalTrainState + UnifiedWagonState + evidence metadata + `per_camera_tracking.json` + raw videos.
- **Output:** `$B/processed_videos/<CAM>_processed.mp4` (4 cameras).
- **Success:** an mp4 per present camera with wagon ids, gap banners, feature bboxes on best frames, anomaly banners; **no detector rerun** (draws from persisted state only).
- **Failure:** a camera mp4 missing (per-camera failure isolated — batch continues); overlays drawn from a model call (must not happen).
- **Files:** `processed_videos/*.mp4`.
- **Logs:** overlay renderer per-camera lines.
- **Benchmark:** render wall time per camera (decode-bound).

## Stage 5 — Report generation (camera + combined, PDF + JSON)
- **Input:** GlobalTrainState + UnifiedWagonState + evidence + wagon_cache.
- **Output:** `$B/reports/{right_up,left_up,right_up_top,left_up_top}_report.pdf` + `$B/reports/combined_train_report.{pdf,json}`.
- **Success:** 4 camera PDFs + combined PDF open; combined PDF shows title/KPI summary (incl. **loco numbers**), 7-col wagon table with `TOP_DMG`/`SIDE_DMG` columns + issue-row highlighting, Damaged-Wagon evidence grid; `combined_train_report.json` schema `wagon_eye.combined_report.v4` with `legacy_view_model` + `wagons[]` carrying `loco_number`/`side_damage`.
- **Failure:** combined PDF crash (batch → `report_failed`; JSON still written; email suppressed); a camera PDF missing (isolated).
- **Files:** `reports/combined_train_report.json` (validate `summary`, `legacy_view_model.summary_kpis.loco_numbers`, per-wagon `side_damage`); the 5 PDFs.
- **Logs:** Stage-5 report lines; `report_failed` if any.
- **Benchmark:** report build wall time; combined PDF size.

## Stage 6a — Dashboard payload (inspection_data)
- **Input:** `combined_train_report.json` + evidence + `delivery/finalization.json`.
- **Output:** `$B/delivery/dashboard/<CAMERA>_inspection.json` + POST to `cctv-receiver/inspections/ingest`.
- **Success:** one `{camera_id, version, inspection_data}` per camera; RIGHT_UP carries populated `loco_number_results`; side cameras' `damaged_wagons` includes side damage; ingest idempotent (sha + revision). **Set `WAGONEYE_INSPECTION_VERSION=v2`.**
- **Failure:** payload missing a camera; ingest POST non-200 (logged, non-fatal); loco_number_results empty on a loco rake.
- **Files:** `delivery/dashboard/RIGHT_UP_inspection.json` (validate against `WAGONEYE_V5_SCHEMA_PARITY.md`); `delivery/finalization.json` (`dashboard_ingested`).
- **Logs:** `dashboard_ingest` per-camera ingest status.
- **Benchmark:** ingest POST latency per camera.

## Stage 6b — PDF + JSON upload (S3)
- **Input:** `reports/combined_train_report.{pdf,json}` (+ camera PDFs).
- **Output:** `s3://end-results/reports/$BK/…`; upload URLs recorded.
- **Success:** PDF upload via report microservice (fallback direct S3 PUT) returns a URL; JSON uploaded; URLs in `finalization.json`.
- **Failure:** both microservice + direct PUT fail (logged); missing URL.
- **Files:** `delivery/finalization.json` (`upload_urls`); the S3 objects.
- **Logs:** `[DELIVERY] upload …`.
- **Benchmark:** upload wall time; object sizes.

## Stage 6c — Email
- **Input:** report PDF/JSON URLs + train summary.
- **Output:** one email per batch to `WAGONEYE_EMAIL_RECEIVER(+_CC)`.
- **Success:** HTTP 200 from the email API; subject `WagonEye Combined Report | v4 | $BK | wagons=N | loco=… | <date>`; exactly-once (idempotency key); recipients = production list (`WAGONEYE_EMAIL_RECEIVER*`).
- **Failure:** non-200 after retries (logged, batch outcome still persisted); duplicate email after restart (should be guarded).
- **Files:** `delivery/finalization.json` (`email_sent` + idempotency key).
- **Logs:** `[DELIVERY] email sent (<status>)`.
- **Benchmark:** email API latency.

## Stage 6d — Archive + cleanup + auto-completion
- **Input:** finalized batch.
- **Output:** batch tree archived to S3 (evidence + processed_videos + states + reports); terminal status in `processed_batches.json`; manifest terminal; temp dirs removed.
- **Success:** terminal status ∈ {`completed`,`completed_partial`,`report_failed`,`failed_no_global_state`,`failed`}; exactly-once upload+email; next train starts clean (no duplicate work on re-poll).
- **Failure:** batch stuck non-terminal; duplicate processing on re-poll; temp dirs not cleaned.
- **Files:** `processed_batches.json`; `$B/manifest.json` (terminal); `$B/archive/`.
- **Logs:** `[BATCH] … completed/…`; finalization lines.
- **Benchmark:** total end-to-end wall time (raw/trimmed → email); peak RSS; peak CPU%.

---

## Cross-cutting: production-comparison acceptance
For each feature, compare v5 vs the production Train-Inspection-Engine output on
the **same trimmed clips** (see `DOOR_PRODUCTION_COMPARISON_CHECKLIST.md`;
replicate the pattern for OCR/Load/Damage). Align wagons by **rake position**
(v5 uses one GlobalTrainState; production counts per camera). Record:
door_state, wagon_number (+prefix-manip), loco_number, load_status, top/side
damage booleans, evidence frame selection, and any false pos/neg. Overall
per-feature agreement % + explanations for every mismatch feed the sign-off.
