# WagonEye v5 — Production Sign-Off Checklist

Final gate before declaring v5 production-ready. Complete on the EC2 host after
following `WAGONEYE_V5_EC2_VALIDATION_GUIDE.md`. Every item = a checkbox + how to
verify + the pass bar. Nothing is a code change; this is acceptance only.

**Run under test** — batch key `$BK = ______________`  ·  build SHA `______________`
·  host `______________`  ·  device `cuda / cpu`  ·  date (UTC) `______________`.

---

## A. Prerequisites

- [ ] **Production models present** — `python -c "from core import production_models as P; print(P.status())"` → all six `True` (`side_damage, top_left_damage, right_top_damage, wagon_number, ltop, top_classification`). Class names spot-checked with `YOLO(path).names`.
- [ ] **Reconstruction models present** — `models/reconstruction/{right_up_gap,left_up_gap,top_gap,side_classification}.pt` all present; `right_up_gap.pt` emits `locono` (loco OCR). (Extraction classifiers present too if Service 1 is in scope.)
- [ ] **Environment set** — `WAGONEYE_INSPECTION_VERSION=v2`; device correct; delivery vars set if delivery is in scope; IAM/S3 reachable (`aws s3 ls s3://wagon-eye-models/`).

## B. Pipeline execution

- [ ] **Pipeline completes** — `master_runner` exits 0; batch reaches a terminal status in `processed_batches.json` (`completed` / `completed_partial`); no unhandled exception in `logs/wagon_eye.log`.
- [ ] **Stage 1 — GlobalTrainState** — `global_train_state.json` `total_wagons > 0`, `master_camera=RIGHT_UP`; classification distribution sane (engine/wagon/brake_van).
- [ ] **Stage 2 — materialization** — `wagon_cache/` populated for all present cameras; on-disk JPEG count == `total_frames`; JPEG quality q95.
- [ ] **Stage 3 — features produce results** (with models present, not `NO_DATA`):
  - [ ] Door — `door/{RIGHT_UP,LEFT_UP}/GW_*.json` `status=OK`, doors ∈ {OPEN,CLOSED}.
  - [ ] OCR — `ocr/RIGHT_UP/GW_*.json` wagon numbers on WAGONs; **loco numbers on ENGINE wagons**.
  - [ ] Load — `load/{RIGHT_UP_TOP,LEFT_UP_TOP}/GW_*.json` load ∈ {LOADED,EMPTY}.
  - [ ] Damage — `damage/<CAMERA>/GW_*.json` for **all 4 cameras** (top 4-class + **side damage**).
- [ ] **Stage 4 — fusion** — `wagon_states/unified/GW_*.json` for every wagon; `side_damage` + `loco_number` populated where applicable; anomalies + `result_state` set; no false OK on missing/pending.

## C. Outputs

- [ ] **Reports generated** — `reports/{right_up,left_up,right_up_top,left_up_top}_report.pdf` + `reports/combined_train_report.pdf` all open and render (title/KPI/wagon table with TOP_DMG + SIDE_DMG columns + evidence grid; loco numbers in KPI).
- [ ] **inspection_data validated** — per-camera dashboard payload (`delivery/dashboard/<CAMERA>_inspection.json`) validated against `WAGONEYE_V5_SCHEMA_PARITY.md`: `version=v2`; RIGHT_UP `loco_number_results` populated; side `damaged_wagons` includes side damage; degraded fields self-declared in `_adapter.degraded_fields`; no fabricated data.
- [ ] **Dashboard updated** — ingest POST returned success for each camera (idempotent); `finalization.json.dashboard_ingested` recorded. (Or explicitly deferred with the dashboard team's sign-off on the payload.)
- [ ] **PDF generated** — `combined_train_report.pdf` non-empty, opens, matches the report expectations above; `combined_train_report.json` schema `wagon_eye.combined_report.v4` valid.
- [ ] **Upload successful** — combined PDF + JSON in `s3://end-results/reports/$BK/`; URLs recorded in `finalization.json`. (microservice or direct-PUT fallback.)
- [ ] **Email successful** — one email per batch, HTTP 200; subject includes wagon count + **loco numbers**; production recipient list; `finalization.json.email_sent` set; no duplicate on a restart.
- [ ] **Archive successful** — batch tree (evidence + processed_videos + states + reports) archived to S3; batch marked terminal; temp dirs cleaned; a re-poll does **no** duplicate work.

## D. Benchmarks (recorded into `docs/benchmarks/`)

- [ ] **Runtime benchmark recorded** — total end-to-end wall time + per-stage `STAGE …` times + per-feature `[FEAT/…] done in …`; per-wagon averages. (Dev CPU ref: Stage 1 ≈ 47 min, Stage 2 ≈ 22 s.)
- [ ] **Memory benchmark recorded** — peak RSS (`/usr/bin/time -v` "Maximum resident set size"); per-model footprint noted.
- [ ] **CPU benchmark recorded** — peak CPU% (and GPU util/VRAM if GPU); thread/oversubscription behaviour if multi-process.

## E. Correctness vs production

- [ ] **Outputs compared against the production pipeline** — on the same trimmed clips, per aligned wagon (by rake position), agreement recorded for: door_state, wagon_number (+ prefix-manip), loco_number, load_status, top_damage (per class), side_damage, evidence frame selection. Agreement % per feature + an explanation for **every** mismatch (wagon-boundary/count differences from GlobalTrainState are expected and must be labelled as such, not as feature defects). Uses `DOOR_PRODUCTION_COMPARISON_CHECKLIST.md` (replicated per feature).

## F. Known, accepted differences (must be acknowledged, not silently passed)

- [ ] Wagon count comes from **one GlobalTrainState** (RIGHT_UP master + cross-camera gap fusion), replacing production's per-camera independent counts — accepted architectural change.
- [ ] Combined-PDF S3 object key is `combined_train_report.pdf` (v4 convention); loco numbers appear in report/email/dashboard, not the filename.
- [ ] Dashboard `direction`→"unknown" (compensated by measured `rake_status`); `loco_frames`/`total_loco_frames` empty (loco **numbers** present). Self-declared in `_adapter.degraded_fields`.
- [ ] Any `WAGONEYE_V5_SCHEMA_PARITY.md` "confirm with dashboard team" items resolved: `wagon_number_results` key convention; optional additive per-class/`door_close_detected` fields.

---

## Sign-off

| Role | Name | Verdict (PASS / PASS-WITH-NOTES / FAIL) | Date | Notes |
|---|---|---|---|---|
| ML / pipeline owner | | | | |
| Ops / deployment | | | | |
| Dashboard / backend | | | | |

**Overall verdict:** ☐ PRODUCTION-READY  ☐ CONDITIONAL (notes)  ☐ NOT READY

**Blocking issues (if any):** ____________________________________________

**Follow-ups (non-blocking):** ___________________________________________
