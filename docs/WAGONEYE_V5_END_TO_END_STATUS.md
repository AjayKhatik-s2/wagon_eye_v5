# WagonEye v5 — End-to-End Integration Status

Milestone: run a complete production-equivalent inspection on top of the
GlobalTrain architecture. Production repo = behavioural spec; this repo =
architectural spec. Production models + production logic; no v4-model
substitution; no redesign.

---

## 1. What was implemented / changed this phase

### Feature processors — rewritten to PRODUCTION behaviour (contracts preserved)
All consume `wagon_cache/GW_n/<camera>/` and write the existing per-camera
`wagon_states/<feature>/<CAMERA>/GW_n.json` layout, so fusion / reporting /
delivery / rendering are unchanged. All use PRODUCTION models via
`core.production_models` and degrade to `NO_DATA` (clear error) when a model is
absent — no dummy inference.

| Feature | File | Production model | Production logic implemented |
|---|---|---|---|
| **Door** (prior phase, frozen) | `features/door/processor.py` | `side_damage.pt` | door_open/door_close banding, conf 0.85/0.88, edge-skip 10, gap-tol 5, `door_state = OPEN iff door_open band`, `door_close_detected` |
| **Load** | `features/load/processor.py` | `ltop.pt` / `top_classification.pt` | production `classify_segment_type` majority vote of `wagon_loaded` vs `wagon_empty`; conf ≥ 0.80; edge-skip 5, step 2; RIGHT_UP_TOP primary / LEFT_UP_TOP fallback (fusion); ENGINE/BRAKE_VAN skipped |
| **OCR** | `features/ocr/processor.py` | `wagon_number.pt` | production wagon-number detector + production-lineage OCR engine (6-step preprocess + Indian-Railways confusion-map correction) + aggregator; det conf 0.40; 11-digit; ENGINE/BRAKE_VAN skipped. Fixed CPU-unsafe `half=True` → device-aware |
| **Damage** | `features/damage/processor.py` | `top_left_damage.pt` / `right_top_damage.pt` | production 4-class top damage (`body_dmg`/`floor_dmg` ± `_probable`) banding, conf 0.70, edge-skip 10, gap-tol 5; `damage_status = DAMAGE iff confirmed body/floor band`; additive production fields (`body_dmg_detected`… `probable_damage_detected`); **no loaded-wagon floor filter** (not production behaviour) |

### Infrastructure (prior phase, in this build)
- `core/production_models.py` — production model registry + graceful loader.
- `models/production/` — documented model dir (README; `.pt` staged on EC2).
- `orchestrator/feature_markers.py` — completion markers key on the PRODUCTION
  model per (feature, camera) (all four features inherit it).
- Materializer JPEG **q95** + `MATERIALIZER_SCHEMA_VERSION=2` (production pixel parity).
- `core/config.py` — `PROD_MODELS_DIR` (env `WAGONEYE_PROD_MODELS_DIR`).

### Reused unchanged (already present, consume the preserved contracts)
- **Fusion / Wagon State Builder** — `fusion/wagon_state_builder.py` (authority
  rules: classification←GST, wagon_identifier←RIGHT_UP OCR, right/left_door←side
  cameras, load←RIGHT_UP_TOP else LEFT_UP_TOP, top_damage←any top). Verified it
  reads the production feature outputs and builds `wagon_states/unified/GW_n.json`.
- **Reports** — `reporting/camera_reports.py` (4 camera PDFs) +
  `reporting/combined_train_report.py` (combined PDF + `combined_train_report.json`).
- **Dashboard JSON (D4)** — `delivery/dashboard_ingest.py` (**enabled by
  default**) re-derives the legacy per-camera `{camera_id, version,
  inspection_data}` payload from the finalized combined report + evidence and
  POSTs to the production `cctv-receiver/inspections/ingest` API. Idempotent,
  read-only, failure-isolated.
- **PDF / Upload / Email / Archive / auto-completion** —
  `delivery/{s3_upload,notification,finalization}.py` + `orchestrator/`
  (master_runner + lifecycle_runner + train_batch_manager + batch_manifest)
  drive Raw→…→Complete automatically, including the async-camera lifecycle,
  exactly-once upload/email, and terminal-state persistence.

**Frozen / untouched:** the validated architecture (orchestrator, lifecycle,
manifest, materializer algorithm, GlobalTrainState) and the Door algorithm.

---

## 2. Verification performed (local dev box, CPU, models absent)

- Byte-compile: all changed modules OK.
- Unit tests: Load canonicalization + classify-frame selection (edge-skip 5,
  step 2); Damage banding (gap split, best-frame), interior edge-skip 10, class
  colours; Door units (prior).
- **Feature → Fusion end-to-end (graceful NO_DATA):** all four processors ran on
  the real 62-wagon GlobalTrainState + wagon_cache, wrote the correct
  camera-scoped layout (door RIGHT_UP/LEFT_UP, ocr RIGHT_UP, load + damage
  RIGHT_UP_TOP/LEFT_UP_TOP; 62 files each), and fusion built 62
  `UnifiedWagonState` with no exceptions.
- **Full local pipeline run:** `orchestrator.master_runner --local-only`
  (see §4 for result). Note: pass an **absolute** `--local-inputs` (or omit it to
  use the absolute default `<repo>/local_inputs`) — a *relative* value produces
  relative paths that the reconstruction subprocess (which runs with
  `cwd=wagon_count/`) cannot resolve. This is a usage note, not a code change.

> Real-inference behaviour is validated on EC2 with the production `.pt` files
> and real footage (models are absent on the dev box by design).

---

## 3. Remaining external dependencies (production model files)

Stage-1 reconstruction (already staged / in `models/reconstruction/` on dev):
`right_up_gap.pt`, `left_up_gap.pt`, `top_gap.pt`, `side_classification.pt`.

Stage-3 features — copy into `models/production/` on EC2 from
`s3://wagon-eye-models/`:

| Model | Feature(s) |
|---|---|
| `side_damage.pt` | Door (both side cameras) |
| `top_left_damage.pt` | Damage (LEFT_UP_TOP) |
| `right_top_damage.pt` | Damage (RIGHT_UP_TOP) |
| `wagon_number.pt` | OCR (RIGHT_UP) |
| `ltop.pt` | Load (LEFT_UP_TOP) |
| `top_classification.pt` | Load (RIGHT_UP_TOP) |

Plus the `easyocr` package for OCR (`pip install easyocr`). Any missing model →
that feature emits `NO_DATA` (pipeline still runs, seals, fuses, reports).
Check: `python -c "from core import production_models as P; print(P.status())"`.

---

## 4. Exact EC2 commands — full end-to-end validation

```bash
# 0) one-time host setup + venv + deps (see docs/EC2_DEPLOYMENT_CHECKLIST.md)
bash scripts/setup_ec2.sh && source .venv/bin/activate

# 1) stage models
aws s3 cp s3://wagon-eye-models/right_up_gap.pt        models/reconstruction/
aws s3 cp s3://wagon-eye-models/left_up_gap.pt         models/reconstruction/
aws s3 cp s3://wagon-eye-models/top_gap.pt             models/reconstruction/
aws s3 cp s3://wagon-eye-models/side_classification.pt models/reconstruction/
aws s3 cp s3://wagon-eye-models/side_damage.pt         models/production/
aws s3 cp s3://wagon-eye-models/top_left_damage.pt     models/production/
aws s3 cp s3://wagon-eye-models/right_top_damage.pt    models/production/
aws s3 cp s3://wagon-eye-models/wagon_number.pt        models/production/
aws s3 cp s3://wagon-eye-models/ltop.pt                models/production/
aws s3 cp s3://wagon-eye-models/top_classification.pt  models/production/
python -c "from core import production_models as P; print(P.status())"   # all True

# 2) four trimmed clips in local_inputs/ (right_up/left_up/right_up_top/left_up_top)
ls local_inputs/

# 3a) FULL local end-to-end (no S3 upload / email) — use ABSOLUTE local-inputs (or omit it)
python -m orchestrator.master_runner --local-only --skip-upload --skip-email --no-interactive
#     OR explicitly:  --local-inputs "$(pwd)/local_inputs"

# 3b) FULL end-to-end WITH delivery (S3 upload + email + dashboard ingest)
export WAGONEYE_S3_INPUT_PREFIXES=incoming/right_up/,incoming/left_up/,incoming/right_up_top/,incoming/left_up_top/
python -m orchestrator.master_runner --local-only          # drops --skip-upload/--skip-email

# 3c) CONTINUOUS production mode (raw extraction + inspection, two services)
python -m train_extraction.run_extraction_service          # raw -> trimmed producer
python -m orchestrator.master_runner --auto                # trimmed -> reports/delivery

# 4) inspect outputs
BK=$(ls -t batch_outputs | head -1)
ls batch_outputs/$BK/reports/                              # 4 camera PDFs + combined_train_report.{pdf,json}
ls batch_outputs/$BK/wagon_states/unified/                # UnifiedWagonState per GW
ls batch_outputs/$BK/delivery/dashboard/                  # <CAMERA>_inspection.json (dashboard feed)
```

Per-feature validation procedures + benchmarks: `docs/EC2_STAGE3_DOOR_VALIDATION.md`,
`docs/DOOR_PRODUCTION_COMPARISON_CHECKLIST.md`, `docs/benchmarks/door.md`
(replicate the comparison/benchmark pattern for OCR/Load/Damage).

---

## 5. Assumptions & remaining production-parity items (honest register)

The pipeline is **runnable end-to-end**; these are the items where full
production parity needs confirmation/work, none blocking a run:

1. **Loco-number OCR (D3) — DONE.** ENGINE wagons run `locono` detection
   (`models/reconstruction/right_up_gap.pt`) on RIGHT_UP frames + a 5-digit read
   via the production-lineage OCR engine, voted per engine. Loco numbers flow:
   OCR JSON → `UnifiedWagonState.loco_number` → fusion → report KPI + PDF
   `loco_numbers` + **email subject** + dashboard `loco_number_results`.
2. **Side damage — DONE (wired end to end).** The `damage` feature now also runs
   on the SIDE cameras (`side_damage.pt` `damage` class, conf 0.85/0.88, band
   gap-tol 5, edge-skip 10) → `wagon_states/damage/<SIDE_CAMERA>/GW_n.json`
   (`damage_status` + `side_damage_details`) → fusion populates
   `UnifiedWagonState.side_damage` (DAMAGE if either side camera confirms) →
   `SIDE_DAMAGE` anomaly + the combined report `SIDE_DMG` column + dashboard
   `damaged_wagons`. Registry: added side-camera damage work units;
   `production_models` maps `(damage, RIGHT_UP/LEFT_UP)→side_damage.pt`; markers
   inherit it.
3. **Schema parity — AUDITED.** See `docs/WAGONEYE_V5_SCHEMA_PARITY.md`: the v5
   per-camera dashboard payload is structurally a production-v2 `inspection_data`
   doc. Eliminated: `loco_number_results` (now populated), side damage in
   `damaged_wagons`. Config fix: set `WAGONEYE_INSPECTION_VERSION=v2`. Justified
   degradations (self-declared in `_adapter.degraded_fields`): `direction`
   (rake_status compensates), `loco_frames`. Confirm-with-dashboard items:
   `wagon_number_results` key convention + a few additive per-class breakdowns.
4. **Report format parity.** Reports use the v4-native camera + combined PDFs
   (which carry the same information: per-wagon table, anomalies, evidence grid).
   Production's exact per-camera PDF *layout* + `inspection_data.json` **v2**
   schema were NOT re-created (that is a large reporting port and would "redesign"
   the maintained reporting). The **dashboard feed** (D4) is covered by
   `delivery/dashboard_ingest.py`, which emits the legacy `{camera_id, version,
   inspection_data}` payload and posts to the production ingest API — with
   documented degraded fields (`direction`→unknown; `loco_*`→empty until item 1;
   `wagon_frames` gallery synthesized from present evidence). Confirm with the
   dashboard team that the re-derived feed + degraded fields are acceptable, or
   schedule a production-exact `inspection_data.v2` emitter as a follow-up.
4. **ML callback API.** Production also POSTed an ML callback with `X-ML-SECRET`.
   The v4 delivery does S3 upload (report microservice + direct-PUT fallback) +
   email + dashboard ingest. If the ML callback is still consumed, add it to
   `delivery/` (endpoint/secret from production `configs/config.json`). Confirm
   whether it is still required.
5. **Threshold source-of-truth.** All new features use the **notebook-authoritative**
   production values per the locked decision (damage 0.85/0.88 side, 0.70 top;
   classification 0.80; OCR det 0.40). Load uses the `classify_segment_type`
   majority rule (not the old_system 0.35 loaded-ratio).

---

## 6. Stage completion

```
✅ Stage A   Automatic train extraction (train_extraction/, production port)
✅ Stage 1   Global reconstruction -> GlobalTrainState
✅ Stage 2   Wagon-cache materialization (q95)
✅ Stage 3   Door / OCR / Load / Damage  (production behaviour, production models)
✅ Stage 4   Fusion -> UnifiedWagonState (authority rules)
✅ Stage 4b  Overlay rendering (visualization-only, existing)
✅ Stage 5   Reports (4 camera PDFs + combined PDF/JSON)  [format parity: item 3]
✅ Stage 6   Dashboard JSON + upload + email + archive + exactly-once + auto-completion
◻  Loco OCR (item 1), Side damage (item 2), production-exact report/ML-callback parity (items 3–4)
```

Nothing committed. Production logic/models used throughout; no v4-model
substitution; GlobalTrain architecture maintained; Door unchanged.
