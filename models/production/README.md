# models/production/ — PRODUCTION feature models (milestone 1, authoritative)

These are the **production `.pt` weights** that WagonEye v5 milestone 1 uses to
reproduce production feature behaviour **exactly**. They are the source of truth
for Stage 3 (feature inference). The v4-native models in `../features/` are
**shelved** until the post-milestone model-swap phase.

> **The `.pt` files are intentionally NOT committed to this repo.** This
> directory ships with documentation only. Copy the real weights here on EC2
> (or set `WAGONEYE_PROD_MODELS_DIR`) from `s3://wagon-eye-models/`.

## How the code uses this directory

- Path: `core.config.PROD_MODELS_DIR` — env `WAGONEYE_PROD_MODELS_DIR`, default
  `<repo>/models/production`.
- Registry + loader: `core/production_models.py` — resolves each `(feature,
  camera)` pair to a filename and loads it **cached, once per process**.
- **Graceful failure (no fake inference):** if a model file is absent, loading
  raises `MissingProductionModel("Production model not found:
  models/production/<name>.pt")`. The owning processor catches it and marks that
  feature `NO_DATA` for every wagon (recording the reason). The batch still
  reconstructs, materializes, fuses and reports.
- **Zero-code EC2 activation:** drop the real `.pt` files here and the pipeline
  runs unchanged — no code edits, no config edits.

## Staging on EC2

```bash
aws s3 cp s3://wagon-eye-models/side_damage.pt        models/production/
aws s3 cp s3://wagon-eye-models/top_left_damage.pt    models/production/
aws s3 cp s3://wagon-eye-models/right_top_damage.pt   models/production/
aws s3 cp s3://wagon-eye-models/wagon_number.pt       models/production/
aws s3 cp s3://wagon-eye-models/ltop.pt               models/production/
aws s3 cp s3://wagon-eye-models/top_classification.pt models/production/
# quick check (prints {filename: present}):
python -c "from core import production_models as P; print(P.status())"
```

## Required models

> Class names below are the **expected** production vocabularies; verify them on
> EC2 with `YOLO(path).names` when the real weights are present. Exact class
> names never differ from what the audit records, but this note keeps the
> milestone honest — no behaviour is assumed from a model we cannot yet load.

### 1. `side_damage.pt`
- **Filename:** `side_damage.pt`
- **Purpose:** side-camera **door state** (open/close) **and side body damage** — production runs ONE side model that emits both.
- **Production stage:** Stage 3 — Door + Side Damage (side cameras).
- **Expected input:** per-wagon side-camera JPEG frames — `wagon_cache/GW_n/{right_up,left_up}/frame_*.jpg` (stable interior, edge frames skipped).
- **Expected output:** YOLO **detect** boxes, classes `{damage, door_open, door_close}`.
- **Loaded by:** `features/door` (both side cameras) and `features/side_damage` (shares the same per-frame detections — one model pass yields door + damage).
- **Notes:** run at detection conf **0.85 (RIGHT_UP) / 0.88 (LEFT_UP)** [notebook-authoritative]; band `gap_tolerance=5`; `door_status = "open" if door_open bands else "closed"`; annotate colors door_open=red, door_close=green, damage=orange.

### 2. `top_left_damage.pt`
- **Filename:** `top_left_damage.pt`
- **Purpose:** 4-class top-view damage for the **LEFT** top camera.
- **Production stage:** Stage 3 — Top Damage (`LEFT_UP_TOP`).
- **Expected input:** `wagon_cache/GW_n/left_up_top/frame_*.jpg` (stable interior).
- **Expected output:** YOLO **detect** boxes, classes `{body_dmg, body_dmg_probable, floor_dmg, floor_dmg_probable}`.
- **Loaded by:** `features/damage` for camera `LEFT_UP_TOP`.
- **Notes:** detection conf **0.70**; confirmed `{body_dmg, floor_dmg}` vs probable `{*_probable}`; band `gap_tolerance=5`.

### 3. `right_top_damage.pt`
- **Filename:** `right_top_damage.pt`
- **Purpose:** 4-class top-view damage for the **RIGHT** top camera.
- **Production stage:** Stage 3 — Top Damage (`RIGHT_UP_TOP`).
- **Expected input:** `wagon_cache/GW_n/right_up_top/frame_*.jpg` (stable interior).
- **Expected output:** YOLO **detect** boxes, classes `{body_dmg, body_dmg_probable, floor_dmg, floor_dmg_probable}`.
- **Loaded by:** `features/damage` for camera `RIGHT_UP_TOP`.
- **Notes:** identical semantics to `top_left_damage.pt`; conf **0.70**. Top damage fuses as `DAMAGE` if **either** top camera confirms.

### 4. `wagon_number.pt`
- **Filename:** `wagon_number.pt`
- **Purpose:** detect the **wagon-number plate bounding box** (11-digit) on the master camera.
- **Production stage:** Stage 3 — Wagon OCR (`RIGHT_UP`).
- **Expected input:** `wagon_cache/GW_n/right_up/frame_*.jpg` (WAGON-class wagons only; engine/brakevan skipped).
- **Expected output:** YOLO **detect** boxes locating the number plate; crops are fed to EasyOCR.
- **Loaded by:** `features/ocr` (RIGHT_UP).
- **Notes:** detector conf **0.40**, band `gap_tolerance=8`; crop padding 0.25; EasyOCR digit-only, 3×primary/1×fallback runs, row-clustering, temporal voting; **prefix manipulation ON** (Indian-Railways 10–39 rule; emit `is_manipulated` + `original_number`) [notebook-authoritative].

### 5. `ltop.pt`
- **Filename:** `ltop.pt`
- **Purpose:** top-view **classification** for the LEFT top camera — emits `wagon_empty` / `wagon_loaded` (+ engine/brakevan/track). Drives **LOAD** status (production has no dedicated load model; load = classification).
- **Production stage:** Stage 3 — Load (`LEFT_UP_TOP`, supporting authority).
- **Expected input:** `wagon_cache/GW_n/left_up_top/frame_*.jpg`.
- **Expected output:** YOLO **classify** top-1 among `{wagon_empty, wagon_loaded, engine, brakevan, empty_track}` (raw names may be `wagon_filled`/`track`; canonicalized via the production class aliases).
- **Loaded by:** `features/load` for camera `LEFT_UP_TOP`.
- **Notes:** load = LOADED if `wagon_loaded` wins the majority vote; classification conf **0.80**; every-other-frame vote, 5 edge frames skipped. Loaded with ultralytics `task="classify"`.

### 6. `top_classification.pt`
- **Filename:** `top_classification.pt`
- **Purpose:** top-view **classification** for the RIGHT top camera — same vocabulary as `ltop.pt`. **Authoritative** for LOAD (RIGHT_UP_TOP), `ltop.pt` is the fallback.
- **Production stage:** Stage 3 — Load (`RIGHT_UP_TOP`, authoritative).
- **Expected input:** `wagon_cache/GW_n/right_up_top/frame_*.jpg`.
- **Expected output:** YOLO **classify** top-1 among `{wagon_empty, wagon_loaded, engine, brakevan, empty_track}`.
- **Loaded by:** `features/load` for camera `RIGHT_UP_TOP`.
- **Notes:** same voting/thresholds as `ltop.pt`; RIGHT_UP_TOP wins when present, else LEFT_UP_TOP fallback (matches production/fusion authority). Loaded with `task="classify"`.

## Not a production-only model (already staged)

- **Loco-number OCR** locates its bands from the `locono` class emitted by
  `../reconstruction/right_up_gap.pt` (already present). It does **not** require
  a separate file here. EasyOCR then reads the 5-digit number (per loco band).

## Also required, but NOT here (already staged elsewhere)

- **Gap / counting / reconstruction** models live in `../reconstruction/`
  (`right_up_gap.pt`, `left_up_gap.pt`, `top_gap.pt`, `side_classification.pt`) —
  production equivalents `right_gap_1.pt` / `left_gap_det.pt` / `top_gap_2.pt`.
- **Train extraction** classifiers live in `../extraction/`
  (`side_classification.pt`, `top_classification.pt`) — see `train_extraction/`.
