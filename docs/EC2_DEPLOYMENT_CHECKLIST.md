# EC2 Deployment Checklist — WagonEye v5 (milestone 1: Door validation build)

Scope: bring up **this exact build** on EC2 to validate the **Door** processor
against the production pipeline. This complements the repo's existing
`DEPLOYMENT.md` (full production runbook) and `scripts/setup_ec2.sh` — it focuses
on the milestone-1 additions (production models in `models/production/`, the
q95 materializer, the production-model markers).

> Do not proceed to OCR/Load/Damage. This build's only new feature is Door.

---

## 0. Instance sizing

| | Minimum | Recommended for validation |
|---|---|---|
| Type | any x86-64 with 8 GB RAM | GPU box (`g4dn.xlarge`+) |
| Why | CPU works but is slow | Stage-1 reconstruction took **~47 min on CPU** for ~264 s of 4-camera video in dev; door inference adds per-frame YOLO on both side cameras. A T4 GPU cuts this dramatically. |
| Disk | ≥ 30 GB free | wagon_cache writes ~15k JPEGs/train (q95) + evidence + logs |

CPU-only is supported (`WAGONEYE_DEVICE=cpu`); expect long runtimes — capture them in `docs/benchmarks/door.md`.

---

## 1. Repository clone / update

```bash
# fresh:
git clone <repo-url> wagon_eye_v4_new
cd wagon_eye_v4_new
git checkout <this-build-branch-or-sha>       # the milestone-1 Door build

# update an existing checkout:
cd wagon_eye_v4_new && git fetch && git checkout <this-build-branch-or-sha>
git rev-parse HEAD                            # record the SHA in the benchmark doc
```

Confirm the milestone-1 files are present:
```bash
ls core/production_models.py \
   models/production/README.md \
   features/door/processor.py \
   docs/EC2_STAGE3_DOOR_VALIDATION.md
```

---

## 2. Python environment + dependencies

Use the provided setup (idempotent; installs OS libs, venv, deps, GPU torch if present):
```bash
bash scripts/setup_ec2.sh          # or WAGONEYE_FORCE_CPU=1 bash scripts/setup_ec2.sh
source .venv/bin/activate
```
Manual equivalent:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt    # ultralytics, opencv-headless, torch, easyocr, reportlab, boto3, filterpy, scipy
ffmpeg -version                    # required by Stage 1 / rendering
```
Python **3.10+**. (Dev validated on 3.14 + ultralytics 8.4; keep the EC2 interpreter ≥3.10.)

---

## 3. Model placement

`setup_ec2.sh` creates `models/reconstruction/` and `models/features/` but **not**
`models/production/` (it ships with the repo via its README). Stage the weights:

```bash
mkdir -p models/production          # exists after clone (has README.md); ensure it's there

# --- Stage-1 reconstruction/counting models (REQUIRED for any run) ---
aws s3 cp s3://wagon-eye-models/right_up_gap.pt        models/reconstruction/   # or right_up_wagon_gap.pt
aws s3 cp s3://wagon-eye-models/left_up_gap.pt         models/reconstruction/   # or left_up_wagon_gap.pt
aws s3 cp s3://wagon-eye-models/top_gap.pt             models/reconstruction/   # prod: top_gap_2.pt
aws s3 cp s3://wagon-eye-models/side_classification.pt models/reconstruction/

# --- Door validation model (REQUIRED for Door) ---
aws s3 cp s3://wagon-eye-models/side_damage.pt         models/production/

# --- (later milestones; NOT needed for Door) ---
# aws s3 cp s3://wagon-eye-models/wagon_number.pt        models/production/
# aws s3 cp s3://wagon-eye-models/top_left_damage.pt     models/production/
# aws s3 cp s3://wagon-eye-models/right_top_damage.pt    models/production/
# aws s3 cp s3://wagon-eye-models/ltop.pt                models/production/
# aws s3 cp s3://wagon-eye-models/top_classification.pt  models/production/
```

Verify presence + classes:
```bash
python -c "from core import production_models as P; print(P.status())"
# side_damage.pt should be True; others may be False for Door-only validation.
python -c "from ultralytics import YOLO; print(YOLO('models/production/side_damage.pt').names)"
# expect door + damage classes, e.g. {..., 'door_open', 'door_close', 'damage'}
python -c "from ultralytics import YOLO; print(YOLO('models/reconstruction/right_up_gap.pt').names)"
# expect {..., 'gap', 'locono', 'engine_head'}
```

> Model files are NOT committed to git. If a `.pt` is absent the pipeline runs
> and emits `NO_DATA` for that feature (clear error), never a fake result.

---

## 4. Input videos

Door validation runs from four **trimmed** per-train clips in `local_inputs/`
(the same four the dev validation used, or a fresh train):
```bash
ls local_inputs/     # right_up.mp4  left_up.mp4  right_up_top.mp4  left_up_top.mp4
```
Filenames must contain `right_up` / `left_up` / `right_up_top` / `left_up_top`.
(Raw→trimmed extraction is a separate producer service and is NOT needed for
this local Door validation.)

---

## 5. Environment variables

For Door validation, minimal overrides (all optional; defaults reproduce prod behaviour):
```bash
export WAGONEYE_DEVICE=cuda          # or cpu; unset = auto-detect
export WAGONEYE_LOG_LEVEL=INFO       # DEBUG for verbose per-wagon lines
# models/production is the default; override only if you stage elsewhere:
# export WAGONEYE_PROD_MODELS_DIR=/data/wagon_eye/models/production
```
See `deploy/wagon-eye.env.example` for the full list (S3, email, lifecycle
deadlines) — **not required** for a local `--local-only` Door run.

---

## 6. Expected directory structure (after a Door validation run)

```
wagon_eye_v4_new/
├── models/
│   ├── reconstruction/  right_up_gap.pt left_up_gap.pt top_gap.pt side_classification.pt
│   ├── features/        (v4-native; unused in milestone 1)
│   └── production/      side_damage.pt   (+ README.md)
├── local_inputs/        right_up.mp4 left_up.mp4 right_up_top.mp4 left_up_top.mp4
├── logs/                wagon_eye.log
└── batch_outputs/<batch_key>/
    ├── downloads/
    ├── global_state/    global_train_state.json  per_camera_tracking.json
    ├── wagon_cache/     GW_1/{right_up,left_up,right_up_top,left_up_top}/frame_*.jpg
    ├── wagon_states/
    │   ├── door/RIGHT_UP/GW_*.json
    │   ├── door/LEFT_UP/GW_*.json
    │   └── .features/{RIGHT_UP,LEFT_UP}/door.json     (completion markers)
    ├── evidence/        GW_*/door/{RIGHT_UP,LEFT_UP}/{right_best,right_crop,metadata}.jpg/json
    ├── reports/         (door-only; ocr/load/damage disabled)
    └── manifest.json
```

---

## 7. Required AWS permissions

| Purpose | Actions | Resource |
|---|---|---|
| Pull models | `s3:GetObject`, `s3:ListBucket` | `s3://wagon-eye-models/*` |
| (Auto mode only) discover source videos | `s3:GetObject`, `s3:ListBucket` | input bucket/prefixes |
| (Auto mode only) upload reports/state | `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` | `s3://biro-wagon-report-biro-copy/*` |

For **local Door validation** (`--local-only`, `--skip-upload`, `--skip-email`)
only the **model-pull** permission is needed (and even that only for the initial
`aws s3 cp`). Prefer an **EC2 IAM instance role** over static keys.

---

## 8. Verification commands (smoke, pre-validation)

```bash
source .venv/bin/activate
# imports + device + project root
python - <<'PY'
import importlib
for m in ("cv2","numpy","torch","ultralytics","reportlab","boto3"):
    importlib.import_module(m); print("ok", m)
from core import config as CFG, production_models as PM
print("device:", CFG.resolve_device(), "| root:", CFG.PROJECT_ROOT)
print("prod models:", PM.status())
PY
# byte-compile the milestone-1 modules
python -m py_compile core/production_models.py features/door/processor.py \
    orchestrator/feature_markers.py materializer/wagon_cache_builder.py core/constants.py
# confirm the two infra fixes are live
python -c "from core import constants as C; from materializer import wagon_cache_builder as M; \
print('JPEG_QUALITY', C.JPEG_QUALITY, 'MAT_SCHEMA', M.MATERIALIZER_SCHEMA_VERSION)"   # 95 / 2
python -c "from orchestrator import feature_markers as FM; \
print(FM.compute_identity(camera_id='RIGHT_UP',feature='door',source_key=None,etag=None,global_state_version='v',feat_models_dir='x')['model_filename'])"  # side_damage.pt
```

Full Door run + inspection commands are in **`docs/EC2_STAGE3_DOOR_VALIDATION.md`**.

---

## 9. Expected outputs (high level)

- `global_state/global_train_state.json` with `total_wagons > 0` (Stage 1 OK).
- `wagon_cache/GW_*/…` populated for all 4 cameras (Stage 2 OK, q95 JPEGs).
- `wagon_states/door/{RIGHT_UP,LEFT_UP}/GW_*.json` — one per wagon per side camera, `status: OK` (with `side_damage.pt` present).
- `evidence/GW_*/door/…` for wagons with a door detection.
- Completion markers under `wagon_states/.features/` keyed on `side_damage.pt`.

---

## 10. Troubleshooting

| Symptom | Cause / action |
|---|---|
| `Production model not found: models/production/side_damage.pt` in door output | model not staged → `aws s3 cp s3://wagon-eye-models/side_damage.pt models/production/`. Every wagon is `NO_DATA` until then (by design). |
| Stage 1 aborts `failed_no_global_state` | fewer than needed cameras, or `right_up_gap.pt`/`side_classification.pt` missing in `models/reconstruction/`. Check `global_state/stage1_wagon_count.log`. |
| `ModuleNotFoundError: global_train_state` | reconstruction must run with `cwd=wagon_count/` (the orchestrator does this; only relevant if invoking `run_global_count.py` by hand). |
| Very slow run | CPU box — expect long runtimes; set `WAGONEYE_DEVICE=cuda` on a GPU host. Record timings in `docs/benchmarks/door.md`. |
| `door.json` has `status: OK` but `door_state: CLOSED` everywhere | verify `side_damage.pt` emits `door_open`/`door_close` (`YOLO(...).names`); a class-name mismatch yields all-closed. |
| ffmpeg errors in Stage 1/rendering | `ffmpeg` not installed — `apt-get install ffmpeg` / `yum install ffmpeg`. |
| wagon_cache looks stale after the q95 change | expected: `MATERIALIZER_SCHEMA_VERSION=2` forces a rebuild of any q90 cache on the next run. |
| easyocr import error | not needed for Door; ignore until OCR milestone. |
