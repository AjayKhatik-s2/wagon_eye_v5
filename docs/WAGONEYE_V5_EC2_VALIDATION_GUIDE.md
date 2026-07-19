# WagonEye v5 — EC2 Validation Guide (step-by-step, no source reading)

Follow top to bottom on a fresh EC2 host to run and validate the complete
pipeline. You do **not** need to read any source. Commands assume you are in the
repo root with the venv active. Companion: `WAGONEYE_V5_VALIDATION_CHECKLIST.md`
(per-stage criteria), `WAGONEYE_V5_PRODUCTION_SIGNOFF.md` (final gate).

Notation: `$` = shell prompt. Placeholders in `<…>`.

---

## Step 0 — Host prep (once)

```bash
git clone <repo-url> wagon_eye_v4_new && cd wagon_eye_v4_new
git checkout <build-sha> && git rev-parse HEAD          # record the SHA
bash scripts/setup_ec2.sh                               # OS libs + venv + deps (GPU torch if present)
source .venv/bin/activate
ffmpeg -version                                         # must exist
python -c "import cv2,numpy,torch,ultralytics,reportlab,boto3,easyocr; print('deps ok')"
```

GPU strongly recommended for validation (CPU Stage-1 ≈ 47 min/train). Force
device if needed: `export WAGONEYE_DEVICE=cuda` (or `cpu`).

## Step 1 — Stage the models

```bash
mkdir -p models/reconstruction models/production models/extraction
# reconstruction (Stage 1)
for m in right_up_gap left_up_gap top_gap side_classification; do
  aws s3 cp s3://wagon-eye-models/$m.pt models/reconstruction/; done
# production features (Stage 3)
for m in side_damage top_left_damage right_top_damage wagon_number ltop top_classification; do
  aws s3 cp s3://wagon-eye-models/$m.pt models/production/; done
# extraction classifiers (only if validating Service 1)
aws s3 cp s3://wagon-eye-models/side_classification.pt models/extraction/
aws s3 cp s3://wagon-eye-models/top_classification.pt  models/extraction/

# verify
python -c "from core import production_models as P; print('production:', P.status())"   # all True
python -c "import os; print('reconstruction:', {f: os.path.exists('models/reconstruction/'+f) for f in ['right_up_gap.pt','left_up_gap.pt','top_gap.pt','side_classification.pt']})"
python -c "from ultralytics import YOLO; print('side_damage classes:', YOLO('models/production/side_damage.pt').names)"   # expect door_open, door_close, damage
python -c "from ultralytics import YOLO; print('top_classification classes:', YOLO('models/production/top_classification.pt').names)"  # expect wagon_loaded/wagon_empty (or aliases)
```

If any production model is absent, that feature emits `NO_DATA` (the pipeline
still runs); stage it before sign-off.

## Step 2 — Environment

```bash
cp deploy/wagon-eye.env.example deploy/wagon-eye.env
# minimum for validation:
export WAGONEYE_DEVICE=cuda                 # or cpu
export WAGONEYE_INSPECTION_VERSION=v2       # dashboard payload version (schema parity)
# for continuous / delivery mode also set (see the env example):
#   WAGONEYE_S3_INPUT_BUCKET, WAGONEYE_S3_INPUT_PREFIXES,
#   WAGONEYE_EMAIL_RECEIVER(+_CC), WAGONEYE_UPLOAD_API_URL, WAGONEYE_EMAIL_API_URL
```

AWS credentials: prefer the EC2 IAM instance role (no keys). Confirm:
`aws s3 ls s3://wagon-eye-models/ >/dev/null && echo "s3 ok"`.

## Step 3 — Provide input clips

Four **trimmed** train clips in `local_inputs/`, filenames containing the camera
substrings + a `YYYYMMDD_HHMMSS` stamp:

```bash
ls local_inputs/   # right_up*.mp4  left_up*.mp4  right_up_top*.mp4  left_up_top*.mp4
```

(To validate Service 1 instead, put **raw** clips in the raw bucket and run
`python -m train_extraction.run_extraction_service --once --dry-run` first.)

## Step 4 — Run the full pipeline (local, no S3 delivery)

```bash
# IMPORTANT: use the absolute default local-inputs (do NOT pass a relative path).
/usr/bin/time -v python -m orchestrator.master_runner \
    --local-only --skip-upload --skip-email --no-interactive \
    2>&1 | tee logs/validation_run.log
echo "exit=$?"
BK=$(ls -t batch_outputs | grep -E '^[0-9]{8}_' | head -1); export B=batch_outputs/$BK
echo "batch=$BK  dir=$B"
```

## Step 5 — Run WITH delivery (upload + email + dashboard) — optional

```bash
# ensure the delivery env vars from Step 2 are set, then drop the skips:
python -m orchestrator.master_runner --local-only --no-interactive 2>&1 | tee logs/validation_delivery.log
```

Continuous production mode (two services):
```bash
python -m train_extraction.run_extraction_service        # terminal 1 (raw -> trimmed)
python -m orchestrator.master_runner --auto              # terminal 2 (trimmed -> reports)
```

## Step 6 — Inspect outputs (per stage)

```bash
# Stage 1
python -m json.tool $B/global_state/global_train_state.json | grep -E '"total_wagons"|"master_camera"'
# Stage 2
find $B/wagon_cache -name '*.jpg' | wc -l ; du -sh $B/wagon_cache
# Stage 3 (status distribution per feature/camera)
for f in door ocr load damage; do for d in $B/wagon_states/$f/*/; do \
  echo "$d: $(ls $d/*.json 2>/dev/null | wc -l) files"; done; done
python -m json.tool $B/wagon_states/door/RIGHT_UP/GW_1.json
python -m json.tool $B/wagon_states/ocr/RIGHT_UP/GW_1.json
python -m json.tool $B/wagon_states/damage/RIGHT_UP/GW_1.json    # side damage (side=true)
# Stage 4
python -m json.tool $B/wagon_states/unified/GW_1.json | grep -E 'side_damage|loco_number|top_damage|load_status|door|anomalies|result_state'
# Stage 4b / 5
ls $B/processed_videos/ ; ls $B/reports/
python -m json.tool $B/reports/combined_train_report.json | grep -E 'schema|total_wagons|loco_numbers|top_damaged|side_damaged'
# Stage 6
ls $B/delivery/dashboard/ 2>/dev/null
python -m json.tool $B/delivery/dashboard/RIGHT_UP_inspection.json | grep -E '"version"|loco_number_results|damaged_wagons'
python -m json.tool $B/delivery/finalization.json 2>/dev/null
```

## Step 7 — Capture benchmarks

```bash
grep -E "STAGE|FEAT/|done in|Maximum resident|Percent of CPU" logs/validation_run.log
du -sh $B/wagon_cache $B/evidence $B/processed_videos $B/reports
```
Record into `docs/benchmarks/door.md` (replicate the template per feature): total
runtime, per-stage times, per-wagon avg, model-load time, peak RSS, peak CPU%,
#wagons, #detections per feature, disk used.

## Step 8 — Compare against production

On the **same trimmed clips**, run the production Train-Inspection-Engine and
diff per aligned wagon (align by rake position — v5 uses one GlobalTrainState).
Use `DOOR_PRODUCTION_COMPARISON_CHECKLIST.md` (replicate for OCR/Load/Damage):
door_state, wagon_number (+prefix), loco_number, load_status, top/side damage,
evidence selection, false pos/neg. Record agreement % + explain every mismatch.

## Step 9 — Service install (production)

```bash
sudo cp deploy/wagon-eye-extraction.service deploy/wagon-eye.service /etc/systemd/system/
# point EnvironmentFile at deploy/wagon-eye.env in each unit
sudo systemctl daemon-reload
sudo systemctl enable --now wagon-eye-extraction wagon-eye
systemctl status wagon-eye --no-pager
journalctl -u wagon-eye -f
```

## Troubleshooting (fast index)

| Symptom | Action |
|---|---|
| Stage-1 `master camera RIGHT_UP video is not present` | you passed a **relative** `--local-inputs`; omit it (absolute default) or pass `"$(pwd)/local_inputs"`. |
| A feature all `NO_DATA` with model present | check `YOLO(<model>.pt).names` matches expected classes (door_open/door_close/damage; body_dmg/floor_dmg…; wagon_loaded/empty; locono). |
| `failed_no_global_state` | reconstruction models missing or <4 cameras; see `$B/global_state/stage1_wagon_count.log`. |
| Loco numbers empty | `models/reconstruction/right_up_gap.pt` missing or no `locono` class; check ENGINE wagons exist in GlobalTrainState. |
| Dashboard on wrong tab | set `WAGONEYE_INSPECTION_VERSION=v2`. |
| Very slow | CPU box; use GPU (`WAGONEYE_DEVICE=cuda`) or accept the runtime and record it. |
| easyocr import error | `pip install easyocr` (OCR only). |
| Stale q90 cache | expected rebuild on next run (materializer schema bumped to v2). |
