# EC2 Stage-3 Door Validation Procedure

Complete procedure to run and validate the **Door** processor (production
`side_damage.pt` behaviour on the GlobalTrain wagon cache) on EC2, and to verify
it reproduces the production pipeline's door output.

Prerequisites: `docs/EC2_DEPLOYMENT_CHECKLIST.md` completed — venv active,
`models/reconstruction/*` + `models/production/side_damage.pt` staged, four
trimmed clips in `local_inputs/`.

> Milestone-1 door behaviour (what we are validating): per side camera
> (RIGHT_UP→right door, LEFT_UP→left door), `side_damage.pt` at conf **0.85
> (RIGHT_UP) / 0.88 (LEFT_UP)**, fixed **10-frame edge skip**, band `door_open`/
> `door_close` at **gap_tolerance 5**, `door_state = OPEN if any door_open band
> else CLOSED`, `door_close_detected = any door_close band`, evidence = winning
> band's best frame. Full spec: `WAGONEYE_V5_STAGE3_FEATURE_AUDIT.md` §2.A.

---

## 1. Verify model loading first

```bash
source .venv/bin/activate
python - <<'PY'
from core import production_models as PM
print("status:", PM.status())                      # side_damage.pt -> True
m = PM.load_for("door", "RIGHT_UP")                 # loads models/production/side_damage.pt
print("names:", m.names)                            # must include door_open, door_close
print("model_for door/LEFT_UP:", PM.model_for("door","LEFT_UP"))   # side_damage.pt
PY
```
PASS when: `side_damage.pt` is `True`, `load_for` returns a model, and `names`
contains `door_open` and `door_close`. If the file is absent, `load_for` raises
`MissingProductionModel` — stage the model (checklist §3).

---

## 2. Run ONLY the Door processor

### Option A — real entry point, Door-only (recommended)
Runs Stage 1 (reconstruction) → Stage 2 (materializer, q95) → Stage 3 **Door only**
→ fusion/reports, with OCR/Load/Damage disabled:
```bash
python -m orchestrator.master_runner \
    --local-only --local-inputs ./local_inputs \
    --disable-features ocr,load,damage \
    --skip-upload --skip-email --no-interactive \
    2>&1 | tee logs/door_validation_run.log
```
The batch key + output dir are printed; outputs land in
`batch_outputs/<batch_key>/`.

### Option B — isolated Door pass on an existing state+cache (fastest re-runs)
Once Stage 1+2 have produced `global_train_state.json` + `wagon_cache/` for a
batch (from Option A, or the standalone Stage-1/2 commands), re-run **only** the
door processor without redoing reconstruction/materialization:
```bash
BATCH=batch_outputs/<batch_key>
python - <<PY
import os
from core.global_state_loader import load_global_train_state
from core import constants as C
from features.door import processor as door
b = os.path.abspath("$BATCH")
state = load_global_train_state(os.path.join(b, "global_state", "global_train_state.json"))
summary = door.run(
    state=state,
    cache_root=os.path.join(b, "wagon_cache"),
    feature_models_dir=os.path.join("models", "features"),   # ignored; model from models/production
    output_dir=os.path.join(b, "wagon_states"),
    evidence_root=os.path.join(b, "evidence"),
    cameras=[C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP],
    verbose=True,
)
from collections import Counter
print("DOOR status dist:", dict(Counter(summary.values())))
PY
```
Use `WAGONEYE_LOG_LEVEL=DEBUG` (or `verbose=True` above) for per-wagon lines like
`[door/RIGHT_UP/GW_7] right=OPEN (0.91) open_bands=1 close_bands=0 frames=88`.

---

## 3. Inspect `wagon_states/door`

```bash
BATCH=batch_outputs/<batch_key>
# per-camera per-wagon files
ls $BATCH/wagon_states/door/RIGHT_UP/ | head
ls $BATCH/wagon_states/door/LEFT_UP/  | head
# status distribution across all wagons (both cameras)
python - <<'PY'
import glob, json, collections, sys
b = sys.argv[1] if len(sys.argv)>1 else "batch_outputs"
for cam in ("RIGHT_UP","LEFT_UP"):
    files = glob.glob(f"{b}/wagon_states/door/{cam}/GW_*.json")
    st = collections.Counter(json.load(open(f))["status"] for f in files)
    ds = collections.Counter(json.load(open(f)).get(f"{'right' if cam=='RIGHT_UP' else 'left'}_door") for f in files)
    print(cam, "n=", len(files), "status=", dict(st), "door_state=", dict(ds))
PY
```
Then open a single wagon and confirm the schema (see §7):
```bash
python -m json.tool $BATCH/wagon_states/door/RIGHT_UP/GW_1.json
```

---

## 4. Inspect generated evidence

```bash
BATCH=batch_outputs/<batch_key>
# evidence tree for one wagon
ls -R $BATCH/evidence/GW_1/door/
# expect (when a door detection existed):
#   RIGHT_UP/right_best.jpg  RIGHT_UP/right_crop.jpg  RIGHT_UP/metadata.json
#   LEFT_UP/left_best.jpg    LEFT_UP/left_crop.jpg    LEFT_UP/metadata.json
python -m json.tool $BATCH/evidence/GW_1/door/RIGHT_UP/metadata.json
# open the annotated best frame to eyeball the box (scp to a workstation or use an image viewer):
#   right_best.jpg  = full frame with door bbox drawn (door_open=red, door_close=green)
#   right_crop.jpg  = padded crop of the door region
```
CLOSED wagons with no door detection correctly have an empty `evidence: {}` and
no evidence files (production did not emit a problem frame for a clean door).

---

## 5. Inspect logs

```bash
# per-stage timing + door summary
grep -E "STAGE|FEAT/door|door/" logs/door_validation_run.log | tail -40
# door completion line: "[FEAT/door] done in <s>s  ok=<n>/<total>"
# per-wagon lines (DEBUG/verbose): "[door/<CAM>/<GW>] <side>=<STATE> (<conf>) open_bands=.. close_bands=.. frames=.."
# rotating file (all stages):
tail -100 logs/wagon_eye.log
```

---

## 6. Expected JSON schema (`wagon_states/door/<CAMERA>/GW_n.json`)

RIGHT_UP writes `right_door`; LEFT_UP writes `left_door`. Status `OK`:
```json
{
  "global_id": "GW_7",
  "feature": "door",
  "camera_id": "RIGHT_UP",
  "side": "right",
  "status": "OK",
  "door_state": "OPEN",
  "door_confidence": 0.9123,
  "right_door": "OPEN",
  "right_door_confidence": 0.9123,
  "door_close_detected": false,
  "tracks": [
    {"camera_id":"RIGHT_UP","track_id":1,"state":"OPEN","confidence":0.9123,
     "first_frame":812,"last_frame":868,"total_hits":41,"mean_center_x":712.4}
  ],
  "supporting_cameras": ["RIGHT_UP"],
  "frame_count": 88,
  "evidence": {
    "right_best": ".../evidence/GW_7/door/RIGHT_UP/right_best.jpg",
    "right_crop": ".../evidence/GW_7/door/RIGHT_UP/right_crop.jpg"
  }
}
```
- `door_state`/`{side}_door` ∈ `{OPEN, CLOSED}` only (production 2-state; never PARTIAL/DAMAGED).
- `NO_DATA` payload (model absent) carries the same keys with `door_state=NO_DATA`, empty `tracks`/`evidence`, and an `error` string.
- `FAILED` payload carries `error` + `traceback` (per-wagon isolation).

---

## 7. Expected evidence structure (`evidence/GW_n/door/<CAMERA>/`)

```
right_best.jpg     full frame, door bbox annotated (door_open=red / door_close=green), label "<CLASS> <conf>"
right_crop.jpg     padded crop of the door bbox
metadata.json      { global_id, feature:"door", camera_id, side,
                     sides: { right: { camera_id, frame_idx, bbox:[x1,y1,x2,y2],
                                       state, confidence, raw_class } } }
```
(LEFT_UP mirrors with `left_*`.) Metadata `frame_idx` is the absolute cache
frame index; the renderer draws the box on that frame ±window.

---

## 8. Runtime benchmarks to capture

Record these in `docs/benchmarks/door.md` (template provided) for both a CPU and
(if available) a GPU run:
- model load time (first `load_for("door", cam)`),
- total door-stage wall time (`[FEAT/door] done in …`),
- average time per wagon (per camera),
- evidence generation time (subset of the above),
- #wagons, #wagons with a door detection, #door_open, #door_close,
- peak RSS + CPU% (`/usr/bin/time -v`, `top`, or `psrecord`),
- device used (`WAGONEYE_DEVICE`).

Suggested capture:
```bash
/usr/bin/time -v python -m orchestrator.master_runner --local-only \
    --local-inputs ./local_inputs --disable-features ocr,load,damage \
    --skip-upload --skip-email --no-interactive 2>&1 | tee logs/door_bench.log
grep -E "STAGE|FEAT/door|Maximum resident|Percent of CPU" logs/door_bench.log
```

---

## 9. Pass / fail criteria

**Infrastructure PASS (all required):**
- [ ] `PM.status()['side_damage.pt']` is `True`; `load_for("door", cam)` returns a model whose `names` include `door_open`,`door_close`.
- [ ] Door stage completes without an unhandled exception; markers written under `wagon_states/.features/{RIGHT_UP,LEFT_UP}/door.json` with `model_filename: side_damage.pt`.
- [ ] Every wagon has a `door/RIGHT_UP/GW_n.json` and `door/LEFT_UP/GW_n.json`; schema matches §6; no `FAILED` (or each `FAILED` explained).
- [ ] Materializer wrote q95 frames (`MATERIALIZER_SCHEMA_VERSION=2` forced rebuild).

**Behavioural PASS (vs production — see `DOOR_PRODUCTION_COMPARISON_CHECKLIST.md`):**
- [ ] On identical frame sets, v5 `door_state` matches production `door_status` for **≥ the agreed threshold** of comparable wagons (target: exact on aligned wagons, since model + thresholds + banding are identical).
- [ ] `door_close_detected` matches production per aligned wagon.
- [ ] No systematic false-open / false-closed bias.
- [ ] Evidence frame selection is reasonable (annotated best frame actually shows the reported door state).

**FAIL triggers:** all-`CLOSED` output with `side_damage.pt` present (class-name
mismatch); door stage exceptions; systematic disagreement with production not
explained by wagon-boundary/count differences; evidence missing for OPEN wagons.
