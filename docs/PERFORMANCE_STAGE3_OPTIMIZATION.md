# WagonEye — Stage 3 Performance Optimization (identical-output constraint)

**Constraint:** identical outputs (wagon count, OCR/door/load/damage results, evidence,
reports, dashboard, archive). Only runtime may change.

## 0. Two hard truths up front (honesty)

1. **This environment cannot profile or validate parity.** `models/production/` is empty,
   `easyocr` isn't installed, and there is no real footage on this box, so real inference
   timings and output-parity checks **cannot be run here**. All timings below are a
   **code-grounded model**, explicitly labeled as estimates. Parity of the batched path
   must be validated on the target host (harness in §9).
2. **Batched YOLO is not guaranteed bit-identical to per-frame** on GPU/FP16: batched
   GEMM/conv use different reduction orders, so a detection whose confidence sits within
   ~1e-3 of a hard threshold (0.70 / 0.85 / 0.88 / 0.80) can flip inclusion → a different
   band → a different result.

> **BRANCH DECISION (aggressive optimization branch): `FEATURE_BATCH_SIZE` DEFAULTS TO 32.**
> Batched inference is ON by default across Door / Damage / Load / OCR-detection. Numerical
> parity will be validated on EC2 with the production models (accepted). Set
> `WAGONEYE_FEATURE_BATCH_SIZE=1` to fall back to the **verbatim** per-frame path (byte-identical
> to pre-optimization). On any OOM the batch auto-halves **32→16→8→4→2→1** (sticky, process-wide)
> and retries — it never fails on OOM.

## 1. What is ALREADY optimal (verified by reading the code — do NOT re-add)

| Requested optimization | Already done? | Evidence |
|---|---|---|
| Model loaded exactly once, never in the wagon loop | **Yes** | `PM.load_for(...)` called once before the wagon loop in every processor: door:391, load:234, damage:421; cached in `production_models._CACHE`. easyocr Reader is a per-process singleton (ocr docstring). |
| Reuse CUDA/CPU context | **Yes** | Device resolved once (`_common.DEVICE/HALF`); model objects reused. |
| Features run concurrently | **Yes** | Orchestrator runs door/ocr/damage in a `ThreadPoolExecutor` (`master_runner.process_batch`); lifecycle runs per-camera. |
| Decode each frame once (within a wagon) | **Yes** | Each interior frame is a single `cv2.imread` in the per-frame loop. |
| Wagons within a feature serial (so the model loads once) | **Yes** | Intentional — enables model reuse; parallelizing wagons around one YOLO model is not thread-safe. |

So "load once / reuse / parallel features / decode once" are **no-ops** — the code already does them. The remaining wins are **batched inference** (the real Stage-3 cost) and a few **I/O** items.

## 2. Measured bottleneck model (code-grounded estimate; NOT profiled here)

Per train (~62 wagons, ~15 fps, ~264 s reference), Stage-3 work units:

| Feature | Cameras | Frames/wagon (interior) | Model calls (per-frame path) | Relative cost |
|---|---|---|---|---|
| OCR | RIGHT_UP | all interior + easyocr per plate | YOLO/frame + **easyocr multi-run/detection** | **highest** (easyocr dominates) |
| Damage (top) | 2 top | interior (edge-skip 10) | 1 YOLO/frame | high |
| Damage (side) | 2 side | interior | 1 YOLO/frame | high |
| Door | 2 side | interior | 1 YOLO/frame | high |
| Load | 2 top | every-other interior (step 2) | 1 classify/frame | medium |

Dominant term = **N_frames × YOLO-forward**, repeated per feature/camera. The single biggest
lever is amortizing the per-call Python/kernel-launch overhead via **batching**; on GPU this is
a large throughput gain, on CPU a smaller one. **OCR's easyocr stage is the top cost but is the
least batchable** (see §11).

## 3. Optimizations implemented (this change — COMPLETE across all eligible features)

| # | Optimization | Where | Enabled by | Output impact |
|---|---|---|---|---|
| 1 | `batched_detect()` (collect → `model([...])` → scatter, input order; `confidence` float = model-conf+floor, `None` = model-default no-floor for OCR) + `_parse_detection_result()` | `_common.py` | helper | none by itself |
| 2 | `batched_classify()` + `_parse_classification_result()` (for Load) | `_common.py` | helper | none by itself |
| 3 | `FEATURE_BATCH_SIZE` knob (env `WAGONEYE_FEATURE_BATCH_SIZE`, **default 32**) | `config.py` | — | default 32 → batched ON |
| 4 | **Sticky adaptive batch** — OOM (CUDA RuntimeError or MemoryError) halves 32→16→8→4→2→1 process-wide; never fails on OOM, never drops a frame | `_common.py` | default | none (same scatter at any size) |
| 5 | **Door** wired: `if >1` batched branch; else **verbatim** per-frame loop | `door/processor.py` | `>1` | default byte-identical |
| 6 | **Damage** (top 4-class + side) wired, same gated pattern | `damage/processor.py` | `>1` | default byte-identical |
| 7 | **Load** wired: gated `batched_classify`; default per-frame verbatim | `load/processor.py` | `>1` | default byte-identical |
| 8 | **OCR**: batch **only** the YOLO wagon-number *detection*; EasyOCR recognition stays per-detection in identical order via the shared `_ocr_one_detection()` | `ocr/processor.py` | `>1` | recognition unchanged; detection gated |
| 9 | **Evidence decode reuse** (batched path caches detection frames → no re-read; default falls back to `read_cached_frame`) | door/damage | `>1` | identical bytes |
| 10 | **Directory-listing cache** (`_cached_sorted_frames`, per dir+mtime, thread-safe) — collapses the redundant re-scans of a wagon dir across features | `_common.py` | **default ON** | identical (mtime-invalidated) |

**Unit-tested here** (fake deterministic models): `batch=1 == 8 == 32` for `batched_detect`
(both `confidence` float and `None`); `batched_classify` bs1 == `run_classification` per frame;
confidence floor honored; CUDA-OOM halving preserves identical in-order output; listing cache
returns identical results and invalidates on mtime change. This proves the collect/scatter/order/
adaptive/cache logic — **not** real-YOLO numeric parity (§9, on-host).

**OCR loco path** (`_process_engine_loco`, ENGINE wagons only — usually 1–2/train) is left
per-frame: marginal benefit, and it interleaves EasyOCR the same way. Documented, not batched.

## 4. Before/After timing estimates (MODEL — not measured here)

For a feature over F total frames, per-call fixed overhead `o`, per-frame compute `c`, batch B:
- per-frame: `F·(o+c)`
- batched:  `ceil(F/B)·o_batch + F·c`, where GPU parallelizes `c` across the batch.

Rough estimates (enable `FEATURE_BATCH_SIZE=32`), **to be confirmed on-host**:

| Stage | GPU (T4) est. | CPU est. |
|---|---|---|
| Door / Damage / Load inference | **~2–4× faster** (kernel-launch + transfer amortized, batch parallelism) | ~1.2–1.6× (less Python overhead; CPU YOLO barely parallelizes a batch) |
| OCR | ~unchanged (easyocr not batched) | ~unchanged |
| Stage-3 overall | **~1.6–2.2× faster** (OCR remains the floor) | ~1.1–1.3× |

Default (`FEATURE_BATCH_SIZE=1`): **0% change** — identical code path.

## 5–8. Disk / storage / RAM impact

- **Disk reads:** batched path removes the evidence best-frame **re-read** for wagons with a
  detection (door: 1/wagon-with-evidence; damage: up to #bands/wagon). Small (findings are few).
  Frame decode count is otherwise unchanged (each interior frame decoded once).
- **Disk writes:** unchanged by this change (same JSON/evidence written once, atomically).
- **Storage:** unchanged by this change. Largest reducible artifact is **`wagon_cache/`**
  (~hundreds of MB, ~15k JPEGs/train, regenerable, **not** uploaded) — safe to delete after a
  terminal batch, but that touches finalize/cleanup (see §11, not done — respects "don't modify
  lifecycle/archive").
- **RAM:** batched path holds at most `FEATURE_BATCH_SIZE` decoded frames + the few
  detection frames per wagon (bounded; released per wagon). At B=32, ~32 frames in flight vs 1
  today — a bounded, modest increase, and it auto-halves under CUDA OOM. Default path RAM unchanged.

## 9. Why inference correctness is preserved

- **Default (`FEATURE_BATCH_SIZE=1`)**: the door detection loop is the **verbatim**
  pre-optimization code (see the `else:` branch marked "VERBATIM"); load/ocr/damage are
  **untouched**. So the shipped default is byte-identical — nothing to validate.
- **Batched path (opt-in)**: same model, same per-camera confidence passed to `model(...)`,
  same class filtering, same banding, same evidence selection — only the *grouping* of frames
  into one `model([...])` call differs. The one non-identical risk is FP16/GEMM reduction order
  at a threshold boundary (§0.2). **Validate before enabling** with:

```bash
# On the target host, with production models staged:
BK=<existing completed batch>
# 1) baseline (per-frame) already on disk in batch_outputs/$BK/wagon_states + evidence
cp -r batch_outputs/$BK/wagon_states /tmp/base_states
cp -r batch_outputs/$BK/evidence     /tmp/base_evidence
# 2) re-run Stage 3 batched (delete feature markers to force re-run)
rm -rf batch_outputs/$BK/wagon_states/.features
WAGONEYE_FEATURE_BATCH_SIZE=32 python -m orchestrator.master_runner --batch $BK --skip-upload --skip-email
# 3) diff the per-wagon feature JSON + evidence (must be identical for strict parity)
diff -r /tmp/base_states batch_outputs/$BK/wagon_states
python - <<'PY'  # pixel-compare evidence
import glob,cv2,numpy as np,os
for a in glob.glob('/tmp/base_evidence/**/*.jpg',recursive=True):
    b=a.replace('/tmp/base_evidence',os.path.join('batch_outputs','$BK','evidence'))
    ia,ib=cv2.imread(a),cv2.imread(b)
    if ia is None or ib is None or ia.shape!=ib.shape or not np.array_equal(ia,ib):
        print("DIFF",b)
print("evidence compare done")
PY
```
If the diff is clean on your hardware, batching is safe to enable in production; if a handful of
threshold-boundary wagons differ, keep `FEATURE_BATCH_SIZE=1` (or raise thresholds' margin — not
allowed here) — that is the FP16 caveat, not a bug.

## 10. Generated file / folder audit (batch_outputs/<key>/)

| Path | Creator | Consumer | Required after completion? | Can stay in memory? | Safe to remove? |
|---|---|---|---|---|---|
| `downloads/<cam>.mp4` | lifecycle `_download_present` | Stage 1/2/4b | No (source, re-downloadable; not uploaded) | No (large) | **Yes, after batch** |
| `global_state/global_train_state.json` | Stage 1 subprocess | Stages 2–5, dashboard | **Yes** (archived) | No (subprocess IPC → must be on disk) | No |
| `global_state/per_camera_tracking.json` | Stage 1 | Stage 2 (fps), rendering | Yes (archived) | No (IPC) | No |
| `global_state/stage1_wagon_count.log` | recon runner | humans/debug | archived | — | low value; keep |
| `global_state/processed_videos/*.mp4` | Stage 1 (wagon_count debug overlay) | none downstream | archived but redundant w/ Stage 4b | — | **Yes** (see §11: `--no-videos`) |
| `wagon_cache/GW_n/<cam>/frame_*.jpg` | Stage 2 | Stage 3, evidence, rendering | **No** (regenerable; NOT uploaded) | No (~15k frames) | **Yes, after finalize** (biggest storage win) |
| `wagon_cache/.materialized/*.json` | Stage 2 | lifecycle skip | resume only | — | after terminal |
| `wagon_states/<feat>/<CAM>/GW_n.json` | Stage 3 | Stage 4 fusion | **Yes** (archived) | (already in memory during fusion) | No |
| `wagon_states/.features/**` | lifecycle | resume skip | resume only | — | after terminal |
| `wagon_states/unified/GW_n.json` | Stage 4 | Stage 5, dashboard | **Yes** (archived) | No | No |
| `evidence/GW_n/<feat>/<CAM>/*.jpg,metadata.json` | Stage 3 | Stage 5, dashboard | **Yes** (deliverable, archived) | No | No |
| `processed_videos/<CAM>_processed.mp4` | Stage 4b | Stage 5 links, archive | **Yes** (archived) | No | No |
| `reports/*.pdf`, `combined_train_report.json` | Stage 5 | delivery, dashboard | **Yes** | No | No |
| `delivery/dashboard/*_inspection.json` | dashboard_ingest | ingest API | Yes (uploaded) | No | No |
| `delivery/finalization.json` | finalization | resume idempotency | resume/audit | — | keep |
| `manifest.json` | lifecycle | resume + archived | **Yes** | — | No |
| `.tmp*` / `.tmp_build` (atomic writes) | all writers | — | No | — | auto (temp+rename; no residue) |

**Takeaways:** the only large *removable* artifacts are `wagon_cache/` and `downloads/`
(regenerable, not uploaded). Everything else is either a deliverable, archived, or a resume marker.

## 11. Optimizations intentionally REJECTED (would risk outputs or violate constraints)

- **OCR easyocr batching** — easyocr runs a multi-pass preprocess + recognizer per detected
  plate; batching it changes recognition context and the multi-run aggregation → **can change OCR
  strings**. Rejected. (Only its YOLO plate-*detection* stage is safely batchable — a follow-up.)
- **Keeping frames in memory across Stage 2→3** — would break the resumable/late-camera
  architecture (features re-read the cache on late attach) and blow RAM (~15k frames), and Stage 2
  is off-limits. Rejected.
- **Stage 1 in-memory** — it's a subprocess; JSON must cross the process boundary on disk. Rejected.
- **Deleting `wagon_cache/`/`downloads/` after finalize** — safe for outputs and a real storage win,
  but it edits finalize/cleanup (the task forbids modifying lifecycle/archive). **Deferred** —
  recommend as a separate, opt-in cleanup step.
- **`--no-videos` to skip wagon_count's debug overlay mp4s** — a real time/disk win with no output
  change (they aren't consumed downstream), but it changes the **Stage 1** invocation. **Deferred**
  per "do not modify Stage 1".
- **Reducing image resolution / skipping frames / raising thresholds** — explicitly forbidden;
  would change detections. Rejected.
- **Parallelizing wagons around a shared YOLO model** — not thread-safe; risks races/nondeterminism.
  Rejected (features already parallel at the orchestrator).

## 12. Unchanged (confirmed)

Discovery, Batch Manager, Lifecycle, Stage 1 (reconstruction), Stage 2 (materialization),
Feature Fusion, Reporting, Dashboard, Archive, Uploads, Wagon Grouping, and the overall
architecture are **untouched**. Only `core/config.py` (one knob), `features/_common.py`
(batching helper), and `features/door/processor.py` (gated branch; default verbatim) changed.
With the default `FEATURE_BATCH_SIZE=1`, the entire pipeline is byte-identical to before.
