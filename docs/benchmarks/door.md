# Door Processor — Benchmark Template

Capture one filled copy of this table per validation run (CPU and, if available,
GPU). Numbers feed EC2 sizing + regression tracking. Source of timings:
`logs/door_validation_run.log` (`STAGE` lines, `[FEAT/door] done in …`) and
`/usr/bin/time -v`.

> Reference (dev workstation, Windows, CPU, Python 3.14, ultralytics 8.4):
> Stage-1 reconstruction alone took **~2842 s (~47 min)** for a 62-wagon, ~264 s,
> 15 fps, 4-camera train; Stage-2 materialization **~22 s** (15,533 q95 JPEGs).
> Door timing was not measured on dev (no `side_damage.pt`). Fill real numbers on EC2.

---

## Run metadata

| Field | Value |
|---|---|
| Date (UTC) | |
| Build SHA (`git rev-parse HEAD`) | |
| Instance type | |
| vCPUs / RAM | |
| GPU (name / VRAM) | |
| `WAGONEYE_DEVICE` | cuda / cpu |
| Python / ultralytics / torch | |
| Batch key | |
| #wagons (GlobalTrainState `total_wagons`) | |
| Cameras run | RIGHT_UP, LEFT_UP |
| Model | models/production/side_damage.pt |

---

## Timing

| Metric | RIGHT_UP | LEFT_UP | Total | Notes |
|---|---|---|---|---|
| Model load time (first `load_for`) | | — (cached) | | side_damage.pt shared → loaded once |
| Door stage wall time (`[FEAT/door] done in`) | | | | |
| Avg runtime per wagon | | | | stage_time / #wagons |
| Median / p95 per wagon | | | | from verbose per-wagon lines |
| Evidence generation time | | | | subset of per-wagon (annotate + crop + write) |
| Frames processed (interior, all wagons) | | | | sum of `frame_count` |
| Avg YOLO inference / frame | | | | stage_time / frames (approx) |

Per-stage context (from `STAGE` log lines, for the same run):

| Stage | Wall time |
|---|---|
| Stage 1 reconstruction | |
| Stage 2 materialization (q95) | |
| Stage 3 Door | |

---

## Resource usage

| Metric | Value | Source |
|---|---|---|
| Peak RSS (memory) | | `/usr/bin/time -v` → "Maximum resident set size" |
| Peak CPU % | | `top` / `psrecord` / time -v "Percent of CPU" |
| GPU util % / VRAM (if GPU) | | `nvidia-smi dmon` during the run |
| Disk written (wagon_cache + evidence) | | `du -sh batch_outputs/<key>/{wagon_cache,evidence}` |

---

## Detection counts

| Metric | RIGHT_UP | LEFT_UP | Total |
|---|---|---|---|
| Wagons with a door detection | | | |
| door_open bands (sum) | | | |
| door_close bands (sum) | | | |
| Wagons door_state=OPEN | | | |
| Wagons door_state=CLOSED | | | |
| Wagons door_close_detected=True | | | |
| Wagons NO_DATA / FAILED | | | |
| Evidence images written | | | |

---

## Capture commands

```bash
/usr/bin/time -v python -m orchestrator.master_runner --local-only \
    --local-inputs ./local_inputs --disable-features ocr,load,damage \
    --skip-upload --skip-email --no-interactive 2>&1 | tee logs/door_bench.log

# timings + memory
grep -E "STAGE|FEAT/door|Maximum resident|Percent of CPU" logs/door_bench.log

# detection counts (fill the tables)
python - <<'PY'
import glob, json, collections
b = "batch_outputs"  # or the specific batch dir
for cam in ("RIGHT_UP","LEFT_UP"):
    files = glob.glob(f"{b}/*/wagon_states/door/{cam}/GW_*.json") or \
            glob.glob(f"{b}/wagon_states/door/{cam}/GW_*.json")
    side = "right" if cam=="RIGHT_UP" else "left"
    st  = collections.Counter(); ds = collections.Counter()
    nclose = ntracks = nframes = 0
    for f in files:
        d = json.load(open(f)); st[d["status"]] += 1; ds[d.get(f"{side}_door")] += 1
        nclose += int(bool(d.get("door_close_detected")))
        ntracks += len(d.get("tracks",[])); nframes += int(d.get("frame_count",0))
    print(cam, "n=",len(files), "status=",dict(st), "door_state=",dict(ds),
          "close_detected=",nclose, "tracks=",ntracks, "frames=",nframes)
PY

# disk
du -sh batch_outputs/*/wagon_cache batch_outputs/*/evidence 2>/dev/null
```

---

## Observations / regressions

- (free text: bottleneck stage, CPU vs GPU delta, anomalies, action items)
