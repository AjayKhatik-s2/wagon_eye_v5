# Wagon Eye — Phase 1 (standalone wagon counter)

Self-contained 4-camera global wagon counting + classification.
Designed to zip and run on a plain Linux server (EC2) with no other code.

> **Phase 1 only.** No doors, no damage, no OCR, no PDF, no email,
> no S3 upload. Just global wagon segmentation + classification +
> processed videos + per-wagon frame folders.

---

## Folder layout

```
wagon_count/
├── run_global_count.py          # entry point
├── global_train_state.py        # data classes
├── tracker_engine.py            # per-camera gap tracking + master classifier
├── global_alignment.py          # cross-camera fusion
├── video_segmenter.py           # overlay + frame extraction
├── requirements.txt
├── inputs/                      # drop your 4 trimmed train videos here
│   ├── right_up.mp4
│   ├── left_up.mp4
│   ├── right_up_top.mp4
│   └── left_up_top.mp4
├── models/                      # drop the 4 YOLO weights here
│   ├── right_up_wagon_gap.pt    # RIGHT_UP (master) gap model
│   ├── left_up_wagon_gap.pt     # LEFT_UP gap model
│   ├── top_gap.pt               # top cameras (RIGHT_UP_TOP, LEFT_UP_TOP)
│   └── side_classification.pt   # RIGHT_UP only: ENGINE / WAGON / BRAKE_VAN
└── results/                     # created on first run
```

---

## Quick start (EC2 / any Linux server)

```bash
# 1) Unzip
unzip wagon_count.zip
cd wagon_count

# 2) Install Python dependencies (into a venv on EC2)
pip install -r requirements.txt

# 3) Drop the 4 videos in ./inputs/ and the 3 .pt models in ./models/

# 4) Run with defaults -- all 4 inputs and 3 models auto-discovered
python run_global_count.py
```

That's it. The pipeline writes everything to `./results/`.

---

## Inputs

The 4 videos must be **synchronized** — i.e., trimmed by an upstream
service to the same train pass so they share a `t=0` alignment.
RIGHT_UP is the **master camera**; its gap detections and
classifications are authoritative.

Default filenames (auto-discovered from `inputs/`):

| Camera           | Filename                                   | Model                                            |
|------------------|--------------------------------------------|--------------------------------------------------|
| RIGHT_UP (master)| `right_up.mp4`                             | `right_up_wagon_gap.pt` + `side_classification.pt` |
| LEFT_UP          | `left_up.mp4`                              | `left_up_wagon_gap.pt`                            |
| RIGHT_UP_TOP     | `right_up_top.mp4`                         | `top_gap.pt`                                      |
| LEFT_UP_TOP      | `left_up_top.mp4`                          | `top_gap.pt`                                      |

You can also override any path explicitly:

```bash
python run_global_count.py \
  --right_up     /data/cam_right_up_20260408.mp4 \
  --left_up      /data/cam_left_up_20260408.mp4 \
  --right_up_top /data/cam_right_up_top_20260408.mp4 \
  --left_up_top  /data/cam_left_up_top_20260408.mp4 \
  --models-dir   /opt/models \
  --output       /opt/results
```

---

## Outputs

After a run:

```
results/
├── global_train_state.json          ← canonical Phase-1 output
├── per_camera_tracking.json         ← per-camera gap timelines for debug
├── processed_videos/
│   ├── RIGHT_UP_processed.mp4       ← overlay: GW IDs, gaps, classification
│   ├── LEFT_UP_processed.mp4
│   ├── RIGHT_UP_TOP_processed.mp4
│   └── LEFT_UP_TOP_processed.mp4
└── frames/
    ├── RIGHT_UP/
    │   ├── GW_1/frame_000000.jpg, frame_000001.jpg, ...
    │   ├── GW_2/...
    │   └── ...
    ├── LEFT_UP/      (same GW_n layout)
    ├── RIGHT_UP_TOP/ (same GW_n layout)
    └── LEFT_UP_TOP/  (same GW_n layout)
```

All four cameras use the **same** `GW_n` ids — they refer to the same
physical wagon. This is the contract Phase-2 (door / damage / OCR /
loaded-empty) will consume.

`global_train_state.json` shape:

```json
{
  "schema": "wagon_eye.global_train_state.v1",
  "master_camera": "RIGHT_UP",
  "master_fps": 25.0,
  "master_total_frames": 7321,
  "total_wagons": 47,
  "regular_wagon_count": 45,
  "engine_count": 1,
  "brake_van_count": 1,
  "wagons": [
    {
      "global_id": "GW_1",
      "wagon_index": 1,
      "start_frame_master": 0,
      "end_frame_master": 312,
      "start_time": 0.0,
      "end_time": 12.52,
      "classification": "ENGINE",
      "classification_confidence": 0.94,
      "supporting_cameras": ["RIGHT_UP","LEFT_UP","RIGHT_UP_TOP","LEFT_UP_TOP"],
      "split_from_global_id": null,
      "leading_gap":  {"source": "video_start"},
      "trailing_gap": {"source": "master", "camera_id": "RIGHT_UP", "track_id": 1, "center_time": 12.51}
    },
    ...
  ],
  "per_camera_local_counts": { "RIGHT_UP": 46, "LEFT_UP": 45, "RIGHT_UP_TOP": 47, "LEFT_UP_TOP": 47 },
  "per_camera_gap_counts":   { "RIGHT_UP": 45, "LEFT_UP": 44, "RIGHT_UP_TOP": 46, "LEFT_UP_TOP": 46 },
  "corrections_applied":     [ {"inserted_at_master_time": 134.4, "supporting_cameras": ["LEFT_UP_TOP","RIGHT_UP_TOP"], ...} ],
  "fallback_used": false
}
```

---

## How the fusion works (one paragraph)

1. **Per-camera gap tracking** — each camera runs its own YOLO gap model.
   RIGHT_UP uses `right_up_wagon_gap.pt`; LEFT_UP uses
   `left_up_wagon_gap.pt`; both top cameras share `top_gap.pt`. A
   constant-velocity Kalman filter on the gap bounding-box center plus a
   hit/miss persistence rule emits one `GapEvent` per stable track.

2. **Master classification** — RIGHT_UP's pre-fusion segments (the spans
   between consecutive RIGHT_UP gaps) are labeled ENGINE / WAGON / BRAKE_VAN
   by `side_classification.pt` via majority vote on sampled frames.

3. **Cross-camera fusion** — each support camera's gaps are matched to
   the master gap timeline by **temporal IoU**. Unmatched support gaps
   are clustered across cameras. A cluster becomes an **inserted gap**
   only when it has **≥2 supporting cameras**, time spread ≤ 1.5 s,
   and mean confidence ≥ 0.4 — and is ≥ 1.0 s from any existing
   master gap. The fused master gap list is the original RIGHT_UP gaps
   plus accepted inserts.

4. **Global wagon rebuild** — segments between consecutive fused gaps
   become `GW_1 .. GW_N`. If an inserted gap splits a RIGHT_UP segment,
   children inherit the parent's classification (so ENGINE and BRAKE_VAN
   stay stable; a merged WAGON splits into two WAGONs).

5. **Fallback** — if support fusion throws or produces zero wagons,
   the system falls back to pure RIGHT_UP wagon counting and sets
   `fallback_used: true` in the JSON.

---

## Tuning knobs

All optional; defaults are usually fine.

| Flag                         | Default | Meaning                                      |
|------------------------------|---------|----------------------------------------------|
| `--side-confidence`          | 0.4     | YOLO conf threshold for side gap models (`right_up_wagon_gap.pt`, `left_up_wagon_gap.pt`) |
| `--top-confidence`           | 0.4     | YOLO conf threshold for `top_gap.pt`         |
| `--classification-samples`   | 5       | Frames per segment voted in classification   |
| `--fuse-min-support`         | 2       | Min cameras needed to insert a missed gap    |
| `--fuse-max-spread`          | 1.5     | Max time spread (s) inside a fusion cluster  |
| `--fuse-min-conf`            | 0.4     | Min mean confidence to insert a fused gap    |
| `--every-nth-frame`          | 1       | Keep 1 of every N frames when extracting     |
| `--no-videos`                | off     | Skip overlay video rendering                 |
| `--no-frames`                | off     | Skip per-wagon frame extraction              |
| `--no-raw-detections`        | off     | Save RAM by not storing per-frame bboxes     |
| `--quiet`                    | off     | Reduce log verbosity                         |

---

## Packaging for AWS

From the directory **above** `wagon_count/`:

```powershell
# Windows PowerShell
Compress-Archive -Path wagon_count -DestinationPath wagon_count.zip -Force
```

```bash
# Linux/macOS
zip -r wagon_count.zip wagon_count -x 'wagon_count/results/*' \
                                   -x 'wagon_count/inputs/*' \
                                   -x 'wagon_count/models/*'
```

The `-x` excludes keep the zip small — your AWS instance fetches videos
and models separately (S3 download, EBS volume, etc.).

---

## Phase-2 hook

The Phase-2 pipeline (door state, damage, OCR, loaded/empty, report)
will consume `results/frames/<CAMERA>/<GW_n>/` directly. Same GW ids
across cameras mean each downstream feature extractor can correlate
findings without re-running synchronization.
