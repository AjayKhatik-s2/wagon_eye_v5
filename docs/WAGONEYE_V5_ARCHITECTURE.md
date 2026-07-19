# WagonEye v5 — Production Architecture (raw S3 → dashboard)

Complete production pipeline: two independent EC2 services connected only through
S3. **Extraction** turns continuous CCTV into trimmed train clips; **inspection**
(the GlobalTrain pipeline) turns trimmed clips into reports + dashboard + email.

```
                        ┌──────────────────────────────────────────────┐
                        │  4 fixed CCTV cameras @ Hazaribagh (ECR-DHN)   │
                        │  RIGHT_UP(master) LEFT_UP RIGHT_UP_TOP LEFT_UP_TOP │
                        └───────────────────────┬──────────────────────┘
                                                │ (external uploader)
                                                ▼
        s3://biro-wagon-raw-video-copy/<camera_folder>/YYYY-MM-DD/*_YYYYMMDD_HHMMSS.mp4
                                                │
   ══════════════ SERVICE 1: train_extraction.run_extraction_service ═════════════
        poll raw bucket → classify frames (side_/top_classification.pt) →
        3-phase segment finder → stitch cross-clip trains → ffmpeg trim →
        upload "<raw>_train.mp4"      (ledger: logs/extraction_state/)
                                                │
                                                ▼
        s3://biro-wagon-pre-processed-video-copy/<camera_folder>/*_train.mp4
                                                │  (WAGONEYE_S3_INPUT_BUCKET/PREFIXES)
   ═══════════ SERVICE 2: orchestrator.master_runner --auto (inspection) ═════════
                                                │
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 0  batch acquisition (train_batch_manager + batch_manifest +          │
   │          lifecycle_runner): cluster 4 cameras by filename timestamp;         │
   │          async-camera lifecycle (wait master → support window → SEAL).       │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ STAGE 1  reconstruction.runner → wagon_count subprocess                      │
   │   models/reconstruction/{right_up_gap,left_up_gap,top_gap,side_classification}.pt│
   │   → global_state/global_train_state.json  (GlobalTrainState: GW_n, class,    │
   │      time windows)  + per_camera_tracking.json  [IMMUTABLE once sealed]      │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ STAGE 2  materializer.wagon_cache_builder (single decode, q95)              │
   │   → wagon_cache/GW_n/<camera>/frame_*.jpg   [LAST video decode for analysis] │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ STAGE 3  feature processors (PRODUCTION models, per camera authority)        │
   │   ┌ Door   side_damage.pt (RIGHT_UP→right, LEFT_UP→left)                     │
   │   ├ OCR    wagon_number.pt (11-digit, prefix-manip) + right_up_gap.pt        │
   │   │        `locono` → 5-digit LOCO OCR  (RIGHT_UP)                           │
   │   ├ Load   top_classification.pt / ltop.pt (wagon_loaded/empty vote, TOP)   │
   │   └ Damage top_left/right_top_damage.pt (4-class, TOP) + side_damage.pt      │
   │            `damage` (SIDE)                                                    │
   │   → wagon_states/<feature>/<CAMERA>/GW_n.json  + evidence/GW_n/<feat>/<CAM>/ │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ STAGE 4  fusion.wagon_state_builder (authority rules)                        │
   │   classification←GST · wagon_identifier + loco_number←RIGHT_UP OCR ·         │
   │   right_door←RIGHT_UP · left_door←LEFT_UP · load←RIGHT_UP_TOP(else LEFT_UP_TOP)│
   │   top_damage←any top · side_damage←either side                              │
   │   → wagon_states/unified/GW_n.json (UnifiedWagonState + anomalies)          │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ STAGE 4b rendering.feature_overlay_renderer (VISUALIZATION ONLY, no model)   │
   │   → processed_videos/<CAM>_processed.mp4                                     │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ STAGE 5  reporting.camera_reports (4 camera PDFs) +                          │
   │          reporting.combined_train_report (combined PDF + JSON schema v4;     │
   │          KPI incl. loco numbers; TOP_DMG/SIDE_DMG columns; evidence grid)    │
   │   → reports/{<cam>_report.pdf} + reports/combined_train_report.{pdf,json}   │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ STAGE 6  delivery (exactly-once via delivery/finalization.json)             │
   │   ├ dashboard_ingest → delivery/dashboard/<CAMERA>_inspection.json          │
   │   │     POST cctv-receiver/inspections/ingest  (version v2; loco_number_results)│
   │   ├ s3_upload → s3://biro-wagon-report-biro-copy/train_batch/$BK/reports/…  │
   │   ├ notification → ONE email (subject incl. loco numbers; prod recipients)  │
   │   └ archive batch tree + mark terminal in master_runner/processed_batches.json│
   └───────────────────────────────────┬────────────────────────────────────────┘
                                        ▼
                    WagonEye dashboard  +  ops inbox  (COMPLETE → next train)
```

## Invariants
- **GlobalTrainState is immutable** downstream of Stage 1 (count / GW_n / boundaries / class never change; late cameras only attach features).
- **Frames decoded exactly once for inference** (Stage 2). Stage 4b decodes only for visualization (no detector).
- **Production models + thresholds everywhere in Stage 3** (`models/production/` + `models/reconstruction/`); v4-native `models/features/` shelved.
- **Per-camera write isolation:** each camera writes only `wagon_states/<feature>/<CAMERA>/` and `evidence/<GW>/<feat>/<CAMERA>/`.
- **Exactly-once delivery:** `finalization.json` guards one upload + one email + idempotent dashboard ingest.

## Buckets
| Bucket | Role |
|---|---|
| `biro-wagon-raw-video-copy` | raw CCTV (extraction input) |
| `biro-wagon-pre-processed-video-copy` | trimmed clips (extraction output = inspection input) |
| `biro-wagon-report-biro-copy` | `train_batch/$BK/…` reports/JSON/archive + `master_runner/processed_batches.json` |
| `wagon-eye-models` | production + reconstruction `.pt` (staged to `models/*` on EC2) |

## Models
| Dir | Files | Stage |
|---|---|---|
| `models/reconstruction/` | right_up_gap, left_up_gap, top_gap, side_classification (`.pt`) | 1 (+ `right_up_gap` locono → loco OCR in 3) |
| `models/production/` | side_damage, top_left_damage, right_top_damage, wagon_number, ltop, top_classification (`.pt`) | 3 |
| `models/extraction/` | side_classification, top_classification (`.pt`) | A |

## Processes / deploy
- `deploy/wagon-eye-extraction.service` — Service 1 (producer).
- `deploy/wagon-eye.service` — Service 2 (`master_runner --auto`).
- Both reboot-safe (systemd); graceful SIGTERM finishes the current unit of work.
