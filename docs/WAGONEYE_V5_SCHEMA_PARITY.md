# WagonEye v5 — inspection_data / dashboard payload parity vs production

Compares the v5 **dashboard payload** (`delivery/dashboard_ingest.py`, the
per-camera `{camera_id, version, inspection_data}` document POSTed to the
production `cctv-receiver/inspections/ingest` API) against the **production**
Train-Inspection-Engine `inspection_data.json` **v2** schema
(`reporting/json_builder.build_inspection_json`). Every difference is listed with
an **ELIMINATE** (fixed / how to fix) or **JUSTIFY** (safe, why) disposition.

> The v5 `reports/combined_train_report.json` (`schema
> wagon_eye.combined_report.v4`) is a *different, additional* artifact (one
> combined train doc, not the per-camera feed). It is a superset carrying the
> full fused state + `legacy_view_model`; it is **not** what the production
> dashboard consumes, so it is out of scope for per-field parity here (it never
> replaces the per-camera feed).

---

## 1. Envelope

| Field | Production v2 | v5 dashboard | Disposition |
|---|---|---|---|
| `camera_id` | `CCTV_HZBN_DHN_n_<ANGLE>` (prefix-stripped) | `full_camera_id(camera)` (same ids) | **OK** — matches. |
| `version` | `"v2"` | `"v1"` (default; env `WAGONEYE_INSPECTION_VERSION`) | **ELIMINATE** — set `WAGONEYE_INSPECTION_VERSION=v2` on the EC2 env so the dashboard reads the payload on its V2 tab. Left as `v1` default only because `dashboard_ingest` was originally modeled on the pre-v2 legacy feed. **Action: set the env var in `deploy/wagon-eye.env`.** |
| `inspection_data` | object | object | OK. |

---

## 2. `inspection_data` — common fields

| Field | Production v2 | v5 dashboard | Disposition |
|---|---|---|---|
| `raw_video_name` | raw clip basename | from `source_video_urls[camera]` (or synthesized) | **OK** (equivalent). |
| `identified_by` | `"model-v3"` | `"model-v3"` (`_model_id`) | **OK** — matches. |
| `upload_timestamp` | `%Y-%m-%dT%H:%M:%S` | same | **OK**. |
| `upload_timestamp_readable` | `%d-%m-%Y %H:%M:%S IST` | same | **OK**. |
| `direction` | `left-to-right` / `right-to-left` / `unknown` | `"unknown"` (always) | **JUSTIFY (degraded)** — optical-flow direction is not persisted in any finalized artifact; recomputing it is out of scope for a read-only adapter. Safe because the **rake load label** is provided by the measured `rake_status` (fused load counts), a stronger signal than the direction heuristic. Recorded in `_adapter.degraded_fields`. |
| `pdf_report_url` | per-camera PDF URL | camera PDF URL | **OK**. |
| `trimmed_video_url` | trimmed clip URL | `src_url` | **OK**. |
| `segment_type_map` | `{str(seg_id): {type, number[, wagon_count]}}` | `{str(idx): {type, number}}` | **JUSTIFY** — keyed by wagon_index; `type`/`number` present. Top-flavour `wagon_count` sub-field is omitted (v5 side-style map). Safe: consumers key by the map key; the per-wagon `wagon_count` is still available in `wagon_segments`. |
| `damage_model_active` | bool | (not emitted) | **JUSTIFY** — v5 always runs damage when the model is present; when absent the wagon fields read `NO_DATA`, which already tells the consumer. Additive-only omission. Can be added if the dashboard reads it. |

---

## 3. `inspection_data` — side-camera fields (RIGHT_UP / LEFT_UP)

| Field | Production v2 | v5 dashboard | Disposition |
|---|---|---|---|
| `total_wagons` | fused/global | `summary.total_wagons` (global fused) | **OK**. |
| `doors_open` / `doors_closed` | per-camera door counts | computed from this camera's `{side}_door` | **OK**. |
| `damaged_wagons` | side damage count | side-camera damaged count (now incl. **side damage**) | **OK** — side damage wired (this milestone). |
| `num_engines` | engine count | `summary.engine_count` | **OK**. |
| `wagon_number_results` | keyed by `str(wagon_count)` | keyed by `str(wagon_index)` | **JUSTIFY / partial ELIMINATE** — both are per-wagon dicts `{is_valid_11_digit, display_number}`. Key differs: production uses the wagon-only 1-based counter, v5 uses `wagon_index`. For a rake with no engines/brakevans before a wagon these coincide; otherwise the keys shift. If the dashboard joins on this key, re-key to the wagon-only counter. **Action: confirm the dashboard's join key; re-key if required (one-line change in `dashboard_ingest`).** |
| `loco_number_results` | keyed by `str(loco_id)`, `{is_valid_5_digit, display_number, raw_number, confidence}` | **now populated** on RIGHT_UP from fused ENGINE-wagon loco OCR, same shape | **ELIMINATE** — implemented this milestone (was empty). |
| `wagon_segments[]` | `{segment_id, segment_type, wagon_count, door_status, door_close_detected, damage_detected, wagon_frames[], wagon_number, is_valid_wagon_id}` | `{segment_id, ... door_status, damage, wagon gallery, ...}` (adapter shape) | **JUSTIFY** — carries the same per-wagon facts (door status, damage, number, frames) in the adapter's field names. `door_close_detected` is available in the door feature JSON; add to the segment if the dashboard requires the exact key. |
| `loco_frames` / `total_loco_frames` | loco frame gallery + count | `[]` / `0` (degraded) | **JUSTIFY (degraded)** — v5 has no separate loco *frame* materialization; the loco **numbers** are now present in `loco_number_results`. The frame gallery is a nice-to-have gallery, not an inspection result. Recorded in `_adapter.degraded_fields`. |
| `problem_frames` / `problem_frames_by_type` | annotated evidence list | synthesized from present evidence JPEGs | **OK** (equivalent; only real files referenced). |

---

## 4. `inspection_data` — top-camera fields (RIGHT_UP_TOP / LEFT_UP_TOP)

| Field | Production v2 (top flavour) | v5 dashboard | Disposition |
|---|---|---|---|
| `wagons_loaded` / `wagons_empty` | load counts | via `summary.loaded/empty` + `rake_status` | **JUSTIFY** — the fused load counts are present at train level (`summary`); the top-camera payload reports `rake_status`. Per-camera `wagons_loaded/empty` can be added if the dashboard reads them per camera. |
| `body_dmg_wagons` / `floor_dmg_wagons` / `probable_damage_wagons` | per-class top-damage counts | not broken out per class in the dashboard payload | **JUSTIFY** — the v5 **damage feature JSON** and combined report carry the full per-class booleans (`body_dmg_detected`, `floor_dmg_detected`, `*_probable_detected`); the dashboard payload rolls them into `damaged_wagons`. Add the per-class counts to the top payload if the dashboard surfaces them. |
| `damaged_wagons` | confirmed top damage | top-camera damaged count | **OK**. |

---

## 5. v5-only additive fields (not in production v2)

| Field | Why present | Disposition |
|---|---|---|
| `rake_status` | measured load proxy for the degraded `direction` | **JUSTIFY** — additive, informative; ignore-safe. |
| `detected_video_url` / `raw_video_urls` | processed-video links | **JUSTIFY** — additive. |
| `_adapter` (`degraded_fields`, provenance) | transparency about what was faithfully reproduced vs degraded | **JUSTIFY** — additive metadata; lets a consumer see exactly what differs. |

---

## 6. Summary of actions

**Eliminated this milestone:** `loco_number_results` populated (loco OCR); side
damage flows into `damaged_wagons`.

**Eliminate via config (no code):** set `WAGONEYE_INSPECTION_VERSION=v2`
(envelope `version`).

**Confirm with dashboard team, then re-key/add if required (small, localized):**
1. `wagon_number_results` / `wagon_segments` key = wagon-only counter vs
   `wagon_index` — re-key `dashboard_ingest` if the dashboard joins on it.
2. Add `door_close_detected`, `damage_model_active`, per-class top-damage counts,
   and per-camera `wagons_loaded/empty` to the payload **only if** the dashboard
   reads those exact keys (each is an additive one-liner; the underlying data
   already exists in the feature JSON / combined report).

**Justified degradations (safe, documented in `_adapter.degraded_fields`):**
`direction`→"unknown" (rake_status compensates); `loco_frames`/`total_loco_frames`
empty (loco *numbers* present instead).

**Net:** the v5 per-camera dashboard payload is structurally a production-v2
`inspection_data` document. The only *value* differences are (a) the `version`
label (config), (b) the degraded `direction`/`loco_frames` (safe, compensated),
and (c) a wagon-key convention + a few additive-only per-class breakdowns to
confirm against the live dashboard. None fabricate data; every degraded field is
self-declared.
