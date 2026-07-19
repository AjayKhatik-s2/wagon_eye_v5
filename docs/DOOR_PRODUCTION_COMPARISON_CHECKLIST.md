# Door — Production vs v5 Comparison Checklist

Purpose: confirm the v5 Door processor reproduces the **production** door output.
Production source of truth = the `Train-Inspection-Engine` side-camera pipeline
(`left_up` / `right_up`), whose `inspection_data.json` carries per-wagon
`door_status`, `door_close_detected`, and door problem-frame evidence.

---

## 0. CRITICAL: how to align wagons before comparing

Production counts wagons **independently per camera** (its own gap segments);
v5 uses **one GlobalTrainState** (`GW_n`, shared across cameras). The counts can
differ (dev sample: production RIGHT_UP=60 vs v5=62). **This is expected — it is
the Stage-1 change, already validated separately — and must NOT be read as a
door defect.** Therefore run the comparison at two levels:

### Level 1 — Algorithm parity (PRIMARY; isolates the door logic)
Feed the **identical frame set** to both door implementations and compare. v5
uses the same model (`side_damage.pt`), same per-camera confidence (0.85/0.88),
same band `gap_tolerance` (5), same edge skip (10), and the same
`OPEN-iff-door_open` rule — so on the same frames the result must be **identical**.
Procedure:
1. Pick a wagon's frames (e.g. v5 `wagon_cache/GW_k/right_up/`).
2. Run production's side door/damage detection + banding on those exact frames
   (production `DamageDetector._detect_side` / notebook
   `detect_damage_doors_on_all_frames`).
3. Compare `door_status`, `door_close_detected`, the door bands (count,
   start/end, best_frame), and the best-frame bbox. **Expect exact match.**
This isolates the door algorithm from wagon-count differences and is the
strongest evidence of behavioural equivalence.

### Level 2 — End-to-end, position-aligned (secondary)
Run the **full production pipeline** and **full v5** on the same four trimmed
clips. Align wagons by **rake position / time window**, not by id:
- production side `wagon_segments[i]` (or `segment_type_map`) ordered along the rake
- v5 `GW_n` ordered by `wagon_index`
Match on overlapping `[start_time, end_time]`. Where counts differ, some wagons
have no 1:1 partner — record those separately (a Stage-1 boundary difference, not
a door mismatch). Compare door fields only on matched pairs.

---

## 1. Per-wagon comparison table (fill one row per aligned wagon)

| Rake pos | prod wagon id | v5 GW_n | Field | Production | v5 | Match? | Note |
|---|---|---|---|---|---|---|---|
| | | | door_state (right) | open/closed | OPEN/CLOSED | ☐ | RIGHT_UP↔right_up |
| | | | door_state (left) | open/closed | OPEN/CLOSED | ☐ | LEFT_UP↔left_up |
| | | | door_close_detected (R) | bool | bool | ☐ | |
| | | | door_close_detected (L) | bool | bool | ☐ | |
| | | | door confidence (R) | — / band conf | door_confidence | ☐ | prod had no explicit conf; compare band best conf |
| | | | evidence best frame (R) | frame # | metadata.frame_idx | ☐ | same/near frame? |
| | | | evidence bbox (R) | [x1,y1,x2,y2] | sides.right.bbox | ☐ | IoU ≥ 0.7 target |
| | | | #door_open bands (R) | n | len(tracks where OPEN) | ☐ | |
| | | | #door_close bands (R) | n | door_close via tracks/flag | ☐ | |

Repeat for every aligned wagon (both cameras).

---

## 2. Field-by-field mapping (production → v5)

| Production (`inspection_data.json`, side) | v5 (`wagon_states/door/<CAM>/GW_n.json`) |
|---|---|
| `wagon_segments[i].door_status` = "open"/"closed" | `right_door` (RIGHT_UP file) / `left_door` (LEFT_UP file) = OPEN/CLOSED |
| `wagon_segments[i].door_close_detected` | `door_close_detected` |
| `door_band_info` / `door_best_frames` | `tracks[]` (state=OPEN) + evidence `sides.<side>.frame_idx` |
| `door_close_band_info` / `door_close_best_frames` | `door_close_detected` + (bands summarized in `tracks` when CLOSED) |
| problem_frame (door_open, annotated red) | `evidence/GW_n/door/<CAM>/<side>_best.jpg` (red box) |
| train counts `doors_open` / `doors_closed` | derived by fusion/reporting from per-wagon `*_door` |

> Note: v5 splits door authority by camera (RIGHT_UP→right, LEFT_UP→left), which
> matches production (each side camera observes its own side's doors). Compare
> the RIGHT_UP file to production `right_up` and the LEFT_UP file to `left_up`.

---

## 3. Comparison dimensions (check each)

- [ ] **door_state** — per aligned wagon, per side: production `door_status` vs v5 `{side}_door`.
- [ ] **left_door** — LEFT_UP file `left_door` vs production `left_up` door_status.
- [ ] **right_door** — RIGHT_UP file `right_door` vs production `right_up` door_status.
- [ ] **confidences** — v5 `*_door_confidence` vs production door band best confidence (production has no single "door confidence"; use the winning band's best-frame conf).
- [ ] **evidence image** — same door instance shown; annotation colour matches (open=red, close=green); box actually on the door.
- [ ] **frame selection** — v5 evidence `frame_idx` equals/near production door best_frame (same band → same best frame expected under Level 1).
- [ ] **JSON fields** — all expected keys present, correct types, OPEN/CLOSED vocabulary (never PARTIAL/DAMAGED).
- [ ] **missing detections** — wagons production flags OPEN that v5 marks CLOSED (false negatives). Investigate: frame set difference (edge skip / boundary), or count-alignment artefact.
- [ ] **false detections** — wagons v5 marks OPEN that production marks CLOSED (false positives). Investigate the same causes.

---

## 4. Discrepancy triage

| Discrepancy | Likely cause | Action |
|---|---|---|
| door_state differs but frames differ | wagon-boundary/count difference (Stage 1) | Re-check via Level 1 on identical frames; if Level 1 matches, it's not a door defect. |
| door_state differs on identical frames | thresholds/banding drift | Confirm v5 uses 0.85/0.88 + gap_tol 5 + edge skip 10; diff the detection lists. |
| v5 all CLOSED | class-name mismatch | `YOLO('models/production/side_damage.pt').names` must include `door_open`/`door_close`. |
| confidence off | prod aggregates differently | Document the definition used; not a pass/fail blocker if state matches. |
| evidence frame differs | best-frame tiebreak | Acceptable if same band + comparable conf; note it. |
| bbox differs | detection jitter | IoU ≥ 0.7 acceptable; large gaps → investigate. |

---

## 5. Summary scorecard (fill after the run)

```
Batch: __________________________  build SHA: __________________  device: ______
Comparison level used: [ ] Level 1 (identical frames)  [ ] Level 2 (end-to-end)

Aligned wagons compared: ____ / prod ____ / v5 ____   (unmatched: ____)
door_state agreement (right): ____ / ____   ( ____ %)
door_state agreement (left):  ____ / ____   ( ____ %)
door_close_detected agreement: ____ / ____
False negatives (prod OPEN, v5 CLOSED): ____   ids: __________
False positives (prod CLOSED, v5 OPEN): ____   ids: __________
Evidence spot-check (n=___): ____ correct

VERDICT: [ ] PASS   [ ] PASS-WITH-NOTES   [ ] FAIL
Notes: ______________________________________________________________________
```

Target for PASS: on Level-1 (identical frames) comparison, **100%** door_state
agreement (the algorithm is a faithful port). On Level-2, high agreement with all
mismatches explained by wagon-boundary/count differences, not door logic.
