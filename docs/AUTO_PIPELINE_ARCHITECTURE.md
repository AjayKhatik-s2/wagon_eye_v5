# WagonEye — Automatic Pipeline Architecture (Definitive Design Specification)

> **Purpose of this document.** This is the permanent architectural reference for
> WagonEye's automatic railway-inspection pipeline. It is not a deployment guide,
> a user manual, or API documentation. It explains **every architectural decision
> and why it exists**, so that an experienced engineer who has never seen the
> repository could rebuild the entire system from this document alone and
> reproduce its external behaviour exactly. It favours reasoning and contracts
> over code.

**Audience:** a senior software engineer, new to the project.
**Scope:** the automatic pipeline from raw CCTV video in S3 to dashboard
ingestion, email, and archive.
**Companion docs (operational, not required to understand the design):**
`WAGONEYE_V5_ARCHITECTURE.md` (one-page diagram), `WAGONEYE_V5_VALIDATION_CHECKLIST.md`,
`WAGONEYE_V5_EC2_VALIDATION_GUIDE.md`, `WAGONEYE_V5_SCHEMA_PARITY.md`,
`WAGONEYE_V5_PRODUCTION_SIGNOFF.md`.

---

## Table of contents
1. System goals
2. Complete system architecture
3. End-to-end lifecycle (Stage A → Stage 6)
4. Batch lifecycle
5. GlobalTrainState
6. Materialization
7. Feature processors
8. Fusion
9. Reporting
10. Delivery
11. Configuration
12. Deployment architecture
13. Performance engineering
14. Failure recovery
15. Repository structure
16. Development philosophy
17. Building a new implementation (the immutable contracts)

---

## 1. System goals

### 1.1 The business problem
A freight train (a **rake**) is a sequence of coupled vehicles: one or more
**locomotives** (engines) at the front, dozens of **wagons** carrying cargo, and
usually a **brake van** at the tail. Railways must inspect every passing rake for
safety and operational faults **without stopping the train**:

- **Doors** left open (safety-critical; cargo loss, fouling the loading gauge).
- **Structural damage** to the wagon body and floor (top view) and body sides.
- **Load state** (loaded vs empty) for operational accounting.
- **Wagon identity** — the 11-digit wagon number stencilled on the side — and the
  **5-digit locomotive number**, to tie the inspection to a specific vehicle.

WagonEye automates this: fixed cameras film the passing rake, and computer-vision
models extract per-wagon findings, producing an inspection report and a
structured record for a dashboard, per train, automatically, 24×7.

### 1.2 Why a single camera is not enough
A rake is a long 3-D object moving past fixed cameras. No single viewpoint sees
everything an inspection needs:

- The **side** of a wagon (doors, the stencilled number, side damage) is only
  visible from a **side camera**, and each side (left/right) needs its own camera
  — the right camera cannot see the left doors and vice-versa.
- The **top** of a wagon (roof/body damage, floor damage, whether it is loaded)
  is only visible from a **top-down camera**.
- One camera can be **occluded, glared, or mis-triggered** on any given pass; a
  second camera viewing the same physical gap provides redundancy.

Therefore WagonEye uses **four cameras**: `LEFT_UP` and `RIGHT_UP` (side views,
left and right), and `LEFT_UP_TOP` and `RIGHT_UP_TOP` (top views). Each camera is
**authoritative** for what only it can see, and the top cameras additionally
**corroborate** each other and the master.

| Camera | Physical view | Authoritative for |
|---|---|---|
| `RIGHT_UP` | side, right | **master timeline**, right door, wagon number (OCR), loco number, right side damage |
| `LEFT_UP` | side, left | left door, left side damage |
| `RIGHT_UP_TOP` | top, right rail | load (primary), top damage |
| `LEFT_UP_TOP` | top, left rail | load (support), top damage (support) |

### 1.3 Why asynchronous cameras make this hard
The four cameras are independent capture devices feeding an external uploader.
Their clips for the **same train** do **not** arrive in S3 at the same moment:
network jitter, per-camera encoding time, and per-camera upload schedules mean
one camera's clip can lag another by seconds to minutes, and occasionally a
camera fails to produce a clip at all.

This creates three hard requirements the architecture must satisfy:

1. **Grouping without a shared clock.** There is no message that says "these four
   clips are the same train." The system must **infer** the grouping from the
   timestamp embedded in each filename, within a tolerance.
2. **Progress before completeness.** Waiting for all four cameras before doing any
   work would stall the pipeline and abandon trains where a camera never arrives.
   The system must start as soon as the **master** camera is present, then attach
   the others as they arrive — **without recomputing** what was already decided.
3. **Bounded waiting.** A camera that never arrives must not block a train
   forever; there must be deadlines after which the train is finalized as a
   partial report.

The **incremental lifecycle** (§4) exists specifically to solve asynchronous
arrival, and **GlobalTrainState** (§5) exists specifically to make "attach later
without recomputing" safe.

### 1.4 Why production systems must run continuously
Trains arrive unpredictably around the clock. A batch job is the wrong model: it
would either miss trains between runs or waste resources polling with a cron. The
system runs as **long-lived services** that continuously poll S3, process each
train as its clips appear, and survive restarts by persisting their state to S3
so no train is lost or double-processed across a reboot.

### 1.5 Latency requirements
Latency is defined per train: from the moment the **last needed clip** is
available to the moment the dashboard/email is delivered. It is **soft**: a rake
report that arrives minutes after the train passed is fully useful (there is no
real-time control loop). The dominant cost is model inference over thousands of
frames across four cameras; on CPU this is tens of minutes per train, on GPU a
few minutes. The design therefore optimises for **not doing redundant work**
(decode once, infer once, never recompute the count) rather than for
sub-second response.

### 1.6 Throughput requirements
Throughput is "trains per hour the host can finish." A single station produces a
train every several minutes at peak. The design keeps up by:
- processing one train at a time to completion (bounded memory), while the
  **incremental lifecycle** lets multiple partially-arrived trains coexist as
  cheap on-disk manifests;
- running the four feature processors in parallel within a train;
- skipping any work whose inputs are unchanged (completion markers).
Scaling beyond one host is horizontal: one service instance per station (each
polls its own prefixes); there is no cross-train shared state to coordinate.

### 1.7 Why GlobalTrainState solves historical problems
The predecessor system counted wagons **independently on each camera** and tried
to reconcile the four differing counts at the very end (by matching upload
timestamps). This produced a family of chronic problems:

- **Divergent counts.** The four cameras routinely disagreed (e.g. 60/61/63/62
  wagons) because each missed or invented different gaps; the final report had to
  guess which count was "right."
- **No stable wagon identity.** "Wagon 7" on the left camera was not necessarily
  "wagon 7" on the right; cross-camera findings could not be reliably joined.
- **Repeated, redundant work.** Each camera re-decoded video, re-detected gaps,
  re-segmented wagons.
- **Fragile reconciliation.** Timestamp pairing broke when clocks drifted or two
  trains passed close together.

**GlobalTrainState** replaces this with a single, authoritative understanding of
the train computed **once**: the master camera (RIGHT_UP) defines the wagon
count, order, boundaries, `GW_n` ids, and ENGINE/WAGON/BRAKE_VAN class; the other
cameras only **repair** gaps the master missed (when they agree) and then
**enrich** the shared wagons with their own features. Every downstream stage
consumes this one truth. The chronic problems dissolve: one count, one stable id
space, decode/segment/count exactly once, and no end-of-pipeline reconciliation.

---

## 2. Complete system architecture

### 2.1 The two-service topology
```
Raw Cameras
    │  (external uploader writes clips)
    ▼
Raw S3 Bucket  ──────────────►  Extraction Service  ──────────►  Trimmed S3 Bucket
 biro-wagon-raw-video-copy     (train_extraction)              complete-train
                                                                     │
                                                                     ▼
                                                              Inspection Service
                                                              (orchestrator.master_runner --auto)
                                                                     │
                                                                     ▼
                                                              GlobalTrain Pipeline
                                                       (Stage 1..4b: reconstruct → materialize
                                                        → features → fuse → render)
                                                                     │
                                     ┌───────────────┬──────────────┼───────────────┐
                                     ▼               ▼              ▼               ▼
                                  Reports        Dashboard        Email          Archive
                              (PDF + JSON)   (inspection_data)  (one/batch)   (S3 batch tree)
```

**Every box explained:**
- **Raw cameras / external uploader.** Outside WagonEye. Produces short raw CCTV
  clips and uploads them to the raw bucket under a per-camera prefix, with a
  `YYYYMMDD_HHMMSS` timestamp in the filename. WagonEye treats this as an
  untrusted, at-least-once source.
- **Raw S3 bucket** (`biro-wagon-raw-video-copy`). Durable landing zone for raw
  clips. It is the extraction service's only input.
- **Extraction service** (`train_extraction`). Watches the raw bucket, detects
  when a clip contains a train pass, trims exactly the train portion (stitching
  across clip boundaries when a train spans two raw clips), and uploads the
  trimmed clip. It performs **no inspection**.
- **Trimmed S3 bucket** (`complete-train`). Holds
  `"<raw>_train.mp4"` clips. It is the boundary between the two services and the
  inspection service's only input.
- **Inspection service** (`orchestrator.master_runner --auto`). Watches the
  trimmed bucket, groups the four cameras' clips into trains, and runs the
  GlobalTrain pipeline per train.
- **GlobalTrain pipeline.** Stages 1–4b (reconstruction, materialization,
  feature inference, fusion, overlay rendering) — the analytical core.
- **Reports.** Per-camera PDFs + a combined PDF + a machine-readable combined
  JSON.
- **Dashboard.** A per-camera `inspection_data` payload POSTed to the railway's
  ingest API.
- **Email.** One summary email per train to operations.
- **Archive.** The full per-batch artifact tree persisted to S3, and the batch
  marked terminal so it is never reprocessed.

### 2.2 Why split into two services
Extraction and inspection have **different triggers, resource profiles, failure
modes, and change cadences**, so coupling them would make the whole system more
fragile:

- **Different work shapes.** Extraction is I/O-bound and cheap (a light classifier
  + ffmpeg trims); inspection is compute-heavy (many YOLO models over thousands
  of frames). Separating them lets each be sized, scaled, and restarted
  independently — e.g. inspection can run on a GPU box while extraction runs on a
  small CPU box.
- **Different cadence.** Extraction runs continuously per raw clip; inspection
  runs per *train* (a group of four trimmed clips). Their loops have different
  natural intervals and state.
- **Fault isolation.** A crash or backlog in inspection must not stop clips being
  trimmed and preserved; a bug in extraction must not corrupt an in-flight
  inspection. A clean S3 boundary means each can fail and recover alone.
- **Independent evolution.** The trim algorithm and the inspection models change
  on different schedules and by different teams. A stable file contract between
  them lets either side be replaced without touching the other.

### 2.3 Why S3 is the communication layer
The two services communicate **only** by reading and writing S3 objects — no
queue, no RPC, no shared database. This is deliberate:

- **Durability and replay.** Every intermediate artifact (trimmed clip, state,
  evidence, report) is a durable object. A consumer that was down simply reads
  what accumulated; there is nothing to "miss." Reprocessing is re-reading.
- **Decoupling in time.** Producer and consumer need not be up simultaneously;
  the bucket buffers arbitrary lag.
- **Zero coupling.** The only contract is object **naming + content**; neither
  service imports the other's code or shares a process. This is what makes "two
  services" real rather than cosmetic.
- **Native to the platform.** The deployment target is AWS; S3 gives durability,
  IAM-scoped access, and effectively unlimited retention for free, versus running
  and monitoring a broker.
- **Idempotent by object identity.** An object's key + ETag is a natural
  idempotency token; the pipeline keys its "already done" markers on them (§4,
  §13), so duplicate notifications and restarts cost nothing.

The trade-off — S3 is poll-based, not push — is acceptable because latency is
soft (§1.5); the services poll on an interval.

---

## 3. End-to-end lifecycle (Stage A → Stage 6)

This section walks a single train from the instant it begins entering the camera
until the dashboard updates. Each stage is described by: **purpose, inputs,
outputs, internal algorithm, state transitions, failure handling, performance,
recovery, exactly-once, expected runtime.**

### Stage A — Automatic extraction (raw → trimmed)
- **Purpose.** Convert continuous CCTV into bounded per-train clips, so the
  inspection service never has to reason about idle track or partial trains.
- **Inputs.** Raw clips under `raw-bucket/<camera_folder>/…`, each a few seconds
  to minutes of footage that may contain no train, part of a train, or a whole
  train.
- **Outputs.** Trimmed clips `"<raw_basename>_train.mp4"` in
  `trimmed-bucket/<camera_folder>/`, each containing exactly one train pass
  (padded with a small buffer), named after the **first** raw clip of the train
  so its timestamp identifies the train.
- **Internal algorithm.** For each new raw clip: a lightweight **frame
  classifier** (side/top classification model) labels frames as `train` vs
  `empty_track` (a parallel train on a second track is explicitly ignored so it
  cannot end our train early). A **three-phase segment finder** locates train
  spans: (1) scan frames (strided for speed) accumulating a "train ended" streak
  once enough consecutive empty-track frames are seen, applying multi-second
  start/end buffers; (2) merge nearby spans; (3) determine travel direction via
  optical flow. A train that reaches the end of a clip still "in train" is held
  as **ongoing**; when the next clip arrives it is checked for continuation and,
  if continuous, the clips are concatenated before trimming. ffmpeg trims the
  final span (GPU NVENC with CPU libx264 fallback).
- **State transitions.** Per camera: `idle → in-train → (ongoing across clips) →
  emitted`. Held ongoing state lives in the extractor's own S3 state store plus a
  local processed-key ledger.
- **Failure handling.** ffmpeg NVENC failure falls back to CPU; a clip that
  raises is logged and skipped (retried next sweep); a raw key is marked
  processed only **after** a successful extract call.
- **Performance.** I/O + a cheap classifier; frame scanning is strided; the
  heavy inspection models are *not* used here.
- **Recovery.** The local ledger + S3 ongoing-state mean a restart never
  re-extracts a handled clip and preserves cross-clip continuity.
- **Exactly-once.** Best-effort: the ledger prevents re-extraction; a duplicate
  raw upload with the same key is skipped.
- **Expected runtime.** Seconds per raw clip.

### Stage 0 — Batch acquisition (grouping the four cameras)
- **Purpose.** Turn independently-arriving trimmed clips into a **TrainBatch**
  (the four cameras of one train) and drive it through the incremental lifecycle.
- **Inputs.** Trimmed clips in the input bucket/prefixes; the persistent set of
  already-processed batch keys; per-batch manifests.
- **Outputs.** An active **BatchManifest** per train; the four local clip paths
  under `batch_outputs/<batch_key>/downloads/`.
- **Internal algorithm.** List source clips, extract each filename's
  `YYYYMMDD_HHMMSS`, and **cluster** clips whose timestamps fall within a
  tolerance (≈120 s) into batches keyed by the cluster's representative stamp.
  Decide which batch is ready to run (master present, or an aged partial). This
  is pure grouping — **no detection logic**.
- **State transitions.** See §4 (the batch lifecycle state machine).
- **Failure/recovery/exactly-once.** The manifest is the durable checkpoint;
  terminal batch keys are recorded so a restart never reprocesses them; an
  ambiguous clip (within tolerance of two active batches) is held for review
  rather than mis-attached.
- **Expected runtime.** Milliseconds (listing + string parsing).

### Stage 1 — Global reconstruction → GlobalTrainState
- **Purpose.** Produce the single authoritative understanding of the train:
  how many wagons, their order, `GW_n` ids, time boundaries, and
  ENGINE/WAGON/BRAKE_VAN class. This is the **only** stage allowed to count
  wagons or detect gaps.
- **Inputs.** The four trimmed clips (RIGHT_UP master + up to three support);
  the reconstruction models.
- **Outputs.** `global_state/global_train_state.json` (the immutable
  GlobalTrainState) + `per_camera_tracking.json` (per-camera fps, dimensions,
  tracked gaps) + Stage-1 debug overlay videos.
- **Internal algorithm.** Per camera, a **gap-detection YOLO** runs on every
  frame; each inter-wagon gap is **tracked** over time (Kalman filter +
  hit/miss persistence) into a stable gap event. On the master (RIGHT_UP), each
  segment between gaps is **classified** ENGINE/WAGON/BRAKE_VAN by voting.
  **Cross-camera fusion** then aligns the support cameras' gaps against the
  master timeline and **recovers** gaps the master missed **only when ≥2 support
  cameras agree** (recorded as an audited correction). The result is assembled
  into `GW_1..GW_N` with deterministic ids and master-clock time windows.
- **State transitions.** Runs during `RECONSTRUCTING`; on success the batch
  transitions to `GLOBAL_STATE_SEALED` (the state becomes immutable).
- **Failure handling.** Subprocess exit ≠ 0, missing JSON, or `total_wagons == 0`
  → the batch is `failed_no_global_state` and aborts (nothing to report). A
  missing master → the batch waits, then fails safely (it will not seal an
  unvalidated non-master timeline).
- **Performance.** The heaviest stage (gap YOLO on every frame of four videos).
- **Recovery.** If the batch restarts before sealing, reconstruction re-runs
  from the trimmed clips; after sealing it is never re-run.
- **Exactly-once.** Guarded by the lifecycle: sealing happens once; a sealed
  state is never recomputed.
- **Expected runtime.** Minutes on GPU; tens of minutes on CPU (dev reference:
  ~47 min for a 62-wagon, ~264 s, 15 fps train on CPU).

### Stage 2 — Materialization (wagon_cache)
- **Purpose.** Decode each video **exactly once** and lay every wagon's frames on
  disk, so no later stage ever touches a video for analysis again.
- **Inputs.** GlobalTrainState (time windows) + the four trimmed clips.
- **Outputs.** `wagon_cache/GW_n/<camera>/frame_NNNNNN.jpg` (JPEG q95) + a
  per-camera materialization marker.
- **Internal algorithm.** Per camera, open the video once, map each `GW_n`'s
  master-clock window to that camera's local frame range (`start_time × local_fps`
  … `end_time × local_fps`), walk frames sequentially, and write each frame's
  JPEG into the wagon bucket it falls in. Cameras run in parallel (one thread
  each; OpenCV releases the GIL during decode).
- **State transitions.** Runs in `PROCESSING_AVAILABLE_CAMERAS`; per-camera,
  idempotent.
- **Failure handling.** One camera's unreadable video → an empty subtree for that
  camera (batch continues as partial); the stage never fabricates frames.
- **Performance.** I/O-bound; the single most important performance decision in
  the system (§13). Fast (dev reference: ~22 s for ~15,500 JPEGs).
- **Recovery / exactly-once.** The marker keys on (video ETag + GST version +
  materializer schema); a matching, non-empty cache is skipped; a changed ETag
  rebuilds just that camera via a temp build + atomic swap (a failed rebuild
  keeps the previous cache).
- **Expected runtime.** Seconds to low minutes.

### Stage 3 — Feature inference (door / OCR / load / damage)
- **Purpose.** Enrich each shared wagon with per-feature findings, using the
  camera that is authoritative for each feature.
- **Inputs.** `wagon_cache` frames + the production feature models + (OCR) the
  loco-region model + easyocr.
- **Outputs.** `wagon_states/<feature>/<CAMERA>/GW_n.json` (one per wagon per
  authoritative camera) + `evidence/GW_n/<feature>/<CAMERA>/…` snapshots.
- **Internal algorithm.** Each processor is a thin driver over a pure cached-frame
  iterator; it runs the production model at the production thresholds, groups
  detections into bands (or votes classifications), decides the per-wagon result,
  and persists best-frame evidence. Details in §7.
- **State transitions.** Runs in `PROCESSING_AVAILABLE_CAMERAS` /
  `PROCESSING_LATE_CAMERA`; the four features run in parallel; wagons within a
  feature run serially so each model loads once per process.
- **Failure handling.** Missing model → `NO_DATA` for every wagon (clear error,
  no dummy inference); a per-wagon exception → `FAILED` for that wagon only; a
  whole-feature crash → that feature's wagons `FAILED`, others continue.
- **Performance.** The bulk of inference cost; parallel across features; the
  slowest is OCR (per-frame detect + multi-run easyocr).
- **Recovery / exactly-once.** A per-`(camera, feature)` completion marker keys
  on ETag + GST version + **production model SHA-256** + processor schema +
  threshold hash; an up-to-date marker skips the feature; any input change
  invalidates exactly that marker.
- **Expected runtime.** Minutes (GPU) to tens of minutes (CPU), dominated by OCR
  and damage.

### Stage 4 — Fusion → UnifiedWagonState
- **Purpose.** Merge the four per-feature results per wagon into one authoritative
  record, applying fixed camera-authority rules and computing anomalies.
- **Inputs.** `wagon_states/<feature>/<CAMERA>/GW_n.json` + GlobalTrainState.
- **Outputs.** `wagon_states/unified/GW_n.json` (one UnifiedWagonState per wagon).
- **Internal algorithm & authority rules.** See §8. Deterministic and idempotent;
  a missing/pending/failed feature yields `NO_DATA`, never a false OK.
- **State transitions.** Runs after the present cameras' features; re-runs
  cheaply when a late camera adds features (fusion revision increments).
- **Failure handling.** A per-wagon fusion error keeps the previous unified JSON.
- **Performance.** Fast (no inference; JSON merge).
- **Expected runtime.** Sub-second to seconds.

### Stage 4b — Overlay rendering (visualization only)
- **Purpose.** Produce human-viewable overlay videos for the report links.
- **Inputs.** GlobalTrainState + UnifiedWagonState + evidence metadata +
  per-camera tracking + the raw videos.
- **Outputs.** `processed_videos/<CAM>_processed.mp4`.
- **Internal algorithm.** Re-open each raw video **only to draw** wagon ids, gap
  banners, feature boxes on the recorded best frames, and anomaly banners. It runs
  **no detector** — every overlay is drawn from already-persisted state.
- **Failure handling.** A per-camera render failure is isolated (that mp4 is
  omitted; the batch continues).
- **Expected runtime.** Decode-bound; low minutes.

### Stage 5 — Reporting (camera + combined)
- **Purpose.** Emit the human report (PDFs) and the machine record (JSON).
- **Inputs.** GlobalTrainState + UnifiedWagonState + evidence + wagon_cache.
- **Outputs.** four camera PDFs + `combined_train_report.pdf` +
  `combined_train_report.json`.
- **Internal algorithm.** Pure presentation: reshape the fused state into the
  report view model and render. **No inference is ever re-run** in reporting.
  Details in §9.
- **Failure handling.** Combined-PDF failure → `report_failed` (JSON still
  written, email suppressed); a camera PDF failure is isolated.
- **Expected runtime.** Seconds.

### Stage 6 — Delivery (dashboard, upload, email, archive)
- **Purpose.** Ship the outputs exactly once: dashboard ingest, S3 upload, one
  email, and archive; mark the batch terminal.
- **Inputs.** The finalized reports + evidence.
- **Outputs.** Per-camera dashboard payloads POSTed to the ingest API; report
  objects in S3; one email; the archived batch tree; a terminal status.
- **Internal algorithm & exactly-once.** See §10. A `finalization.json` marker
  records what was already uploaded/emailed/ingested (keyed by report hash +
  revision + idempotency key), so a restart never double-delivers.
- **Failure handling.** Ingest/upload/email failures are logged and non-fatal to
  the batch outcome; the exactly-once markers make retries safe.
- **Expected runtime.** Seconds (network-bound).

### 3.1 Stage timeline (one train, happy path)
```
t0  master trimmed clip appears ──► batch discovered
t1  support clips arrive (seconds..minutes) OR support window expires
t2  SEAL GlobalTrainState (Stage 1)                [immutable from here]
t3  materialize present cameras (Stage 2)
t4  features on present cameras, parallel (Stage 3)
t5  fusion (Stage 4) → interim report (Stage 5) [local only by default]
t6  late camera arrives ─► materialize+features for it ─► re-fuse ─► re-report
t7  all cameras present OR final deadline ─► FINALIZE
t8  ONE upload + ONE email + dashboard ingest (Stage 6) ─► archive ─► terminal
```

---

## 4. Batch lifecycle

### 4.1 Why a lifecycle exists
Because cameras arrive asynchronously (§1.3), a train is not a single atomic job;
it is a **long-lived entity** that accumulates cameras over time and must survive
restarts. The lifecycle is an explicit **state machine** persisted in a
**BatchManifest** so that at any instant the system knows exactly what has been
done and what remains — and a crash/restart resumes precisely there.

### 4.2 The state machine
```
DISCOVERED
   │ (a clip for this train appeared)
   ▼
COLLECTING_CAMERAS ──► WAITING_FOR_MASTER ──(master absent past deadline)──► FAILED_NO_GLOBAL_STATE
   │ (master present)
   ▼
WAITING_FOR_SUPPORT   (short window so support cameras can improve gap recovery)
   │ (support window expires OR all present)
   ▼
RECONSTRUCTING ──(exit≠0 / 0 wagons)──► FAILED_NO_GLOBAL_STATE
   │
   ▼
GLOBAL_STATE_SEALED     ◄══════ IMMUTABLE: count / GW_n / boundaries / class fixed
   │
   ▼
PROCESSING_AVAILABLE_CAMERAS   (materialize + features + fusion + interim report)
   │
   ├──(some cameras still missing, before final deadline)──► WAITING_FOR_LATE_CAMERAS
   │        │ (a late camera arrives)
   │        ▼
   │   PROCESSING_LATE_CAMERA ──► (attach its features, re-fuse, re-report) ──┐
   │        ▲───────────────────────────────────────────────────────────────┘
   │ (all cameras present) OR (final deadline reached)
   ▼
FINALIZING ──► ONE upload + ONE email + archive ──► terminal:
        COMPLETED | COMPLETED_PARTIAL | REPORT_FAILED | FAILED
```

### 4.3 The three deadlines (why there are exactly three)
They are **semantically distinct** and must not be conflated:
- **Master wait** (default 10 min from first-seen): how long to wait for the
  RIGHT_UP master before the train fails safely. Without a master there is no
  authoritative timeline, so there is nothing to seal.
- **Support-fusion window** (default 3 min after the master arrives): a *short*
  window to let support cameras show up so cross-camera gap recovery is better.
  When it expires the train **seals from the master + whatever support exists** —
  it does **not** wait for the final deadline. This trades a little latency for
  count accuracy.
- **Final-camera wait** (default 30 min from first-seen): the hard close.
  Still-missing cameras become `CAMERA_MISSING_FINAL` and the train finalizes as
  `COMPLETED_PARTIAL`. This bounds total latency.

### 4.4 Persistence and markers
| Artifact | Location | Role |
|---|---|---|
| Batch manifest | `batch_outputs/<key>/manifest.json` (+ S3 mirror) | resumable lifecycle state; schema-versioned; atomic writes; an unknown/newer schema is refused rather than misread |
| Materialization marker | `wagon_cache/.materialized/<CAMERA>.json` | skip re-extraction; keyed on ETag + GST version + materializer schema |
| Feature marker | `wagon_states/.features/<CAMERA>/<feature>.json` | skip re-inference; keyed on ETag + GST version + **production model SHA-256** + processor schema + threshold hash |
| Finalization marker | `delivery/finalization.json` | exactly-once upload/email/ingest; records report hashes, URLs, email status, idempotency key |
| Processed batches | `processed_batches.json` (S3) | terminal batch keys; never reprocessed |

### 4.5 Camera events
- **Arrival.** A clip whose timestamp joins this batch's cluster → the camera is
  marked present; if the state is already sealed, only that camera's
  materialize+features+fusion run (a late attach).
- **Late camera.** Arrives after sealing but before the final deadline → attached
  with **no reseal and no renumbering**; it can only add door/OCR/load/damage to
  existing wagons. This is the entire reason the state is immutable (§5).
- **Missing camera.** Absent at the final deadline → `CAMERA_MISSING_FINAL`; its
  fields report as missing (never a false OK); the report is partial.
- **Terminal late camera.** Arrives after the batch is terminal → logged and
  **ignored**; a sealed report is never reopened.
- **Ambiguous camera.** A clip within tolerance of two active batches and too
  close to decide → held in `videos_for_review`, not silently attached.

### 4.6 Resume / restart / recovery / archive / deletion
- **Restart at any point** resumes from the manifest + on-disk markers, repeating
  no completed reconstruction, materialization, feature inference, upload, or
  email.
- **Archive.** On finalize, the batch tree (evidence + processed videos + states +
  reports) is uploaded to S3 under `archive/<key>/`; the batch is recorded
  terminal.
- **Deletion.** Local `batch_outputs/<key>/` may be pruned after archive; the S3
  archive + terminal record are the durable truth. Only terminal batches are safe
  to delete locally.

---

## 5. GlobalTrainState

GlobalTrainState is the **spine** of the entire system. Everything downstream is a
deterministic function of it.

### 5.1 Why it is immutable after sealing
The four cameras arrive at different times. If the wagon count or `GW_n` ids could
change when a late camera arrives, then every artifact already produced from the
earlier state (wagon_cache windows, per-feature JSON, evidence, interim reports)
would be invalidated and would have to be recomputed — and worse, findings
already delivered could silently change identity ("the damage on wagon 7" might
refer to a different wagon after a renumber). Immutability makes late attachment
**safe and cheap**: a late camera can only add features to wagons that already
exist with fixed ids. This is the property that turns asynchronous arrival from a
correctness hazard into a routine, incremental enrichment.

### 5.2 Why wagon numbering never changes
`GW_n` is assigned once by the master timeline in rake order and is the **join key**
across cameras, features, evidence, reports, and the dashboard. If it changed,
cross-camera joins would break and historical references would rot. Recovered gaps
(from support cameras) are **inserted** into the master timeline with provenance
(`split_from_global_id`), but the resulting ids are assigned deterministically at
seal time and then frozen.

### 5.3 The data model
- **`GlobalWagon`** — one physical wagon after fusion:
  - `global_id` (`"GW_7"`) — the immutable id / join key.
  - `wagon_index` — 1-based position in the rake.
  - `start_frame_master` / `end_frame_master` — master-frame boundaries.
  - `start_time` / `end_time` — master-clock seconds (the canonical boundary;
    each camera's local frame range is derived from these × that camera's fps).
  - `classification` — `ENGINE | WAGON | BRAKE_VAN | UNKNOWN` (RIGHT_UP is the
    only authority).
  - `classification_confidence`.
  - `supporting_cameras` — which cameras corroborated this wagon.
  - `split_from_global_id` — provenance when created by a recovered gap.
  - `leading_gap` / `trailing_gap` — the boundary gap events (or an edge marker).
- **`GlobalTrainState`** — the train:
  - `total_wagons`, `wagons[]`.
  - `master_camera` (`RIGHT_UP`), `master_fps`, `master_total_frames`.
  - `per_camera_local_counts` / `per_camera_gap_counts` / `per_camera_status` —
    per-camera bookkeeping (the raw divergent counts, retained for audit).
  - `corrections_applied[]` — each recovered gap: master time/frame, supporting
    cameras, mean confidence, time spread.
  - `participating_cameras` / `missing_at_reconstruction` — who was present at
    seal.
  - `reconstruction_mode` (MASTER_ONLY / MASTER_WITH_SUPPORT_AVAILABLE /
    MASTER_WITH_FUSED_SUPPORT), `support_fusion_used`, `support_gap_recoveries`,
    `reconstruction_confidence`, `sealed_at`, `sealing_reason`.

### 5.4 Example (abbreviated)
```json
{
  "schema": "wagon_eye.global_train_state.v1",
  "master_camera": "RIGHT_UP", "master_fps": 15.0, "master_total_frames": 3969,
  "total_wagons": 62, "regular_wagon_count": 58, "engine_count": 2, "brake_van_count": 1,
  "wagons": [
    {"global_id": "GW_1", "wagon_index": 1, "start_time": 5.42, "end_time": 12.58,
     "classification": "ENGINE", "classification_confidence": 0.86,
     "supporting_cameras": ["LEFT_UP_TOP","RIGHT_UP_TOP"],
     "leading_gap": {"source": "edge"}, "trailing_gap": {"track_id": 3, "confidence": 0.81}},
    {"global_id": "GW_2", "wagon_index": 2, "start_time": 12.58, "end_time": 17.90,
     "classification": "WAGON", "classification_confidence": 0.79, "split_from_global_id": null}
  ],
  "per_camera_gap_counts": {"RIGHT_UP": 59, "LEFT_UP": 60, "RIGHT_UP_TOP": 62, "LEFT_UP_TOP": 61},
  "corrections_applied": [
    {"inserted_at_master_time": 12.58, "supporting_cameras": ["LEFT_UP","LEFT_UP_TOP","RIGHT_UP_TOP"],
     "mean_confidence": 0.76, "time_spread_sec": 0.80}
  ],
  "reconstruction_mode": "MASTER_WITH_FUSED_SUPPORT", "support_gap_recoveries": 3,
  "sealed_at": "2026-07-19T06:00:12+05:30", "sealing_reason": "all_cameras_present"
}
```

### 5.5 How support cameras repair the master (gap recovery)
The master may miss a gap (occlusion, glare, coupling geometry). Support cameras,
viewing the same physical gap, catch it. Fusion aligns each support camera's gaps
to the master timeline; where a cluster of unmatched support gaps agrees in time
across **≥2** cameras, a gap is **inserted** into the master timeline as a
recovered wagon boundary (audited in `corrections_applied`). Requiring ≥2 cameras
prevents a single support camera's false gap from splitting a wagon. This is why
the fused count can exceed the master's raw count (e.g. master 60 → fused 62).

### 5.6 Wagon ordering & classification
Order is the master timeline order (rake order) — deterministic and stable.
Classification (ENGINE/WAGON/BRAKE_VAN) is a **structural** property (where a
vehicle sits and what it is), so it belongs to the master reconstruction, not to a
per-camera feature. Load state (loaded/empty) is *not* structural — it is a
per-wagon appearance — so it is a **feature** (§7), not part of GlobalTrainState.

### 5.7 Why this replaces per-camera state
Per-camera counting produced N conflicting truths reconciled late and fragilely
(§1.7). GlobalTrainState produces **one** truth up-front with a stable id space, so
downstream stages never count, never reconcile, and never disagree. It is the
architectural core that makes everything else simple.

---

## 6. Materialization

### 6.1 Why repeated video decoding is expensive
Video decode is CPU-heavy and sequential. The predecessor decoded each clip
**3–4 times** (gap detection, then again for segmentation, then per feature),
multiplied by four cameras — the dominant cost after inference and pure waste, as
every decode produced the same pixels.

### 6.2 Why decoding once is correct (and sufficient)
Every analytical stage needs the **same pixels**: the frames of each wagon on each
camera. Nothing downstream needs random access to the video or a different
decode. So decode each video **once**, write the frames to disk keyed by wagon and
camera, and let every later stage read JPEGs. This is both faster and simpler
(later stages iterate files, not video), and it removes an entire class of bugs
(seek errors, container quirks) from the analytical path.

### 6.3 Why wagon_cache exists (and its layout)
`wagon_cache/GW_n/<camera>/frame_NNNNNN.jpg` is the shared substrate. The layout
is chosen so that:
- **A feature reads exactly the frames it needs** by opening one directory
  (`GW_n/<its authoritative camera>/`).
- **A wagon's evidence is co-located** across cameras (`GW_n/…`) so the combined
  report can pull all four cameras' snapshots for one wagon from one place.
- **Frame filenames carry the absolute frame index**, preserving each frame's
  identity for evidence metadata and overlay rendering.

### 6.4 How JPEG quality is selected
Quality is **q95**. These JPEGs are the exact pixels fed to the damage and OCR
models; lowering quality could shift a detection or an OCR read. q95 matches the
production reference frames, so validation compares like-for-like pixels. Quality
is a single config constant, and the materializer schema version is bumped when it
changes so stale lower-quality caches are rebuilt rather than reused.

### 6.5 Why frame timestamps are preserved / how cameras stay synchronized
Boundaries live in **master-clock seconds** in GlobalTrainState. Each camera's
local frame range for a wagon is computed as `time × that camera's fps`, so the
four cameras are aligned to the same wagon spans even at different frame rates.
The frame **index** in the filename preserves each frame's position for evidence
and overlays. Synchronization is thus a property of the shared master clock, not
of any per-camera assumption.

### 6.6 Cache invalidation
The per-camera materialization marker keys on **(source ETag, GST version,
materializer schema)**. If the source clip changes (ETag), or the state is
re-sealed (new GST version), or the cache format changes (schema bump), only the
affected camera's cache is rebuilt — into a temp directory, then atomically
swapped in, so a failed rebuild never destroys a valid cache. An unchanged,
non-empty cache is skipped, which is what makes restarts and late-camera attaches
cheap.

---

## 7. Feature processors

### 7.1 Philosophy: independent, stateless, authoritative
Each feature (door, OCR, load, damage) is a **completely independent** processor.
This independence is deliberate and load-bearing:
- **Fault isolation.** One feature crashing or a model missing must not affect the
  others; each writes only its own namespace and degrades to `NO_DATA` alone.
- **Parallelism.** Independence lets the four run concurrently.
- **Replaceability.** Any one feature's model or logic can be swapped without
  touching the others (§17).
- **Single responsibility.** A processor knows only: read the cache, run its
  model at its thresholds, write its result + evidence. It never counts wagons,
  never reads another feature (with one documented exception below), and **never
  modifies GlobalTrainState**.

A processor **never modifies GlobalTrainState** because the state is the immutable
authority; if a feature could change the count or ids, immutability (and therefore
safe late attachment) would be lost, and two features could race to redefine the
train. Features **consume** the state and **produce** per-wagon findings — a
strict one-way dependency.

### 7.2 Common shape
Every processor:
1. Iterates the cached frames for its authoritative camera(s) and each wagon.
2. Runs the production model at the production thresholds.
3. Groups detections into **bands** (contiguous frame runs) or **votes**
   classifications, then decides the per-wagon result.
4. Persists **best-frame evidence** (the clearest frame showing the finding).
5. Writes `wagon_states/<feature>/<CAMERA>/GW_n.json` with a stable key surface.
6. Skips ENGINE/BRAKE_VAN where the feature is meaningless (load/wagon-OCR).

### 7.3 Door
- **Inputs / authority.** Side cameras: RIGHT_UP → right door, LEFT_UP → left
  door.
- **Model.** `side_damage.pt` (one side model emits `door_open`, `door_close`,
  `damage`; the door processor consumes only the door classes).
- **Thresholds.** Detection confidence 0.85 (RIGHT_UP) / 0.88 (LEFT_UP); band gap
  tolerance 5 frames; a fixed 10-frame edge skip per wagon.
- **Decision.** `door_state = OPEN if any door_open band else CLOSED` (a strict
  two-state rule — production doors are never PARTIAL/DAMAGED here);
  `door_close_detected` records explicit closed bands.
- **Evidence.** The winning band's best frame, annotated (open=red, close=green).
- **Output keys.** `right_door`/`left_door` (+ confidence), `door_state`,
  `door_close_detected`, `tracks[]`, `evidence`.
- **Errors/retry/parallelism/markers.** Missing model → `NO_DATA`; per-wagon
  exception → `FAILED`; runs in parallel with other features; the completion
  marker keys on the door model + thresholds.
- **Performance.** Detect on every interior frame of both side cameras.

### 7.4 OCR (wagon number + loco number)
- **Inputs / authority.** RIGHT_UP only (the OCR authority).
- **Models.** Wagon: `wagon_number.pt` (plate bbox) + easyocr. Loco:
  `right_up_gap.pt`'s `locono` class (loco-region) + easyocr.
- **Thresholds.** Wagon detector 0.40; loco detector 0.60; wagon = 11 digits,
  loco = 5 digits.
- **Decision.** WAGON wagons → detect plate, crop, preprocess (upscale, denoise,
  CLAHE, sharpen), OCR digits, apply the Indian-Railways prefix-manipulation
  correction (first two digits must fall in the valid range; corrected digit
  recorded as manipulated), validate 11 digits, and vote across frames.
  ENGINE wagons → detect `locono`, OCR 5-digit, vote. BRAKE_VAN → skipped.
- **Evidence.** Best plate/loco crop + annotated frame.
- **Output keys.** `wagon_identifier` (+ conf, candidates), `loco_number` (+
  conf), `is_valid_5_digit`.
- **Performance.** The slowest feature (per-frame detect + multi-run easyocr).

### 7.5 Load
- **Inputs / authority.** Top cameras: RIGHT_UP_TOP primary, LEFT_UP_TOP fallback.
- **Model.** Top classification (`top_classification.pt` / `ltop.pt`) emitting
  `wagon_loaded` / `wagon_empty`. Production has **no dedicated load model** —
  load is a classification, so it is derived from the top classifier.
- **Thresholds.** Only predictions with confidence ≥ 0.80 vote; frames sampled
  every-other-frame with a 5-frame edge skip; **majority** of loaded vs empty
  wins.
- **Decision.** `load_status = LOADED if loaded votes > empty votes else EMPTY`
  (else `NO_DATA`). ENGINE/BRAKE_VAN skipped.
- **Output keys.** `load_status` (+ conf, counts, ratio).

### 7.6 Damage (top 4-class + side)
- **Inputs / authority.** Top: RIGHT_UP_TOP + LEFT_UP_TOP (either confirming wins).
  Side: RIGHT_UP + LEFT_UP.
- **Models.** Top: `right_top_damage.pt` / `top_left_damage.pt` (4 classes:
  `body_dmg`, `body_dmg_probable`, `floor_dmg`, `floor_dmg_probable`). Side:
  `side_damage.pt`'s `damage` class.
- **Thresholds.** Top confidence 0.70; side 0.85/0.88; band gap tolerance 5;
  edge skip 10. No loaded-wagon floor filter (not production behaviour).
- **Decision.** Top: `damage_status = DAMAGE if any confirmed body/floor band`
  (probable recorded separately). Side: `damage_status = DAMAGE if any damage
  band`.
- **Output keys.** `damage_status`, `top_damage_details[]` (top) /
  `side_damage_details[]` (side), plus the per-class booleans.
- **The one cross-feature read (documented).** Historically the damage processor
  read the sibling load result to suppress floor damage on loaded wagons; that
  filter is **omitted** here because it is not production behaviour. No feature
  reads another in the current design.

### 7.7 Why processors never modify GlobalTrainState (restated)
See §7.1: the state is the immutable authority and the join key space. Features
form a strict one-way dependency on it. This is the invariant that makes late
attachment, restart, and parallelism all correct.

---

## 8. Fusion

### 8.1 Purpose
Collapse the four cameras' per-feature results for a wagon into one
**UnifiedWagonState**, using fixed authority so there is never ambiguity about
which camera "wins" a field.

### 8.2 Authority rules (and why these)
| Field | Authority | Why |
|---|---|---|
| `classification` | GlobalTrainState | structural; fixed at seal |
| `wagon_identifier`, `loco_number` | RIGHT_UP OCR | only the master side sees the stencilled number |
| `right_door` | RIGHT_UP | the right camera sees the right doors |
| `left_door` | LEFT_UP | symmetric |
| `load_status` | RIGHT_UP_TOP, else LEFT_UP_TOP | top view; primary/fallback for redundancy |
| `top_damage` | any TOP camera reporting DAMAGE | damage anywhere on top is damage; two views reduce misses |
| `side_damage` | either SIDE camera reporting DAMAGE | symmetric to top |

### 8.3 Conflict resolution, priority, confidence
- **Doors/OCR/load** are single-authority: no conflict possible; the authoritative
  camera's value is used (load falls back to the second top camera only when the
  primary produced no data).
- **Damage (top/side)** is "any-camera-wins": if either authoritative camera
  confirms DAMAGE, the wagon is damaged (evidence from both is merged). This
  biases toward **not missing** a safety finding.
- **Confidence** is a combined score over the populated per-field confidences (a
  mean of the non-zero confidences), used for reporting sort order, not for
  overriding authority.

### 8.4 Missing features, NO_DATA, pending
- A feature JSON that is `FAILED` / `NO_FRAMES` / `NO_DATA` → that field is
  `NO_DATA` — never a false OK.
- A camera that has **not yet arrived** → its owned fields are `PENDING_CAMERA`;
  at the final deadline a still-missing camera → `CAMERA_MISSING_FINAL`. These are
  distinct from "arrived but produced no data."
- A feature the operator **disabled** → its owned fields carry the
  `DISABLED_BY_USER` sentinel and never raise an anomaly.

### 8.5 Anomalies
`LEFT_DOOR_OPEN`, `RIGHT_DOOR_OPEN`, `TOP_DAMAGE`, `SIDE_DAMAGE`, and
`OCR_MISSING` (a WAGON-class wagon with no valid number, only once OCR has
actually run). Anomalies drive report row highlighting and the email summary.

### 8.6 Example
```json
{
  "global_id": "GW_7", "wagon_index": 7, "classification": "WAGON",
  "wagon_identifier": "31234567890", "wagon_identifier_confidence": 0.83,
  "loco_number": "NO_DATA",
  "left_door": "CLOSED", "right_door": "OPEN", "right_door_confidence": 0.91,
  "load_status": "LOADED", "load_confidence": 0.86,
  "top_damage": "OK", "side_damage": "DAMAGE",
  "field_sources": {"right_door": "RIGHT_UP", "load_status": "RIGHT_UP_TOP", "side_damage": "LEFT_UP"},
  "field_status": {"left_door": "OK", "top_damage": "OK"},
  "anomalies": ["RIGHT_DOOR_OPEN", "SIDE_DAMAGE"],
  "result_state": "COMPLETE_WITH_ANOMALY"
}
```

---

## 9. Reporting

### 9.1 Camera reports vs combined report
- **Camera reports** (one per camera) present **only what that camera is
  authoritative for**, using that camera's snapshots — so an inspector can drill
  into a single viewpoint.
- **Combined report** aggregates the four into the unified train view: a KPI
  summary (totals, loaded/empty, doors open, damage, **loco numbers**, rake
  type/status), a per-wagon table with `TOP_DMG` and `SIDE_DMG` columns and
  issue-row highlighting, links to the four camera reports and the processed
  videos, and a "Damaged Wagon" evidence grid.

### 9.2 inspection_data and the dashboard payload
The machine record is `combined_train_report.json` (a schema-versioned superset of
the fused train state, including a `legacy_view_model` the PDF renderer and
external consumers can read verbatim). The **dashboard payload** is a per-camera
`{camera_id, version, inspection_data}` document re-derived from the finalized
report + evidence for the railway's ingest API (§10, and the field-by-field
parity in `WAGONEYE_V5_SCHEMA_PARITY.md`).

### 9.3 PDF generation & evidence layout
PDFs are rendered from the view model with the product's visual identity. Evidence
images are the best-frame snapshots persisted by the feature processors under
`evidence/GW_n/<feature>/<CAMERA>/` (co-located per wagon so the combined report
pulls all four cameras' proof for one wagon from one directory).

### 9.4 Why reporting never re-runs inference
Reporting is **pure presentation**: it reshapes and renders already-computed state
and reads already-saved evidence images. Re-running a model in reporting would
(a) risk producing a *different* result than the one fused and delivered, breaking
determinism and auditability, and (b) duplicate the most expensive work. The
architectural invariant is that inference happens **only** in Stage 1
(reconstruction) and Stage 3 (features); every later stage is a deterministic
function of persisted artifacts.

---

## 10. Delivery

### 10.1 The four deliverables
- **Dashboard.** Per-camera `inspection_data` payloads POSTed to
  `cctv-receiver/inspections/ingest`. Read-only w.r.t. the pipeline (it only reads
  finalized artifacts), idempotent, and failure-isolated.
- **S3 upload.** The combined PDF (via a report microservice, falling back to a
  direct S3 PUT) + the combined JSON + the batch tree.
- **Email.** Exactly one summary email per train to the production recipient list;
  the subject carries the wagon count and loco numbers.
- **Archive.** The batch tree persisted to S3; the batch marked terminal.

### 10.2 Exactly-once semantics & idempotency
Delivery is guarded by `delivery/finalization.json`, which records what has
already been uploaded, emailed, and ingested — keyed by the final report hash +
report revision + an idempotency key. On a restart, delivery reads this marker and
skips anything already done. The email idempotency key is sent both as a payload
field and an `Idempotency-Key` header so a compliant email service de-duplicates a
resend. Exactly-once is best-effort across the narrow window between an API
returning 200 and the marker being persisted (a crash there could cause one
resend); everywhere else it is guaranteed.

### 10.3 Partial failures & recovery
Each deliverable is independent and non-fatal to the batch outcome: a failed
dashboard ingest, upload, or email is logged, recorded in the marker, and does not
prevent the others or block the batch from reaching a terminal state. On the next
poll or restart, only the not-yet-succeeded deliverables are retried (the marker
prevents re-doing the successful ones).

---

## 11. Configuration

### 11.1 Principles
- **No absolute paths in code.** The project root is auto-detected from the source
  tree, so the system runs wherever it is cloned.
- **Every path and knob is an environment variable** with a default that
  reproduces production behaviour; an unset environment behaves like production.
- **Configuration over hardcoding**, but **thresholds that define behaviour are
  fixed in the feature processors** (they are production truth, not per-deployment
  tunables) — see §16.

### 11.2 Environment variables (by group)
| Group | Variables | Meaning |
|---|---|---|
| Paths | `WAGONEYE_WORKSPACE_ROOT`, `WAGONEYE_MODELS_DIR`, `WAGONEYE_RECON_MODELS_DIR`, `WAGONEYE_PROD_MODELS_DIR`, `WAGONEYE_FEAT_MODELS_DIR`, `WAGONEYE_EXTRACTION_MODELS_DIR`, `WAGONEYE_LOCAL_INPUTS_DIR`, `WAGONEYE_LOG_DIR` | filesystem locations (defaults under the repo root) |
| Device | `WAGONEYE_DEVICE` | `cuda` / `cpu` / auto |
| S3 | `WAGONEYE_S3_REGION`, `WAGONEYE_S3_OUTPUT_BUCKET`, `WAGONEYE_S3_INPUT_BUCKET`, `WAGONEYE_S3_INPUT_PREFIXES`, `WAGONEYE_S3_TRAIN_BATCH_PREFIX`, `WAGONEYE_S3_STATE_KEY` | buckets/prefixes for discovery, output, state |
| Delivery | `WAGONEYE_UPLOAD_API_URL`, `WAGONEYE_EMAIL_API_URL`, `WAGONEYE_EMAIL_RECEIVER`, `WAGONEYE_EMAIL_RECEIVER_CC`, `WAGONEYE_INSPECTION_VERSION`, `WAGONEYE_DASHBOARD_INGEST_ENABLED` | report/email/dashboard endpoints + recipients + payload version |
| Lifecycle | `WAGONEYE_MASTER_WAIT_MINUTES`, `WAGONEYE_SUPPORT_FUSION_WAIT_MINUTES`, `WAGONEYE_FINAL_CAMERA_WAIT_MINUTES`, `WAGONEYE_ACTIVE_BATCH_POLL_INTERVAL`, `WAGONEYE_LATE_CAMERA_POLICY` | the three deadlines + poll cadence + late-camera policy |
| Logging | `WAGONEYE_LOG_LEVEL` | root log level |

### 11.3 Directory layout (runtime)
```
<repo>/
├── models/{reconstruction,production,extraction,features}/   *.pt (staged, not committed)
├── local_inputs/                                             4 trimmed clips (local mode)
├── logs/{wagon_eye.log, extraction_state/}
└── batch_outputs/<batch_key>/
    ├── downloads/ global_state/ wagon_cache/ wagon_states/{<feature>/<CAMERA>/, unified/, .features/}
    ├── evidence/<GW>/<feature>/<CAMERA>/ processed_videos/ reports/ delivery/ archive/ manifest.json
```

### 11.4 Model locations, bucket names, camera naming
- Models: `models/reconstruction` (Stage 1 + loco region), `models/production`
  (Stage 3), `models/extraction` (Stage A); staged from `s3://wagon-eye-models`.
- Buckets: raw `biro-wagon-raw-video-copy`, trimmed
  `complete-train`, output/archive
  `end-results`.
- Cameras: the four canonical ids (`RIGHT_UP` master, `LEFT_UP`, `RIGHT_UP_TOP`,
  `LEFT_UP_TOP`); lowercase cache folder names; a filename must contain the camera
  substring + a `YYYYMMDD_HHMMSS` stamp.

### 11.5 Feature switches & thresholds
Features are a registry: each has a key, display name, and the UnifiedWagonState
fields it owns. A feature can be disabled per run (CLI/interactive); disabled
fields read `DISABLED_BY_USER` and never raise an anomaly. Thresholds are fixed in
the processors (production values).

---

## 12. Deployment architecture

### 12.1 EC2 + systemd
Two systemd services run the two long-lived processes: an **extraction** unit
(producer) and an **inspection** unit (`master_runner --auto`). systemd gives
reboot-safe auto-restart; both trap `SIGTERM`/`SIGINT` to finish the current unit
of work before exiting (graceful `systemctl stop`). The host auto-detects its
project root, so no SageMaker/Jupyter runtime is needed.

### 12.2 IAM & S3
The instance uses an **IAM instance role** (no static keys): `GetObject/ListBucket`
on the raw + models buckets, `PutObject/GetObject/ListBucket` on the trimmed +
output buckets. All artifact durability and cross-service communication is S3.

### 12.3 Model synchronization & bootstrap
A one-shot setup script installs OS libs (ffmpeg, OpenCV/reportlab runtime),
creates a venv, installs dependencies (GPU torch if a GPU is detected), and
creates the runtime directory skeleton. Models are then copied from
`s3://wagon-eye-models` into `models/{reconstruction,production,extraction}`. A
missing model does not crash startup — the owning feature emits `NO_DATA`.

### 12.4 Health checks & monitoring
Liveness = the service process is up (systemd) and the log shows recent poll
lines. Progress = new terminal entries in `processed_batches.json` and new
`archive/<key>/` trees in S3. Per-stage `STAGE …` and `[FEAT/…] done in …`
log lines give timing; batch outcomes give success/failure. Alerting is on
service-down, on batches stuck non-terminal past the final deadline, and on
repeated delivery failures.

### 12.5 Scaling
Horizontal by station: one inspection instance per station polls its own input
prefixes; there is no cross-train shared state to coordinate. Within an instance,
one train is processed to completion at a time (bounded memory), with the four
features parallel. Vertical: a GPU cuts per-train time by an order of magnitude.

### 12.6 GPU vs CPU
The device is resolved once per process (CUDA if present, else CPU; overridable).
FP16 is used only on CUDA (a CPU footgun otherwise). CPU is fully supported but
an order of magnitude slower for the inference stages; GPU is recommended for
production throughput.

---

## 13. Performance engineering

The performance model follows from one principle: **do each expensive thing
exactly once, and never redo work whose inputs are unchanged.**

- **Decode once.** Materialization (§6) is the only analytical video decode;
  everything else reads JPEGs. This removes the predecessor's 3–4× redundant
  decode per camera.
- **Infer once.** Reconstruction (Stage 1) and features (Stage 3) are the only
  model calls; reporting and rendering never re-infer. A given `(camera, feature)`
  runs once and is skipped thereafter via its completion marker.
- **Parallel features.** The four features run concurrently; within a feature,
  wagons run serially so each model/OCR reader loads once per process and is
  reused.
- **Memory strategy.** One train at a time; per-wagon frames streamed from disk;
  models loaded once and cached; evidence written incrementally. Multiple
  partially-arrived trains are cheap on-disk manifests, not in-memory jobs.
- **Threading.** OpenCV decode releases the GIL, so per-camera materialization
  threads scale; per-process compute-thread bounding avoids oversubscription when
  multiple model processes share a CPU box.
- **CPU bottlenecks.** Gap detection (every frame ×4 videos) and OCR (per-frame
  detect + multi-run easyocr) dominate on CPU.
- **GPU bottlenecks.** Model load/VRAM; batching frames helps GPU utilisation
  (marginal on CPU).
- **Caching.** wagon_cache (frames), model cache (weights), completion markers
  (skip logic) — three caches that together make restarts and late attaches
  near-free.
- **Avoiding duplicate work / exactly-once processing.** Markers keyed on ETag +
  GST version + model SHA + schema + thresholds make every stage skippable when
  its inputs are unchanged; the finalization marker makes delivery exactly-once.

---

## 14. Failure recovery

For each scenario: what happens and how recovery occurs.

| Scenario | Behaviour & recovery |
|---|---|
| **Missing camera** | Batch proceeds with present cameras; missing fields → `PENDING_CAMERA` then `CAMERA_MISSING_FINAL` at the deadline; report is `COMPLETED_PARTIAL`. A late arrival before the deadline is attached (materialize+features+re-fuse) with no reseal. |
| **Corrupt / unreadable video** | That camera's wagon_cache subtree is empty; its features → `NO_DATA`; other cameras unaffected; batch is partial. |
| **Missing model** | The owning feature emits `NO_DATA` with a clear "model not found" error; the pipeline still seals, fuses, and reports. Stage a real model and the marker (which keyed on the model hash) re-runs that feature. |
| **Failed inference (one wagon)** | That `(feature, GW_n)` JSON → `FAILED`; fusion treats it as `NO_DATA`; rest unaffected. |
| **EC2 reboot** | systemd restarts both services; each resumes from the manifest + on-disk markers, repeating no completed reconstruction/materialization/feature/upload/email. |
| **S3 outage** | Reads/writes fail and are retried on the next poll; nothing is lost (artifacts are re-read when S3 returns); no state is corrupted because writes are atomic (temp + rename/replace). |
| **Network failure (delivery)** | Ingest/upload/email fail and are logged; the finalization marker records what did succeed; the not-yet-done deliverables retry later. |
| **Partial upload** | The upload path uses a microservice with a direct-PUT fallback; a failed PDF upload leaves the batch marked accordingly and retries; a truncated write is avoided by atomic object creation. |
| **Dashboard offline** | Ingest POST fails, is logged, non-fatal; retried; idempotent (sha + revision), so a later success does not duplicate. |
| **Email failure** | Logged; batch outcome still persisted; retried; idempotency key prevents a duplicate on resend. |
| **Worker crash mid-stage** | The last-completed stage's markers/artifacts survive; the batch resumes at the next incomplete stage. A crash between an API 200 and its marker write is the only exactly-once gap (one possible resend). |
| **Power failure** | Equivalent to a reboot: manifest + markers on durable disk/S3 drive resumption; no train is lost or double-processed. |

The unifying mechanism: **atomic writes + durable markers keyed on content
identity**. Every stage is safe to re-enter because it either finds its marker and
skips, or rebuilds into a temp location and atomically swaps.

---

## 15. Repository structure

| Directory | Owns | Allowed dependencies | Prohibited |
|---|---|---|---|
| `train_extraction/` | Stage A producer (raw→trimmed) | its own vendored extractor + S3 | importing the inspection package |
| `orchestrator/` | batch acquisition + lifecycle + markers (Stage 0) | core, reconstruction, materializer, features, fusion, reporting, delivery | analytical logic (no counting/inference here) |
| `reconstruction/` | Stage 1 driver | `wagon_count/` subprocess, core | doing anything but launching + parsing reconstruction |
| `wagon_count/` | the counting brain (gaps, fusion, classification) | self-contained | importing the inspection package (stays standalone) |
| `materializer/` | Stage 2 (decode once → wagon_cache) | core | inference / model loading |
| `features/` | Stage 3 processors + shared helpers + inference_lib | core, production_models, cache | modifying GlobalTrainState; reading another feature (one documented exception) |
| `fusion/` | Stage 4 (UnifiedWagonState) | core, feature JSON | inference; changing the count |
| `rendering/` | Stage 4b overlays | core, state, evidence, tracking JSON | invoking any detector/model |
| `reporting/` | Stage 5 (PDF/JSON) | core, state, evidence, cache | inference |
| `delivery/` | Stage 6 (dashboard/upload/email/archive/finalization) | core, finalized artifacts | mutating pipeline state |
| `core/` | config, constants, GlobalTrainState loader, UnifiedWagonState, production_models, camera/feature registries, logging | nothing analytical | loading models, reading frames |
| `models/` | the `.pt` weights (reconstruction/production/extraction/features) | — | code |
| `deploy/`, `scripts/` | systemd units + EC2 setup | — | — |
| `docs/` | this spec + operational guides | — | — |

**Dependency direction is strictly one-way:** `core` depends on nothing; every
stage depends on `core` and on the artifacts of earlier stages, never on later
stages. `wagon_count` and `train_extraction` are deliberately standalone (they can
run alone), which is why they do not import the inspection package.

---

## 16. Development philosophy

- **Single responsibility.** Each module does one thing (extract, count,
  materialize, one feature, fuse, report, deliver). This is what makes fault
  isolation, parallelism, and replaceability possible.
- **Immutable state.** GlobalTrainState is frozen at seal; UnifiedWagonState is a
  pure function of feature outputs. Immutability is what makes asynchronous
  camera arrival safe.
- **No duplicated business logic.** Counting exists in exactly one place
  (reconstruction); authority rules in exactly one place (fusion); the S3 naming
  contract in exactly one place (delivery/artifacts). Duplication is how the
  predecessor drifted into inconsistency.
- **No hidden global state.** All cross-stage communication is explicit files with
  documented schemas; there is no shared mutable singleton driving behaviour.
- **Production-first behaviour.** The models, thresholds, and outputs reproduce
  the deployed production system; "improvements" are a separate, later phase, never
  mixed with the architectural migration.
- **Deterministic outputs.** Given the same inputs (clips + models + thresholds),
  the pipeline produces the same artifacts. This is required for auditability,
  for the completion-marker skip logic, and for behavioural parity testing.
- **Configuration over hardcoding** for *deployment* concerns (paths, buckets,
  endpoints, deadlines); **fixed constants** for *behavioural* concerns
  (thresholds), because those are production truth, not per-deployment knobs.
- **Testability.** Pure helpers (band grouping, canonicalization, frame selection,
  authority rules) are unit-testable without models; every stage has a clear
  file contract that can be validated in isolation; graceful `NO_DATA` lets the
  whole pipeline run end-to-end without models present.

---

## 17. Building a new implementation (the immutable contracts)

A future engineer may rebuild any component in any language, provided the
**external contracts** below are preserved. As long as these hold, the rest of the
system continues to work unchanged, and behavioural parity with production is
maintained.

### 17.1 The contracts that must NEVER change
1. **Camera identity & authority.** Four cameras `RIGHT_UP` (master), `LEFT_UP`,
   `RIGHT_UP_TOP`, `LEFT_UP_TOP`; the authority map in §8.2.
2. **S3 naming.** Raw `raw-bucket/<camera_folder>/…`; trimmed
   `trimmed-bucket/<camera_folder>/<raw>_train.mp4` (timestamp in name); output
   `output-bucket/archive/<key>/…`; state `processed_batches.json`.
3. **GlobalTrainState JSON schema** (§5.3): `total_wagons`, `wagons[]` with
   `global_id`/`wagon_index`/`start_time`/`end_time`/`classification` + master
   clock fields + correction provenance. `GW_n` is the immutable join key.
4. **wagon_cache layout** (§6.3): `wagon_cache/GW_n/<camera>/frame_NNNNNN.jpg`,
   frames windowed by the master-clock boundaries, quality that preserves
   detections.
5. **Per-feature output schema** (§7): `wagon_states/<feature>/<CAMERA>/GW_n.json`
   with the documented keys (`door_state`/`right_door`/`left_door`;
   `wagon_identifier`/`loco_number`; `load_status`; `damage_status`/
   `top_damage_details`/`side_damage_details`), plus `status ∈ {OK, NO_FRAMES,
   FAILED, NO_DATA}`.
6. **UnifiedWagonState schema** (§8.6) and the anomaly vocabulary.
7. **Evidence layout** `evidence/GW_n/<feature>/<CAMERA>/…` (co-located per wagon).
8. **Dashboard payload** `{camera_id, version, inspection_data}` with the fields
   in `WAGONEYE_V5_SCHEMA_PARITY.md`.
9. **Lifecycle terminal states** and the exactly-once delivery guarantee.

### 17.2 What may be freely replaced (behind the contracts)
- The **models** (any detector/classifier/OCR), as long as the per-feature output
  values match production semantics on the same footage.
- The **reconstruction engine** (any counting/tracking algorithm), as long as it
  emits the GlobalTrainState schema and preserves the master-authority + ≥2-support
  gap-recovery semantics.
- The **implementation language / framework** of any stage.
- **Reporting rendering** (any PDF/JSON toolkit), as long as it consumes the
  fused state and evidence and emits the report artifacts + the dashboard payload.
- The **materializer, feature drivers, fusion, delivery** internals — any
  implementation that honours the file contracts and the exactly-once markers.

### 17.3 Outputs that must remain identical (parity targets)
- Per-wagon **findings**: door state per side, wagon number (with the
  Indian-Railways prefix correction), loco number, load state, top/side damage
  (per class), evidence frame selection.
- The **train count and classification** for a given rake (allowing that the
  GlobalTrainState count legitimately differs from the predecessor's per-camera
  counts — that is the intended improvement, not a regression).
- The **dashboard inspection_data** fields consumed by the railway, and the
  **email** semantics (one per train, subject content).

### 17.4 How to verify behavioural parity
1. **Per-feature, identical-frame parity (strongest).** Run the reference and the
   new implementation on the **same** cached frames and compare per wagon:
   door_state, wagon/loco number, load, damage bands + best frame. With the same
   model + thresholds + banding this should be exact; this isolates a feature from
   the count.
2. **End-to-end, position-aligned parity.** Run both full pipelines on the same
   trimmed clips; align wagons by **rake position** (the count/ids differ by
   construction); compare findings on matched wagons; label every mismatch as a
   feature difference or a boundary/count difference.
3. **Schema validation.** Validate every emitted artifact against the schemas in
   §17.1; a consumer (dashboard) must accept the payload unchanged.
4. **Deterministic replay.** Re-run the same inputs and confirm byte-stable
   conclusion-level outputs (paths/timestamps aside).
5. **Lifecycle & exactly-once tests.** Kill/restart mid-stage; confirm no repeated
   work, no double upload/email, correct partial-report behaviour, and correct
   late-camera attachment without reseal/renumber.

Meeting §17.1 + passing §17.4 means a completely new implementation is a faithful
WagonEye: same external behaviour, same contracts, same guarantees — which is the
definition of "production-ready" for this system.

---

*End of specification.*
