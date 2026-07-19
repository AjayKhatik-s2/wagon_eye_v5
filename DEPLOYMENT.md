# WagonEye v4 — EC2 Deployment Guide

This guide takes a fresh Amazon EC2 Linux instance to a running, continuously
polling WagonEye v4 pipeline. The pipeline is **EC2-native**: it needs no
SageMaker notebook, no Jupyter runtime, and no hardcoded paths — it runs from
wherever you clone it.

> All pipeline logic (GlobalTrainState, wagon counting, feature inference,
> fusion, rendering, reports, delivery) is unchanged from the SageMaker
> version. Only the runtime/infrastructure layer changed.

---

## 0. Prerequisites

- An EC2 instance running **Ubuntu 22.04/24.04** or **Amazon Linux 2023**.
  - CPU-only works. For production throughput use a GPU instance
    (e.g. `g4dn.xlarge`) with the NVIDIA driver installed.
- An **IAM instance role** attached to the EC2 instance granting S3 access to
  the input and output buckets (recommended over static keys).
- The **8 model files** (`.pt`) — 4 reconstruction + 4 feature models.

---

## 1. Clone the repository

Clone anywhere — the project auto-detects its own root; no path is hardcoded.

```bash
sudo mkdir -p /opt && cd /opt
git clone <your-repo-url> wagon_eye_v4      # or copy the folder up via scp/rsync
cd wagon_eye_v4
```

(If you use a different location, adjust the paths in
`deploy/wagon-eye.service` accordingly — they're marked `<<EDIT>>`.)

---

## 2. Install dependencies (one command)

```bash
bash scripts/setup_ec2.sh
```

This script:
- installs OS packages: `ffmpeg`, OpenCV/reportlab runtime libs
  (`libgl1`/`mesa-libGL`, `glib2`, `fontconfig` + DejaVu fonts), `python3-venv`,
  `pip`, and a C compiler;
- creates a virtualenv at `.venv/`;
- installs `requirements.txt`;
- **auto-detects a GPU** (`nvidia-smi`) and, if present, reinstalls the CUDA
  build of torch from `https://download.pytorch.org/whl/cu121`;
- creates the runtime directory skeleton (`models/`, `logs/`, `batch_outputs/`,
  `local_inputs/`);
- runs an import + device sanity check.

Force CPU torch even on a GPU box: `WAGONEYE_FORCE_CPU=1 bash scripts/setup_ec2.sh`.
Pick a specific interpreter: `PYTHON_BIN=python3.11 bash scripts/setup_ec2.sh`.

---

## 3. Drop in the models

```
models/reconstruction/     right_up_gap.pt   left_up_gap.pt   top_gap.pt   side_classification.pt
                           (long names right_up_wagon_gap.pt / left_up_wagon_gap.pt also accepted)
models/features/           door_state.pt   loaded.pt   damage.pt   wagon_id_counting.pt
```

(Or point `WAGONEYE_MODELS_DIR` / `WAGONEYE_RECON_MODELS_DIR` /
`WAGONEYE_FEAT_MODELS_DIR` elsewhere — see step 4.)

---

## 4. Configure the environment

Every setting has a working default (the original production values). Override
only what differs on this host:

```bash
cp deploy/wagon-eye.env.example deploy/wagon-eye.env
nano deploy/wagon-eye.env
```

For **continuous `--auto` mode you must set the input prefixes** so the poller
knows where the source videos land:

```ini
WAGONEYE_S3_INPUT_BUCKET=end-results
WAGONEYE_S3_INPUT_PREFIXES=incoming/right_up/,incoming/left_up/,incoming/right_up_top/,incoming/left_up_top/
```

Other common overrides: `WAGONEYE_DEVICE=cpu|cuda`, `WAGONEYE_WORKSPACE_ROOT`,
`WAGONEYE_LOG_DIR`, `WAGONEYE_LOG_LEVEL`, `WAGONEYE_S3_OUTPUT_BUCKET`,
`WAGONEYE_EMAIL_RECEIVER`. Full list with defaults is in
`deploy/wagon-eye.env.example`.

---

## 5. Smoke test before going live (no S3)

Put 4 trimmed videos (filenames containing `right_up` / `left_up` /
`right_up_top` / `left_up_top`) into `local_inputs/`, then:

```bash
source .venv/bin/activate
set -a; source deploy/wagon-eye.env; set +a      # load your overrides
python -m orchestrator.master_runner --local-only \
       --local-inputs ./local_inputs --no-interactive
```

Success looks like: `[BATCH <key>] completed (…s)` and a
`batch_outputs/<key>/reports/combined_train_report.pdf` that now includes the
company logo. Check `logs/wagon_eye.log` for the timestamped per-stage trace.

---

## 6. Install as a service (continuous mode, restarts on reboot)

```bash
# Edit the three <<EDIT>> placeholders (User, WorkingDirectory, paths):
sudo cp deploy/wagon-eye.service /etc/systemd/system/wagon-eye.service
sudo nano /etc/systemd/system/wagon-eye.service

sudo systemctl daemon-reload
sudo systemctl enable --now wagon-eye        # start now + on every boot
```

`enable` makes it **auto-start after a reboot**. Verify:

```bash
systemctl is-enabled wagon-eye     # -> enabled
systemctl status wagon-eye         # -> active (running)
```

---

## 7. Monitor

```bash
# Application log (structured, timestamped, rotates at 50 MB × 10):
tail -f logs/wagon_eye.log

# Or via systemd's journal:
journalctl -u wagon-eye -f

# Belt-and-braces raw stdout/stderr captured by the unit:
tail -f logs/service.out logs/service.err
```

Every stage logs a start line, an elapsed-time completion line, and any
warnings/errors with full tracebacks. The full Stage-1 (`wagon_count`)
subprocess trace for each batch is saved at
`batch_outputs/<key>/global_state/stage1_wagon_count.log`.

---

## 8. Control the service

```bash
sudo systemctl stop wagon-eye       # SIGTERM: finishes the CURRENT batch, then exits
sudo systemctl restart wagon-eye
sudo systemctl start wagon-eye
```

Graceful stop waits for the in-flight batch. If your batches can run longer
than `TimeoutStopSec` (default 1800 s in the unit), raise it, or systemd will
SIGKILL the batch when the timer expires.

Manual (non-service) alternatives:

```bash
python -m orchestrator.master_runner --auto        # continuous, foreground
python -m orchestrator.master_runner --once         # one batch then exit
python -m orchestrator.master_runner --batch <key>  # replay a specific batch
```

---

## 9. Verify it's actually working

1. `systemctl status wagon-eye` → `active (running)`.
2. `grep 'logging initialized' logs/wagon_eye.log` → confirms log rotation is set up.
3. Drop a complete set of 4 videos into the input prefix(es); within one poll
   interval (`--poll-interval`, default 60 s) the log shows
   `[BATCH] discovered … batch(es)` → the 6 stage lines → `[BATCH <key>] completed`.
4. Check the output bucket for `reports/<key>/combined_train_report.pdf`
   and the archived tree; confirm the notification email arrived.
5. Confirm the device line at startup:
   `WagonEye v4 orchestrator starting (device=cuda)` on a GPU box.

---

## Reboot behavior & disk notes

- **After a reboot**, `systemd` restarts the service automatically (because of
  `enable`). Processed-batch state lives in S3
  (`s3://<bucket>/processed_batches.json`), so no batch is
  reprocessed after a restart.
- **Disk growth**: each batch leaves its full working tree under
  `batch_outputs/<key>/` (downloads, wagon_cache JPEGs, evidence, processed
  videos). Nothing prunes these automatically. Add a retention policy suited to
  your audit requirements, e.g. a cron job:
  ```bash
  # delete batch working dirs older than 14 days
  find /opt/wagon_eye_v4/batch_outputs -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
  ```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `--auto` idles, "WAGONEYE_S3_INPUT_PREFIXES is empty" | Set the input prefixes in `deploy/wagon-eye.env` (step 4). |
| `train_batch_manager not importable` | Should not happen — the module now ships in `orchestrator/`. Ensure the repo wasn't partially copied. |
| `libGL.so.1: cannot open shared object` | OS libs missing; re-run `scripts/setup_ec2.sh` (installs `libgl1`/`mesa-libGL`). |
| Reports render without fonts / boxes | Install `fontconfig` + DejaVu fonts (setup script does this). |
| Runs on CPU on a GPU box | Driver/CUDA torch not installed; `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`, or set `WAGONEYE_DEVICE=cuda`. |
| S3 `AccessDenied` | Attach an IAM instance role with read on the input bucket + write on the output bucket. |

---

## Incremental lifecycle (v4 async-camera) — deployment notes

The `--auto` service is now a **manifest-driven, multi-batch scheduler** that
handles cameras arriving at different times. No change to service names or paths:
the existing `wagon-eye.service` unit and `EnvironmentFile=` layout are unchanged.
An old env file that does not set the new variables uses safe defaults (all new
`WAGONEYE_*` keys are optional).

**New environment variables** (see `deploy/wagon-eye.env.example` for full comments):

| Variable | Default | Meaning |
|---|---|---|
| `WAGONEYE_MASTER_WAIT_MINUTES` | `10` | wait for RIGHT_UP before it's "late" |
| `WAGONEYE_SUPPORT_FUSION_WAIT_MINUTES` | `3` | support window (armed on RIGHT_UP arrival); must be ≤ final wait |
| `WAGONEYE_FINAL_CAMERA_WAIT_MINUTES` | `30` | hard close → `COMPLETED_PARTIAL` |
| `WAGONEYE_ENABLE_LEFT_UP_FALLBACK_MASTER` | `false` | experimental; keep off |
| `WAGONEYE_GENERATE_INTERIM_REPORTS` | `true` | regenerate reports on disk as cameras arrive |
| `WAGONEYE_UPLOAD_INTERIM_REPORTS` | `false` | interim reports are local-only unless set |
| `WAGONEYE_EMAIL_INTERIM_REPORTS` | `false` | one email at closure unless set |
| `WAGONEYE_LATE_CAMERA_POLICY` | `IGNORE` | terminal batches never reopened |
| `WAGONEYE_ACTIVE_BATCH_POLL_INTERVAL` | `60` | scheduler poll cadence (s) |
| `WAGONEYE_MANIFEST_S3_PREFIX` | *(empty)* | manifest S3 prefix override |

**Startup validation.** The orchestrator validates configuration before polling
and **fails fast** (exit 2) with clear `[CONFIG]` errors on: negative deadlines,
`support_wait > final_wait`, interim upload/email enabled without interim
generation, empty S3 input prefixes / output bucket in `--auto`, missing email
endpoint/recipients when email is enabled, or a non-writable workspace/log/temp
dir. It then logs a **redacted** effective-settings summary (recipients shown as
counts; no secrets).

**Backward compatibility.** Existing CLI modes (`--auto`, `--once`, `--batch`,
`--local-only`) parse and behave as before; `--local-only` still runs a single
complete-set batch offline. Legacy flat `wagon_states/<feature>/GW_n.json`
batches remain readable. New batches write only the camera-scoped layout.

> ⚠️ **Production gate.** The code is structurally production-ready, but a real
> deployment should still be preceded by one full four-camera run and one
> delayed-camera run using **real models, real videos, live S3, and the real
> email/upload services** — the automated tests use mocks/fixtures and do not
> exercise those live integrations or `.pt` inference quality.
