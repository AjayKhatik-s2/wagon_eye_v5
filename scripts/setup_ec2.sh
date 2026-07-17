#!/usr/bin/env bash
#
# setup_ec2.sh -- prepare a fresh EC2 (Ubuntu or Amazon Linux) host to run the
# WagonEye v4 pipeline.  Idempotent: safe to re-run.
#
# What it does:
#   1. Installs OS packages (ffmpeg, OpenCV/reportlab runtime libs, fonts,
#      python venv + build tools).
#   2. Creates a Python virtualenv at <repo>/.venv.
#   3. Installs requirements.txt (CPU torch by default).
#   4. If an NVIDIA GPU is detected, reinstalls the CUDA build of torch.
#   5. Creates the runtime directory skeleton (models/, logs/, etc.).
#
# Usage:
#   bash scripts/setup_ec2.sh                 # auto-detect GPU
#   WAGONEYE_FORCE_CPU=1 bash scripts/setup_ec2.sh   # force CPU torch
#   PYTHON_BIN=python3.11 bash scripts/setup_ec2.sh  # pick interpreter
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve the repo root from this script's location (scripts/ is a child).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${REPO_ROOT}/.venv"
CUDA_INDEX_URL="https://download.pytorch.org/whl/cu121"

log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[setup:warn]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup:error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Distro detection + OS package install
# ---------------------------------------------------------------------------
install_os_packages() {
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
  else
    warn "cannot read /etc/os-release; skipping OS package install"
    return
  fi

  log "Detected distro: ${PRETTY_NAME:-$ID}"
  case "${ID}" in
    ubuntu|debian)
      export DEBIAN_FRONTEND=noninteractive
      sudo apt-get update -y
      sudo apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        fontconfig \
        fonts-dejavu-core \
        python3-venv \
        python3-pip \
        build-essential \
        ca-certificates
      ;;
    amzn|rhel|centos|fedora)
      # Amazon Linux 2023 uses dnf; AL2 uses yum -- both accept `yum`.
      sudo yum install -y \
        ffmpeg \
        mesa-libGL \
        glib2 \
        fontconfig \
        dejavu-sans-fonts \
        python3 \
        python3-pip \
        gcc gcc-c++ make \
        ca-certificates || \
      warn "some yum packages may be unavailable on this AMI (ffmpeg often \
needs the RPM Fusion / EPEL repo). Install ffmpeg manually if Stage-1/4b video \
rendering fails."
      ;;
    *)
      warn "unrecognized distro '${ID}'. Install these manually: ffmpeg, \
libGL, glib2, fontconfig + DejaVu fonts, python3-venv, pip, a C compiler."
      ;;
  esac
}

# ---------------------------------------------------------------------------
# 2 + 3. venv + Python deps
# ---------------------------------------------------------------------------
setup_venv() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} not found"
  log "Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
  log "Installing ${REPO_ROOT}/requirements.txt"
  python -m pip install -r "${REPO_ROOT}/requirements.txt"
}

# ---------------------------------------------------------------------------
# 4. GPU-aware torch
# ---------------------------------------------------------------------------
setup_torch() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  if [ "${WAGONEYE_FORCE_CPU:-0}" = "1" ]; then
    log "WAGONEYE_FORCE_CPU=1 -- keeping the CPU torch wheel from requirements.txt"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA GPU detected -- installing CUDA build of torch from ${CUDA_INDEX_URL}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    python -m pip install --upgrade --index-url "${CUDA_INDEX_URL}" \
      "torch>=2.0.0" "torchvision>=0.15.0" || \
      warn "CUDA torch install failed. The pipeline will still run on CPU. \
To retry: pip install --index-url ${CUDA_INDEX_URL} torch torchvision"
  else
    log "No NVIDIA GPU detected -- keeping the CPU torch wheel (pipeline will \
run on CPU; set WAGONEYE_DEVICE=cpu to make this explicit)."
    warn "If this box DOES have a GPU, install the CUDA driver + \
'pip install --index-url ${CUDA_INDEX_URL} torch torchvision' then re-run."
  fi
}

# ---------------------------------------------------------------------------
# 5. Runtime directory skeleton
# ---------------------------------------------------------------------------
make_dirs() {
  log "Creating runtime directory skeleton under ${REPO_ROOT}"
  mkdir -p \
    "${REPO_ROOT}/models/reconstruction" \
    "${REPO_ROOT}/models/features" \
    "${REPO_ROOT}/logs" \
    "${REPO_ROOT}/batch_outputs" \
    "${REPO_ROOT}/local_inputs"
}

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
verify() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  log "Verifying imports + resolved device"
  python - <<'PY'
import importlib
for m in ("cv2", "numpy", "torch", "ultralytics", "easyocr", "reportlab", "boto3"):
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
from core import config as CFG
print("  device:", CFG.resolve_device())
print("  project root:", CFG.PROJECT_ROOT)
PY
}

main() {
  log "WagonEye v4 EC2 setup -- repo root: ${REPO_ROOT}"
  install_os_packages
  setup_venv
  setup_torch
  make_dirs
  verify
  cat <<EOF

$(printf '\033[1;32m[setup] Done.\033[0m')

Next steps:
  1. Drop the 8 model .pt files into:
       ${REPO_ROOT}/models/reconstruction/   (4 files)
       ${REPO_ROOT}/models/features/         (4 files)
  2. Configure environment (optional overrides):
       cp deploy/wagon-eye.env.example deploy/wagon-eye.env  &&  edit it
  3. Smoke test (no S3):
       source ${VENV_DIR}/bin/activate
       python -m orchestrator.master_runner --local-only \\
           --local-inputs ./local_inputs --no-interactive
  4. Install the service (continuous S3 polling):
       see deploy/wagon-eye.service and DEPLOYMENT.md

EOF
}

main "$@"
