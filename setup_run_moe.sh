#!/usr/bin/env bash
# Setup a uv venv for Prims-TS MoE (CUDA 13 / SM100+SM103).
#
# Usage:
#   ./setup_run_moe.sh          # create .venv and install flashinfer
#   source .venv/bin/activate
#   ./run_moe.sh                # BF16/NVFP4 bench (see run_moe.sh)
#
# Requires: CUDA 13 toolkit on PATH, NVIDIA GPU, network for wheels/cubins.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
# Skip NIXL/NCCL-EP native builds; not needed for prims_ts MoE.
export BUILD_NVEP="${BUILD_NVEP:-0}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv into ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating uv venv at ${VENV_DIR} (Python ${PYTHON_VERSION}) ..."
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

# Ensure submodules (cutlass, spdlog, cccl, nixl) exist for the editable install.
if [[ ! -f 3rdparty/cutlass/include/cutlass/cutlass.h ]]; then
  echo "Initializing git submodules ..."
  git submodule update --init --recursive
fi

echo "Installing torch (${UV_TORCH_BACKEND}) ..."
uv pip install --python "${VENV_DIR}/bin/python" torch --torch-backend="${UV_TORCH_BACKEND}"

echo "Installing flashinfer editable with .[cu13] (BUILD_NVEP=${BUILD_NVEP}) ..."
# Dynamic deps in pyproject.toml are not lockable; use uv pip, not uv sync.
# --no-build-isolation: build_backend needs ambient torch/ninja/CUDA.
uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation -e ".[cu13]"

echo
echo "Done. Activate and run MoE:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  ./run_moe.sh"
echo
"${VENV_DIR}/bin/python" - <<'PY'
import torch
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.utils import get_compute_capability

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)} cc={get_compute_capability(torch.device('cuda'))}")
print(f"prims_ts_available={is_prims_ts_available()}")
PY
