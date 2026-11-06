#!/usr/bin/env bash
# Install a dedicated public Framework environment; run on a compute worker.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 UV_COMPILE_BYTECODE=0
workspace=${1:?Usage: bash tools/dreams/bootstrap.sh WORKSPACE [cu130-train|cu128-train]}
cuda_group=${2:-cu130-train}
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$workspace"
workspace=$(cd "$workspace" && pwd)
export UV_CACHE_DIR="$workspace/cache/uv" UV_PYTHON_INSTALL_DIR="$workspace/cache/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy OMP_NUM_THREADS=1
export HF_HOME="$workspace/cache/huggingface" TORCH_HOME="$workspace/cache/torch"
unset TRANSFORMERS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1
export UV_TOOL_DIR="$workspace/tools/uv-envs" UV_TOOL_BIN_DIR="$workspace/bin"
# The framework lockfile requires this uv version; leave the host installation alone.
cd "$workspace"
if [[ ! -x "$workspace/bin/uv" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        printf '%s\n' 'uv is missing in this runtime. Install uv 0.11.29 inside the worker/container using the tutorial environment recipe, then rerun bootstrap.' >&2
        exit 2
    fi
    uv tool install --python 3.13 uv==0.11.29
fi
uv_cmd="$workspace/bin/uv"
framework="$workspace/sources/cosmos-framework"
revision=1a2f03ab6111b457633827736f94f67d55bd5202
mkdir -p "$workspace/sources" "$workspace/reports"
if [[ ! -d "$framework/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/NVIDIA/cosmos-framework.git "$framework"
fi
if [[ $(git -C "$framework" rev-parse HEAD) != "$revision" ]]; then
    git -C "$framework" fetch --depth 1 origin "$revision"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$framework" checkout --detach "$revision"
fi
cd "$framework"
if [[ -x .venv/bin/python ]]; then
    .venv/bin/python -c 'import os, pathlib, sys; root=pathlib.Path(os.environ["UV_PYTHON_INSTALL_DIR"]).resolve(); base=pathlib.Path(sys.base_prefix).resolve(); assert base.is_relative_to(root), "Existing venv uses a host Python; choose a fresh workspace for managed Python"'
fi
# uv 0.11.29 rejects --managed-python together with UV_PYTHON_PREFERENCE.
# Keep the explicit managed-interpreter requirement for this command.
env -u UV_PYTHON_PREFERENCE "$uv_cmd" sync --managed-python --python 3.13 --all-extras --group "$cuda_group"
"$uv_cmd" pip install --python .venv/bin/python -e "$package_root"
"$uv_cmd" pip install --python .venv/bin/python pytest==8.4.2 lpips==0.1.4
.venv/bin/python -m cosmos3_h_embodied.dreams.patch_framework "$framework"
.venv/bin/python -c 'import sys, torch, cosmos_framework; print(sys.version); print("base_python:", sys.base_prefix); print(torch.__version__, torch.version.cuda); print(cosmos_framework.__file__)' > "$workspace/reports/environment.txt"
printf '%s\n' "Environment: $framework/.venv" "Report: $workspace/reports/environment.txt"
