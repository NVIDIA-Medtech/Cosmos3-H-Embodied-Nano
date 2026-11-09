#!/usr/bin/env bash
# Run on the chosen workstation/compute worker with GPU_COUNT visible GPUs.
# Eight H100s are the qualified reference; other configurations need qualification.
# Setup/download/migration are CPU operations; actual training is torchrun.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
workspace=${1:?Usage: bash tools/dreams/run_smoke.sh WORKSPACE [GPU_COUNT]}
gpus=${2:-8}
setup_only=${DREAMS_SETUP_ONLY:-0}
if [[ "$setup_only" != 0 && "$setup_only" != 1 ]]; then
    printf '%s\n' 'DREAMS_SETUP_ONLY must be 0 or 1' >&2
    exit 2
fi
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$workspace"
workspace=$(cd "$workspace" && pwd)
framework="$workspace/sources/cosmos-framework"
python="$framework/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$workspace/cache/huggingface" TORCH_HOME="$workspace/cache/torch"
export IMAGINAIRE_OUTPUT_ROOT="$workspace/runs"
unset COSMOS_SMOKE TRANSFORMERS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
mkdir -p "$workspace/reports"
cd "$framework"
"$python" -m cosmos3_h_embodied.dreams.patch_framework "$framework"
# Check CUDA and FFmpeg/TorchCodec before model/data downloads or conversion.
"$python" -m cosmos3_h_embodied.dreams.runtime_check --gpus "$gpus" --output "$workspace/reports/devices.json"
"$python" -m pytest "$package_root/tools/dreams/tests" -q --junitxml="$workspace/reports/unit_tests.xml" -p no:cacheprovider
if [[ "$setup_only" == 1 ]]; then
    printf '%s\n' "Setup checks passed. Reports: $workspace/reports. Model/data downloads and training were not started."
    exit 0
fi
if [[ ! -f "$workspace/run.json" ]]; then
    "$python" -m cosmos3_h_embodied.dreams.smoke init --workspace "$workspace" --gpus "$gpus"
fi
if [[ ! -f "$workspace/assets/receipt.json" ]]; then
    "$python" -m cosmos3_h_embodied.dreams.prepare --workspace "$workspace" --download-models
fi
if [[ ! -f "$workspace/prepared/c72.json" || ! -f "$workspace/prepared/c24.json" ]]; then
    "$python" -m cosmos3_h_embodied.dreams.prepare --workspace "$workspace" --public-smoke --allow-unreviewed-smoke
fi
if [[ ! -f "$workspace/migrated_h16_28d/action_contract.json" ]]; then
    "$python" -m cosmos3_h_embodied.dreams.migrate \
        --source "$workspace/assets/initializer/checkpoints/h16_50000/c3-h-e-n_model_bf16.safetensors" \
        --output "$workspace/migrated_h16_28d"
fi
"$python" -m cosmos3_h_embodied.dreams.preflight --spec "$workspace/run.json" --output "$workspace/reports/preflight" --loaders
"$python" -m cosmos3_h_embodied.dreams.smoke run --workspace "$workspace"
for stage in sf4 sf2; do
    stage_root="$workspace/runs/cosmos3_h_dreams/public_smoke/$stage"
    if [[ ! -f "$workspace/native/$stage/export.json" ]]; then
        "$python" -m cosmos3_h_embodied.dreams.export --source "$stage_root/checkpoints/iter_000000012" \
            --config "$stage_root/config.yaml" --output "$workspace/native/$stage"
    fi
    "$python" -m cosmos3_h_embodied.dreams.evaluate \
        --checkpoint "$workspace/native/$stage/reload.dcp/model" --config "$workspace/native/$stage/inference.yaml" \
        --spec "$workspace/run.json" --output "$workspace/review/$stage" --samples 1 --modes recorded energy_off
done
printf '%s\n' "Training, export and inference completed. Review $workspace/reports and $workspace/review."
