#!/usr/bin/env bash
# Evaluate released students without training, initializer download, or migration.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
workspace=${1:?Usage: bash tools/dreams/run_released.sh WORKSPACE [4|2|both]}
choice=${2:-both}
case "$choice" in
  4) stages=(4); model_set=sf4 ;;
  2) stages=(2); model_set=sf2 ;;
  both) stages=(4 2); model_set=students ;;
  *) printf 'Choose 4, 2, or both\n' >&2; exit 2 ;;
esac
workspace=$(cd "$workspace" && pwd)
framework="$workspace/sources/cosmos-framework"
python="$framework/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$workspace/cache/huggingface" TORCH_HOME="$workspace/cache/torch"
unset COSMOS_SMOKE TRANSFORMERS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
cd "$framework"
"$python" -m cosmos3_h_embodied.dreams.patch_framework "$framework"
"$python" -m cosmos3_h_embodied.dreams.runtime_check --gpus 1 --output "$workspace/reports/devices.json"
if [[ ! -f "$workspace/run.json" ]]; then
    "$python" -m cosmos3_h_embodied.dreams.smoke init --workspace "$workspace" --gpus 1 --profile inference --run-name released_evaluation
fi
"$python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["purpose"] == "inference_only", "Use a separate inference workspace"' "$workspace/run.json"
"$python" -m cosmos3_h_embodied.dreams.prepare --workspace "$workspace" --download-models --model-set "$model_set"
if [[ ! -f "$workspace/prepared/c24.json" || ! -f "$workspace/prepared/c72.json" ]]; then
    "$python" -m cosmos3_h_embodied.dreams.prepare --workspace "$workspace" --public-smoke --allow-unreviewed-smoke
fi
for steps in "${stages[@]}"; do
    "$python" -m cosmos3_h_embodied.dreams.inference_config --spec "$workspace/run.json" \
        --stage "sf$steps" --output "$workspace/configs/sf$steps.yaml"
    "$python" -m cosmos3_h_embodied.dreams.released --workspace "$workspace" --steps "$steps"
    "$python" -m cosmos3_h_embodied.dreams.evaluate \
        --checkpoint "$workspace/released_sf$steps.dcp/model" --config "$workspace/configs/sf$steps.yaml" \
        --spec "$workspace/run.json" --output "$workspace/review/released_sf$steps" \
        --split test --samples 3 --seed 28003 --full-metrics \
        --modes recorded translation_hold rotation_hold energy_off
done
printf 'Review %s/review/ and %s/reports/\n' "$workspace" "$workspace"
