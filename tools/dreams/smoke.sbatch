#!/usr/bin/env bash
# Caller supplies site-specific account/partition/container flags to sbatch.
# Example: sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
#   --gpus-per-node=2 tools/dreams/smoke.sbatch \
#   /absolute/package/root /absolute/workspace cu130-train 2
# Resource headers below are reference defaults; override for your target hardware.
# No GPU GRES header: pass --gpus-per-node=N on GRES sites or --exclusive on
# whole-node GPU sites. The positional GPU count controls torchrun, not allocation.
#SBATCH --job-name=cosmos3-h-dreams-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=04:00:00
set -euo pipefail
package_root=${1:?package root required}
workspace=${2:?workspace required}
cuda_group=${3:-cu130-train}
gpu_count=${4:-8}
if [[ ! "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
    printf 'GPU count must be a positive integer: %s\n' "$gpu_count" >&2
    exit 2
fi
mkdir -p "$workspace"
worker=(bash -c 'set -euo pipefail; bash "$1/tools/dreams/bootstrap.sh" "$2" "$3"; bash "$1/tools/dreams/run_smoke.sh" "$2" "$4"' dreams-smoke "$package_root" "$workspace" "$cuda_group" "$gpu_count")
options=(--ntasks=1 --kill-on-bad-exit=1)
if [[ -n "${DREAMS_CONTAINER_IMAGE:-}" ]]; then
    options+=(--container-image="$DREAMS_CONTAINER_IMAGE"
        --container-mounts="${DREAMS_CONTAINER_MOUNTS:-$package_root:$package_root,$workspace:$workspace}"
        --container-workdir="$workspace" --container-entrypoint)
fi
srun "${options[@]}" "${worker[@]}"
