# From an H16 world model to four-step and two-step students

This tutorial supplies a public-framework implementation of the teacher and
Self-Forcing recipe used for Cosmos3-H-Dreams. Start from the public
Cosmos3-H-Embodied-Nano H16 checkpoint, adapt supplied-action video prediction to
CMR cholecystectomy, and distill four-step and two-step causal students.

**This page is the sole starting document.** Use the repository checkout containing
this tutorial and its companion code; the commands below cover setup through
student evaluation. No separate implementation handoff, previous conversation,
private repository or pre-existing agent workspace is required.

A manual download or copy of this complete repository is sufficient; set
`C3HEN_DIR` below to its extracted root. Keep `cosmos3_h_embodied/`, `tools/dreams/`
and `pyproject.toml` together, including the shipped JSON and patch files. You do
not need this repository's Git history to run the tutorial. Bootstrap downloads
the pinned public framework; later steps download the documented public assets.

**Validation status (2026-11-05):** the bounded six-stage training smoke,
C24/SF4 resume checks, both student exports/reloads and all four evaluation
rollouts completed on eight H100 80 GB GPUs with the documented image
prerequisites. A separate fresh setup run then verified the documented image
recipe, scheduler launcher, managed Python bootstrap, FFmpeg/TorchCodec loading,
BF16 on all eight GPUs, **55 tests**, and small-MP4 decoding. A transient GitHub
download error required retrying bootstrap in the same workspace; no code fixes
were needed. The setup run stopped before model/data downloads and training.

These checks establish bounded integration, not full-training convergence or
released-checkpoint quality. Both smoke students scored below the repeat-first-
frame baseline. Evaluation used H0 and already-zero energy controls; exported
H16 inference and sensitivity to a nonzero energy intervention remain unqualified.
The released-student launcher and full research curriculum have CPU regression
coverage but were not exercised by these GPU smoke/setup runs. The final setup-only
exit option and image-build shell guards were checked locally.

The private reviewed selections are not shipped. The public-only smoke fixture
below tests integration. It is **telemetry-filtered but not visually reviewed**,
and its recordings may have been seen during Open-H pretraining.

Choose a workflow before downloading large files:

| Goal | Entry point | Hardware evidence |
|---|---|---|
| Use released four-/two-step students | Section 2: released-student inference | Reported successful on one 48 GB GPU; no initializer or training required |
| Check training integration | Section 2: bounded training smoke | Passed on eight H100 80 GB GPUs; failed at C24 A on a tested pair of 48 GB GPUs |
| Train on reviewed CMR selections | Sections 3 and 6 | Full budgets and checkpoint selection supplied; full-curriculum convergence unqualified |

The independent portability review confirmed setup, tests, migration and real
loaders, but exposed missing inference-only and research commands. Those commands
are now included and covered by CPU regression tests; this does not establish
GPU execution of the independent inference workflow or full research curriculum.

## 1. Recipe and implementation map

| Term | Meaning |
|---|---|
| C24 | 24 future images at 10 Hz plus the initial image; 2.4 seconds |
| C72 | 72 future images plus the initial image; 7.2 seconds |
| H16 | Up to 16 genuine preceding action transitions, encoded at the initial image's anchor |
| TF | Causal teacher forcing on recorded video context |
| SF | Self-Forcing on generated context with distribution-matching supervision |
| Four-step / two-step | Denoising evaluations per block, not the number of actions |

A block is one latent temporal frame: four future video frames and four supplied
action transitions (0.4 seconds). The model generates video, not controller commands.

```text
Public H16 initializer → migrate CMR 20D to 28D
  → C24 adaptation A: 2,000 updates on an initial reviewed selection
  → C24 adaptation B: 2,000 updates on an expanded reviewed selection
  → C72 teacher: 4,400 C72-stage updates (max_iter)
  → causal TF: 1,000 updates
  → four-step SF: 300 logical updates
  → two-step SF: 300 additional logical updates
```

There is no separate DCM stage. This was an empirical reconstruction result, not
a claim that DCM is generally ineffective. SF still uses distribution matching.
Each SF stage uses `student_update_freq=6`, with no warmup: 300 logical updates
mean **50 student updates and 250 critic updates**.

[exp15_distillation_reference.json](exp15_distillation_reference.json) contains the
frozen numerical settings. [dreams/experiments.py](dreams/experiments.py) translates
those settings into actual registered experiments. A/B describe successive
adaptations; the smoke test deliberately uses the same tiny fixture for both.
That simplification is not equivalent to the reference dataset curriculum.

| Component | Shipped implementation |
|---|---|
| Immutable public asset identities | [assets.json](dreams/assets.json) |
| Native 60 Hz validity, anchored poses, controls, PTS decoding | [data.py](dreams/data.py) |
| Source downloads, reviewed selections, camera8000 manifests | [prepare.py](dreams/prepare.py) |
| Uniform accepted-start sampling and H16 collation | [loader.py](dreams/loader.py) |
| Strict 20D → 28D projector migration | [migrate.py](dreams/migrate.py) |
| Pinned causal H16 framework extension | [framework_h16.patch](dreams/framework_h16.patch), [history_attention.py](dreams/history_attention.py) |
| Six stages and regular-net teacher/critic loading | [experiments.py](dreams/experiments.py), [training.py](dreams/training.py) |
| Restartable stage driver | [smoke.py](dreams/smoke.py) |
| Regular-net BF16 exports and native inference | [export.py](dreams/export.py), [evaluate.py](dreams/evaluate.py) |
| Inference without training | [inference_config.py](dreams/inference_config.py), [released.py](dreams/released.py), [run_released.sh](../tools/dreams/run_released.sh) |
| Full-budget research specs and checkpoint selection | [research.py](dreams/research.py) |

## 2. Setup, resource planning and executable workflows

Use a **separate checkout/environment** from an existing C3HEN teacher setup. The
companion pins public `NVIDIA/cosmos-framework` to
`1a2f03ab6111b457633827736f94f67d55bd5202` and applies a checksum-guarded, narrow
H16 extension. It preserves newer upstream functionality outside that extension.
The older validated C3HEN teacher revision in the main README is a separate
compatibility statement.

The framework supplies causal TF and Self-Forcing. The companion supplies the
CMR28/H16 integration; this integration is not already present in the generic
Bimanual example. That example uses a different 59D action contract, normalization,
chunk size, optimization budget and initialization behavior.

### Environment and resources

On a shared cluster, run installation, data preparation, checkpoint conversion
and training on allocated compute workers. On a dedicated workstation, run them
locally in the chosen compatible environment; Slurm is optional. Follow the pinned
framework's [setup guide](https://github.com/NVIDIA/cosmos-framework/blob/1a2f03ab6111b457633827736f94f67d55bd5202/docs/setup.md),
including CUDA/toolchain and FFmpeg dependencies. The tested base image is
`nvcr.io/nvidia/pytorch:26.06-py3`, **with the system packages and uv added below**.
The unmodified base image is insufficient: the fresh test found neither uv nor
the FFmpeg shared libraries needed for evaluation. Prerequisites must be installed
**inside the worker/container that runs bootstrap**, even if they exist on the
host. Prepare that runtime before starting the quickstart:

```bash
# Run as root inside an image build/writable container, or use sudo on a workstation.
apt-get update
apt-get install -y --no-install-recommends \
  ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev curl wget tree
curl -LsSf https://astral.sh/uv/0.11.29/install.sh \
  | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
uv --version
ffmpeg -version
```

For Slurm/Pyxis, the following Enroot recipe derives a reusable image from that
public base. Run the build on an allocated worker. It requires site-supported
Enroot root remapping and network access; use your site's image builder if those
are unavailable. The import/start/export procedure was used successfully in the
fresh rerun; these generic paths replace that run's site-specific paths.

<details>
<summary>Build the prerequisite image with Enroot</summary>

```bash
export DREAMS_CONTAINER_IMAGE=/path/to/writable/images/pytorch-26.06-py3-dreams.sqsh
(
set -euo pipefail
mkdir -p "$(dirname "$DREAMS_CONTAINER_IMAGE")"
if [[ -e "$DREAMS_CONTAINER_IMAGE" || -e "$DREAMS_CONTAINER_IMAGE.partial" ]]; then
  printf '%s\n' 'Choose a new image path; an image or partial export already exists.' >&2
  exit 1
fi
dreams_image_tmp=$(mktemp -d "${TMPDIR:-/tmp}/dreams-image.XXXXXXXX")
trap 'rm -rf -- "$dreams_image_tmp"' EXIT
export ENROOT_TEMP_PATH="$dreams_image_tmp/tmp" ENROOT_RUNTIME_PATH="$dreams_image_tmp/runtime"
export ENROOT_DATA_PATH="$dreams_image_tmp/data" ENROOT_CACHE_PATH="$dreams_image_tmp/cache"
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"
export ENROOT_MAX_PROCESSORS=1 ENROOT_MAX_CONNECTIONS=1
enroot import --output "$dreams_image_tmp/base.sqsh" 'docker://nvcr.io#nvidia/pytorch:26.06-py3'
enroot create --name dreams-image "$dreams_image_tmp/base.sqsh"
enroot start --root --rw dreams-image bash -c '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev curl wget tree
curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
rm -rf /var/lib/apt/lists/*
uv --version
ffmpeg -version
'
enroot export --output "$DREAMS_CONTAINER_IMAGE.partial" dreams-image
mv "$DREAMS_CONTAINER_IMAGE.partial" "$DREAMS_CONTAINER_IMAGE"
)
```

The exported image is the input to the Slurm example below. Keep it outside your
training workspace so a fresh training run can use a new workspace. The exit trap
cleans this build's temporary container files. The subshell stops on a failed
build command and keeps its exit trap and environment changes local.

</details>

Bootstrap installs the framework-required `uv 0.11.29` inside `WORKSPACE`, without
changing the host installation. Bootstrap now requires a uv-managed Python 3.13
under `WORKSPACE/cache/python`, rather than inheriting a host/Conda interpreter.
An existing environment based on a host interpreter is rejected: use a fresh
workspace. `reports/environment.txt` records the base interpreter.

The launch scripts set `PYTHONDONTWRITEBYTECODE=1` to suppress Python/pytest
`__pycache__` writes in the checkout; bootstrap also disables uv bytecode compilation.
Use the same setting for manual commands below. Editable installs can still use
build metadata: use a writable extracted copy if your build tooling requires it,
rather than relying on a shared read-only checkout.

```bash
export PYTHONDONTWRITEBYTECODE=1
```

The scripts relocate Hugging Face caches into `WORKSPACE`, so a login stored under
your normal `HF_HOME` may not be found. Supply `HF_TOKEN` through your environment
or secret manager before downloads, or log in to the workspace cache after setup:

```bash
HF_HOME="$WORKSPACE/cache/huggingface" \
  "$WORKSPACE/sources/cosmos-framework/.venv/bin/hf" auth login
```

Never put tokens in committed scripts, run specifications or reports. The scripts
preserve `HF_TOKEN`. Authentication can affect access and download behavior;
network speeds are not guaranteed.

Run setup **and training in the same compatible runtime**. Successfully importing
Torch does not establish CUDA compatibility: a host's older driver may need the
supported compatibility libraries provided by your training container. Preserve
those library paths. After bootstrap installs Python dependencies, both launch scripts check
`ffmpeg -version`, the framework's `load_torchcodec()` shared-library loader,
CUDA initialization and a small BF16 operation before model/data downloads or
conversion. A standalone ffmpeg executable does not establish that TorchCodec
can load its shared libraries. Media failures are recorded in `reports/devices.json`
with the full chained loader error in the worker log. If this fails, correct the
runtime or choose the compatible CUDA dependency group; do not bypass the check.
The pinned upstream setup guide documents the CUDA/driver requirements.

**Eight GPUs are a reference configuration, not a required GPU count.** Adapt the
launcher and resource request to your target hardware, then qualify that setup
with the smoke workflow. This does not promise that the unchanged model and recipe
fit every device or GPU count.

### Model size, memory and disk

The initializer contains approximately **15.17 billion parameters**, about
**30.3 GB in BF16**, with approximately **6.97 billion selected for training**.
“Nano” does not mean this is a small workstation training workload.

A rough sharded persistent-state estimate for the teacher is:

```text
P = all parameters; T = trainable parameters; N = data-parallel shards
bytes per rank ≈ (4P + 4T + 8T) / N
                 FP32 parameters + gradients + two Adam moments
For P≈15.17B and T≈6.97B: ≈144 GB total / N, before activations and other buffers.
```

This estimate is not an exact allocator model: transient unsharding, communication,
kernels, frozen parameters and activation lifetimes affect peaks. SF also holds a
critic and fixed teacher. Two shards imply about 72 **decimal GB** per rank for
the estimated teacher state alone.

**The unchanged recipe failed at C24 A step 1 on two 48 GB GPUs**, before the first
optimizer update. An allocator retry did not resolve the failure. Use the
inference path on hardware of that class. Fewer updates do not solve per-step
memory pressure, and no qualified offload/low-VRAM training preset is shipped.
No minimum count of other GPU types is claimed.

Eight H100 80 GB GPUs are the successful training-smoke reference. Production used
48 H100s, batch 1 per rank and no accumulation. Host RAM of 512 GiB was a planning
request, not a measured minimum. A separate 125 GiB host completed migration and
preflight, but did not reach training export; that does not qualify export memory.

Plan disk before launch (GB/TB below are decimal; GiB/TiB are binary):

| Item | Approximate size or budget |
|---|---:|
| Bootstrap wheel downloads | 10 GB; installed environment/cache uses additional space |
| H16 initializer | 30.3 GB |
| Video VAE | 2.7 GB |
| Each released student | 30.3 GB, plus approximately 30 GB for its DCP conversion |
| Three-episode smoke data | About 170 MB in the reported run; bounded search can fetch additional telemetry |
| Full-state teacher/TF checkpoint | Estimated 115–117 GB |
| Full-state SF checkpoint | Estimated 230–234 GB |
| Default smoke checkpoint retention | Five teacher/TF and three SF saves: about 1.3 TB |
| Both student exports | About 180 GB for `.pt`, safetensors and reload DCP, plus temporary space |

**Reserve at least 2 TiB free scratch for the documented training smoke.** This is
a planning requirement for its retention/export policy, not a universal measured
minimum. Checkpoints, assets and exports approach roughly 1.5 TB, with caches and
temporary files on top. Full research data and periodic research saves require a
separate budget. For inference only, reserve about 200 GiB for both students and
conversions, dependencies and small evaluation data; larger datasets and rollouts
need more. Download time depends on bandwidth and access.

Do not delete intermediate smoke saves mid-pipeline. After completion, archive
reports, selected weights and resume checkpoints before removing unneeded copies.
Research retention must preserve selected parents, the fixed C72 scoring teacher,
each latest full-state resume point and candidates still under review. A receipt
alone cannot replace weights/optimizer state needed for restart. No automatic
pruning is performed.

### Bounded training smoke

Use this path with adequate resources. For inference alone, skip the training
commands and use the released-student subsection below. The driver removes
`COSMOS_SMOKE` so it always tests the full model, never a two-layer replacement.

```bash
export C3HEN_DIR=/path/to/Cosmos3-H-Embodied-Nano
export WORKSPACE=/path/to/new/writable/dreams-smoke
export GPU_COUNT=8  # Reference; choose the number of visible GPUs for your run.
# Run in the prepared environment above (NGC 26.06-py3 + system packages + uv).
bash "$C3HEN_DIR/tools/dreams/bootstrap.sh" "$WORKSPACE" cu130-train
export PY="$WORKSPACE/sources/cosmos-framework/.venv/bin/python"
bash "$C3HEN_DIR/tools/dreams/run_smoke.sh" "$WORKSPACE" "$GPU_COUNT"
```

For CUDA 12.8, use the framework-supported `cu128-train` group instead. Caches,
model downloads, data and results live under `WORKSPACE`; no previous agent's
virtualenv, prepared selection, checkpoint cache or private repository is required.
The helper pins pytest/LPIPS and uses the framework's lockfile for training dependencies.

A scheduler-neutral shell script is provided, together with a [Slurm example](../tools/dreams/smoke.sbatch).
Supply your own account and partition. On Slurm/Pyxis sites, set
`DREAMS_CONTAINER_IMAGE` to your supported training image; the example explicitly
starts a containerized `srun` worker. Use `DREAMS_CONTAINER_MOUNTS` if additional
mounts are required. Both the repository and workspace must be visible at the same
absolute paths inside the container. Without an image setting, the allocated
worker's host runtime must already be compatible.

```bash
# Optional on sites with Slurm/Pyxis container support:
export DREAMS_CONTAINER_IMAGE=/path/to/writable/images/pytorch-26.06-py3-dreams.sqsh
# Choose ONE allocation style for your site:
GPU_REQUEST=(--gpus-per-node="$GPU_COUNT")  # GPU GRES sites
# GPU_REQUEST=(--exclusive)                 # Whole-node GPU sites
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  "${GPU_REQUEST[@]}" \
  "$C3HEN_DIR/tools/dreams/smoke.sbatch" "$C3HEN_DIR" "$WORKSPACE" cu130-train "$GPU_COUNT"
```

Some clusters allocate GPUs through exclusive whole-node requests
rather than GPU GRES. Choose `GPU_REQUEST=(--exclusive)` on those sites; an
`Invalid generic resource` error is a scheduler request issue. `smoke.sbatch` has
no GPU GRES header: you must supply the appropriate allocation request. Its
positional GPU count controls training ranks, not scheduler allocation. The
host-memory and time headers are reference defaults; override them as needed. Keep one node and one
launcher process for this smoke, and match its GPU-count argument to the GPUs
available to the worker. `torchrun` launches one training rank per selected GPU.
If your scheduler enforces a job-name convention, pass `--job-name=YOUR_JOB_NAME`
to `sbatch` to override the example's name.

For **setup validation only**, set `DREAMS_SETUP_ONLY=1`. Bootstrap still installs
the environment; `run_smoke.sh` then checks media/GPU dependencies and runs the
tests before exiting successfully. It does not initialize a run, download
model/data assets, migrate weights or train. No placeholder files are needed:

```bash
# After bootstrap, in the prepared worker environment:
DREAMS_SETUP_ONLY=1 bash "$C3HEN_DIR/tools/dreams/run_smoke.sh" "$WORKSPACE" "$GPU_COUNT"

# Or bootstrap and validate through the same containerized scheduler launcher:
DREAMS_SETUP_ONLY=1 sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  "${GPU_REQUEST[@]}" \
  "$C3HEN_DIR/tools/dreams/smoke.sbatch" "$C3HEN_DIR" "$WORKSPACE" cu130-train "$GPU_COUNT"
```

These are alternative launch methods. To continue with the full smoke in the same
workspace, rerun the normal command with `DREAMS_SETUP_ONLY` unset or set to `0`.

If Pyxis/Enroot fails before bootstrap with curl `(23) Failed to truncate file`,
see [container-import troubleshooting](#container-import-troubleshooting) at the
end of this tutorial. This intermittent site-specific issue did not occur in
the latest fresh rerun; apply the workaround only if that exact failure occurs.

### Adapting to a workstation or another GPU count

Start by recording GPU models, memory available on **each** device, driver/runtime
versions, host RAM and free disk. For a mixed-GPU workstation, verify both devices
with the installed runtime; do not assume identical kernels, performance or memory
headroom. The standard sharding configuration does not assign less work to a
smaller GPU. Adding the two VRAM capacities together is not a fit check.

After bootstrap, run this small check inside the same environment/container:

```bash
nvidia-smi
"$PY" - <<'PY'
import torch
for device in range(torch.cuda.device_count()):
    with torch.cuda.device(device):
        free, total = torch.cuda.mem_get_info()
        x = torch.ones((16, 16), device=f"cuda:{device}", dtype=torch.bfloat16)
        (x @ x).sum().item()
        torch.cuda.synchronize()
        print(device, torch.cuda.get_device_name(device),
              f"free={free / 2**30:.1f} GiB total={total / 2**30:.1f} GiB; BF16 passed")
PY
```

GPU count is configurable, but the unchanged recipe is known to fail on the
tested two-48-GB setup. For another sufficiently provisioned setup, choose a new
workspace and set `GPU_COUNT` to the visible device count. The driver records that
world size. Batch 1 per rank with accumulation 1 gives effective batch `GPU_COUNT`,
not automatically 8 or the production batch of 48.

Use GPU UUIDs to avoid confusing CUDA indices with `nvidia-smi` indices:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used --format=csv
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Optional: substitute a UUID from the command above to select a device.
# export CUDA_VISIBLE_DEVICES=GPU-UUID-OF-SELECTED-DEVICE
```

CUDA indices are relative to the selected list. A display GPU may already have
substantial VRAM in use. The launcher now checks BF16 on **every** selected GPU and
writes identity/free-memory information to `reports/devices.json`.

Measured training peaks from the eight-H100 smoke, taking the maximum over ranks:

| Stage | Allocated GPU memory (GiB) | Reserved GPU memory (GiB) |
|---|---:|---:|
| C24 A/B | 23.2 | 29.2 |
| C72 | 24.7 | 30.1 |
| Causal TF | 38.2 | 47.9 |
| SF4/SF2 | 57.4 | 73.3 |

These are observations for that exact eight-GPU run, not minimum VRAM requirements
or predictions for two GPUs. Fewer shards can increase the model/optimizer memory
per rank; TF/SF also have different activation and scoring-network costs. A
successful teacher stage does not establish that SF will fit. Fewer updates or a
smaller dataset shorten the run but do not solve per-step memory pressure. Host
checkpoint conversion/export has separate memory and disk requirements.

If the unchanged smoke runs out of memory, preserve the failed stage and measured
peak. Use more/larger compatible GPUs, or develop and qualify a memory-saving
configuration before continuing. This package does **not** yet ship a validated
low-VRAM/offload preset. Do not silently reduce C72, H16, resolution, model size or
the student/critic update schedule to obtain a passing result: label such changes
as a different experiment and rerun its semantic and GPU checks. Gradient
accumulation can change effective batch but is not an automatic memory-fit fix;
the shipped registrations set it to 1, and SF phase/counter behavior must be
qualified before changing it.

Use a fresh workspace when changing world size or contracts; do not edit an
existing run's world size and claim a same-stage resume. Preserve the effective
batch, resolved settings, peak memory and update times in the qualification
record. For a longer run, explicitly decide how a different effective batch
changes the optimization budget. Do not repeat a known unsupported two-48-GB
training attempt just to reproduce the OOM; use released-student inference.

### Released-student inference without training

This path needs no trained teacher, initializer migration, smoke export or
external helper scripts. Use a **new inference workspace**, select one suitable
GPU, and run setup and inference inside the same runtime:

```bash
export C3HEN_DIR=/path/to/Cosmos3-H-Embodied-Nano
export WORKSPACE=/path/to/new/writable/dreams-inference
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Optionally select a GPU UUID with CUDA_VISIBLE_DEVICES.
bash "$C3HEN_DIR/tools/dreams/bootstrap.sh" "$WORKSPACE" cu130-train
export PY="$WORKSPACE/sources/cosmos-framework/.venv/bin/python"
bash "$C3HEN_DIR/tools/dreams/run_released.sh" "$WORKSPACE" both
```

Use `4` or `2` instead of `both` to evaluate only that student. The script downloads
the VAE/tokenizer and chosen weights, prepares the public fixture, generates the
inference config without training, converts weights and evaluates three test
windows. Conversion uses an atomic output directory and source/DCP hashes; reruns
verify existing files. Incomplete or changed conversions fail explicitly. No
CMR20→28 migration is applied to released students.

A reported single-48-GB Ada GPU test used approximately 38.5 GB peak VRAM. One
7.2-second rollout took about 45 seconds **including model loading**, not measured
steady-state throughput. Both released students completed H0/H16 evaluation.
These observations are not guaranteed resource requirements for other hardware.

For custom reviewed data, initialize an inference spec without starting training:

```bash
"$PY" -m cosmos3_h_embodied.dreams.smoke init \
  --workspace "$WORKSPACE" --gpus 1 --profile inference --run-name released_evaluation
```

Run this only if `run.json` does not exist. Prepare the reviewed data using
Section 3, set `manifests.c72` in the inference spec to the prepared C72 manifest,
and set `data_root` to its source-data root. Then use the explicit Section 7
commands. `run_released.sh` is the small-fixture entry point. An inference-profile
workspace cannot accidentally launch training.

### Included compatibility changes and patch handling

The tutorial, companion implementation and tests must come from the same
repository revision. There are two distinct kinds of changes:

| Change | Where it lives | What the reader does |
|---|---|---|
| Upstream causal H16 extension | [framework_h16.patch](dreams/framework_h16.patch), [framework_h16_files.json](dreams/framework_h16_files.json), [patch_framework.py](dreams/patch_framework.py) | Run bootstrap; it applies and verifies the patch against the pinned framework revision automatically. |
| Dataset-backed evaluator fixes | [evaluate.py](dreams/evaluate.py), [regression tests](../tools/dreams/tests/test_evaluate_args.py) | Use this repository version; these are ordinary integrated source changes, with no additional manual patch command. |

The independent run found that the public forward-dynamics argument builder
requires an `action_path` even when generation receives an in-memory dataset
batch. The evaluator now writes the actual normalized CMR28 action rows to
`conditioning_actions.json`, with history/sequence metadata in
`conditioning_metadata.json`, and passes that real file to argument validation.
The in-memory batch remains authoritative for history and masks; the action JSON
alone is not a complete H16 replay bundle.

The evaluator also explicitly sets `autoregressive` from the checkpoint's causal
configuration, because upstream forward-dynamics defaults otherwise select
`false`. Distilled sampling uses the checkpoint's fixed four-/two-step schedule.
The regression tests cover the public argument builder for teacher, SF4 and SF2.
An older checkout lacking these changes can fail with an `action_path` error or
select the wrong generation path. Upgrade the companion and tutorial together;
do not bypass the validation or run a causal student bidirectionally.

Repository maintainers must include these source changes and tests when publishing
this tutorial. An integration diff used during development is not a downstream
dependency. Any future external patch must be shipped, version-pinned and
documented here with its application and verification steps.

The complete **training** smoke sequence is:

```text
Download pinned assets and three eligible public episodes from distinct recordings
→ native C24/C72 preparation, with explicit unreviewed-smoke status
→ CPU geometry/projector/packing tests and actual-loader checks
→ migrate the initializer (regular weights only)
→ C24 A: 1 update, save, full-state resume to 2
→ C24 B: 2 → C72: 2 → TF: 2
→ SF4: 6 logical updates, save, full-state resume to 12
→ SF2: 12 logical updates
→ export both students → recorded/energy-off native inference
```

Twelve SF logical updates exercise two student and ten critic updates. The shell
fails immediately if a command fails; it does not label a partial run successful.
Re-run the same command to resume an interrupted stage. Keep `run.json` and its
output directories together. Do not change world size, selected data or contracts
between a same-stage save/resume test.

Outputs include:

- `reports/environment.txt`, `reports/unit_tests.xml`, `reports/preflight/*.yaml`
  and `reports/preflight/preflight.json` with real loader shapes. Preflight YAMLs
  now use the trainer serialization path and validate native config/architecture
  reload; they are no longer inspection-only dumps. Use `inference_config` to
  project a training configuration to an inference-only model.
- `reports/pipeline.json` with stage checkpoints, logs, elapsed times and failures.
- `runs/cosmos3_h_dreams/public_smoke/<stage>/smoke_health_rank*.jsonl`
  with finite losses, delivered sample identifiers and GPU memory usage.
- `native/sf4/` and `native/sf2/`: `model_bf16.pt`, `model_bf16.safetensors`,
  inference configuration, contract and export hashes.
- `review/sf4/index.html` and `review/sf2/index.html`, with rollouts and metric JSON.

The native framework accepts a model bundle or DCP, not a bare `.pt`. Each export
therefore re-reads the actual `.pt`, verifies exact tensor equality, and writes a
`reload.dcp/model` used for the inference test. The `.pt`/safetensors files remain
available for downstream integrations. A successful smoke does not imply that the
few-update student produces useful surgical motion.

### Inspect failures and resume

On Python exceptions, `failure_rankNN.json` in the stage directory records the
last iteration, error and available CUDA allocation/reservation peaks, even when
the first step fails. This is best-effort: an OS kill, node failure or process
abort may prevent the hook from running. Preserve the OOM message too; other
processes' GPU allocations are not included in PyTorch allocator peaks.

Keep the scheduler output (or capture the shell's stdout/stderr) as well as
`WORKSPACE/reports/`. A failure during setup or preparation happens before
`pipeline.json` exists. Once training starts, `pipeline.json` identifies the failed
stage and its log. Fix the first reported error before resuming; a missing later
student is usually a consequence of that earlier failure.

Transient bootstrap download failures, including HTTP 504 Gateway Timeout from
GitHub during `uv sync`, are retry cases: rerun bootstrap with the **same workspace**
once connectivity recovers. Retain completed downloads and the managed environment;
do not change dependency pins to work around a transient network error.

After correcting an environment problem, re-enter the supported worker/container
and rerun the two commands in the quickstart with the **same workspace**. After a
training interruption, rerun `run_smoke.sh` in that runtime. Completed artifacts
are reused; same-stage training resumes from its saved full state. The driver
rejects changed manifests, parents, stages or world size rather than silently
reusing incompatible checkpoints. Migration and exports publish only after
successful write/readback checks. Do not delete the workspace to retry.

If you submit setup and training as separate dependent scheduler jobs, a failed
setup leaves the dependent job unable to start. Retry setup and update the training
job's dependency, or resubmit the combined Slurm example after confirming no
previous attempt is running. Do not submit duplicate training writers into one
workspace.

Call the complete smoke successful only when:

1. `reports/pipeline.json` records completion of all six requested training stages.
2. Both `native/sf4/export.json` and `native/sf2/export.json` exist, with verified
   exported tensors and native reload checkpoints.
3. Both `review/sf4/results.json` and `review/sf2/results.json` contain the requested
   recorded/action-intervention outputs, and their `index.html` videos are playable.

A completed setup job, passing CPU tests, or a saved teacher alone does not satisfy
these criteria. The save/resume probes exercise restart paths; they do not yet
establish full-model numerical equivalence to a duplicate uninterrupted run.
The exact CPU delivered-sample/dropout restart check is a narrower guarantee.

## 3. Supply your own reviewed data

Here, “your own data” means reviewed data conforming to the **CMR Versius
cholecystectomy layout and reference instrument pair**. This is not a generic
robot/procedure importer. Another robot, action layout or instrument pair requires
changes to `dreams/data.py` (native fields, geometry and instrument validity),
`dreams/prepare.py` (instrument declarations), `dreams/contract.py` (schema/domain
identity), and the reference JSON (action map and normalization). Requalify
packing, migration, training and inference. Matching shapes or reusing domain 25
does not establish semantic compatibility.

The full dataset is separate from the smoke's three-episode download. Query its
size before allocating space; this writes the pinned file/byte inventory to
`reports/dataset_inventory.json` without downloading media:

```bash
"$PY" -m cosmos3_h_embodied.dreams.prepare --workspace "$WORKSPACE" --dataset-size
"$PY" -m cosmos3_h_embodied.dreams.prepare --workspace "$WORKSPACE" --download-dataset
```

The second command downloads the complete pinned cholecystectomy prefix into
`WORKSPACE/data/Surgical/cmr_surgical/cholecystectomy/`. Budget its reported bytes
**in addition** to checkpoints and exports. The smoke's generated `selection.json`
is a concrete path-format example, but must not simply be relabeled as reviewed.

The source dataset is [PhysicalAI-Robotics-Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment),
under `Surgical/cmr_surgical/cholecystectomy`. The smoke downloader uses the pinned
native LeRobot v2.1 files directly. **Conversion to v3 is not required for that
path.** For v3 data, explicitly list every parquet shard containing an episode and
its video shard offset in the descriptor below; preserve contiguous episode-local
frame indices. The loader reads both through the same column-projected path.

For research training, use manual visual review to
retain the two controlled instruments: **left Cadiere Grasper, right Monopolar
Hook**. Telemetry instrument IDs establish tool identity, not visible tool count.
Document sparse versus dense review and limitations. Native Hook enums 8 and 11
are treated as the same instrument; the left Cadiere enum is 0.

Inspect each
proposed interval at 10 Hz, recording episode-local `[start, end)` native row
ranges only where both controlled tools are visible and no additional instrument
is visible. Include the 1.6-second history margin in the review. Reject ambiguous
frames or shorten the interval; do not treat tool telemetry as a visual label.
A second reviewer should inspect uncertain intervals and a sample of accepted
ones. Record reviewer, sampling rate, decision and uncertainty in your selection
provenance. Sparse review must be labeled as such, rather than called frame-exact.
Convert those accepted ranges into `reviewed_intervals` below; the index builder
then intersects them with the telemetry rules independently for C24 and C72.

Split by recording, and case where known, before selecting overlapping windows.
Protect recordings used by ancestors' validation/test and report remaining
pretraining overlap. Do not use the public smoke fixture to claim untouched test
performance. The original reviewed selections and their private manifests are
not included; a raw-dataset revision and threshold cannot reconstruct them.

A selection JSON uses paths relative to `WORKSPACE/data`, native 60 Hz frame
indices and half-open reviewed intervals:

```json
{
  "review_status": "visually_reviewed_two_instruments",
  "source": {"repository": "nvidia/PhysicalAI-Robotics-Open-H-Embodiment", "revision": "e29dda7cabf2a2626634c7822db27695553ae523"},
  "ancestry_overlap_status": "DESCRIBE_YOUR_AUDIT",
  "episodes": [
    {
      "episode_id": 123,
      "recording_id": "recording-a",
      "split": "train",
      "data_files": ["Surgical/cmr_surgical/cholecystectomy/data/chunk-000/episode_000123.parquet"],
      "video": "Surgical/cmr_surgical/cholecystectomy/videos/chunk-000/observation.images.endoscope/episode_000123.mp4",
      "video_offset_seconds": 0.0,
      "instrument_names": ["Cadiere Grasper", "Monopolar Hook"],
      "reviewed_intervals": [[0, 3600]]
    }
  ]
}
```

Add nonempty `val` and `test` recordings; preparation rejects recording overlap
across splits. For v3, `data_files` may contain multiple shards and
`video_offset_seconds` is the episode's start in the shared video. Review spans
must cover the retained history as well as the future. Use source-relative
identifiers so a selection manifest can be shared without media or internal paths.

```bash
PY="$WORKSPACE/sources/cosmos-framework/.venv/bin/python"
"$PY" -m cosmos3_h_embodied.dreams.prepare \
  --workspace "$WORKSPACE" --selection /path/to/reviewed-selection-a.json \
  --output-dir "$WORKSPACE/prepared_a"
"$PY" -m cosmos3_h_embodied.dreams.prepare \
  --workspace "$WORKSPACE" --selection /path/to/reviewed-selection-b.json \
  --output-dir "$WORKSPACE/prepared_b"
```

Each output directory contains `c24.json` and `c72.json`. Existing nonempty
directories are rejected to protect previous selections and smoke manifests.
Choose a new directory for each revision. Both manifests are published together
only after preparation succeeds. They record accepted starts, file checksums,
unique minutes, split counts, review status and rejection reasons. Sampling is
uniform **over accepted starts**, with replacement—not uniform over episodes.
The initial loader uses zero workers for an easily audited delivered-sample
resume contract. Qualify worker/prefetch changes separately before scaling up.

The camera filter requires, on both hands,
`sqrt(thumbstick_x**2 + thumbstick_y**2) <= 8000` on **every native 60 Hz row**
from the earliest genuine history transition through the future endpoint,
inclusive. A spike between 10 Hz endpoints rejects the window. History is not
shortened to hide camera activity. This is a camera-command proxy, not proof of
stationary imagery: tissue/tool interaction can still move the view.
Both arms must remain engaged; invalid telemetry, timestamp gaps, linkage or
motion-scaling changes also reject a window. Energy/thumbstick buttons remain
available as conditioning channels.

## 4. Preserve the 28D action and H16 contract

Public starting points in this repository:

- `data/cmr_clinical_dataset.py`: source columns, pose convention, motion scaling.
- `data/history_dataset.py`: contiguous history and anchoring.
- `data/transforms.py`: normalization, clean actions and H16 sequence plans.
- `data/joint_dataloader.py`: history-aware token accounting and collation.
- `configs/experiment_h16.py`: C3HEN architecture and tokenizer configuration.

Paths above are relative to `cosmos3_h_embodied/`. The companion `dreams/data.py`
and `dreams/loader.py` implement this contract for CMR28 and decode approved starts
from the prepared manifest, rather than relying on random rejection.

| Channels, zero-based | Meaning |
|---|---|
| 0–2 / 14–16 | Left/right anchored translation |
| 3–8 / 17–22 | Left/right anchored rotation, two matrix columns in 6D |
| 9 / 23 | Left/right jaw |
| 10 / 24 | Left/right energy |
| 11 / 25 | Left/right thumbstick press |
| 12–13 / 26–27 | Left/right thumbstick X/Y |
| 28–63 | Padding, masked out as semantic channels |

There are **six thumbstick channels**, plus two energy channels. CMR uses domain
ID **25** in the existing 32-domain model.

For a window anchored at native row `a`, transition `s → s+6` uses:

- Source quaternion order **XYZW**.
- Translation `(p[s+6] - p[a]) * translation_scale[a]`, in the source translation
  coordinates, without applying `R[a].T` to the translation.
- Rotation `R[a].T @ R[s+6]`; multiply its axis-angle rotation vector by the
  recorded rotation scale, convert back to a matrix and concatenate the first
  two **columns**: `[r00, r10, r20, r01, r11, r21]`.
- Jaw and thumbstick axes at transition start `s`.
- Energy and thumbstick press as the binary maximum over rows `[s, s+6)`;
  this retains brief native-rate presses.

Poses are **relative to the fixed initial anchor**, not frame-to-frame increments
or cumulative Euler-angle sums. History uses the same anchor as the future.

Preserve pretrained pose/jaw normalization when migrating channels. The JSON
contract includes frozen reference coefficients:
`offset=(q01+q99)/2`, `scale=(q99-q01)/2`, normalized clamp `[-1.5, 1.5]`.
Binary controls use offset 0 / scale 1; thumbstick axes use fixed symmetric scales.
Do not re-fit them after camera filtering if retaining the reference representation.
For a new robot, define a suitable new contract rather than borrowing CMR scales.
The 20 preserved coefficients match this repository's
[public CMR normalizer](data/normalizers/cmr_clinical_all4_backward_anchored_worldtrans_rot6d.json).

Migrate the CMR **semantic projector channels**: old `0:10` stay at `0:10`, old
`10:20` move to `14:24`, and new control contributions start at zero. Preserve
all other domains/modules. Map the reverse action projection and its bias too,
even though FDM does not train action prediction. Record the migration and
normalizer identity. Both layouts pad to 64, so a shape check cannot detect an
unmigrated checkpoint. See the projection-equivalence test below.

### Native CMR arrays and timing

Confirm the downloaded dataset's `meta/modality.json` before applying indices.
For the reference 100-wide `action` and `observation.state` arrays, zero-based:

| Source field | Left | Right |
|---|---:|---:|
| `action`: XYZ | 0:3 | 13:16 |
| `action`: quaternion XYZW | 3:7 | 16:20 |
| `action`: energy | 8 | 21 |
| `action`: thumbstick press | 9 | 22 |
| `action`: jaw | 10 | 23 |
| `action`: thumbstick X/Y | 11:13 | 24:26 |
| `observation.state`: engagement | 16 | 17 |
| `observation.state`: arm linkage | 20 | 21 |

The common translation/rotation motion scales are state columns 12/13. Preserve
the public CMR geometry, including its zero-translation-scale fallback of 1.0.
The action contract above defines endpoint poses, start-row jaws/axes and interval-max
buttons. Validate binary inputs as 0/1 before reduction. Invalid inputs must not
be silently coerced into neutral commands.

At 60 Hz, consecutive native timestamps must differ from 1/60 s by no more than
1e-4 s for a retained interval. Read episodes across parquet shard boundaries;
confirm contiguous episode-local frame indices and finite telemetry. Validate
nonzero quaternion norms. Video PTS must match the requested row-derived time,
including the episode's offset into a video shard (reference tolerance: half a
native frame). Decode C+1 images at stride 6; aspect-preserving bilinear resize
to fit 832×480, then black letterbox. Do not apply random video augmentation or
another reflection-padding pass to already-sized reference inputs.

History is the most recent valid suffix of up to 16 transitions, not 16 arbitrarily
selected earlier rows. Check invalid telemetry at native rate, not only transition
endpoints. For an accepted window, apply camera8000 across the retained history
and future together; do not shorten history merely to hide an above-threshold
camera command that was otherwise valid.

### Model-side batch

Before transforms: video `[3,C+1,480,832]`, future actions `[C,28]`, history actions
`[H,28]`, domain 25, both conditioning rates 10, mode `forward_dynamics` and the
fixed caption from the tutorial. After transforms: actions `[H+C,64]`, semantic
mask for only channels 0–27, all action rows conditioned, only the first video
frame conditioned, and sequence-plan offset `1-H`.

Use `max_samples_per_batch=1`, no token-driven multi-sample packing and accumulation
1. Retain H16-aware token counting; cap neither H16 nor the future implicitly to
fit an old C24 token budget. Inspect the actual collated batch and all modality
loss masks. Disable image augmentation and set caption dropout to zero for evaluation.

### Projector migration

Create a new model-only checkpoint and a migration receipt; do not overwrite the
public initializer. Map old channels `[0..19]` to `[0..9,14..23]` for domain 25.
Zero the newly introduced control contributions and padding for that domain.
Leave other domains and all unrelated tensors unchanged.

The domain-aware projections store flattened matrices. Inspect shapes and the
public `DomainAwareLinear` implementation before reshaping: input projection is
conceptually `[64,hidden]`, output projection `[hidden,64]`, not the usual assumption
that every linear weight is `[out,in]`. Map the output bias too. Apply consistent
handling to whichever weight variants are retained; this reference uses regular
weights, not EMA. Do not carry optimizer state across this semantic migration.

Required test: for normalized 20D vectors and their mapped 28D equivalents with
new controls zero, the pre/post-migration input embeddings must agree numerically.
Left and right arms must both pass. Verify unchanged tensors in other domains and
read back the saved checkpoint. Reject a missing/mismatched schema or normalizer
receipt even if all padded tensor shapes match.

### H16 is real input, once per rollout

Each independent sample includes its available contiguous genuine history
(`0 <= H <= 16`), followed by C24/C72 future actions. Stop history at episode
boundaries, invalid timestamps, disengagement or changed linkage/scaling.
Do not bridge invalid intervals or invent missing actions.

The combined action tensor is `[H+C, 64]` with sequence-plan
`action_start_frame_offset = 1-H`. All actions are clean conditioning in FDM.
For H16, history occupies temporal positions −15 through 0; future starts at 1.

TF trains causal attention against recorded video context. SF and autoregressive
inference process the initial image and explicit H16 prefix once; later blocks
receive new actions and cached preceding context. KV caching therefore matters
during **SF training as well as inference**. Do not prepend sliding H16 repeatedly
or change the anchor halfway through C72. Re-anchoring/resetting each C24 is a
separate protocol, not an equivalent implementation.

For H16/C72, the first future latent block must receive combined action rows
`16:20`, and the last `84:88`. Rows `0:16` are history. Test this explicitly:
a future-only slicer can silently treat history as commands.

The caption is fixed: **“Continue the ongoing surgical task.”** Training uses 10%
caption dropout and appends duration/FPS/timestamps and resolution metadata;
evaluation has no caption dropout. Captions do not vary with task labels or
trajectory content. No unpublished task annotations are needed for this recipe.

## 5. H16 through causal training and inference

The supplied `patch_framework` command checks every touched upstream file against
its pinned hash before applying the patch. It accepts an already applied patch,
but refuses mixed or otherwise modified files. Start from a fresh pinned checkout
instead of forcing it onto another framework version.

The extension carries explicit history lengths from `SequencePlan` through clean
generation data, packed sequences and attention metadata. It packs real action
prefix tokens without inventing preceding video frames. TF queries see history,
strictly preceding clean frames and the current noisy frame; they cannot see
future action blocks. Null frame-zero action values are zeroed **after** the real
history prefix. SF keeps the prefix when truncating/replaying samples and slices
future actions with the correct offset.

Every new training sample gets its own genuine H0..H16 prefix. In causal inference
it is prefetched once at frame zero; subsequent blocks receive new commands and
accumulate context in the KV cache. Do not prepend a sliding H16 window at every
block. Cache reset/re-anchoring is a different deployment protocol requiring its
own qualification. Starting without recorded history uses H0, not invented rows.

## 6. From smoke to a longer experiment

The six registrations are `chole_c24_a`, `chole_c24_b`, `chole_c72`, `chole_tf`,
`chole_sf4`, and `chole_sf2`. They read `COSMOS_DREAMS_SPEC`. The research-spec
generator supplies every path, budget and save interval; do not simply increase
the smoke counters and treat its unreviewed data as a research setup.

Use a **new research workspace**. Run bootstrap, download/review data and prepare
the separate A/B directories from Section 3. Then download training assets,
migrate the initializer and generate the specification:

```bash
export GPU_COUNT=8  # Total training world size for the chosen hardware.
"$PY" -m cosmos3_h_embodied.dreams.prepare --workspace "$WORKSPACE" --download-models
"$PY" -m cosmos3_h_embodied.dreams.migrate \
  --source "$WORKSPACE/assets/initializer/checkpoints/h16_50000/c3-h-e-n_model_bf16.safetensors" \
  --output "$WORKSPACE/migrated_h16_28d"
"$PY" -m cosmos3_h_embodied.dreams.research init \
  --workspace "$WORKSPACE" --gpus "$GPU_COUNT" --run-name research_run \
  --selection-a "$WORKSPACE/prepared_a" --selection-b "$WORKSPACE/prepared_b"
cd "$WORKSPACE/sources/cosmos-framework"
"$PY" -m cosmos3_h_embodied.dreams.preflight \
  --spec "$WORKSPACE/run.json" --output "$WORKSPACE/reports/preflight" --loaders
```

The generator checks reviewed status, action contracts and recording separation
across **both** selections. Preserve `run.json` with the manifests. Keep fixed
validation/test recordings and reviewed windows for comparing candidates. Split
checks cannot establish that the initializer never saw a recording.

| Stage | Stage updates | Periodic checkpoint interval |
|---|---:|---:|
| C24 A | 2,000 | 200 |
| C24 B | 2,000 | 200 |
| C72 | 4,400 | 400 |
| TF | 1,000 | 100 |
| SF4 | 300 | 60 |
| SF2 | 300 | 60 |

Intervals are editable before launch and are preserved by the research driver.
It runs one explicitly requested stage, then stops for offline validation and
selection. It does not enable an in-loop validation loader. The smoke's 1/2 and
6/12 restart probes remain exclusive to the smoke profile. Keeping all periodic
research checkpoints needs substantially more disk than the smoke.

| Stage | Student/model initialization | Fixed scoring teacher and fresh critic |
|---|---|---|
| C24 A | Migrated public H16 regular weights | — |
| C24 B | Selected C24 A | — |
| C72 | Selected C24 B | — |
| TF | Selected C72 | — |
| Four-step SF | Selected TF | Selected C72 |
| Two-step SF | Selected four-step student | The same selected C72 |

New stages load weights with fresh optimizers, schedulers and counters. A same-stage
resume loads full training and delivered-sample state. The driver tests both paths.
The critic is initialized afresh at the two-step stage; it is not carried over
from four-step SF. EMA is disabled, and the SF subclass explicitly requests
`net` rather than relying on the upstream loader's default `net_ema` or a filename
heuristic. Receipts reject unmapped checkpoints even though padded shapes match.

Key reference settings:

- 832×480, 10 Hz, BF16 mixed precision, FP32 master weights, full activation
  checkpointing; 48 GPUs, context parallelism 1, one sample per rank, accumulation 1.
- Teacher/TF: constant `3e-6`, FusedAdam, betas `(0.9, 0.999)`, weight decay `0.01`.
- SF: student `1e-6`, critic `2e-7`, betas `(0, 0.999)`, weight decay `0.01`;
  `student_update_freq=6`, no warmup, guidance 1, SID/fake-score loss scales 1.
- TF replay: one latent frame per chunk, clean KV gradients retained. SF replay:
  block size 1, no conditioned frames in replay loss, one backward gradient step.
- Four-step SDE ladder: `[1, 0.9375, 0.8333333333333334, 0.625]`.
  Two-step: `[1, 0.8333333333333334]`. The sampler appends terminal zero.
- Train only the reference generator/action-conditioning parameter groups, as
  listed in the reference JSON. Do not unfreeze action-output policy training.
- Fixed text: `Continue the ongoing surgical task.` with duration/FPS/timestamp
  and resolution metadata; 10% training text dropout, zero evaluation dropout.
  No phase/gesture labels or future-derived captions are conditioning.

Run and review stages in order. For a single-node allocation:

```bash
STAGE=c24_a  # Then c24_b, c72, tf, sf4, sf2, each after selecting its predecessor.
"$PY" -m cosmos3_h_embodied.dreams.smoke run --workspace "$WORKSPACE" --stages "$STAGE"
```

Evaluate periodic candidates on the **validation** split using Section 7. For
C24 use `--future 24`; for C72/TF/SF use `--future 72`. Review fixed-window videos
and interventions, then record the chosen checkpoint:

```bash
ITER=200  # Example C24 A candidate: choose an actually saved and reviewed iteration.
printf -v ITER_DIR 'iter_%09d' "$ITER"
CANDIDATE="$WORKSPACE/runs/cosmos3_h_dreams/research_run/$STAGE/checkpoints/$ITER_DIR"
"$PY" -m cosmos3_h_embodied.dreams.research select \
  --workspace "$WORKSPACE" --stage "$STAGE" --checkpoint "$CANDIDATE" \
  --evaluation "$WORKSPACE/review/${STAGE}_${ITER}/results.json" \
  --reason "Describe the fixed validation comparison and visual review supporting this choice"
```

Selection checks stamped stage/data/world-size/parent identity and completed
validation evidence for that exact checkpoint. It records hashes and updates the
next stage's parent. Selecting C72 fixes the scoring teacher for both SF stages.
Ancestry changes are rejected once downstream checkpoints exist. The next stage
refuses to run without required selections. Selecting the last iteration is
allowed but still requires evaluation; it is not implicitly the best checkpoint.
Reserve test data for final assessment, not selection.

The driver launches **one node**. For multinode training, generate the spec with
the total world size, validate the handoff, then launch the training entry point
on every node with your scheduler's rendezvous settings:

```bash
export COSMOS_DREAMS_SPEC="$WORKSPACE/run.json"
export IMAGINAIRE_OUTPUT_ROOT="$WORKSPACE/runs"
export HF_HOME="$WORKSPACE/cache/huggingface" TORCH_HOME="$WORKSPACE/cache/torch"
unset COSMOS_SMOKE
"$PY" -m cosmos3_h_embodied.dreams.research validate --workspace "$WORKSPACE" --stage "$STAGE"
# Set NNODES, NODE_RANK, GPUS_PER_NODE, MASTER_ADDR and MASTER_PORT for the allocation.
# Launch once per node; NNODES * GPUS_PER_NODE must equal run.json's world_size.
"$PY" -m torch.distributed.run \
  --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node="$GPUS_PER_NODE" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m cosmos_framework.scripts.train_experiment \
  --config cosmos_framework/configs/simulation/config.py \
  --experiment-module cosmos3_h_embodied.dreams.experiments -- "experiment=chole_$STAGE"
```

Repository/workspace/assets must resolve at identical paths on all nodes. For a
same-stage multinode resume, set `resume[STAGE]` in `run.json` to the latest complete
stamped checkpoint root before relaunching; preserve manifests, world size and
ancestry. A new stage leaves its resume entry absent. Qualify the chosen launcher
and topology; the numerical reference does not qualify a particular scheduler.

Select longer-run checkpoints using fixed validation examples, seeds and action
interventions in addition to denoising loss. A successful save or a decreasing
loss alone does not establish action following. Stop on nonfinite gradients,
contract mismatches, missing outputs or shape/PTS errors; do not skip bad batches
silently. Keep test examples separate from checkpoint-selection examples.

## 7. Evaluate and compare exported students

`run_smoke.sh` exports both students, reloads their actual exported tensors, and
runs recorded-action and energy-off rollouts. It uses the **validation split**
(`val`, the evaluator default), with one window per student and seed 28003.
This selection may have H0 and already-zero energy inputs; the default smoke
does not guarantee H16 coverage or a changed energy intervention. The fresh
latest 2026-11-05 rerun selected such a window: SF4/SF2 PSNR was 16.27/16.17 dB versus
21.45 dB for repeating the first frame. These minimally trained smoke outputs
establish integration, not useful-motion or release-quality performance.
For a richer comparison:

```bash
"$PY" -m cosmos3_h_embodied.dreams.evaluate \
  --checkpoint "$WORKSPACE/native/sf2/reload.dcp/model" \
  --config "$WORKSPACE/native/sf2/inference.yaml" \
  --spec "$WORKSPACE/run.json" --output "$WORKSPACE/review/sf2_extended" \
  --split test --samples 3 --seed 28003 --full-metrics \
  --modes recorded translation_hold rotation_hold pose_hold energy_off
```

Use the same selected starts and seeds to compare teachers and students. These
commands generate a model-only configuration without loading weights and evaluate
a periodic **regular-net** training checkpoint directly. No separate teacher
export is necessary:

```bash
STAGE=c72  # Also supports c24_a, c24_b, tf, sf4 and sf2.
ITER=400   # Choose an existing checkpoint.
FUTURE=72  # Use 24 for either C24 stage.
printf -v ITER_DIR 'iter_%09d' "$ITER"
CANDIDATE="$WORKSPACE/runs/cosmos3_h_dreams/research_run/$STAGE/checkpoints/$ITER_DIR"
"$PY" -m cosmos3_h_embodied.dreams.inference_config \
  --spec "$WORKSPACE/run.json" --stage "$STAGE" --output "$WORKSPACE/configs/$STAGE.yaml"
"$PY" -m cosmos3_h_embodied.dreams.evaluate \
  --checkpoint "$CANDIDATE/model" --config "$WORKSPACE/configs/$STAGE.yaml" \
  --spec "$WORKSPACE/run.json" --output "$WORKSPACE/review/${STAGE}_${ITER}" \
  --split val --future "$FUTURE" --samples 3 --seed 28003 --full-metrics \
  --modes recorded translation_hold rotation_hold energy_off
```

C72 uses bidirectional generation with 35 denoising steps; causal students use
their fixed four-/two-step schedule. Inference loads regular `net`, not EMA, and
the projected config excludes SF teacher/critic networks. `evaluation.json`
records checkpoint metadata/config/manifest identities, split, seed and completion.
`results.json` records each case and whether conditioning actually changed.
Compare like-for-like horizons; C24 selection is not a C72 quality comparison.

Recorded-action
metrics exclude the conditioning frame and report L1, PSNR, optional SSIM and
AlexNet LPIPS, plus a repeat-first-frame PSNR baseline. Shapes/frame counts must
match exactly. Scores are computed from decoded output MP4s, so codec settings
are part of this evaluator; do not compare them directly to historical raw-tensor
metrics without qualifying that difference. Intervention videos have no recorded
counterfactual ground truth and receive no reconstruction score.

Translation/rotation holds change only their named future pose channels. Energy
and thumbstick inputs, jaws and genuine history remain unchanged. Energy-off
changes only the two future energy channels. Correct responsiveness requires
visual review; reconstruction metrics cannot prove that a student uses actions.

Inspect `results.json` for the actual history length and compare the saved
`conditioning_actions.json` files before calling an intervention meaningful.
The qualified default example had H0 and zero energy in the recorded input, so
energy-off exercised the inference command but did not test a changed input.
To qualify exported H16 inference and energy responsiveness, select examples
with genuine H16 and nonzero recorded energy, respectively, and verify those
conditions in the outputs. A larger `--samples` value spreads examples over the
split, but does not guarantee either condition. A separate reported test passed
H16 inference for released students, not smoke-exported weights; its energy inputs
were also zero. These checks remain separate from completing the default smoke.

For a continuous 21.6-second AR probe, use `--future 216` on an explicitly selected
start with enough valid recorded future data. This uses a single initial anchor,
real prefix and continuously accumulated generated context; it does not reset to
recorded images at C72 boundaries. The command fails if the selected trajectory
is too short or violates the native filter. Choose valid starts rather than
looping/padding an action trajectory to manufacture a long test.

The public released weights are pinned in `assets.json`. Section 2 provides the
one-command workflow. Its individual commands, also usable with a custom
inference spec, are:

```bash
cd "$WORKSPACE/sources/cosmos-framework"
"$PY" -m cosmos3_h_embodied.dreams.prepare --workspace "$WORKSPACE" \
  --download-models --model-set students
for steps in 4 2; do
  "$PY" -m cosmos3_h_embodied.dreams.inference_config \
    --spec "$WORKSPACE/run.json" --stage "sf$steps" --output "$WORKSPACE/configs/sf$steps.yaml"
  "$PY" -m cosmos3_h_embodied.dreams.released --workspace "$WORKSPACE" --steps "$steps"
  "$PY" -m cosmos3_h_embodied.dreams.evaluate \
    --checkpoint "$WORKSPACE/released_sf${steps}.dcp/model" \
    --config "$WORKSPACE/configs/sf$steps.yaml" \
    --spec "$WORKSPACE/run.json" --output "$WORKSPACE/review/released_sf${steps}" \
    --split test --samples 3 --seed 28003 --full-metrics \
    --modes recorded translation_hold rotation_hold energy_off
done
```

This gives a measured comparison on your own fixed examples. The released students
already use CMR28; applying `migrate.py` to them is an error, and its initializer
hash guard rejects that mistake. This comparison does not establish that either
model's pretraining excluded these recordings.

Report per-example metrics, mean and variability across seeds, action-following
review, long-rollout drift, speed and memory under the same inference settings.
Few-update smoke outputs must not be called release-quality or close to the
released checkpoints on the strength of a successful process exit.

## 8. Reproduction scope and remaining qualification

This is an executable public adaptation of the reference method. Exact released
weights are not promised: the public initializer, reviewed data selection,
framework revision, numerical kernels and stochastic training may differ.
Private reviewed manifests and internal paths are not dependencies of this package.

Before claiming readiness for sustained training, retain successful evidence for:

1. Fresh installation, asset hashes and all six resolved configurations.
2. Actual C24/C72 batches including genuine history and correct 28D normalization.
3. TF/SF forward/backward with finite losses and gradients.
4. C24 and SF save/resume, including delivered sample order and optimizer phase.
5. Regular-weight student exports, reload, native rollouts and fixed-example review.

Your `WORKSPACE/reports/` captures evidence for your target hardware;
the reference result does not qualify a different GPU count or a memory-saving
configuration. Do not promote an unreviewed smoke dataset to research training
merely by increasing its update count.

### Container-import troubleshooting

<details>
<summary>Optional workaround for the reported Enroot curl error</summary>

This optional workaround applies only to Slurm/Pyxis/Enroot sites failing
before bootstrap with curl `(23) Failed to truncate file`.

The tested workaround copied the installed Enroot helper library into the
workspace and changed `common::curl` to write registry response bodies to a regular
temporary file before forwarding them to stdout. It then ran **explicit
`enroot import` in the batch shell**, and gave Pyxis the resulting local `.sqsh`
file. The system installation was not modified.

For the same symptom and matching helper implementation, the following prepares
that private change. Run on an allocated worker, with `WORKSPACE` set as in the
tutorial; adjust the installed library path if your site uses another location.
The exact-match guard stops if the installed helper differs; in that case ask
the site administrator for the appropriate fix rather than applying it blindly.

```bash
export ENROOT_LIBRARY_PATH="$WORKSPACE/site-enroot"
mkdir -p "$WORKSPACE"
# Use a new destination so an existing private library is not overwritten.
test ! -e "$ENROOT_LIBRARY_PATH" || exit 1
cp -a /usr/lib/enroot "$ENROOT_LIBRARY_PATH"
python3 - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["ENROOT_LIBRARY_PATH"]) / "common.sh"
text = path.read_text()
old = '''    { code=$(curl -o "/proc/self/fd/${stdout}" ${CURL_ERROUT+--stderr "/proc/self/fd/${stdout}"} -w '%{http_code}' "$@") || rv=$?; } {stdout}>&1'''
new = '''    local response_file
    response_file=$(mktemp -p "${ENROOT_TEMP_PATH}" curl-response.XXXXXXXX)
    { code=$(curl -o "${response_file}" ${CURL_ERROUT+--stderr "/proc/self/fd/${stdout}"} -w '%{http_code}' "$@") || rv=$?; } {stdout}>&1
    cat "${response_file}"
    rm -f "${response_file}"'''
assert text.count(old) == 1, "Installed Enroot helper differs; consult your site administrator"
path.write_text(text.replace(old, new))
PY
```

Then import in the same batch shell, using job-local runtime/temp storage. Merge
the cleanup into your launcher's existing exit trap if it already has one:

```bash
dreams_enroot_tmp=$(mktemp -d "${TMPDIR:-/tmp}/dreams-enroot.XXXXXXXX")
trap 'rm -rf -- "$dreams_enroot_tmp"' EXIT
export ENROOT_TEMP_PATH="$dreams_enroot_tmp/tmp"
export ENROOT_RUNTIME_PATH="$dreams_enroot_tmp/runtime"
export ENROOT_DATA_PATH="$dreams_enroot_tmp/data"
export ENROOT_CACHE_PATH="$WORKSPACE/cache/enroot"
export ENROOT_MAX_PROCESSORS=1 ENROOT_MAX_CONNECTIONS=1
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"
image="$WORKSPACE/pytorch-26.06-py3.sqsh"
enroot import --output "$image" 'docker://nvcr.io#nvidia/pytorch:26.06-py3'
# In the subsequent srun command, use --container-image="$image".
# Alternatively set DREAMS_CONTAINER_IMAGE="$image" for the tutorial Slurm wrapper.
```

The commands above generalize the reported helper/import procedure; this exact
documentation block has not been independently rerun. Keep any scheduler options,
mounts and prerequisite installation consistent with your site. Whole-node GPU
allocations may require omitting `--gpus-per-node`, as described in the tutorial.

</details>
