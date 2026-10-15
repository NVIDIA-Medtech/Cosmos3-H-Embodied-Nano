# Cosmos3-H-Embodied-Nano: Open-H surgical robotics on cosmos-framework

`cosmos3_h_embodied` adds the Open-H surgical robotics recipe of **Cosmos3-H-Embodied-Nano** to
[cosmos-framework](https://github.com/NVIDIA/cosmos-framework): data loading for the
[Open-H Embodiment dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment),
the **H16 action history** (up to 16 past actions given to the model as clean conditioning), training,
evaluation and a policy server. It is a plugin package: cosmos-framework is used unmodified.

One model handles three tasks on 2.4 s clips (25 frames and 24 actions at 10 Hz, 256-pixel resolution tier):

| Mode | Input | Output |
|---|---|---|
| Forward dynamics | first frame, actions | video |
| Inverse dynamics | video | actions |
| Policy (`wam` in cosmos-framework) | first frame, text, action history | actions and video |

Design decisions, and every place where this code deliberately differs from the training code of the
released checkpoint, are recorded in [`design_and_decisions_log.md`](../design_and_decisions_log.md).

## Contents

| Path | What |
|---|---|
| `data/` | Open-H datasets (generalized dVRK, CMR clinical), H16 history, normalization, transforms, family mixing, history-matched sequence packing |
| `data/normalizers/` | Action normalization statistics of the released checkpoint (15 families) |
| `data/exclusions/` | Episodes excluded from training (302), keyed by dataset path |
| `model/` | `HistoryOmniMoTModel`: action loss averaged over the noised action rows only |
| `callbacks/` | Dataloader resume state (per-worker sample index and RNG state) |
| `configs/` | Dataset mixture and experiments (`config.py` registers them with cosmos-framework) |
| `scripts/eval.py` | Dataset-driven evaluation: video PSNR, action MSE on predicted rows |
| `scripts/policy_server.py` | HTTP policy server with action history |
| `../tools/data_prep/` | Rebuild the training data from the Hugging Face dataset ([README](../tools/data_prep/README.md)) |
| `../tools/convert_checkpoint_to_dcp.py` | Released `.pt` weights → DCP checkpoint for training |
| `../tools/download_openh_samples.py` | Download a few episodes per dataset for a local smoke test |

## Install

1. Install cosmos-framework following its instructions. This package was validated against revision
   `7cfaf78` (`git checkout 7cfaf7854b2114eda5deb36250a7e8f00c64d69e`).
2. In the same environment, from this repository's root: `pip install -e .`

Commands below run from this repository's root with the cosmos-framework environment active.

## Experiments

| Experiment | Data | Captions | LR schedule |
|---|---|---|---|
| `openh_h16_task_prompt` | Open-H roots available on Hugging Face | per-root caption context | constant 1e-5 after 100 warm-up steps (the rate training ended at) |
| `openh_h16_task_prompt_reference` | the exact 55-root training mixture (some roots are not public) | as trained (`caption_prefix_mode: legacy`) | as trained (linear 8e-5 → 1e-5 over 50,000 iterations) |
| `openh_h16_task_prompt_smoke` | 5 families, window cap, per-sample family mixing | per-root | constant |

The default experiment uses **rank-based family mixing**: each GPU rank serves one dataset family, with
ranks allocated by mixture ratio, which needs at least as many ranks as families (11 for the Hugging Face mixture, 15 for the reference). With fewer ranks, set
`shard_across_workers=false` on the dataset (per-sample mixing: each worker draws a family per sample).

Not on Hugging Face (used by the reference experiment only): SutureBot canonical roots ×3, SutureBot
mono, USTC (7 roots), CMR dry-box peg transfer, STAR-IL (261 episodes), the wound-closure clean-video
twin, and the CMR clinical per-frame instruction texts (without them CMR captions fall back to the
procedure description). See the design log, section 6.

## 1. Prepare data

Follow [`tools/data_prep/README.md`](../tools/data_prep/README.md): download from Hugging Face, convert
LeRobot v2.1 → v3.0, apply the per-root repairs, install `modality.json` files, and build the CMR
valid-window index into `<data root>/indices/`. Then:

```bash
export OPENH_DATA_ROOT=<data root>     # mirrors the Hugging Face layout below Surgical/
```

## 2. Checkpoint

```bash
python tools/convert_checkpoint_to_dcp.py --checkpoint <released .pt> -o <ckpt_dcp>
```

`<ckpt_dcp>/model` is used for evaluation and serving; `<ckpt_dcp>` is the training `checkpoint.load_path`.

## 3. Train

```bash
python -m cosmos_framework.scripts.export_config -o h16.yaml \
    --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt
torchrun --nproc_per_node=8 --nnodes=<N> ... -m cosmos_framework.scripts._train \
    --config-file h16.yaml -o outputs \
    --config-overrides checkpoint.load_path=<ckpt_dcp> model.config.parallelism.data_parallel_shard_degree=<world size>
```

The recipe was trained on 32 GPUs (FSDP, 2-step gradient accumulation, 5,040 tokens per packed sequence).
Resuming is automatic from `outputs/.../checkpoints`; the action dataloader position and RNG state are
restored per worker.

Smoke test on one GPU with a 2-layer model (local samples from `tools/download_openh_samples.py`):

```bash
python -m cosmos_framework.scripts.export_config -o smoke.yaml \
    --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt_smoke
COSMOS_SMOKE=1 torchrun --nproc_per_node=1 -m cosmos_framework.scripts._train --config-file smoke.yaml -o outputs \
    --config-overrides model.config.parallelism.data_parallel_shard_degree=1 model.config.compile.enabled=false
```

## 4. Evaluate

```bash
python -m cosmos3_h_embodied.scripts.eval --no-guardrails \
    --checkpoint-path <ckpt_dcp>/model \
    --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt \
    --dataset.dataset dvrk_hamlyn --dataset.model-mode joint --dataset.dataset-split val \
    --dataset.num-samples 32 --seed 20260908 --num-steps 35 --guidance 6.0 -o outputs/eval
```

The datasets switch the Hugging Face hub to offline mode (`HF_HUB_OFFLINE=1`) unless it is set; the first
evaluation or serving run downloads the VAE and tokenizer, so start it with `HF_HUB_OFFLINE=0`.

On the `val` and `full` splits the eval turns off the training-time video augmentation
(`EVAL_DATASET_OVERRIDES` in `configs/experiment_h16.py`), so the model is conditioned on and scored against
clean frames.

`--dataset.model-mode joint` runs forward dynamics, inverse dynamics and policy (`wam`). Per-sample
`metrics.json` and `metrics_aggregate.json` hold video PSNR (forward dynamics, policy) and action MSE
(inverse dynamics, policy) in normalized action space; history rows are excluded from the action MSE.

## 5. Serve a policy

```bash
python -m cosmos3_h_embodied.scripts.policy_server \
    --checkpoint-path <ckpt_dcp>/model --config-file h16.yaml \
    --normalizer dvrk_hamlyn_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json \
    --domain-name hamlyn --port 8000
```

`POST /predict` takes the current frame, the caption, and up to 16 past actions (normalized); it returns
24 actions (denormalized) and the predicted video. See the module docstring for the request format.

