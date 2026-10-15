# Cosmos3-H-Embodied-Nano

**Cosmos3-H-Embodied-Nano** is [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) (16B) post-trained for
surgical robotics on the [Open-H Embodiment dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment).
One model predicts robot actions and endoscopic video for dVRK-style and other surgical robots, conditioned on a
short **history of past actions** (up to 16, "H16").

This repository contains everything needed to use and adapt it, built as a plugin package (`cosmos3_h_embodied`)
on an unmodified [cosmos-framework](https://github.com/NVIDIA/cosmos-framework):

- data loading for Open-H (and your own LeRobot data) with the H16 action history;
- scripts that rebuild the training data from the Hugging Face dataset;
- the training recipe of the released checkpoint, to fine-tune it further;
- dataset-driven evaluation and an HTTP policy server.

## The model

One model handles three tasks on 2.4 s clips (25 frames and 24 actions at 10 Hz, 256-pixel resolution tier):

| Mode | Input | Output |
|---|---|---|
| Forward dynamics | first frame, actions | video |
| Inverse dynamics | video | actions |
| Policy (`wam` in cosmos-framework) | first frame, text, action history | actions and video |

- **Actions.** Per arm and step: the end-effector pose relative to the current pose (translation as a world-frame
  offset, rotation in the current pose's frame as a 6D representation) and the gripper command, 20 values for a
  two-arm robot. Each dataset family is normalized
  with its own statistics (1st/99th percentiles to [-1, 1]); an embodiment ID selects the action input and output
  projections.
- **Action history.** Up to 16 past action steps, expressed in the current pose's frame, are given to the model as
  clean conditioning. At episode boundaries the history is shorter.
- **Text.** The task instruction, prefixed with a short description of the robot and setting for each dataset.
- **Training.** 50,000 iterations on 15 dataset families (55 dataset roots; 11 families are on Hugging Face), with
  the three modes mixed 0.4 / 0.2 / 0.4, on 32 GPUs. The last 3,063 iterations used the H16 recipe in this
  repository; the earlier ones an Open-H recipe without action history (see
  [training history](cosmos3_h_embodied/training_differences.md#0-training-history-of-the-released-checkpoint)).

Released weights: [`nvidia/Cosmos3-H-Embodied-Nano`](https://huggingface.co/nvidia/Cosmos3-H-Embodied-Nano) on Hugging Face,
file `model_bf16.pt` (the network weights in bf16).

## Install

1. Install cosmos-framework following its instructions. This package was validated against revision `7cfaf78`
   (`git checkout 7cfaf7854b2114eda5deb36250a7e8f00c64d69e`).
2. In the same environment, from this repository's root: `pip install -e .`

All commands below run from this repository's root with the cosmos-framework environment active.

Convert the released weights to a DCP checkpoint once; cosmos-framework loads checkpoints in this format:

```bash
python tools/convert_checkpoint_to_dcp.py --checkpoint model_bf16.pt -o <ckpt_dcp>
```

`<ckpt_dcp>/model` is used for evaluation and serving; `<ckpt_dcp>` is the training `checkpoint.load_path`.

The first evaluation or serving run downloads the VAE and the text tokenizer from Hugging Face. The datasets switch
the hub to offline mode (`HF_HUB_OFFLINE=1`) unless the variable is set, so start the first run with
`HF_HUB_OFFLINE=0`.

## Serve a policy

```bash
python -m cosmos_framework.scripts.export_config -o h16.yaml \
    --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt
python -m cosmos3_h_embodied.scripts.policy_server \
    --checkpoint-path <ckpt_dcp>/model --config-file h16.yaml \
    --normalizer dvrk_hamlyn_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json \
    --domain-name hamlyn --port 8000
```

`POST /predict` takes the current frame, the caption and up to 16 past actions (normalized); it returns 24 actions
(denormalized) and the predicted video. The request format is described in the module docstring of
`cosmos3_h_embodied/scripts/policy_server.py`. `--normalizer` and `--domain-name` select the action statistics and
the embodiment (see [Fine-tune on your own data](#fine-tune-on-your-own-data)).

## Evaluate

Evaluation runs the three modes on samples of a dataset family and scores them against the recorded data.
It needs the prepared data (next section):

```bash
python -m cosmos3_h_embodied.scripts.eval --no-guardrails \
    --checkpoint-path <ckpt_dcp>/model \
    --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt \
    --dataset.dataset dvrk_hamlyn --dataset.model-mode joint --dataset.dataset-split val \
    --dataset.num-samples 32 --seed 20260908 --num-steps 35 --guidance 6.0 -o outputs/eval
```

`--dataset.model-mode joint` runs forward dynamics, inverse dynamics and policy (`wam`). Per-sample `metrics.json`
and `metrics_aggregate.json` hold video PSNR (forward dynamics, policy) and action MSE (inverse dynamics, policy)
in normalized action space; history rows are excluded from the action MSE. On the `val` and `full` splits the eval
turns off the training-time video augmentation (`EVAL_DATASET_OVERRIDES` in
`cosmos3_h_embodied/configs/experiment_h16.py`), so the model is conditioned on and scored against clean frames.

## Fine-tune on Open-H

### 1. Prepare the data

Follow [`tools/data_prep/README.md`](tools/data_prep/README.md): download from Hugging Face, convert LeRobot
v2.1 → v3.0, apply the per-root repairs, install `modality.json` files, and build the CMR valid-window index into
`<data root>/indices/`. Then:

```bash
export OPENH_DATA_ROOT=<data root>     # mirrors the Hugging Face layout below Surgical/
```

### 2. Choose an experiment

| Experiment | Data | Captions | LR schedule |
|---|---|---|---|
| `openh_h16_task_prompt` | the Open-H roots on Hugging Face (11 families) | per-root robot description | constant 1e-5 after 100 warm-up steps (the rate training ended at) |
| `openh_h16_task_prompt_reference` | the exact 55-root training mixture (15 families; some roots are not public) | as trained (`caption_prefix_mode: legacy`) | as trained (linear 8e-5 → 1e-5 over 50,000 iterations) |
| `openh_h16_task_prompt_smoke` | 5 families, window cap, per-sample family mixing | per-root | constant |

The experiments use **rank-based family mixing**: each GPU rank serves one dataset family, and ranks are allocated
by mixture ratio. This needs at least as many ranks as families (11 for the Hugging Face mixture). With fewer ranks,
set `shard_across_workers=false` on the dataset (per-sample mixing: each dataloader worker draws a family per sample).

### 3. Train

```bash
export IMAGINAIRE_OUTPUT_ROOT=<checkpoint directory>   # default: /tmp/imaginaire4-output
python -m cosmos_framework.scripts.export_config -o h16.yaml \
    --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt
torchrun --nproc_per_node=8 --nnodes=<N> ... -m cosmos_framework.scripts._train \
    --config-file h16.yaml -o outputs \
    --config-overrides checkpoint.load_path=<ckpt_dcp> model.config.parallelism.data_parallel_shard_degree=<world size>
```

The recipe was trained on 32 GPUs (FSDP, 2-step gradient accumulation, 5,040 tokens per packed sequence).
Checkpoints (model, EMA and optimizer state, several times the size of the released weights) are written to
`$IMAGINAIRE_OUTPUT_ROOT/cosmos3_h_embodied/openh_h16/<experiment>/checkpoints/iter_<N>/`; `-o outputs` holds the logs
and a `job` link to that directory. Restarting the same command resumes from the latest checkpoint, with the action
dataloader position and RNG state restored per worker.

Smoke test on one GPU with a 2-layer model, on a few episodes per dataset from `tools/download_openh_samples.py`:

```bash
python -m cosmos_framework.scripts.export_config -o smoke.yaml \
    --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt_smoke
COSMOS_SMOKE=1 torchrun --nproc_per_node=1 -m cosmos_framework.scripts._train --config-file smoke.yaml -o outputs \
    --config-overrides model.config.parallelism.data_parallel_shard_degree=1 model.config.compile.enabled=false
```

## Fine-tune on your own data

Data from a bimanual (or single-arm) surgical robot with Cartesian end-effector kinematics fits the
`GeneralizedDvrkHistoryDataset` loader. Prepare it once, then fine-tune for the mode you need.

### Prepare your dataset

1. **LeRobot v3.0 format**, one root per task, below `OPENH_DATA_ROOT` (for example
   `$OPENH_DATA_ROOT/my_lab/task_a`); the loaders sample clips at 10 Hz. The state holds each arm's measured
   end-effector pose, the action each arm's gripper command (and, with `pose_source="action"`, the commanded pose).
2. **`meta/modality.json`** that names the columns per arm, in both the `state` and the `action` block:
   `<arm>_pose` (7 columns: position + quaternion, set `quaternion_order` to `xyzw` or `wxyz`; or 6 columns:
   position + Euler XYZ in radians) and `<arm>_gripper` (1 column, optional). See
   `tools/data_prep/modality/hamlyn/peg_transfer/modality.json` for an example.
3. **An embodiment.** The model has 32 embodiment slots for its action projections, all of them assigned
   (`EMBODIMENT_TO_DOMAIN_ID` in `cosmos3_h_embodied/data/domain_utils.py`). Reuse the embodiment of the most similar
   robot: for example, `hamlyn` or `dvrk_obuda` for a dVRK with two patient-side manipulators. Captions are
   prefixed with a robot and setting description; add one for your data to `EMBODIMENT_CAPTION_CONTEXT` in the same
   file (key `my_lab` below).
4. **A dataset family.** Append it to `FAMILIES` in `cosmos3_h_embodied/configs/openh_datasets.py`:

   ```python
   Family(
       "my_lab",
       1.0,  # sampling ratio relative to the other families of an experiment
       GeneralizedDvrkHistoryDataset,
       (Root("my_lab/task_a", "my_lab"), Root("my_lab/task_b", "my_lab")),  # path, caption key
       _dvrk(
           "hamlyn",  # embodiment
           "my_lab_backward_anchored_worldtrans_rot6d.json",  # action statistics (next step)
           "observation.images.left",
           quaternion_order="wxyz",
       ),
   ),
   ```

5. **Action statistics** of the family with
   [`tools/compute_action_normalizer.py`](tools/compute_action_normalizer.py), written to the family's normalizer
   file under `cosmos3_h_embodied/data/normalizers/` (the procedure of the released checkpoint's most recent
   statistics files, which it reproduces exactly):

   ```bash
   python tools/compute_action_normalizer.py --family my_lab
   ```

Each tutorial below registers an experiment, trains from the released checkpoint (`<ckpt_dcp>`, see
[Install](#install)), and uses the result. Experiments are registered at the end of
`cosmos3_h_embodied/configs/experiment_h16.py` with `register_experiment`, which builds them like
`openh_h16_task_prompt` (the `continue` learning rate: constant 1e-5, the rate the released checkpoint finished its
training at). `shard_across_workers=False` mixes families per sample, needed when there are fewer GPU ranks than
families. Checkpoints are written to `$IMAGINAIRE_OUTPUT_ROOT/cosmos3_h_embodied/openh_h16/<experiment>/checkpoints/`
(see [Train](#3-train)).

### Tutorial: forward dynamics (world model)

Train the model to predict video from the first frame and an action sequence:

```python
register_experiment("my_lab_fd", families=["my_lab"], shard_across_workers=False, mode="forward_dynamics")
```

```bash
python -m cosmos_framework.scripts.export_config -o my_lab_fd.yaml \
    --config-file cosmos3_h_embodied/configs/config.py --experiment my_lab_fd
torchrun --nproc_per_node=8 -m cosmos_framework.scripts._train --config-file my_lab_fd.yaml -o outputs \
    --config-overrides checkpoint.load_path=<ckpt_dcp> model.config.parallelism.data_parallel_shard_degree=8
```

Score the predicted video on held-out episodes (PSNR against the recorded video):

```bash
python -m cosmos3_h_embodied.scripts.eval --no-guardrails \
    --checkpoint-path $IMAGINAIRE_OUTPUT_ROOT/cosmos3_h_embodied/openh_h16/my_lab_fd/checkpoints/iter_<N>/model \
    --config-file cosmos3_h_embodied/configs/config.py --experiment my_lab_fd \
    --dataset.dataset my_lab --dataset.model-mode forward_dynamics --dataset.dataset-split val \
    --dataset.num-samples 32 --num-steps 35 --guidance 6.0 -o outputs/eval_my_lab_fd
```

### Tutorial: policy

Train the model to predict the next 24 actions (and the video) from the current frame, the task text and the action
history:

```python
register_experiment("my_lab_policy", families=["my_lab"], shard_across_workers=False, mode="policy")
```

Export the config and train as above with `my_lab_policy`. Then serve it:

```bash
python -m cosmos3_h_embodied.scripts.policy_server \
    --checkpoint-path $IMAGINAIRE_OUTPUT_ROOT/cosmos3_h_embodied/openh_h16/my_lab_policy/checkpoints/iter_<N>/model \
    --config-file my_lab_policy.yaml \
    --normalizer my_lab_backward_anchored_worldtrans_rot6d.json --domain-name hamlyn --port 8000
```

`--dataset.model-mode wam` in the eval command above scores the policy offline (action MSE). If your data labels
failed episodes (`splits` in `meta/info.json`), the default joint training keeps them out of the policy mode; a
policy-only run does not, so list them in the root's `meta/exclude_episodes.json` (a JSON list of episode indices).

Training on a single mode specializes the model. To keep its other abilities, keep `mode="joint"` (the recipe mixes
forward dynamics, inverse dynamics and policy 0.4 / 0.2 / 0.4, set by `joint_mode_weights`) or add Open-H families to
`families`.

## Repository layout

| Path | What |
|---|---|
| `cosmos3_h_embodied/data/` | Open-H datasets (generalized dVRK, CMR clinical), H16 history, normalization, transforms, family mixing, history-matched sequence packing |
| `cosmos3_h_embodied/data/normalizers/` | Action normalization statistics of the released checkpoint (15 families) |
| `cosmos3_h_embodied/data/exclusions/` | Episodes excluded from training (302), keyed by dataset path |
| `cosmos3_h_embodied/model/` | `HistoryOmniMoTModel`: action loss averaged over the noised action rows only |
| `cosmos3_h_embodied/callbacks/` | Dataloader resume state (per-worker sample index and RNG state) |
| `cosmos3_h_embodied/configs/` | Dataset mixture and experiments (`config.py` registers them with cosmos-framework) |
| `cosmos3_h_embodied/scripts/eval.py` | Dataset-driven evaluation |
| `cosmos3_h_embodied/scripts/policy_server.py` | HTTP policy server with action history |
| `tools/data_prep/` | Rebuild the training data from the Hugging Face dataset ([README](tools/data_prep/README.md)) |
| `tools/convert_checkpoint_to_dcp.py` | Released `.pt` weights → DCP checkpoint |
| `tools/compute_action_normalizer.py` | Action normalization statistics for your own dataset |
| `tools/download_openh_samples.py` | Download a few episodes per dataset for a local smoke test |

## Reproducibility

The training data of the released checkpoint can be rebuilt from Hugging Face for 11 of its 15 dataset families;
some roots, captions and one video stream were never published. The code was validated against the original
training code on the original data: identical samples, matching training loss, and matching evaluation metrics.
[`cosmos3_h_embodied/training_differences.md`](cosmos3_h_embodied/training_differences.md) lists every difference
between the released checkpoint's training run and what this repository reproduces. Design decisions and validation
results are recorded in [`design_and_decisions_log.md`](design_and_decisions_log.md).

## Citation and license

```bibtex
@techreport{nvidia2026cosmos3,
  title  = {Cosmos 3: Omnimodal World Models for physical AI},
  author = {{NVIDIA Cosmos Team}},
  year   = {2026},
  url    = {https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf}
}
```

Source code and models are released under [OpenMDW-1.1](https://openmdw.ai/license/1-1/). The Open-H datasets
carry their own licenses; see the dataset card.
