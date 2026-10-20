# From an H16 world model to four-step and two-step students

This guide explains how an H16 action-conditioned world model can be adapted to your data and distilled into four-step and two-step students for interactive video generation. It describes the purpose of each training stage, data preparation and evaluation. It is a conceptual walkthrough, not yet an executable distillation tutorial.

> **Implementation status.** This repository supports H16 bidirectional teacher fine-tuning and evaluation. Causal teacher-forcing and Self-Forcing training are not yet supported by [cosmos-framework](https://github.com/NVIDIA/cosmos-framework). **The tutorial will be updated with runnable distillation instructions once cosmos-framework supports them.** Until then, the distillation sections below describe the reference recipe, not an executable end-to-end pipeline. A 28D CMR control adapter is also not provided by the current 20D loader.

For the earlier model family, see the [Cosmos-H-Surgical-Simulator training tutorial](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator/blob/main/docs/tutorial_teacher_training_and_self_forcing.md). Its commands, action representation and trajectory-cache stage are not interchangeable with this Cosmos3 recipe.

## 1. Understand the task and terminology

The distilled model is a **forward-dynamics video model**: it receives an initial image, supplied robot actions, a task description and optional genuine action history, then predicts video. It does not choose the controller commands.

| Term | Meaning in this recipe |
|---|---|
| C24 | 24 future video frames/actions at 10 Hz: 2.4 seconds, plus the initial image |
| C72 | 72 future video frames/actions at 10 Hz: 7.2 seconds, plus the initial image |
| H16 | Up to 16 preceding action steps supplied as clean conditioning |
| TF | Causal teacher-forcing training using recorded context |
| SF | Self-Forcing training using the model's generated context |
| Four-step / two-step | Denoising evaluations per generated block, **not** the number of actions or video frames |

The reference causal model generates blocks of four future video frames, corresponding to one latent temporal frame. At 10 Hz, a block spans 0.4 seconds. A C72 rollout therefore spans multiple autoregressive blocks; it is not 72 separate denoising steps.

The reference uses H16 as an **initial history prefix**. Do not insert a fresh 16-action prefix at every block within the same cached rollout. If beginning a new independent window, build its history from the actual preceding commands according to the loader's coordinate contract. With a standalone starting image and no history, use H0 rather than inventing past commands.

## 2. Prepare your data and action contract

Start with the repository's [installation and custom-data instructions](../README.md#fine-tune-on-your-own-data) and [Open-H preparation guide](../tools/data_prep/README.md). The loaders accept LeRobot data with synchronized video, poses, gripper commands and modality metadata.

Before training:

1. Identify the pose source, quaternion ordering, coordinate frames, units and timestamp alignment explicitly.
2. Use the loader's pose transforms and normalization consistently in training and deployment. Read the [action-space specification](README_ACTION_SPACE.md); do not substitute raw controller Euler angles for its six rotation channels.
3. Reserve validation and test recordings before selecting windows. Where case identity is available, split by case as well. Protect recordings used by any warm-start ancestor when constructing a genuinely held-out test set.
4. Fit new normalization statistics on training data only. For an existing compatible checkpoint, preserve its normalization unless deliberately adapting the input mapping.
5. Check initial images, future actions and H16 alignment through the actual loader and collation path, not only the manifest.

### Example: curate two-instrument surgical scenes

For a bimanual surgical world model, use Astra-assisted visual selection with human review to target **exactly two visible instruments** matching the controlled arms. A cholecystectomy example uses a left Cadiere Grasper and right Monopolar Hook. Check that both controlled arms are engaged as well: engagement telemetry and visible instrument count are separate checks.

Record accepted intervals, review coverage and unresolved cases. Sparse inspection should not be represented as exhaustive frame-by-frame verification. The objective is to align the visible scene with the instruments represented by the action input.

A useful additional training-only filter limits camera commands using native thumbstick telemetry. For each hand, compute the raw two-axis magnitude:

`radius = sqrt(thumbstick_x² + thumbstick_y²)`

Accept a window only when **both radii are at most 8,000 on every native 60 Hz row**, from the earliest retained H16 history row through the future endpoint, inclusive. Checking only the 10 Hz sampled rows can miss intervening commands.

- **8,000 is a raw CMR telemetry threshold**, not a normalized model-space value or a portable threshold for other controllers.
- Keep accepted action values unchanged; this is a sample-selection rule, not an instruction to zero controls.
- The rule targets small camera commands. It does not guarantee a stationary camera or rule out visual motion caused by instrument/tissue interaction.
- Keep energy activations and the existing action semantics. The camera filter is not an energy-off filter.

Count unique covered video time as well as accepted window starts. Overlapping windows are not independent clips; summing their durations overstates dataset coverage. Short- and long-horizon training windows may cover different amounts of usable data.

### Additional control channels

A 28D surgical extension can retain the pretrained pose/gripper representation and add eight controls: **two energy values, four thumbstick axes and two thumbstick buttons**, across the two hands. Thus, there are six thumbstick dimensions, not eight. The 28 meaningful dimensions are padded to the model's 64-dimensional action input.

The current Open-H loader supplies the original two-arm 20D pose/gripper representation. Merely changing a dimension count does not implement the 28D extension. A compatible adapter must preserve pose transforms, normalization, channel ordering, control timing and the checkpoint's action-projection mapping. Equal padded tensor shapes do not prove semantic compatibility. Until that adapter is available, use the supported action contract for your own-data experiments rather than assuming that the extra controls are already supported.

## 3. Establish a teacher-training baseline

Follow the main README to install the pinned compatible `cosmos-framework` revision, prepare your dataset family and convert the released weights to DCP. Register a forward-dynamics experiment as described in the [forward-dynamics tutorial](../README.md#tutorial-forward-dynamics-world-model).

For a registered `my_lab_fd` experiment, the entry point is:

```bash
python -m cosmos_framework.scripts.export_config -o my_lab_fd.yaml \
    --config-file cosmos3_h_embodied/configs/config.py --experiment my_lab_fd
```

Train and evaluate with the commands in that tutorial. This establishes a loader-verified **teacher baseline**. It does not automatically configure the higher-resolution distillation recipe below. In particular, the Open-H defaults of this repository use a 256-pixel tier and packed sequences; the illustrative settings below use 832 × 480 and one example per rank.

Keep the exported configuration, dataset version, normalization statistics and parent checkpoint identity with every run. Distinguish a new stage initialized from weights from a full-state continuation restoring optimizer, scheduler, RNG and data position.

### Initialize from the released checkpoint

Use the released **task-prompt Open-H H16 50k checkpoint** linked in the [main README](../README.md#the-model) as the starting point for your own-data adaptation. Convert its weights to DCP using the conversion tool before teacher training.

Begin C24 adaptation on your own prepared dataset, then select and qualify a teacher before proceeding to longer horizons and distillation. Your dataset, initialization and intended deployment determine the appropriate training budget.

## 4. How the distillation stages fit together

```text
Released Open-H H16 checkpoint
  → short-horizon bidirectional teacher (C24)
  → long-horizon bidirectional teacher (C72)
  → causal teacher forcing (TF)
  → four-step Self-Forcing student
  → optional two-step Self-Forcing student
```

| Stage | Initialization and purpose |
|---|---|
| C24 teacher | Start from the released checkpoint and adapt bidirectional video prediction to your domain. Short clips provide an initial check of appearance and action conditioning. |
| C72 teacher | Start from the adapted C24 teacher and train on longer clips to learn temporal behavior over a larger window. |
| TF | Start from the selected C72 teacher and train with causal attention and recorded conditioning, preparing the model for sequential generation. |
| Four-step SF | Start the student from the causal TF checkpoint. Train with its own generated context to address the mismatch between recorded training context and autoregressive inference. A fixed C72 scoring teacher and a trainable critic provide distribution-matching supervision. |
| Two-step SF | Start from the four-step student, retain the scoring teacher and train with a two-step sampling schedule. This further reduces the denoising budget; evaluate the quality/latency tradeoff separately. |

This recipe proceeds directly from TF to SF, without a separate intermediate distillation stage. That does not remove distribution-matching supervision from SF itself. Exact configuration names and training commands will be added once cosmos-framework supports these stages.

An **illustrative update budget** is C24 2,000 → C72 4,400 → TF 1,000 → four-step SF 300 → two-step SF 300. These values describe a useful starting scale, not a validated schedule for every dataset or for every pretrained initialization. Select continuation budgets from validation rather than treating these numbers as fixed requirements.

### Reference settings

| Setting | Reference value |
|---|---|
| Hardware and batch | 6 nodes × 8 H100 GPUs; one example per rank; accumulation 1; global batch 48 |
| Video | 832 × 480, 10 Hz |
| Context | H16; C24 then C72 |
| Compute precision | BF16 mixed precision; full activation checkpointing |
| Teacher and TF learning rate | Constant 3e-6 |
| SF student / critic learning rates | 1e-6 / 2e-7 |
| SF update ratio | One student update per six logical iterations |
| Four-step sampling | Shift-5 schedule, four denoising steps per generated block |
| Two-step sampling | Shift-5 schedule `[1, 5/6, 0]`, two denoising steps per generated block |

**Each SF300 stage means 50 student updates and 250 critic updates**, not 300 student optimizer updates. Preserve this distinction when comparing compute budgets or implementing the recipe.

Treat these budgets as a reference, not a convergence guarantee. Monitor held-out generation, action response and longer-rollout drift alongside loss. More data does not automatically justify an identical budget, and a lower denoising loss does not guarantee a better distilled student.

## 5. Qualification before a full run

Once distillation training is supported, qualify the complete path on a small dataset before allocating a full training run:

- Verify a forward/backward pass at C24, C72, TF and each SF sampling budget; require finite losses and gradients.
- Exercise both student and critic optimizer updates, then save and resume. Check stage iteration counts and restored state.
- Verify causal masks, action/frame alignment, initial H16 handling and conditioning across generated blocks.
- Check that action-projection mappings and normalization survive checkpoint conversion and export.
- Compare exported inference against training validation with matched inputs, sampling settings and seeds.

The H16 teacher and this recipe description alone do not establish those checks for a new distillation implementation.

## 6. Evaluate and deploy both students

Compare the selected teacher, four-step student and two-step student on identical recorded-action examples. Fix initial images, histories, trajectories and seeds, while recognizing that different sampling budgets consume randomness differently.

Use complementary evaluations:

1. **Reconstruction:** L1, PSNR, SSIM and LPIPS against recorded video; include dynamic-region measurements and a repeat-first-frame baseline.
2. **Temporal stability:** continuous rollouts of at least 30 seconds, using generated conditioning between blocks rather than resetting to recorded frames. Inspect early and late segments separately.
3. **Action response:** isolated translation, positive/negative rotation about each axis, jaw motion and nominal/energy-off pairs, with other inputs held fixed.
4. **Interactive use:** test the actual controller-to-model transforms and four-/two-step schedules in the intended accelerated runtime, such as FlashDreams.

Reconstruction metrics alone cannot establish action fidelity. A plausible scene that ignores commands is a failure; a stable camera with an immobile tool is not evidence that rotation control was learned. The reference camera filter was motivated by unwanted camera motion under isolated tool rotations, but comparative rollouts cannot by themselves attribute all improvements to that filter.

Keep validation for checkpoint selection and a separate final test set where feasible. Document ancestry or review limitations when strict independence cannot be established, and label training-set reconstructions as such.

For deployment, package weights with the action layout, normalization, resolution, frame rate, history convention and sampling schedule. A two-step student is separately trained; merely requesting two steps from four-step weights does not replace training a two-step student. Two steps also do not guarantee twice the end-to-end speed: decoding, caching and runtime overhead contribute to latency.

## Next update

Once `cosmos-framework` supports distillation training, this tutorial will add the supported version, executable C24/C72/TF/SF configurations, four-step → two-step continuation commands, and tested export/deployment instructions. Until then, use the current teacher-training documentation and the reference methodology above without treating the missing distillation stages as runnable commands.
