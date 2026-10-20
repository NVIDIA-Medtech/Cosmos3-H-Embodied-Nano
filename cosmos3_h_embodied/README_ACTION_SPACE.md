# Cosmos3-H-Embodied-Nano: Action Space for Open-H Multi-Embodiment Training

This document describes the action representation of Cosmos3-H-Embodied-Nano: how each robot's recorded kinematics
become the action rows the model consumes and predicts, how embodiments with different arms share one 64-wide action
vector, how actions are normalized, and how the dataset families are mixed in training. It also compares the
representation with that of [Cosmos-H-Surgical-Simulator](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator)
(see its [action space document](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator/blob/main/scripts/README_ACTION_SPACE.md)).

---

## Table of Contents

1. [Design Motivation](#design-motivation)
2. [The Action Vector](#the-action-vector)
   - [Per-Arm Layout](#per-arm-layout)
   - [Relative Pose Representation](#relative-pose-representation)
   - [Time Alignment and Action History](#time-alignment-and-action-history)
3. [Embodiment Mapping](#embodiment-mapping)
   - [Summary Table](#summary-table)
   - [Per-Embodiment Notes](#per-embodiment-notes)
4. [CMR Versius Clinical](#cmr-versius-clinical)
5. [Normalization](#normalization)
6. [Padding, Channel Masking and Embodiment Projections](#padding-channel-masking-and-embodiment-projections)
7. [Actions per Training Mode](#actions-per-training-mode)
8. [Data Mixture](#data-mixture)
9. [Training Statistics](#training-statistics)
10. [Comparison with Cosmos-H-Surgical-Simulator](#comparison-with-cosmos-h-surgical-simulator)
11. [Source Code Pointers](#source-code-pointers)

---

## Design Motivation

Cosmos3-Nano generates video and actions jointly. The Open-H post-training covers 15 dataset families from 13 robot
embodiments: dual-arm dVRK systems with grippers, a four-arm system without grippers, single-arm systems, and the CMR
Versius clinical robot. A shared representation is needed because:

1. **Packed batches**: samples of different embodiments are packed into one sequence per GPU, so every action row
   has the same width.
2. **Transfer across embodiments**: every arm is described the same way (end-effector motion relative to the current
   pose, plus a gripper command), so what the model learns about one dVRK applies to the others.
3. **Embodiment-specific projections**: an embodiment ID selects that robot's action input and output projections,
   so the same channel can mean different things on different robots.

The model's action width is `max_action_dim = 64`; every embodiment's action is zero-padded to it.

---

## The Action Vector

### Per-Arm Layout

Each action row describes one time step of every arm, arms concatenated in the order of the dataset's
`meta/modality.json`:

| Dims (per arm) | Size | Content |
|---|---|---|
| 0–2 | 3 | Translation of the end effector relative to the current pose, in the world frame |
| 3–8 | 6 | Rotation relative to the current orientation, as 6D rotation |
| 9 | 1 | Gripper command (absolute), if the arm has a gripper |

A dual-arm robot with grippers therefore has a 20-dimensional action (`[arm1: 10 | arm2: 10]`), padded to 64.

### Relative Pose Representation

All families use the pose convention `backward_anchored_world_trans`. For the window anchor (the current frame,
`t = 0`) with pose `(R_0, t_0)` and an endpoint `t = i` with pose `(R_i, t_i)`:

| Component | Value |
|---|---|
| Translation (3D) | `t_i − t_0`: displacement from the anchor, in the frame the poses are recorded in (not rotated into the anchor frame) |
| Rotation (6D) | `R_0ᵀ · R_i`: orientation relative to the anchor, expressed in the anchor's own frame |

Every row is relative to the same anchor (not frame-to-frame differences). The 6D rotation is the first two
**columns** of the rotation matrix, flattened as `[r00, r10, r20, r01, r11, r21]` (the convention of Zhou et al.,
"On the Continuity of Rotation Representations in Neural Networks").

Source poses come in three formats, set per family:

| Pose columns | Format | Setting |
|---|---|---|
| 7 | position + quaternion `x, y, z, w` | `quaternion_order="xyzw"` (default) |
| 7 | position + quaternion `w, x, y, z` | `quaternion_order="wxyz"` (Hamlyn, UCSD) |
| 6 | position + Euler angles, extrinsic XYZ, radians | inferred from the width (Stanford, Rob Surgical) |

Quaternions are normalized to unit length before conversion. Translations keep the dataset's units; the per-family
normalization absorbs scale differences.

By default the pose is read from the dataset's `observation.state` (measured kinematics) and the gripper from
`action` (the commanded gripper). Families whose state pose is unusable read the pose from `action` instead
(`pose_source="action"`: Rob Surgical, Turin MITIC, STAR-IL).

### Time Alignment and Action History

Data is sampled at 10 Hz. A sample covers 2.4 s: 25 video frames (the current frame and 24 future frames) and
24 future action rows. Row `i` (1–24) holds the pose at `t = i` relative to the anchor at `t = 0`, and the gripper
command at `t = i − 1`.

Up to 16 **history** rows (H16, 1.6 s) are prepended. History row `j` (0–15) holds the pose at `t = −15 + j`
relative to the same anchor at `t = 0`, and the gripper command at `t = −16 + j`; the last history row is the
current pose. History is the contiguous valid stretch ending at the current frame: it is shorter at episode starts
and is cut at padding, non-finite values and, for CMR, at clutch, arm-assignment or scaling changes. A sample's
action tensor is therefore `[24 + h, 64]` with `h ≤ 16`. History rows are normalized with the same statistics as
the future rows.

---

## Embodiment Mapping

### Summary Table

| Family | Embodiment (ID) | Arms | Gripper | Action dim | Padding | Pose input | Pose source |
|---|---|---|---|---|---|---|---|
| `cmr_clinical` | `needle_pickup` (25) | 2 | yes | **20** | 44 | quaternion xyzw (hand controllers) | action |
| `cmr_versius_drybox_peg_transfer` | `needle_pickup` (25) | 2 | yes | **20** | 44 | quaternion xyzw (hand controllers) | action |
| `dvrk_obuda` | `dvrk_obuda` (11) | 2 | yes | **20** | 44 | quaternion xyzw | state |
| `dvrk_ucsd` | `dvrk_ucsd` (17) | 2 | yes | **20** | 44 | quaternion wxyz | state |
| `dvrk_jhu` | `suturebot` (22) | 2 | yes | **20** | 44 | quaternion xyzw | state |
| `dvrk_suturebot_mono` | `suturebot` (22) | 2 | yes | **20** | 44 | quaternion xyzw | state |
| `dvrk_stanford` | `dvrk_stanford` (14) | 2 | yes | **20** | 44 | Euler XYZ | state |
| `dvrk_ustc` | `ustc_torin` (31) | 2 | yes | **20** | 44 | quaternion xyzw | state |
| `dvrk_hamlyn` | `hamlyn` (16) | 2 | yes | **20** | 44 | quaternion wxyz | state |
| `dvrk_ucberkeley` | `dvrk_ucberkeley` (18) | 2 | yes | **20** | 44 | quaternion xyzw | state |
| `dvrk_smarts` | `smarts` (30) | 2 | yes | **20** | 44 | quaternion xyzw | state |
| `dvrk_rob_surgical` | `rob_surgical` (27) | 4 | no | **36** | 28 | Euler XYZ | action |
| `turin_mitic` | `turin_mitic` (19) | 2 | no | **18** | 46 | quaternion xyzw | action |
| `dvrk_star_il` | `star_il` (29) | 1 | no | **9** | 55 | quaternion xyzw | action |
| `dvrk_tud_grasping_retraction` | `tud_tundra` (28) | 1 | no | **9** | 55 | quaternion xyzw | state |

The embodiment IDs index the 32 slots of the action projections (`EMBODIMENT_TO_DOMAIN_ID` in
[`data/domain_utils.py`](data/domain_utils.py)); all 32 are assigned. New data reuses the slot of the most similar
robot (see the README, "Fine-tune on your own data").

### Per-Embodiment Notes

#### Dual-Arm with Grippers (20D)

Eleven families (dVRK systems and CMR) share this layout:

```
Arm 1: translation (3) + rot6d (6) + gripper (1) = 10
Arm 2: translation (3) + rot6d (6) + gripper (1) = 10
Total: 20 → [20 data | 44 zeros]
```

Arm order follows `meta/modality.json` (for example `psm1, psm2`; Hamlyn `left_arm, right_arm`; UCSD
`psm_retraction, psm_cutter`).

#### Rob Surgical (36D)

Four arms (`left, right, lap, aux`), no gripper channels, Euler-angle poses:

```
4 × (translation (3) + rot6d (6)) = 36 → [36 data | 28 zeros]
```

The `lap` arm holds the laparoscope: windows in which it translates by more than the camera-motion threshold
(5.0) are skipped, as camera motion would dominate the video target.

#### Turin MITIC (18D)

Two arms without gripper channels; poses from the action block: `2 × 9 = 18 → [18 data | 46 zeros]`.

#### Single Arm (9D)

STAR-IL (KUKA arm, poses from the action block) and TUD TUNDRA (UR5e end effector), no gripper channel:
`9 → [9 data | 55 zeros]`.

---

## CMR Versius Clinical

The CMR clinical data (Open-H `cmr_surgical`) uses a blocked 100-column schema. The loader builds the same 20D
dual-arm action from it:

| Source | Columns |
|---|---|
| Pose per hand (`action`) | `x, y, z` + quaternion `xyzw`: left 0–6, right 13–19 |
| Gripper (`pince`, `action`) | left 10, right 23 |
| Motion scaling (`observation.state`) | `translation_scaling` 12, `rotation_scaling` 13 |
| Clutch engagement (`observation.state`) | `hapticengaged` left 16, right 17 |
| Arm assignment (`observation.state`) | `armlinkedtohaptic` left 20, right 21 |

The poses are **hand-controller** poses. Per the CMR dataset documentation they are expressed in the camera
(endoscope) frame. The loader converts them to instrument motion:

- **Motion scaling**: the relative translation is multiplied by `translation_scaling` and the relative rotation
  angle by `rotation_scaling`, both read at the window anchor and applied to the whole window.
- **Window rejection**: windows in which any frame has a disengaged hand (clutch out, `clutch_policy="reject_any"`),
  an arm-assignment change, or endoscope motion (a thumbstick deflection above 5,000 raw units; the controls rest
around 1,200–1,800) are skipped. The valid windows are
  precomputed in the CMR valid-window index (`tools/data_prep/build_cmr_valid_window_index.py`).
- **History** additionally stops at any clutch, arm-assignment or scaling change.

Buttons, thumbsticks and other console state are not part of the action vector; the thumbsticks are only used to
reject windows.

---

## Normalization

Actions are normalized per family with **quantile normalization**, using the `global` block of the family's
statistics file in [`data/normalizers/`](data/normalizers/):

```
x_norm = clip( 2 · (x − q01) / (q99 − q01) − 1,  −1.5, 1.5 )
```

The 1st/99th percentiles map to −1/+1; values beyond them are kept up to ±1.5 instead of being clipped at the
boundary. In the `global` block the 6D-rotation dimensions have `q01 = −1`, `q99 = 1`, so rotations pass through
unchanged (their components already lie in [−1, 1]); translations and grippers are rescaled. One exception: the
CMR dry-box statistics (`drybox_backward_anchored_worldtrans_rot6d.json`) rescale the rotation dimensions as well,
as trained. The `global_raw` block holds the unmodified statistics of all dimensions.

The statistics come from 4,000 training windows of a family (24 rows each, no history); the older shipped files were
computed with a simpler window selection (see [`training_differences.md`](training_differences.md)).
[`tools/compute_action_normalizer.py`](../tools/compute_action_normalizer.py) computes them for a new family with
the procedure of the most recent shipped files.

The policy server returns denormalized actions (`--normalizer`), and its `history_action` input is in the
normalized space (`QuantileDenormalizer.normalize` in
[`scripts/policy_server.py`](scripts/policy_server.py) applies the training normalization).

---

## Padding, Channel Masking and Embodiment Projections

After normalization, the transform zero-pads each action row to 64 dimensions and records the true width as
`raw_action_dim` (`pad_action_to_max_dim` in [`data/transforms.py`](data/transforms.py)). In the model
(cosmos-framework `OmniMoTModel`):

- each action row is one token; `action2llm` and `llm2action` are domain-aware linear layers whose weights are
  selected per sample by the embodiment ID;
- the noise and the velocity target are zeroed beyond `raw_action_dim`, and the action loss covers only the
  `raw_action_dim` valid channels;
- the action loss is averaged over the noised rows only (`HistoryOmniMoTModel`), so clean history rows do not
  dilute it.

The packing loader charges every sample `24 + 16` action tokens whatever its history length, so the number of
samples per packed sequence does not depend on history.

---

## Actions per Training Mode

Training mixes three modes per sample (weights 0.4 / 0.2 / 0.4):

| Mode | Video | Future action rows | History rows |
|---|---|---|---|
| Forward dynamics | first frame given, future generated | given (clean) | given |
| Inverse dynamics | all frames given | generated | given |
| Policy (`wam` in cosmos-framework) | first frame given, future generated | generated | given |

All modes are conditioned on the text caption (robot and setting description plus task instruction).

---

## Data Mixture

Each family has a mixture ratio. Training used **rank-based mixing** on 32 GPUs: every GPU rank serves one family,
ranks are allocated by ratio with at least one rank per family (largest-remainder method), and every rank packs its
own samples. The share of training compute per family is therefore its share of ranks:

| Family | Ratio | Share of ratios | GPU ranks (of 32) | Share of ranks |
|---|---|---|---|---|
| `dvrk_jhu` | 2.5928 | 32.5% | 7 | 21.9% |
| `cmr_clinical` | 1.59626 | 20.0% | 4 | 12.5% |
| `dvrk_obuda` | 0.6949 | 8.7% | 3 | 9.4% |
| `dvrk_rob_surgical` | 0.6031 | 7.5% | 2 | 6.3% |
| `turin_mitic` | 0.56609 | 7.1% | 2 | 6.3% |
| `dvrk_stanford` | 0.5253 | 6.6% | 2 | 6.3% |
| `dvrk_hamlyn` | 0.3272 | 4.1% | 2 | 6.3% |
| `dvrk_ustc` | 0.30749 | 3.8% | 2 | 6.3% |
| `dvrk_suturebot_mono` | 0.28775 | 3.6% | 2 | 6.3% |
| `dvrk_ucsd` | 0.1892 | 2.4% | 1 | 3.1% |
| `dvrk_ucberkeley` | 0.1333 | 1.7% | 1 | 3.1% |
| `dvrk_star_il` | 0.0704 | 0.9% | 1 | 3.1% |
| `dvrk_smarts` | 0.0619 | 0.8% | 1 | 3.1% |
| `dvrk_tud_grasping_retraction` | 0.0287 | 0.4% | 1 | 3.1% |
| `cmr_versius_drybox_peg_transfer` | 0.00374 | 0.05% | 1 | 3.1% |

On a different number of GPUs the allocation, and so the effective mixture, changes. With fewer GPUs than families,
`shard_across_workers=False` mixes families per sample by ratio instead. The roots and settings of every family are
in [`configs/openh_datasets.py`](configs/openh_datasets.py); four families are not on Hugging Face (see
[`training_differences.md`](training_differences.md)).

---

## Training Statistics

From the final training stage of the released checkpoint (iterations 49,000 → 50,000):

| Metric | Value |
|---|---|
| GPUs | 32 × NVIDIA H100 80GB (4 nodes), FSDP |
| Gradient accumulation | 2 |
| Packed sequence capacity | 5,040 tokens per micro-batch |
| Samples per optimizer update | ~545 (mean; packing varies) |
| Tokens per optimizer update | ~305,000 |
| Time per optimizer update | ~9 s |
| Resolution tier | 256 (aspect ratio kept, reflection padding) |
| Video frames per sample | 25 (1 current + 24 future) at 10 Hz |
| Action rows per sample | 24 future + up to 16 history |
| Validation split | 5% of the episodes of each root (seed 42) |

---

## Comparison with Cosmos-H-Surgical-Simulator

| | Cosmos-H-Surgical-Simulator | Cosmos3-H-Embodied-Nano |
|---|---|---|
| Model | Cosmos Predict 2.5, action-conditioned video | Cosmos3-Nano, joint video and action generation |
| Action width | 44 | 64 |
| Per-arm pose | `rel_xyz_rot6d`: translation relative to the reference pose, rotation `R_refᵀ · R_t` as 6D | the same (`backward_anchored_world_trans`): translation `t_i − t_0`, rotation `R_0ᵀ · R_i` as 6D |
| 6D rotation | first two columns, `[r00, r10, r20, r01, r11, r21]` | the same |
| Gripper | absolute | absolute (commanded gripper) |
| Dual-arm dVRK | 20D | 20D |
| CMR Versius | 44D: poses, grippers, energy buttons, thumbsticks, clutch buttons and 14 state-conditioning channels; clutch-aware delta re-integration | 20D (poses and grippers only, motion-scaled); windows with clutch-out, arm swaps or endoscope motion are skipped |
| Embodiment conditioning | one action MLP for all embodiments | per-embodiment action projections (32 slots) |
| Normalization | mean/std per dataset | quantile (q01/q99 → ±1, clip ±1.5) per family; rotations unchanged |
| Action rows per sample | 12 | 24 future + up to 16 history, at 10 Hz |
| Role of actions | conditioning (forward dynamics) | conditioning (forward dynamics, history) or generated (inverse dynamics, policy) |

---

## Source Code Pointers

| Component | File | Key classes / functions |
|---|---|---|
| Per-arm layout from `modality.json` | [`data/dvrk_schema.py`](data/dvrk_schema.py) | `parse_arm_specs`, `build_arm_specs`, `build_relative_action` |
| Relative pose and 6D rotation | [`data/pose_utils.py`](data/pose_utils.py) | `pose_abs_to_rel`, `convert_rotation` |
| Generalized dVRK loader | [`data/generalized_dvrk_dataset.py`](data/generalized_dvrk_dataset.py) | `GeneralizedDvrkLeRobotDataset` |
| CMR clinical loader | [`data/cmr_clinical_dataset.py`](data/cmr_clinical_dataset.py) | `CmrClinicalLeRobotDataset` |
| Action history | [`data/history_dataset.py`](data/history_dataset.py) | `anchored_history`, `GeneralizedDvrkHistoryDataset`, `CmrClinicalHistoryDataset` |
| Normalization | [`data/action_normalization.py`](data/action_normalization.py) | `normalize_action`, `NORM_CLIP` |
| Padding, sequence plan | [`data/transforms.py`](data/transforms.py) | `pad_action_to_max_dim`, `build_sequence_plan_from_mode` |
| Embodiment IDs, caption contexts | [`data/domain_utils.py`](data/domain_utils.py) | `EMBODIMENT_TO_DOMAIN_ID`, `EMBODIMENT_CAPTION_CONTEXT` |
| Family mixing | [`data/unified_dataset.py`](data/unified_dataset.py) | `ActionUnifiedIterableDataset` |
| Packing | [`data/joint_dataloader.py`](data/joint_dataloader.py) | `HistoryMatchedJointDataLoader` |
| Action loss over noised rows | [`model/omni_mot_model.py`](model/omni_mot_model.py) | `HistoryOmniMoTModel` |
| Mixture and family settings | [`configs/openh_datasets.py`](configs/openh_datasets.py) | `FAMILIES`, `COMMON_KWARGS` |
| Statistics for new data | [`tools/compute_action_normalizer.py`](../tools/compute_action_normalizer.py) | |
