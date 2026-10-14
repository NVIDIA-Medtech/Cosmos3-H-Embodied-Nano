# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal recent-action context (H16: up to 16 past action transitions) for Open-H datasets.

Only tabular queries are extended. Existing loaders still select, filter, caption,
and normalize exactly the same future window. At episode/clutch/arm boundaries we
retain the contiguous valid suffix, rather than inventing padded observations or
rejecting a future window that the H0 control would have used.
"""

from __future__ import annotations

import numpy as np
import torch

from cosmos3_h_embodied.data.cmr_clinical_dataset import CmrClinicalLeRobotDataset
from cosmos3_h_embodied.data.dvrk_schema import ArmSpec, build_relative_action
from cosmos3_h_embodied.data.generalized_dvrk_dataset import SHARD_ROOT_KEY, GeneralizedDvrkLeRobotDataset


def valid_suffix_start(valid: torch.Tensor) -> int:
    """First row after the most recent invalid transition (or zero)."""
    invalid = torch.where(~valid.bool())[0]
    return int(invalid[-1]) + 1 if invalid.numel() else 0


def anchored_history(state, action, history_length, arm_specs, pose_source="state"):
    """Past transitions ending at t=0, expressed in the current window's frame.

    Row j uses pose at t=-H+j+1 and jaw command at t=-H+j. Thus the last
    supplied pose is the measured/current anchor, never a future observation.
    This matches SequencePlan's transition timestamps 1-H,...,0.
    """
    h = history_length
    if not h:
        dim = sum(9 + (a.gripper_col is not None) for a in arm_specs)
        return action.new_empty((0, dim))
    order = [h, *range(1, h + 1)]
    poses = action[order].clone() if pose_source == "action" else action[:h].clone()
    if pose_source == "action":
        # build_relative_action reads pose and jaw from the same array. Override
        # only disjoint jaw columns to retain the step-start command pairing.
        for spec in arm_specs:
            if spec.gripper_col is not None:
                poses[:h, spec.gripper_col] = action[:h, spec.gripper_col]
    return build_relative_action(
        state[order].numpy().astype(np.float32),
        poses,
        arm_specs,
        pose_convention="backward_anchored_world_trans",
        pose_source=pose_source,
    )


class _HistoryMixin:
    def __init__(self, *args, num_history_actions: int = 0, fixed_instruction: str | None = None, **kwargs):
        if fixed_instruction is not None and (not isinstance(fixed_instruction, str) or not fixed_instruction.strip()):
            raise ValueError("fixed instruction must be a nonempty decision-time instruction")
        self._fixed_instruction = fixed_instruction
        if isinstance(num_history_actions, bool) or num_history_actions not in (0, 16):
            raise ValueError("history datasets support num_history_actions=0 or 16")
        self._num_history_actions = num_history_actions
        self._sample_history = None
        super().__init__(*args, **kwargs)
        if self._pose_convention != "backward_anchored_world_trans":
            raise ValueError("history datasets require pose_convention=backward_anchored_world_trans")
        if self._latent_cache_active:
            raise ValueError("history datasets require live video decoding")
        if num_history_actions:
            for key in ("action", "observation.state"):
                self._delta_timestamps[key] = [
                    i * self._dt for i in range(-num_history_actions, 0)
                ] + self._delta_timestamps[key]

    def _fetch_sample(self, idx, **kwargs):
        mode, dataset_idx, row_idx, sample = super()._fetch_sample(idx, **kwargs)
        self._sample_history = None
        h = self._num_history_actions
        if h:
            valid = torch.ones(h, dtype=torch.bool)
            for key in ("action", "observation.state"):
                # LeRobot clamps out-of-episode requests and exposes is_pad.
                # Missing masks must fail closed, never count padding as history.
                pad = sample.get(f"{key}_is_pad")
                if pad is None or len(pad) < h + 1:
                    raise RuntimeError(f"history requires LeRobot padding mask for {key}")
                valid &= ~pad[:h].bool() & ~pad[1 : h + 1].bool()
                finite = torch.isfinite(sample[key][: h + 1]).all(dim=1)
                valid &= finite[:-1] & finite[1:]
            valid &= self._history_validity(sample, h)
            start = valid_suffix_start(valid)
            past = {key: sample[key][start : h + 1] for key in ("action", "observation.state")}
            if SHARD_ROOT_KEY in sample:
                past[SHARD_ROOT_KEY] = sample[SHARD_ROOT_KEY]
            history = self._compute_history(past, h - start)
            if not torch.isfinite(history).all():
                raise RuntimeError("nonfinite causal history")
            self._sample_history = self._normalize_action(history)
            # Parent filtering and target construction see byte-identical future
            # tensors; scalar frame/episode IDs and all video queries stay intact.
            sample = dict(sample)
            for key in ("action", "observation.state"):
                sample[key] = sample[key][h:]
                sample[f"{key}_is_pad"] = sample[f"{key}_is_pad"][h:]
        return mode, dataset_idx, row_idx, sample

    def _history_validity(self, sample, h):
        return torch.ones(h, dtype=torch.bool)

    def _build_result(self, **kwargs):
        if getattr(self, "_fixed_instruction", None) is not None:
            # Before the shared transform tokenizes text. Corrected recipes also
            # disable append_idle_frames: those counts remain available only to
            # existing sample filters and must not become conditioning text.
            kwargs["ai_caption"] = self._fixed_instruction
        result = super()._build_result(**kwargs)
        # Fetch/build run serially in each dataset worker, including CMR's
        # tabular preflight and second fetch of the same accepted video window.
        if self._sample_history is not None:
            result["history_action"] = self._sample_history
        return result


class GeneralizedDvrkHistoryDataset(_HistoryMixin, GeneralizedDvrkLeRobotDataset):
    def _load_known_bad_samples(self):
        # A shared append-only audit changes while workers start. Reading it
        # lazily gave H0/H16 different initial skip sets despite identical RNG
        # checkpoints. Both experimental arms use the frozen committed set;
        # worker-local failure memory and append-only rejection audits remain.
        super()._load_known_bad_samples(runtime_log="")

    def _compute_history(self, sample, h):
        return anchored_history(
            sample["observation.state"], sample["action"], h, self._arm_specs_for_sample(sample), self._pose_source
        )


class CmrClinicalHistoryDataset(_HistoryMixin, CmrClinicalLeRobotDataset):
    def _history_validity(self, sample, h):
        state = sample["observation.state"][: h + 1]
        # BLOCKED 100D clinical schema, not the interleaved 26D benchtop Versius layout.
        engaged = (state[:, [16, 17]] == 1).all(dim=1)
        same_arms = (state[:, [20, 21]] == state[h, [20, 21]]).all(dim=1)
        same_scaling = (state[:, [12, 13]] == state[h, [12, 13]]).all(dim=1)
        valid = engaged & same_arms & same_scaling
        return valid[:-1] & valid[1:]

    def _compute_history(self, sample, h):
        specs = [ArmSpec("left", (0, 7), 10, "quat_xyzw"), ArmSpec("right", (13, 20), 23, "quat_xyzw")]
        history = anchored_history(sample["observation.state"], sample["action"], h, specs, "action")
        if not h:
            return history
        ts, rs = sample["observation.state"][h, [12, 13]].tolist()
        ts = ts or 1.0
        for offset in (0, 10):
            history[:, offset : offset + 9] = torch.from_numpy(
                self._apply_motion_scaling(history[:, offset : offset + 9].numpy(), ts, rs)
            )
        return history
