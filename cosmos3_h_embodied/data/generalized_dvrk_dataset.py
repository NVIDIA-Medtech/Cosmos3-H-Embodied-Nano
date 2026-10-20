# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generalized dVRK bimanual loader — one class for every Open-H PSM group.

Derives the per-arm column layout from the dataset's ``meta/modality.json`` (via
``dvrk_schema.build_arm_specs``) instead of the hardcoded slices in
``DvrkWoundClosureLeRobotDataset``/``SutureBotLeRobotDataset``, so a single class
serves the Open-H dVRK-family groups (arcade, obuda, hamlyn, ucsd, ucberkeley,
turin[pose-only], …). Reuses the wound-closure machinery (letterbox-free,
NO task-progress dim). Pose is read from ``observation.state`` (realized
kinematics), gripper from ``action`` (setpoint), exactly as the hand-written
loaders do. Only the ``backward_anchored_world_trans`` convention is generalized
(the one all current runs use); other conventions are rejected up front so a
non-PSM layout can't be silently mislabelled by the base class's hardcoded slices.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from cosmos3_h_embodied.data.cosmos3_action_lerobot import (
    ActionSpec,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos3_h_embodied.data.dvrk_schema import build_arm_specs, build_relative_action
from cosmos3_h_embodied.data.dvrk_wound_closure_dataset import DvrkWoundClosureLeRobotDataset
from cosmos3_h_embodied.data.pose_utils import build_abs_pose_from_components

_POSE_CONVENTION = "backward_anchored_world_trans"
SHARD_ROOT_KEY = "_dvrk_source_root"


def load_shard_arm_specs(roots, pose_source, quaternion_order="xyzw"):
    """Bind each physical shard to its columns, with one stable output arm order.

    Match named arms when columns/formats differ. Name-only variants may retain
    their original slot order only when all ordered column ranges and formats
    are identical (the audited USTC psm1/psm2 versus left/right case). This preserves
    numerical channels, not an assumption of physical identity across recordings.
    Validate every configured root before starting the sample retry loop.
    """
    if pose_source not in ("state", "action"):
        raise ValueError(f"unsupported pose source: {pose_source!r}")
    by_root, reference = {}, None
    for root in roots:
        key = os.path.realpath(root)
        with open(os.path.join(key, "meta", "modality.json")) as fh:
            modality = json.load(fh)
        specs = build_arm_specs(
            modality["action"], modality["state"], pose_source=pose_source, quaternion_order=quaternion_order
        )
        if not specs:
            raise ValueError(f"shard has no pose arms: {root}")
        if reference is None:
            reference = specs
        named = {arm.name: arm for arm in specs}
        if set(named) == {arm.name for arm in reference}:
            ordered = [named[arm.name] for arm in reference]
        elif [(s.pose_cols, s.gripper_col, s.pose_format) for s in specs] == [
            (s.pose_cols, s.gripper_col, s.pose_format) for s in reference
        ]:
            ordered = specs
        else:
            raise ValueError(f"shard changes output arm/gripper semantics: {root}")
        if any((s.gripper_col is None) != (r.gripper_col is None) for s, r in zip(ordered, reference, strict=True)):
            raise ValueError(f"shard changes output arm/gripper semantics: {root}")
        by_root[key] = ordered
    if reference is None:
        raise ValueError("at least one physical shard is required")
    return reference, by_root


class GeneralizedDvrkLeRobotDataset(DvrkWoundClosureLeRobotDataset):
    """dVRK bimanual loader whose per-arm layout comes from ``modality.json``."""

    def __init__(
        self,
        root: str | list[str] | None = None,
        roots: list[str] | None = None,
        *,
        embodiment_type: str = "dvrk_generalized",
        pose_source: str = "state",
        quaternion_order: str = "xyzw",
        camera_motion_arm: str | None = None,
        camera_motion_threshold: float | None = None,
        # Policy-gated minimal-action filter ON by default for the Open-H dVRK mixture
        # (resample POLICY-mode windows >= 80% idle; kept for FDM/IDM). Scoped here
        # so it doesn't change other experiments that use the base loaders directly.
        policy_idle_max_fraction: float | None = 0.9,
        **kwargs: Any,
    ) -> None:
        resolved = roots if roots is not None else ([root] if isinstance(root, str) else list(root or []))
        if not resolved:
            raise ValueError("GeneralizedDvrkLeRobotDataset requires `root` or `roots`.")
        # pose_source='state' (default): pose from STATE, gripper from ACTION.
        # pose_source='action': pose (+gripper) from ACTION — for Turin, whose state
        # has overlapping joints+pose columns while action carries clean quat-7 poses.
        self._pose_source = pose_source
        self._quaternion_order = quaternion_order
        self._arm_specs, self._arm_specs_by_root = load_shard_arm_specs(resolved, pose_source, quaternion_order)

        # Optional camera-motion filter: for rigs where an arm drives the endoscope
        # (e.g. rob_surgical's `lap` laparoscope arm), reject windows where that arm's
        # position moves past a threshold — camera panning corrupts the FDM/video target
        # (same idea as the CMR joystick filter). Off unless both are set.
        self._camera_motion_threshold = camera_motion_threshold
        self._camera_motion_arm = camera_motion_arm
        if camera_motion_arm is not None:
            for shard, specs in self._arm_specs_by_root.items():
                if not any(a.name == camera_motion_arm for a in specs):
                    raise ValueError(f"camera_motion_arm={camera_motion_arm!r} absent in {shard}")

        pose_convention = kwargs.get("pose_convention", _POSE_CONVENTION)
        if pose_convention != _POSE_CONVENTION:
            raise ValueError(
                f"GeneralizedDvrkLeRobotDataset only supports pose_convention={_POSE_CONVENTION!r}, "
                f"got {pose_convention!r} (other conventions fall back to the base class's hardcoded "
                "PSM slices, which would mislabel a modality-driven layout)."
            )
        super().__init__(
            root=root,
            roots=roots,
            embodiment_type=embodiment_type,
            policy_idle_max_fraction=policy_idle_max_fraction,
            **kwargs,
        )
        # Pose-from-action needs T+1 action rows to form T relative-pose deltas
        # (the base requests only T action rows).
        if pose_source == "action":
            self._delta_timestamps["action"] = [i * self._dt for i in range(0, self._chunk_length + 1)]

    def _arm_specs_for_root(self, root):
        key = os.path.realpath(root)
        if key not in self._arm_specs_by_root:
            # This is a configuration/schema failure, not a tolerance rejection
            # that the parent is allowed to hide by randomly selecting a new row.
            raise AssertionError(f"unregistered physical shard schema: {root}")
        return self._arm_specs_by_root[key]

    def _arm_specs_for_sample(self, sample):
        root = sample.get(SHARD_ROOT_KEY)
        if root is None:
            if len(self._arm_specs_by_root) != 1:
                raise AssertionError("multi-shard action computation requires the fetched physical root")
            return self._arm_specs
        return self._arm_specs_for_root(root)

    def _fetch_sample(self, idx, **kwargs):
        mode, dataset_idx, row_idx, sample = super()._fetch_sample(idx, **kwargs)
        # Registered indices are local to a worker's subset of the configured
        # roots. A worker assigned only root 1 still fetches dataset_idx=0.
        root = self._dataset_build_args[dataset_idx]["root"]
        if root is None:
            raise AssertionError("generalized action samples require a physical registered root")
        self._arm_specs_for_root(root)
        # Bind identity to the actual fetched sample, not the last fetched shard:
        # history, retries and delayed computation must all use the same schema.
        sample = sample.copy()
        sample[SHARD_ROOT_KEY] = root
        return mode, dataset_idx, row_idx, sample

    def _build_action_spec(self) -> ActionSpec:
        """``[Pos, Rot6d, (Gripper)] × n_arms`` from the parsed arm specs."""
        components: list[Any] = []
        for arm in self._arm_specs:
            components.extend([Pos(), Rot("rot6d")])
            if arm.gripper_col is not None:
                components.append(Gripper())
        return build_action_spec(*components)

    @property
    def action_dim(self) -> int:
        return sum(9 + (1 if arm.gripper_col is not None else 0) for arm in self._arm_specs)

    def _compute_backward_anchored_world_trans_action(
        self, sample: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Modality-driven generalization of the hand-written worldtrans compute."""
        state = sample["observation.state"].numpy().astype(np.float32)  # (T+1, state_dim)
        action = sample["action"]  # (T or T+1, action_dim) tensor
        specs = self._arm_specs_for_sample(sample)
        full_action = build_relative_action(
            state,
            action,
            specs,
            pose_convention=_POSE_CONVENTION,
            rotation_format="rot6d",
            pose_source=self._pose_source,
        )
        # Anchor pose (first arm at t=0) for downstream trajectory reconstruction.
        pose_arr = np.asarray(action, dtype=np.float32) if self._pose_source == "action" else state

        # Camera-arm motion filter: reject windows where the endoscope-driving arm (e.g.
        # rob_surgical's `lap`) moves past the threshold — camera panning corrupts the
        # video target. Caught by the tolerance-retry loop -> resamples another window.
        if self._camera_motion_threshold is not None and self._camera_motion_arm is not None:
            cs = next(arm.pose_cols[0] for arm in specs if arm.name == self._camera_motion_arm)
            cam_pos = pose_arr[:, cs : cs + 3]
            if float(np.linalg.norm(cam_pos - cam_pos[0], axis=1).max()) > self._camera_motion_threshold:
                raise AssertionError("tolerance: camera arm (laparoscope) panning within window")

        spec0 = specs[0]
        start, end = spec0.pose_cols
        arm0 = pose_arr[:, start:end]
        if spec0.pose_format in ("quat_xyzw", "quat_wxyz"):
            norms = np.clip(np.linalg.norm(arm0[:, 3:7], axis=-1, keepdims=True), 1e-8, None)
            abs0 = build_abs_pose_from_components(arm0[:, 0:3], arm0[:, 3:7] / norms, spec0.pose_format)
        elif spec0.pose_format == "euler_xyz":
            abs0 = build_abs_pose_from_components(arm0[:, 0:3], arm0[:, 3:6], "euler_xyz")
        else:
            raise ValueError(f"Unsupported pose_format {spec0.pose_format!r} for anchor pose")
        initial_pose = torch.from_numpy(abs0[0].copy()).float()
        return full_action, initial_pose
