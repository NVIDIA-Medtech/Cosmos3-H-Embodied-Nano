# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Schema parsing for the generalized dVRK bimanual loader.

Derives per-arm column layouts from a LeRobot ``modality.json`` action/state
block, so one dataset class can serve every bimanual-PSM Open-H group
(arcade, obuda, stanford, ucsd, ucberkeley, hamlyn, turin, …) instead of the
hardcoded slices in ``suturebot_dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cosmos3_h_embodied.data.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

_POSE_SUFFIX = "_pose"
_GRIPPER_SUFFIX = "_gripper"


@dataclass(frozen=True)
class ArmSpec:
    """Column layout for one manipulator arm within a modality block.

    Attributes:
        name: Arm prefix (e.g. ``"psm1"``, ``"left_arm"``, ``"psm_retraction"``).
        pose_cols: ``(start, end)`` half-open column range of the arm's pose.
        gripper_col: Column index of the arm's gripper scalar, or ``None`` if the
            arm has no gripper channel (e.g. Turin is pose-only).
    """

    name: str
    pose_cols: tuple[int, int]
    gripper_col: int | None
    pose_format: str


def _infer_pose_format(pose_width: int, quaternion_order: str = "xyzw") -> str:
    """Map a pose column width to its SE(3)-construction format.

    7 = position(3) + explicitly ordered quaternion(4); 6 = position(3) + euler_xyz(3) radians
    (Stanford: Absolute Cartesian pos + roll/pitch/yaw). Other widths are added
    when a concrete group needs them; fail loudly until then so a
    silently-mislabelled pose can't slip through.
    """
    if quaternion_order not in ("xyzw", "wxyz"):
        raise ValueError(f"Unsupported quaternion order {quaternion_order!r}")
    if pose_width == 7:
        return "quat_" + quaternion_order
    if pose_width == 6:
        return "euler_xyz"
    raise ValueError(f"Unsupported pose width {pose_width}; only quaternion (7) and euler_xyz (6) are handled so far.")


def parse_arm_specs(action_modality: dict, *, quaternion_order: str = "xyzw") -> list[ArmSpec]:
    """Group a modality action block's keys into per-arm column specs.

    Keys are grouped by prefix: ``<arm>_pose`` gives the pose columns and
    ``<arm>_gripper`` (optional) the gripper column. Arms are returned in
    first-seen order. Quaternion order is a source-format choice; the default
    preserves legacy XYZW behavior and does not infer semantics from width.
    """
    specs: dict[str, dict] = {}
    order: list[str] = []
    for key, span in action_modality.items():
        if key.endswith(_POSE_SUFFIX):
            name = key[: -len(_POSE_SUFFIX)]
        elif key.endswith(_GRIPPER_SUFFIX):
            name = key[: -len(_GRIPPER_SUFFIX)]
        else:
            continue
        if name not in specs:
            specs[name] = {}
            order.append(name)
        if key.endswith(_POSE_SUFFIX):
            specs[name]["pose_cols"] = (span["start"], span["end"])
        else:
            specs[name]["gripper_col"] = span["start"]
    result: list[ArmSpec] = []
    for name in order:
        start, end = specs[name]["pose_cols"]
        result.append(
            ArmSpec(
                name=name,
                pose_cols=(start, end),
                gripper_col=specs[name].get("gripper_col"),
                pose_format=_infer_pose_format(end - start, quaternion_order),
            )
        )
    _assert_no_overlap(result)
    return result


def build_arm_specs(
    action_modality: dict, state_modality: dict, pose_source: str = "state", *, quaternion_order: str = "xyzw"
) -> list[ArmSpec]:
    """Per-arm I/O layout for the loader: pose read from the STATE block (realized
    kinematics, used to build SE(3) transitions), gripper from the ACTION block
    (commanded setpoint), matched by arm name. Their column layouts differ across
    groups (e.g. Stanford), so both blocks must be parsed independently.

    ``pose_source='action'`` reads pose (and gripper) from the ACTION block instead —
    for groups like Turin whose STATE block has overlapping joints+pose columns
    (unusable) while ACTION carries clean quat-7 poses.
    """
    if pose_source == "action":
        return parse_arm_specs(action_modality, quaternion_order=quaternion_order)
    state_by_name = {s.name: s for s in parse_arm_specs(state_modality, quaternion_order=quaternion_order)}
    action_by_name = {s.name: s for s in parse_arm_specs(action_modality, quaternion_order=quaternion_order)}
    result: list[ArmSpec] = []
    for name, sspec in state_by_name.items():
        aspec = action_by_name.get(name)
        result.append(
            ArmSpec(
                name=name,
                pose_cols=sspec.pose_cols,
                gripper_col=aspec.gripper_col if aspec is not None else None,
                pose_format=sspec.pose_format,
            )
        )
    return result


def build_relative_action(
    state: np.ndarray,
    action: torch.Tensor,
    arm_specs: list[ArmSpec],
    pose_convention: str,
    rotation_format: str = "rot6d",
    pose_source: str = "state",
) -> torch.Tensor:
    """Generalized per-arm relative action, mirroring SutureBot's compute path.

    For each arm: read the pose columns from ``state`` (T+1, ...), build absolute
    SE(3) poses, convert to the requested relative convention (rot6d), and append
    the arm's gripper column from ``action`` (T, ...) if present.

    Args:
        state: (T+1, state_dim) array; pose read from ``spec.pose_cols``.
        action: (T, action_dim) tensor; gripper read from ``spec.gripper_col``.
        arm_specs: per-arm layout (pose from state, gripper from action).
        pose_convention: passed to ``pose_abs_to_rel`` (e.g. backward_anchored_world_trans).
        rotation_format: rotation encoding of the relative pose (default rot6d).

    Returns:
        (T, sum_arms(rel_dim [+1 if gripper])) tensor, arms concatenated in order.
    """
    # Pose source: STATE (realized kinematics) by default, or the ACTION array for
    # groups whose state pose is unusable (Turin). Gripper always comes from action.
    pose_arr = np.asarray(action if pose_source == "action" else state, dtype=np.float32)
    blocks: list[torch.Tensor] = []
    for spec in arm_specs:
        start, end = spec.pose_cols
        arm = pose_arr[:, start:end]
        if spec.pose_format in ("quat_xyzw", "quat_wxyz"):
            xyz = arm[:, 0:3]
            quat = arm[:, 3:7]
            norms = np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
            abs_poses = build_abs_pose_from_components(xyz, quat / norms, spec.pose_format)
        elif spec.pose_format == "euler_xyz":
            # width-6 pose: pos3 + euler_xyz3 (radians, extrinsic xyz). Stanford's
            # Absolute Cartesian [ee_x, ee_y, ee_z, roll, pitch, yaw].
            abs_poses = build_abs_pose_from_components(arm[:, 0:3], arm[:, 3:6], "euler_xyz")
        else:
            raise ValueError(f"Unsupported pose_format {spec.pose_format!r} for arm {spec.name!r}")
        rel = pose_abs_to_rel(poses_abs=abs_poses, rotation_format=rotation_format, pose_convention=pose_convention)
        block = torch.from_numpy(np.asarray(rel)).float()
        if spec.gripper_col is not None:
            # gripper always from the action setpoint; align to the T relative rows
            # (pose-from-action gives T+1 action rows, so slice to block length).
            grip = action[: block.shape[0], [spec.gripper_col]].float()
            block = torch.cat([block, grip], dim=-1)
        blocks.append(block)
    return torch.cat(blocks, dim=-1)


def _assert_no_overlap(specs: list[ArmSpec]) -> None:
    """Reject modality blocks whose column spans overlap (e.g. malformed smarts,
    where every key has start:0 and both arms alias the same columns)."""
    intervals: list[tuple[int, int, str]] = []
    for a in specs:
        intervals.append((a.pose_cols[0], a.pose_cols[1], f"{a.name}_pose"))
        if a.gripper_col is not None:
            intervals.append((a.gripper_col, a.gripper_col + 1, f"{a.name}_gripper"))
    intervals.sort()
    for (s0, e0, k0), (s1, e1, k1) in zip(intervals, intervals[1:]):
        if s1 < e0:
            raise ValueError(
                f"modality columns overlap: {k0}[{s0}:{e0}] and {k1}[{s1}:{e1}] "
                "-- malformed modality.json (non-cumulative start offsets?)"
            )
