# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rollout building blocks without a model: history re-anchoring and forward-dynamics batches."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from cosmos3_h_embodied.data.pose_utils import pose_abs_to_rel
from cosmos3_h_embodied.scripts.policy_server import build_policy_batch, make_policy_transform
from cosmos3_h_embodied.scripts.rollout_utils import arm_layout, make_raw_sample, reanchor_history

CONVENTION = "backward_anchored_world_trans"


def _trajectory(num_poses: int, seed: int) -> np.ndarray:
    """Smooth random absolute poses ``[T, 4, 4]``."""
    rng = np.random.default_rng(seed)
    rotvec = np.cumsum(rng.normal(scale=0.05, size=(num_poses, 3)), axis=0)
    poses = np.tile(np.eye(4), (num_poses, 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    poses[:, :3, 3] = np.cumsum(rng.normal(scale=0.002, size=(num_poses, 3)), axis=0)
    return poses


def _relative(poses: np.ndarray) -> np.ndarray:
    """Rows relative to ``poses[0]`` (the dataset convention), ``[T - 1, 9]``."""
    return pose_abs_to_rel(poses, rotation_format="rot6d", pose_convention=CONVENTION)


@pytest.mark.parametrize("gripper", [True, False])
def test_reanchored_history_equals_the_next_chunks_history(gripper) -> None:
    # Two arms, absolute poses at t = 0..24; a chunk holds the poses at t = 1..24 relative to t = 0.
    arms = [_trajectory(25, seed) for seed in (0, 1)]
    blocks, expected = [], []
    for arm, poses in enumerate(arms):
        jaw = np.linspace(-1, 1, 24)[:, None] * (arm + 1)
        chunk = _relative(poses)
        # The next chunk is anchored at t = 24: history row j = pose at t = 9 + j relative to it.
        history = _relative(np.concatenate([poses[24:25], poses[9:25]]))
        blocks.append(np.concatenate([chunk, jaw], axis=1) if gripper else chunk)
        expected.append(np.concatenate([history, jaw[8:]], axis=1) if gripper else history)
    action, expected = np.concatenate(blocks, axis=1), np.concatenate(expected, axis=1)
    assert arm_layout(action.shape[1]) == (2, gripper)
    np.testing.assert_allclose(reanchor_history(action), expected, atol=1e-5)


def test_arm_layouts() -> None:
    assert arm_layout(20) == (2, True) and arm_layout(36) == (4, False)
    assert arm_layout(18) == (2, False) and arm_layout(9) == (1, False)
    with pytest.raises(ValueError):
        arm_layout(25)


def test_forward_dynamics_batch_conditions_on_all_actions() -> None:
    image = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
    action, history = torch.randn(24, 20), torch.randn(5, 20)
    raw = make_raw_sample(image, "Transfer the peg", 0, 20, mode="forward_dynamics", action=action, history=history)
    out = make_policy_transform(64)(raw, resolution="256")
    assert out["sequence_plan"].condition_frame_indexes_action == list(range(29))
    torch.testing.assert_close(out["action"][:5, :20], history)
    torch.testing.assert_close(out["action"][5:, :20], action)
    with pytest.raises(ValueError):
        make_raw_sample(image, "p", 0, 20, mode="forward_dynamics")  # actions missing
    with pytest.raises(ValueError):
        make_raw_sample(image, "p", 0, 20, mode="forward_dynamics", action=torch.randn(12, 20))


def test_server_forward_dynamics_request() -> None:
    image = np.random.default_rng(0).integers(0, 256, size=(3, 480, 640), dtype=np.uint8)
    action = torch.randn(24, 20)
    req = {"prompt": "Transfer the peg", "domain_name": "hamlyn", "mode": "forward_dynamics", "action": action.tolist()}
    batch, meta = build_policy_batch(req, transform=make_policy_transform(64), raw_action_dim=20, image=torch.from_numpy(image))
    assert batch["mode"] == ["forward_dynamics"] and meta["num_history_actions"] == 0
    torch.testing.assert_close(batch["action"][0][0][:, :20], action)
    assert batch["sequence_plan"][0].condition_frame_indexes_action == list(range(24))
