# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Helpers of ``tools/compute_action_normalizer.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "compute_action_normalizer", Path(__file__).resolve().parents[1] / "tools" / "compute_action_normalizer.py"
)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_rotation_dims_follow_the_packed_arm_layout() -> None:
    two_arms = [SimpleNamespace(gripper_col=7), SimpleNamespace(gripper_col=15)]
    assert tool.rotation_dims(two_arms) == [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]
    no_gripper = [SimpleNamespace(gripper_col=None), SimpleNamespace(gripper_col=None)]
    assert tool.rotation_dims(no_gripper) == [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]


def test_global_block_sets_rotation_dims_to_the_identity_range() -> None:
    actions = np.random.default_rng(0).normal(size=(1000, 20)).astype(np.float32)
    raw = tool.compute_stats(actions)
    np.testing.assert_allclose(raw["q01"], np.quantile(actions, 0.01, axis=0), rtol=1e-6)
    overlaid = tool.identity_overlay(raw, [3, 4])
    for d in (3, 4):
        assert [overlaid[k][d] for k in ("q01", "q99", "mean", "std")] == [-1.0, 1.0, 0.0, 1.0]
    assert overlaid["q99"][0] == raw["q99"][0] and raw["q01"][3] != -1.0
