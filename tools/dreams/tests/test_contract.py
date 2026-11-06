# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Semantic tests: padding shapes alone must never establish compatibility."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from cosmos3_h_embodied.dreams.contract import DOMAIN, OLD_TO_NEW, require_contract, stamp
from cosmos3_h_embodied.dreams.data import encode, history_starts, normalize, window_reason
from cosmos3_h_embodied.dreams.migrate import remap
from scipy.spatial.transform import Rotation


def native(length: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action = np.zeros((length, 100), dtype=np.float32)  # [L,100]
    state = np.zeros_like(action)  # [L,100]
    action[:, [6, 19]] = 1
    state[:, 12:14] = 1
    state[:, 16:18] = 1
    state[:, 20:22] = [1, 2]
    state[:, 22:24] = [0, 8]
    return action, state, np.arange(length, dtype=np.float64) / 60  # [L]


def test_pose_is_anchor_relative_with_column_rot6d_and_scaled_rotvec() -> None:
    a, s, _ = native()
    s[:, 12:14] = [0.4, 0.5]
    base = Rotation.from_euler("z", 55, degrees=True)
    delta = Rotation.from_euler("x", 30, degrees=True)
    a[:, 3:7] = base.as_quat()  # [L,4]
    a[102, 3:7] = (base * delta).as_quat()  # [4]
    a[108, 3:7] = (base * Rotation.from_euler("x", 60, degrees=True)).as_quat()
    a[96, :3], a[102, :3], a[108, :3] = [1, 2, 3], [2, 2, 3], [3, 2, 3]
    a[96, 10], a[102, 10] = 0.2, 0.6
    out = encode(a, s, 96, np.array([96, 102]))  # [2,28]
    np.testing.assert_allclose(out[:, 0], [0.4, 0.8])
    np.testing.assert_allclose(out[:, 9], [0.2, 0.6])
    matrix = Rotation.from_euler("x", 15, degrees=True).as_matrix()  # [3,3]
    np.testing.assert_allclose(out[0, 3:9], np.r_[matrix[:, 0], matrix[:, 1]], atol=1e-6)
    np.testing.assert_allclose(encode(a, s, 96, np.array([90]))[0, :3], [0, 0, 0], atol=1e-6)


def test_buttons_aggregate_half_open_intervals_axes_are_start_values() -> None:
    a, s, _ = native()
    a[99, 8], a[102, 9] = 1, 1
    a[96, 11:13] = [500, -1000]
    a[99, 11:13] = [4000, 3000]
    out = encode(a, s, 96, np.array([96, 102]))  # [2,28]
    np.testing.assert_array_equal(out[:, 10:12], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(out[0, 12:14], [500, -1000])
    np.testing.assert_array_equal(normalize(out)[:, 10:12], out[:, 10:12])


def test_native_camera_pulse_in_true_history_rejects_without_shortening() -> None:
    a, s, t = native()
    a[7, 11:13] = [6000, 6000]  # radius >8000; missed by 10 Hz endpoint-only checks
    reason, history = window_reason(a, s, t, 96, 72)
    assert reason == "camera" and len(history) == 16
    a[7, 11:13] = [4800, 6400]  # exact radius=8000 accepted
    assert window_reason(a, s, t, 96, 72)[0] is None
    a[528, 24] = 8001  # inclusive future endpoint
    assert window_reason(a, s, t, 96, 72)[0] == "camera"


def test_invalid_history_suffix_timestamps_and_engagement() -> None:
    a, s, t = native()
    assert len(history_starts(a, s, t, 18)) == 3
    s[70, 16] = 0
    np.testing.assert_array_equal(history_starts(a, s, t, 96), [72, 78, 84, 90])
    s[100, 17] = 0
    assert window_reason(a, s, t, 96, 24)[0] == "telemetry_or_engagement"
    with pytest.raises(ValueError, match="engagement"):
        encode(a, s, 96, np.array([96]))
    s[100, 17] = 1
    t[103] += 0.002
    assert window_reason(a, s, t, 96, 24)[0] == "timestamps"


def test_projector_mapping_preserves_both_arms_other_domains_and_gradients() -> None:
    torch.manual_seed(7)
    width = 12
    state = {
        "action2llm.fc.weight": torch.randn(32, 64 * width),
        "llm2action.fc.weight": torch.randn(32, width * 64),
        "action2llm.bias.weight": torch.randn(32, width),
        "llm2action.bias.weight": torch.randn(32, 64),
    }
    old = {k: v.clone() for k, v in state.items()}
    remap(state)
    keep = [i for i in range(32) if i != DOMAIN]
    for key in old:
        torch.testing.assert_close(state[key][keep], old[key][keep], rtol=0, atol=0)
    source = torch.randn(5, 20)  # [T,20]
    target = torch.zeros(5, 64)  # [T,64]
    target[:, list(OLD_TO_NEW)] = source
    wi = state["action2llm.fc.weight"][DOMAIN].reshape(64, width).requires_grad_()  # [64,C]
    torch.testing.assert_close(target @ wi, source @ old["action2llm.fc.weight"][DOMAIN].reshape(64, width)[:20])
    controls = [10, 11, 12, 13, 24, 25, 26, 27]
    assert torch.count_nonzero(wi[controls]) == 0
    target[:, controls] = 1
    (target @ wi).sum().backward()
    assert torch.count_nonzero(wi.grad[controls]) == len(controls) * width
    torch.testing.assert_close(
        state["llm2action.fc.weight"][DOMAIN].reshape(width, 64)[:, list(OLD_TO_NEW)],
        old["llm2action.fc.weight"][DOMAIN].reshape(width, 64)[:, :20],
        rtol=0,
        atol=0,
    )


def test_checkpoint_requires_semantic_receipt(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model/.metadata").write_bytes(b"fixture")
    with pytest.raises(FileNotFoundError):
        require_contract(tmp_path)
    stamp(tmp_path, {"test": True})
    require_contract(tmp_path)
    receipt = json.loads((tmp_path / "action_contract.json").read_text())
    receipt["contract_sha256"] = "wrong"
    (tmp_path / "action_contract.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="contract mismatch"):
        require_contract(tmp_path)
