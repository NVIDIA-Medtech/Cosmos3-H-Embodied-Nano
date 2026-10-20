# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rollout metrics and the frame geometry recorded by ``quant_eval``."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos3_h_embodied.scripts.quant_eval import _boxes
from cosmos3_h_embodied.scripts.rollout_metrics import rollout_metrics


def _moving_square(num_frames: int) -> np.ndarray:
    video = np.full((num_frames, 32, 48, 3), 100, dtype=np.uint8)
    for t in range(num_frames):
        video[t, 8:16, t % 40 : t % 40 + 8] = 250
    return video


def test_identical_rollout_is_perfect() -> None:
    video = _moving_square(49)
    m = rollout_metrics(video, video)
    assert m["fds"] == 0 and m["ssim"] == pytest.approx(1.0) and math.isinf(m["psnr"])
    assert m["num_frames"] == 48 and len(m["per_frame"]["l1"]) == 48
    assert set(m["phases"]) == {"early", "mid"}  # "late" needs 72 generated frames


def test_dynamic_psnr_penalizes_a_frozen_prediction_more() -> None:
    gt = _moving_square(73)
    frozen = np.repeat(gt[:1], 73, axis=0)  # copies the conditioning frame
    m = rollout_metrics(gt, frozen)
    assert m["dynamic_psnr"] < m["psnr"]
    assert set(m["phases"]) == {"early", "mid", "late"}
    # FDS is the mean of the per-frame L1 (frames in [-1, 1]).
    assert m["fds"] == pytest.approx(np.mean(m["per_frame"]["l1"]))


def test_boxes_without_letterbox() -> None:
    # The generated frame (176 rows) is the top of the 181-row content region.
    pred_box, source_box = _boxes(SimpleNamespace(), [480, 848], [181, 320], [176, 320])
    assert pred_box == [0.0, 1.0, 0.0, 1.0]
    assert source_box == pytest.approx([0.0, 176 / 181, 0.0, 1.0])


def test_boxes_with_cmr_letterbox() -> None:
    # 1080x1920 letterboxed to 480x832: 468 image rows between 6-row bars.
    ds = SimpleNamespace(_letterbox_hw=(480, 832))
    pred_box, source_box = _boxes(ds, [1080, 1920], [185, 320], [176, 320])
    covered = 176 / 185 * 480
    assert pred_box == pytest.approx([6 / covered, 1.0, 0.0, 1.0])
    assert source_box == pytest.approx([0.0, (covered - 6) / 468, 0.0, 1.0])
