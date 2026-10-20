# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frechet distance, time windows and stability ratios of ``tools/benchmark/realism_metrics.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torchcodec")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "benchmark"))
from compare_rollouts import time_windows  # noqa: E402
from realism_metrics import frechet_distance, motion, sharpness  # noqa: E402


def test_frechet_distance_of_gaussians() -> None:
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(4000, 8)))
    assert frechet_distance(x, x) == pytest.approx(0.0, abs=1e-6)
    # A mean shift adds its squared norm; a scaled covariance adds d * (s - 1)^2 for std scale s.
    shift = torch.full((8,), 0.5, dtype=torch.float64)
    assert frechet_distance(x + shift, x) == pytest.approx(8 * 0.25, rel=1e-6)
    assert frechet_distance(2 * x, x) == pytest.approx(frechet_distance(x, x) + 8 * 1.0, rel=0.05)


def test_time_windows() -> None:
    assert time_windows(71, 0) == {}
    assert time_windows(71, 60) == {"0-7 s": (1, 72)}
    w = time_windows(311, 60)
    assert list(w.values()) == [(1, 61), (61, 121), (121, 181), (181, 241), (241, 312)]
    assert list(w)[-1] == "24-31 s"


def test_stability_ratios_of_a_blurred_still_video() -> None:
    rng = np.random.default_rng(0)
    video = torch.from_numpy(rng.integers(0, 256, size=(4, 3, 32, 32), dtype=np.uint8))
    still = video[:1].expand(4, -1, -1, -1)
    assert np.all(motion(still) == 0) and np.all(motion(video)[1:] > 0)
    blurred = torch.nn.functional.avg_pool2d(video.float(), 3, 1, 1).round().to(torch.uint8)
    assert np.all(sharpness(blurred) < sharpness(video))
