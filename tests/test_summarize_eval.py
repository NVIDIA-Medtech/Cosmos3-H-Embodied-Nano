# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Seed matching, best-of-seeds and the mean-trajectory baseline of ``tools/benchmark/summarize_eval.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "benchmark"))
from summarize_eval import load_runs, summarize  # noqa: E402


def _sample(run: Path, mode: str, index: int, gt: list[list[float]] | None = None, **metrics: float) -> None:
    d = run / "fam" / "fam" / mode / str(index)
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"mode": mode, "name": str(index), **metrics}))
    if gt is not None:
        (d / "gt_action.json").write_text(json.dumps(gt))


def test_policy_best_of_seeds_and_baseline(tmp_path: Path) -> None:
    s0, s1 = tmp_path / "seed0", tmp_path / "seed1"
    _sample(s0, "forward_dynamics", 0, psnr=20.0)
    _sample(s0, "inverse_dynamics", 0, action_mse=0.01)
    # Two policy samples whose recorded trajectories are 0 and 2: the mean trajectory is 1, MSE 1.
    _sample(s0, "wam", 0, gt=[[0.0], [0.0]], action_mse=0.2, psnr=18.0)
    _sample(s0, "wam", 1, gt=[[2.0], [2.0]], action_mse=0.4, psnr=17.0)
    _sample(s1, "wam", 0, action_mse=0.1, psnr=19.0)
    _sample(s1, "wam", 1, action_mse=0.6, psnr=16.0)
    summary = summarize(load_runs([s0, s1]))
    fam = summary["fam"]
    assert fam["fd_psnr"] == 20.0 and fam["id_mse"] == pytest.approx(0.01)
    assert fam["policy_mse_mean"] == pytest.approx((0.15 + 0.5) / 2)
    assert fam["policy_mse_best"] == pytest.approx((0.1 + 0.4) / 2)
    assert fam["policy_psnr_best"] == pytest.approx((19.0 + 17.0) / 2)
    assert fam["policy_runs"] == 2 and fam["baseline_mse"] == pytest.approx(1.0)
    assert list(summary) == ["fam", "All"]
