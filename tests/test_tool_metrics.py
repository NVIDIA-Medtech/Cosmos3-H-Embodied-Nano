# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GATC and TCD of ``tools/benchmark/tool_metrics.py`` on synthetic tools (no segmenter needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torchcodec")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "benchmark"))
from tool_metrics import GATCConfig, gatc_per_frame, summarize_frames, tcd_per_frame  # noqa: E402


def _tool_video(num_frames: int, offset: int = 0) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """A textured background with a bright bar ("tool") moving right; returns frames and tool masks."""
    rng = np.random.default_rng(0)
    background = rng.integers(60, 120, size=(64, 96, 3), dtype=np.uint8)
    frames, masks = [], []
    for t in range(num_frames):
        x = 10 + 2 * t + offset
        frame = background.copy()
        frame[20:40, x : x + 12] = 230
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[20:40, x : x + 12] = 1
        frames.append(frame)
        masks.append([mask])
    return np.stack(frames), masks


def test_identical_rollout_scores_best() -> None:
    video, masks = _tool_video(12)
    gatc = gatc_per_frame(video, video, [m[0] for m in masks], GATCConfig())
    assert gatc == pytest.approx(np.ones(12), abs=1e-4)
    tcd, matched, missed = tcd_per_frame(masks, masks, 64, 96)
    assert np.all(tcd == 0) and np.all(matched == 0) and np.all(missed == 0)


def test_misplaced_and_missing_tools_score_worse() -> None:
    video, masks = _tool_video(12)
    shifted, shifted_masks = _tool_video(12, offset=20)
    gatc = gatc_per_frame(video, shifted, [m[0] for m in masks], GATCConfig())
    assert np.nanmedian(gatc) < 0.9
    assert tcd_per_frame(masks, shifted_masks, 64, 96)[0] == pytest.approx(np.full(12, 20.0))
    # A frame without any generated tool costs half the image diagonal.
    tcd, matched, missed = tcd_per_frame(masks, [[]] * 12, 64, 96)
    assert tcd[0] == pytest.approx(0.5 * np.hypot(64, 96)) and np.isnan(matched[0]) and missed[0] == 1.0


def test_frames_without_reference_tools_are_not_scored() -> None:
    per_frame = np.array([np.nan, 1.0, 3.0, np.nan], dtype=np.float32)
    s = summarize_frames(per_frame)
    assert s["score"] == 2.0 and s["coverage"] == 50.0 and s["phases"] == {}
