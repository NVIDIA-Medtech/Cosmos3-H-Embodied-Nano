# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Video metrics of an autoregressive rollout against the recorded video (see README_EVALUATION.md).

Frame 0 is the conditioning frame and is never scored. Frames are compared in [-1, 1].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from cosmos_framework.inference.metrics import compute_dynamic_psnr, compute_psnr, compute_ssim

# Scored frame ranges [start, stop) by time since the conditioning frame (10 frames per second):
# the 12-frame chunks 1, 2-3 and 4-6 of Cosmos-H-Surgical-Simulator's evaluation.
PHASES = {"early": (1, 13), "mid": (13, 37), "late": (37, 73)}
METRICS = ("fds", "ssim", "psnr", "dynamic_psnr")


def _to_cthw(video_thwc: np.ndarray) -> torch.Tensor:
    if video_thwc.dtype != np.uint8 or video_thwc.ndim != 4 or video_thwc.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3) uint8 video, got {video_thwc.dtype} {video_thwc.shape}")
    return torch.from_numpy(np.ascontiguousarray(video_thwc)).permute(3, 0, 1, 2).float() / 127.5 - 1.0


def _dynamic_psnr(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """PSNR over the 20% of pixels with the largest recorded frame-to-frame change.

    ``torch.quantile`` takes at most 2**24 values, so long clips use the NumPy path (same percentile).
    """
    if gt[0, 1:].numel() > 2**24:
        return float(compute_dynamic_psnr(gt.numpy(), pred.numpy())[0])
    return compute_dynamic_psnr(gt, pred)[0]


def _scores(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    """FDS (mean per-frame L1), mean SSIM, PSNR and dynamic PSNR of (C, T, H, W) clips in [-1, 1]."""
    return {
        "fds": float((pred - gt).abs().mean()),
        "ssim": compute_ssim(gt, pred),
        "psnr": compute_psnr(gt, pred),
        "dynamic_psnr": _dynamic_psnr(gt, pred),
    }


def rollout_metrics(gt_thwc: np.ndarray, pred_thwc: np.ndarray) -> dict[str, Any]:
    """Scores of a rollout over all generated frames, per phase, and per frame.

    Args:
        gt_thwc: Recorded video, (T, H, W, 3) uint8; frame 0 is the conditioning frame.
        pred_thwc: Generated video of the same shape.
    """
    if gt_thwc.shape != pred_thwc.shape:
        raise ValueError(f"shape mismatch: gt {gt_thwc.shape} vs pred {pred_thwc.shape}")
    gt, pred = _to_cthw(gt_thwc), _to_cthw(pred_thwc)
    num_frames = gt.shape[1]
    out: dict[str, Any] = {"num_frames": num_frames - 1, **_scores(gt[:, 1:], pred[:, 1:])}
    out["phases"] = {
        name: _scores(gt[:, start:stop], pred[:, start:stop])
        for name, (start, stop) in PHASES.items()
        if stop <= num_frames
    }
    mse = ((pred[:, 1:] - gt[:, 1:]) ** 2).mean(dim=(0, 2, 3))
    out["per_frame"] = {
        "l1": (pred[:, 1:] - gt[:, 1:]).abs().mean(dim=(0, 2, 3)).tolist(),
        "psnr": [20.0 * math.log10(2.0 / math.sqrt(v)) if v > 0 else float("inf") for v in mse.tolist()],
    }
    return out


def summarize(values: list[float]) -> dict[str, float]:
    """Mean, standard deviation, 95% confidence half-width (normal approximation) and count."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "ci95": 1.96 * std / math.sqrt(arr.size), "n": int(arr.size)}
