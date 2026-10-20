# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action normalization helpers."""

import functools
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.utils import log


@functools.lru_cache(maxsize=64)
def load_action_stats(stats_path: str, stats_key: str = "global") -> dict[str, np.ndarray]:
    """Load pre-computed action normalization stats from a JSON file.

    lru_cache: each worker process only reads the file once per (path, key) pair
    regardless of how many dataset entries share the same normalizer file.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found at {stats_path}.")
    log.info(f"Loading action normalization stats from {stats_path}")
    with path.open("r") as f:
        raw = json.load(f)
    if stats_key in raw:
        raw = raw[stats_key]
        if not isinstance(raw, dict):
            raise TypeError(f"Action normalization stats block {stats_key!r} in {stats_path} must be a dict.")
    elif stats_key != "global":
        raise KeyError(f"Action normalization stats block {stats_key!r} not found in {stats_path}.")
    stat_keys = {"mean", "std", "min", "max", "q01", "q99"}
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items() if k in stat_keys}


# Normalized-action clip bound. NOT 1.0: we clip to [-1.5, 1.5] so values beyond the
# q01/q99 (or min/max) range survive as outliers instead of being flattened to the
# boundary — LBM-style headroom. q99 still maps to +1.0.
NORM_CLIP = 1.5


def normalize_action(
    action: torch.Tensor,
    method: str,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Normalize action tensor (all dimensions including gripper)."""
    if method == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        denom = (q99 - q01).clamp(min=1e-8)
        return (2.0 * (action - q01) / denom - 1.0).clamp(-NORM_CLIP, NORM_CLIP)
    if method == "meanstd":
        return (action - stats["mean"]) / stats["std"].clamp(min=1e-8)
    if method == "minmax":
        lo, hi = stats["min"], stats["max"]
        denom = (hi - lo).clamp(min=1e-8)
        return (2.0 * (action - lo) / denom - 1.0).clamp(-NORM_CLIP, NORM_CLIP)
    raise ValueError(f"Unknown normalization method: {method!r}")
