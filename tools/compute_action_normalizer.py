# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute the action normalization statistics of a dataset family.

The loaders normalize actions per family with a JSON file under ``cosmos3_h_embodied/data/normalizers/``
(``normalizer_filename`` of the family in ``cosmos3_h_embodied/configs/openh_datasets.py``). This script
writes that file for a family of the generalized dVRK loader (``GeneralizedDvrkHistoryDataset``), for
example your own data once it is added to ``FAMILIES``. The procedure is the one of the released
checkpoint's most recent statistics files:

1. Build the family's dataset as the recipe configures it (roots, sample stride, exclusions, arm layout,
   quaternion order), train split, without action history and without normalization.
2. Draw ``20 x --num-samples`` candidate windows at random (``random.Random(seed).sample``), read their
   tabular data, and keep the first ``--num-samples`` windows whose actions and states are complete
   (no padding at episode ends) and finite (all of them for a smaller dataset). Each window holds
   ``chunk_length`` (24) actions: per arm, translation, 6D rotation and gripper, relative to the
   window's current pose.
3. Per action dimension, compute mean, std, min, max and the 1st/99th percentiles over all kept actions
   (``global_raw``).
4. ``global`` is the same with the 6D-rotation dimensions set to the identity range (q01 = min = -1,
   q99 = max = 1, mean 0, std 1): the recipe's quantile normalization then leaves rotations unchanged
   (rot6d components already lie in [-1, 1]).

Only tabular data is read (no video decoding). Example, after adding a family ``my_lab``::

    OPENH_DATA_ROOT=<data root> python tools/compute_action_normalizer.py --family my_lab

writes ``cosmos3_h_embodied/data/normalizers/<the family's normalizer_filename>`` (or ``-o <path>``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from cosmos3_h_embodied.configs.openh_datasets import COMMON_KWARGS, FAMILIES
from cosmos3_h_embodied.data.history_dataset import GeneralizedDvrkHistoryDataset

NORMALIZERS_DIR = Path(__file__).resolve().parents[1] / "cosmos3_h_embodied/data/normalizers"


def compute_stats(actions: np.ndarray) -> dict[str, list[float]]:
    """Per-dimension statistics of ``actions`` [N, D]."""
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
    }


def identity_overlay(stats: dict[str, list[float]], dims: list[int]) -> dict[str, list[float]]:
    """``stats`` with the given dimensions set to the identity range [-1, 1]."""
    out = {k: list(v) for k, v in stats.items()}
    for d in dims:
        out["mean"][d], out["std"][d] = 0.0, 1.0
        out["min"][d], out["q01"][d] = -1.0, -1.0
        out["max"][d], out["q99"][d] = 1.0, 1.0
    return out


def rotation_dims(arm_specs) -> list[int]:
    """6D-rotation dimensions of the packed action: each arm is translation (3) + rot6d (6) [+ gripper (1)]."""
    dims, offset = [], 0
    for arm in arm_specs:
        dims += list(range(offset + 3, offset + 9))
        offset += 9 + (1 if arm.gripper_col is not None else 0)
    return dims


def build_dataset(family_name: str, data_root: str, include_nonpublic: bool):
    """The family's training dataset (train split, no history, no normalization) and its relative roots."""
    family = next((f for f in FAMILIES if f.name == family_name), None)
    if family is None:
        raise SystemExit(f"unknown family {family_name!r}; add it to FAMILIES in cosmos3_h_embodied/configs/openh_datasets.py")
    if family.cls is not GeneralizedDvrkHistoryDataset:
        raise SystemExit(f"{family_name}: only GeneralizedDvrkHistoryDataset families are supported")
    roots = [r for r in family.roots if include_nonpublic or r.public]
    kwargs = dict(COMMON_KWARGS, **family.kwargs)
    kwargs.update(
        roots=[f"{data_root}/{r.path}" for r in roots],
        caption_embodiments=[r.caption for r in roots],
        split="train",
        mode="forward_dynamics",
        action_normalization=None,
        num_history_actions=0,
    )
    # Resolve the config interpolations (data root, exclusion list) as Hydra would.
    os.environ.setdefault("OPENH_DATA_ROOT", data_root)
    kwargs = OmegaConf.to_container(OmegaConf.create(kwargs), resolve=True)
    dataset = family.cls(**kwargs)
    dataset._skip_video_loading = True
    dataset._register_sources()
    return family, dataset, [r.path for r in roots]


def collect(dataset, count: int, seed: int) -> tuple[np.ndarray, Counter, Counter]:
    """Actions of the first ``count`` complete, finite windows among ``20 * count`` random candidates."""
    candidates = random.Random(seed).sample(range(len(dataset)), min(len(dataset), count * 20))
    chunks, per_root, rejected = [], Counter(), Counter()
    for index in candidates:
        root = dataset._dataset_build_args[dataset._resolve_index(index)[0]]["root"]
        try:
            _, _, _, raw = dataset._fetch_sample(index, decode_video=False, mode_override="forward_dynamics")
        except (OSError, RuntimeError, ValueError, AssertionError) as exc:
            if isinstance(exc, AssertionError) and "tolerance" not in str(exc):
                raise
            rejected["fetch"] += 1
            continue
        keys = ("action", "observation.state")
        if any(raw[k + "_is_pad"].any() for k in keys):
            rejected["padding"] += 1
            continue
        if not all(raw[k].isfinite().all() for k in keys):
            rejected["nonfinite"] += 1
            continue
        try:
            action, _ = dataset._compute_convention_action(raw)
        except (ValueError, AssertionError) as exc:
            if "zero-norm" not in str(exc) and "tolerance" not in str(exc):
                raise
            rejected["geometry"] += 1
            continue
        chunks.append(action.numpy())
        per_root[root] += 1
        if len(chunks) % 500 == 0:
            print(f"  {len(chunks)}/{count}", flush=True)
        if len(chunks) == count:
            break
    if not chunks:
        raise SystemExit(f"no complete windows among {len(candidates)} candidates")
    if len(chunks) < count:
        print(f"WARNING: only {len(chunks)} complete windows among {len(candidates)} candidates; using all of them")
    return np.concatenate(chunks).astype(np.float32), per_root, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--family", required=True, help="family name in FAMILIES (configs/openh_datasets.py)")
    parser.add_argument("--data-root", default=os.environ.get("OPENH_DATA_ROOT"), help="default: $OPENH_DATA_ROOT")
    parser.add_argument("--num-samples", type=int, default=4000, help="number of windows")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--include-nonpublic", action="store_true", help="also the roots marked public=False")
    parser.add_argument("-o", "--output", type=Path, help="default: the family's normalizer file")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("set OPENH_DATA_ROOT or --data-root")

    family, dataset, roots = build_dataset(args.family, args.data_root.rstrip("/"), args.include_nonpublic)
    rot_dims = rotation_dims(dataset._arm_specs)
    print(f"{family.name}: {len(dataset)} training windows, action dim {dataset.action_dim}, rotation dims {rot_dims}")
    actions, per_root, rejected = collect(dataset, args.num_samples, args.seed)
    print(f"kept {len(actions) // dataset._chunk_length} windows; rejected: {dict(rejected)}")

    raw = compute_stats(actions)
    by_path = {f"{args.data_root.rstrip('/')}/{r}": r for r in roots}
    out = {
        "metadata": {
            "family": family.name,
            "embodiment_type": dataset._embodiment_type,
            "pose_convention": dataset._pose_convention,
            "rotation_format": "rot6d",
            "quaternion_order": family.kwargs.get("quaternion_order", "xyzw"),
            "action_dim": int(actions.shape[1]),
            "skip_rotation_dims": rot_dims,
            "chunk_length": dataset._chunk_length,
            "fps": COMMON_KWARGS["fps"],
            "split": "train",
            "split_seed": COMMON_KWARGS["split_seed"],
            "split_val_ratio": COMMON_KWARGS["split_val_ratio"],
            "seed": args.seed,
            "dataset_roots": roots,
            "num_chunks_sampled": int(actions.shape[0]) // dataset._chunk_length,
            "num_steps_used": int(actions.shape[0]),
            "root_chunk_counts": {by_path.get(k, k): v for k, v in sorted(per_root.items())},
            "selection": "Uniform random candidate windows from the family's train split; the first "
            f"{args.num_samples} (or all) with complete, finite actions and states. global: rotation dims set to the "
            "identity range; global_raw: statistics of all dims.",
        },
        "global": identity_overlay(raw, rot_dims),
        "global_raw": raw,
    }
    path = args.output or NORMALIZERS_DIR / family.kwargs["normalizer_filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
