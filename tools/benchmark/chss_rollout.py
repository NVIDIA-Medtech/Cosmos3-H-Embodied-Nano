# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos-H-Surgical-Simulator rollouts of a benchmark plan, for ``compare_rollouts.py``.

Runs in a Cosmos-H-Surgical-Simulator checkout and environment (it imports that repository's
dataset and inference classes) and follows its ``inference_open_h.py``: per embodiment the
released transforms, 12-frame chunks conditioned on the last generated frame, actions zero-padded
to 44D, guidance 0. Each rollout starts at the plan's ``start_frame`` and is written as
``<output>/<root>/ep<episode>_f<start>_s<seed>/rollout.{npz,json}``, the layout of
``cosmos3_h_embodied.scripts.quant_eval``.

Example (from the Cosmos-H-Surgical-Simulator checkout)::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python <this repo>/tools/benchmark/chss_rollout.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \\
        --ckpt_path model_ema_bf16.safetensors --plan <this repo>/tools/benchmark/chss_benchmark_plan.json \\
        --data_root <Open-H>/Surgical --output outputs/chss_rollouts
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import WrappedLeRobotSingleDataset
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    EMBODIMENT_REGISTRY,
    MAX_ACTION_DIM,
    construct_modality_config_and_transforms,
)
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import ActionVideo2WorldInference

NUM_FRAMES = 13
CHUNK_SIZE = NUM_FRAMES - 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--s3_cred", default="credentials/s3_checkpoint.secret")
    p.add_argument("--plan", type=Path, required=True, help="Benchmark plan JSON (rollouts with CHSS embodiment tags)")
    p.add_argument("--data_root", required=True, help="Open-H 'Surgical' directory (plan roots are relative to it)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num_chunks", type=int, default=6, help="12-frame chunks per rollout (unless the plan sets num_frames)")
    p.add_argument("--num_seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="Base seed")
    p.add_argument("--guidance", type=float, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    return p.parse_args()


def load_dataset(path: str, embodiment: str, exclude_splits: list[str] | None) -> WrappedLeRobotSingleDataset:
    config, _, test_transform = construct_modality_config_and_transforms(
        num_frames=NUM_FRAMES, embodiment=embodiment, downscaled_res=False
    )
    modality_filename = config.pop("modality_filename", None) if isinstance(config, dict) else None
    return WrappedLeRobotSingleDataset(
        dataset_path=path,
        modality_configs=config,
        transforms=test_transform,
        embodiment_tag=embodiment,
        data_split="full",
        modality_filename=modality_filename,
        exclude_splits=exclude_splits,
    )


def pad_action(action: np.ndarray) -> np.ndarray:
    return np.pad(action, [(0, 0), (0, MAX_ACTION_DIM - action.shape[-1])]) if action.shape[-1] < MAX_ACTION_DIM else action


def to_thwc(video) -> np.ndarray:
    video = video.numpy() if isinstance(video, torch.Tensor) else video
    return np.transpose(video, (1, 2, 3, 0))  # (C, T, H, W) -> (T, H, W, C)


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = [(r, s) for r in plan for s in range(args.num_seeds)][args.shard_index :: args.num_shards]
    logger.info(f"{len(plan)} episodes x {args.num_seeds} seeds; this shard runs {len(jobs)} rollouts")

    video2world = ActionVideo2WorldInference(args.experiment, args.ckpt_path, args.s3_cred, context_parallel_size=1)
    datasets: dict[str, tuple[WrappedLeRobotSingleDataset, dict, int]] = {}
    for i, (rollout, seed_idx) in enumerate(jobs):
        root, episode, start = rollout["root"], int(rollout["episode"]), int(rollout["start_frame"])
        name = f"ep{episode:06d}_f{start:06d}_s{seed_idx}"
        out_dir = args.output / root / name
        if (out_dir / "rollout.json").is_file():
            continue
        embodiment = rollout["chss_embodiment"]
        if root not in datasets:
            ds = load_dataset(os.path.join(args.data_root, root), embodiment, rollout.get("chss_exclude_splits"))
            index = {(int(e), int(b)): j for j, (e, b) in enumerate(ds._all_steps)}
            interval = 6 if embodiment == "cmr_versius" else EMBODIMENT_REGISTRY[embodiment]["timestep_interval"]
            datasets[root] = (ds, index, interval)
        ds, index, interval = datasets[root]
        num_chunks = rollout.get("num_frames", CHUNK_SIZE * args.num_chunks) // CHUNK_SIZE
        bases = [start + k * CHUNK_SIZE * interval for k in range(num_chunks)]
        missing = [b for b in bases if (episode, b) not in index]
        if missing:
            raise ValueError(f"{root} episode {episode}: no samples at base indices {missing}")
        logger.info(f"[{i + 1}/{len(jobs)}] {root} episode {episode} from frame {start}, seed {seed_idx}")

        seed = args.seed + 1000 * seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        gt_chunks, pred_chunks, current = [], [], None
        for k, base in enumerate(bases):
            data = ds[index[(episode, base)]]
            video = to_thwc(data["video"])
            action = data["action"].numpy() if isinstance(data["action"], torch.Tensor) else data["action"]
            if current is None:
                current = video[0]
            next_frame, chunk = video2world.step_inference(
                img_array=current,
                action=pad_action(action).astype(np.float32),
                guidance=args.guidance,
                seed=seed + k,
                num_latent_conditional_frames=1,
            )
            gt_chunks.append(video)
            pred_chunks.append(chunk)
            current = next_frame
        gt = np.concatenate([gt_chunks[0]] + [c[1:] for c in gt_chunks[1:]])
        pred = np.concatenate([pred_chunks[0]] + [c[1:] for c in pred_chunks[1:]])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "rollout.npz", gt=gt.astype(np.uint8), pred=pred.astype(np.uint8))
        meta = {
            "model": "cosmos_h_surgical_simulator",
            "root": root,
            "episode": episode,
            "start_frame": start,
            "seed_index": seed_idx,
            "seed": seed,
            "embodiment": embodiment,
            "source_frames": [start + t * interval for t in range(gt.shape[0])],
            # The generated frame shows the whole source frame (resized to the model resolution).
            "pred_box": [0.0, 1.0, 0.0, 1.0],
            "source_box": [0.0, 1.0, 0.0, 1.0],
            "num_chunks": num_chunks,
            "guidance": args.guidance,
        }
        (out_dir / "rollout.json").write_text(json.dumps(meta, indent=2) + "\n")
    video2world.cleanup()


if __name__ == "__main__":
    main()
