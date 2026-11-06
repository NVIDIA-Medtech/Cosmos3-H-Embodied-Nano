# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compose all six real configurations and optionally verify actual loader batches."""

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from cosmos_framework.utils.config import load_config
from cosmos_framework.utils.lazy_config import LazyConfig, instantiate
from omegaconf import OmegaConf

from cosmos3_h_embodied.dreams.contract import write_json
from cosmos3_h_embodied.dreams.smoke import STAGES
from cosmos3_h_embodied.dreams.inference_config import validate_model_config


def single_packed_item(batch: dict[str, Any], key: str) -> torch.Tensor:
    """Extract one item from the public joint loader's sample/item nesting.

    CMR uses one sample with one camera/action item. Metadata such as the
    sequence plan is a flat per-sample list and must not use this helper.
    """
    samples = batch[key]
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError(f"Expected one packed sample for {key}")
    items = samples[0]
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], torch.Tensor):
        raise ValueError(f"Expected one tensor item in the {key} sample")
    return items[0]  # action [H+C,64] or video [3,C+1,H,W]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loaders", action="store_true")
    args = parser.parse_args()
    os.environ["COSMOS_DREAMS_SPEC"] = str(args.spec.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    # Do not leave a previous successful report visible if this attempt fails.
    write_json(args.output / "preflight.json", report)
    for stage in STAGES:
        cfg = load_config(
            "cosmos_framework/configs/simulation/config.py",
            ["--", f"experiment=chole_{stage}"],
            experiment_module="cosmos3_h_embodied.dreams.experiments",
        )
        assert cfg.model.config.ema.enabled is False
        assert cfg.model.config.max_action_dim == 64
        assert cfg.model.config.action_schema == "native"
        assert cfg.model.config.parallelism.context_parallel_shard_degree == 1
        assert cfg.trainer.grad_accum_iter == 1
        config_path = args.output / f"{stage}.yaml"
        LazyConfig.save_yaml(cfg, str(config_path))
        restored = OmegaConf.load(config_path)
        validate_model_config(OmegaConf.to_container(restored.model, resolve=True))
        report[stage] = {"config": "composed_and_reloaded", "loader": "not_requested"}
        if args.loaders and stage in ("c24_a", "c72"):
            loader = instantiate(cfg.dataloader_train)
            batch = next(iter(loader))
            action = single_packed_item(batch, "action")  # [H+C,64]
            video = single_packed_item(batch, "video")  # [3,C+1,480,832]
            future = 24 if stage == "c24_a" else 72
            history = 1 - batch["sequence_plan"][0].action_start_frame_offset
            assert 0 <= history <= 16
            assert tuple(action.shape) == (history + future, 64)
            assert tuple(video.shape) == (3, future + 1, 480, 832)
            assert torch.count_nonzero(action[:, 28:]) == 0
            assert int(batch["raw_action_dim"][0]) == 28
            assert torch.isfinite(action).all()
            report[stage]["loader"] = {
                "action_shape": list(action.shape),
                "video_shape": list(video.shape),
                "history": history,
            }
        write_json(args.output / "preflight.json", report)
        print(stage, report[stage], flush=True)


if __name__ == "__main__":
    main()
