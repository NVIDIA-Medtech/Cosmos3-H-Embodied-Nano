# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-seed recorded/action-intervention evaluation through public native inference."""

from cosmos_framework.inference.common.init import init_script

init_script(env={"COSMOS_TRAINING": "1"})

import argparse
import copy
import html
import json
from pathlib import Path

import numpy as np
import torch
from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.init import is_rank0
from cosmos_framework.inference.dataset_samples import _collate_sample
from cosmos_framework.inference.metrics import compute_psnr, compute_ssim
from cosmos_framework.inference.vision import read_media_frames
from omegaconf import OmegaConf

from cosmos3_h_embodied.dreams.contract import file_sha256, write_json
from cosmos3_h_embodied.dreams.data import normalize
from cosmos3_h_embodied.dreams.loader import CholeDataset
from cosmos3_h_embodied.scripts.eval_utils import _compute_lpips


def intervene(actions: torch.Tensor, history: int, mode: str) -> torch.Tensor:  # [H+C,64] -> [H+C,64]
    """Change future commands only; never change the genuine recorded prefix."""
    result = actions.clone()  # [H+C,64]
    if mode == "energy_off":
        result[history:, [10, 24]] = 0
    elif mode in ("translation_hold", "rotation_hold", "pose_hold"):
        neutral = np.zeros((1, 28), dtype=np.float32)  # [1,28]
        neutral[:, [3, 7, 17, 21]] = 1  # two identity rotations in column-6D encoding
        neutral = torch.from_numpy(normalize(neutral))[0]  # [28]
        channels = []
        if mode in ("translation_hold", "pose_hold"):
            channels += [0, 1, 2, 14, 15, 16]
        if mode in ("rotation_hold", "pose_hold"):
            channels += [3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22]
        result[history:, channels] = neutral[channels]
    elif mode != "recorded":
        raise ValueError(mode)
    return result


def build_dataset_sample_args(item, model_config, *, name, destination, future, seed):
    """Build validated args while retaining the dataset batch as conditioning.

    The public forward-dynamics argument builder requires a real action file
    even when generate_batch receives in-memory conditioning. Save the exact
    normalized semantic rows, including history, with their sequence metadata.
    The supplied dataset batch remains authoritative for masks and history.
    """
    action_path = destination / "conditioning_actions.json"
    write_json(action_path, item["action"][:, :28].tolist())
    write_json(
        destination / "conditioning_metadata.json",
        {
            "schema": "cmr28-h16-normalized",
            "source_key": item["__key__"],
            "action_start_frame_offset": item["sequence_plan"].action_start_frame_offset,
            "history": 1 - item["sequence_plan"].action_start_frame_offset,
            "future_frames": future,
            "conditioning_source": "in_memory_dataset_batch",
        },
    )
    return OmniSampleOverrides(
        name=name,
        model_mode="forward_dynamics",
        prompt=item["ai_caption"],
        resolution="480",
        raw_action_dim=28,
        action_path=str(action_path.resolve()),
        autoregressive=bool(model_config.video_temporal_causal),
        seed=seed,
        guidance=1.0,
        num_frames=future + 1,
        fps=10,
        num_steps=None if model_config.fixed_step_sampler_config is not None else 35,
        output_dir=destination,
    ).build_sample(model_config=model_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="exported inference.yaml (or bidirectional teacher config)"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=28003)
    parser.add_argument(
        "--future", type=int, default=72, help="72 by default; 216 for a continuous 21.6-second AR probe"
    )
    parser.add_argument("--modes", nargs="+", default=["recorded", "translation_hold", "rotation_hold", "energy_off"])
    parser.add_argument(
        "--full-metrics", action="store_true", help="also SSIM and AlexNet LPIPS (downloads LPIPS backbone)"
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.future < 4 or args.future % 4:
        parser.error("--future must be a positive multiple of four")
    spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation = {
        "status": "running",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata_sha256": file_sha256(Path(args.checkpoint) / ".metadata"),
        "config_sha256": file_sha256(args.config),
        "spec_sha256": file_sha256(args.spec),
        "manifest_sha256": file_sha256(Path(spec["manifests"]["c72"])),
        "split": args.split,
        "seed": args.seed,
        "samples": args.samples,
        "modes": args.modes,
    }
    write_json(args.output / "evaluation.json", evaluation)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    dataset = CholeDataset(
        spec["manifests"]["c72"], spec["data_root"], config["model"]["config"]["vlm_config"]["tokenizer"], args.split
    )
    dataset.future = args.future
    # Native validation happens again for extended trajectories: do not pad,
    # loop or silently substitute another trajectory when it is too short.
    indices = np.unique(np.linspace(0, len(dataset.windows) - 1, args.samples).astype(int)).tolist()
    register_checkpoints()
    setup = OmniSetupOverrides(
        checkpoint_path=args.checkpoint,
        config_file=str(args.config),
        output_dir=args.output,
        use_ema_weights=False,
        guardrails=False,
        use_torch_compile=False,
        use_cuda_graphs=False,
        dp_shard_size=1,
        cp_size=1,
        cfgp_size=1,
    ).build_setup()
    pipe = setup.get_inference_cls().create(setup)
    records = []
    for case, index in enumerate(indices):
        sample = dataset.sample(index)
        gt = sample["video"].clone()  # [3,C+1,H,W], uint8 before inference mutates inputs
        history = 1 - sample["sequence_plan"].action_start_frame_offset
        for mode in args.modes:
            item = copy.deepcopy(sample)
            item["action"] = intervene(item["action"], history, mode)  # [H+C,64]
            name = f"case_{case:03d}_{mode}"
            destination = args.output / name
            sample_args = build_dataset_sample_args(
                item,
                pipe.model_config,
                name=name,
                destination=destination,
                future=args.future,
                seed=args.seed + case,
            )
            conditioning_changed = not torch.equal(item["action"], sample["action"])
            outputs = pipe.generate_batch([sample_args], _collate_sample(item))
            if not outputs or outputs[0].status != "success":
                raise RuntimeError(f"Inference failed: {name}: {outputs}")
            if not is_rank0():
                continue
            prediction, _ = read_media_frames(destination / "vision.mp4", max_frames=args.future + 2)  # [3,C+1,H,W]
            if prediction.shape != gt.shape:
                raise ValueError(f"Strict frame/shape alignment failed: {prediction.shape} vs {gt.shape}")
            record = {
                "name": name,
                "source_key": sample["__key__"],
                "seed": args.seed + case,
                "history": history,
                "future_frames": args.future,
                "mode": mode,
                "conditioning_changed": conditioning_changed,
                "video": f"{name}/vision.mp4",
            }
            if mode == "recorded":
                pred, target = prediction[:, 1:], gt[:, 1:]  # exclude conditioning frame
                baseline = gt[:, :1].expand_as(target)  # [3,C,H,W]
                record["metrics"] = {
                    "l1_0_1": float((pred.float() - target.float()).abs().mean() / 255),
                    "psnr": compute_psnr(target, pred),
                    "repeat_first_psnr": compute_psnr(target, baseline),
                }
                if args.full_metrics:
                    record["metrics"].update(ssim=compute_ssim(target, pred), lpips=_compute_lpips(target, pred))
            records.append(record)
            write_json(
                args.output / "results.json",
                {
                    "config_sha256": file_sha256(args.config),
                    "cases": records,
                    "scoring": "Decoded MP4, all future pixels, excludes conditioning frame; interventions have no GT target",
                    "qualification": "Fixed-window evaluation; no claim of convergence or ancestry-clean held-out performance",
                },
            )
    if is_rank0():
        evaluation["status"] = "completed"
        evaluation["case_count"] = len(records)
        write_json(args.output / "evaluation.json", evaluation)
        cards = "\n".join(
            f'<article><h3>{html.escape(r["name"])}</h3><p>{html.escape(r["source_key"])} · H{r["history"]}</p><video controls muted loop width="640" src="{r["video"]}"></video></article>'
            for r in records
        )
        (args.output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>CMR model evaluation</title><h1>Fixed-action inference checks</h1><p>Hold interventions change future channels only. Other controls and real history remain unchanged.</p>'
            + cards
        )


if __name__ == "__main__":
    main()
