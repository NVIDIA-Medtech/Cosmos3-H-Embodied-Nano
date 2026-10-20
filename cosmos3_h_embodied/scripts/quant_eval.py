# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Quantitative evaluation: autoregressive forward-dynamics rollouts on held-out episodes.

From an episode's first frame, the model predicts the video chunk by chunk with the recorded
actions (24 frames at 10 fps per chunk, 3 chunks = 7.2 s by default). Each chunk is conditioned
on the last generated frame of the previous one and on the recorded action history. The rollout
is scored against the recorded video: FDS, SSIM, PSNR and dynamic PSNR, overall and for the
early / mid / late phases (see ``cosmos3_h_embodied/README_EVALUATION.md``).

Example::

    python -m cosmos3_h_embodied.scripts.quant_eval --no-guardrails \\
        --checkpoint-path <dcp dir> --config-file cosmos3_h_embodied/configs/config.py \\
        --experiment openh_h16_task_prompt --dataset.dataset dvrk_hamlyn \\
        --episodes-per-root 3 --num-seeds 2 -o outputs/quant_eval
"""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_TRAINING": "1",
    }
)

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import torch
import torchvision.io
import tyro

from cosmos3_h_embodied.configs.experiment_h16 import EVAL_DATASET_OVERRIDES
from cosmos3_h_embodied.data.transforms import remove_reflection_padding
from cosmos3_h_embodied.scripts.rollout_metrics import METRICS, PHASES, rollout_metrics, summarize
from cosmos3_h_embodied.scripts.rollout_utils import generate_chunk
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import SetupOverrides, tyro_cli
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.dataset import DatasetArgs, create_dataset
from cosmos_framework.scripts.dataset_utils import set_dataset_mode
from cosmos_framework.utils import log


class QuantEvalArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    setup: tyro.conf.OmitArgPrefixes[SetupOverrides] = OmniSetupOverrides.model_construct()
    """Model and parallelism configuration (``--seed`` is the base seed of the rollouts)."""
    dataset: DatasetArgs = DatasetArgs.model_construct()
    """Family to evaluate (``--dataset.dataset``) and split (default ``val``: held-out episodes)."""
    num_chunks: int = 3
    """Chunks per rollout (24 generated frames each)."""
    episodes_per_root: int = 3
    """Episodes per dataset root (the first ones of the split that are long enough)."""
    num_seeds: int = 2
    """Rollouts per episode, with different seeds."""
    plan: Path | None = None
    """JSON list of rollouts ``{"root", "episode", "start_frame"[, "num_frames"]}`` to run instead of
    the automatic selection (``root`` relative to ``OPENH_DATA_ROOT``; the episode must be in the
    split; ``num_frames``, a multiple of 24, overrides ``num_chunks``). Entries of other families
    are skipped."""
    num_shards: int = 1
    """Split the rollouts over this many independent processes (e.g. one per GPU)..."""
    shard_index: int = 0
    """...of which this process runs the rollouts ``i`` with ``i % num_shards == shard_index``."""
    save_videos: bool = True
    """Write a side-by-side (recorded | generated) MP4 per rollout."""
    save_inputs: bool = False
    """Also write the model inputs of each rollout (``inputs.npz``: first frame, per-chunk actions,
    history and caption, embodiment, seed), which ``scripts/simulate.py`` replays."""
    summarize_only: bool = False
    """Only aggregate the rollouts already in the output directory."""


def _window_index(ds) -> dict[tuple[str, int], dict[int, int]]:
    """(root, episode) -> {window start (episode-relative source frame): dataset index}."""
    stride = ds._sample_stride
    out: dict[tuple[str, int], dict[int, int]] = {}
    span_start = 0
    for span_idx, (ds_idx, _, valid_len, episode) in enumerate(ds._episode_records):
        base = ds._episode_frame_bases[span_idx]
        windows = out.setdefault((os.path.normpath(str(ds._all_shard_roots[ds_idx])), int(episode)), {})
        for j in range(valid_len):
            windows[(base + j) * stride] = span_start + j
        span_start = ds._episode_cum_ends[span_idx]
    return out


def _fetch_exact(ds, idx: int) -> dict:
    """``ds[idx]``, failing instead of silently substituting another window."""
    fetched: list[int] = []
    fetch = ds._fetch_sample

    def recording_fetch(i, *args, **kwargs):
        fetched.append(int(i))
        return fetch(i, *args, **kwargs)

    ds._fetch_sample = recording_fetch
    try:
        raw = ds[idx]
    finally:
        del ds._fetch_sample
    if not fetched or set(fetched) != {idx}:  # CMR fetches the window twice (actions, then video)
        raise RuntimeError(f"dataset substituted windows {fetched} for {idx}")
    return raw


def _record_inputs(inputs: dict[str, list], raw: dict) -> None:
    """Keep a chunk's model inputs as the dataset returns them (before the transform)."""
    if not inputs["frame"]:
        video = raw["video"]
        if video.dtype != torch.uint8:
            raise TypeError(f"expected a uint8 video, got {video.dtype}")
        inputs["frame"].append(video[:, 0].permute(1, 2, 0).numpy().copy())  # [H,W,3], the first chunk's frame 0
    history = raw.get("history_action", torch.zeros((0, raw["action"].shape[1])))
    inputs["action"].append(raw["action"].float().numpy().copy())
    inputs["history"].append(history.float().numpy().copy())
    inputs["prompt"].append(raw["ai_caption"])
    inputs["domain_id"].append(int(raw["domain_id"]))


def _save_inputs(path: Path, inputs: dict[str, list], seed: int) -> None:
    """``inputs.npz`` in the format of ``scripts/simulate.py`` (history zero-padded to 16 rows)."""
    num_hist = [len(h) for h in inputs["history"]]
    dim = inputs["action"][0].shape[1]
    history = np.zeros((len(num_hist), 16, dim), dtype=np.float32)
    for k, h in enumerate(inputs["history"]):
        history[k, 16 - len(h) :] = h
    np.savez_compressed(
        path,
        frame=inputs["frame"][0],
        action=np.stack(inputs["action"]),
        history=history,
        history_length=np.asarray(num_hist),
        prompt=np.asarray(inputs["prompt"]),
        domain_id=np.asarray(inputs["domain_id"][0]),
        seed=np.asarray(seed),
    )


def _root_info(root: str, video_key: str) -> dict[str, Any]:
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    height, width = info["features"][video_key]["shape"][:2]
    return {"source_fps": float(info["fps"]), "source_hw": [int(height), int(width)]}


def _boxes(ds, source_hw: list[int], content_hw: list[int], pred_hw: list[int]) -> tuple[list[float], list[float]]:
    """Which part of the generated frame shows which part of the source frame.

    Returns ``(pred_box, source_box)``, each ``[top, bottom, left, right]`` as fractions of the
    generated frame and of the source frame. The generated frame is the top-left ``pred_hw`` of the
    content region ``content_hw`` (the model's output size is a multiple of 16). The content region
    is the source frame, letterboxed with black bars for CMR (``letterbox_hw``).
    """
    letterbox = getattr(ds, "_letterbox_hw", None)
    canvas = letterbox if letterbox is not None else source_hw
    pred_box, source_box = [], []
    for axis in (0, 1):
        size, src, content, pred = canvas[axis], source_hw[axis], content_hw[axis], pred_hw[axis]
        image = round(src * min(canvas[0] / source_hw[0], canvas[1] / source_hw[1])) if letterbox else src
        offset = (size - image) // 2
        covered = pred / content * size  # canvas extent of the generated frame
        stop = min(covered, offset + image)
        pred_box += [offset / covered, stop / covered]
        source_box += [0.0, (stop - offset) / image]
    return pred_box, source_box


def _write_side_by_side(path: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    video = torch.from_numpy(np.concatenate([gt, pred], axis=2))
    pad_h, pad_w = video.shape[1] % 2, video.shape[2] % 2
    if pad_h or pad_w:
        video = torch.nn.functional.pad(video.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode="replicate")
        video = video.permute(0, 2, 3, 1)
    torchvision.io.write_video(str(path), video, fps=10, video_codec="h264", options={"crf": "18"})


def _select_rollouts(args: QuantEvalArgs, windows, chunk_src: dict[str, int], data_root: str) -> list[dict]:
    """Rollouts ``{"root", "episode", "start_frame", "num_chunks"}`` whose windows all exist in the split."""

    def chain_ok(root: str, episode: int, start: int, num_chunks: int) -> bool:
        w = windows.get((root, episode), {})
        return all(start + k * chunk_src[root] in w for k in range(num_chunks))

    if args.plan is not None:
        rollouts = []
        for r in json.loads(args.plan.read_text()):
            root = os.path.normpath(os.path.join(data_root, r["root"]))
            if root not in chunk_src:
                continue
            num_frames = r.get("num_frames", 24 * args.num_chunks)
            if num_frames % 24:
                raise ValueError(f"plan rollout {r}: num_frames must be a multiple of 24")
            r = {"root": r["root"], "episode": int(r["episode"]), "start_frame": int(r["start_frame"])}
            r["num_chunks"] = num_frames // 24
            if not chain_ok(root, r["episode"], r["start_frame"], r["num_chunks"]):
                raise ValueError(f"plan rollout {r} has no complete window chain in split {args.dataset.dataset_split!r}")
            rollouts.append(r)
        if not rollouts:
            raise ValueError(f"the plan has no rollouts for dataset {args.dataset.dataset!r}")
        return rollouts
    rollouts, per_root = [], defaultdict(int)
    for root, episode in windows:  # split order (for val: a seeded random permutation)
        if per_root[root] >= args.episodes_per_root:
            continue
        start = next((s for s in sorted(windows[(root, episode)]) if chain_ok(root, episode, s, args.num_chunks)), None)
        if start is not None:
            rel = os.path.relpath(root, data_root)
            rollouts.append({"root": rel, "episode": episode, "start_frame": start, "num_chunks": args.num_chunks})
            per_root[root] += 1
    return rollouts


def run_rollouts(args: QuantEvalArgs) -> None:
    if args.dataset.dataset_split in ("val", "full"):
        args.setup.experiment_overrides = [*args.setup.experiment_overrides, *EVAL_DATASET_OVERRIDES]
    args.dataset.model_mode = "forward_dynamics"
    setup = args.setup.build_setup()
    init_output_dir(setup.output_dir)
    register_checkpoints()
    if not os.environ.get("OPENH_DATA_ROOT"):
        raise ValueError("set OPENH_DATA_ROOT to the prepared data directory")
    data_root = os.path.normpath(os.environ["OPENH_DATA_ROOT"])

    samples = create_dataset(args.dataset, config_args=setup)
    ds = samples._dataset
    set_dataset_mode(ds, "forward_dynamics")
    windows = _window_index(ds)
    roots = {root for root, _ in windows}
    root_info = {root: _root_info(root, ds.VIDEO_KEY) for root in roots}
    chunk_src: dict[str, int] = {}
    for root, info in root_info.items():
        step = info["source_fps"] * ds._dt
        if abs(step - round(step)) > 1e-6:
            raise ValueError(f"{root}: source fps {info['source_fps']} is not a multiple of {1 / ds._dt} fps")
        info["frame_step"] = round(step)
        chunk_src[root] = ds._chunk_length * info["frame_step"]

    rollouts = _select_rollouts(args, windows, chunk_src, data_root)
    jobs = [(r, s) for r in rollouts for s in range(args.num_seeds)]
    jobs = jobs[args.shard_index :: args.num_shards]
    log.info(f"{len(rollouts)} episodes x {args.num_seeds} seeds; this shard runs {len(jobs)} rollouts")
    if not jobs:
        return

    pipe = setup.get_inference_cls().create(setup)
    base_seed = setup.sample_overrides.seed or 0
    for i, (rollout, seed_idx) in enumerate(jobs):
        root = os.path.normpath(os.path.join(data_root, rollout["root"]))
        episode, start = int(rollout["episode"]), int(rollout["start_frame"])
        name = f"ep{episode:06d}_f{start:06d}_s{seed_idx}"
        rollout_dir = setup.output_dir / args.dataset.dataset_label / rollout["root"] / name
        if (rollout_dir / "metrics.json").is_file():
            log.info(f"[{i + 1}/{len(jobs)}] {rollout_dir} exists, skipping")
            continue
        rollout_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[{i + 1}/{len(jobs)}] {rollout['root']} episode {episode} from frame {start}, seed {seed_idx}")
        seed = base_seed + 1000 * seed_idx
        gt_chunks, pred_chunks = [], []
        inputs: dict[str, list] = defaultdict(list)
        condition: torch.Tensor | None = None  # last generated frame, (C, 1, h, w) content region
        for k in range(rollout["num_chunks"]):
            idx = windows[(root, episode)][start + k * chunk_src[root]]
            raw = _fetch_exact(ds, idx)
            if args.save_inputs:
                _record_inputs(inputs, raw)
            sample_args, data_batch = samples._make_sample_from_raw(raw, idx, "forward_dynamics")
            gt_chunk = remove_reflection_padding(data_batch["video"][0], data_batch["image_size"][0]).clone()
            sample_args = sample_args.model_copy(
                update={"name": f"{name}/chunk{k}", "output_dir": rollout_dir / f"chunk{k}", "seed": seed + k}
            ).build_sample(model_config=pipe.model_config)
            pred, _ = generate_chunk(pipe, sample_args, data_batch, condition)
            content_hw, pred_hw = list(gt_chunk.shape[-2:]), list(pred.shape[-2:])
            # Score the region both cover (the generated frame can be smaller than the content region).
            gt_chunks.append(gt_chunk[..., : pred_hw[0], : pred_hw[1]])
            pred_chunks.append(pred[..., : content_hw[0], : content_hw[1]])
            condition = pred[:, -1:]
        gt = torch.cat([gt_chunks[0]] + [c[:, 1:] for c in gt_chunks[1:]], dim=1).permute(1, 2, 3, 0).numpy()
        pred = torch.cat([pred_chunks[0]] + [c[:, 1:] for c in pred_chunks[1:]], dim=1).permute(1, 2, 3, 0).numpy()

        info = root_info[root]
        pred_box, source_box = _boxes(ds, info["source_hw"], content_hw, list(pred.shape[1:3]))
        meta = {
            "model": "cosmos3_h_embodied",
            "family": args.dataset.dataset_label,
            "root": rollout["root"],
            "episode": episode,
            "start_frame": start,
            "seed_index": seed_idx,
            "seed": seed,
            "video_key": ds.VIDEO_KEY,
            "source_fps": info["source_fps"],
            "source_hw": info["source_hw"],
            "source_frames": [start + t * info["frame_step"] for t in range(gt.shape[0])],
            "pred_box": pred_box,
            "source_box": source_box,
            "num_chunks": rollout["num_chunks"],
        }
        np.savez_compressed(rollout_dir / "rollout.npz", gt=gt, pred=pred)
        if args.save_inputs:
            _save_inputs(rollout_dir / "inputs.npz", inputs, seed)
        (rollout_dir / "rollout.json").write_text(json.dumps(meta, indent=2) + "\n")
        metrics = rollout_metrics(gt, pred)
        (rollout_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
        log.info(
            f"  FDS {metrics['fds']:.4f}  SSIM {metrics['ssim']:.4f}  PSNR {metrics['psnr']:.2f}  "
            f"dynamic PSNR {metrics['dynamic_psnr']:.2f}"
        )
        if args.save_videos:
            _write_side_by_side(rollout_dir / "comparison.mp4", gt, pred)


def write_summary(output_dir: Path) -> dict[str, Any]:
    """Aggregate all rollouts in ``output_dir`` per dataset root and overall: summary.json, summary.md."""
    by_root: dict[str, list[dict]] = defaultdict(list)
    for meta_file in sorted(output_dir.rglob("rollout.json")):
        metrics_file = meta_file.with_name("metrics.json")
        if metrics_file.is_file():
            meta = json.loads(meta_file.read_text())
            by_root[meta["root"]].append(json.loads(metrics_file.read_text()))

    def aggregate(items: list[dict]) -> dict[str, Any]:
        out: dict[str, Any] = {m: summarize([x[m] for x in items]) for m in METRICS}
        out["phases"] = {
            p: {m: summarize([x["phases"][p][m] for x in items if p in x["phases"]]) for m in METRICS} for p in PHASES
        }
        return out

    summary = {
        "roots": {root: aggregate(items) for root, items in sorted(by_root.items())},
        "overall": aggregate([x for items in by_root.values() for x in items]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "| Dataset | Rollouts | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for root, s in [*summary["roots"].items(), ("**All**", summary["overall"])]:
        lines.append(
            f"| {root} | {s['fds']['n']} | {s['fds']['mean']:.4f} | {s['ssim']['mean']:.4f} | "
            f"{s['psnr']['mean']:.2f} | {s['dynamic_psnr']['mean']:.2f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    args = tyro_cli(QuantEvalArgs, description=__doc__)
    if args.setup.output_dir is None:
        raise ValueError("'output_dir' is required")
    if not args.summarize_only:
        run_rollouts(args)
    summary = write_summary(Path(args.setup.output_dir))
    log.success(f"{summary['overall']['fds']['n']} rollouts summarized in {args.setup.output_dir}/summary.md")


if __name__ == "__main__":
    main()
