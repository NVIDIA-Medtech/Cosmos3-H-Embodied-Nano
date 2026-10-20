# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Score the rollouts of several models against the same recorded frames at a common resolution.

Inputs are rollout directories written by ``cosmos3_h_embodied.scripts.quant_eval`` and
``tools/benchmark/chss_rollout.py`` (``rollout.npz`` + ``rollout.json`` per rollout). Rollouts are
matched by (root, episode, start frame, seed index); only rollouts that every model completed
are scored. The reference frames are decoded from the Open-H videos at the rollouts' source
frames, cropped to the part of the source frame that every model generates (``source_box``) and
resized to ``--height`` (width from the aspect ratio); each prediction is cropped to that part
(``pred_box``) and resized the same way.

Writes ``results.json`` (per rollout, per dataset, overall, paired differences),
``results.md`` (tables) and ``per_frame.{csv,png}`` (per-frame L1 / PSNR averages per group).

Example::

    python tools/benchmark/compare_rollouts.py --data-root <Open-H>/Surgical \\
        --rollouts c3hen=outputs/quant_eval chss=outputs/chss_rollouts -o outputs/benchmark
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torchcodec.decoders import VideoDecoder

from cosmos3_h_embodied.scripts.rollout_metrics import METRICS, PHASES, rollout_metrics, summarize

HIGHER_IS_BETTER = {"fds": False, "ssim": True, "psnr": True, "dynamic_psnr": True}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", nargs="+", required=True, help="label=directory, one per model")
    p.add_argument("--data-root", type=Path, required=True, help="Open-H 'Surgical' directory")
    p.add_argument("--height", type=int, default=256, help="Common evaluation height in pixels")
    p.add_argument("--groups", type=Path, help="JSON {root: group name} for grouped tables")
    p.add_argument("-o", "--output", type=Path, required=True)
    return p.parse_args()


def index_rollouts(directory: Path) -> dict[tuple, Path]:
    out = {}
    for meta_file in directory.rglob("rollout.json"):
        if (meta_file.parent / "rollout.npz").is_file():
            m = json.loads(meta_file.read_text())
            out[(m["root"], int(m["episode"]), int(m["start_frame"]), int(m["seed_index"]))] = meta_file.parent
    return out


def episode_video(root: Path, episode: int, video_key: str) -> tuple[Path, float, float]:
    """(video file, start timestamp within it, fps) of an episode in a LeRobot v2.1 or v3.0 dataset."""
    info = json.loads((root / "meta" / "info.json").read_text())
    if info["codebase_version"].startswith("v2"):
        chunk = episode // info["chunks_size"]
        rel = info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=episode)
        return root / rel, 0.0, float(info["fps"])
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet")))
    eps = pd.concat([pd.read_parquet(f) for f in files]).set_index("episode_index")
    row = eps.loc[episode]
    rel = info["video_path"].format(
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"]),
    )
    return root / rel, float(row[f"videos/{video_key}/from_timestamp"]), float(info["fps"])


def decode_frames(root: Path, episode: int, video_key: str, frames: list[int]) -> torch.Tensor:
    """Recorded frames as (T, 3, H, W) uint8, selected by timestamp like LeRobot."""
    path, offset, fps = episode_video(root, episode, video_key)
    return VideoDecoder(str(path)).get_frames_played_at(seconds=[offset + f / fps for f in frames]).data


def resize(frames_tchw: torch.Tensor, height: int, width: int) -> torch.Tensor:
    out = F.interpolate(frames_tchw.float(), size=(height, width), mode="bilinear", antialias=True, align_corners=False)
    return out.round().clamp(0, 255).to(torch.uint8)


def crop_box(frames_tchw: torch.Tensor, box: list[float]) -> torch.Tensor:
    """Crop to ``[top, bottom, left, right]`` (fractions of the frame)."""
    h, w = frames_tchw.shape[-2:]
    return frames_tchw[..., round(box[0] * h) : round(box[1] * h), round(box[2] * w) : round(box[3] * w)]


def to_pred_box(region: list[float], pred_box: list[float], source_box: list[float]) -> list[float]:
    """Map a source-frame region to fractions of a generated frame whose ``pred_box`` shows ``source_box``."""
    out = []
    for i in (0, 2):
        p0, p1, s0, s1 = pred_box[i], pred_box[i + 1], source_box[i], source_box[i + 1]
        out += [p0 + (region[i] - s0) / (s1 - s0) * (p1 - p0), p0 + (region[i + 1] - s0) / (s1 - s0) * (p1 - p0)]
    return out


def time_windows(num_generated: int, size: int, fps: float = 10.0) -> dict[str, tuple[int, int]]:
    """Consecutive ``size``-frame windows ``{name: (start, stop)}`` over generated frames 1..num_generated.

    The last window takes the remainder; there are none if ``size`` is 0 or exceeds the rollout.
    """
    starts = list(range(1, num_generated + 2 - size, size)) if size > 0 else []
    stops = starts[1:] + [num_generated + 1]
    return {f"{(a - 1) / fps:.0f}-{(b - 1) / fps:.0f} s": (a, b) for a, b in zip(starts, stops)}


def aligned_clips(
    key: tuple, labels: tuple, indices: list[dict], data_root: Path, height: int
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Reference clip and, per model, (prediction, own input frames), all (T, 3, H, W) uint8 at ``height``."""
    root, episode, _, _ = key
    metas = [json.loads((i[key] / "rollout.json").read_text()) for i in indices]
    frames = metas[0]["source_frames"]
    for m in metas[1:]:
        if m["source_frames"] != frames:
            raise ValueError(f"{key}: models predict different source frames")
    video_key = next(m["video_key"] for m in metas if "video_key" in m)
    source = decode_frames(data_root / root, episode, video_key, frames)
    # Score the part of the source frame that every model generates.
    region = [max(m["source_box"][0] for m in metas), min(m["source_box"][1] for m in metas),
              max(m["source_box"][2] for m in metas), min(m["source_box"][3] for m in metas)]
    source = crop_box(source, region)
    width = 2 * round(height * source.shape[-1] / source.shape[-2] / 2)
    gt = resize(source, height, width)
    clips = {}
    for label, index, meta in zip(labels, indices, metas):
        data = np.load(index[key] / "rollout.npz")
        box = to_pred_box(region, meta["pred_box"], meta["source_box"])
        pred = resize(crop_box(torch.from_numpy(data["pred"]).permute(0, 3, 1, 2), box), height, width)
        own_gt = resize(crop_box(torch.from_numpy(data["gt"]).permute(0, 3, 1, 2), box), height, width)
        clips[label] = (pred, own_gt)
    return gt, clips


def paired(a: list[float], b: list[float], higher_is_better: bool) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    diff = a - b
    better = diff > 0 if higher_is_better else diff < 0
    p = float(wilcoxon(a, b).pvalue) if len(a) > 0 and np.any(diff != 0) else float("nan")
    return {**summarize(diff.tolist()), "win_rate": float(better.mean()) if len(a) else float("nan"), "wilcoxon_p": p}


def main() -> None:
    args = parse_args()
    labels, dirs = zip(*(r.split("=", 1) for r in args.rollouts))
    indices = [index_rollouts(Path(d)) for d in dirs]
    keys = sorted(set.intersection(*(set(i) for i in indices)))
    print(f"{len(keys)} rollouts completed by all of {labels} ({[len(i) for i in indices]})", flush=True)
    groups = json.loads(args.groups.read_text()) if args.groups else {}

    per_rollout = []
    for n, key in enumerate(keys):
        root, episode, start, seed_index = key
        gt, clips = aligned_clips(key, labels, indices, args.data_root, args.height)
        gt_np = gt.permute(0, 2, 3, 1).numpy()
        entry = {"root": root, "episode": episode, "start_frame": start, "seed_index": seed_index, "models": {}}
        for label, (pred, own_gt) in clips.items():
            metrics = rollout_metrics(gt_np, pred.permute(0, 2, 3, 1).numpy())
            # Robustness check: the same scores against the model's own input frames (its resampling).
            own = rollout_metrics(own_gt.permute(0, 2, 3, 1).numpy(), pred.permute(0, 2, 3, 1).numpy())
            metrics["own_reference"] = {m: own[m] for m in METRICS}
            # Agreement of the model's own input pipeline with the decoded reference (frame alignment check).
            mse = ((own_gt.float() - gt.float()) ** 2).mean().item()
            metrics["input_psnr"] = 20 * math.log10(255 / math.sqrt(mse)) if mse > 0 else float("inf")
            entry["models"][label] = metrics
        per_rollout.append(entry)
        scores = "  ".join(f"{lab}: PSNR {entry['models'][lab]['psnr']:.2f}" for lab in labels)
        print(f"[{n + 1}/{len(keys)}] {root} ep {episode} s{seed_index}  {scores}", flush=True)

    def aggregate(entries: list[dict]) -> dict:
        out = {"n": len(entries), "models": {}, "paired": {}}
        for lab in labels:
            ms = [e["models"][lab] for e in entries]
            out["models"][lab] = {m: summarize([x[m] for x in ms]) for m in (*METRICS, "input_psnr")}
            out["models"][lab]["phases"] = {
                p: {m: summarize([x["phases"][p][m] for x in ms if p in x["phases"]]) for m in METRICS} for p in PHASES
            }
        for other in labels[1:]:
            out["paired"][f"{labels[0]}-{other}"] = {
                m: paired(
                    [e["models"][labels[0]][m] for e in entries],
                    [e["models"][other][m] for e in entries],
                    HIGHER_IS_BETTER[m],
                )
                for m in METRICS
            }
            out["paired_own_reference"] = {
                m: paired(
                    [e["models"][labels[0]]["own_reference"][m] for e in entries],
                    [e["models"][other]["own_reference"][m] for e in entries],
                    HIGHER_IS_BETTER[m],
                )
                for m in METRICS
            }
        return out

    by_root, by_group = defaultdict(list), defaultdict(list)
    for e in per_rollout:
        by_root[e["root"]].append(e)
        by_group[groups.get(e["root"], e["root"])].append(e)
    results = {
        "labels": labels,
        "height": args.height,
        "overall": aggregate(per_rollout),
        "groups": {g: aggregate(es) for g, es in sorted(by_group.items())},
        "roots": {r: aggregate(es) for r, es in sorted(by_root.items())},
        "rollouts": per_rollout,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    # Per-frame curves (mean over the rollouts of each group).
    curves = {}
    with open(args.output / "per_frame.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "frame", *(f"{lab}_{m}" for lab in labels for m in ("l1", "psnr"))])
        for g, es in sorted(by_group.items()):
            num = min(len(e["models"][lab]["per_frame"]["l1"]) for e in es for lab in labels)
            curves[g] = {
                lab: {m: np.mean([e["models"][lab]["per_frame"][m][:num] for e in es], axis=0) for m in ("l1", "psnr")}
                for lab in labels
            }
            for t in range(num):
                writer.writerow([g, t + 1, *(float(curves[g][lab][m][t]) for lab in labels for m in ("l1", "psnr"))])
    fig, axes = plt.subplots(2, len(curves), figsize=(4.5 * len(curves), 6), squeeze=False)
    for col, (g, c) in enumerate(curves.items()):
        for lab in labels:
            frames = np.arange(1, len(c[lab]["l1"]) + 1) / 10.0
            axes[0, col].plot(frames, c[lab]["psnr"], label=lab)
            axes[1, col].plot(frames, c[lab]["l1"], label=lab)
        axes[0, col].set_title(f"{g} (n={len(by_group[g])})")
        axes[0, col].set_ylabel("PSNR (dB)")
        axes[1, col].set_ylabel("L1 (FDS per frame)")
        axes[1, col].set_xlabel("seconds after the conditioning frame")
        axes[0, col].legend()
    fig.tight_layout()
    fig.savefig(args.output / "per_frame.png", dpi=120)

    # Markdown tables.
    def fmt(s: dict, m: str) -> str:
        return f"{s['mean']:.4f}" if m in ("fds", "ssim") else f"{s['mean']:.2f}"

    lines = ["| Group | Rollouts | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |", "|---|---:|---|---:|---:|---:|---:|"]
    for g, agg in [*results["groups"].items(), ("**All**", results["overall"])]:
        for lab in labels:
            s = agg["models"][lab]
            lines.append(f"| {g} | {agg['n']} | {lab} | " + " | ".join(fmt(s[m], m) for m in METRICS) + " |")
    lines += ["", "| Group | Phase | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |", "|---|---|---|---:|---:|---:|---:|"]
    for g, agg in results["groups"].items():
        for p in PHASES:
            for lab in labels:
                s = agg["models"][lab]["phases"][p]
                if s["fds"]["n"]:
                    lines.append(f"| {g} | {p} | {lab} | " + " | ".join(fmt(s[m], m) for m in METRICS) + " |")
    pair = f"{labels[0]}-{labels[-1]}"
    lines += ["", f"Paired differences ({pair}; win rate = share of rollouts where {labels[0]} is better):", "",
              "| Group | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |", "|---|---|---:|---:|---:|---:|"]
    for g, agg in [*results["groups"].items(), ("**All**", results["overall"])]:
        if pair in agg["paired"]:
            for m in METRICS:
                d = agg["paired"][pair][m]
                lines.append(f"| {g} | {m} | {d['mean']:+.4f} | ±{d['ci95']:.4f} | {d['win_rate']:.0%} | {d['wilcoxon_p']:.2g} |")
    lines += ["", "Robustness check, each model scored against its own input frames:", "",
              "| Group | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |", "|---|---|---:|---:|---:|---:|"]
    for g, agg in [*results["groups"].items(), ("**All**", results["overall"])]:
        if "paired_own_reference" in agg:
            for m in METRICS:
                d = agg["paired_own_reference"][m]
                lines.append(f"| {g} | {m} | {d['mean']:+.4f} | ±{d['ci95']:.4f} | {d['win_rate']:.0%} | {d['wilcoxon_p']:.2g} |")
    lines += ["", "| Dataset | Rollouts | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |", "|---|---:|---|---:|---:|---:|---:|"]
    for r, agg in results["roots"].items():
        for lab in labels:
            s = agg["models"][lab]
            lines.append(f"| {r} | {agg['n']} | {lab} | " + " | ".join(fmt(s[m], m) for m in METRICS) + " |")
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
