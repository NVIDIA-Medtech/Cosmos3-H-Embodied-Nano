# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Realism (FID, FVD) and long-horizon stability of benchmark rollouts.

The frames are the ones ``compare_rollouts.py`` scores: the decoded reference and each prediction,
cropped to the region every model generates and resized to ``--height``; the conditioning frame is
skipped. Both metrics compare the distribution of generated frames or clips with the distribution
of the recorded ones of the same rollouts (lower is better):

- **FID**: Frechet distance of Inception-v3 pool features (the TensorFlow FID weights, via
  ``torch-fidelity``) of single frames, resized to 299x299.
- **FVD**: Frechet distance of I3D (Kinetics-400) logits of ``--clip-frames``-frame clips taken
  every ``--clip-stride`` frames, resized to 224x224 and scaled to [-1, 1] (the TorchScript I3D of
  StyleGAN-V / common FVD implementations).

Frechet distances depend on the sample size; models are compared on the same rollouts and frames.
Confidence intervals come from a paired bootstrap over clips (episode and start frame; all seeds of
a resampled clip are kept). As a resolution control, each model's own input frames (the recorded
video after the model's resampling) are scored the same way: the best score at that resolution.

With ``--window-frames N`` the scores are also computed per N-frame window of the rollout (the
last window takes the remainder), together with PSNR, SSIM, dynamic PSNR and two no-reference
ratios to the recorded video (1 is ideal): sharpness (mean gradient magnitude) and motion (mean
absolute frame-to-frame change). Blur or a frozen video lowers them, flicker raises motion.

Feature extraction runs on a GPU: run one process per shard, then ``--summarize-only``::

    python tools/benchmark/realism_metrics.py --data-root <Open-H>/Surgical \\
        --inception-weights weights-inception-2015-12-05-6726825d.pth --i3d-weights i3d_torchscript.pt \\
        --rollouts c3hen=outputs/quant_eval chss=outputs/chss_rollouts -o outputs/realism \\
        [--window-frames 60] [--num-shards N --shard-index i]
    python tools/benchmark/realism_metrics.py --summarize-only --rollouts c3hen=... chss=... -o outputs/realism
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from compare_rollouts import aligned_clips, index_rollouts, paired, time_windows
from cosmos3_h_embodied.scripts.rollout_metrics import METRICS, rollout_metrics, summarize

PIXEL_HIGHER_IS_BETTER = {"fds": False, "ssim": True, "psnr": True, "dynamic_psnr": True}
FPS = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", nargs="+", required=True, help="label=directory, one per model")
    p.add_argument("--data-root", type=Path, help="Open-H 'Surgical' directory")
    p.add_argument("--roots", nargs="+", help="Datasets to score (default: all)")
    p.add_argument("--groups", type=Path, help="JSON {root: group name}; scores are pooled per group")
    p.add_argument("--height", type=int, default=176, help="Common evaluation height in pixels")
    p.add_argument("--window-frames", type=int, default=0, help="Also score N-frame windows (0: whole rollouts only)")
    p.add_argument("--clip-frames", type=int, default=16)
    p.add_argument("--clip-stride", type=int, default=8)
    p.add_argument("--inception-weights", type=Path, help="torch-fidelity Inception-v3 weights (.pth)")
    p.add_argument("--i3d-weights", type=Path, help="TorchScript I3D (i3d_torchscript.pt)")
    p.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples for confidence intervals")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("-o", "--output", type=Path, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
class FeatureExtractors:
    def __init__(self, inception_weights: Path, i3d_weights: Path, device: str = "cuda"):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        self.device = device
        self.inception = FeatureExtractorInceptionV3(
            "inception-v3-compat", ["2048"], feature_extractor_weights_path=str(inception_weights)
        ).to(device)
        self.i3d = torch.jit.load(str(i3d_weights), map_location=device).eval()

    @torch.no_grad()
    def frames(self, video_tchw: torch.Tensor, batch: int = 64) -> np.ndarray:
        """(T, 2048) Inception pool features of uint8 frames."""
        out = [self.inception(video_tchw[i : i + batch].to(self.device))[0] for i in range(0, len(video_tchw), batch)]
        return torch.cat(out).float().cpu().numpy()

    @torch.no_grad()
    def clips(self, video_tchw: torch.Tensor, starts: list[int], length: int, batch: int = 8) -> np.ndarray:
        """(len(starts), 400) I3D features of the clips ``video[s : s + length]``."""
        video = F.interpolate(video_tchw.to(self.device).float(), size=(224, 224), mode="bilinear", antialias=True)
        video = video / 127.5 - 1.0
        out = []
        for i in range(0, len(starts), batch):
            x = torch.stack([video[s : s + length] for s in starts[i : i + batch]]).permute(0, 2, 1, 3, 4)
            out.append(self.i3d(x.contiguous(), rescale=False, resize=False, return_features=True))
        return torch.cat(out).float().cpu().numpy() if out else np.zeros((0, 400), dtype=np.float32)


def _gray(video_tchw: torch.Tensor) -> torch.Tensor:
    v = video_tchw.float() / 255.0
    return (0.299 * v[:, 0] + 0.587 * v[:, 1] + 0.114 * v[:, 2]).unsqueeze(1)


def sharpness(video_tchw: torch.Tensor) -> np.ndarray:
    """Per-frame mean Sobel gradient magnitude of the luma."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    g = _gray(video_tchw)
    gx, gy = F.conv2d(g, kx), F.conv2d(g, kx.transpose(2, 3))
    return (gx**2 + gy**2).sqrt().mean(dim=(1, 2, 3)).numpy()


def motion(video_tchw: torch.Tensor) -> np.ndarray:
    """Per-frame mean absolute luma change from the previous frame (frame 0: 0)."""
    g = _gray(video_tchw)
    return np.concatenate([[0.0], (g[1:] - g[:-1]).abs().mean(dim=(1, 2, 3)).numpy()])


# ---------------------------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------------------------
def run_shard(args: argparse.Namespace, labels: tuple, indices: list[dict]) -> None:
    keys = sorted(k for k in set.intersection(*(set(i) for i in indices)) if not args.roots or k[0] in args.roots)
    clips = sorted({k[:3] for k in keys})[args.shard_index :: args.num_shards]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(clips)} clips", flush=True)
    out_dir = args.output / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = FeatureExtractors(args.inception_weights, args.i3d_weights)
    for n, clip in enumerate(clips):
        root, episode, start = clip
        stem = f"{root.replace('/', '__')}_ep{episode}_f{start}"
        if (out_dir / f"{stem}.json").is_file():
            continue
        t0 = time.time()
        seeds = sorted(k[3] for k in keys if k[:3] == clip)
        arrays, meta = {}, {"root": root, "episode": episode, "start_frame": start, "seeds": seeds, "models": {}}
        for seed in seeds:
            gt, aligned = aligned_clips((*clip, seed), labels, indices, args.data_root, args.height)
            num_generated = len(gt) - 1
            starts = list(range(1, num_generated - args.clip_frames + 2, args.clip_stride))
            wins = time_windows(num_generated, args.window_frames, FPS)
            if seed == seeds[0]:
                arrays["reference_fid"] = fx.frames(gt[1:])
                arrays["reference_fvd"] = fx.clips(gt, starts, args.clip_frames)
                arrays["clip_starts"] = np.asarray(starts)
                ref_sharp, ref_motion = sharpness(gt), motion(gt)
                meta.update(num_generated=num_generated, windows=wins)
            for label, (pred, own) in aligned.items():
                if seed == seeds[0]:
                    arrays[f"{label}_input_fid"] = fx.frames(own[1:])
                    arrays[f"{label}_input_fvd"] = fx.clips(own, starts, args.clip_frames)
                arrays[f"{label}_s{seed}_fid"] = fx.frames(pred[1:])
                arrays[f"{label}_s{seed}_fvd"] = fx.clips(pred, starts, args.clip_frames)
                per_frame = {
                    "sharpness": (sharpness(pred) / np.maximum(ref_sharp, 1e-6))[1:].tolist(),
                    "motion": (motion(pred) / np.maximum(ref_motion, 1e-6))[1:].tolist(),
                }
                gt_np, pred_np = gt.permute(0, 2, 3, 1).numpy(), pred.permute(0, 2, 3, 1).numpy()
                per_frame["psnr"] = rollout_metrics(gt_np, pred_np)["per_frame"]["psnr"]
                # Window k scores frames [start, stop); rollout_metrics skips the frame before it.
                win_scores = {
                    name: {m: rollout_metrics(gt_np[a - 1 : b], pred_np[a - 1 : b])[m] for m in METRICS}
                    for name, (a, b) in wins.items()
                }
                meta["models"].setdefault(label, {})[str(seed)] = {"per_frame": per_frame, "windows": win_scores}
        np.savez(out_dir / f"{stem}.npz", **arrays)
        (out_dir / f"{stem}.json").write_text(json.dumps(meta) + "\n")
        print(f"[{n + 1}/{len(clips)}] {root} ep {episode} f{start}: {len(seeds)} seeds ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------
def frechet_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Frechet distance between Gaussians fitted to the rows of ``x`` and ``y`` (float64)."""
    mu_x, mu_y = x.mean(0), y.mean(0)
    cov_x, cov_y = torch.cov(x.T), torch.cov(y.T)
    # tr sqrt(cov_x cov_y) = sum sqrt(eig(A cov_y A)) with A = cov_x^(1/2), all symmetric.
    lam, vec = torch.linalg.eigh(cov_x)
    a = (vec * lam.clamp(min=0).sqrt()) @ vec.T
    tr_covmean = torch.linalg.eigvalsh(a @ cov_y @ a).clamp(min=0).sqrt().sum()
    return float((mu_x - mu_y).square().sum() + cov_x.trace() + cov_y.trace() - 2 * tr_covmean)


def summarize_runs(args: argparse.Namespace, labels: tuple) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_files = sorted((args.output / "clips").glob("*.json"))
    clips = [json.loads(f.read_text()) | {"npz": f.with_suffix(".npz")} for f in clip_files]
    clips = [c for c in clips if not args.roots or c["root"] in args.roots]
    groups = json.loads(args.groups.read_text()) if args.groups else {}
    pools = defaultdict(list)
    for c in clips:
        pools[groups.get(c["root"], "All")].append(c)
    rng = np.random.default_rng(0)
    first, other = labels[0], labels[-1]
    results = {"labels": labels, "height": args.height, "clip_frames": args.clip_frames,
               "clip_stride": args.clip_stride, "bootstrap": args.bootstrap, "pools": {}}

    for pool, members in sorted(pools.items()):
        data = [np.load(c["npz"]) for c in members]
        num_generated = min(c["num_generated"] for c in members)
        wins = [("all", 1, num_generated + 1)] + [(k, *v) for k, v in members[0]["windows"].items()]

        def gather(kind: str, name: str, a: int, b: int) -> list[torch.Tensor]:
            """Per clip, the rows of feature set ``name`` inside frames [a, b) (all seeds for a model)."""
            out = []
            for c, d in zip(members, data):
                if kind == "fid":
                    sel = slice(a - 1, b - 1)
                else:
                    starts = d["clip_starts"]
                    sel = np.flatnonzero((starts >= a) & (starts + args.clip_frames <= b))
                keys = [name] if name in ("reference",) or name.endswith("_input") else [f"{name}_s{s}" for s in c["seeds"]]
                out.append(torch.from_numpy(np.concatenate([d[f"{k}_{kind}"][sel] for k in keys])).double().to(device))
            return out

        pool_out = {"n_clips": len(members), "n_rollouts": sum(len(c["seeds"]) for c in members), "windows": {}}
        for wname, a, b in wins:
            w_out = {"frames": [a, b]}
            for kind in ("fid", "fvd"):
                sets = {lab: gather(kind, lab, a, b) for lab in labels}
                sets |= {f"{lab} input": gather(kind, f"{lab}_input", a, b) for lab in labels}
                ref = gather(kind, "reference", a, b)
                if sum(len(r) for r in ref) < 2:
                    continue
                scores = {k: frechet_distance(torch.cat(v), torch.cat(ref)) for k, v in sets.items()}
                diffs = []
                for _ in range(args.bootstrap):
                    idx = rng.integers(0, len(members), len(members))
                    r = torch.cat([ref[i] for i in idx])
                    diffs.append(frechet_distance(torch.cat([sets[first][i] for i in idx]), r)
                                 - frechet_distance(torch.cat([sets[other][i] for i in idx]), r))
                diffs = np.asarray(diffs)
                w_out[kind] = {
                    "scores": scores,
                    "samples": {"reference": sum(len(r) for r in ref), first: sum(len(r) for r in sets[first])},
                    "paired": {"difference": scores[first] - scores[other],
                               "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                               "share_better": float((diffs < 0).mean())} if len(diffs) else {},
                }
            if wname != "all":
                per = {lab: [c["models"][lab][str(s)]["windows"][wname] for c in members for s in c["seeds"]] for lab in labels}
                w_out["pixel"] = {lab: {m: summarize([x[m] for x in per[lab]]) for m in METRICS} for lab in labels}
                w_out["pixel_paired"] = {
                    m: paired([x[m] for x in per[first]], [x[m] for x in per[other]], PIXEL_HIGHER_IS_BETTER[m])
                    for m in METRICS
                }
            ratios = {}
            for lab in labels:
                pf = [c["models"][lab][str(s)]["per_frame"] for c in members for s in c["seeds"]]
                ratios[lab] = {r: summarize([float(np.mean(p[r][a - 1 : b - 1])) for p in pf]) for r in ("sharpness", "motion")}
            w_out["ratios"] = ratios
            pool_out["windows"][wname] = w_out
            print(f"{pool} {wname}: " + "  ".join(
                f"{k.upper()} " + " / ".join(f"{lab} {w_out[k]['scores'][lab]:.1f}" for lab in labels)
                for k in ("fid", "fvd") if k in w_out), flush=True)
        pool_out["per_frame"] = {
            lab: {r: np.nanmean(np.asarray([c["models"][lab][str(s)]["per_frame"][r][:num_generated]
                                            for c in members for s in c["seeds"]], dtype=np.float64), axis=0).tolist()
                  for r in ("psnr", "sharpness", "motion")}
            for lab in labels
        }
        results["pools"][pool] = pool_out

    (args.output / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    lines = write_tables(results, labels)
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_horizon(results, labels, args.output / "horizon.png")


def write_tables(results: dict, labels: tuple) -> list[str]:
    first, other = labels[0], labels[-1]
    lines = [f"Frames at {results['height']} rows; FVD on {results['clip_frames']}-frame clips every "
             f"{results['clip_stride']} frames. Difference = {first} - {other} (lower is better); 95% CI and the "
             f"share of {results['bootstrap']} paired bootstrap resamples (over clips) in which {first} is better.", ""]
    head = ["| Pool | Window | Metric | " + " | ".join(labels) + " | " + " | ".join(f"{lab} input" for lab in labels)
            + " | Difference | 95% CI | Share better |", "|---|---|---|" + "---:|" * (2 * len(labels) + 3)]
    lines += head
    for pool, p in results["pools"].items():
        for wname, w in p["windows"].items():
            for kind in ("fid", "fvd"):
                if kind not in w:
                    continue
                s, d = w[kind]["scores"], w[kind]["paired"]
                lines.append(
                    f"| {pool} | {wname} | {kind.upper()} | " + " | ".join(f"{s[lab]:.1f}" for lab in labels) + " | "
                    + " | ".join(f"{s[f'{lab} input']:.1f}" for lab in labels)
                    + f" | {d['difference']:+.1f} | [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] | {d['share_better']:.0%} |"
                )
    has_windows = any(len(p["windows"]) > 1 for p in results["pools"].values())
    if has_windows:
        lines += ["", "Per window: pixel metrics (mean over rollouts) and ratios to the recorded video (1 is ideal):", "",
                  "| Pool | Window | Model | PSNR | SSIM | Dynamic PSNR | Sharpness | Motion |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for pool, p in results["pools"].items():
            for wname, w in p["windows"].items():
                if "pixel" not in w:
                    continue
                for lab in labels:
                    px, r = w["pixel"][lab], w["ratios"][lab]
                    lines.append(f"| {pool} | {wname} | {lab} | {px['psnr']['mean']:.2f} | {px['ssim']['mean']:.3f} | "
                                 f"{px['dynamic_psnr']['mean']:.2f} | {r['sharpness']['mean']:.2f} | {r['motion']['mean']:.2f} |")
        lines += ["", f"Per window, paired pixel differences ({first} - {other}):", "",
                  "| Pool | Window | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |", "|---|---|---|---:|---:|---:|---:|"]
        for pool, p in results["pools"].items():
            for wname, w in p["windows"].items():
                for m in ("psnr", "ssim", "dynamic_psnr"):
                    if "pixel_paired" in w:
                        d = w["pixel_paired"][m]
                        lines.append(f"| {pool} | {wname} | {m} | {d['mean']:+.4f} | ±{d['ci95']:.4f} | "
                                     f"{d['win_rate']:.0%} | {d['wilcoxon_p']:.2g} |")
    else:
        lines += ["", "Ratios to the recorded video (1 is ideal):", "", "| Pool | Model | Sharpness | Motion |", "|---|---|---:|---:|"]
        for pool, p in results["pools"].items():
            for lab in labels:
                r = p["windows"]["all"]["ratios"][lab]
                lines.append(f"| {pool} | {lab} | {r['sharpness']['mean']:.2f} | {r['motion']['mean']:.2f} |")
    return lines


def plot_horizon(results: dict, labels: tuple, path: Path) -> None:
    pools = list(results["pools"].items())
    has_windows = any(n != "all" and "fid" in w for _, p in pools for n, w in p["windows"].items())
    cols = 4 if has_windows else 3
    fig, axes = plt.subplots(len(pools), cols, figsize=(4.5 * cols, 3.6 * len(pools)), squeeze=False)
    for row, (pool, p) in zip(axes, pools):
        for ax, r, title in zip(row[:3], ("psnr", "sharpness", "motion"), ("PSNR (dB)", "Sharpness ratio", "Motion ratio")):
            for lab in labels:
                y = np.asarray(p["per_frame"][lab][r])
                t = np.arange(1, len(y) + 1) / FPS
                k = 5  # centered 0.5 s moving average for readability
                ax.plot(t[k // 2 : len(t) - k // 2], np.convolve(y, np.ones(k) / k, mode="valid"), label=lab)
            if r != "psnr":
                ax.axhline(1.0, color="gray", lw=0.8, ls="--")
            ax.set_title(f"{pool}: {title}")
            ax.set_xlabel("time since conditioning frame (s)")
        wins = [(n, w) for n, w in p["windows"].items() if n != "all" and "fid" in w]
        if has_windows:
            x = np.arange(len(wins))
            for i, lab in enumerate(labels):
                row[3].bar(x + (i - 0.5) * 0.4, [w["fid"]["scores"][lab] for _, w in wins], width=0.4, label=lab)
            row[3].set_xticks(x, [n for n, _ in wins])
            row[3].set_title(f"{pool}: FID per window")
        row[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labels, dirs = zip(*(r.split("=", 1) for r in args.rollouts))
    if not args.summarize_only:
        if not (args.data_root and args.inception_weights and args.i3d_weights):
            raise SystemExit("--data-root, --inception-weights and --i3d-weights are required unless --summarize-only")
        run_shard(args, labels, [index_rollouts(Path(d)) for d in dirs])
    if args.summarize_only or args.num_shards == 1:
        summarize_runs(args, labels)


if __name__ == "__main__":
    main()
