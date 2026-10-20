# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool metrics (GATC, TCD) of benchmark rollouts, with a Medical-SAM3 surgical tool segmenter.

GT-Anchored Tool Consistency (GATC, in [-1, 1], higher is better) and Tool Centroid Distance (TCD,
pixels, lower is better) are ported unchanged from Cosmos-H-Surgical-Simulator's
``scripts/cosmos_h_surgical_simulator_quant_eval.py`` (see its ``scripts/README_EVALUATION.md``),
including the segmenter settings (text prompt "surgical tool", score threshold 0.3). The frames are
the ones ``compare_rollouts.py`` scores: the decoded reference and each prediction, cropped to the
region every model generates and resized to ``--height``; the conditioning frame is skipped. TCD
is in pixels at that resolution.

The segmenter is fine-tuned on laparoscopic video (CholecSeg8k), so use these metrics on laparoscopic
video only: the default ``--roots`` are the four CMR Versius procedures.

As a resolution control, each model's own input frames (the recorded video after the model's
resampling) are scored the same way, once per clip: what a perfect prediction at that model's
resolution would get.

The per-phase medians cover the phases of ``rollout_metrics.py`` (the first 7.2 s); with
``--window-frames N`` they cover consecutive N-frame windows instead (long rollouts).

Segmentation runs on a GPU: run one process per shard, then ``--summarize-only``::

    python tools/benchmark/tool_metrics.py --data-root <Open-H>/Surgical \\
        --sam3-repo <Medical-SAM3 checkout> --sam3-checkpoint <fine-tuned checkpoint> \\
        --rollouts c3hen=outputs/quant_eval chss=outputs/chss_rollouts -o outputs/tool_metrics \\
        [--num-shards N --shard-index i]
    python tools/benchmark/tool_metrics.py --summarize-only --rollouts c3hen=... chss=... -o outputs/tool_metrics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from compare_rollouts import aligned_clips, index_rollouts, paired, time_windows
from cosmos3_h_embodied.scripts.rollout_metrics import PHASES, summarize

TOOL_METRICS = ("gatc", "tcd")
HIGHER_IS_BETTER = {"gatc": True, "tcd": False}
DEFAULT_ROOTS = [
    "cmr_surgical/cholecystectomy",
    "cmr_surgical/hysterectomy",
    "cmr_surgical/inguinal_hernia",
    "cmr_surgical/prostatectomy",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", nargs="+", required=True, help="label=directory, one per model")
    p.add_argument("--data-root", type=Path, help="Open-H 'Surgical' directory")
    p.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Datasets to score")
    p.add_argument("--height", type=int, default=256, help="Common evaluation height in pixels")
    p.add_argument("--sam3-repo", type=Path, help="Medical-SAM3 checkout (provides the sam3 package)")
    p.add_argument("--sam3-checkpoint", type=Path, help="Medical-SAM3 checkpoint fine-tuned for surgical tools")
    p.add_argument("--prompt", default="surgical tool")
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--gatc-k", type=int, default=3, help="GATC translation search radius in pixels")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--overlays", type=int, default=3, help="Save mask overlays of the first N clips of a shard")
    p.add_argument("--window-frames", type=int, default=0, help="Report N-frame windows instead of the phases")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("-o", "--output", type=Path, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------------------------
class ToolSegmenter:
    """Per-instance tool masks from Medical-SAM3 with a text prompt, as in Cosmos-H-Surgical-Simulator."""

    def __init__(self, repo: Path, checkpoint: Path, prompt: str, score_threshold: float):
        sys.path.insert(0, str(repo))
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        state = ckpt["model"] if isinstance(ckpt.get("model"), dict) else ckpt
        if "detector." in next(iter(state)):
            state = {k.replace("detector.", ""): v for k, v in state.items() if "detector" in k}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"segmenter: {len(state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
        if missing or unexpected:
            print(f"  missing (first 5): {missing[:5]}\n  unexpected (first 5): {unexpected[:5]}", flush=True)
        self.processor = Sam3Processor(model, confidence_threshold=0.1)
        self.prompt = prompt
        self.score_threshold = score_threshold

    def instances(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """Binary (H, W) uint8 masks, one per detected tool."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous())
            state = self.processor.set_text_prompt(self.prompt, state)
        masks = state["masks"].reshape(-1, *frame_rgb.shape[:2]).cpu().numpy()
        scores = state["scores"].float().cpu().numpy()
        return [m.astype(np.uint8) for m in masks[scores >= self.score_threshold] if m.any()]

    def video(self, frames_thwc: np.ndarray) -> list[list[np.ndarray]]:
        return [self.instances(f) for f in frames_thwc]


def union(instances: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for m in instances:
        out = np.maximum(out, m)
    return out


# ---------------------------------------------------------------------------------------------
# GATC and TCD (ported from Cosmos-H-Surgical-Simulator)
# ---------------------------------------------------------------------------------------------
@dataclass
class GATCConfig:
    k_translation_px: int = 3
    dilate_radius_px: int = 10
    time_tolerance: tuple[int, ...] = (-1, 0, 1)
    eps: float = 1e-6
    min_mask_pixels: int = 50


def _gray(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0


def _masked_zncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray, eps: float) -> float:
    if mask.sum() < 10:
        return float("nan")
    x, y = a[mask], b[mask]
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).mean() / (np.sqrt((x * x).mean()) * np.sqrt((y * y).mean()) + eps))


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def gatc_per_frame(gt: np.ndarray, gen: np.ndarray, gt_masks: list[np.ndarray], cfg: GATCConfig) -> np.ndarray:
    """Per-frame GATC: best masked ZNCC over small shifts (±k px, ±1 frame) times a tool-presence factor.

    NaN where the dilated reference tool mask has fewer than ``min_mask_pixels`` pixels.
    """
    num = min(len(gt), len(gen), len(gt_masks))
    gt_gray, gen_gray = [_gray(f) for f in gt[:num]], [_gray(f) for f in gen[:num]]
    gt_grad, gen_grad = [_grad_mag(g) for g in gt_gray], [_grad_mag(g) for g in gen_gray]
    out = np.full(num, np.nan, dtype=np.float32)
    k = cfg.k_translation_px
    for t in range(num):
        m = _dilate(gt_masks[t].astype(bool), cfg.dilate_radius_px)
        if int(m.sum()) < cfg.min_mask_pixels:
            continue
        # Tool presence: gradient energy of the generated frame inside the mask, relative to the reference.
        presence = min(1.0, float(gen_grad[t][m].mean()) / (float(gt_grad[t][m].mean()) + cfg.eps))
        best = -np.inf
        for dt in cfg.time_tolerance:
            if not 0 <= t + dt < num:
                continue
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    s = _masked_zncc(gt_gray[t], _shift(gen_gray[t + dt], dx, dy), m, cfg.eps)
                    if not math.isnan(s) and s > best:
                        best = s
        if best > -np.inf:
            out[t] = best * presence
    return out


def _centroids(instances: list[np.ndarray], min_area_px: int = 50) -> np.ndarray:
    out = []
    for m in instances:
        if int(m.sum()) < min_area_px:
            continue
        mo = cv2.moments(m, binaryImage=True)
        if mo["m00"] > 0:
            out.append((mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]))
    return np.asarray(out, dtype=np.float32).reshape(-1, 2)


def tcd_per_frame(
    gt_inst: list[list[np.ndarray]], gen_inst: list[list[np.ndarray]], height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame TCD: Hungarian-matched centroid distance; an unmatched reference tool costs half the diagonal.

    Returns TCD, the mean distance of the matched tools only, and the share of reference tools left
    unmatched (diagnostics), each NaN where the reference frame has no tool.
    """
    d_miss = 0.5 * math.hypot(height, width)
    num = min(len(gt_inst), len(gen_inst))
    tcd, matched, missed = (np.full(num, np.nan, dtype=np.float32) for _ in range(3))
    for t in range(num):
        c_gt, c_gen = _centroids(gt_inst[t]), _centroids(gen_inst[t])
        if len(c_gt) == 0:
            continue
        if len(c_gen) == 0:
            tcd[t], missed[t] = d_miss, 1.0
            continue
        d = np.linalg.norm(c_gt[:, None] - c_gen[None], axis=2)
        rows, cols = linear_sum_assignment(d)
        tcd[t] = (d[rows, cols].sum() + (len(c_gt) - len(rows)) * d_miss) / len(c_gt)
        matched[t] = d[rows, cols].mean()
        missed[t] = 1.0 - len(rows) / len(c_gt)
    return tcd, matched, missed


def summarize_frames(per_frame: np.ndarray, phases: dict[str, tuple[int, int]] = PHASES) -> dict:
    """Median over valid frames (the rollout score), coverage (% valid frames) and per-phase medians."""
    valid = np.isfinite(per_frame)
    out = {
        "score": float(np.median(per_frame[valid])) if valid.any() else float("nan"),
        "coverage": float(valid.mean() * 100.0) if valid.size else 0.0,
        "phases": {},
    }
    for name, (start, stop) in phases.items():
        seg = per_frame[start - 1 : stop - 1]  # per_frame[0] is generated frame 1
        if stop - 1 <= len(per_frame) and np.isfinite(seg).any():
            out["phases"][name] = float(np.nanmedian(seg))
    return out


# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------
def save_overlay(path: Path, gt: np.ndarray, gen: dict, gt_inst: list, gen_inst: dict, frame: int) -> None:
    """Reference and generated frames side by side, each detected tool tinted."""
    colors = np.array([(60, 255, 60), (255, 80, 80), (80, 160, 255), (255, 220, 60)], dtype=np.float32)

    def draw(img: np.ndarray, inst: list[np.ndarray]) -> np.ndarray:
        img = img.copy()
        for i, m in enumerate(inst):
            img[m > 0] = (0.5 * img[m > 0] + 0.5 * colors[i % len(colors)]).astype(np.uint8)
        return img

    row = [draw(gt[frame], gt_inst[frame])] + [draw(v[frame], gen_inst[k][frame]) for k, v in gen.items()]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(row, axis=1), cv2.COLOR_RGB2BGR))


def score_clip(
    segmenter: ToolSegmenter, gt: np.ndarray, gt_inst: list, gt_masks: list, gen: np.ndarray, cfg: GATCConfig,
    phases: dict[str, tuple[int, int]] = PHASES,
) -> tuple[dict, list]:
    """GATC, TCD and the TCD diagnostics of one generated clip; also returns its tool instances."""
    gen_inst = segmenter.video(gen)
    tcd, matched, missed = tcd_per_frame(gt_inst, gen_inst, *gt.shape[1:3])
    out = {
        "gatc": summarize_frames(gatc_per_frame(gt, gen, gt_masks, cfg), phases),
        "tcd": summarize_frames(tcd, phases),
        "tcd_matched_px": float(np.nanmean(matched)) if np.isfinite(matched).any() else float("nan"),
        "missed_tools": float(np.nanmean(missed)) if np.isfinite(missed).any() else float("nan"),
        "tools_per_frame": float(np.mean([len(i) for i in gen_inst])),
    }
    return out, gen_inst


def run_shard(args: argparse.Namespace, labels: tuple, indices: list[dict]) -> None:
    keys = sorted(k for k in set.intersection(*(set(i) for i in indices)) if k[0] in args.roots)
    # Shard by clip so that the seeds of a clip share the reference segmentation.
    clips = sorted({k[:3] for k in keys})
    mine = set(clips[args.shard_index :: args.num_shards])
    keys = [k for k in keys if k[:3] in mine]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(keys)} rollouts of {len(mine)} clips", flush=True)
    out_dir = args.output / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    segmenter = ToolSegmenter(args.sam3_repo, args.sam3_checkpoint, args.prompt, args.score_threshold)
    cfg = GATCConfig(k_translation_px=args.gatc_k)
    cache: dict[tuple, dict] = {}
    overlays = 0
    for n, key in enumerate(keys):
        root, episode, start, seed_index = key
        out_file = out_dir / f"{root.replace('/', '__')}_ep{episode}_f{start}_s{seed_index}.json"
        if out_file.is_file():
            continue
        t0 = time.time()
        gt, clips_ = aligned_clips(key, labels, indices, args.data_root, args.height)
        gt_np = gt.permute(0, 2, 3, 1).numpy()[1:]  # skip the conditioning frame
        phases = time_windows(len(gt_np), args.window_frames) if args.window_frames else PHASES
        if key[:3] not in cache:
            gt_inst = segmenter.video(gt_np)
            gt_masks = [union(i, gt_np.shape[1:3]) for i in gt_inst]
            # Resolution control: each model's own input frames, scored like a prediction (once per clip).
            inputs = {
                label: score_clip(segmenter, gt_np, gt_inst, gt_masks, own.permute(0, 2, 3, 1).numpy()[1:], cfg, phases)[0]
                for label, (_, own) in clips_.items()
            }
            cache = {key[:3]: {"inst": gt_inst, "masks": gt_masks, "inputs": inputs}}
        ref = cache[key[:3]]
        entry = {
            "root": root, "episode": episode, "start_frame": start, "seed_index": seed_index,
            "reference_tools_per_frame": float(np.mean([len(i) for i in ref["inst"]])), "models": {},
        }
        gen_np, gen_inst = {}, {}
        for label, (pred, _) in clips_.items():
            gen_np[label] = pred.permute(0, 2, 3, 1).numpy()[1:]
            entry["models"][label], gen_inst[label] = score_clip(
                segmenter, gt_np, ref["inst"], ref["masks"], gen_np[label], cfg, phases
            )
            if seed_index == 0:
                entry["models"][label]["input"] = ref["inputs"][label]
        if overlays < args.overlays and seed_index == 0:
            for f in (0, len(gt_np) // 2, len(gt_np) - 1):
                save_overlay(args.output / f"overlay_{out_file.stem}_t{f + 1}.jpg", gt_np, gen_np, ref["inst"], gen_inst, f)
            overlays += 1
        out_file.write_text(json.dumps(entry, indent=1) + "\n")
        scores = "  ".join(
            f"{lab}: GATC {m['gatc']['score']:.3f} TCD {m['tcd']['score']:.1f}" for lab, m in entry["models"].items()
        )
        print(f"[{n + 1}/{len(keys)}] {root} ep {episode} s{seed_index} ({time.time() - t0:.0f} s)  {scores}", flush=True)


def summarize_runs(args: argparse.Namespace, labels: tuple) -> None:
    entries = [json.loads(f.read_text()) for f in sorted((args.output / "rollouts").glob("*.json"))]
    entries = [e for e in entries if e["root"] in args.roots]
    phase_names = list(dict.fromkeys(p for e in entries for m in e["models"].values() for p in m["gatc"]["phases"]))

    def stats(ms: list[dict]) -> dict:
        out = {
            m: {**summarize([x[m]["score"] for x in ms]),
                "median": float(np.nanmedian([x[m]["score"] for x in ms])) if ms else float("nan"),
                "coverage": float(np.mean([x[m]["coverage"] for x in ms])) if ms else float("nan"),
                "phases": {p: summarize([x[m]["phases"][p] for x in ms if p in x[m]["phases"]]) for p in phase_names}}
            for m in TOOL_METRICS
        }
        for d in ("tcd_matched_px", "missed_tools", "tools_per_frame"):
            out[d] = summarize([x[d] for x in ms])
        return out

    def aggregate(es: list[dict]) -> dict:
        out = {"n": len(es), "reference_tools_per_frame": summarize([e["reference_tools_per_frame"] for e in es]),
               "models": {}, "inputs": {}, "paired": {}}
        for lab in labels:
            out["models"][lab] = stats([e["models"][lab] for e in es])
            out["inputs"][lab] = stats([e["models"][lab]["input"] for e in es if "input" in e["models"][lab]])
        for other in labels[1:]:
            out["paired"][f"{labels[0]}-{other}"] = {
                m: paired([e["models"][labels[0]][m]["score"] for e in es], [e["models"][other][m]["score"] for e in es],
                          HIGHER_IS_BETTER[m])
                for m in TOOL_METRICS
            }
        return out

    by_root = defaultdict(list)
    for e in entries:
        by_root[e["root"]].append(e)
    results = {
        "labels": labels, "height": args.height, "prompt": args.prompt, "score_threshold": args.score_threshold,
        "overall": aggregate(entries),
        "roots": {r: aggregate(es) for r, es in sorted(by_root.items())},
        "rollouts": entries,
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=1) + "\n")

    def row(name: str, n: int, lab: str, s: dict) -> str:
        return (f"| {name} | {n} | {lab} | {s['gatc']['mean']:.3f} / {s['gatc']['median']:.3f} | "
                f"{s['tcd']['mean']:.1f} / {s['tcd']['median']:.1f} | {s['tcd_matched_px']['mean']:.1f} | "
                f"{s['missed_tools']['mean']:.0%} | {s['tools_per_frame']['mean']:.2f} |")

    pair = f"{labels[0]}-{labels[-1]}"
    head = ["| Dataset | Rollouts | Model | GATC ↑ (mean / median) | TCD px ↓ (mean / median) | Matched distance px | "
            "Missed tools | Tools per frame |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    rows = [*results["roots"].items(), ("**All**", results["overall"])]
    lines = list(head)
    for r, agg in rows:
        lines.append(f"| {r} | {agg['n']} | reference | | | | | {agg['reference_tools_per_frame']['mean']:.2f} |")
        lines += [row(r, agg["n"], lab, agg["models"][lab]) for lab in labels]
    lines += ["", "Resolution control: each model's own input frames scored as if they were its prediction (one per clip):",
              "", *head]
    for r, agg in rows:
        lines += [row(r, agg["inputs"][lab]["gatc"]["n"], f"{lab} input", agg["inputs"][lab]) for lab in labels]
    lines += ["", "| Phase | Model | GATC ↑ | TCD px ↓ |", "|---|---|---:|---:|"]
    for p in phase_names:
        for lab in labels:
            s = results["overall"]["models"][lab]
            if s["gatc"]["phases"][p]["n"]:
                lines.append(f"| {p} | {lab} | {s['gatc']['phases'][p]['mean']:.3f} | {s['tcd']['phases'][p]['mean']:.1f} |")
    lines += ["", f"Paired differences ({pair}; win rate = share of rollouts where {labels[0]} is better):", "",
              "| Dataset | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |", "|---|---|---:|---:|---:|---:|"]
    for r, agg in rows:
        if pair in agg["paired"]:
            for m in TOOL_METRICS:
                d = agg["paired"][pair][m]
                lines.append(f"| {r} | {m} | {d['mean']:+.4f} | ±{d['ci95']:.4f} | {d['win_rate']:.0%} | {d['wilcoxon_p']:.2g} |")
    (args.output / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    labels, dirs = zip(*(r.split("=", 1) for r in args.rollouts))
    if not args.summarize_only:
        if not (args.data_root and args.sam3_repo and args.sam3_checkpoint):
            raise SystemExit("--data-root, --sam3-repo and --sam3-checkpoint are required unless --summarize-only")
        run_shard(args, labels, [index_rollouts(Path(d)) for d in dirs])
    if args.summarize_only or args.num_shards == 1:
        summarize_runs(args, labels)


if __name__ == "__main__":
    main()
