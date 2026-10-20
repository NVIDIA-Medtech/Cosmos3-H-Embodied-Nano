# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize ``cosmos3_h_embodied.scripts.eval`` runs of several seeds per dataset family and mode.

Each ``--runs`` directory is the output of one ``eval.py`` run (one seed, any families); samples are
matched across runs by family, mode and dataset index. Forward dynamics: video PSNR; inverse
dynamics: action MSE; policy: action MSE and video PSNR, as the mean over the runs and the best of
them per sample (lowest MSE, highest PSNR), since the policy's action distribution is multimodal.
Action MSE is in the normalized action space (most channels in [-1, 1]) over the predicted rows.
The baseline predicts the family's mean recorded trajectory for every policy sample, so its MSE is
the variance of the recorded actions.

Example::

    python tools/benchmark/summarize_eval.py --runs outputs/eval_s0 outputs/eval_s1 -o outputs/eval_summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(runs: list[Path]) -> dict[tuple[str, str, str], list[dict]]:
    """``{(family, mode, sample index): [metrics of each run]}``; sample dirs are ``<family>/<mode>/<index>``."""
    samples: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        for path in sorted(run.rglob("metrics.json")):
            family, mode, index = path.parts[-4:-1]
            samples[(family, mode, index)].append({**json.loads(path.read_text()), "dir": path.parent})
    return samples


def baseline_mse(sample_dirs: list[Path]) -> float:
    """MSE of predicting the mean recorded trajectory of these samples (``gt_action.json``, scored rows)."""
    gts = [np.asarray(json.loads((d / "gt_action.json").read_text()), dtype=np.float64) for d in sample_dirs]
    if not gts or len({g.shape for g in gts}) != 1:
        return float("nan")
    g = np.stack(gts)
    return float(((g - g.mean(axis=0)) ** 2).mean())


def summarize(samples: dict[tuple[str, str, str], list[dict]]) -> dict[str, dict]:
    rows: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    policy_dirs: dict[str, list[Path]] = defaultdict(list)
    for (family, mode, _), runs in sorted(samples.items()):
        for key in (family, "All"):
            row = rows[key]
            if mode == "forward_dynamics":
                row["fd_psnr"].append(np.mean([r["psnr"] for r in runs]))
            elif mode == "inverse_dynamics":
                row["id_mse"].append(np.mean([r["action_mse"] for r in runs]))
            elif mode == "wam":
                mse = [r["action_mse"] for r in runs]
                psnr = [r["psnr"] for r in runs]
                row["policy_runs"].append(len(runs))
                row["policy_mse_mean"].append(np.mean(mse))
                row["policy_mse_best"].append(np.min(mse))
                row["policy_psnr_mean"].append(np.mean(psnr))
                row["policy_psnr_best"].append(np.max(psnr))
                if key != "All":
                    policy_dirs[family].append(runs[0]["dir"])
    out: dict[str, dict] = {}
    for key, row in rows.items():
        out[key] = {name: float(np.mean(values)) for name, values in row.items() if name != "policy_runs"}
        out[key]["samples"] = max((len(v) for n, v in row.items() if n != "policy_runs"), default=0)
        out[key]["policy_runs"] = int(min(row["policy_runs"], default=0))
    families = [k for k in out if k != "All"]
    for family in families:
        out[family]["baseline_mse"] = baseline_mse(policy_dirs[family])
    if "All" in out:
        weights = [len(policy_dirs[f]) for f in families]
        values = [out[f]["baseline_mse"] for f in families]
        out["All"]["baseline_mse"] = float(np.average(values, weights=weights)) if sum(weights) else float("nan")
    return dict(sorted(out.items(), key=lambda kv: (kv[0] == "All", kv[0])))


def write_table(summary: dict[str, dict], path: Path) -> None:
    def fmt(row: dict, key: str, digits: int) -> str:
        value = row.get(key, float("nan"))
        return "" if np.isnan(value) else f"{value:.{digits}f}"

    runs = max((row["policy_runs"] for row in summary.values()), default=0)
    lines = [
        f"Policy: mean over {runs} seeds / best of {runs} per sample. Action MSE in normalized action space "
        "(lower is better); baseline: the family's mean recorded trajectory.",
        "",
        "| Family | Samples | FD PSNR ↑ | ID action MSE ↓ | Policy action MSE ↓ (mean / best) "
        "| Policy PSNR ↑ (mean / best) | Baseline action MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in summary.items():
        name = f"**{family}**" if family == "All" else family
        lines.append(
            f"| {name} | {row['samples']} | {fmt(row, 'fd_psnr', 2)} | {fmt(row, 'id_mse', 4)} "
            f"| {fmt(row, 'policy_mse_mean', 4)} / {fmt(row, 'policy_mse_best', 4)} "
            f"| {fmt(row, 'policy_psnr_mean', 2)} / {fmt(row, 'policy_psnr_best', 2)} | {fmt(row, 'baseline_mse', 4)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="eval.py output directories (one per seed)")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(load_runs(args.runs))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_table(summary, args.output / "eval_summary.md")
    print((args.output / "eval_summary.md").read_text())


if __name__ == "__main__":
    main()
