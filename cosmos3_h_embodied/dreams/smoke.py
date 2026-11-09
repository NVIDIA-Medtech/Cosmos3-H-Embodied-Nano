# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Scheduler-neutral, resumable six-stage smoke driver; all paths are caller-owned."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from cosmos3_h_embodied.dreams.contract import file_sha256, require_contract, write_json
from cosmos3_h_embodied.dreams.prepare import assets

STAGES = ("c24_a", "c24_b", "c72", "tf", "sf4", "sf2")


def checkpoint(workspace: Path, run_name: str, stage: str, iteration: int) -> Path:
    return workspace / "runs/cosmos3_h_dreams" / run_name / stage / "checkpoints" / f"iter_{iteration:09d}"


def initialize(workspace: Path, gpus: int, run_name: str, profile: str = "smoke") -> Path:
    """Write an explicit small-update spec, independent of the historical private data."""
    if gpus < 1:
        raise ValueError("At least one GPU is required")
    workspace = workspace.resolve()
    spec_path = workspace / "run.json"
    if spec_path.exists():
        raise FileExistsError(f"Keep the existing spec to resume: {spec_path}")
    pins = assets()
    steps = dict(c24_a=2, c24_b=2, c72=2, tf=2, sf4=12, sf2=12)
    parents = {
        stage: str(checkpoint(workspace, run_name, STAGES[i - 1], steps[STAGES[i - 1]]))
        for i, stage in enumerate(STAGES)
        if i
    }
    parents["c24_a"] = str(workspace / "migrated_h16_28d")
    spec = {
        "schema": "dreams-smoke-v1",
        "purpose": "inference_only" if profile == "inference" else "integration_only_not_convergence_or_reproduction",
        "run_name": run_name,
        "world_size": gpus,
        "steps": steps,
        "parents": parents,
        "teacher": str(checkpoint(workspace, run_name, "c72", steps["c72"])),
        "manifests": {
            s: str(workspace / "prepared" / ("c24.json" if s.startswith("c24") else "c72.json")) for s in STAGES
        },
        "data_root": str(workspace / "data"),
        "resume": {},
        "vae": str(workspace / "assets/vae" / pins["vae"]["file"]),
        "text_tokenizer": str(workspace / "assets/text_tokenizer" / pins["text_tokenizer"]["prefix"]),
        "save_every": steps.copy(),
        "qualification": {"gpu": "not_run", "resume": "not_run"},
    }
    write_json(spec_path, spec)
    return spec_path


def execute(command: list[str], log: Path, *, env: dict[str, str], cwd: Path) -> None:
    """Persist exact argv and output, failing at the first unsuccessful operation."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    with log.open("a") as output:
        output.write(json.dumps({"argv": command, "cwd": str(cwd)}) + "\n")
        output.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def run(workspace: Path, *, dryrun: bool = False, stages: list[str] | None = None) -> None:
    """Run the pipeline in one allocation, with explicit C24 and SF save/resume probes.

    On failure, re-run the same command: successfully stamped stage checkpoints
    are retained. A partial same-stage checkpoint restores model, optimizer,
    scheduler and delivered-sample counter, not just weights.
    """
    workspace = workspace.resolve()
    original_path = workspace / "run.json"
    spec = json.loads(original_path.read_text())
    if spec["purpose"] == "inference_only" and not dryrun:
        raise ValueError("Inference workspace cannot launch training; use a new training workspace")
    research = spec["purpose"] == "research"
    if research and not dryrun:
        if not stages or len(stages) != 1:
            raise ValueError("Research runs require one explicit --stages value; validate and select between stages")
        from cosmos3_h_embodied.dreams.research import validate_handoff

        validate_handoff(spec, stages[0])
    framework = workspace / "sources/cosmos-framework"
    python = framework / ".venv/bin/python"
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(workspace / "cache/huggingface"),
        TORCH_HOME=str(workspace / "cache/torch"),
        IMAGINAIRE_OUTPUT_ROOT=str(workspace / "runs"),
        COSMOS_DREAMS_SPEC=str(workspace / "active_run.json"),
    )
    env.pop("COSMOS_SMOKE", None)  # Do not silently replace the released model with a two-layer toy.
    report_path = workspace / "reports/pipeline.json"
    report: dict[str, Any] = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {"stages": {}, "kind": "research" if research else "smoke"}
    )
    base = [
        "-m",
        "cosmos_framework.scripts.train_experiment",
        "--config",
        "cosmos_framework/configs/simulation/config.py",
        "--experiment-module",
        "cosmos3_h_embodied.dreams.experiments",
    ]
    for stage in stages or list(STAGES):
        if stage not in STAGES:
            raise ValueError(stage)
        end = spec["steps"][stage]
        final = checkpoint(workspace, spec["run_name"], stage, end)
        if not dryrun and (final / "action_contract.json").is_file():
            receipt = require_contract(final)["provenance"]
            expected = {
                "stage": stage,
                "world_size": spec["world_size"],
                "parent": spec["parents"][stage],
                "manifest_sha256": file_sha256(Path(spec["manifests"][stage])),
            }
            if stage.startswith("sf"):
                expected["scoring_teacher_and_fresh_critic"] = spec["teacher"]
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Completed checkpoint belongs to a different run specification: {final}")
            report["stages"].setdefault(stage, {})["checkpoint"] = str(final)
            continue
        active = json.loads(json.dumps(spec))
        active["resume"] = {}
        if not dryrun:
            require_contract(spec["parents"][stage])
            if stage.startswith("sf"):
                require_contract(spec["teacher"])
        # Prove full-state resume in both the ordinary and phase-aware trainer.
        boundaries = [end] if research else [1, end] if stage == "c24_a" else [6, end] if stage == "sf4" else [end]
        latest_file = final.parent / "latest_checkpoint.txt"
        if not dryrun and latest_file.exists():
            latest = final.parent / Path(latest_file.read_text().strip()).name
            require_contract(latest)
            active["resume"][stage] = str(latest)
            iteration = int(latest.name.removeprefix("iter_"))
            boundaries = [n for n in boundaries if n > iteration]
        if dryrun:
            boundaries = [end]
        for stop in boundaries:
            active["steps"][stage] = stop
            if not research:
                active["save_every"][stage] = stop
            write_json(workspace / "active_run.json", active)
            log = workspace / "reports" / f"{stage}_{stop}_{'compose' if dryrun else 'train'}.log"
            command = (
                [str(python), *base, "--dryrun", "--", f"experiment=chole_{stage}"]
                if dryrun
                else [
                    str(python),
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={spec['world_size']}",
                    *base,
                    "--",
                    f"experiment=chole_{stage}",
                ]
            )
            start_time = time.time()
            try:
                execute(command, log, env=env, cwd=framework)
                saved = checkpoint(workspace, spec["run_name"], stage, stop)
                if not dryrun:
                    require_contract(saved)
                    active["resume"][stage] = str(saved)
                report["stages"].setdefault(stage, {}).setdefault("allocations", []).append(
                    {
                        "end_iteration": stop,
                        "elapsed_seconds": time.time() - start_time,
                        "log": str(log),
                        "dryrun": dryrun,
                        "success": True,
                        "checkpoint": str(saved) if not dryrun else None,
                    }
                )
                write_json(report_path, report)
            except Exception as error:
                report["status"] = "failed"
                report["failure"] = {"stage": stage, "log": str(log), "error": str(error)}
                write_json(report_path, report)
                raise
        if not dryrun:
            report["stages"][stage]["checkpoint"] = str(final)
            report["stages"][stage]["manifest_sha256"] = file_sha256(Path(spec["manifests"][stage]))
    report.pop("failure", None)
    report["status"] = "composition_passed" if dryrun else "requested_training_stages_completed"
    report["not_established"] = ["convergence", "released-student quality parity", "surgical reliability"]
    write_json(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "compose", "run"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--run-name", default="public_smoke")
    parser.add_argument("--profile", choices=["smoke", "inference"], default="smoke")
    parser.add_argument("--stages", nargs="+", choices=STAGES)
    args = parser.parse_args()
    if args.command == "init":
        print(initialize(args.workspace, args.gpus, args.run_name, args.profile))
    else:
        run(args.workspace, dryrun=args.command == "compose", stages=args.stages)


if __name__ == "__main__":
    main()
