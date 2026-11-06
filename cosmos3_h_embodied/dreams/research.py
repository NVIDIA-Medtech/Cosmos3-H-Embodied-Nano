# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Create a full-budget research specification and record reviewed checkpoint handoffs."""

import argparse
import json
from pathlib import Path

from cosmos3_h_embodied.dreams.contract import contract_id, file_sha256, reference, require_contract, write_json
from cosmos3_h_embodied.dreams.smoke import STAGES, checkpoint, initialize


def initialize_research(workspace: Path, gpus: int, run_name: str, selection_a: Path, selection_b: Path) -> Path:
    if run_name == "public_smoke" or not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("Use a distinct simple research run name")
    manifests = {
        s: str(
            (
                (selection_a if s == "c24_a" else selection_b) / ("c24.json" if s.startswith("c24") else "c72.json")
            ).resolve()
        )
        for s in STAGES
    }
    recordings = {}
    for path in set(manifests.values()):
        manifest = json.loads(Path(path).read_text())
        if (
            manifest["contract_sha256"] != contract_id()
            or manifest["review_status"] != "visually_reviewed_two_instruments"
        ):
            raise ValueError(f"Research requires reviewed manifests with the current action contract: {path}")
        for episode in manifest["episodes"]:
            group, split = episode["recording_id"], episode["split"]
            if recordings.setdefault(group, split) != split:
                raise ValueError(f"Recording crosses splits between A/B selections: {group}")
    path = initialize(workspace, gpus, run_name)
    spec = json.loads(path.read_text())
    spec.update(
        purpose="research",
        manifests=manifests,
        steps={s["name"]: s["max_iter"] for s in reference()["stages"]},
        save_every=dict(c24_a=200, c24_b=200, c72=400, tf=100, sf4=60, sf2=60),
        selected={},
        validation="Offline fixed-window evaluation on the validation split before each handoff",
    )
    for i, stage in enumerate(STAGES[1:], 1):
        parent = STAGES[i - 1]
        spec["parents"][stage] = str(checkpoint(workspace.resolve(), run_name, parent, spec["steps"][parent]))
    spec["teacher"] = spec["parents"]["tf"]
    write_json(path, spec)
    return path


def validate_handoff(spec: dict, stage: str) -> None:
    """Require recorded selections, not implicit last-checkpoint choices."""
    required = [] if stage == "c24_a" else [STAGES[STAGES.index(stage) - 1]]
    if stage.startswith("sf"):
        required.append("c72")
    for parent in set(required):
        chosen = spec.get("selected", {}).get(parent)
        if chosen is None:
            raise ValueError(f"Evaluate and select {parent} before launching {stage}")
        path = Path(chosen["checkpoint"])
        receipt = require_contract(path)
        if (
            file_sha256(path / "action_contract.json") != chosen["receipt_sha256"]
            or receipt["provenance"]["stage"] != parent
        ):
            raise ValueError(f"Selected checkpoint receipt changed: {parent}")
        if file_sha256(path / "model/.metadata") != chosen["checkpoint_metadata_sha256"]:
            raise ValueError(f"Selected checkpoint metadata changed: {parent}")
        evaluation = Path(chosen["evaluation"])
        if file_sha256(evaluation) != chosen["evaluation_sha256"]:
            raise ValueError(f"Selected validation evidence changed: {parent}")
        expected = spec["teacher"] if parent == "c72" and stage.startswith("sf") else spec["parents"][stage]
        if Path(expected).resolve() != path.resolve():
            raise ValueError(f"Handoff path disagrees with selected {parent}")


def select(workspace: Path, stage: str, source: Path, evaluation: Path, reason: str) -> None:
    spec_path = workspace / "run.json"
    spec = json.loads(spec_path.read_text())
    if spec["purpose"] != "research":
        raise ValueError("Checkpoint selection is for research specs only")
    source, evaluation = source.resolve(), evaluation.resolve()
    provenance = require_contract(source)["provenance"]
    if provenance.get("stage") != stage or provenance.get("manifest_sha256") != file_sha256(
        Path(spec["manifests"][stage])
    ):
        raise ValueError("Selected checkpoint is from another stage or selection")
    if provenance.get("world_size") != spec["world_size"] or provenance.get("parent") != spec["parents"][stage]:
        raise ValueError("Selected checkpoint is from another run or world size")
    result = json.loads(evaluation.read_text())
    # Evaluator writes an explicit run receipt, separate from its case list.
    run = json.loads((evaluation.parent / "evaluation.json").read_text())
    if (
        not result.get("cases")
        or run["status"] != "completed"
        or "recorded" not in run["modes"]
        or run["split"] != "val"
        or Path(run["checkpoint"]).resolve() != source / "model"
    ):
        raise ValueError("Select using validation results from this exact checkpoint")
    if run["checkpoint_metadata_sha256"] != file_sha256(source / "model/.metadata"):
        raise ValueError("Evaluated checkpoint metadata changed")
    downstream = list(STAGES[STAGES.index(stage) + 1 :])
    for dependent in downstream:
        folder = checkpoint(workspace.resolve(), spec["run_name"], dependent, 0).parent
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Cannot change ancestry after {dependent} checkpoints exist; start a new run")
    chosen = {
        "checkpoint": str(source),
        "receipt_sha256": file_sha256(source / "action_contract.json"),
        "checkpoint_metadata_sha256": file_sha256(source / "model/.metadata"),
        "evaluation": str(evaluation),
        "evaluation_sha256": file_sha256(evaluation),
        "reason": reason,
    }
    spec["selected"][stage] = chosen
    if stage != STAGES[-1]:
        spec["parents"][STAGES[STAGES.index(stage) + 1]] = str(source)
    if stage == "c72":
        spec["teacher"] = str(source)
    write_json(spec_path, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--workspace", type=Path, required=True)
    init.add_argument("--gpus", type=int, required=True)
    init.add_argument("--run-name", default="research_run")
    init.add_argument("--selection-a", type=Path, required=True, help="prepared manifest directory")
    init.add_argument("--selection-b", type=Path, required=True, help="prepared manifest directory")
    choose = sub.add_parser("select")
    choose.add_argument("--workspace", type=Path, required=True)
    choose.add_argument("--stage", choices=STAGES, required=True)
    choose.add_argument(
        "--checkpoint", type=Path, required=True, help="stamped checkpoint root, not its model subdirectory"
    )
    choose.add_argument("--evaluation", type=Path, required=True, help="results.json from validation evaluation")
    choose.add_argument("--reason", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--workspace", type=Path, required=True)
    validate.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    if args.command == "init":
        print(initialize_research(args.workspace, args.gpus, args.run_name, args.selection_a, args.selection_b))
    elif args.command == "select":
        select(args.workspace, args.stage, args.checkpoint, args.evaluation, args.reason)
    else:
        spec = json.loads((args.workspace / "run.json").read_text())
        if spec["purpose"] != "research":
            raise ValueError("Expected research run.json")
        validate_handoff(spec, args.stage)
        require_contract(spec["parents"][args.stage])
        print(f"Handoff validated: {args.stage}")


if __name__ == "__main__":
    main()
