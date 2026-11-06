# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare a portable camera8000 selection, including a public-only smoke fixture.

The smoke fixture is telemetry-filtered, NOT a visually reviewed research set.
Production preparation consumes explicit reviewed intervals and recording splits.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from cosmos3_h_embodied.dreams.contract import contract_id, file_sha256, write_json
from cosmos3_h_embodied.dreams.data import read_episode, resolve, window_reason


def assets() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "assets.json").read_text())


def download_assets(workspace: Path, students: bool = False, model_set: str = "training") -> None:
    """Pin every downloaded model file and record its actual content hash."""
    pins = assets()
    receipt = workspace / "assets/receipt.json"
    records: dict[str, Any] = json.loads(receipt.read_text()) if receipt.exists() else {}
    names = ["vae", "text_tokenizer"]
    names += ["initializer"] if model_set == "training" else ["students"]
    if students and "students" not in names:
        names.append("students")
    for name in names:
        spec = dict(pins[name])
        if name == "students" and model_set in ("sf4", "sf2"):
            spec["files"] = [f for f in spec["files"] if f"_{model_set[-1]}-step_" in f]
        directory = workspace / "assets" / name
        if "prefix" in spec:
            snapshot_download(
                repo_id=spec["repository"],
                revision=spec["revision"],
                local_dir=directory,
                allow_patterns=[spec["prefix"] + "*"],
                max_workers=1,
            )
            paths = sorted(p for p in (directory / spec["prefix"]).iterdir() if p.is_file())
        else:
            filenames = spec.get("files", [spec.get("file")])
            paths = [
                Path(
                    hf_hub_download(
                        repo_id=spec["repository"], revision=spec["revision"], filename=f, local_dir=directory
                    )
                )
                for f in filenames
            ]
        previous_files = records.get(name, {}).get("files", [])
        downloaded = [
            {"path": str(p.relative_to(workspace)), "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in paths
        ]
        by_path = {entry["path"]: entry for entry in previous_files + downloaded}
        records[name] = {
            "repo": spec["repository"],
            "revision": spec["revision"],
            "files": list(by_path.values()),
        }
    write_json(workspace / "assets/receipt.json", records)


def download_dataset(workspace: Path, *, size_only: bool = False) -> None:
    """Report the pinned subset's actual size before optionally downloading it."""
    spec = assets()["dataset"]
    entries = HfApi().list_repo_tree(
        spec["repository"],
        repo_type="dataset",
        revision=spec["revision"],
        path_in_repo=spec["prefix"],
        recursive=True,
    )
    files = [{"path": e.path, "bytes": e.size} for e in entries if hasattr(e, "size")]
    total = sum(e["bytes"] for e in files)
    inventory = {**spec, "files": files, "bytes": total, "gib": total / 2**30}
    write_json(workspace / "reports/dataset_inventory.json", inventory)
    print(f"Pinned subset: {len(files)} files, {total:,} bytes ({total / 2**30:.1f} GiB)", flush=True)
    if not size_only:
        snapshot_download(
            repo_id=spec["repository"],
            repo_type="dataset",
            revision=spec["revision"],
            allow_patterns=[spec["prefix"] + "/*"],
            local_dir=workspace / "data",
            max_workers=1,
        )


def prepare_to_directory(root: Path, selection: Path, output: Path, *, allow_unreviewed_smoke: bool = False) -> None:
    """Publish both manifests together; never overwrite a previous selection."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared output already exists; choose a new --output-dir: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "prepared"
        prepare(root, selection, staging, allow_unreviewed_smoke=allow_unreviewed_smoke)
        if output.exists():
            output.rmdir()  # Empty only; never remove an existing manifest.
        staging.rename(output)


def accepted(
    root: Path, episode: dict[str, Any], future: int, threshold: float, max_starts: int | None = None
) -> tuple[list[int], Counter]:
    """Enumerate accepted 10 Hz anchors, checking every intervening native row."""
    a, s, t = read_episode(str(root), tuple(episode["data_files"]), episode["episode_id"])  # [L,100],[L,100],[L]
    intervals = episode["reviewed_intervals"]
    starts: list[int] = []
    reasons: Counter = Counter()
    for anchor in range(0, len(a) - future * 6, 6):
        reason, history = window_reason(a, s, t, anchor, future, threshold)
        first = int(history[0]) if len(history) else anchor
        if reason is None and not any(lo <= first and anchor + future * 6 < hi for lo, hi in intervals):
            reason = "outside_selected_interval"
        if reason:
            reasons[reason] += 1
            continue
        starts.append(anchor)
        if max_starts is not None and len(starts) >= max_starts:
            break
    return starts, reasons


def union_minutes(starts: list[int], future: int) -> float:
    """Count unique future-time coverage, never overlapping-window duration sums."""
    if not starts:
        return 0.0
    total, begin, end = 0, starts[0], starts[0] + future * 6
    for start in starts[1:]:
        stop = start + future * 6
        if start > end:
            total += end - begin
            begin = start
        end = max(end, stop)
    return (total + end - begin) / 3600


def prepare(
    root: Path, selection: Path, output: Path, *, threshold: float = 8000, allow_unreviewed_smoke: bool = False
) -> None:
    """Create C24/C72 manifests from a portable reviewed selection descriptor."""
    chosen = json.loads(selection.read_text())
    if chosen["review_status"] != "visually_reviewed_two_instruments" and not allow_unreviewed_smoke:
        raise ValueError("Unreviewed data requires explicit --allow-unreviewed-smoke; not a production dataset")
    recordings: dict[str, str] = {}
    for ep in chosen["episodes"]:
        split = ep["split"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if recordings.setdefault(ep["recording_id"], split) != split:
            raise ValueError("A recording crosses train/val/test splits")
        if ep.get("instrument_names") != ["Cadiere Grasper", "Monopolar Hook"]:
            raise ValueError("Selection must declare the reference instrument pair")
    source_hashes = {
        rel: file_sha256(resolve(root, rel)) for ep in chosen["episodes"] for rel in [*ep["data_files"], ep["video"]]
    }
    for future in (24, 72):
        episodes = []
        summary: dict[str, Any] = {split: {"samples": 0, "unique_minutes": 0.0} for split in ("train", "val", "test")}
        reasons: Counter = Counter()
        for ep in chosen["episodes"]:
            starts, rejected = accepted(root, ep, future, threshold)
            reasons.update(rejected)
            if starts:
                episodes.append({**ep, "accepted_starts": starts})
                summary[ep["split"]]["samples"] += len(starts)
                summary[ep["split"]]["unique_minutes"] += union_minutes(starts, future)
        if not all(summary[split]["samples"] for split in summary):
            raise ValueError(f"C{future} needs nonempty train/val/test: {summary}")
        write_json(
            output / f"c{future}.json",
            {
                "schema": "cmr28-selection-v1",
                "contract_sha256": contract_id(),
                "future_frames": future,
                "threshold": threshold,
                "review_status": chosen["review_status"],
                "source": chosen["source"],
                "selection_sha256": file_sha256(selection),
                "source_sha256": source_hashes,
                "episodes": episodes,
                "summary": summary,
                "rejections": dict(reasons),
                "ancestry_overlap_status": chosen.get("ancestry_overlap_status", "not_audited"),
            },
        )


def public_smoke(workspace: Path) -> Path:
    """Find three eligible public episodes from distinct recordings, deterministically.

    Only selected videos are downloaded. At most 80 episode telemetry files are
    inspected. Recording splits protect this fixture's own training, not Open-H
    initialization ancestry: this is an integration test, not a held-out claim.
    """
    spec = assets()["dataset"]
    root = workspace / "data"

    def get(relative: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=spec["repository"],
                repo_type="dataset",
                revision=spec["revision"],
                filename=f"{spec['prefix']}/{relative}",
                local_dir=root,
            )
        )

    info = json.loads(get("meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info["fps"] != 60:
        raise ValueError("Pinned smoke fixture expects native 60 Hz LeRobot v2.1")
    rows = [json.loads(line) for line in get("meta/episodes.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(28003)
    candidates = [4148, 35, *rng.permutation(len(rows)).tolist()]
    selected: list[dict[str, Any]] = []
    groups: set[str] = set()
    for index in candidates[:80]:
        ep = rows[index]
        group = ep["tasks"][0].rsplit("_", 1)[0]
        if group in groups:
            continue
        values = {
            "episode_index": index,
            "episode_chunk": index // info["chunks_size"],
            "video_key": "observation.images.endoscope",
        }
        data = get(info["data_path"].format(**values))
        video = f"{spec['prefix']}/{info['video_path'].format(**values)}"
        descriptor = {
            "episode_id": index,
            "recording_id": group,
            "data_files": [str(data.relative_to(root))],
            "video": video,
            "video_offset_seconds": 0.0,
            "reviewed_intervals": [[0, ep["length"]]],
            "instrument_names": ["Cadiere Grasper", "Monopolar Hook"],
            "split": ("train", "val", "test")[len(selected)],
        }
        starts, _ = accepted(root, descriptor, 72, 8000, max_starts=4)
        if len(starts) < 4:
            continue
        get(info["video_path"].format(**values))
        selected.append(descriptor)
        groups.add(group)
        print(f"Selected smoke {descriptor['split']}: episode {index}, recording {group}", flush=True)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError("No complete fixture in bounded search; supply an explicit reviewed selection")
    result = workspace / "selection.json"
    write_json(
        result,
        {
            "review_status": "telemetry_only_unreviewed_smoke",
            "source": spec,
            "ancestry_overlap_status": "not_audited_smoke_only",
            "episodes": selected,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--public-smoke", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--allow-unreviewed-smoke", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--include-students", action="store_true")
    parser.add_argument("--model-set", choices=["training", "sf4", "sf2", "students"], default="training")
    parser.add_argument("--output-dir", type=Path, help="new directory for C24/C72 manifests")
    parser.add_argument(
        "--dataset-size", action="store_true", help="report pinned full-subset size without downloading"
    )
    parser.add_argument(
        "--download-dataset", action="store_true", help="download the complete pinned cholecystectomy subset"
    )
    args = parser.parse_args()
    if args.public_smoke and not args.allow_unreviewed_smoke:
        parser.error("--public-smoke requires --allow-unreviewed-smoke")
    if args.download_models:
        download_assets(args.workspace, args.include_students, args.model_set)
    if args.dataset_size or args.download_dataset:
        download_dataset(args.workspace, size_only=not args.download_dataset)
    if (
        (args.download_models or args.dataset_size or args.download_dataset)
        and not args.public_smoke
        and args.selection is None
    ):
        return
    selection = public_smoke(args.workspace) if args.public_smoke else args.selection
    if selection is None:
        parser.error("Choose --public-smoke or --selection")
    prepare_to_directory(
        args.workspace / "data",
        selection,
        args.output_dir or args.workspace / "prepared",
        allow_unreviewed_smoke=args.allow_unreviewed_smoke,
    )


if __name__ == "__main__":
    main()
