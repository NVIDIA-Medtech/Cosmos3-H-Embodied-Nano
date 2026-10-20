# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Download a few episodes of the Open-H Embodiment dataset, for smoke tests.

Fetches a few episodes from every public root of the training mixture's families
(``cosmos3_h_embodied/configs/openh_datasets.py``) from the Hugging Face dataset
``nvidia/PhysicalAI-Robotics-Open-H-Embodiment`` (LeRobot v2.1, one file per episode).

Each sampled root is written as a valid, smaller LeRobot v2.1 dataset with its episodes renumbered
0..K-1 (``tools/data_prep/renumber_sample_episodes.py``), so ``tools/data_prep/convert_v21_to_v30.py``
converts it directly. The output mirrors the Hugging Face tree below ``Surgical/``, plus
``sample_manifest.json``. Families whose roots need more than the conversion (see
``tools/data_prep/README.md``) need those steps too.

Examples::

    # 2 episodes from every root of two families
    python tools/download_openh_samples.py --output-dir samples_v21 --families dvrk_obuda dvrk_ucsd
    python tools/data_prep/convert_v21_to_v30.py --src-root samples_v21 --dst-root samples

    # just print the plan (no download)
    python tools/download_openh_samples.py --output-dir samples_v21 --dry-run

The dataset is public; ``HF_TOKEN`` is optional but avoids anonymous rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from cosmos3_h_embodied.configs.openh_datasets import FAMILIES

sys.path.insert(0, str(Path(__file__).resolve().parent / "data_prep"))
from renumber_sample_episodes import renumber  # noqa: E402

HF_REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
HF_PREFIX = "Surgical"

# Minimum episode duration (seconds) for a sampled episode. One training window
# needs 16 history steps plus a 24-step future chunk at 10 Hz (~4 s); 8 s leaves
# room for several strided windows.
DEFAULT_MIN_EPISODE_SECONDS = 8.0

META_FILES = ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl", "stats.json", "modality.json")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Downloader:
    def __init__(self, output_dir: Path, revision: str | None, dry_run: bool, source_root: Path | None = None):
        if source_root is None:
            from huggingface_hub import constants, hf_hub_download

            self._hf_hub_download = hf_hub_download
            # Untouched upstream files, reused across runs; outside the output tree, which holds only sample roots.
            self.raw_dir = Path(constants.HF_HUB_CACHE) / "openh_samples_raw"
        self.output_dir = output_dir
        self.revision = revision
        self.dry_run = dry_run
        self.source_root = source_root  # local v2.1 tree (same layout as HF below Surgical/)

    def _resolve_local(self, repo_path: str) -> Path:
        """Resolve ``repo_path`` under ``source_root``, matching path components case-insensitively.

        Local copies of the dataset do not always preserve the Hugging Face folder casing
        (e.g. ``USTC`` instead of ``ustc``); an exact match always takes precedence.
        """
        path = self.source_root
        for part in Path(repo_path).parts:
            candidate = path / part
            if not candidate.exists() and path.is_dir():
                matches = [c for c in path.iterdir() if c.name.lower() == part.lower()]
                if len(matches) == 1:
                    candidate = matches[0]
            path = candidate
        return path

    def _fetch_local(self, repo_path: str, optional: bool) -> Path | None:
        src = self._resolve_local(repo_path)
        if not src.is_file():
            if optional:
                return None
            raise FileNotFoundError(src)
        return src

    def _fetch_uncompressed(self, repo_path: str) -> Path:
        """Plain HTTP download without content encoding.

        Some large metadata files intermittently fail to decode when the hub serves
        them brotli-compressed; requesting the identity encoding avoids that.
        """
        from huggingface_hub import get_token, hf_hub_url

        url = hf_hub_url(HF_REPO_ID, f"{HF_PREFIX}/{repo_path}", repo_type="dataset", revision=self.revision)
        headers = {"Accept-Encoding": "identity"}
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        dst = self.raw_dir / HF_PREFIX / repo_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as response:
            with tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
        tmp.replace(dst)
        return dst

    def fetch(self, repo_path: str, optional: bool = False, attempts: int = 3) -> Path | None:
        if self.source_root is not None:
            return self._fetch_local(repo_path, optional)
        from huggingface_hub.errors import EntryNotFoundError

        for attempt in range(attempts):
            if attempt == attempts - 1:
                return self._fetch_uncompressed(repo_path)
            try:
                local = self._hf_hub_download(
                    HF_REPO_ID,
                    f"{HF_PREFIX}/{repo_path}",
                    repo_type="dataset",
                    revision=self.revision,
                    local_dir=self.raw_dir,
                )
                return Path(local)
            except EntryNotFoundError:
                if optional:
                    return None
                raise
            except Exception as exc:  # transient transport errors (e.g. truncated compressed responses)
                if attempt == attempts - 1:
                    raise
                delay = 2**attempt
                print(f"  retry {attempt + 1}/{attempts - 1} for {repo_path} in {delay}s: {exc}", file=sys.stderr)
                time.sleep(delay)
        return None

    def sample_root(
        self,
        family,
        root_path: str,
        episodes_per_root: int,
        min_seconds: float,
        video_keys: str,
        seed: int | None,
    ) -> dict:
        meta = {name: self.fetch(f"{root_path}/meta/{name}", optional=name != "info.json") for name in META_FILES}
        info = json.loads(meta["info.json"].read_text())
        if info.get("codebase_version") != "v2.1":
            raise RuntimeError(f"{root_path}: expected LeRobot v2.1, found {info.get('codebase_version')}")
        fps = float(info["fps"])
        episodes = _read_jsonl(meta["episodes.jsonl"])
        eligible = [e for e in episodes if e["length"] >= min_seconds * fps]
        if not eligible:
            # Some roots only hold short clips; take the longest ones instead.
            eligible = sorted(episodes, key=lambda e: -e["length"])[:episodes_per_root]
            print(
                f"  {root_path}: no episode reaches {min_seconds:g}s; using the longest "
                f"({max(e['length'] for e in eligible) / fps:.1f}s)",
                file=sys.stderr,
            )
        if seed is None:
            chosen = eligible[:episodes_per_root]
        else:
            chosen = sorted(random.Random(f"{seed}:{root_path}").sample(eligible, min(episodes_per_root, len(eligible))), key=lambda e: e["episode_index"])

        all_video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        if video_keys == "used":
            video_key = family.kwargs["video_key"]
            if video_key not in all_video_keys:
                raise RuntimeError(f"{root_path}: camera {video_key!r} not in {all_video_keys}")
            keys = [video_key]
        else:
            keys = all_video_keys

        chunks_size = int(info["chunks_size"])
        files = []
        for ep in chosen:
            index = ep["episode_index"]
            chunk = index // chunks_size
            files.append(info["data_path"].format(episode_chunk=chunk, episode_index=index))
            files.extend(
                info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=index) for key in keys
            )

        record = dict(
            family=family.name,
            hf_path=f"{HF_PREFIX}/{root_path}",
            source=str(self.source_root / root_path) if self.source_root else f"hf://datasets/{HF_REPO_ID}",
            fps=fps,
            video_keys=keys,
            episodes=[dict(episode_index=e["episode_index"], length=e["length"], tasks=e.get("tasks")) for e in chosen],
            upstream_total_episodes=info["total_episodes"],
        )
        if self.dry_run:
            return record

        # Subset with the upstream episode indices, then renumbered 0..K-1 into the output tree.
        subset = self.output_dir / "_subset" / root_path
        shutil.rmtree(subset, ignore_errors=True)
        for rel in files:
            _link_or_copy(self.fetch(f"{root_path}/{rel}"), subset / rel)
        self._write_subset_meta(meta, subset / "meta", info, chosen, keys, all_video_keys)
        record["bytes"] = sum((subset / rel).stat().st_size for rel in files)
        dst = self.output_dir / root_path
        shutil.rmtree(dst, ignore_errors=True)
        record["renumbered"] = {str(k): v for k, v in renumber(subset, dst).items()}
        shutil.rmtree(subset)
        return record

    @staticmethod
    def _write_subset_meta(meta, meta_dst: Path, info, chosen, keys, all_video_keys) -> None:
        meta_dst.mkdir(parents=True, exist_ok=True)
        indices = {e["episode_index"] for e in chosen}
        for name, src in meta.items():
            if src is not None:
                stem, ext = name.split(".", 1)
                _link_or_copy(src, meta_dst / f"{stem}.upstream.{ext}")

        _write_jsonl(meta_dst / "episodes.jsonl", chosen)
        if meta["episodes_stats.jsonl"] is not None:
            stats = [s for s in _read_jsonl(meta["episodes_stats.jsonl"]) if s["episode_index"] in indices]
            _write_jsonl(meta_dst / "episodes_stats.jsonl", stats)
        for name in ("tasks.jsonl", "stats.json", "modality.json"):
            if meta[name] is not None:
                _link_or_copy(meta[name], meta_dst / name)

        subset = dict(info)
        if keys != all_video_keys:
            subset["features"] = {
                k: v for k, v in info["features"].items() if v.get("dtype") != "video" or k in keys
            }
        subset["total_episodes"] = len(chosen)
        subset["total_frames"] = sum(e["length"] for e in chosen)
        subset["total_videos"] = len(chosen) * len(keys)
        subset["total_chunks"] = len({e["episode_index"] // int(info["chunks_size"]) for e in chosen})
        (meta_dst / "info.json").write_text(json.dumps(subset, indent=4) + "\n")


def main(argv: list[str] | None = None) -> int:
    known = [f.name for f in FAMILIES if any(r.public for r in f.roots)]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=known, default=known, metavar="FAMILY", help=f"subset of {known}")
    parser.add_argument("--episodes-per-root", type=int, default=2)
    parser.add_argument("--max-roots-per-family", type=int, default=None, help="default: every root")
    parser.add_argument("--min-episode-seconds", type=float, default=DEFAULT_MIN_EPISODE_SECONDS)
    parser.add_argument(
        "--video-keys",
        choices=("all", "used"),
        default="all",
        help="'all' keeps every camera so the sample can be converted to LeRobot v3.0; "
        "'used' fetches only the camera the training loader reads",
    )
    parser.add_argument("--seed", type=int, default=None, help="sample random eligible episodes instead of the first ones")
    parser.add_argument("--revision", default=None, help="Hugging Face dataset revision (branch, tag or commit)")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="read from a local LeRobot v2.1 tree laid out like the HF 'Surgical/' folder instead of Hugging Face",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve metadata and print the plan only")
    args = parser.parse_args(argv)

    downloader = Downloader(args.output_dir, args.revision, args.dry_run, args.source_root)
    report = dict(repo_id=HF_REPO_ID, revision=args.revision, args={k: str(v) for k, v in vars(args).items()}, families=[])
    failures = 0
    for family in FAMILIES:
        if family.name not in args.families:
            continue
        roots = [r.path for r in family.roots if r.public][: args.max_roots_per_family]
        entry = dict(name=family.name, video_key=family.kwargs["video_key"], roots=[])
        print(f"[{family.name}] {len(roots)} root(s)")
        for root_path in roots:
            try:
                record = downloader.sample_root(
                    family, root_path, args.episodes_per_root, args.min_episode_seconds, args.video_keys, args.seed
                )
            except Exception as exc:  # keep going; report every failing root at the end
                failures += 1
                print(f"  FAILED {root_path}: {exc}", file=sys.stderr)
                entry["roots"].append(dict(hf_path=f"{HF_PREFIX}/{root_path}", error=str(exc)))
                continue
            size = f", {record['bytes'] / 1e6:.1f} MB" if "bytes" in record else ""
            eps = [e["episode_index"] for e in record["episodes"]]
            print(f"  {root_path}: episodes {eps} @ {record['fps']:g} fps{size}")
            entry["roots"].append(record)
        report["families"].append(entry)

    if not args.dry_run:
        shutil.rmtree(args.output_dir / "_subset", ignore_errors=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "sample_manifest.json"
        if manifest_path.exists():
            # Keep families sampled by earlier runs.
            sampled = {f["name"] for f in report["families"]}
            previous = json.loads(manifest_path.read_text())
            report["previous_runs"] = previous.pop("previous_runs", []) + [
                {k: v for k, v in previous.items() if k != "families"}
            ]
            kept = [f for f in previous.get("families", []) if f["name"] not in sampled]
            order = {name: i for i, name in enumerate(known)}
            report["families"] = sorted(kept + report["families"], key=lambda f: order.get(f["name"], len(order)))
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output_dir / 'sample_manifest.json'}")
    if failures:
        print(f"{failures} root(s) failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
