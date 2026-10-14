# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Download a small, self-consistent sample of the Open-H Embodiment dataset.

Fetches a few episodes from every dataset root used by the Cosmos3-H-Embodied-Nano
training mixture (15 families) from the public Hugging Face dataset
``nvidia/PhysicalAI-Robotics-Open-H-Embodiment`` (LeRobot v2.1, one file per
episode). The result is meant for smoke tests of the data pipeline, not training.

Each sampled root is written as a valid, smaller LeRobot v2.1 dataset:
``meta/episodes.jsonl``, ``meta/episodes_stats.jsonl`` and ``meta/info.json`` are
rewritten to describe only the downloaded episodes, while the per-episode parquet
and mp4 files are the unmodified upstream files (original episode indices are
kept). The upstream metadata is preserved next to it as ``*.upstream.*``.

Output layout mirrors the Hugging Face tree below ``Surgical/``::

    <output-dir>/
      cmr_surgical/cholecystectomy/{meta,data,videos}/...
      hamlyn/suturing_1/...
      ...
      sample_manifest.json        # what was fetched, per family / root / episode

Examples::

    # 2 episodes from every root of all 15 families (default)
    python tools/download_openh_samples.py --output-dir data/openh_samples

    # quick: first root of each family, 1 episode, only the camera the loader uses
    python tools/download_openh_samples.py --output-dir data/openh_samples \\
        --max-roots-per-family 1 --episodes-per-root 1 --video-keys used

    # a subset of families
    python tools/download_openh_samples.py --output-dir data/openh_samples \\
        --families dvrk_hamlyn dvrk_jhu

    # just print the plan (no download)
    python tools/download_openh_samples.py --output-dir data/openh_samples --dry-run

    # sample from a local copy of the v2.1 tree instead of Hugging Face, e.g. the
    # USTC roots, which are not in the public dataset
    python tools/download_openh_samples.py --output-dir data/openh_samples \\
        --families dvrk_ustc --source-root /path/to/open-h-embodiment/Surgical

Requires ``huggingface_hub`` (not needed with ``--source-root``). The dataset is public; ``HF_TOKEN`` is optional but
avoids anonymous rate limits.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HF_REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
HF_PREFIX = "Surgical"

# Minimum episode duration (seconds) for a sampled episode. One training window
# needs 16 history steps plus a 24-step future chunk at 10 Hz (~4 s); 8 s leaves
# room for several strided windows.
DEFAULT_MIN_EPISODE_SECONDS = 8.0


@dataclasses.dataclass(frozen=True)
class Root:
    hf_path: str  # path below HF_PREFIX (or below --source-root)
    note: str = ""  # set when the mapping to the training root is not 1:1
    on_hf: bool = True  # False: only available from a local copy (--source-root)


@dataclasses.dataclass(frozen=True)
class Family:
    name: str
    video_key: str  # camera stream the training loader reads
    roots: tuple[Root, ...]
    note: str = ""


# Dataset families of the Open-H training mixture and the Hugging Face roots that
# back them. Order follows the training config.
FAMILIES: tuple[Family, ...] = (
    Family(
        "cmr_clinical",
        "observation.images.endoscope",
        (
            Root("cmr_surgical/cholecystectomy"),
            Root("cmr_surgical/hysterectomy"),
            Root("cmr_surgical/inguinal_hernia"),
            Root("cmr_surgical/prostatectomy"),
        ),
    ),
    Family(
        "dvrk_obuda",
        "observation.images.endoscope.left",
        tuple(
            Root(f"obuda/{name}")
            for name in (
                "pork_1",
                "frs_dome_1",
                "needlethreading_1",
                "needlethreading_2",
                "pegtransfer_1",
                "pegtransfer_2",
                "rollercoaster_1",
                "seaspike_1",
                "seaspike_2",
                "seaspike_3",
                "skinphantom_1",
            )
        ),
    ),
    Family(
        "dvrk_ucsd",
        "observation.images.left",
        (Root("ucsd/surgical_learning_dataset2"), Root("ucsd/surgical_learning_dataset")),
    ),
    Family(
        "dvrk_jhu",
        "observation.images.endoscope.left",
        (
            Root("jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled"),
            Root("jhu/imerse/nephfat/nephfat"),
            Root("jhu/imerse/srt_tissue_lift"),
            Root("jhu/imerse/cao_cautery_combined"),
            Root("jhu/imerse/srth_porcine_chole"),
            Root(
                "jhu/imerse/suturebot",
                note="UNVERIFIED: training used three 'suturebot_canon' roots "
                "(suturebot_tissue_2, suturing_jesse, jesse_pickup) that are not published "
                "under these names; jhu/imerse/suturebot is the closest public source.",
            ),
            Root("jhu/lcsr/arcade/cholecystectomy"),
        ),
    ),
    Family(
        "dvrk_suturebot_mono",
        "observation.images.endoscope.left",
        (
            Root(
                "jhu/imerse/suturebot",
                note="UNVERIFIED: training read a separate SutureBot copy whose camera key is "
                "'observation.images.endoscope_left'; the public root names it "
                "'observation.images.endoscope.left'.",
            ),
        ),
    ),
    Family(
        "dvrk_stanford",
        "observation.images.camera_left",
        tuple(
            Root(f"stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/{name}")
            for name in ("peg_transfer", "needle_transfer", "tissue_retraction")
        ),
    ),
    Family("dvrk_rob_surgical", "observation.images.endoscope", (Root("rob_surgical/all_merged_data"),)),
    Family(
        "dvrk_ustc",
        "observation.images.endoscope.left",
        tuple(
            Root(f"ustc/{name}", on_hf=False)
            for name in (
                "needle_pickup",
                "Needle_handover",
                "ustc_knot_tying",
                "grasp_on_liver",
                "exvivo_liver_sep",
                "invivo_liver_sep",
                "tissue_lifting",
            )
        ),
        note="NOT ON HUGGING FACE: the USTC roots are not part of the public dataset; "
        "sample them from a local copy with --source-root.",
    ),
    Family(
        "dvrk_hamlyn",
        "observation.images.color",
        tuple(
            Root(f"hamlyn/{name}")
            for name in (
                "knot_tying",
                "needle_grasp_and_handover",
                "peg_transfer",
                "suturing_1",
                "suturing_2",
                "tissue_retraction",
            )
        ),
    ),
    Family("dvrk_ucberkeley", "observation.images.left", (Root("ucberkeley/debridement_lerobot"),)),
    Family(
        "turin_mitic",
        "observation.images.endoscope.left",
        tuple(
            Root(f"turin/{name}")
            for name in (
                "mitic_lerobot_plastic_tube",
                "mitic_lerobot_ex_vivo",
                "mitic_lerobot_plastic_pad",
                "mitic_lerobot_plastic_pad_3dmed",
            )
        ),
    ),
    Family(
        "cmr_versius_drybox_peg_transfer",
        "observation.images.endoscope",
        (
            Root(
                "cmr_surgical/peg_transfer",
                note="UNVERIFIED: training root was 'drybox_peg_transfer'; both public candidates "
                "(peg_transfer, dry_box) are sampled.",
            ),
            Root(
                "cmr_surgical/dry_box",
                note="UNVERIFIED: see cmr_surgical/peg_transfer.",
            ),
        ),
    ),
    Family("dvrk_star_il", "observation.images.endoscope.left", (Root("jhu/imerse/star_il/star_il"),)),
    Family(
        "dvrk_tud_grasping_retraction",
        "observation.images.laparoscope_left",
        (Root("tud/260131_tundra_dataset/grasping_retraction"),),
    ),
    Family(
        "dvrk_smarts",
        "observation.images.endoscope_left",
        tuple(Root(f"jhu/lcsr/smarts/SurgSync-stitch-coldcut/{p}") for p in ("P1", "P2", "P3")),
    ),
)

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
            from huggingface_hub import hf_hub_download

            self._hf_hub_download = hf_hub_download
        self.output_dir = output_dir
        self.raw_dir = output_dir / "_hf_raw"  # untouched upstream files; reused across runs
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
        family: Family,
        root: Root,
        episodes_per_root: int,
        min_seconds: float,
        video_keys: str,
        seed: int | None,
    ) -> dict:
        meta = {name: self.fetch(f"{root.hf_path}/meta/{name}", optional=name != "info.json") for name in META_FILES}
        info = json.loads(meta["info.json"].read_text())
        if info.get("codebase_version") != "v2.1":
            raise RuntimeError(f"{root.hf_path}: expected LeRobot v2.1, found {info.get('codebase_version')}")
        fps = float(info["fps"])
        episodes = _read_jsonl(meta["episodes.jsonl"])
        eligible = [e for e in episodes if e["length"] >= min_seconds * fps]
        if not eligible:
            # Some roots only hold short clips; take the longest ones instead.
            eligible = sorted(episodes, key=lambda e: -e["length"])[:episodes_per_root]
            print(
                f"  {root.hf_path}: no episode reaches {min_seconds:g}s; using the longest "
                f"({max(e['length'] for e in eligible) / fps:.1f}s)",
                file=sys.stderr,
            )
        if seed is None:
            chosen = eligible[:episodes_per_root]
        else:
            chosen = sorted(random.Random(f"{seed}:{root.hf_path}").sample(eligible, min(episodes_per_root, len(eligible))), key=lambda e: e["episode_index"])

        all_video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        if video_keys == "used":
            if family.video_key not in all_video_keys:
                raise RuntimeError(f"{root.hf_path}: camera {family.video_key!r} not in {all_video_keys}")
            keys = [family.video_key]
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
            hf_path=f"{HF_PREFIX}/{root.hf_path}",
            source=str(self.source_root / root.hf_path) if self.source_root else f"hf://datasets/{HF_REPO_ID}",
            note=root.note,
            fps=fps,
            video_keys=keys,
            episodes=[dict(episode_index=e["episode_index"], length=e["length"], tasks=e.get("tasks")) for e in chosen],
            upstream_total_episodes=info["total_episodes"],
        )
        if self.dry_run:
            return record

        dst = self.output_dir / root.hf_path
        for rel in files:
            _link_or_copy(self.fetch(f"{root.hf_path}/{rel}"), dst / rel)
        self._write_subset_meta(meta, dst / "meta", info, chosen, keys, all_video_keys)
        record["bytes"] = sum((dst / rel).stat().st_size for rel in files)
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
        for name in ("tasks.jsonl", "stats.json"):
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
        # Episode indices keep their upstream values; ``splits`` is left as upstream
        # (informational in v2.1) and ``episodes.jsonl`` is the authoritative list.
        (meta_dst / "info.json").write_text(json.dumps(subset, indent=4) + "\n")


def main(argv: list[str] | None = None) -> int:
    known = [f.name for f in FAMILIES]
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
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        metavar="PATH",
        help="with a single --families entry: sample these roots (relative to 'Surgical/') instead of the "
        "built-in list, e.g. when a local copy uses different task folders",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve metadata and print the plan only")
    args = parser.parse_args(argv)
    if args.roots and len(args.families) != 1:
        parser.error("--roots requires exactly one --families entry")

    downloader = Downloader(args.output_dir, args.revision, args.dry_run, args.source_root)
    report = dict(repo_id=HF_REPO_ID, revision=args.revision, args={k: str(v) for k, v in vars(args).items()}, families=[])
    failures = 0
    for family in FAMILIES:
        if family.name not in args.families:
            continue
        if args.roots:
            on_hf = all(r.on_hf for r in family.roots)
            family = dataclasses.replace(family, roots=tuple(Root(r.strip("/"), on_hf=on_hf) for r in args.roots))
        roots = family.roots[: args.max_roots_per_family] if args.max_roots_per_family else family.roots
        entry = dict(name=family.name, video_key=family.video_key, note=family.note, roots=[])
        print(f"[{family.name}] {len(roots)} root(s)" + (f"  NOTE: {family.note}" if family.note else ""))
        for root in roots:
            if not root.on_hf and args.source_root is None:
                print(f"  SKIPPED {root.hf_path}: not on Hugging Face (use --source-root)")
                entry["roots"].append(dict(hf_path=f"{HF_PREFIX}/{root.hf_path}", skipped="not on Hugging Face"))
                continue
            try:
                record = downloader.sample_root(
                    family, root, args.episodes_per_root, args.min_episode_seconds, args.video_keys, args.seed
                )
            except Exception as exc:  # keep going; report every failing root at the end
                failures += 1
                print(f"  FAILED {root.hf_path}: {exc}", file=sys.stderr)
                entry["roots"].append(dict(hf_path=f"{HF_PREFIX}/{root.hf_path}", error=str(exc)))
                continue
            size = f", {record['bytes'] / 1e6:.1f} MB" if "bytes" in record else ""
            eps = [e["episode_index"] for e in record["episodes"]]
            print(f"  {root.hf_path}: episodes {eps} @ {record['fps']:g} fps{size}" + (f"  NOTE: {root.note}" if root.note else ""))
            entry["roots"].append(record)
        report["families"].append(entry)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "sample_manifest.json"
        if manifest_path.exists():
            # Keep families sampled by earlier runs (e.g. HF run + local-source run).
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
