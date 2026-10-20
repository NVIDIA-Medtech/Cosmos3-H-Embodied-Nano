# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Install the authored ``meta/modality.json`` files shipped in ``tools/data_prep/modality/``.

Some training roots use a ``modality.json`` that differs from (or is missing in) the
Open-H upstream data. Each shipped file lives at ``modality/<root path in the Open-H
repository>/modality.json`` and is copied to ``<data root>/<root path>/meta/modality.json``
of the converted v3.0 dataset, replacing any existing file. Roots that are not present
under ``--data-root`` are reported and skipped.

Usage::

    python tools/data_prep/install_modality.py --data-root <v30 data dir> [ROOT_PATH ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MODALITY_DIR = Path(__file__).resolve().parent / "modality"


def shipped_roots(modality_dir: Path = MODALITY_DIR) -> list[str]:
    return sorted(str(p.parent.relative_to(modality_dir)) for p in modality_dir.rglob("modality.json"))


def install(data_root: Path, roots: list[str] | None = None, modality_dir: Path = MODALITY_DIR) -> list[str]:
    installed = []
    for rel in roots or shipped_roots(modality_dir):
        src = modality_dir / rel / "modality.json"
        if not src.is_file():
            raise FileNotFoundError(f"no shipped modality.json for {rel}")
        json.loads(src.read_text())  # must be valid JSON
        meta = data_root / rel / "meta"
        if not (meta / "info.json").is_file():
            print(f"skip (not a dataset under {data_root}): {rel}")
            continue
        shutil.copyfile(src, meta / "modality.json")
        installed.append(rel)
        print(f"installed {rel}/meta/modality.json")
    return installed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="directory holding the converted roots")
    ap.add_argument("roots", nargs="*", help="root paths to install (default: all shipped)")
    args = ap.parse_args()
    install(args.data_root.resolve(), args.roots or None)


if __name__ == "__main__":
    main()
