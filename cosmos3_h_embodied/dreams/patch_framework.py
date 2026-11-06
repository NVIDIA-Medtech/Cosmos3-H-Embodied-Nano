# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Apply the narrow H16 extension only to the audited public Framework files."""

import argparse
import json
import subprocess
from pathlib import Path

from cosmos3_h_embodied.dreams.contract import file_sha256


def apply(root: Path) -> None:
    """Fail on mixed/modified files; accept an already applied patch idempotently."""
    directory = Path(__file__).parent
    manifest = json.loads((directory / "framework_h16_files.json").read_text())
    states = [
        "after"
        if file_sha256(root / name) == hashes["after"]
        else "before"
        if file_sha256(root / name) == hashes["before"]
        else "unknown"
        for name, hashes in manifest.items()
    ]
    if set(states) == {"after"}:
        print("H16 patch already applied and verified")
        return
    if set(states) != {"before"}:
        raise RuntimeError("Framework sources differ from the pinned originals; use a clean pinned checkout")
    patch = directory / "framework_h16.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    for name, hashes in manifest.items():
        if file_sha256(root / name) != hashes["after"]:
            raise RuntimeError(f"Patched-file checksum mismatch: {name}")
    print("Applied and verified H16 causal extension")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", type=Path)
    apply(parser.parse_args().framework)


if __name__ == "__main__":
    main()
