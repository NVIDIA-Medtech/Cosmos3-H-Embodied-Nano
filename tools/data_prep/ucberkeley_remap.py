# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UC Berkeley debridement (step UB): use the Cartesian poses as state and action, in place.

The upstream ``action`` and ``observation.state`` are 14-D joint positions. Training uses
the 16-D quat-7 bimanual poses (psm1 pos3 + quat xyzw + jaw, psm2 the same) stored in
``action.cartesian_state`` and ``observation.cartesian_state``. On the converted v3.0 root:

  * ``action <- action.cartesian_state`` and ``observation.state <- observation.cartesian_state``
    in the data files, ``meta/info.json`` (float32 [16], Cartesian names), ``meta/stats.json``
    and the per-episode ``stats/*`` columns; the ``*.cartesian_state`` columns are removed;
  * drops ``next.done`` and the leftover v2.1 image-path columns ``observation.images.{left,right}``
    from the data files (the videos stay as video features; the string columns make LeRobot's
    cast fail);
  * adds the global frame ``index`` column (absent upstream; LeRobot needs it).

Then install the shipped quat-7 ``modality.json`` (``install_modality.py``). Idempotent.

    python tools/data_prep/ucberkeley_remap.py --root <v30 data dir>/ucberkeley/debridement_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from drop_columns import _write_inplace, drop_data_columns, drop_metadata

REMAP = {"action": "action.cartesian_state", "observation.state": "observation.cartesian_state"}
DROP = ["next.done"]
IMAGE_PATH_COLUMNS = ["observation.images.left", "observation.images.right"]


def _remap_table(table: pa.Table, start: int) -> tuple[pa.Table, int]:
    for dst, src in REMAP.items():
        if src in table.column_names:
            i = table.column_names.index(dst)
            table = table.set_column(i, dst, table.column(src)).drop_columns([src])
    if "index" not in table.column_names:
        index = pa.array(np.arange(start, start + table.num_rows, dtype=np.int64))
        table = table.append_column("index", index)
    return table, start + table.num_rows


def remap(root: Path) -> None:
    files = sorted((root / "data").rglob("*.parquet"))
    start = 0
    for path in files:
        table, start = _remap_table(pq.read_table(path), start)
        _write_inplace(table, path)
    drop_data_columns(root, DROP + IMAGE_PATH_COLUMNS)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info["features"]
    if any(src in features for src in REMAP.values()):
        fps = features["action"].get("fps")
        for dst, src in REMAP.items():
            features[dst] = dict(features.pop(src), dtype="float32", shape=[16])
        features["index"] = {"dtype": "int64", "shape": [1], "names": None, **({"fps": fps} if fps else {})}
        info_path.write_text(json.dumps(info, indent=4) + "\n")

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        if any(src in stats for src in REMAP.values()):
            for dst, src in REMAP.items():
                stats[dst] = stats.pop(src)
            stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        names = table.column_names
        if not any(n.startswith(f"stats/{src}/") for src in REMAP.values() for n in names):
            continue
        for dst, src in REMAP.items():
            table = table.drop_columns([n for n in names if n.startswith(f"stats/{dst}/")])
            table = table.rename_columns(
                [f"stats/{dst}/" + n[len(f"stats/{src}/") :] if n.startswith(f"stats/{src}/") else n for n in table.column_names]
            )
            names = table.column_names
        _write_inplace(table, path)

    drop_metadata(root, DROP)
    total = int(info.get("total_frames", start))
    if start != total:
        raise RuntimeError(f"{root}: {start} rows in data files, info.json total_frames={total}")
    print(f"{root}: remapped to Cartesian state/action, {start} frames indexed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="converted ucberkeley/debridement_lerobot root")
    remap(ap.parse_args().root)


if __name__ == "__main__":
    main()
