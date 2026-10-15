# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Episode exclusions: the shipped list and how its keys are matched to dataset roots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos3_h_embodied.configs.openh_datasets import FAMILIES
from cosmos3_h_embodied.data.cosmos3_action_lerobot import match_overlay_root

SHIPPED = Path(__file__).resolve().parents[1] / "cosmos3_h_embodied/data/exclusions/openh_h16_exclusions.json"


def test_shipped_list_names_recipe_roots() -> None:
    roots = json.loads(SHIPPED.read_text())["roots"]
    recipe_roots = {r.path for f in FAMILIES for r in f.roots}
    assert set(roots) <= recipe_roots
    assert sum(len(v["exclude"]) for v in roots.values()) == 302


def test_relative_keys_match_trailing_path_components(tmp_path) -> None:
    root = tmp_path / "data" / "jhu" / "imerse" / "nephfat" / "nephfat"
    root.mkdir(parents=True)
    assert match_overlay_root({"jhu/imerse/nephfat/nephfat": {}}, root) == "jhu/imerse/nephfat/nephfat"
    assert match_overlay_root({str(root.resolve()): {}}, root) == str(root.resolve())
    assert match_overlay_root({"imerse/srt_tissue_lift": {}}, root) is None
    assert match_overlay_root({"fat/nephfat": {}}, root) is None  # whole components only
    with pytest.raises(ValueError):
        match_overlay_root({"nephfat": {}, "nephfat/nephfat": {}}, root)


def test_keys_match_through_symlinked_data_directories(tmp_path) -> None:
    target = tmp_path / "storage" / "cmr_clinical_v30" / "inguinal_hernia"
    target.mkdir(parents=True)
    link = tmp_path / "data" / "cmr_surgical" / "inguinal_hernia"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    assert match_overlay_root({"cmr_surgical/inguinal_hernia": {}}, link) == "cmr_surgical/inguinal_hernia"
    assert match_overlay_root({"cmr_clinical_v30/inguinal_hernia": {}}, link) == "cmr_clinical_v30/inguinal_hernia"
