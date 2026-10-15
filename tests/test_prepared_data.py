# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Samples of prepared dataset families have the shapes and ranges the recipe expects.

Checks data prepared with ``tools/data_prep`` or your own family (see the README)::

    OPENH_DATA_ROOT=<data root> OPENH_TEST_FAMILIES=dvrk_hamlyn,my_lab pytest tests/test_prepared_data.py

Reads a few training samples per family (CPU video decoding).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cosmos3_h_embodied.configs.openh_datasets import COMMON_KWARGS, FAMILIES
from cosmos3_h_embodied.data.action_normalization import NORM_CLIP

TEST_FAMILIES = [f for f in os.environ.get("OPENH_TEST_FAMILIES", "").split(",") if f]
SAMPLES = 4


@pytest.mark.skipif(
    not TEST_FAMILIES or "OPENH_DATA_ROOT" not in os.environ, reason="set OPENH_DATA_ROOT and OPENH_TEST_FAMILIES"
)
@pytest.mark.parametrize("family_name", TEST_FAMILIES)
def test_family_samples(family_name: str) -> None:
    family = next((f for f in FAMILIES if f.name == family_name), None)
    assert family is not None, f"{family_name} is not in FAMILIES (configs/openh_datasets.py)"
    roots = [r for r in family.roots if r.public]
    data_root = os.environ["OPENH_DATA_ROOT"].rstrip("/")
    kwargs = {**COMMON_KWARGS, **family.kwargs, "roots": [f"{data_root}/{r.path}" for r in roots], "mode": "policy"}
    if all(r.caption for r in roots):
        kwargs["caption_embodiments"] = [r.caption for r in roots]
    dataset = family.cls(**OmegaConf.to_container(OmegaConf.create(kwargs), resolve=True))
    dataset._register_sources()
    assert len(dataset) > 0, "no training windows"
    for index in np.random.default_rng(0).integers(0, len(dataset), size=SAMPLES):
        sample = dataset[int(index)]
        video, action, history = sample["video"], sample["action"], sample.get("history_action")
        assert video.dtype == torch.uint8 and video.shape[0] == 3 and video.shape[1] == 25, tuple(video.shape)
        assert action.shape[0] == 24 and torch.isfinite(action).all()
        assert action.abs().max() <= NORM_CLIP
        assert history is None or (history.shape[0] <= 16 and history.shape[1] == action.shape[1])
        assert isinstance(sample["ai_caption"], str) and sample["ai_caption"].strip()
