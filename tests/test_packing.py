# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Token accounting of the history-matched packing loader."""

from __future__ import annotations

import pytest
import torch

from cosmos3_h_embodied.data.joint_dataloader import HistoryMatchedJointDataLoader


def _loader() -> HistoryMatchedJointDataLoader:
    loader = object.__new__(HistoryMatchedJointDataLoader)
    loader.__dict__.update(
        tokenizer_spatial_compression_factor=16, tokenizer_temporal_compression_factor=4, patch_spatial=2,
        chunk_length=24, max_history_actions=16,
    )  # fmt: skip
    return loader


def _batch(h: int, w: int, rows: int, text: int = 100) -> dict:
    return {
        "text_token_ids": [torch.zeros(text, dtype=torch.long)],
        "video": [torch.zeros(1, 1, 1, 1, dtype=torch.uint8).expand(3, 25, h, w)],
        "action": [torch.zeros(rows, 64)],
    }


@pytest.mark.parametrize("hw", [(256, 320), (320, 256), (272, 336)])
@pytest.mark.parametrize("history", [0, 7, 16])
def test_every_sample_is_charged_the_full_history_budget(hw, history) -> None:
    h, w = hw
    und, gen = _loader()._compute_token_split_per_sample(_batch(h, w, 24 + history))
    assert und == 100 + 1
    # Patch grid rounded down (as in training), 7 latent frames for 25 video frames, 2 extra tokens,
    # then 24 + 16 action rows whatever the history length.
    assert gen == (h // 32) * (w // 32) * 7 + 2 + 40


@pytest.mark.parametrize("rows", [23, 41])
def test_unexpected_action_rows_are_rejected(rows) -> None:
    with pytest.raises(ValueError):
        _loader()._compute_token_split_per_sample(_batch(256, 320, rows))
