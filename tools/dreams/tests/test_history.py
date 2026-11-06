# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exercise the actual patched packer, causal visibility and null-action cache handling."""

import pytest
import torch
from cosmos3_h_embodied.dreams.history_attention import history_visibility_indices
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
from cosmos_framework.data.generator.sequence_packing.temporal_causal import pack_supertokens_temporal_causal
from cosmos_framework.model.generator.utils.kv_cache import zero_null_action_values


@pytest.mark.parametrize("history", [0, 1, 4, 7, 16])
@pytest.mark.parametrize("future", [24, 72])
def test_prefix_is_real_tokens_not_invented_frames(history: int, future: int) -> None:
    builder = PackedSequenceBuilder()
    actions = torch.arange((history + future) * 64).reshape(history + future, 64).float()  # [H+C,64]
    frames = future // 4 + 1
    length, null = pack_supertokens_temporal_causal(
        builder,
        torch.zeros(1, 48, frames, 4, 4),
        actions,
        [0],
        0.5,
        2,
        4,
        64,
        vision_fps=10.0,
        action_fps=10.0,
        enable_fps_modulation=True,
        history_action_prefix_length=history,
    )
    assert length == history + frames * (4 + 4)
    assert null is True and builder.history_action_prefix_length == history
    packed = builder.action.tokens[0]  # [H+C+4,64]
    torch.testing.assert_close(packed[:history], actions[:history])
    assert torch.count_nonzero(packed[history : history + 4]) == 0
    torch.testing.assert_close(packed[history + 4 :], actions[history:])
    assert builder.vision.token_shapes == [(frames, 2, 2)]


@pytest.mark.parametrize("replay", [False, True])
def test_frame_queries_cannot_see_future_actions_or_video(replay: bool) -> None:
    h, frames, spatial, text = 16, 4, 8, 3
    qo, ko, gather = history_visibility_indices(h, frames, spatial, text, replay)
    assert qo == [0, 16, 24, 32, 40, 48]
    assert gather[: ko[1]] == list(range(text + h))
    clean_base = text + h + frames * spatial if replay else text
    for frame in range(frames):
        keys = gather[ko[frame + 1] : ko[frame + 2]]
        expected = (
            list(range(text))
            + list(range(clean_base, clean_base + h + frame * spatial))
            + list(range(text + h + frame * spatial, text + h + (frame + 1) * spatial))
        )
        assert keys == expected


def test_null_action_values_preserve_the_history_prefix() -> None:
    values = torch.ones(1, 16 + 3 * 8, 2, 4)  # [B,H+T*S,heads,D]
    out = zero_null_action_values(values, [(3, 2, 2)], 4, True, history_action_prefix_length=16)
    assert torch.count_nonzero(out[:, :16]) == out[:, :16].numel()
    assert torch.count_nonzero(out[:, 16:20]) == 0
    assert torch.count_nonzero(out[:, 20:]) == out[:, 20:].numel()
    assert torch.all(values == 1)  # cached caller tensor is not mutated
