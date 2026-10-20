# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Policy-server request handling without a model: batch construction and action (de)normalization."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from cosmos3_h_embodied.data.action_normalization import NORM_CLIP, load_action_stats, normalize_action
from cosmos3_h_embodied.data.domain_utils import get_domain_id
from cosmos3_h_embodied.scripts.policy_server import (
    NORMALIZERS_DIR,
    QuantileDenormalizer,
    build_policy_batch,
    make_policy_transform,
)

NORMALIZER = "dvrk_hamlyn_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json"
PROMPT = "Hamlyn dVRK peg transfer. Transfer the peg"


def _png_b64(h: int, w: int) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, size=(h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build(**req):
    req = {"image": _png_b64(480, 640), "prompt": PROMPT, "domain_name": "hamlyn", **req}
    return build_policy_batch(req, transform=make_policy_transform(64), raw_action_dim=20)


@pytest.mark.parametrize("history", [0, 7, 16])
def test_history_rows_are_prepended_as_clean_conditioning(history) -> None:
    req = {"history_action": torch.randn(history, 20).tolist()} if history else {}
    batch, meta = _build(image_size=256, **req)
    assert meta["num_history_actions"] == history
    action = batch["action"][0][0]
    assert action.shape == (24 + history, 64)
    if history:
        torch.testing.assert_close(action[:history, :20], torch.tensor(req["history_action"]))
    assert batch["sequence_plan"][0].condition_frame_indexes_action == list(range(history))
    assert batch["sequence_plan"][0].condition_frame_indexes_vision == [0]
    assert int(batch["raw_action_dim"][0]) == 20
    assert int(batch["domain_id"][0]) == get_domain_id("hamlyn")
    assert batch["ai_caption"][0].startswith(PROMPT)
    assert batch["video"][0][0].shape[1] == 25


@pytest.mark.parametrize(
    "req",
    [
        {"history_action": torch.zeros(17, 20).tolist()},
        {"history_action": torch.zeros(4, 19).tolist()},
        {"history_action": [[float("nan")] * 20]},
        {"prompt": None},
        {"image_size": 0},
    ],
)
def test_invalid_requests_are_rejected(req) -> None:
    with pytest.raises(ValueError):
        _build(**req)


def test_denormalizer_inverts_training_normalization() -> None:
    den = QuantileDenormalizer(NORMALIZER)
    stats = {k: torch.as_tensor(v) for k, v in load_action_stats(str(NORMALIZERS_DIR / NORMALIZER)).items()}
    raw = stats["q01"] + torch.rand(24, den.dim) * (stats["q99"] - stats["q01"])
    normalized = normalize_action(raw, "quantile", stats)
    torch.testing.assert_close(den.normalize(raw), normalized)
    torch.testing.assert_close(den.denormalize(normalized), raw, rtol=1e-5, atol=1e-5)
    # Out-of-range values are clipped as in training.
    assert den.normalize(stats["q99"] + 10 * (stats["q99"] - stats["q01"])).max() == NORM_CLIP
