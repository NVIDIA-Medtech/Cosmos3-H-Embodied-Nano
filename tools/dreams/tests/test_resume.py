# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Delivered-sample restart and text-dropout RNG without a model or media decode."""

import random
from types import MethodType
from typing import Any

import numpy as np
import pytest
import torch
from cosmos3_h_embodied.dreams import loader as loader_module
from cosmos3_h_embodied.dreams.loader import CholeDataset
from cosmos3_h_embodied.dreams.preflight import single_packed_item


def synthetic_sample(self: CholeDataset, index: int) -> dict[str, Any]:
    return {"index": index, "dropout_draws": [random.random(), float(np.random.random()), float(torch.rand(()))]}


def dataset() -> CholeDataset:
    # Exercise the actual iterator, replacing only expensive media/tokenizer I/O.
    result = object.__new__(CholeDataset)
    result.seed = 28003
    result.split = "train"
    result.windows = [(0, i * 6) for i in range(100)]
    result.sample = MethodType(synthetic_sample, result)
    return result


@pytest.mark.parametrize("rank", [0, 7])
def test_delivered_counter_and_dropout_restart_exactly(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("ACTION_DATALOADER_STATE_WORKER_0_INDEX", "0")
    iterator = iter(dataset())
    delivered = [next(iterator) for _ in range(5)]
    # Simulate loader lookahead: only the third sample was delivered at the save.
    monkeypatch.setenv("ACTION_DATALOADER_STATE_WORKER_0_INDEX", str(delivered[2]["_action_sample_index"]))
    resumed = iter(dataset())
    assert [next(resumed), next(resumed)] == delivered[3:]


def test_loader_does_not_advance_model_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTION_DATALOADER_STATE_WORKER_0_INDEX", "0")
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    next(iter(dataset()))
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_public_joint_loader_accepts_iterable_population(monkeypatch: pytest.MonkeyPatch) -> None:
    source = dataset()
    source.future = 24
    assert len(source) == 100
    monkeypatch.setattr(loader_module, "CholeDataset", lambda *args, **kwargs: source)
    packed = loader_module.make_loader("unused.json", ".", {})
    assert packed is not None  # Constructor validates child length without decoding/prewarming.


def packed_sample(self: CholeDataset, index: int) -> dict[str, Any]:
    # Replace media/tokenizer I/O only; exercise the real collation and packing.
    return {
        "video": torch.zeros(3, self.future + 1, 32, 32, dtype=torch.uint8),  # [3,C+1,H,W]
        "action": torch.ones(self.future + 16, 64),  # [H16+C,64]
        "text_token_ids": torch.tensor([1, 2, 3]),
        "raw_action_dim": 28,
    }


@pytest.mark.parametrize("future", [24, 72])
def test_preflight_reads_real_packed_batch(monkeypatch: pytest.MonkeyPatch, future: int) -> None:
    source = dataset()
    source.future = future
    source.sample = MethodType(packed_sample, source)
    monkeypatch.setattr(loader_module, "CholeDataset", lambda *args, **kwargs: source)
    batch = next(iter(loader_module.make_loader("unused.json", ".", {})))
    action = single_packed_item(batch, "action")
    video = single_packed_item(batch, "video")
    assert tuple(action.shape) == (16 + future, 64)
    assert tuple(video.shape) == (3, future + 1, 32, 32)
    assert torch.all(action == 1)
    assert int(batch["raw_action_dim"][0]) == 28
