# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-worker dataloader resume state (``ActionDataLoaderStateCallback``)."""

from __future__ import annotations

import base64
import os

import pytest
import torch

from cosmos3_h_embodied.callbacks.action_dataloader_state import (
    RESUME_INDEX_ENV,
    RESUME_RNG_ENV,
    ActionDataLoaderStateCallback,
)


def _packed_batch(step: int) -> dict:
    # As produced by the packing loader: per-sample tensors of shape (1,) and RNG blobs.
    workers = [(step + k) % 4 for k in range(3)]
    return {
        "_action_worker_id": [torch.tensor([w]) for w in workers],
        "_action_sample_index": [torch.tensor([10 * step + w]) for w in workers],
        "_action_rng_state": [bytes([step, w]) * 8 for w in workers],
    }


def test_keeps_the_newest_sample_per_worker_and_restores_it(monkeypatch) -> None:
    # load_state_dict exports the state as environment variables; register them so they are restored afterwards.
    for w in range(4):
        monkeypatch.setenv(RESUME_INDEX_ENV.format(worker_id=w), "")
        monkeypatch.setenv(RESUME_RNG_ENV.format(worker_id=w), "")
    callback = ActionDataLoaderStateCallback()
    for step in range(5):
        callback.on_training_step_batch_end(None, _packed_batch(step), {}, torch.tensor(0.0))
    saved = callback.state_dict()
    # Step 4 delivered workers 0, 1, 2 (indices 40, 41, 42); worker 3 last appeared at step 3.
    assert {w: s["index"] for w, s in saved.items()} == {0: 40, 1: 41, 2: 42, 3: 33}
    assert saved[3]["rng_state"] == bytes([3, 3]) * 8

    restored = ActionDataLoaderStateCallback()
    restored.load_state_dict(saved)
    for w, s in saved.items():
        assert os.environ[RESUME_INDEX_ENV.format(worker_id=w)] == str(s["index"])
        assert base64.b64decode(os.environ[RESUME_RNG_ENV.format(worker_id=w)]) == s["rng_state"]


def test_misaligned_metadata_is_rejected() -> None:
    batch = _packed_batch(0)
    batch["_action_rng_state"] = batch["_action_rng_state"][:2]
    with pytest.raises(RuntimeError):
        ActionDataLoaderStateCallback().on_training_step_batch_end(None, batch, {}, torch.tensor(0.0))
