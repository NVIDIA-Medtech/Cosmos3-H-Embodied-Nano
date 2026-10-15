# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint the action dataloader's per-worker position so training resumes on the same samples."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback

RESUME_INDEX_ENV = "ACTION_DATALOADER_STATE_WORKER_{worker_id}_INDEX"
RESUME_RNG_ENV = "ACTION_DATALOADER_STATE_WORKER_{worker_id}_RNG_B64"


@dataclass
class ActionWorkerState:
    """Last action sample delivered to training by one DataLoader worker."""

    index: int = 0
    rng_state: bytes | None = None


def _flatten_ints(value: Any) -> list[int]:
    """Flatten packed-batch worker metadata into ordinary integers."""
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [item for sub in value for item in _flatten_ints(sub)]
    return [int(value)]


class ActionDataLoaderStateCallback(Callback):
    """Per-rank dataloader state for :class:`~cosmos3_h_embodied.data.unified_dataset.ActionUnifiedIterableDataset`.

    Each sample carries ``_action_worker_id``, ``_action_sample_index`` (samples produced by
    that worker so far) and ``_action_rng_state`` (the worker's global RNG state after
    producing it). The callback keeps the newest entry per worker and saves it in the
    checkpoint's per-rank ``dataloader`` component. On load it exports the state as
    environment variables that the dataset workers read in ``worker_init_fn``: each worker
    skips the samples it already delivered and restores its RNG.

    Workers must be spawned after the checkpoint is loaded, so the joint loader needs
    ``lazy_initialize_child_iterators=True``.
    """

    checkpoint_component: str = "dataloader"

    def __init__(self) -> None:
        super().__init__()
        self.config: Any = None
        self.state: dict[int, ActionWorkerState] = {}

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, Any],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        worker_ids = _flatten_ints(data_batch["_action_worker_id"])
        indices = _flatten_ints(data_batch["_action_sample_index"])
        rng_states = data_batch.get("_action_rng_state")
        if rng_states is None:
            blobs: list[bytes | None] = [None] * len(indices)
        elif isinstance(rng_states, (bytes, bytearray)):
            blobs = [bytes(rng_states)]
        else:
            blobs = [bytes(value) for value in rng_states]
        if not len(worker_ids) == len(indices) == len(blobs):
            raise RuntimeError(
                "Action dataloader checkpoint metadata is misaligned: "
                f"workers={worker_ids}, indices={indices}, rng_states={len(blobs)}"
            )
        for worker_id, index, rng_state in zip(worker_ids, indices, blobs):
            previous = self.state.get(worker_id)
            if index > (previous.index if previous is not None else -1):
                self.state[worker_id] = ActionWorkerState(index=index, rng_state=rng_state)

    def has_checkpoint_state(self) -> bool:
        return True

    def state_dict(self) -> dict[int, dict[str, int | bytes]]:
        state_dict: dict[int, dict[str, int | bytes]] = {}
        for worker_id, worker_state in self.state.items():
            state_dict[worker_id] = {"index": int(worker_state.index)}
            if worker_state.rng_state is not None:
                state_dict[worker_id]["rng_state"] = worker_state.rng_state
        log.info(f"Saved action dataloader state for {len(state_dict)} worker(s)", rank0_only=False)
        return state_dict

    def load_state_dict(self, state_dict: dict[int, dict[str, int | bytes]]) -> None:
        if not state_dict:
            log.info("No action dataloader state found in checkpoint")
            return
        self.state = {}
        for worker_id, worker_state in state_dict.items():
            worker_id = int(worker_id)
            index = int(worker_state["index"])
            rng_state = worker_state.get("rng_state")
            rng_state = bytes(rng_state) if rng_state is not None else None
            self.state[worker_id] = ActionWorkerState(index=index, rng_state=rng_state)
            os.environ[RESUME_INDEX_ENV.format(worker_id=worker_id)] = str(index)
            rng_key = RESUME_RNG_ENV.format(worker_id=worker_id)
            if rng_state is not None:
                os.environ[rng_key] = base64.b64encode(rng_state).decode("ascii")
            else:
                os.environ.pop(rng_key, None)
            log.info(
                f"Loaded action dataloader state for worker {worker_id}: index={index} "
                f"rng_bytes={len(rng_state or b'')}",
                rank0_only=False,
            )
