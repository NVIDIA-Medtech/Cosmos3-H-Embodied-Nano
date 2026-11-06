# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Uniform accepted-start sampling with deterministic delivered-sample resume."""

import json
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn
from torch.utils.data import DataLoader, IterableDataset

from cosmos3_h_embodied.data.joint_dataloader import HistoryMatchedJointDataLoader
from cosmos3_h_embodied.data.transforms import ActionTransformPipeline
from cosmos3_h_embodied.dreams.contract import DOMAIN, contract_id, file_sha256
from cosmos3_h_embodied.dreams.data import decode, encode, normalize, read_episode, resolve, window_reason


def draw_index(seed: int, rank: int, draw: int, size: int) -> int:
    """Counter-derived sampling with replacement: every accepted start is equiprobable."""
    return int(np.random.default_rng(np.random.SeedSequence([seed, rank, draw])).integers(size))


class CholeDataset(IterableDataset):
    """One view, C24/C72, up to H16, 28 valid channels padded by the public transform.

    The smoke path intentionally uses zero workers. Each sample's
    draw counter is checkpointed only after delivery, so loader lookahead cannot
    skip examples on resume. Increasing workers needs a separate qualification.
    """

    def __init__(
        self,
        manifest: str,
        data_root: str,
        tokenizer_config: dict[str, Any],
        split: str = "train",
        seed: int = 28003,
        verify_hashes: bool = False,
    ) -> None:
        self.root = Path(data_root)
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest["contract_sha256"] != contract_id():
            raise ValueError("Prepared-data action contract mismatch")
        if verify_hashes:
            for path, expected in self.manifest["source_sha256"].items():
                if file_sha256(resolve(self.root, path)) != expected:
                    raise ValueError(f"Source file changed: {path}")
        self.future = int(self.manifest["future_frames"])
        if self.future not in (24, 72):
            raise ValueError("The reference loader supports C24 and C72")
        self.episodes = [ep for ep in self.manifest["episodes"] if ep["split"] == split]
        self.windows = [(i, start) for i, ep in enumerate(self.episodes) for start in ep["accepted_starts"]]
        if not self.windows:
            raise ValueError(f"Empty {split} split")
        self.seed = seed
        self.split = split
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=0.1 if split == "train" else 0.0,
            max_action_dim=64,
            action_channel_masking=True,
            append_viewpoint_info=False,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            append_idle_frames=False,
            format_prompt_as_json=False,
            video_augment=False,
        )

    def __len__(self) -> int:
        """Return the accepted-start population required by the public joint loader.

        Iteration still samples with replacement indefinitely; this is the nominal
        population size, not an epoch boundary or a limit on training updates.
        """
        return len(self.windows)

    def sample(self, index: int) -> dict[str, Any]:
        ep_idx, anchor = self.windows[index]
        ep = self.episodes[ep_idx]
        a, s, times = read_episode(str(self.root), tuple(ep["data_files"]), ep["episode_id"])  # [L,100],[L,100],[L]
        reason, history = window_reason(a, s, times, anchor, self.future, self.manifest["threshold"])
        if reason is not None:
            raise ValueError(f"Prepared start became invalid: {ep['episode_id']}:{anchor}: {reason}")
        starts = anchor + np.arange(self.future) * 6  # [C]
        sample = {
            "video": torch.from_numpy(decode(self.root, ep, anchor, self.future)),  # [3,C+1,480,832]
            "action": torch.from_numpy(normalize(encode(a, s, anchor, starts))),  # [C,28]
            "history_action": torch.from_numpy(normalize(encode(a, s, anchor, history))),  # [H,28]
            "domain_id": torch.tensor(DOMAIN),
            "conditioning_fps": 10.0,
            "fps": 10.0,
            "mode": "forward_dynamics",
            "ai_caption": "Continue the ongoing surgical task.",
            "__key__": f"episode_{ep['episode_id']:06d}_frame_{anchor:06d}",
            "_action_family": "cmr_chole_28d",
        }
        return self.transform(sample, "480")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("Use num_workers=0 for the deterministic smoke loader")
        rank = int(os.environ.get("RANK", "0"))
        draw = int(os.environ.get("ACTION_DATALOADER_STATE_WORKER_0_INDEX", "0")) if self.split == "train" else 0
        while True:
            index = (
                draw_index(self.seed, rank, draw, len(self.windows))
                if self.split == "train"
                else draw % len(self.windows)
            )
            # Isolate augmentation/caption RNG from model RNG and make dropout restartable.
            py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
            sample_seed = int(np.random.SeedSequence([self.seed, rank, draw, 1]).generate_state(1)[0])
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.random.default_generator.manual_seed(sample_seed)
            try:
                sample = self.sample(index)
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
            draw += 1
            sample.update(
                _action_worker_id=0, _action_sample_index=draw, action_sample_local_start_frame=self.windows[index][1]
            )
            yield sample


def make_loader(
    manifest: str, data_root: str, tokenizer_config: dict[str, Any], split: str = "train", seed: int = 28003
) -> HistoryMatchedJointDataLoader:
    """Build the actual packed training loader, not an approximation of its batches."""
    dataset = CholeDataset(manifest, data_root, tokenizer_config, split, seed)
    child = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=custom_collate_fn, pin_memory=True)
    return HistoryMatchedJointDataLoader(
        dataloaders={"action_data": {"dataloader": child, "ratio": 1}},
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,
        max_samples_per_batch=1,
        chunk_length=dataset.future,
        max_history_actions=16,
        prewarm=False,
        lazy_initialize_child_iterators=True,
        seed=seed,
    )
