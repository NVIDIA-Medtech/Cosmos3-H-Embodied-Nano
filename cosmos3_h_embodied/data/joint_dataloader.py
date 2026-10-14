# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sequence-packing loader for history-conditioned action samples."""

from __future__ import annotations

from typing import Any

from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader


class HistoryMatchedJointDataLoader(IterativeJointDataLoader):
    """``IterativeJointDataLoader`` that charges every action sample the full history budget.

    A sample carries ``chunk_length`` future action rows plus 0..``max_history_actions``
    clean history rows (fewer near an episode start). Each sample is charged as if it had
    the full ``chunk_length + max_history_actions`` rows, so the number of samples per
    packed batch does not depend on how much history a sample happens to have, and H0 and
    H16 runs see the same samples. Reserved slots are not fed to the model.

    The token count reproduces the H16 training run exactly: text tokens + 1, then per
    video ``(H // (s * p)) * (W // (s * p)) * (1 + (T - 1) // t) + 2``, then the action
    rows. (The base class rounds the patch grid up instead of down, which differs when
    ``H // s`` or ``W // s`` is odd.)
    """

    def __init__(self, *args: Any, chunk_length: int = 24, max_history_actions: int = 16, **kwargs: Any) -> None:
        if chunk_length <= 0 or max_history_actions < 0:
            raise ValueError(f"invalid chunk_length={chunk_length} / max_history_actions={max_history_actions}")
        # Set before super().__init__(), which may prewarm and count tokens.
        self.chunk_length = int(chunk_length)
        self.max_history_actions = int(max_history_actions)
        super().__init__(*args, **kwargs)

    def _compute_token_split_per_sample(self, data_batch: dict) -> tuple[int, int]:
        text_token_ids = data_batch["text_token_ids"]
        if isinstance(text_token_ids, list):
            num_text_tokens = text_token_ids[0].shape[0]
        else:
            num_text_tokens = text_token_ids.shape[1]
        und_tokens = num_text_tokens + 1

        gen_tokens = 0
        is_image_batch = "images" in data_batch
        media_list = data_batch["images" if is_image_batch else "video"]
        spatial_downsample = self.tokenizer_spatial_compression_factor * self.patch_spatial
        for media in media_list if isinstance(media_list, list) else [media_list]:
            if is_image_batch:
                _, H, W = media.shape
                T = 1
            else:
                _, T, H, W = media.shape
            latent_t = 1 + (T - 1) // self.tokenizer_temporal_compression_factor
            gen_tokens += (H // spatial_downsample) * (W // spatial_downsample) * latent_t + 2

        actions = data_batch.get("action", [])
        if len(actions) != 1 or actions[0] is None:
            raise ValueError("history-matched packing requires exactly one action-bearing sample")
        num_rows = actions[0].shape[0]
        budget = self.chunk_length + self.max_history_actions
        if not self.chunk_length <= num_rows <= budget:
            raise ValueError(
                f"expected {self.chunk_length} future rows plus at most {self.max_history_actions} history rows, "
                f"got {num_rows}"
            )
        gen_tokens += budget
        return und_tokens, gen_tokens
