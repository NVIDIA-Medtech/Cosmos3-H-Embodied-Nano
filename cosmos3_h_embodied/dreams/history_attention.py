# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal attention with a native, action-only history prefix.

The prefix is never replicated or represented by invented video frames. A
variable-length attention batch implements one history query block and one
query block per latent frame. Teacher forcing reads clean strictly earlier
frames and the current noisy frame, preserving the same visibility as AR.
"""

import torch
import torch.nn.functional as F
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_mode_splits,
    get_causal_seq,
    get_full_only_seq,
)
from cosmos_framework.model.attention import attention
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.model.generator.utils.kv_cache import (
    ARMemoryValue,
    MemoryValue,
    TFNoisyMemoryValue,
    TFReplayCleanMemoryValue,
)


def history_visibility_indices(
    history: int, frames: int, spatial: int, text: int, replay: bool
) -> tuple[list[int], list[int], list[int]]:
    """Build exact query boundaries and gather indices into [text, live, clean].

    History queries see text and history only. Frame queries see the prefix,
    strictly earlier clean frames, and their current live frame. No query can
    reach future commands or pixels. Returned indices are CPU Python metadata.
    """
    if not 1 <= history <= 16 or frames < 1 or spatial < 1 or text < 1:
        raise ValueError("Invalid single-sample history attention geometry")
    gen_len = history + frames * spatial
    past_base = text + gen_len if replay else text
    indexes = list(range(text)) + list(range(text, text + history))
    query_offsets = [0, history]
    key_offsets = [0, len(indexes)]
    for frame in range(frames):
        past_end = history + frame * spatial
        indexes.extend(range(text))
        indexes.extend(range(past_base, past_base + past_end))
        indexes.extend(range(text + past_end, text + past_end + spatial))
        query_offsets.append(history + (frame + 1) * spatial)
        key_offsets.append(len(indexes))
    return query_offsets, key_offsets, indexes


def action_history_attention(
    query: SequencePack,
    key: SequencePack,
    value: SequencePack,
    metadata: SplitInfo,
    memory: MemoryValue | None,
    normalized_key: SequencePack | None,
) -> SequencePack:
    """Evaluate prefix-aware eager attention for one clip, preserving gradients."""
    if memory is not None and not isinstance(memory, (TFNoisyMemoryValue, TFReplayCleanMemoryValue, ARMemoryValue)):
        raise ValueError("Action history supports replay teacher forcing and eager AR prefill")
    if isinstance(memory, ARMemoryValue) and memory.frame_idx != 0:
        raise ValueError("Action history is prefetched once, at frame zero")
    if isinstance(memory, (TFNoisyMemoryValue, TFReplayCleanMemoryValue)) and memory.frames_per_chunk != 1:
        raise ValueError("Action history currently requires one latent frame per causal chunk")
    history = metadata.history_action_prefix_length
    if len(metadata.vision_token_shapes) != 1:
        raise ValueError("Action history requires one single-view sample")
    frames, height, width = metadata.vision_token_shapes[0]
    spatial = metadata.num_action_tokens_per_supertoken + height * width
    gen_len = history + frames * spatial
    text_len = query["_num_causal_tokens"]
    if query["_num_full_tokens"] != gen_len:
        raise ValueError("History attention geometry does not match packed tokens")
    tq, _ = get_causal_seq(query)  # [N_text_pad,H,D]
    tk, _ = get_causal_seq(key)  # [N_text_pad,H_kv,D]
    tv, _ = get_causal_seq(value)  # [N_text_pad,H_kv,D]
    cross_k, _ = get_causal_seq(key if normalized_key is None else normalized_key)  # [N_text_pad,H_kv,D]
    gq, _ = get_full_only_seq(query)  # [N_gen_pad,H,D]
    gk, _ = get_full_only_seq(key)  # [N_gen_pad,H_kv,D]
    gv, _ = get_full_only_seq(value)  # [N_gen_pad,H_kv,D]
    live_v = gv[:gen_len].clone()  # [N_gen,H_kv,D]
    if metadata.null_action_supertokens:
        live_v[history : history + metadata.num_action_tokens_per_supertoken] = 0  # [TCF,H_kv,D]
    keys = [cross_k[:text_len], gk[:gen_len]]  # list[N_i,H_kv,D]
    values = [tv[:text_len], live_v]  # list[N_i,H_kv,D]
    replay = isinstance(memory, TFNoisyMemoryValue)
    if replay:
        keys.append(memory.cached_clean_gen_k[0, :gen_len])  # [N_gen,H_kv,D]
        values.append(memory.cached_clean_gen_v[0, :gen_len])  # [N_gen,H_kv,D]
    qo, ko, gather = history_visibility_indices(history, frames, spatial, text_len, replay)
    q_offsets = torch.tensor(qo, device=gq.device, dtype=torch.int32)  # [T+2]
    k_offsets = torch.tensor(ko, device=gq.device, dtype=torch.int32)  # [T+2]
    indices = torch.tensor(gather, device=gq.device, dtype=torch.long)  # [N_gather]
    gathered_k = torch.cat(keys).index_select(0, indices)  # [N_gather,H_kv,D]
    gathered_v = torch.cat(values).index_select(0, indices)  # [N_gather,H_kv,D]
    generated = attention(
        gq[:gen_len].unsqueeze(0),
        gathered_k.unsqueeze(0),
        gathered_v.unsqueeze(0),
        cumulative_seqlen_Q=q_offsets,
        cumulative_seqlen_KV=k_offsets,
        max_seqlen_Q=max(history, spatial),
        max_seqlen_KV=max(b - a for a, b in zip(ko, ko[1:])),
        backend="natten",
    )  # [1,N_gen,H,D]
    text = attention(
        tq[:text_len].unsqueeze(0),
        tk[:text_len].unsqueeze(0),
        tv[:text_len].unsqueeze(0),
        is_causal=True,
        causal_type=CausalType.TopLeft,
        backend="natten",
    )  # [1,N_text,H,D]
    generated = F.pad(generated[0].flatten(-2), (0, 0, 0, gq.shape[0] - gen_len))  # [N_gen_pad,H*D]
    text = F.pad(text[0].flatten(-2), (0, 0, 0, tq.shape[0] - text_len))  # [N_text_pad,H*D]
    return from_mode_splits(text, generated, query)
