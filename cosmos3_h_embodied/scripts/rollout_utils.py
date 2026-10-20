# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Building blocks shared by the rollout scripts (``quant_eval``, ``simulate``) and the policy server.

- :func:`make_policy_transform`, :class:`QuantileDenormalizer`: the training transform (without
  augmentation) and the action normalization of a family.
- :func:`make_raw_sample`: a model input in the format the datasets produce, before the training
  transform (conditioning frame, actions, action history, caption, embodiment, mode).
- :func:`generate_chunk`: one 2.4 s chunk through the inference pipeline, conditioned on a frame.
- :func:`reanchor_history`: the action history of the next chunk from the actions of this one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos3_h_embodied.data.action_normalization import load_action_stats, normalize_action
from cosmos3_h_embodied.data.pose_utils import pose_abs_to_rel, pose_rel_to_abs
from cosmos3_h_embodied.data.transforms import (
    ActionTransformPipeline,
    reflection_pad_to_target,
    remove_reflection_padding,
)

CHUNK = 24  # action rows (and generated frames) per chunk
MAX_HISTORY_ACTIONS = 16
DEFAULT_MAX_ACTION_DIM = 64
RESOLUTION = "256"  # the recipe's resolution tier (every family's dataset entry)
POSE_CONVENTION = "backward_anchored_world_trans"
NORMALIZERS_DIR = Path(__file__).resolve().parents[1] / "data" / "normalizers"


def make_policy_transform(max_action_dim: int = DEFAULT_MAX_ACTION_DIM) -> ActionTransformPipeline:
    """The training transform of the H16 recipe without tokenizer, dropout and augmentation."""
    return ActionTransformPipeline(
        pad_keys=["video"],
        keep_aspect_ratio=True,
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        video_temporal_downsample=4,
        max_action_dim=max_action_dim,
        action_channel_masking=True,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
        format_prompt_as_json=False,
        video_augment=None,
    )


class QuantileDenormalizer:
    """Training ``quantile`` normalization (``global`` q01/q99 stats) and its inverse."""

    def __init__(self, normalizer: str | Path) -> None:
        path = Path(normalizer)
        if not path.is_file():
            path = NORMALIZERS_DIR / str(normalizer)
        stats = load_action_stats(str(path), stats_key="global")
        self.path = path
        self.stats = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in stats.items()}
        self.q01 = self.stats["q01"]  # [D]
        self.range = (self.stats["q99"] - self.q01).clamp(min=1e-8)  # [D]

    @property
    def dim(self) -> int:
        return int(self.q01.shape[0])

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1.0) / 2.0 * self.range.to(action.device) + self.q01.to(action.device)  # [...,D]

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """As in training, including the clip to ``[-NORM_CLIP, NORM_CLIP]`` (e.g. for ``history_action``)."""
        stats = {k: v.to(action.device) for k, v in self.stats.items()}
        return normalize_action(action, "quantile", stats)  # [...,D]


def make_raw_sample(
    image: torch.Tensor,
    prompt: str,
    domain_id: int,
    raw_action_dim: int,
    *,
    mode: str = "policy",
    action: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    fps: int = 10,
) -> dict[str, Any]:
    """A sample as the datasets return it, for :class:`~cosmos3_h_embodied.data.transforms.ActionTransformPipeline`.

    Args:
        image: Conditioning frame ``[3, H, W]`` uint8.
        prompt: Caption (robot and setting context + task), as in training.
        domain_id: Embodiment ID (``domain_utils.get_domain_id``).
        raw_action_dim: Unpadded action width of the embodiment.
        mode: ``"forward_dynamics"`` (``action`` required) or ``"policy"``.
        action: ``[24, raw_action_dim]`` normalized actions (forward dynamics).
        history: ``[nH, raw_action_dim]`` normalized past actions, ``nH <= 16``, oldest first.
    """
    if mode not in ("forward_dynamics", "policy"):
        raise ValueError(f"mode must be 'forward_dynamics' or 'policy', got {mode!r}")
    if action is None:
        if mode == "forward_dynamics":
            raise ValueError("forward dynamics needs the chunk's actions")
        action = torch.zeros((CHUNK, raw_action_dim), dtype=torch.float32)
    action = torch.as_tensor(action, dtype=torch.float32)
    if action.shape != (CHUNK, raw_action_dim):
        raise ValueError(f"'action' must be [{CHUNK}, {raw_action_dim}], got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise ValueError("'action' must be finite")
    sample: dict[str, Any] = {
        "video": image.unsqueeze(1).repeat(1, CHUNK + 1, 1, 1),  # [3,T,H,W]; only frame 0 is conditioning
        "action": action,
        "ai_caption": prompt,
        "mode": mode,
        "conditioning_fps": torch.tensor(fps, dtype=torch.long),
        "domain_id": torch.tensor(domain_id, dtype=torch.long),
    }
    if history is not None:
        history = torch.as_tensor(history, dtype=torch.float32)
        if history.numel() == 0:
            history = history.reshape(0, raw_action_dim)
        if history.ndim != 2 or history.shape[1] != raw_action_dim:
            raise ValueError(f"'history_action' must be [nH, {raw_action_dim}], got {tuple(history.shape)}")
        if history.shape[0] > MAX_HISTORY_ACTIONS:
            raise ValueError(f"at most {MAX_HISTORY_ACTIONS} history actions, got {history.shape[0]}")
        if not torch.isfinite(history).all():
            raise ValueError("'history_action' must be finite")
        sample["history_action"] = history
    return sample


def arm_layout(raw_action_dim: int) -> tuple[int, bool]:
    """``(number of arms, has gripper)``: arms of 10 (pose + gripper) or 9 values (pose only)."""
    gripper = raw_action_dim % 10 == 0
    width = 10 if gripper else 9
    if raw_action_dim % width:
        raise ValueError(f"cannot split a {raw_action_dim}-dim action into arms of 9 or 10 values")
    return raw_action_dim // width, gripper


def reanchor_history(action: np.ndarray, num_history: int = MAX_HISTORY_ACTIONS) -> np.ndarray:
    """History of the next chunk from this chunk's raw (denormalized) actions ``[24, D]``.

    Row ``i`` (1-24) of a chunk holds the pose at ``t = i`` relative to the anchor ``t = 0`` and
    the gripper command at ``t = i - 1``. The next chunk is anchored at ``t = 24``; its history row
    ``j`` holds the pose at ``t = 24 - (num_history - 1) + j`` relative to that anchor and the gripper
    command one step earlier: the last ``num_history`` rows of this chunk, re-anchored, with their
    gripper commands. Exact for the dataset conventions without motion scaling (CMR scales rotation
    angles by the console's scaling, so the re-anchored rotations are an approximation there).
    """
    action = np.asarray(action, dtype=np.float64)
    num_arms, gripper = arm_layout(action.shape[1])
    width = 10 if gripper else 9
    rows = action[-num_history:].copy()
    for arm in range(num_arms):
        pose = action[:, arm * width : arm * width + 9]
        poses_abs = pose_rel_to_abs(pose, rotation_format="rot6d", pose_convention=POSE_CONVENTION)
        anchored = np.concatenate([poses_abs[-1:], poses_abs[-num_history:]])  # new anchor, then the poses
        rows[:, arm * width : arm * width + 9] = pose_abs_to_rel(
            anchored, rotation_format="rot6d", pose_convention=POSE_CONVENTION
        )
    return rows.astype(np.float32)


class CaptureVideo:
    """Keeps the generated video, which the inference pipeline only writes to a lossy MP4."""

    def __init__(self) -> None:
        import cosmos_framework.inference.inference as cf_inference

        self._module = cf_inference
        self.video: torch.Tensor | None = None
        self._save = cf_inference.save_img_or_video

    def __enter__(self) -> "CaptureVideo":
        def save(sample: torch.Tensor, *args, **kwargs):
            self.video = sample.detach().cpu()
            return self._save(sample, *args, **kwargs)

        self._module.save_img_or_video = save
        return self

    def __exit__(self, *exc) -> None:
        self._module.save_img_or_video = self._save


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    if video.dtype == torch.uint8:
        return video
    return (video.float().clamp(0, 1) * 255).round().to(torch.uint8)


def generate_chunk(
    pipe, sample_args, data_batch: dict[str, Any], condition: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Generate one chunk with the inference pipeline.

    Args:
        pipe: The inference pipeline (``setup.get_inference_cls().create(setup)``).
        sample_args: Built sample arguments (seed, output directory, sampling settings).
        data_batch: The collated sample; its video is overwritten with the conditioning frame.
        condition: Conditioning frame ``[3, 1, h, w]`` uint8 in the content region (the last
            generated frame of the previous chunk), or ``None`` for the sample's own first frame.

    Returns:
        The generated video ``[3, 25, h, w]`` uint8 without padding (frame 0 is the conditioning
        frame as the model saw it), and the predicted actions (all rows, normalized) or ``None``.
    """
    image_size = data_batch["image_size"][0]
    video = data_batch["video"][0]  # (C, T, H, W) uint8, content top-left, reflection-padded
    if condition is None:
        first = video[:, :1].clone()
    else:
        target_h, target_w = int(image_size[0]), int(image_size[1])
        padded = reflection_pad_to_target({"video": condition.float()}, ["video"], True, target_w, target_h)
        first = padded["video"].round().clamp(0, 255).to(torch.uint8)
    # The model sees only the conditioning frame: blank out any recorded future frames.
    video.copy_(first.expand_as(video))
    with CaptureVideo() as capture:
        outputs = pipe.generate_batch([sample_args], data_batch)
    if not outputs or outputs[0].status != "success" or capture.video is None:
        raise RuntimeError(f"generation failed for {sample_args.name}: {outputs[0].message if outputs else ''}")
    action = outputs[0].outputs[0].content.get("action") if outputs[0].outputs else None
    if action is not None:
        action = torch.as_tensor(action, dtype=torch.float32)
    return remove_reflection_padding(to_uint8(capture.video), image_size), action
