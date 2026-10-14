# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""dVRK wound-closure action dataset (dual-PSM, 20D action).

LeRobot v2.1 layout from the JHU Imerse wound-closure release. Each frame has:
- 4 camera views (we use only ``observation.images.endoscope.left``)
- ``observation.state`` (16D): per-arm ``[x, y, z, qx, qy, qz, qw, jaw]``
- ``action`` (16D): per-arm ``[x_sp, y_sp, z_sp, qx_sp, qy_sp, qz_sp, qw_sp, jaw_sp]``

Action representation emitted to the model is 20D per frame:
``[psm1_pos(3), psm1_rot6d(6), psm1_jaw(1), psm2_pos(3), psm2_rot6d(6), psm2_jaw(1)]``.
Translation+rot6d come from per-arm absolute SE(3) poses derived from the state
quaternions; jaw values come from the action setpoint columns.
"""

import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# Append-only log of samples skipped due to tolerance violations. Each line
# is one JSON object — parallel DataLoader workers can append concurrently
# because writes are < PIPE_BUF and stay atomic on POSIX/shared filesystems.
_BAD_SAMPLES_LOG = os.environ.get("DVRK_BAD_SAMPLES_LOG", "")

# Pre-curated set of known-bad sample indices, distilled from
# ``_BAD_SAMPLES_LOG`` and committed alongside the dataset code. Lets
# ``__getitem__`` fast-skip these indices instead of doing the
# attempt→catch→log dance every epoch. The runtime catch is still kept as a
# safety net for any newly-discovered bad samples.
_KNOWN_BAD_SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "bad_samples", "dvrk_wound_closure.json")


def load_runtime_bad_samples(jsonl_path: str) -> "dict[str, set[int]]":
    """Distill the cumulative runtime bad-sample JSONL (``_BAD_SAMPLES_LOG``) into a
    ``{shard_key: {row_idx, ...}}`` map so bad samples discovered in ANY prior run are
    skipped from the START of the next run — makes the per-run ``_mark_bad`` memory
    PERSISTENT across runs/chain-links WITHOUT manual curation of the committed file.

    ``shard_key`` is the last-two path components of the record's ``shard_root`` (mount-
    independent), matching the key ``_is_known_bad`` looks up. Records with a missing/
    negative ``row_idx`` or ``shard_root`` are skipped; malformed lines are ignored
    (best-effort — never block dataset init on a corrupt audit line).
    """
    out: "dict[str, set[int]]" = {}
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return out
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001 - skip a single corrupt line
                    continue
                root = str(rec.get("shard_root") or "")
                # New records state the decision explicitly. For legacy
                # records, reconstruct it from the captured exception. This
                # heals ledgers polluted by transient HF-cache EAGAIN/ENODEV
                # failures without deleting the audit evidence.
                persistent = rec.get("persistent")
                if persistent is False:
                    continue
                if persistent is None and not is_persistent_sample_failure(
                    exc_type=rec.get("exc_type"), message=rec.get("exc")
                ):
                    continue
                try:
                    row = int(rec.get("row_idx", -1))
                except (TypeError, ValueError):
                    continue
                if row < 0 or not root:
                    continue
                parts = root.rstrip("/").split("/")
                key = "/".join(parts[-2:]) if len(parts) >= 2 else root
                out.setdefault(key, set()).add(row)
    except Exception:  # noqa: BLE001 - never block init on a read failure
        pass
    return out


from cosmos_framework.utils import log
from cosmos3_h_embodied.data.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
    is_persistent_sample_failure,
)
from cosmos3_h_embodied.data.domain_utils import caption_prefix_for_root, default_caption_prefix
from cosmos3_h_embodied.data.pose_utils import (
    PoseConvention,
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
)
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint

# State / action column layout per arm.
# Both ``observation.state`` and ``action`` use the same 8-D layout per arm:
# ``[x, y, z, qx, qy, qz, qw, jaw]``.  The two arms are concatenated, so PSM1
# occupies columns 0..7 and PSM2 occupies columns 8..15.
_ARM_DIM = 8
_PSM1_SLICE = slice(0, _ARM_DIM)
_PSM2_SLICE = slice(_ARM_DIM, 2 * _ARM_DIM)
_POS_SLICE = slice(0, 3)
_QUAT_SLICE = slice(3, 7)
_JAW_COL = 7


class DvrkWoundClosureLeRobotDataset(BaseActionLeRobotDataset):
    """Action dataset for the JHU Imerse dVRK wound-closure recordings."""

    VIDEO_KEY = "observation.images.endoscope.left"

    def __init__(
        self,
        root: str | list[str] | None = None,
        roots: list[str] | None = None,
        fps: float = 5.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.05,
        split: str = "train",
        mode: str = "forward_dynamics",
        pose_convention: PoseConvention = "backward_framewise",
        action_normalization: ActionNormalization | None = None,
        viewpoint: Viewpoint = "ego_view",
        enable_fast_init: bool = False,
        normalizer_filename: str | None = None,
        prompt_source: str = "task",
        video_key: str | None = None,
        caption_prefix: str | None = None,
        caption_prefix_mode: str = "fixed",
        caption_embodiments: list[str] | None = None,
        joint_mode_weights: list[float] | None = None,
        fd_mono_condition_prob: float = 0.0,
        fd_mono_condition_views: tuple[str, ...] | list[str] = ("endoscope",),
        # Q3 experiment: decouple the VIDEO temporal rate from the action/state
        # (kinematics) rate. Defaults (None) keep video locked to fps/chunk_length
        # (symmetric, legacy behavior). Set e.g. fps=30/chunk_length=48 (kinematics)
        # + video_fps=10/video_chunk_length=16 to train 30 Hz actions against 10 Hz
        # video over the same time window. NOTE: a full training run also needs the
        # model VAE config (encode_chunk_frames / encode_exact_durations / state_t)
        # aligned to video_chunk_length; this only changes what the loader emits.
        video_fps: float | None = None,
        video_chunk_length: int | None = None,
        # Subsample windows: keep every k-th chunk-start (implemented in the base).
        # Used to bound the latent cache to a tractable window count — stride≈chunk
        # gives non-overlapping windows (every frame used once). Forwarded to base.
        sample_stride: int = 1,
        # Latent cache (hybrid): when latent_cache_root is set, read precomputed VAE
        # latents (skip video decode) instead of encoding on the fly. Forwarded to base.
        latent_cache_root: str | None = None,
        latent_cache_name: str | None = None,
        latent_cache_version_hash: str | None = None,
        latent_cache_required: bool = False,
        video_source: str = "mode_conditional",
        required_clean_video_roots: list[str] | tuple[str, ...] | None = None,
        exclude_episodes_overlay: str | None = None,
        # dVRK rig is shared across procedures (wound closure, suturebot, srth chole,
        # cao cautery) with identical 16-D dual-PSM kinematics -> this loader is
        # reusable; set a distinct embodiment_type per dataset for its own domain id
        # + normalizer. Default preserves the wound-closure behavior.
        embodiment_type: str = "dvrk_wound_closure",
        # Duplicate-frame filter (forwarded to base): reject frozen / re-encoded
        # windows whose consecutive frames stay near-identical after the 10 Hz
        # downsample. None disables it.
        duplicate_frame_max_fraction: float | None = None,
        duplicate_frame_eps: float = 1e-3,
        # Policy-gated minimal-action filter (forwarded to base): resample POLICY-mode
        # windows that are >= this fraction idle. None disables it.
        policy_idle_max_fraction: float | None = None,
        # Loader options forwarded to the base (see BaseActionLeRobotDataset).
        camera_overlap_max: float = 0.25,
        coarse_dropout: bool | dict | None = None,
        max_windows: int | None = None,
    ) -> None:
        # prompt_source: which dataset column to pull `ai_caption` from.
        #   "task"             — LeRobot's task_index → tasks.parquet[task] lookup.
        #                        Every wound-closure batch we have collapses to a
        #                        single task descriptor ("Wound Closure"), so this
        #                        gives a constant prompt across all samples.
        #   "instruction.text" — per-frame surgical sub-task description from the
        #                        data parquet (hundreds of unique strings per batch
        #                        like "Use the right gripper to drive the needle
        #                        through the top of the wound"). Aggregated to the
        #                        chunk's first frame.
        if prompt_source not in ("task", "instruction.text"):
            raise ValueError(f"prompt_source must be 'task' or 'instruction.text', got {prompt_source!r}")
        self._prompt_source = prompt_source
        # Per-instance camera override. Default keeps the single endoscope view so
        # existing experiments are unaffected; the multiview SFT points this at the
        # composed canvas key ("observation.images.multiview"). Shadows the class
        # attribute and must be set before super().__init__ / _delta_timestamps.
        if video_key is not None:
            self.VIDEO_KEY = video_key
        # Optional sentence prepended to every caption (e.g. the multiview view-layout
        # description) so the model learns the tiled layout alongside instruction.text.
        # When not explicitly set, fall back to the per-embodiment caption context
        # (robot / setting / task / instruments) distilled from the dataset READMEs,
        # so the model gets that context the raw task strings lack.
        self._explicit_caption_prefix = caption_prefix.strip() if caption_prefix else None
        self._default_embodiment_type = embodiment_type
        self._caption_prefix = (
            self._explicit_caption_prefix if self._explicit_caption_prefix else default_caption_prefix(embodiment_type)
        )
        self._normalizer_filename_override = normalizer_filename
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=embodiment_type,
            viewpoint=viewpoint,
            pose_convention=pose_convention,
            rotation_format="rot6d",
            action_normalization=action_normalization,
            tolerance_s=0.1,
            enable_fast_init=enable_fast_init,
            joint_mode_weights=joint_mode_weights,
            fd_mono_condition_prob=fd_mono_condition_prob,
            fd_mono_condition_views=fd_mono_condition_views,
            sample_stride=sample_stride,
            latent_cache_root=latent_cache_root,
            latent_cache_name=latent_cache_name,
            latent_cache_version_hash=latent_cache_version_hash,
            latent_cache_required=latent_cache_required,
            video_source=video_source,
            required_clean_video_roots=required_clean_video_roots,
            exclude_episodes_overlay=exclude_episodes_overlay,
            duplicate_frame_max_fraction=duplicate_frame_max_fraction,
            duplicate_frame_eps=duplicate_frame_eps,
            policy_idle_max_fraction=policy_idle_max_fraction,
            camera_overlap_max=camera_overlap_max,
            coarse_dropout=coarse_dropout,
            max_windows=max_windows,
        )

        # Accept either `root` (single string, single list, back-compat) or
        # `roots` (explicit list). The base class iterates `_all_shard_roots`
        # transparently, so multiple shards just become a longer list.
        if root is not None and roots is not None:
            raise ValueError("Pass `root` or `roots`, not both.")
        if roots is None:
            if root is None:
                raise ValueError("DvrkWoundClosureLeRobotDataset requires `root` or `roots`.")
            roots = [root] if isinstance(root, str) else list(root)
        if not roots:
            raise ValueError("`roots` must contain at least one dataset path.")
        self._all_shard_roots = list(roots)

        # Per-ROOT caption prefix: a combined entry (e.g. dvrk_jhu = 9 procedures under
        # one embodiment_type) must not label every root with the entry's single prefix.
        # Resolve each root to its own embodiment context; an explicit caption_prefix
        # (e.g. multiview view-layout) overrides for all roots. Indexed by dataset_idx.
        # ``caption_embodiments`` names each root's caption context explicitly (one entry per
        # root); without it the context is resolved from substrings of the root path.
        self._set_caption_prefix_mode(caption_prefix_mode)
        if caption_embodiments is not None and len(caption_embodiments) != len(self._all_shard_roots):
            raise ValueError(
                f"caption_embodiments has {len(caption_embodiments)} entries for {len(self._all_shard_roots)} roots"
            )
        if self._explicit_caption_prefix:
            self._caption_prefix_by_idx = {i: self._explicit_caption_prefix for i in range(len(self._all_shard_roots))}
        elif caption_embodiments is not None:
            self._caption_prefix_by_idx = {}
            for i, emb in enumerate(caption_embodiments):
                prefix = default_caption_prefix(emb)
                if prefix is None:
                    raise ValueError(f"no caption context registered for embodiment {emb!r}")
                self._caption_prefix_by_idx[i] = prefix
        else:
            self._caption_prefix_by_idx = {
                i: caption_prefix_for_root(r, self._default_embodiment_type)
                for i, r in enumerate(self._all_shard_roots)
            }

        # T+1 frames/states (for backward_framewise we need the next-frame pose
        # to form the per-step delta); T action setpoints for the jaw. Video may run
        # on its own (slower) rate — see video_fps/video_chunk_length (Q3).
        v_dt = (1.0 / video_fps) if video_fps else None
        self._delta_timestamps = self._build_delta_timestamps(
            self.VIDEO_KEY, self._dt, self._chunk_length, v_dt, video_chunk_length
        )

    @staticmethod
    def _build_delta_timestamps(
        video_key: str,
        dt: float,
        chunk_length: int,
        video_dt: float | None = None,
        video_chunk_length: int | None = None,
    ) -> dict[str, list[float]]:
        """LeRobot ``delta_timestamps`` for one sample window.

        ``observation.state`` and ``video`` get ``chunk+1`` stamps (T+1 poses/frames
        for per-step deltas); ``action`` gets ``chunk`` stamps. When ``video_dt`` /
        ``video_chunk_length`` are given the video stream runs on its own (typically
        slower) rate while action/state stay on ``dt``/``chunk_length`` — the Q3
        30 Hz-kinematics / 10 Hz-video recipe. Both streams start at t=0 and are
        meant to cover the same wall-clock window.
        """
        v_dt = video_dt if video_dt is not None else dt
        v_n = video_chunk_length if video_chunk_length is not None else chunk_length
        return {
            video_key: [i * v_dt for i in range(0, v_n + 1)],
            "observation.state": [i * dt for i in range(0, chunk_length + 1)],
            "action": [i * dt for i in range(0, chunk_length)],
        }

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    @staticmethod
    def _arm_abs_poses(state_arm: np.ndarray) -> np.ndarray:
        """Build (T+1, 4, 4) absolute poses for one arm from its state slice."""
        xyz = state_arm[:, _POS_SLICE]
        quat = state_arm[:, _QUAT_SLICE]
        # Normalize quaternions defensively — raw dVRK telemetry can drift.
        norms = np.linalg.norm(quat, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        quat = quat / norms
        return build_abs_pose_from_components(xyz, quat, "quat_xyzw")

    def _compute_backward_framewise_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-arm framewise relative pose + jaw setpoint.

        Returns:
            action: (T, 20) tensor laid out as
                ``[psm1_pos(3), psm1_rot6d(6), psm1_jaw(1),
                   psm2_pos(3), psm2_rot6d(6), psm2_jaw(1)]``.
            initial_pose: (4, 4) PSM1 absolute pose at t=0 (used downstream for
                reconstructing the trajectory during eval / visualization). PSM2
                initial pose is currently dropped; revisit if eval needs both.
        """
        state = sample["observation.state"].numpy().astype(np.float32)  # (T+1, 16)
        action = sample["action"]  # (T, 16) tensor

        psm1_abs = self._arm_abs_poses(state[:, _PSM1_SLICE])
        psm2_abs = self._arm_abs_poses(state[:, _PSM2_SLICE])

        psm1_rel = pose_abs_to_rel(
            poses_abs=psm1_abs, rotation_format="rot6d", pose_convention="backward_framewise"
        )  # (T, 9)
        psm2_rel = pose_abs_to_rel(
            poses_abs=psm2_abs, rotation_format="rot6d", pose_convention="backward_framewise"
        )  # (T, 9)

        psm1_jaw = action[:, _PSM1_SLICE][:, [_JAW_COL]].float()
        psm2_jaw = action[:, _PSM2_SLICE][:, [_JAW_COL]].float()

        psm1_block = torch.cat([torch.from_numpy(psm1_rel).float(), psm1_jaw], dim=-1)  # (T, 10)
        psm2_block = torch.cat([torch.from_numpy(psm2_rel).float(), psm2_jaw], dim=-1)  # (T, 10)

        full_action = torch.cat([psm1_block, psm2_block], dim=-1)  # (T, 20)
        initial_pose = torch.from_numpy(psm1_abs[0].copy()).float()
        return full_action, initial_pose

    def _compute_backward_anchored_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-arm pose **anchored to the chunk's frame-0** (GR00T-style).

        Identical to ``_compute_backward_framewise_action`` except the relative
        delta is ``delta_T = T_0^{-1} @ T_{i+1}`` instead of ``T_i^{-1} @ T_{i+1}``.
        That means every action in the chunk is expressed in the frame-0
        coordinate frame — matches GR00T's ``REL_XYZ_ROT6D`` representation.

        Returns same layout as ``_compute_backward_framewise_action``.
        """
        state = sample["observation.state"].numpy().astype(np.float32)  # (T+1, 16)
        action = sample["action"]  # (T, 16) tensor

        psm1_abs = self._arm_abs_poses(state[:, _PSM1_SLICE])
        psm2_abs = self._arm_abs_poses(state[:, _PSM2_SLICE])

        psm1_rel = pose_abs_to_rel(
            poses_abs=psm1_abs, rotation_format="rot6d", pose_convention="backward_anchored"
        )  # (T, 9)
        psm2_rel = pose_abs_to_rel(
            poses_abs=psm2_abs, rotation_format="rot6d", pose_convention="backward_anchored"
        )  # (T, 9)

        psm1_jaw = action[:, _PSM1_SLICE][:, [_JAW_COL]].float()
        psm2_jaw = action[:, _PSM2_SLICE][:, [_JAW_COL]].float()

        psm1_block = torch.cat([torch.from_numpy(psm1_rel).float(), psm1_jaw], dim=-1)  # (T, 10)
        psm2_block = torch.cat([torch.from_numpy(psm2_rel).float(), psm2_jaw], dim=-1)  # (T, 10)

        full_action = torch.cat([psm1_block, psm2_block], dim=-1)  # (T, 20)
        initial_pose = torch.from_numpy(psm1_abs[0].copy()).float()
        return full_action, initial_pose

    def _compute_backward_anchored_world_trans_action(
        self, sample: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Anchor-relative rotation + WORLD-frame translation (cosmos2-aligned).

        Identical to ``_compute_backward_anchored_action`` except the translation
        block is the raw world-frame difference ``t_{i+1} - t_0`` (not rotated into
        the frame-0 frame). Matches cosmos_predict2's ``compute_diff_actions``.

        Returns same layout as ``_compute_backward_anchored_action``.
        """
        state = sample["observation.state"].numpy().astype(np.float32)  # (T+1, 16)
        action = sample["action"]  # (T, 16) tensor

        psm1_abs = self._arm_abs_poses(state[:, _PSM1_SLICE])
        psm2_abs = self._arm_abs_poses(state[:, _PSM2_SLICE])

        psm1_rel = pose_abs_to_rel(
            poses_abs=psm1_abs,
            rotation_format="rot6d",
            pose_convention="backward_anchored_world_trans",
        )  # (T, 9)
        psm2_rel = pose_abs_to_rel(
            poses_abs=psm2_abs,
            rotation_format="rot6d",
            pose_convention="backward_anchored_world_trans",
        )  # (T, 9)

        psm1_jaw = action[:, _PSM1_SLICE][:, [_JAW_COL]].float()
        psm2_jaw = action[:, _PSM2_SLICE][:, [_JAW_COL]].float()

        psm1_block = torch.cat([torch.from_numpy(psm1_rel).float(), psm1_jaw], dim=-1)  # (T, 10)
        psm2_block = torch.cat([torch.from_numpy(psm2_rel).float(), psm2_jaw], dim=-1)  # (T, 10)

        full_action = torch.cat([psm1_block, psm2_block], dim=-1)  # (T, 20)
        initial_pose = torch.from_numpy(psm1_abs[0].copy()).float()
        return full_action, initial_pose

    def _compute_absolute_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-arm absolute pose + jaw setpoint (skips the first/anchor frame).

        Returns (T, 20) action and PSM1 anchor pose, mirroring framewise output
        layout so downstream code does not need to special-case the mode.
        """
        state = sample["observation.state"].numpy().astype(np.float32)[1:]  # (T, 16)
        action = sample["action"]  # (T, 16)

        psm1_abs = self._arm_abs_poses(state[:, _PSM1_SLICE])
        psm2_abs = self._arm_abs_poses(state[:, _PSM2_SLICE])

        def _flatten(poses_abs: np.ndarray) -> torch.Tensor:
            translation = torch.from_numpy(poses_abs[:, :3, 3]).float()
            rot_matrix = torch.from_numpy(poses_abs[:, :3, :3]).float()
            rot6d = convert_rotation(rot_matrix, input_format="matrix", output_format="rot6d").float()
            return torch.cat([translation, rot6d], dim=-1)  # (T, 9)

        psm1_pose = _flatten(psm1_abs)
        psm2_pose = _flatten(psm2_abs)
        psm1_jaw = action[:, _PSM1_SLICE][:, [_JAW_COL]].float()
        psm2_jaw = action[:, _PSM2_SLICE][:, [_JAW_COL]].float()

        full_action = torch.cat([psm1_pose, psm1_jaw, psm2_pose, psm2_jaw], dim=-1)  # (T, 20)
        # ``_compute_backward_framewise_action`` exposes the t=0 PSM1 pose;
        # do the same here so callers see a uniform return shape.
        initial_pose = torch.eye(4, dtype=torch.float32)
        initial_pose[:3, 3] = torch.from_numpy(psm1_abs[0, :3, 3]).float()
        initial_pose[:3, :3] = torch.from_numpy(psm1_abs[0, :3, :3]).float()
        return full_action, initial_pose

    # ------------------------------------------------------------------
    # Episode filtering
    # ------------------------------------------------------------------

    def _filter_valid_episodes(self, meta: LeRobotDatasetMetadata, episode_ids: list[int]) -> list[int]:
        """Drop episodes whose ``tasks`` metadata is empty/whitespace."""
        kept: list[int] = []
        dropped = 0
        for ep_id in episode_ids:
            ep = meta.episodes[ep_id]
            tasks = ep.get("tasks", [])
            if isinstance(tasks, str):
                tasks = [tasks]
            if any(t and str(t).strip() for t in (tasks or [])):
                kept.append(ep_id)
            else:
                dropped += 1
        if dropped:
            log.info(
                f"DvrkWoundClosureLeRobotDataset: dropped {dropped} / {len(episode_ids)} episodes with empty prompt"
            )
        return kept

    # ------------------------------------------------------------------
    # Action spec / __getitem__
    # ------------------------------------------------------------------


    def _extract_caption(self, sample: dict, dataset_idx: int | None = None) -> str:
        """Pull the per-sample text caption per ``self._prompt_source``.

        - ``"task"``: LeRobot's resolved task descriptor (scalar string,
          collapses to ``"Wound Closure"`` for all our batches).
        - ``"instruction.text"``: per-frame surgical sub-task string from
          the data parquet. Per-frame values can vary across the chunk;
          we collapse to the FIRST frame's instruction so each chunk's
          caption matches its starting action. Falls back to ``sample["task"]``
          when the column is missing or empty.

        ``dataset_idx`` (source shard root, from ``_fetch_sample``) picks the PER-ROOT
        caption prefix so a combined entry labels each root with its own procedure.
        """
        caption_idx = self._caption_root_index(dataset_idx)
        prefix = (
            self._caption_prefix_by_idx.get(caption_idx, self._caption_prefix)
            if caption_idx is not None
            else self._caption_prefix
        )
        def _prefixed(text: str) -> str:
            # Prepend the per-root caption context so the caption describes both the
            # robot/procedure and the per-frame sub-task.
            return f"{prefix} {text}" if prefix else text

        if self._prompt_source == "instruction.text":
            it = sample.get("instruction.text")
            if it is None:
                pass  # fall through to task
            elif isinstance(it, str):
                if it.strip():
                    return _prefixed(it)
            elif hasattr(it, "__iter__"):
                # Sequence of per-frame strings (or torch/np container).
                try:
                    first = it[0] if hasattr(it, "__getitem__") else next(iter(it))
                except (IndexError, StopIteration):
                    first = None
                if isinstance(first, bytes):
                    first = first.decode("utf-8", errors="replace")
                if isinstance(first, str) and first.strip():
                    return _prefixed(first)
        return _prefixed(sample["task"])

    def _build_action_spec(self) -> ActionSpec:
        """Dual-PSM 20D: ``[Pos, Rot6d, Gripper] × 2``."""
        return build_action_spec(
            Pos(),
            Rot("rot6d"),
            Gripper(),
            Pos(),
            Rot("rot6d"),
            Gripper(),
        )

    def _normalizer_filename(self) -> str:
        """Override the bundled-stats filename via the optional ctor arg.

        Lets a YAML config point at a different normalizer (e.g. one computed
        on a different fps / dataset mix) without renaming the canonical file,
        so existing checkpoints keep loading the stats they were trained against.
        """
        if self._normalizer_filename_override:
            return self._normalizer_filename_override
        return super()._normalizer_filename()

    @property
    def action_dim(self) -> int:
        return 20

    def _log_bad_sample(self, idx: int, exc: BaseException) -> None:
        """Append a single JSONL record describing a tolerance-failed sample.

        Workers may write concurrently; one open()+write()+close() per call
        keeps each append below PIPE_BUF (atomic on POSIX). Worst case the
        log misses entries (best-effort), never blocks training.
        """
        try:
            dataset_idx, row_idx, _, _ = self._resolve_index(idx)
        except Exception:
            dataset_idx, row_idx = -1, -1
        persistent = is_persistent_sample_failure(exc)
        # Only row-specific defects belong in the in-session skip set. A
        # shared-cache outage or a policy-only idle rejection must not poison
        # this row for FDM/IDM or future retries.
        if persistent:
            self._mark_bad(dataset_idx, row_idx)
        record = {
            "timestamp": time.time(),
            "idx": int(idx),
            "dataset_idx": int(dataset_idx),
            "row_idx": int(row_idx),
            "shard_root": self._registered_shard_root(dataset_idx) or "",
            "exc_type": type(exc).__name__,
            "exc": str(exc)[:500],
            "persistent": persistent,
        }
        if not _BAD_SAMPLES_LOG:
            return
        try:
            os.makedirs(os.path.dirname(_BAD_SAMPLES_LOG) or ".", exist_ok=True)
            with open(_BAD_SAMPLES_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # never block training on a logging failure

    def _load_known_bad_samples(self, *, runtime_log: str | None = None) -> None:
        """Load the committed known-bad-samples JSON into a per-shard-key set.

        Idempotent + lazy: only reads the file once per process. Stored as
        ``self._known_bad_by_shard_key`` mapping ``"<batch>/<sub>"`` to
        ``frozenset[int]`` of bad row indices.
        """
        if getattr(self, "_known_bad_loaded", False):
            return
        self._known_bad_by_shard_key: dict[str, frozenset[int]] = {}
        self._known_bad_loaded = True
        # (1) Committed, hand-curated known-bad file (optional).
        merged: dict[str, set[int]] = {}
        n_committed = 0
        if os.path.isfile(_KNOWN_BAD_SAMPLES_PATH):
            try:
                with open(_KNOWN_BAD_SAMPLES_PATH) as _f:
                    data = json.load(_f)
                for key, indices in (data.get("bad_samples") or {}).items():
                    merged.setdefault(key, set()).update(int(i) for i in indices)
                n_committed = sum(len(v) for v in merged.values())
            except Exception as _exc:  # noqa: BLE001
                log.warning(f"DvrkWoundClosureLeRobotDataset: failed to load known-bad-samples file: {_exc}")
        # (2) Cumulative runtime JSONL — makes _mark_bad PERSIST across runs without
        # manual curation. Folded in ON TOP of the committed set (union).
        runtime_log = _BAD_SAMPLES_LOG if runtime_log is None else runtime_log
        runtime = load_runtime_bad_samples(runtime_log)
        for key, rows in runtime.items():
            merged.setdefault(key, set()).update(rows)
        n_runtime = sum(len(v) for v in runtime.values())
        self._known_bad_by_shard_key = {k: frozenset(v) for k, v in merged.items()}
        total = sum(len(v) for v in self._known_bad_by_shard_key.values())
        log.info(
            f"DvrkWoundClosureLeRobotDataset: known-bad skip-set = {total} indices across "
            f"{len(self._known_bad_by_shard_key)} shard key(s) "
            f"(committed={n_committed}, runtime-jsonl={n_runtime} from {runtime_log})"
        )

    def _is_known_bad(self, dataset_idx: int, row_idx: int) -> bool:
        """Return True iff ``(dataset_idx, row_idx)`` is in the curated bad set."""
        if not self._known_bad_by_shard_key:
            return False
        key = self._shard_key(dataset_idx)
        if key is None:
            return False
        # Match on the last two path components so the lookup is independent
        # of mount point (e.g. /data vs a full shared-filesystem path both resolve to the
        # same canonical key like "<batch>/<dataset>").
        bad_set = self._known_bad_by_shard_key.get(key)
        return bad_set is not None and row_idx in bad_set

    @staticmethod
    def _next_skip_index(cur_idx: int, n: int, rng: random.Random | None = None) -> int:
        """Pick the next index to try after a bad sample.

        Jump to a RANDOM index instead of ``cur_idx + 1``. Sequential +1 gets
        trapped inside a contiguous bad-video block (a truncated/unreadable
        region can span thousands of consecutive samples) and blows past the
        256-attempt bound -> the worker dies -> the rank dies -> NCCL timeout.
        A random draw escapes any contiguous bad region in a handful of tries
        (unless the dataset is almost entirely bad, in which case the bound
        still fails loudly as intended).

        ``rng`` should be seeded from the original ``idx`` so that joint-eval
        calls for the same sample_id (FDM / IDM / policy) pick the same
        replacement window instead of diverging via the global RNG state.
        """
        if n <= 1:
            return 0
        return (rng or random).randrange(n)

    def _compute_convention_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch action-compute on the pose convention. Called inside the retry
        loop so any ``tolerance:`` AssertionError it raises (camera-motion, etc.)
        triggers a resample rather than crashing the worker."""
        if self._pose_convention == "backward_framewise":
            return self._compute_backward_framewise_action(sample)
        elif self._pose_convention == "backward_anchored":
            return self._compute_backward_anchored_action(sample)
        elif self._pose_convention == "backward_anchored_world_trans":
            return self._compute_backward_anchored_world_trans_action(sample)
        elif self._pose_convention == "absolute":
            return self._compute_absolute_action(sample)
        else:
            raise ValueError(f"Unknown pose_convention: {self._pose_convention}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Wrap _fetch_sample so tolerance-violating samples are logged + skipped
        # instead of crashing the DataLoader worker. The committed
        # known-bad-samples JSON gives us a fast O(1) pre-skip for the
        # ~3000 samples that have already been seen to violate tolerance;
        # the slow attempt→catch→log path catches anything new.
        self._load_known_bad_samples()
        attempts = 0
        cur_idx = idx
        # Seed from original idx so joint-eval calls (FDM/IDM/policy) for the
        # same sample_id pick identical replacement windows on skip, preventing
        # episode misalignment across modes when camera-motion or bad-video
        # causes a jump.
        jump_rng = random.Random(idx)
        sample_dataset_idx: int | None = None
        while attempts < 256:
            # Fast path: skip known-bad before paying the parquet+video load.
            try:
                dataset_idx, row_idx, _, _ = self._resolve_index(cur_idx)
                if self._is_known_bad(dataset_idx, row_idx):
                    attempts += 1
                    cur_idx = self._next_skip_index(cur_idx, len(self), jump_rng)
                    continue
            except Exception:
                pass  # let the existing slow-path handle it
            try:
                mode, sample_dataset_idx, _, sample = self._fetch_sample(cur_idx)
                # Compute the action INSIDE the retry loop so tolerance filters raised
                # during compute — camera-arm panning (rob_surgical) and the
                # minimal-action policy gate below — resample another window instead
                # of propagating out and crashing the DataLoader worker.
                action, initial_pose = self._compute_convention_action(sample)
                if mode == "policy" and self._policy_idle_max_fraction is not None:
                    # Policy-gated minimal-action filter: drop windows the policy
                    # expert would learn "do nothing" from. Kept for FDM/IDM (static
                    # is a valid dynamics target), hence the mode gate.
                    n_idle = self._compute_idle_frames(action)
                    if n_idle is not None and int(n_idle) >= (self._policy_idle_max_fraction * int(action.shape[0])):
                        raise AssertionError("tolerance: minimal-action window (policy mode)")
                break
            except AssertionError as e:
                msg = str(e)
                if "tolerance" not in msg:
                    raise
                self._log_bad_sample(cur_idx, e)
                attempts += 1
                cur_idx = self._next_skip_index(cur_idx, len(self), jump_rng)
            except (FileNotFoundError, OSError, IndexError, EOFError, StopIteration, RuntimeError, ValueError) as e:
                # Missing/truncated/undecodable video (missing files or missing
                # frames -> out-of-range seek), or a torchcodec/ffmpeg decode
                # error. Skip this sample instead of letting the exception kill
                # the DataLoader worker -> which crashes the rank -> NCCL collective
                # timeout across the whole job. The 256-attempt bound below still
                # fails loudly if an entire shard is unreadable (not just sparse
                # bad frames), so this can't silently mask a fully-broken dataset.
                self._log_bad_sample(cur_idx, e)
                attempts += 1
                cur_idx = self._next_skip_index(cur_idx, len(self), jump_rng)
        else:
            raise RuntimeError(
                f"DvrkWoundClosureLeRobotDataset: 256 consecutive samples failed to "
                f"load (tolerance violations or unreadable video) starting from "
                f"idx={idx}. See {_BAD_SAMPLES_LOG}."
            )

        ai_caption = self._extract_caption(sample, sample_dataset_idx)
        # None when the latent cache is active (base skipped video decode); the
        # cached latent is injected into the result below. `action`/`initial_pose`
        # were computed inside the retry loop above (so tolerance filters resample).
        video = sample.get(self.VIDEO_KEY)

        result = self._build_result(
            mode=mode, video=video, action=action, ai_caption=ai_caption, initial_pose=initial_pose
        )
        if self._latent_cache_active:
            # Replace the (never-decoded) video with the precomputed VAE latent and
            # set vision_is_latent=True so the transform skips resize/pad and the
            # model forward skips encode. Keyed by cur_idx (the post-skip index the
            # precompute encoded), matching the CMR/Versius cached loaders.
            result = self._inject_cached_latent(result, cur_idx)
        elif "video" in result:
            # Live sample in a possibly-hybrid batch: tag it so the model's
            # per-sample gating encodes raw items alongside cached latents.
            result["vision_is_latent"] = torch.tensor(False)
        return result
