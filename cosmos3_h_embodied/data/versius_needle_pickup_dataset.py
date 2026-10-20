# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CMR Versius needle-pickup action dataset (dual hand-controller, 20D action).

LeRobot v3.0 layout converted from the Open-H ``cmr-surgical-60hz`` release
(``needle-pickup``: 249 episodes, 60 fps, stereo endoscope). Each frame has:

- 2 camera views (we use only ``observation.images.endoscope_left``)
- ``action`` (26D): the haptic LEADER (hand-controller) command stream,
  L/R INTERLEAVED per component (matches ``meta/info.json`` ``names``)::

    0 x_L  1 x_R  2 y_L  3 y_R  4 z_L  5 z_R
    6 qx_L 7 qx_R 8 qy_L 9 qy_R 10 qz_L 11 qz_R 12 qw_L 13 qw_R
    14 clutch_L 15 clutch_R  16 energy_L 17 energy_R  18 thumbBtn_L 19 thumbBtn_R
    20 pince_L 21 pince_R  22 thumbX_L 23 thumbX_R  24 thumbY_L 25 thumbY_R

  Verified against the data: under this interleaved indexing the
  L quat [6,8,10,12] and R quat [7,9,11,13] both norm to *exactly* 1.0
  (std 0.0); the binary clutch buttons land on 14/15, the continuous 0-1 pince
  on 20/21. The earlier "blocked, 13-dims-per-hand" reading ([0,1,2]/[13,14,15],
  quats [3..6]/[16..19]) was WRONG — it read the all-zero energy/thumbstick
  button columns as the right-arm pose, producing a degenerate right arm.
  The emitted 20-D action is invariant to the source layout as long as these
  indices match the data.
- ``observation.state`` (24D): discrete rig-configuration flags — NOT poses,
  so unlike the dVRK datasets the absolute pose trajectory is taken from the
  action columns (the leader pose command IS the commanded tool pose).

Action representation emitted to the model is 20D per frame, matching the
dVRK/suturebot surgical convention so the checkpoints transfer:
``[left_pos(3), left_rot6d(6), left_pince(1), right_pos(3), right_rot6d(6), right_pince(1)]``.
The 6 button/thumbstick dims (14-19, 22-25) are dropped.

NOTE: ``fd_mono_*`` kwargs are accepted here but forwarded to the base only
when that base actually defines them (signature-guarded in ``__init__``); on a base that lacks them a mono-FD
request raises loudly instead of silently no-op'ing. This single-view
needle-pickup dataset itself defaults ``fd_mono_condition_prob=0`` — mono-FD is
meaningful only for the multiview suturing subclass.
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.utils import log
from cosmos3_h_embodied.data.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos3_h_embodied.data.pose_utils import (
    PoseConvention,
    build_abs_pose_from_components,
    compute_idle_frames,
    convert_rotation,
    pose_abs_to_rel,
)
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint

# INTERLEAVED action-column layout — L/R alternate per component (the ORIGINAL
# CMR export layout; matches meta/info.json `names`). Verified against the data:
# L/R quaternions norm to *exactly* 1.0 only under this indexing. The old
# "blocked, 13-dims-per-hand" reading ([0,1,2]/[13,14,15], quats [3..6]/[16..19])
# read the all-zero energy/thumbstick button columns as the right-arm pose,
# producing a degenerate right arm. The whole codebase uses this ONE (interleaved)
# convention.
_L_POS_COLS = [0, 2, 4]
_L_QUAT_COLS = [6, 8, 10, 12]  # (qx, qy, qz, qw), unit-norm in the raw data
_L_PINCE_COL = 20
_R_POS_COLS = [1, 3, 5]
_R_QUAT_COLS = [7, 9, 11, 13]
_R_PINCE_COL = 21
# Clutch buttons — verified against needle_pickup_v30 meta/info.json action names
# (clutchBtn_left=14, clutchBtn_right=15). 0 = engaged (tracking), 1 = pressed
# (disengaged / repositioning). Same layout as VersiusSuturingLeRobotDataset.
_L_CLUTCH_COL = 14
_R_CLUTCH_COL = 15
# rot6d identity = first two columns of the 3x3 identity, flattened (a "no rotation" delta).
_ROT6D_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


class VersiusNeedlePickupLeRobotDataset(BaseActionLeRobotDataset):
    """Action dataset for the Open-H CMR Versius needle-pickup recordings."""

    VIDEO_KEY = "observation.images.endoscope_left"

    def __init__(
        self,
        root: str | list[str] | None = None,
        roots: list[str] | None = None,
        fps: float = 10.0,
        chunk_length: int = 16,
        split_seed: int = 42,
        split_val_ratio: float = 0.05,
        split: str = "train",
        mode: str = "forward_dynamics",
        pose_convention: PoseConvention = "backward_anchored_world_trans",
        action_normalization: ActionNormalization | None = None,
        viewpoint: Viewpoint = "ego_view",
        enable_fast_init: bool = False,
        normalizer_filename: str | None = None,
        prompt_source: str = "task",
        video_key: str | None = None,
        letterbox_hw: tuple[int, int] | None = None,
        caption_prefix: str | None = None,
        joint_mode_weights: list[float] | None = None,
        latent_cache_root: str | None = None,
        latent_cache_name: str | None = None,
        latent_cache_version_hash: str | None = None,
        latent_cache_required: bool = False,
        video_source: str = "mode_conditional",
        exclude_episodes_overlay: str | None = None,
        # Subsample windows (every k-th chunk-start); bounds the latent cache.
        sample_stride: int = 1,
        latent_cache_variant_roots: list[str] | tuple[str, ...] | None = None,
        policy_view_dropout_mono_prob: float = 0.0,
        policy_view_dropout_single_prob: float = 0.0,
        embodiment_type: str = "needle_pickup",
        # mono-FD (endoscope-only forward dynamics). Forwarded to the base ONLY
        # when that base supports it (see the super().__init__ guard below); the
        # ("endoscope",) default maps to the top-row endoscope tile of the
        # Versius multiview canvas resolved by the network's cross-view mask.
        fd_mono_condition_prob: float = 0.0,
        fd_mono_condition_views: tuple[str, ...] | list[str] = ("endoscope",),
        image_aug_strength: float = 0.0,
        image_aug_pixel_drop: float = 0.0,
        idle_filter_frac: float | None = None,
        action_chunk_length: int | None = None,
        action_fps: float | None = None,
        # Loader options forwarded to the base (see BaseActionLeRobotDataset).
        camera_overlap_max: float = 0.25,
        coarse_dropout: bool | dict | None = None,
        max_windows: int | None = None,
    ) -> None:
        # Only "task" exists for this dataset (single descriptor: "Needle
        # pickup and handover"); there is no per-frame instruction column.
        if prompt_source != "task":
            raise ValueError(f"prompt_source must be 'task' for needle-pickup, got {prompt_source!r}")
        self._prompt_source = prompt_source
        # Per-instance camera override; shadows the class attribute and must be
        # set before super().__init__ / _delta_timestamps.
        if video_key is not None:
            self.VIDEO_KEY = video_key
        self._caption_prefix = caption_prefix.strip() if caption_prefix else None
        # Distinguish "caller said nothing" (None -> subclasses may synthesize a prefix)
        # from "caller explicitly asked for NO prefix" (""), which the truthiness test
        # above cannot express. Needed to evaluate a checkpoint trained before the
        # procedure/instrument prefix existed — feeding it that prefix is out-of-
        # distribution. Mirrors _explicit_caption_prefix in dvrk_wound_closure_dataset.
        self._explicit_caption_prefix = caption_prefix is not None
        # Optional cosmos2-style letterbox (out_h, out_w). Coerce to int: a
        # saved config.yaml round-trips list scalars as strings.
        self._letterbox_hw = tuple(int(v) for v in letterbox_hw) if letterbox_hw is not None else None
        self._normalizer_filename_override = normalizer_filename
        # Forward mono-FD kwargs only if the installed base accepts them, so this
        # file still works with a base class that lacks fd_mono_* (see the
        # module docstring). Requesting mono-FD against an unsupporting base is a
        # loud error rather than a silent no-op.
        import inspect

        _base_params = inspect.signature(BaseActionLeRobotDataset.__init__).parameters
        _fd_mono_kwargs: dict[str, Any] = {}
        if "fd_mono_condition_prob" in _base_params:
            _fd_mono_kwargs["fd_mono_condition_prob"] = fd_mono_condition_prob
            _fd_mono_kwargs["fd_mono_condition_views"] = fd_mono_condition_views
        elif float(fd_mono_condition_prob) > 0.0:
            raise ValueError(
                "fd_mono_condition_prob > 0 was requested but the installed "
                "BaseActionLeRobotDataset does not support mono-FD (fd_mono_* "
                "kwargs). Upgrade the base class or set fd_mono_condition_prob=0."
            )
        # Forward image-aug / idle-filter kwargs only if the installed base
        # supports them (same portability guard as the mono-FD kwargs above).
        _aug_kwargs: dict[str, Any] = {}
        for _k, _v in (
            ("image_aug_strength", image_aug_strength),
            ("image_aug_pixel_drop", image_aug_pixel_drop),
            ("idle_filter_frac", idle_filter_frac),
            ("action_chunk_length", action_chunk_length),
            ("action_fps", action_fps),
        ):
            if _k in _base_params:
                _aug_kwargs[_k] = _v
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
            # Native 60 fps: 10/30 Hz queries land exactly on frames, so half a
            # native frame period catches timestamp corruption instead of
            # silently accepting multi-frame slop.
            tolerance_s=0.008,
            enable_fast_init=enable_fast_init,
            joint_mode_weights=joint_mode_weights,
            latent_cache_root=latent_cache_root,
            latent_cache_name=latent_cache_name,
            latent_cache_version_hash=latent_cache_version_hash,
            latent_cache_required=latent_cache_required,
            video_source=video_source,
            exclude_episodes_overlay=exclude_episodes_overlay,
            sample_stride=sample_stride,
            latent_cache_variant_roots=latent_cache_variant_roots,
            policy_view_dropout_mono_prob=policy_view_dropout_mono_prob,
            policy_view_dropout_single_prob=policy_view_dropout_single_prob,
            camera_overlap_max=camera_overlap_max,
            coarse_dropout=coarse_dropout,
            max_windows=max_windows,
            **_fd_mono_kwargs,
            **_aug_kwargs,
        )

        # Accept either `root` (single string, single list, back-compat) or
        # `roots` (explicit list).
        if root is not None and roots is not None:
            raise ValueError("Pass `root` or `roots`, not both.")
        if roots is None:
            if root is None:
                raise ValueError("VersiusNeedlePickupLeRobotDataset requires `root` or `roots`.")
            roots = [root] if isinstance(root, str) else list(root)
        if not roots:
            raise ValueError("`roots` must contain at least one dataset path.")
        self._all_shard_roots = list(roots)

        # T+1 action rows: frame-0 anchor pose + T target poses. Both the pose
        # trajectory and the pince (jaw) columns come from `action`;
        # observation.state (rig flags) is never requested.
        # Video on the VIDEO rate; the pose+pince SOURCE (`action`, T+1) on the ACTION
        # rate. Defaults make action_* == video rate (unchanged for single-fps runs).
        self._delta_timestamps = {
            self.VIDEO_KEY: [i * self._dt for i in range(0, self._chunk_length + 1)],
            "action": [i * self._action_dt for i in range(0, self._action_chunk_length + 1)],
        }

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    @staticmethod
    def _arm_abs_poses(action_np: np.ndarray, pos_cols: list[int], quat_cols: list[int]) -> np.ndarray:
        """Build (T+1, 4, 4) absolute poses for one hand from the action columns."""
        xyz = action_np[:, pos_cols]
        quat = action_np[:, quat_cols]
        # Normalize quaternions defensively — raw telemetry can drift.
        norms = np.linalg.norm(quat, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        quat = quat / norms
        return build_abs_pose_from_components(xyz, quat, "quat_xyzw")

    def _compute_relative_action(
        self, sample: dict[str, Any], pose_convention: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-hand relative pose + pince for any backward_* convention.

        Returns:
            action: (T, 20) tensor laid out as
                ``[left_pos(3), left_rot6d(6), left_pince(1),
                   right_pos(3), right_rot6d(6), right_pince(1)]``.
            initial_pose: (4, 4) LEFT hand absolute pose at t=0 (used downstream
                for reconstructing the trajectory during eval / visualization).
        """
        action_np = sample["action"].numpy().astype(np.float32)  # (T+1, 26)
        T = action_np.shape[0] - 1  # number of action steps

        left_abs = self._arm_abs_poses(action_np, _L_POS_COLS, _L_QUAT_COLS)
        right_abs = self._arm_abs_poses(action_np, _R_POS_COLS, _R_QUAT_COLS)

        left_rel = pose_abs_to_rel(
            poses_abs=left_abs, rotation_format="rot6d", pose_convention=pose_convention
        )  # (T, 9)
        right_rel = pose_abs_to_rel(
            poses_abs=right_abs, rotation_format="rot6d", pose_convention=pose_convention
        )  # (T, 9)

        # --- Clutch gating (parity with VersiusSuturingLeRobotDataset) ---
        # When the clutch is pressed (col 14 left / 15 right == 1) at frame i, the
        # controller is repositioning, not commanding surgical motion: zero the
        # position delta and set rotation to identity rot6d so the model never learns
        # to imitate repositioning. Clutch at frame i gates the action i -> i+1.
        left_mask = (action_np[:T, _L_CLUTCH_COL] > 0.5)[:, np.newaxis]   # (T, 1)
        right_mask = (action_np[:T, _R_CLUTCH_COL] > 0.5)[:, np.newaxis]  # (T, 1)
        if left_mask.any():
            left_rel[:, :3] = np.where(left_mask, 0.0, left_rel[:, :3])
            left_rel[:, 3:9] = np.where(left_mask, np.broadcast_to(_ROT6D_IDENTITY, (T, 6)), left_rel[:, 3:9])
        if right_mask.any():
            right_rel[:, :3] = np.where(right_mask, 0.0, right_rel[:, :3])
            right_rel[:, 3:9] = np.where(right_mask, np.broadcast_to(_ROT6D_IDENTITY, (T, 6)), right_rel[:, 3:9])
        # --- End clutch gating ---

        # Pince setpoints for frames 0..T-1 — step-start pairing, mirroring the
        # dVRK datasets (jaw at frame i pairs with the pose delta into frame i+1).
        pince = sample["action"][:-1]  # (T, 26) tensor
        left_pince = pince[:, [_L_PINCE_COL]].float()
        right_pince = pince[:, [_R_PINCE_COL]].float()

        left_block = torch.cat([torch.from_numpy(left_rel).float(), left_pince], dim=-1)  # (T, 10)
        right_block = torch.cat([torch.from_numpy(right_rel).float(), right_pince], dim=-1)  # (T, 10)

        full_action = torch.cat([left_block, right_block], dim=-1)  # (T, 20)
        initial_pose = torch.from_numpy(left_abs[0].copy()).float()
        return full_action, initial_pose

    def _compute_absolute_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-hand absolute pose + pince (skips the first/anchor frame)."""
        action_np = sample["action"].numpy().astype(np.float32)[1:]  # (T, 26)

        left_abs = self._arm_abs_poses(action_np, _L_POS_COLS, _L_QUAT_COLS)
        right_abs = self._arm_abs_poses(action_np, _R_POS_COLS, _R_QUAT_COLS)

        def _flatten(poses_abs: np.ndarray) -> torch.Tensor:
            translation = torch.from_numpy(poses_abs[:, :3, 3]).float()
            rot_matrix = torch.from_numpy(poses_abs[:, :3, :3]).float()
            rot6d = convert_rotation(rot_matrix, input_format="matrix", output_format="rot6d").float()
            return torch.cat([translation, rot6d], dim=-1)  # (T, 9)

        pince = sample["action"][:-1]  # (T, 26)
        left_pince = pince[:, [_L_PINCE_COL]].float()
        right_pince = pince[:, [_R_PINCE_COL]].float()

        full_action = torch.cat(
            [_flatten(left_abs), left_pince, _flatten(right_abs), right_pince], dim=-1
        )  # (T, 20)
        initial_pose = torch.from_numpy(left_abs[0].copy()).float()
        return full_action, initial_pose

    # ------------------------------------------------------------------
    # Video letterbox (cosmos2 canvas)
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox(video: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        """Aspect-preserving resize + symmetric black pad to (out_h, out_w).

        Replicates cosmos_predict2's ``resize_with_padding``. Input/return
        layout is ``(T, C, H, W)`` float.
        """
        if video.ndim != 4:
            raise ValueError(f"_letterbox expects (T, C, H, W), got shape {tuple(video.shape)}")
        _, _, h, w = video.shape
        scale = min(out_w / w, out_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
        canvas = video.new_zeros((video.shape[0], video.shape[1], out_h, out_w))
        top = (out_h - new_h) // 2
        left = (out_w - new_w) // 2
        canvas[:, :, top : top + new_h, left : left + new_w] = resized
        return canvas

    def _convert_video(self, video_tchw: torch.Tensor | None) -> torch.Tensor | None:
        """Letterbox to the cosmos2 canvas (if configured) before base conversion."""
        if (
            self._letterbox_hw is not None
            and video_tchw is not None
            and not self._skip_video_loading
        ):
            video_tchw = self._letterbox(video_tchw, self._letterbox_hw[0], self._letterbox_hw[1])
        return super()._convert_video(video_tchw)

    # ------------------------------------------------------------------
    # Action spec / __getitem__
    # ------------------------------------------------------------------

    def _extract_caption(self, sample: dict) -> str:
        text = sample["task"]  # "Needle pickup and handover"
        return f"{self._caption_prefix} {text}" if self._caption_prefix else text

    def _build_action_spec(self) -> ActionSpec:
        """Dual hand-controller 20D: ``[Pos, Rot6d, Gripper] × 2``."""
        return build_action_spec(
            Pos(), Rot("rot6d"), Gripper(),
            Pos(), Rot("rot6d"), Gripper(),
        )

    def _normalizer_filename(self) -> str:
        """Override the bundled-stats filename via the optional ctor arg.

        The bundled stats are horizon-dependent (anchored displacement range
        grows with chunk length), so configs point at an explicit
        ``needle_pickup_..._c<chunk>.json`` instead of the default scheme.
        """
        if self._normalizer_filename_override:
            return self._normalizer_filename_override
        return super()._normalizer_filename()

    @property
    def action_dim(self) -> int:
        return 20

    def _versius_idle_frames(self, sample: dict[str, Any]) -> torch.Tensor | None:
        """Idle-frame count for the anchored/absolute conventions.

        The shared ``_compute_idle_frames`` only supports ``backward_framewise``
        (its metric needs per-step deltas), so for the anchored conventions the
        worldtrans configs use it returns None. Here we recompute the action under
        ``backward_framewise`` (which also applies clutch gating, so clutched
        frames read as idle) purely for idle detection and run the shared detector
        on it, reusing the declared ActionSpec + ``idle_eps_*`` thresholds. Returns
        None (=> no idle filtering) if the convention is already framewise or
        anything is unavailable.
        """
        if self._pose_convention == "backward_framewise":
            return None  # base _compute_idle_frames handles it natively
        spec = self.action_spec
        if spec is None:
            return None
        try:
            fw_action, _ = self._compute_relative_action(sample, "backward_framewise")
        except Exception:
            return None
        eps_t, eps_r, eps_g, joint_thr = self._resolve_idle_thresholds()
        try:
            n = compute_idle_frames(
                fw_action, spec, eps_t=eps_t, eps_r=eps_r, eps_g=eps_g,
                joint_threshold=joint_thr, min_streak=self.idle_min_streak,
            )
        except (ValueError, TypeError):
            return None
        return torch.tensor(n, dtype=torch.long)

    def __getitem__(self, idx: int, _idle_depth: int = 0) -> dict[str, Any]:
        # Skip (and log) tolerance-violating samples instead of crashing the
        # DataLoader worker. No curated known-bad list exists for this dataset
        # yet; the runtime catch is the only path.
        attempts = 0
        cur_idx = idx
        # Seed from original idx so joint-eval calls (FDM/IDM/policy) for the
        # same sample_id pick identical replacement windows, preventing episode
        # misalignment across modes when tolerance or bad-video causes a jump.
        jump_rng = random.Random(idx)
        while attempts < 256:
            try:
                mode, _, _, sample = self._fetch_sample(cur_idx)
                break
            except AssertionError as e:
                if "tolerance" not in str(e):
                    raise
                log.warning(
                    f"VersiusNeedlePickupLeRobotDataset: tolerance-failed sample idx={cur_idx}: "
                    f"{str(e)[:200]}"
                )
                attempts += 1
                cur_idx = jump_rng.randrange(len(self))
            except (FileNotFoundError, OSError, IndexError, EOFError, StopIteration,
                    RuntimeError, ValueError) as e:
                # Out-of-range video frame (e.g. a chunk near an episode's last frame),
                # missing/truncated/undecodable video, etc. Skip instead of letting it
                # crash the DataLoader worker -> rank -> NCCL timeout. Random resample
                # (not +1) escapes any contiguous bad region (e.g. episode-tail windows).
                log.warning(
                    f"VersiusNeedlePickupLeRobotDataset: unreadable/out-of-range sample "
                    f"idx={cur_idx}: {type(e).__name__}: {str(e)[:160]}"
                )
                attempts += 1
                cur_idx = jump_rng.randrange(len(self))
        else:
            raise RuntimeError(
                f"VersiusNeedlePickupLeRobotDataset: 256 consecutive samples failed "
                f"tolerance check starting from idx={idx}."
            )

        ai_caption = self._extract_caption(sample)
        video = sample[self.VIDEO_KEY]

        if self._pose_convention in ("backward_framewise", "backward_anchored", "backward_anchored_world_trans"):
            action, initial_pose = self._compute_relative_action(sample, self._pose_convention)
        elif self._pose_convention == "absolute":
            action, initial_pose = self._compute_absolute_action(sample)
        else:
            raise ValueError(f"Unknown pose_convention: {self._pose_convention}")

        _idle = self._versius_idle_frames(sample)
        result = self._build_result(
            mode=mode, video=video, action=action, ai_caption=ai_caption,
            initial_pose=initial_pose,
            **({"idle_frames": _idle} if _idle is not None else {}),
        )
        return self._reject_if_idle(idx, result, _idle_depth)
