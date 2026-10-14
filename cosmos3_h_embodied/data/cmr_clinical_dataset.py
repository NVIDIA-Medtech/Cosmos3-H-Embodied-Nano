# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CMR Versius **clinical** action dataset (Open-H ``cmr_surgical``).

Distinct from ``versius_needle_pickup``/``versius_suturing`` (benchtop Versius, LeRobot
v3.0, INTERLEAVED 26-D action). The clinical Open-H release is a **BLOCKED**
100-D schema (verified against ``meta/modality.json`` + the parquet):

``action`` (100-D, first 26 named), per hand blocked with 13 cols/hand::

    L: x,y,z(0-2)  quat_xyzw(3-6)  clutchBtn(7) energyBtn(8) thumbBtn(9)
       pince(10)   thumbstick_xy(11-12)
    R: same, offset +13  -> pose(13-19), pince(23)

``observation.state`` (100-D, first 24 named) carries rig flags used here::

    translation_scaling(12)  rotation_scaling(13)
    hapticengaged_l(16) hapticengaged_r(17)   # clutch engagement (1=engaged)
    armlinkedtohaptic_l(20) armlinkedtohaptic_r(21)  # which arm each hand drives

Emitted action is **20-D**, matching the dVRK/suturebot/Versius convention so
checkpoints transfer::

    [left_pos(3), left_rot6d(6), left_pince(1), right_pos(3), right_rot6d(6), right_pince(1)]

with ``pose_convention="backward_anchored_world_trans"`` — world-frame translation
``t_{i+1} - t0`` + anchor-relative rot6d ``R0^T R_{i+1}``. This is EXACTLY GR00T's
``REL_XYZ_ROT6D`` (verified: ``gr00t/data/state_action/pose.py::convert_to_rel_xyz_rot6d``).

Clinical-specific processing (GR00T-faithful):
  * **Motion scaling** (hand->instrument space): scale the per-window rel vector by
    ``translation_scaling``/``rotation_scaling`` read at the window anchor frame
    (matches ``gr00t ... apply_motion_scaling_to_rel_xyz_rot6d``; scalar per window).
  * **Clutch gating**: per target frame where ``hapticengaged==0`` for an arm,
    zero that arm's pos delta and set rot6d to identity ([1,0,0,0,1,0]); pince held.
  * **Window rejection**: drop windows that span an ``armlinkedtohaptic`` change
    (arm/instrument swap) or in which both arms are fully disengaged. Rejection
    reuses the base ``__getitem__`` tolerance-retry (raise ``AssertionError`` whose
    message contains ``"tolerance"`` -> the loop advances to the next index).

  * **Prompt**: clip-level from ``meta/episodes.*_vlm_qwen3.jsonl`` ``weak_labels``
    -> ``"do a <procedure>. phase: <phase>. step: <step>"``.
  * **Video**: single ``observation.images.endoscope`` view.

Phase 1 targets a single procedure root (cholecystectomy). Multi-root prompt
attribution (per-source ``weak_labels``) is a Phase-2 item; with multiple roots
the weak-label map merges by ``episode_index`` (last-wins) — fine for single root.

Latent-cache / augmentation hybrid
----------------------------------
The latent cache and pixel augmentation are **mutually exclusive on one entry**: a
cached VAE latent freezes one fixed augmentation, defeating the point of aug. And a
single batch cannot mix a cached latent ``[C, T/4, H/16, W/16]`` with a raw video
``[C, T, H, W]`` (shape mismatch at collate). So the proposed "0.5 cached / 0.5
live-aug" hybrid is realized as **two dataset entries over the same roots** — one
with ``latent_cache_root=...`` (no aug), one with ``augment=True`` (no cache) —
blended by ``ratio`` in the joint dataloader, so each *entry* still yields uniform
batches. ``augment`` is passed through per-entry; see ``ActionVideoAugment``
(``video_aug.py``) for the temporally-consistent, action-safe augmentation.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.utils import log
from cosmos3_h_embodied.data.domain_utils import cmr_caption_prefix
from cosmos3_h_embodied.data.pose_utils import (
    convert_rotation,
    pose_abs_to_rel,
)
from cosmos3_h_embodied.data.versius_needle_pickup_dataset import (
    VersiusNeedlePickupLeRobotDataset,
)
from cosmos3_h_embodied.data.video_aug import (
    ActionVideoAugment,
    ActionVideoAugmentConfig,
)

# --- BLOCKED clinical action column layout (13 cols/hand) ---
_L_POS_COLS = [0, 1, 2]
_L_QUAT_COLS = [3, 4, 5, 6]  # (qx, qy, qz, qw)
_L_PINCE_COL = 10
_R_POS_COLS = [13, 14, 15]
_R_QUAT_COLS = [16, 17, 18, 19]
_R_PINCE_COL = 23

# --- observation.state indices ---
_ST_TRANS_SCALE = 12
_ST_ROT_SCALE = 13
_ST_HAPTIC_L = 16
_ST_HAPTIC_R = 17
_ST_ARMLINK_L = 20
_ST_ARMLINK_R = 21

# Handgrip thumbsticks (action cols) that pan the Versius endoscope. Raw units: ~1200-1800
# baseline, spikes to ~19000+ during active camera panning.
_L_THUMBSTICK_COLS = [11, 12]
_R_THUMBSTICK_COLS = [24, 25]

_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_CLUTCH_POLICY_CHOICES = ("legacy_mask", "reject_any")


def _stable_root_key(root: str | Path) -> str:
    parts = Path(root).resolve().parts
    return "/".join(parts[-2:])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_valid_window_runs(
    path: str | Path,
    *,
    expected_sha256: str | None,
    chunk_length: int,
    sample_stride: int,
    camera_motion_threshold: float | None,
    clutch_policy: str = "legacy_mask",
) -> tuple[
    dict[str, dict[int, list[tuple[int, int]]]],
    dict[str, tuple[int, int]],
]:
    """Load and validate the CMR action-validity sidecar (``tools/data_prep/build_cmr_valid_window_index.py``).

    ``expected_sha256`` optionally pins the file. Unpinned, the geometry and policy
    fields and each root's row/episode totals are still checked against the loader.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CMR valid-window index missing: {source}")
    actual_sha256 = _sha256_file(source) if expected_sha256 else None
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "CMR valid-window index checksum mismatch: "
            f"expected={expected_sha256} actual={actual_sha256} path={source}"
        )

    with np.load(source, allow_pickle=False) as data:
        schema_version = int(data["schema_version"].item())
        indexed_chunk = int(data["chunk_length"].item())
        indexed_stride = int(data["sample_stride"].item())
        indexed_threshold = float(data["camera_motion_threshold"].item())
        if schema_version not in (1, 2):
            raise RuntimeError(f"unsupported CMR valid-window schema {schema_version}")
        indexed_clutch_policy = (
            str(data["clutch_policy"].item()) if schema_version >= 2 else "legacy_mask"
        )
        if indexed_clutch_policy != clutch_policy:
            raise RuntimeError(
                "CMR valid-window clutch policy mismatch: "
                f"index={indexed_clutch_policy!r} loader={clutch_policy!r}"
            )
        if indexed_chunk != int(chunk_length) or indexed_stride != int(sample_stride):
            raise RuntimeError(
                "CMR valid-window geometry mismatch: "
                f"index=c{indexed_chunk}/stride{indexed_stride} "
                f"loader=c{chunk_length}/stride{sample_stride}"
            )
        expected_threshold = (
            float("nan") if camera_motion_threshold is None else float(camera_motion_threshold)
        )
        thresholds_match = (
            np.isnan(indexed_threshold) and np.isnan(expected_threshold)
        ) or indexed_threshold == expected_threshold
        if not thresholds_match:
            raise RuntimeError(
                "CMR valid-window camera threshold mismatch: "
                f"index={indexed_threshold} loader={expected_threshold}"
            )

        root_keys = [str(value) for value in data["root_keys"].tolist()]
        root_rows = data["root_total_rows"].astype(np.int64, copy=False)
        root_episodes = data["root_total_episodes"].astype(np.int64, copy=False)
        if len(set(root_keys)) != len(root_keys):
            raise RuntimeError("CMR valid-window index contains duplicate root keys")
        roots_meta = {
            key: (int(root_rows[i]), int(root_episodes[i]))
            for i, key in enumerate(root_keys)
        }

        run_roots = data["run_root"].astype(np.int64, copy=False)
        run_episodes = data["run_episode"].astype(np.int64, copy=False)
        run_starts = data["run_start"].astype(np.int64, copy=False)
        run_lengths = data["run_length"].astype(np.int64, copy=False)
        sizes = {len(run_roots), len(run_episodes), len(run_starts), len(run_lengths)}
        if sizes != {len(run_roots)}:
            raise RuntimeError("CMR valid-window run arrays have inconsistent lengths")

        runs: dict[str, dict[int, list[tuple[int, int]]]] = {
            key: {} for key in root_keys
        }
        for root_idx, episode, start, length in zip(
            run_roots, run_episodes, run_starts, run_lengths
        ):
            if not 0 <= root_idx < len(root_keys) or start < 0 or length <= 0:
                raise RuntimeError("CMR valid-window index contains an invalid run")
            runs[root_keys[root_idx]].setdefault(int(episode), []).append(
                (int(start), int(length))
            )
    return runs, roots_meta


def camera_is_moving(
    action_np: np.ndarray,
    threshold: float,
    l_cols: list[int] = _L_THUMBSTICK_COLS,
    r_cols: list[int] = _R_THUMBSTICK_COLS,
) -> bool:
    """True if the endoscope joystick is deflected past ``threshold`` on any window frame.

    The Versius camera is panned by the handgrip thumbsticks, so a deflection means the
    scene moves independent of the instrument action — corrupting the FDM/video target.
    This is NOT part of GR00T's CMR filter (GR00T only handles arm-swap + full disengage);
    it's an added filter for camera-motion windows.
    """
    a = np.asarray(action_np, dtype=np.float32)
    lmag = np.hypot(a[:, l_cols[0]], a[:, l_cols[1]])
    rmag = np.hypot(a[:, r_cols[0]], a[:, r_cols[1]])
    return bool((np.maximum(lmag, rmag) > threshold).any())


def _scale_rot6d_by_angle(rot6d_np: np.ndarray, scaling: float) -> np.ndarray:
    """Scale each rotation's angle by ``scaling`` (axis-angle), preserving axis.

    Mirrors GR00T ``scale_rot6d_by_angle``: rot6d -> matrix -> axis-angle,
    multiply the angle by ``scaling``, back to rot6d.
    """
    if abs(float(scaling) - 1.0) < 1e-9:
        return rot6d_np
    from scipy.spatial.transform import Rotation

    R = convert_rotation(
        torch.from_numpy(np.ascontiguousarray(rot6d_np)).float(),
        input_format="rot6d",
        output_format="matrix",
    ).numpy()  # (T, 3, 3)
    rotvec = Rotation.from_matrix(R).as_rotvec() * float(scaling)
    R2 = Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)  # (T, 3, 3)
    out = convert_rotation(
        torch.from_numpy(R2), input_format="matrix", output_format="rot6d"
    ).numpy()
    return out.astype(np.float32)


def instruction_for_frame(segments: list[list] | None, frame_index: int) -> str | None:
    """Instruction text for ``frame_index`` from piecewise-constant ``segments``.

    ``segments`` is ``[[start_frame, text], ...]`` sorted by ``start_frame`` (as
    stored in ``meta/instruction_segments.json``). Returns the text of the last segment whose
    ``start_frame <= frame_index``. Returns ``None`` for empty/missing segments so
    the caller can fall back. Defensive: a frame before the first start uses the
    first segment (starts should begin at 0, but never index-error).
    """
    if not segments:
        return None
    starts = [s[0] for s in segments]
    i = bisect.bisect_right(starts, frame_index) - 1
    if i < 0:
        i = 0
    return segments[i][1]


class CmrClinicalLeRobotDataset(VersiusNeedlePickupLeRobotDataset):
    """Action dataset for the Open-H CMR Versius *clinical* recordings."""

    VIDEO_KEY = "observation.images.endoscope"
    # Correct per-frame caption column (grafted from the source recordings). Declared
    # scalar-string in info.json, so LeRobot returns it directly at the anchor frame.
    INSTR_KEY = "instruction.text_with_state"

    def __init__(
        self,
        root: str | list[str] | None = None,
        roots: list[str] | None = None,
        video_key: str | None = None,
        prompt_source: str = "task",
        augment: dict | bool | None = None,
        camera_motion_threshold: float | None = 5000.0,
        clutch_policy: str = "legacy_mask",
        valid_window_index_path: str | None = None,
        valid_window_index_sha256: str | None = None,
        caption_prefix_mode: str = "fixed",
        **kwargs: Any,
    ) -> None:
        # Clinical prompts come from weak_labels, not the (UUID) task column, but
        # the base needle-pickup __init__ hard-requires prompt_source=="task".
        # Pass "task" through and override _extract_caption below.
        if prompt_source != "task":
            raise ValueError(
                f"CmrClinicalLeRobotDataset prompt_source must be 'task' (clinical builds "
                f"the caption from weak_labels internally), got {prompt_source!r}"
            )
        if clutch_policy not in _CLUTCH_POLICY_CHOICES:
            raise ValueError(
                f"clutch_policy must be one of {_CLUTCH_POLICY_CHOICES}, got {clutch_policy!r}"
            )
        self._clutch_policy = clutch_policy
        self._set_caption_prefix_mode(caption_prefix_mode)
        super().__init__(
            root=root,
            roots=roots,
            video_key=video_key if video_key is not None else self.VIDEO_KEY,
            prompt_source="task",
            **kwargs,
        )
        # Reject windows where the handgrip joystick pans the endoscope (camera motion
        # corrupts the FDM/video target). None disables. Beyond GR00T (arm-swap+disengage).
        self._camera_motion_threshold = camera_motion_threshold
        # Clinical derives BOTH the pose trajectory and the clutch/scaling signals
        # from the action + state columns, so observation.state must be loaded too.
        self._delta_timestamps = {
            self.VIDEO_KEY: [i * self._dt for i in range(0, self._chunk_length + 1)],
            "action": [i * self._dt for i in range(0, self._chunk_length + 1)],
            "observation.state": [i * self._dt for i in range(0, self._chunk_length + 1)],
        }
        self._valid_window_runs: dict[str, dict[int, list[tuple[int, int]]]] | None = None
        self._valid_window_roots_meta: dict[str, tuple[int, int]] = {}
        if valid_window_index_sha256 and not valid_window_index_path:
            raise ValueError("valid_window_index_sha256 requires valid_window_index_path")
        if valid_window_index_path:
            self._valid_window_runs, self._valid_window_roots_meta = load_valid_window_runs(
                valid_window_index_path,
                expected_sha256=str(valid_window_index_sha256) if valid_window_index_sha256 else None,
                chunk_length=self._chunk_length,
                sample_stride=self._sample_stride,
                camera_motion_threshold=self._camera_motion_threshold,
                clutch_policy=self._clutch_policy,
            )
            log.info(
                f"CmrClinicalLeRobotDataset: valid-window index "
                f"{valid_window_index_path} ({len(self._valid_window_runs)} roots)"
            )
        # Per-FRAME instruction labels: {episode_index: [[start_frame, text], ...]}.
        # Grafted from the source recordings (arm/instrument-state + task + phase
        # text); not part of the public Open-H release. Replaces the old (WRONG) qwen VLM
        # weak_labels phase/step captions from episodes.*_vlm_qwen3.jsonl.
        (
            self._procedure,
            self._instr_segments,
            self._procedure_by_idx,
            self._instr_segments_by_idx,
        ) = self._load_instruction_segments(self._all_shard_roots)
        # Procedure-aware caption context (robot + setting + procedure + instruments),
        # prepended to the grafted per-frame instruction.text_with_state label. CMR
        # shares one embodiment so this can't come from the embodiment table; build it
        # PER-ROOT from _procedure_by_idx so a 4-procedure entry doesn't label every
        # sample with the last root. Respect an explicit caption_prefix (e.g. multiview).
        _explicit_prefix = getattr(self, "_caption_prefix", None)
        # An explicit caption_prefix wins even when EMPTY. caption_prefix="" means "no
        # prefix — the caption is the parquet task/instruction text alone", which is how
        # some earlier checkpoints were trained. A truthiness test alone
        # cannot express that and would silently re-add the procedure/instrument prefix,
        # feeding that checkpoint a prompt it never saw.
        if getattr(self, "_explicit_caption_prefix", False) and not _explicit_prefix:
            self._caption_prefix_by_idx = {i: "" for i in self._procedure_by_idx}
            self._caption_prefix = ""
        elif _explicit_prefix:
            self._caption_prefix_by_idx = {i: _explicit_prefix for i in self._procedure_by_idx}
            self._caption_prefix = _explicit_prefix
        else:
            self._caption_prefix_by_idx = {
                i: cmr_caption_prefix(p) for i, p in self._procedure_by_idx.items()
            }
            self._caption_prefix = self._caption_prefix_by_idx.get(0, cmr_caption_prefix(self._procedure))

        # Optional temporally-consistent, action-safe pixel augmentation (live path
        # only — mutually exclusive with the latent cache, see the class docstring's
        # hybrid note). ``augment`` may be a bool, a dict of overrides, or None.
        if augment is None or augment is False:
            self._aug: ActionVideoAugment | None = None
        elif augment is True:
            self._aug = ActionVideoAugment(ActionVideoAugmentConfig(enabled=True))
        elif isinstance(augment, dict):
            cfg = {"enabled": True, **augment}
            self._aug = ActionVideoAugment(ActionVideoAugmentConfig(**cfg))
        else:
            self._aug = ActionVideoAugment(augment)
        if self._aug is not None and self._latent_cache_active:
            raise ValueError(
                "CmrClinicalLeRobotDataset: `augment` and the latent cache are mutually "
                "exclusive on one dataset entry — a cached latent freezes one fixed "
                "augmentation. Use TWO dataset entries over the same roots (one cache-mode, "
                "one augment=True) blended by ratio for the hybrid; see the class docstring."
            )

    def _filter_episode_spans(
        self,
        meta: Any,
        episode_spans: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Replace raw candidate spans with action-valid contiguous runs."""
        if self._valid_window_runs is None:
            return episode_spans

        root_key = _stable_root_key(meta.root)
        if root_key not in self._valid_window_runs:
            raise RuntimeError(f"CMR valid-window index has no entry for {root_key!r}")
        expected_rows, expected_episodes = self._valid_window_roots_meta[root_key]
        actual_rows = int(meta.info.get("total_frames", -1))
        actual_episodes = int(meta.total_episodes)
        if (actual_rows, actual_episodes) != (expected_rows, expected_episodes):
            raise RuntimeError(
                f"CMR valid-window source changed for {root_key}: "
                f"index rows/episodes={(expected_rows, expected_episodes)} "
                f"actual={(actual_rows, actual_episodes)}"
            )

        runs_by_episode = self._valid_window_runs[root_key]
        filtered: list[tuple[int, int, int, int]] = []
        raw_count = sum(span[2] for span in episode_spans)
        for episode_id, sample_start, valid_len, _ in episode_spans:
            for run_start, run_length in runs_by_episode.get(int(episode_id), ()):
                run_stop = min(valid_len, run_start + run_length)
                if run_start >= valid_len or run_stop <= run_start:
                    continue
                filtered.append(
                    (
                        episode_id,
                        sample_start + run_start * self._sample_stride,
                        run_stop - run_start,
                        run_start,
                    )
                )
        log.info(
            f"CmrClinicalLeRobotDataset [{root_key}]: valid-window index kept "
            f"{sum(span[2] for span in filtered)} / {raw_count} candidate starts"
        )
        return filtered

    # ------------------------------------------------------------------
    # Per-frame instruction prompts (grafted instruction.text_with_state)
    # ------------------------------------------------------------------

    @staticmethod
    def _procedure_from_root(root: str) -> str:
        """Procedure name from a shard-root dir (e.g. ``.../cholecystectomy`` -> ``cholecystectomy``)."""
        name = os.path.basename(os.path.normpath(root))
        for suf in ("_v30_multiview", "_v30", "_v21", "_multiview"):
            if name.endswith(suf):
                name = name[: -len(suf)]
        return name.replace("_", " ") or "surgical procedure"

    @staticmethod
    def _procedures_by_idx(roots: list[str]) -> dict[int, str]:
        """Map each shard-root index -> its own procedure (no last-root-wins collapse)."""
        return {i: CmrClinicalLeRobotDataset._procedure_from_root(r) for i, r in enumerate(roots)}

    @staticmethod
    def _load_instruction_segments(
        roots: list[str],
    ) -> tuple[str, dict[int, list], dict[int, str], dict[int, dict[int, list]]]:
        """Load per-episode instruction segments from each root's meta/instruction_segments.json.

        The JSON is ``{episode_index: [[start_frame, text], ...]}`` (piecewise-constant
        runs of ``instruction.text_with_state``). The public Open-H release does not
        include these labels.

        Returns ``(procedure0, merged_segments, procedures_by_idx, segments_by_idx)``.
        With MULTIPLE roots (4-procedure CMR entry) the merged map collides across roots
        by ``episode_index`` (last-wins), so callers that know the sample's source root
        must use ``segments_by_idx``/``procedures_by_idx`` keyed by ``dataset_idx``. The
        merged map is retained only for the single-root legacy fallback.
        """
        merged: dict[int, list] = {}
        procedures_by_idx: dict[int, str] = {}
        segments_by_idx: dict[int, dict[int, list]] = {}
        for i, r in enumerate(roots):
            procedure = CmrClinicalLeRobotDataset._procedure_from_root(r)
            procedures_by_idx[i] = procedure
            seg: dict[int, list] = {}
            path = os.path.join(r, "meta", "instruction_segments.json")
            if not os.path.exists(path):
                log.warning(
                    f"CmrClinical: no instruction_segments.json at {path} — captions "
                    f"fall back to 'do a {procedure}'."
                )
            else:
                try:
                    with open(path) as fh:
                        raw = json.load(fh)
                    for k, v in raw.items():
                        seg[int(k)] = v
                except (OSError, json.JSONDecodeError) as e:
                    log.warning(f"CmrClinical: failed reading {path}: {e}")
            segments_by_idx[i] = seg
            merged.update(seg)  # legacy single-root fallback only
        procedure0 = procedures_by_idx.get(0, "surgical procedure")
        log.info(
            f"CmrClinical: loaded instruction segments from {len(roots)} root(s); "
            f"procedures={list(procedures_by_idx.values())}"
        )
        return procedure0, merged, procedures_by_idx, segments_by_idx

    @staticmethod
    def _as_int(v: Any) -> int | None:
        """Coerce a possibly-tensor/array scalar (episode/frame index) to int."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return int(v.reshape(-1)[0].item())
        if isinstance(v, np.ndarray):
            return int(v.reshape(-1)[0])
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _extract_caption(self, sample: dict, dataset_idx: int | None = None) -> str:
        """Caption = the correct per-frame instruction at the window's anchor frame.

        Prefers the grafted per-frame ``instruction.text_with_state`` column carried
        directly in the sample (declared scalar-string, so LeRobot returns the anchor
        frame's value). Falls back to the ``instruction_segments.json`` sidecar (keyed
        by anchor ``episode_index``/``frame_index``), then to ``do a <procedure>``.

        ``dataset_idx`` (the sample's source shard root, from ``_fetch_sample``) selects
        the PER-ROOT procedure / caption prefix / segment map, so a 4-procedure entry
        labels each sample with its own procedure instead of the last root's.
        """
        caption_idx = self._caption_root_index(dataset_idx)
        if caption_idx is not None:
            procedure = self._procedure_by_idx.get(caption_idx, self._procedure)
            prefix = self._caption_prefix_by_idx.get(caption_idx, self._caption_prefix)
            segments = self._instr_segments_by_idx.get(caption_idx, self._instr_segments)
        else:
            procedure, prefix, segments = self._procedure, self._caption_prefix, self._instr_segments
        text = None
        val = sample.get(self.INSTR_KEY)
        if isinstance(val, str):
            text = val
        elif isinstance(val, (list, tuple)) and val and isinstance(val[0], str):
            text = val[0]
        if not text:  # sidecar fallback (robust if the column isn't surfaced)
            ep = self._as_int(sample.get("episode_index"))
            fr = self._as_int(sample.get("frame_index"))
            if ep is not None and fr is not None:
                text = instruction_for_frame(segments.get(ep), fr)
        if not text:
            text = f"do a {procedure}"
        return f"{prefix} {text}" if prefix else text

    # ------------------------------------------------------------------
    # Action computation (clinical: blocked cols + motion scaling + clutch)
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_invalid_window(
        state_np: np.ndarray, *, clutch_policy: str = "legacy_mask"
    ) -> None:
        """Raise a tolerance-tagged AssertionError for windows we must skip."""
        al = state_np[:, _ST_ARMLINK_L]
        ar = state_np[:, _ST_ARMLINK_R]
        if np.unique(al).size > 1 or np.unique(ar).size > 1:
            raise AssertionError("tolerance: armlinkedtohaptic change within window (arm swap)")
        hl = state_np[:, _ST_HAPTIC_L]
        hr = state_np[:, _ST_HAPTIC_R]
        if clutch_policy == "reject_any":
            if bool((hl < 0.5).any()) or bool((hr < 0.5).any()):
                raise AssertionError("tolerance: clutch-out in either arm within window")
            return
        if clutch_policy != "legacy_mask":
            raise ValueError(
                f"clutch_policy must be one of {_CLUTCH_POLICY_CHOICES}, got {clutch_policy!r}"
            )
        if bool((hl < 0.5).all()) and bool((hr < 0.5).all()):
            raise AssertionError("tolerance: both arms disengaged across window")

    @staticmethod
    def _apply_motion_scaling(rel: np.ndarray, ts: float, rs: float) -> np.ndarray:
        """hand->instrument space: xyz *= ts, rot6d angle *= rs (GR00T-faithful)."""
        if abs(ts - 1.0) < 1e-9 and abs(rs - 1.0) < 1e-9:
            return rel
        out = rel.copy()
        out[:, 0:3] = rel[:, 0:3] * float(ts)
        out[:, 3:9] = _scale_rot6d_by_angle(rel[:, 3:9], rs)
        return out

    @staticmethod
    def _zero_disengaged(rel: np.ndarray, haptic_targets: np.ndarray) -> np.ndarray:
        """Zero pos delta + identity rot6d on frames where the arm is clutched out."""
        mask = haptic_targets < 0.5  # (T,)
        if not mask.any():
            return rel
        out = rel.copy()
        out[mask, 0:3] = 0.0
        out[mask, 3:9] = _IDENTITY_ROT6D
        return out

    def _compute_relative_action(
        self, sample: dict[str, Any], pose_convention: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_np = sample["action"].numpy().astype(np.float32)  # (T+1, 100)
        state_np = sample["observation.state"].numpy().astype(np.float32)  # (T+1, 100)

        # Skip clinically-invalid windows (arm swap / both disengaged).
        self._reject_invalid_window(state_np, clutch_policy=self._clutch_policy)

        # Skip windows where the surgeon is panning the endoscope with the handgrip
        # joystick — the camera moves independent of the instrument action, corrupting
        # the FDM/video target. (Not in GR00T's CMR filter; added here.)
        if self._camera_motion_threshold is not None and camera_is_moving(
            action_np, self._camera_motion_threshold
        ):
            raise AssertionError("tolerance: endoscope camera panning (joystick) within window")

        left_abs = self._arm_abs_poses(action_np, _L_POS_COLS, _L_QUAT_COLS)
        right_abs = self._arm_abs_poses(action_np, _R_POS_COLS, _R_QUAT_COLS)

        left_rel = pose_abs_to_rel(
            poses_abs=left_abs, rotation_format="rot6d", pose_convention=pose_convention
        )  # (T, 9)
        right_rel = pose_abs_to_rel(
            poses_abs=right_abs, rotation_format="rot6d", pose_convention=pose_convention
        )

        # Motion scaling: scalar read at the window anchor frame (matches GR00T's
        # single-reference-state scaling).
        ts = float(state_np[0, _ST_TRANS_SCALE])
        rs = float(state_np[0, _ST_ROT_SCALE])
        if ts == 0.0:  # guard degenerate telemetry
            ts = 1.0
        left_rel = self._apply_motion_scaling(left_rel, ts, rs)
        right_rel = self._apply_motion_scaling(right_rel, ts, rs)

        # Clutch zeroing per target frame (rel[i] targets frame i+1 -> state[1:]).
        left_rel = self._zero_disengaged(left_rel, state_np[1:, _ST_HAPTIC_L])
        right_rel = self._zero_disengaged(right_rel, state_np[1:, _ST_HAPTIC_R])

        # Pince (jaw) setpoints, step-start pairing (frames 0..T-1).
        pince = sample["action"][:-1]  # (T, 100)
        left_pince = pince[:, [_L_PINCE_COL]].float()
        right_pince = pince[:, [_R_PINCE_COL]].float()

        left_block = torch.cat([torch.from_numpy(left_rel).float(), left_pince], dim=-1)  # (T,10)
        right_block = torch.cat([torch.from_numpy(right_rel).float(), right_pince], dim=-1)

        full_action = torch.cat([left_block, right_block], dim=-1)  # (T, 20)
        initial_pose = torch.from_numpy(left_abs[0].copy()).float()
        return full_action, initial_pose

    def _compute_absolute_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "CmrClinicalLeRobotDataset only supports backward_* (relative) pose "
            "conventions; absolute is not used for clinical mid-training."
        )

    def _convert_video(self, video_tchw: torch.Tensor | None) -> torch.Tensor | None:
        """Apply live augmentation on the raw [0,1] clip, then letterbox/format.

        Fires only on the LIVE path: when the latent cache is active the base never
        decodes video (``video_tchw is None``) and this is a no-op. Input is
        ``(T, C, H, W)`` float in [0, 1]; ``ActionVideoAugment`` wants ``(C, T, H, W)``,
        so we permute around it. Aug runs BEFORE the base resize/normalize so the
        photometric ops operate on natural [0, 1] pixels.
        """
        if (
            self._aug is not None
            and video_tchw is not None
            and not self._skip_video_loading
            and video_tchw.ndim == 4
        ):
            aug_in = video_tchw.permute(1, 0, 2, 3).contiguous()  # (T,C,H,W) -> (C,T,H,W)
            video_tchw = self._aug(aug_in).permute(1, 0, 2, 3).contiguous()
        return super()._convert_video(video_tchw)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retry-on-rejection wrapper covering BOTH fetch and action compute.

        The needle-pickup base's retry loop only guards ``_fetch_sample`` (the
        timestamp-tolerance check). Clinical action computation ALSO rejects
        windows (clutch-out / arm-linkage change) by raising a ``"tolerance"``-
        tagged ``AssertionError`` from ``_compute_relative_action``, so that call
        must be inside the retry loop too — otherwise a rejected window crashes
        the worker instead of advancing to the next index.
        """
        attempts = 0
        cur_idx = idx
        failure_counts: dict[str, int] = {}
        attempted_indices: list[int] = []

        def remember_failure(exc: BaseException) -> None:
            reason = f"{type(exc).__name__}: {str(exc)[:240]}"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
            if len(attempted_indices) < 16:
                attempted_indices.append(int(cur_idx))

        # Deterministic resample: seed the retry-jump RNG by the ORIGINAL idx so the
        # same requested window always resolves to the same episode. Without this, the
        # jump used the global `random` state, which differs between the FD/ID/policy
        # __getitem__ calls for the same sample_id -> the three modes landed on DIFFERENT
        # CMR procedures (multi-root entry), misaligning the eval 3-up. Per-idx seeding
        # also keeps a jumped window identical across modes AND reproducible across runs.
        jump_rng = random.Random(idx)
        while attempts < 256:
            try:
                # Validate action/timestamp/clutch semantics before touching video.
                # Rejected clinical windows are common and may otherwise open several
                # large MP4 files before the deterministic retry finds a valid one.
                mode, dataset_idx, row_idx, sample = self._fetch_sample(
                    cur_idx, decode_video=False
                )
                if self._pose_convention in (
                    "backward_framewise",
                    "backward_anchored",
                    "backward_anchored_world_trans",
                ):
                    action, initial_pose = self._compute_relative_action(sample, self._pose_convention)
                else:
                    action, initial_pose = self._compute_absolute_action(sample)
                ai_caption = self._extract_caption(sample, dataset_idx)
                if getattr(self, "_skip_video_loading", False) or self._latent_cache_active:
                    # None when the latent cache is active; it is injected below.
                    video = sample.get(self.VIDEO_KEY)
                else:
                    video_mode, video_dataset_idx, video_row_idx, video_sample = self._fetch_sample(
                        cur_idx, decode_video=True, mode_override=mode
                    )
                    if (video_mode, video_dataset_idx, video_row_idx) != (
                        mode,
                        dataset_idx,
                        row_idx,
                    ):
                        raise RuntimeError("CMR tabular/video preflight resolved inconsistently")
                    video = video_sample.get(self.VIDEO_KEY)
                result = self._build_result(
                    mode=mode,
                    video=video,
                    action=action,
                    ai_caption=ai_caption,
                    initial_pose=initial_pose,
                )
                if self._latent_cache_active:
                    # The cache is keyed by the index requested from this dataset,
                    # not by the index that the deterministic rejection retry
                    # ultimately resolves to.  Precompute calls ``ds[idx]`` and
                    # writes the returned (possibly retried) clip under ``idx``;
                    # training must therefore read that same key.  Reading
                    # ``cur_idx`` instead silently paired the retried action/caption
                    # with a latent produced for a different request.
                    result = self._inject_cached_latent(result, idx)
                elif "video" in result:
                    # Mark live samples so a HYBRID (mixed cache + live-aug) batch carries
                    # a per-sample vision_is_latent flag for EVERY sample — the model's
                    # per-sample gating (omni_mot_model) reads it to encode raw items and
                    # reuse cached latents in the same batch.
                    result["vision_is_latent"] = torch.tensor(False)
                return result
            except AssertionError as e:
                if "tolerance" not in str(e):
                    raise
                remember_failure(e)
                self._remember_bad_sample(cur_idx, e)  # never resample this window this run
                attempts += 1
                # Retry at a RANDOM index, not idx+1: clinical clutch-out stretches
                # are long and contiguous (multi-second), so sequential advance can
                # exhaust the retry budget inside one bad region. A random jump lands
                # in a valid window with high probability (~85% frames engaged).
                cur_idx = jump_rng.randint(0, len(self) - 1)
            except (FileNotFoundError, OSError, IndexError, EOFError, StopIteration,
                    RuntimeError, ValueError) as e:
                # Out-of-range video frame (a chunk near an episode's last frame, e.g.
                # drybox), missing/undecodable video, etc. Skip instead of crashing the
                # DataLoader worker -> rank -> NCCL timeout. Random resample escapes any
                # contiguous bad region.
                log.warning(
                    f"CmrClinicalLeRobotDataset: unreadable/out-of-range sample "
                    f"idx={cur_idx}: {type(e).__name__}: {str(e)[:160]}"
                )
                remember_failure(e)
                self._remember_bad_sample(cur_idx, e)  # never resample this window this run
                attempts += 1
                cur_idx = jump_rng.randint(0, len(self) - 1)
        ranked_failures = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
        raise RuntimeError(
            f"CmrClinicalLeRobotDataset: {attempts} retries all failed the "
            f"tolerance / clutch-rejection check starting from idx={idx}; "
            f"failure_counts={ranked_failures}; first_attempted_indices={attempted_indices}."
        )
