# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared LeRobot adapter utilities for Action datasets.

These helpers centralize common behavior across Action wrappers:
- deterministic train/val episode splitting
- valid per-episode index range construction
- a reusable BaseActionLeRobotDataset class with lazy init, video formatting,
  and common result building
"""

from __future__ import annotations

import importlib
import json
import logging as _logging
import math
import os as _os
import random
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Literal

import huggingface_hub.constants as _hf_const
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_delta_indices
from torch.utils.data import Dataset

_hf_offline_applied = False

# Failures caused by shared filesystems, cache locking, or service interruption
# are not properties of a dataset row. Remembering them as bad poisons otherwise
# valid rows across workers/runs (and can eventually blacklist a whole family).
_TRANSIENT_SAMPLE_ERRNOS = {
    4,  # EINTR
    11,  # EAGAIN / resource temporarily unavailable
    16,  # EBUSY
    19,  # ENODEV (observed from a temporarily unavailable HF cache mount)
    110,  # ETIMEDOUT
    116,  # ESTALE
}


def is_persistent_sample_failure(
    exc: BaseException | None = None,
    *,
    exc_type: str | None = None,
    message: str | None = None,
) -> bool:
    """Return whether a failure is evidence that the sample itself is bad.

    ``exc_type``/``message`` support classification of legacy JSONL records.
    Records predating failure metadata are retained for backward compatibility.
    """
    if exc is not None:
        exc_type = type(exc).__name__
        message = str(exc)
        errno_value = getattr(exc, "errno", None)
    else:
        errno_value = None

    if exc_type is None and message is None:
        return True

    kind = str(exc_type or "")
    lower = str(message or "").lower()
    if kind in {"BlockingIOError", "TimeoutError", "ConnectionError", "InterruptedError"}:
        return False
    if errno_value in _TRANSIENT_SAMPLE_ERRNOS:
        return False
    if kind == "OSError" and any(
        marker in lower
        for marker in (
            "[errno 4]",
            "[errno 11]",
            "[errno 16]",
            "[errno 19]",
            "[errno 110]",
            "[errno 116]",
            "resource temporarily unavailable",
            "stale file handle",
            "timed out",
        )
    ):
        return False

    # These are retry decisions for one invocation, not globally bad rows.
    if "tolerance: minimal-action window (policy mode)" in lower:
        return False
    if "tolerance: session-remembered-bad sample" in lower:
        return False

    return True


def _ensure_hf_hub_offline() -> None:
    """Force HF Hub into offline mode for local-only datasets (repo_id="local").

    Sets the ``HF_HUB_OFFLINE`` env var (for any future imports in worker
    processes), patches the already-imported constant, and suppresses the
    expected "Returning existing local_dir" fallback warning.

    Safe to call multiple times; only applies once per process.
    """
    global _hf_offline_applied
    if _hf_offline_applied:
        return
    if "HF_HUB_OFFLINE" not in _os.environ:
        _os.environ["HF_HUB_OFFLINE"] = "1"
    if not _hf_const.HF_HUB_OFFLINE:
        _hf_const.HF_HUB_OFFLINE = True
    _logging.getLogger("huggingface_hub._snapshot_download").setLevel(_logging.ERROR)
    _hf_offline_applied = True


from functools import cached_property

from cosmos_framework.utils import log
from cosmos3_h_embodied.data.action_normalization import (
    load_action_stats,
    normalize_action,
)

# Re-export the action_spec DSL from this module so that subclass datasets
# only need a single import block (alongside ``BaseActionLeRobotDataset``).
from cosmos_framework.data.generator.action.utils.action_spec import (  # noqa: F401  (re-export)
    ActionSpec,
    DimType,
    Gripper,
    Joint,
    Pos,
    Reserved,
    Rot,
    build_action_spec,
)
from cosmos3_h_embodied.data.domain_utils import get_domain_id
from cosmos3_h_embodied.data.pose_utils import compute_idle_frames
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint


# Worker memory profiling is not part of the public pipeline; these stand-ins keep the
# (disabled) profiling call sites inert.
def _memprofile_enabled() -> bool:
    return False


def rss_tracker(*_args: Any, **_kwargs: Any) -> Any:
    from contextlib import nullcontext

    return nullcontext()

# ---------------------------------------------------------------------------
# LRU-capped VideoDecoderCache
# ---------------------------------------------------------------------------
# 256 (was 64): with many small surgical shards sampled at random, a small
# cache thrashes — every miss reopens the video file and re-seeks. Node
# RAM (~2 TB) makes hundreds of warm decoders effectively free.
_LRU_VIDEO_CACHE_MAX_SIZE: int = 256
_LRU_DATASET_MAX_LOADED: int = 32
ActionNormalization = Literal["quantile", "quantile_rot", "meanstd", "minmax"]
_ACTION_NORMALIZATION_CHOICES: tuple[str, ...] = ("quantile", "quantile_rot", "meanstd", "minmax")

_decoder_cache_patched = False

CAPTION_PREFIX_MODES: tuple[str, ...] = ("fixed", "legacy")
# cosmos-framework mode names -> this package's mode names.
MODE_ALIASES: dict[str, str] = {"wam": "policy"}


def _create_video_decoder(video_decoder_cls: Any, file_handle: Any) -> Any:
    """Create a CPU TorchCodec decoder (approximate seek mode, as in training)."""
    return video_decoder_cls(file_handle, seek_mode="approximate", device="cpu")


class _LRUVideoDecoderCache:
    """Drop-in replacement for ``lerobot.datasets.video_utils.VideoDecoderCache``
    with LRU eviction.  When the cache exceeds *max_size* entries the
    least-recently-used decoder (and its file handle) is evicted.
    """

    def __init__(self, max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_decoder(self, video_path: str) -> Any:
        if importlib.util.find_spec("torchcodec"):  # type: ignore[attr-defined]
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        import fsspec

        video_path = str(video_path)

        with self._lock:
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
                self._hits += 1
                return self._cache[video_path][0]

            self._misses += 1
            file_handle = fsspec.open(video_path).__enter__()
            try:
                decoder = _create_video_decoder(VideoDecoder, file_handle)
            except Exception:
                file_handle.close()
                raise
            self._cache[video_path] = (decoder, file_handle)

            evicted = 0
            while len(self._cache) > self._max_size:
                _, (_, old_fh) = self._cache.popitem(last=False)
                try:
                    old_fh.close()
                except Exception:
                    pass
                evicted += 1
            self._evictions += evicted

            if evicted and self._evictions % 50 <= evicted:
                log.debug(
                    f"[VideoDecoderCache pid={_os.getpid()}] "
                    f"evicted={self._evictions} total, size={len(self._cache)}/{self._max_size}, "
                    f"hits={self._hits}, misses={self._misses}, "
                    f"hit_rate={100 * self._hits / max(1, self._hits + self._misses):.1f}%"
                )

            return decoder

    def clear(self) -> None:
        with self._lock:
            for _, file_handle in self._cache.values():
                try:
                    file_handle.close()
                except Exception:
                    pass
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


def _patch_decoder_cache(max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
    """Replace the module-level ``_default_decoder_cache`` in LeRobot with an
    LRU-capped version to prevent unbounded memory growth in workers."""
    global _decoder_cache_patched
    if _decoder_cache_patched:
        return

    import lerobot.datasets.video_utils as _vu

    lru_cache = _LRUVideoDecoderCache(max_size=max_size)
    _vu._default_decoder_cache = lru_cache
    _decoder_cache_patched = True
    log.debug(f"Patched LeRobot VideoDecoderCache with LRU max_size={max_size}")


def _parallel_map(
    fn: Callable[[Any], Any],
    items: list[Any],
    *,
    max_workers: int,
    label: str,
) -> list[Any]:
    """Thread-pool ``map`` — returns results in input order.

    Intended for IO-bound prefetch (``LeRobotDatasetMetadata`` loads,
    parquet column reads).  Preserves item-order so callers can ``zip``
    with their ``indices`` / ``roots`` list.  Skips the thread pool
    entirely when there is 0 or 1 task — avoids per-worker
    ``ThreadPoolExecutor`` setup cost and log spam under
    ``shard_across_workers=True`` where each worker typically gets
    only 1-2 shards.
    """
    if not items:
        return []
    if len(items) == 1 or max_workers <= 1:
        return [fn(items[0])] if len(items) == 1 else [fn(x) for x in items]
    log.info(f"{label}: {len(items)} tasks (workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))


def split_episode_ids(total_episodes: int, seed: int, val_ratio: float, split: str) -> list[int]:
    """Create deterministic random episode ids for train/val/full splits."""
    num_val = int(round(total_episodes * val_ratio))
    g = torch.Generator().manual_seed(seed)
    episode_ids = torch.randperm(total_episodes, generator=g).tolist()

    if split == "train":
        return episode_ids[num_val:]
    if split == "val":
        return episode_ids[:num_val]
    return episode_ids


def parse_split_ranges(spec: object) -> set[int]:
    """Episode indices from an Open-H ``"a:b, c:d"`` split range string.

    Ranges are inclusive on both ends (gr00t semantics: ``range(a, b+1)``).
    ``None``/empty -> empty set.
    """
    if not spec:
        return set()
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition(":")
        lo = int(a)
        hi = int(b) if b else lo
        out.update(range(lo, hi + 1))
    return out


def split_episode_set(info: dict | None, names: list[str]) -> set[int]:
    """Union of episode indices over the named splits in ``info['splits']`` (case-insensitive)."""
    splits = (info.get("splits") or {}) if info else {}
    lower = {str(k).lower(): v for k, v in splits.items()}
    out: set[int] = set()
    for name in names:
        out |= parse_split_ranges(lower.get(name.lower()))
    return out


def load_exclude_episodes(root: "str | Path | None", overlay_path: "str | Path | None" = None) -> set[int]:
    """Episode ids flagged by the offline prefilter (``meta/exclude_episodes.json``).

    An offline prefilter may write this sidecar per dataset root: whole
    episodes that are frozen (high duplicate-frame fraction), video/action-length
    misaligned, or mostly idle. Read at index-build so the runtime loader never
    enumerates them (zero dataloader resamples). No-op (empty set) when ``root`` is
    ``None`` or the sidecar is absent. Accepts a bare ``[ids]`` list or a dict with
    an ``"exclude"`` key (extra keys, e.g. per-id reasons, are ignored). Ids are
    coerced to ``int``.
    """
    if root is None:
        return set()
    excluded: set[int] = set()
    path = Path(root) / "meta" / "exclude_episodes.json"
    if path.exists():
        data = json.loads(path.read_text())
        ids = data.get("exclude", []) if isinstance(data, dict) else data
        excluded.update(int(x) for x in ids)
    if overlay_path:
        overlay = Path(overlay_path)
        if not overlay.is_file():
            raise FileNotFoundError(f"Configured exclusion overlay does not exist: {overlay}")
        try:
            overlay_roots = json.loads(overlay.read_text()).get("roots", {})
        except Exception as e:
            raise RuntimeError(f"Failed to read exclusion overlay {overlay}: {e}") from e
        key = match_overlay_root(overlay_roots, root)
        if key is None:
            log.warning(f"Exclusion overlay {overlay} has no entry for root {root}")
        else:
            excluded.update(int(x) for x in overlay_roots[key].get("exclude", []))
    return excluded


def match_overlay_root(overlay_roots: "dict[str, Any]", root: "str | Path") -> str | None:
    """Key of ``overlay_roots`` that names dataset ``root``, or None.

    An absolute key must equal the resolved root path. A relative key (e.g. the
    dataset's path inside the Open-H repository, ``jhu/imerse/nephfat/nephfat``)
    matches when it equals the trailing components of the root path, as given or with
    symlinks resolved, so the same overlay file works wherever the data is stored
    (including a data directory of symlinks). More than one match is an error.
    """
    resolved = Path(root).resolve()
    if str(resolved) in overlay_roots:
        return str(resolved)
    candidates = (Path(_os.path.abspath(root)).parts, resolved.parts)
    matches = []
    for key in overlay_roots:
        parts = Path(key).parts
        if parts and not Path(key).is_absolute() and any(c[-len(parts) :] == parts for c in candidates):
            matches.append(key)
    if len(matches) > 1:
        raise ValueError(f"Exclusion overlay keys {matches} all match root {resolved}")
    return matches[0] if matches else None


def load_camera_motion_segments(root: "str | Path | None") -> "dict[int, list[list[int]]]":
    """Per-episode camera-panning frame segments (``meta/camera_motion_frames.json``).

    An offline video detector may write this sidecar for rigs
    where the endoscope pans with no camera channel in the action (e.g. smarts): a
    map ``{episode_index: [[start, end), ...]}`` of half-open frame ranges the video
    detector flagged as camera motion. Read at index-build so the loader can resample
    any *window* that lands on a pan — keeping the rest of the (good) episode, unlike
    the whole-episode ``exclude_episodes.json`` drop. No-op (empty dict) when ``root``
    is ``None`` or the sidecar is absent. Accepts a ``{"segments": {...}}`` wrapper or a
    bare ``{episode: segments}`` mapping; keys are coerced to ``int``.
    """
    if root is None:
        return {}
    path = Path(root) / "meta" / "camera_motion_frames.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    seg_map = data.get("segments", data) if isinstance(data, dict) else {}
    return {int(k): [[int(a), int(b)] for a, b in v] for k, v in seg_map.items()}


def window_camera_overlap(segments: "list[list[int]]", win_start: int, win_len: int) -> float:
    """Fraction of a window ``[win_start, win_start+win_len)`` covered by any
    camera-motion segment (half-open ``[start, end)`` frame ranges). Pure index math
    — no decode — so the runtime cost is a handful of comparisons per sampled window.
    """
    if win_len <= 0:
        return 0.0
    win_end = win_start + win_len
    covered = 0
    for s, e in segments:
        lo, hi = max(win_start, s), min(win_end, e)
        if hi > lo:
            covered += hi - lo
    return covered / win_len


def should_reject_window(segments: "list[list[int]]", win_start: int, win_len: int, overlap_max: float = 0.25) -> bool:
    """True when a window overlaps camera-motion segments by more than ``overlap_max``
    (strictly greater — a window exactly at the threshold is kept). Empty ``segments``
    (the common case: no sidecar) short-circuits to False."""
    if not segments:
        return False
    return window_camera_overlap(segments, win_start, win_len) > overlap_max


def frame_duplicate_fraction(video: "torch.Tensor | None", eps: float = 1e-3) -> float:
    """Fraction of consecutive frame-pairs that are near-identical.

    ``video`` is a ``[T, ...]`` tensor (LeRobot ``[T, C, H, W]`` float in [0, 1]).
    A pair ``(t-1, t)`` counts as a duplicate when its mean-absolute-difference is
    ``<= eps``. Returns ``duplicates / (T - 1)``, or ``0.0`` when ``T < 2`` or the
    video is ``None`` (nothing to compare -> not a duplicate window).
    """
    if video is None or video.shape[0] < 2:
        return 0.0
    flat = video.reshape(video.shape[0], -1).float()  # [T, C*H*W]
    mad = (flat[1:] - flat[:-1]).abs().mean(dim=1)  # [T-1]
    return float((mad <= eps).float().mean().item())


_VIDEO_SOURCE_CHOICES = ("mode_conditional", "clean", "annotated")


def select_video_delta_timestamps(
    delta_timestamps: dict[str, list[float]],
    video_key: str | None,
    *,
    clean_twin_available: bool,
    video_source: str = "mode_conditional",
    clean_required: bool = False,
) -> dict[str, list[float]]:
    """Request only the video streams needed by ``video_source``.

    ``mode_conditional`` needs both streams because the joint mode is sampled after
    LeRobot has decoded the row. Fixed ``clean`` and ``annotated`` recipes can avoid
    decoding the stream that will be discarded. The clean stream is returned under
    ``<video_key>_clean``; :func:`apply_mode_conditional_video` maps it back to the
    canonical key before downstream transforms run.

    The input mapping is never mutated. Datasets without a clean twin are unchanged.
    """
    if video_source not in _VIDEO_SOURCE_CHOICES:
        raise ValueError(f"video_source must be one of {_VIDEO_SOURCE_CHOICES}, got {video_source!r}")
    if clean_required and video_source != "clean":
        raise ValueError(f"a required-clean root must use video_source='clean', got {video_source!r}")
    if clean_required and not clean_twin_available:
        raise FileNotFoundError(f"required clean video stream is unavailable for {video_key!r}")
    if not video_key or video_key not in delta_timestamps or not clean_twin_available:
        return dict(delta_timestamps)

    clean_key = f"{video_key}_clean"
    selected = dict(delta_timestamps)
    if video_source == "clean":
        selected.pop(video_key, None)
        selected[clean_key] = delta_timestamps[video_key]
    elif video_source == "annotated":
        selected.pop(clean_key, None)
    else:
        selected.setdefault(clean_key, delta_timestamps[video_key])
    return selected


def retain_requested_video_features(
    features: dict[str, dict[str, Any]], requested_video_keys: set[str]
) -> dict[str, dict[str, Any]]:
    """Drop video metadata for streams the loader did not request.

    LeRobot decodes every key advertised as ``dtype=video`` even when that key is
    absent from ``delta_timestamps`` (it decodes one current frame in that case).
    Restricting metadata to the requested keys prevents those otherwise invisible
    reads while preserving every non-video feature.
    """
    return {
        key: value for key, value in features.items() if value.get("dtype") != "video" or key in requested_video_keys
    }


def apply_mode_conditional_video(
    sample: dict,
    mode: str,
    video_key: str,
    clean_key: str,
    video_source: str = "mode_conditional",
) -> dict:
    """Select annotated/clean video without changing existing callers by default.

    wound_closure burns needle entry/exit dots onto its endoscope video for the whole
    episode. That's a useful goal signal for the **policy**, but the forward-dynamics
    expert must not learn to *generate* the dots and the inverse-dynamics expert must
    not *shortcut* on them — so ``forward_dynamics``/``inverse_dynamics`` should see the
    clean frames. The clean frames live under ``<video_key>_clean`` (a twin video stream, when the
    dataset provides one). This picks the right one for the
    mode, then drops the extra key so downstream is mode-agnostic. Pure + in-place;
    no-op unless ``clean_key`` is present in ``sample`` (i.e. all other datasets).

    ``video_source`` is explicit because a latent cache is keyed only by
    ``(dataset_name, idx)`` and cannot represent a different pixel target per sampled
    joint mode. ``mode_conditional`` preserves the historical behavior, ``clean``
    always selects the clean twin when present, and ``annotated`` keeps the original.
    """
    if video_source not in _VIDEO_SOURCE_CHOICES:
        raise ValueError(f"video_source must be one of {_VIDEO_SOURCE_CHOICES}, got {video_source!r}")
    if not clean_key or clean_key not in sample:
        return sample
    if video_source == "clean" or (
        video_source == "mode_conditional" and mode in ("forward_dynamics", "inverse_dynamics")
    ):
        sample[video_key] = sample[clean_key]
    sample.pop(clean_key, None)
    return sample


_COARSE_DROPOUT_DEFAULTS = {
    "num_holes": 2,
    "hole_h_frac": 0.25,
    "hole_w_frac": 0.25,
    "fill": 0.0,
    "p": 1.0,
}
# Modes that get coarse dropout: ONLY inverse_dynamics, where every video frame is
# CONDITIONING context (build_sequence_plan_from_mode: condition_frame_indexes_vision =
# all frames) so there is no vision loss to corrupt. forward_dynamics AND policy both
# condition only on frame 0 and NOISE frames 1..T as a video generation target
# (vision flow-matching loss) -- occluding the whole clip there would train the model to
# generate the dropout boxes. cross_view_inpainting likewise generates video. Excluded.
_COARSE_DROPOUT_MODES = ("inverse_dynamics",)


def coarse_dropout_config(value: bool | dict | None) -> dict | None:
    """Resolve the ``coarse_dropout`` option: None/False = off, True = defaults, dict = defaults + overrides."""
    if value is None or value is False:
        return None
    cfg = dict(_COARSE_DROPOUT_DEFAULTS)
    if value is not True:
        unknown = set(value) - set(_COARSE_DROPOUT_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown coarse_dropout keys {sorted(unknown)}; known: {sorted(_COARSE_DROPOUT_DEFAULTS)}")
        cfg.update(value)
    return cfg


def apply_coarse_dropout(
    video: "torch.Tensor", cfg: dict, generator: "torch.Generator | None" = None
) -> "torch.Tensor":
    """Zero ``num_holes`` coarse rectangles (temporally consistent) in a (T,C,H,W) clip.

    Returns a NEW tensor (clone) so the source (e.g. a shared clean twin) is untouched.
    Same rectangles across all T frames -> occlusion, not per-frame noise. No-op with
    probability ``1 - p`` and for non-4D inputs (returned unchanged).
    """
    if video is None or getattr(video, "ndim", 0) != 4:
        return video
    p = float(cfg.get("p", 1.0))
    if p < 1.0 and float(torch.rand((), generator=generator).item()) >= p:
        return video
    _t, _c, h, w = video.shape
    hh = max(1, int(round(h * float(cfg.get("hole_h_frac", 0.25)))))
    hw = max(1, int(round(w * float(cfg.get("hole_w_frac", 0.25)))))
    fill = float(cfg.get("fill", 0.0))
    out = video.clone()
    for _ in range(int(cfg.get("num_holes", 2))):
        y = int(torch.randint(0, max(1, h - hh + 1), (1,), generator=generator).item())
        x = int(torch.randint(0, max(1, w - hw + 1), (1,), generator=generator).item())
        out[:, :, y : y + hh, x : x + hw] = fill
    return out


def required_future_frame_offset(delta_timestamps: dict[str, list[float]], source_fps: float) -> int:
    """Last queried source row, using LeRobot's actual seconds-to-frames rounding.

    History's negative offsets do not shorten the valid future span. The maximum
    includes every requested modality, including asymmetric video/action rates.
    """
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source FPS must be finite and positive")
    indices = get_delta_indices(delta_timestamps, source_fps)
    return max(0, max((offset for offsets in indices.values() for offset in offsets), default=0))


def build_episode_spans(
    episodes: Any,
    episode_ids: Sequence[int],
    chunk_length: int,
    sample_stride: int = 1,
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Build valid episode spans for LeRobot frame queries.

    ``chunk_length`` is the last required offset in SOURCE rows. Callers must
    convert their query times using the source FPS; model token count is not a
    source-row horizon when the two frame rates differ.

    Returns:
        - episode spans as ``(episode_id, sample_start, valid_len)``
        - total valid sample count across selected episodes
        - total raw frame count across selected episodes
    """
    assert sample_stride >= 1, f"sample_stride must be >= 1, got {sample_stride}"

    dataset_from_index = list(episodes["dataset_from_index"])
    dataset_to_index = list(episodes["dataset_to_index"])
    length = list(episodes["length"])

    spans: list[tuple[int, int, int]] = []
    valid_count = 0
    sample_count = 0
    for episode_id in episode_ids:
        start = dataset_from_index[episode_id]
        stop = dataset_to_index[episode_id]
        raw_valid_len = stop - start - chunk_length
        if raw_valid_len > 0:
            valid_len = (raw_valid_len + sample_stride - 1) // sample_stride
            spans.append((episode_id, start, valid_len))
            valid_count += valid_len
        sample_count += int(length[episode_id])

    return spans, valid_count, sample_count


def cap_episode_spans(episode_spans: list[tuple], max_windows: int | None) -> list[tuple]:
    """Truncate ``(episode_id, sample_start, valid_len)`` spans to the first
    ``max_windows`` windows total (clamping the last kept episode's ``valid_len``).

    ``sample_stride`` can only thin windows WITHIN an episode (floor 1/episode), so
    it cannot shrink a many-episode dataset below its episode count. This caps the
    flat window index instead, giving a genuinely small, deterministic subset that
    precompute and training share (same env -> same idx-range -> all-cached).

    Returns the list unchanged when ``max_windows`` is None (default / real runs).
    """
    if max_windows is None:
        return episode_spans
    out: list[tuple] = []
    total = 0
    for span in episode_spans:
        ep_id, start, valid_len, *extras = span
        if total >= max_windows:
            break
        take = min(valid_len, max_windows - total)
        out.append((ep_id, start, take, *extras))
        total += take
    return out


def _normalize_split(split: str) -> str:
    """Normalize split name to one of ``'train'``, ``'val'``, ``'full'``."""
    s = split.lower().strip()
    if s in {"val", "valid", "validation", "eval", "test"}:
        return "val"
    if s in {"train", "full"}:
        return s
    raise ValueError(f"Unsupported {split=}. Use train/val/full.")


class BaseActionLeRobotDataset(Dataset):
    """Reusable base class for Action LeRobot-backed map-style datasets.

    Subclasses typically:
    1) call ``_register_source`` to register one or more LeRobot sources
    2) implement ``__getitem__`` for dataset-specific sample parsing
    3) call ``_build_result`` to assemble the return dict
    """

    # Applied as: R_opencv = R_native @ _to_opencv
    # Subclasses override in __init__; default is identity (no correction).

    # Bundled normalization stats directory.  Stats are committed at
    # ``<_NORMALIZERS_DIR>/<embodiment>_<pose>_<rotation_format>.json`` (flat
    # layout matching the existing UMI files) and produced by
    # ``projects/cosmos3/vfm/datasets/action/compute_action_stats.py``.
    # Subclasses that need a different filename scheme can override
    # :meth:`_normalizer_filename`.
    _NORMALIZERS_DIR: ClassVar[Path] = Path(__file__).parent / "normalizers"

    # Sub-modes sampled per request when ``mode == "joint"``. Order MUST match the
    # order of any ``joint_mode_weights`` passed in, e.g. [0.4, 0.2, 0.4, 0.0] for
    # FD 0.4 / ID 0.2 / policy 0.4 and no cross-view inpainting.
    # cross_view_inpainting requires the multiview canvas (dVRK); single-view configs
    # must set its weight to 0.
    _JOINT_MODES: ClassVar[tuple[str, ...]] = (
        "forward_dynamics",
        "inverse_dynamics",
        "policy",
        "cross_view_inpainting",
    )

    # Outcome-split filtering (from meta/info.json 'splits'). failure/fail episodes gate
    # the policy expert off (FDM/IDM keep them); bad_frames episodes are dropped entirely;
    # recovery is KEPT (a policy skill). No-op for datasets without a 'splits' field.
    _EXCLUDE_BAD_FRAMES: ClassVar[bool] = True
    _GATE_FAILURE_POLICY: ClassVar[bool] = True

    def __init__(
        self,
        *,
        fps: float,
        chunk_length: int,
        split_seed: int,
        split_val_ratio: float,
        split: str,
        mode: str,
        embodiment_type: str,
        viewpoint: Viewpoint,
        pose_convention: str | None = None,
        rotation_format: str | None = None,
        action_normalization: ActionNormalization | None = None,
        tolerance_s: float = 1e-4,
        max_loaded_datasets: int = _LRU_DATASET_MAX_LOADED,
        skip_video_loading: bool = False,
        sample_stride: int = 1,
        enable_fast_init: bool = False,
        fast_init_max_workers: int = 64,
        joint_mode_weights: list[float] | None = None,
        fd_mono_condition_prob: float = 0.0,
        fd_mono_condition_views: tuple[str, ...] | list[str] = ("endoscope",),
        latent_cache_root: str | None = None,
        latent_cache_name: str | None = None,
        latent_cache_version_hash: str | None = None,
        latent_cache_required: bool = False,
        video_source: str = "mode_conditional",
        required_clean_video_roots: list[str] | tuple[str, ...] | None = None,
        exclude_episodes_overlay: str | None = None,
        duplicate_frame_max_fraction: float | None = None,
        duplicate_frame_eps: float = 1e-3,
        policy_idle_max_fraction: float | None = None,
        latent_cache_variant_roots: list[str] | tuple[str, ...] | None = None,
        policy_view_dropout_mono_prob: float = 0.0,
        policy_view_dropout_single_prob: float = 0.0,
        image_aug_strength: float = 0.0,
        image_aug_pixel_drop: float = 0.0,
        idle_filter_frac: float | None = None,
        action_chunk_length: int | None = None,
        action_fps: float | None = None,
        camera_overlap_max: float = 0.25,
        coarse_dropout: bool | dict | None = None,
        max_windows: int | None = None,
    ) -> None:
        super().__init__()
        _ensure_hf_hub_offline()
        _patch_decoder_cache()
        self._memprofile = _memprofile_enabled()

        assert sample_stride >= 1, f"sample_stride must be >= 1, got {sample_stride}"
        assert fast_init_max_workers >= 1, f"fast_init_max_workers must be >= 1, got {fast_init_max_workers}"
        assert action_normalization is None or action_normalization in _ACTION_NORMALIZATION_CHOICES, (
            f"action_normalization must be None or one of {_ACTION_NORMALIZATION_CHOICES}, got {action_normalization!r}"
        )

        with rss_tracker(f"{self.__class__.__name__}.__init__", enabled=self._memprofile):
            self._fps = float(fps)
            self._dt = 1.0 / self._fps
            self._chunk_length = int(chunk_length)
            # Decoupled ACTION temporal rate (mixed-fps mRoPE, Cosmos3 report §2.4):
            # actions may be sampled DENSER than video over the SAME window, so mRoPE
            # FPS modulation places both streams on the shared real-world time axis.
            # Default (None) => identical to the video rate (current behavior, byte-for-byte).
            self._action_fps = float(action_fps) if action_fps is not None else self._fps
            self._action_dt = 1.0 / self._action_fps
            self._action_chunk_length = (
                int(action_chunk_length) if action_chunk_length is not None else self._chunk_length
            )
            # The action window must fit inside the video window: build_episode_spans
            # reserves the VIDEO chunk_length at each episode's tail, so a longer action
            # extent would read past the episode end.
            assert (self._action_chunk_length - 1) * self._action_dt <= self._chunk_length * self._dt + 1e-6, (
                f"action window {(self._action_chunk_length - 1) * self._action_dt:.4f}s exceeds "
                f"video window {self._chunk_length * self._dt:.4f}s "
                f"(action_chunk_length={self._action_chunk_length}, action_fps={self._action_fps}, "
                f"chunk_length={self._chunk_length}, fps={self._fps})"
            )
            # Latent-cache mode: when a cache root is set we (a) tell lerobot to
            # skip video decode entirely (same mechanism as skip_video_loading)
            # and (b) inject the precomputed VAE latent per sample. The reader is
            # built lazily per worker process (safe_open handles are not fork-safe).
            self._latent_cache_root = latent_cache_root
            self._latent_cache_name = latent_cache_name
            self._latent_cache_version_hash = latent_cache_version_hash
            self._latent_cache_required = bool(latent_cache_required)
            if video_source not in _VIDEO_SOURCE_CHOICES:
                raise ValueError(f"video_source must be one of {_VIDEO_SOURCE_CHOICES}, got {video_source!r}")
            self._video_source = video_source
            self._required_clean_video_roots = {
                str(Path(root).resolve()) for root in (required_clean_video_roots or [])
            }
            self._exclude_episodes_overlay = exclude_episodes_overlay
            if self._latent_cache_required and not latent_cache_version_hash:
                raise ValueError("latent_cache_required=True requires a pinned latent_cache_version_hash")
            # Graceful fallback: if the cache was configured but the manifest doesn't
            # exist yet (precompute still in progress) or this dataset isn't in it yet,
            # fall back to live video decode + VAE encode rather than crashing. The
            # check runs at __init__ (before any shard loads) because _get_dataset
            # strips video keys based on this flag — per-worker lazy flipping is too late.
            self._latent_cache_active = self._check_latent_cache_ready(latent_cache_root, latent_cache_name)
            if self._latent_cache_required and not self._latent_cache_active:
                raise RuntimeError(
                    "Required latent cache is not ready for "
                    f"{latent_cache_name!r} under {latent_cache_root!r}; refusing live-VAE fallback"
                )
            if self._latent_cache_required:
                from cosmos3_h_embodied.data._unsupported import read_manifest

                manifest = read_manifest(latent_cache_root)
                top_hash = manifest.get("version_hash") if manifest else None
                entry_hash = (
                    manifest.get("datasets", {}).get(str(latent_cache_name), {}).get("version_hash")
                    if manifest
                    else None
                )
                if top_hash != latent_cache_version_hash or entry_hash != latent_cache_version_hash:
                    raise RuntimeError(
                        "Required latent cache signature mismatch for "
                        f"{latent_cache_name!r}: top={top_hash!r}, entry={entry_hash!r}, "
                        f"expected={latent_cache_version_hash!r}"
                    )
            # Precomputed color-jitter variant roots (see _pick_latent_cache_root).
            # Each must hold the SAME entry name / window enumeration as the original.
            self._latent_cache_variant_roots = list(latent_cache_variant_roots or [])
            self._latent_readers: dict[str, Any] | None = None
            # Policy-mode view dropout (multiview canvas only). Probabilities are
            # disjoint: mono (drop BOTH wrist tiles) / single (drop ONE wrist tile);
            # the remainder keeps the full canvas. Applied in latent space so it is
            # latent-cache compatible. Train split only.
            if not 0.0 <= float(policy_view_dropout_mono_prob) <= 1.0:
                raise ValueError(
                    f"policy_view_dropout_mono_prob must be in [0,1], got {policy_view_dropout_mono_prob!r}"
                )
            if not 0.0 <= float(policy_view_dropout_single_prob) <= 1.0:
                raise ValueError(
                    f"policy_view_dropout_single_prob must be in [0,1], got {policy_view_dropout_single_prob!r}"
                )
            if float(policy_view_dropout_mono_prob) + float(policy_view_dropout_single_prob) > 1.0:
                raise ValueError(
                    "policy_view_dropout_mono_prob + policy_view_dropout_single_prob must be <= 1.0, got "
                    f"{policy_view_dropout_mono_prob!r} + {policy_view_dropout_single_prob!r}"
                )
            self._policy_view_dropout_mono_prob = float(policy_view_dropout_mono_prob)
            self._policy_view_dropout_single_prob = float(policy_view_dropout_single_prob)
            # On-the-fly TRAIN-only image aug (skipped entirely under the latent
            # cache, since _convert_video returns early when video isn't decoded).
            self._image_aug_strength = float(image_aug_strength)
            self._image_aug_pixel_drop = float(image_aug_pixel_drop)
            # Idle-window filtering: drop TRAIN windows whose action chunk is
            # >idle_filter_frac idle (None = off). Built after the span index.
            # Either a single float (applies to every mode) or a per-mode mapping,
            # e.g. {policy: 0.3, forward_dynamics: 0.5, inverse_dynamics: 0.5}. An
            # absent mode key => no idle filtering for that mode.
            if idle_filter_frac is None:
                self._idle_filter_frac = None
            elif hasattr(idle_filter_frac, "items"):
                # dict OR OmegaConf DictConfig (Hydra passes the latter, which is NOT
                # a dict subclass) — coerce to a plain {mode: float} dict.
                self._idle_filter_frac = {str(k): float(v) for k, v in idle_filter_frac.items()}
            else:
                self._idle_filter_frac = float(idle_filter_frac)
            self._allowed_indices: list[int] | None = None
            self._split_seed = int(split_seed)
            self._split_val_ratio = float(split_val_ratio)
            self._split = _normalize_split(split)
            self._mode = mode
            # Joint-mode sampling weights (P1). None == uniform, which reproduces
            # the previous random.choice behavior exactly, so existing configs are
            # unaffected until they pass joint_mode_weights.
            self._joint_mode_weights = self._validate_mode_weights(joint_mode_weights)
            # Mono-FD (partial-observation forward dynamics): with this probability a
            # forward_dynamics sample conditions ONLY on the fd_mono_condition_views
            # tiles of frame 0 (e.g. endoscope) and generates everything else — wrist
            # tiles at t=0 included. Only meaningful on multiview-canvas entries; keep
            # 0.0 (default) for single-view entries. Applied on the train split only
            # (eval keeps standard FD comparable); set COSMOS3_FD_MONO_FORCE=1 to force
            # it on every FD sample regardless of split (dedicated mono-FD evals).
            if not 0.0 <= float(fd_mono_condition_prob) <= 1.0:
                raise ValueError(f"fd_mono_condition_prob must be in [0, 1], got {fd_mono_condition_prob!r}")
            self._fd_mono_condition_prob = float(fd_mono_condition_prob)
            self._fd_mono_condition_views = list(fd_mono_condition_views)
            # Per-worker realized tally of the FD/ID/policy split, logged every
            # _mode_log_every joint samples so we can confirm the configured split.
            self._mode_counts: dict[str, int] = {m: 0 for m in self._JOINT_MODES}
            self._mode_log_every = 2000
            self._embodiment_type = embodiment_type
            self._viewpoint: Viewpoint = viewpoint
            self._pose_convention = pose_convention
            self._rotation_format = rotation_format
            self._action_normalization = action_normalization
            # Lazy-loaded stats cache, populated on first call to
            # :meth:`_normalize_action`.  Per-process (workers get their own).
            self._norm_stats: dict[str, torch.Tensor] | None = None
            self._tolerance_s = tolerance_s
            # Duplicate-frame filter: some clips freeze or were re-encoded from a
            # higher rate, so consecutive frames stay byte-identical even after the
            # 10 Hz delta_timestamps downsample -> no visual dynamics -> poisoned
            # FDM/video target. When the fraction of near-identical consecutive
            # frame-pairs in a window exceeds this, _fetch_sample raises a
            # "tolerance:" AssertionError so the subclass retry loop resamples.
            # None (default) disables it; enabled per-entry for suturebot/cao/srth.
            self._duplicate_frame_max_fraction = (
                None if duplicate_frame_max_fraction is None else float(duplicate_frame_max_fraction)
            )
            self._duplicate_frame_eps = float(duplicate_frame_eps)
            # Per-frame camera-motion mask (data-driven, off unless a shard has
            # meta/camera_motion_frames.json): reject any *window* that lands on an
            # endoscope pan for rigs whose action has no camera channel (smarts). Keeps
            # the rest of the episode — unlike the whole-episode exclude_episodes drop.
            # Populated per-shard in _append_index_records; empty map => pure no-op.
            self._camera_motion_segments: dict[int, dict[int, list[list[int]]]] = {}
            self._camera_overlap_max = float(camera_overlap_max)
            # Mode-conditional coarse dropout (inverse_dynamics only, see _COARSE_DROPOUT_MODES). None = off.
            self._coarse_dropout = coarse_dropout_config(coarse_dropout)
            # Optional cap on the number of windows per root (smoke tests). None = all windows.
            self._max_windows = None if max_windows is None else int(max_windows)
            # Universal in-session bad-sample memory (ALL loaders): once a sample fails
            # (tolerance / decode / etc.) it is remembered by stable shard key + row and
            # never resampled this run — see _mark_bad / _is_remembered_bad, checked in
            # _fetch_sample. Per-worker (each DataLoader worker keeps its own set).
            self._session_bad_by_shard_key: dict[str, set[int]] = {}
            # Policy-gated minimal-action filter: for POLICY-mode samples only, reject
            # windows whose idle-frame fraction exceeds this (arms barely moved) so the
            # policy expert isn't trained to output "do nothing". Kept for FDM/IDM (a
            # static window is a valid dynamics target). None (default) disables it.
            self._policy_idle_max_fraction = (
                None if policy_idle_max_fraction is None else float(policy_idle_max_fraction)
            )
            self._max_loaded_datasets = max_loaded_datasets
            self._skip_video_loading = (
                skip_video_loading
                if isinstance(skip_video_loading, bool)
                else str(skip_video_loading).lower() == "true"
            )
            self._sample_stride = int(sample_stride)
            self._enable_fast_init = (
                enable_fast_init if isinstance(enable_fast_init, bool) else str(enable_fast_init).lower() == "true"
            )
            self._fast_init_max_workers = int(fast_init_max_workers)
            self._delta_timestamps: dict[str, list[float]] = {}
            self._to_opencv: np.ndarray | dict[str, np.ndarray] = np.eye(3, dtype=np.float32)

            if pose_convention is None:
                log.warning(
                    f"{self.__class__.__name__}: pose_convention is not set. "
                    "Consider specifying 'backward_framewise' or 'backward_anchored'."
                )

            self._datasets: list[LeRobotDataset | None] = []
            self._dataset_build_args: list[dict[str, Any] | None] = []
            self._loaded_lru: OrderedDict[int, None] = OrderedDict()

            # -- Flat index structures (populated by _append_index_records) --
            # Together these two lists form a searchable map from a flat
            # global index to (dataset, row, episode, frame).  One entry per
            # episode span across *all* registered sources.
            #
            # _episode_records[i] = (ds_idx, sample_start, valid_len, episode_id)
            #   ds_idx       – which source dataset (index into _datasets)
            #   sample_start – first row of this span in that dataset's table
            #   valid_len    – number of usable frames in this span
            #   episode_id   – the episode this span belongs to
            #
            # _episode_cum_ends[i] = running total of valid_len through span i
            #   Used for O(log N) lookup via bisect_right in _resolve_index.
            self._episode_records: list[tuple[int, int, int, int]] = []
            self._episode_frame_bases: list[int] = []
            self._episode_cum_ends: list[int] = []
            self._num_valid_indices = 0
            self._domain_id = get_domain_id(self._embodiment_type)

            # Deferred-init shard roots — a list of root paths.
            # Subclasses populate this in __init__; _register_sources()
            # reads _delta_timestamps and _tolerance_s from self (both
            # initialised above, with _delta_timestamps overridden by
            # each subclass).
            # ActionUnifiedIterableDataset.assign_worker uses len() for
            # round-robin shard distribution and _register_sources(indices)
            # for deferred loading.  When empty, shard distribution is
            # skipped (every worker iterates the full dataset).
            self._all_shard_roots: list[str] = []

    # -- caption lookups -----------------------------------------------------

    # How per-root caption data (prefix, CMR procedure and instruction segments) is looked
    # up for a sample. "fixed" uses the sample's own root. "legacy" reproduces the
    # training run of the released checkpoint: it indexes the per-root tables with the
    # *registered* dataset index, which is 0 whenever a DataLoader worker serves a single
    # shard, so every sample of a family gets the first root's caption data.
    _caption_prefix_mode: str = "fixed"

    def _set_caption_prefix_mode(self, caption_prefix_mode: str) -> None:
        if caption_prefix_mode not in CAPTION_PREFIX_MODES:
            raise ValueError(f"caption_prefix_mode must be one of {CAPTION_PREFIX_MODES}, got {caption_prefix_mode!r}")
        self._caption_prefix_mode = caption_prefix_mode

    def _caption_root_index(self, dataset_idx: int | None) -> int | None:
        """Index into ``_all_shard_roots`` for the per-root caption tables of a registered dataset."""
        if dataset_idx is None or self._caption_prefix_mode == "legacy":
            return dataset_idx
        return self._all_shard_roots.index(self._dataset_build_args[dataset_idx]["root"])

    # -- public properties ---------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def split(self) -> str:
        return self._split

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def domain_id(self) -> int:
        return self._domain_id

    # -- source registration -------------------------------------------------

    def _register_source(
        self,
        *,
        delta_timestamps: dict[str, list[float]],
        tolerance_s: float,
        root: str | None = None,
        repo_id: str = "local",
        force_cache_sync: bool = False,
        download_videos: bool = False,
        video_backend: str | None = None,
        revision: str | None = None,
        dataset_label: str | None = None,
        prefetched_meta: LeRobotDatasetMetadata | None = None,
    ) -> LeRobotDatasetMetadata:
        """Register a LeRobot dataset source lazily (metadata-only at init).

        ``prefetched_meta`` lets subclasses load metadata in a thread pool
        (``LeRobotDatasetMetadata`` reads are pure I/O — ``info.json`` +
        ``episodes.parquet`` + ``tasks.parquet``) and then hand the ready
        object to the serial append-path below, which still manages the
        order-sensitive shared state (``_datasets`` / ``_dataset_build_args``
        / ``_episode_records`` / ``_episode_cum_ends``).  When ``None`` the
        caller gets the original single-threaded behavior.
        """
        label_str = f" [{dataset_label}]" if dataset_label else ""
        cls = self.__class__.__name__
        # "local" is not a valid PEP 440 version, so LeRobot's
        # is_valid_version() check skips the get_safe_version() HF API call.
        if repo_id == "local" and revision is None:
            revision = "local"

        with rss_tracker(f"{cls}{label_str} — metadata load", enabled=self._memprofile):
            if prefetched_meta is not None:
                meta = prefetched_meta
            else:
                meta = LeRobotDatasetMetadata(
                    repo_id=repo_id,
                    root=root,
                    revision=revision,
                    force_cache_sync=force_cache_sync,
                )
            if root is not None and str(Path(root).resolve()) in self._required_clean_video_roots:
                video_key = getattr(self, "VIDEO_KEY", None)
                clean_key = f"{video_key}_clean" if video_key else None
                clean_available = bool(
                    clean_key
                    and clean_key in meta.video_keys
                    and _os.path.isdir(_os.path.join(root, "videos", clean_key))
                )
                select_video_delta_timestamps(
                    delta_timestamps,
                    video_key,
                    clean_twin_available=clean_available,
                    video_source=self._video_source,
                    clean_required=True,
                )
                log.info(f"{cls}{label_str}: required clean stream verified: {clean_key}")
            ds_idx = len(self._datasets)
            self._datasets.append(None)
            self._dataset_build_args.append(
                {
                    "repo_id": repo_id,
                    "root": root,
                    "delta_timestamps": delta_timestamps,
                    "tolerance_s": tolerance_s,
                    "force_cache_sync": force_cache_sync,
                    "download_videos": download_videos,
                    "video_backend": video_backend,
                    "revision": revision,
                }
            )

        self._append_index_records(meta=meta, ds_idx=ds_idx, dataset_label=dataset_label)

        return meta

    def _append_index_records(
        self,
        *,
        meta: LeRobotDatasetMetadata,
        ds_idx: int,
        dataset_label: str | None = None,
    ) -> None:
        """Populate episode split / index records from dataset metadata."""
        episode_ids = split_episode_ids(
            total_episodes=meta.total_episodes,
            seed=self._split_seed,
            val_ratio=self._split_val_ratio,
            split=self._split,
        )

        if hasattr(self, "_filter_valid_episodes"):
            episode_ids = self._filter_valid_episodes(meta, episode_ids)

        # Offline-prefilter overlay (meta/exclude_episodes.json sidecar and/or the
        # configured exclusion overlay): drop whole episodes flagged frozen /
        # video-action-misaligned / mostly-idle so the runtime loader never
        # enumerates them (zero dataloader resamples). No-op when the sidecar is
        # absent. Parallel to the bad_frames drop below.
        excluded = load_exclude_episodes(getattr(meta, "root", None), self._exclude_episodes_overlay)
        if excluded:
            before = len(episode_ids)
            episode_ids = [e for e in episode_ids if e not in excluded]
            log.info(
                f"{self.__class__.__name__}: exclude-index dropped {before - len(episode_ids)} episodes (of {before})"
            )

        # Per-frame camera-motion mask overlay (parallel to the episode drop above,
        # but window-level): keep the episode, resample only the windows that land on
        # a pan. No-op when the sidecar is absent (every dataset but smarts today).
        cam_segs = load_camera_motion_segments(getattr(meta, "root", None))
        if cam_segs:
            self._camera_motion_segments[ds_idx] = cam_segs
            log.info(
                f"{self.__class__.__name__}: camera-motion mask on {len(cam_segs)} "
                f"episodes (window-level resample, overlap_max={self._camera_overlap_max})"
            )

        # Outcome-split overlay (from meta/info.json 'splits'; no-op when absent):
        #  - bad_frames -> drop those episodes entirely
        #  - failure/fail -> record so the policy expert is gated off per-episode
        info = getattr(meta, "info", None) or {}
        if not hasattr(self, "_failure_episodes"):
            self._failure_episodes: dict[int, set[int]] = {}
        if self._EXCLUDE_BAD_FRAMES:
            bad = split_episode_set(info, ["bad_frames"])
            if bad:
                before = len(episode_ids)
                episode_ids = [e for e in episode_ids if e not in bad]
                log.info(f"{self.__class__.__name__}: dropped {before - len(episode_ids)} bad_frames episodes")
        if self._GATE_FAILURE_POLICY:
            fail = split_episode_set(info, ["failure", "fail"])
            if fail:
                self._failure_episodes.setdefault(ds_idx, set()).update(fail)
                log.info(f"{self.__class__.__name__}: {len(fail)} failure episodes -> policy-gated (FDM/IDM keep them)")

        source_args = self._dataset_build_args[ds_idx]
        future_offset = required_future_frame_offset(source_args["delta_timestamps"], float(meta.fps))
        source_args["future_frame_offset"] = future_offset
        source_args["source_fps"] = float(meta.fps)
        episode_spans, valid_count, sample_count = build_episode_spans(
            episodes=meta.episodes,
            episode_ids=episode_ids,
            chunk_length=future_offset,
            sample_stride=self._sample_stride,
        )
        indexed_episode_spans = [(*span, 0) for span in episode_spans]
        if hasattr(self, "_filter_episode_spans"):
            indexed_episode_spans = self._filter_episode_spans(meta, indexed_episode_spans)
            valid_count = sum(span[2] for span in indexed_episode_spans)

        if self._max_windows is not None:
            indexed_episode_spans = cap_episode_spans(indexed_episode_spans, self._max_windows)
            valid_count = sum(span[2] for span in indexed_episode_spans)

        class_name = self.__class__.__name__
        label = f" [{dataset_label}]" if dataset_label else ""
        log.info(f"{class_name}{label}: split={self._split}, num episodes={len(episode_ids)}")
        if sample_count > 0:
            log.info(
                f"{class_name}{label}: kept {valid_count} / {sample_count} "
                f"({100 * valid_count / sample_count:.2f} %) samples"
            )

        for episode_id, sample_start, valid_len, episode_frame_base in indexed_episode_spans:
            self._episode_records.append((ds_idx, sample_start, valid_len, episode_id))
            self._episode_frame_bases.append(int(episode_frame_base))
            self._num_valid_indices += valid_len
            self._episode_cum_ends.append(self._num_valid_indices)

    # -- deferred shard registration -----------------------------------------

    def _register_sources(self, indices: list[int] | None = None) -> None:
        """Register a subset (or all) of the shard roots in ``_all_shard_roots``.

        Called by ``ActionUnifiedIterableDataset.assign_worker`` during training,
        or explicitly by eval/visualization scripts after construction.

        ``_all_shard_roots`` is a list of root paths.  Per-shard args that are
        shared across all shards (``delta_timestamps``, ``tolerance_s``) are
        taken from ``self``.  Subclasses may override this for extra per-shard
        setup (e.g. loading instruction segments).

        When ``enable_fast_init=True``, ``LeRobotDatasetMetadata`` (a pure-IO
        read of ``info.json`` + ``episodes.parquet`` + ``tasks.parquet``) is
        prefetched in a thread pool and handed to the order-sensitive
        serial register loop via ``prefetched_meta=``.  Shard count scales
        the speedup; for single-shard datasets the two paths are
        equivalent.

        Args:
            indices: Which entries of ``_all_shard_roots`` to register.
                ``None`` means all.
        """
        if indices is None:
            indices = list(range(len(self._all_shard_roots)))
        if not indices:
            return

        roots = [self._all_shard_roots[i] for i in indices]

        if self._enable_fast_init:
            # ``_ensure_hf_hub_offline`` already ran in ``__init__`` and is
            # idempotent; no need to re-invoke here.
            workers = max(1, min(self._fast_init_max_workers, len(roots)))
            metas: list[LeRobotDatasetMetadata | None] = _parallel_map(
                lambda root: LeRobotDatasetMetadata(repo_id="local", root=root, revision="local"),
                roots,
                max_workers=workers,
                label=f"{type(self).__name__}: LeRobotDatasetMetadata prefetch",
            )
        else:
            metas = [None] * len(roots)

        for root, meta in zip(roots, metas):
            label = root.rsplit("/", 1)[-1] if "/" in root else root
            self._register_source(
                root=root,
                delta_timestamps=self._delta_timestamps,
                tolerance_s=self._tolerance_s,
                dataset_label=label,
                prefetched_meta=meta,
            )

    # -- lazy dataset access -------------------------------------------------

    def _get_dataset(self, ds_idx: int) -> LeRobotDataset:
        """Get or lazily construct the LeRobot dataset for the given source index.

        Loaded datasets are tracked with LRU ordering.  When the number of
        loaded datasets exceeds ``_max_loaded_datasets`` the least-recently-used
        dataset is evicted (set back to ``None``) so the GC can reclaim it.
        """
        ds = self._datasets[ds_idx]
        if ds is not None:
            self._loaded_lru.move_to_end(ds_idx)
            return ds

        _ensure_hf_hub_offline()

        build_args = self._dataset_build_args[ds_idx]
        if build_args is None:
            raise RuntimeError(f"Missing dataset build args for dataset index {ds_idx}")

        # Evict least-recently-used datasets before loading a new one.
        while len(self._loaded_lru) >= self._max_loaded_datasets:
            evict_idx, _ = self._loaded_lru.popitem(last=False)
            self._datasets[evict_idx] = None

        with rss_tracker(
            f"[WORKER {_os.getpid()}] Lazy-loaded ds[{ds_idx}]",
            enabled=self._memprofile,
            extras_fn=lambda: [f"total loaded={len(self._loaded_lru)}/{len(self._datasets)}"],
        ):
            delta_ts = build_args["delta_timestamps"]
            if self._skip_video_loading or self._latent_cache_active:
                # Covers both LeRobot v2 (``observation.images.<name>``) and
                # v3 (``observation.image.<name>``) video-column conventions.
                # Latent-cache mode also skips decode: the VAE latent is injected
                # from cache, so the raw frames are never needed.
                delta_ts = {k: v for k, v in delta_ts.items() if not k.startswith("observation.image")}

            # Request only the stream(s) the configured video policy can use. The
            # historical mode-conditional policy needs both because mode is sampled
            # after decode; fixed clean/annotated recipes avoid the discarded decode.
            _vkey = getattr(self, "VIDEO_KEY", None)
            if _vkey and _vkey in delta_ts:
                _clean = f"{_vkey}_clean"
                delta_ts = select_video_delta_timestamps(
                    delta_ts,
                    _vkey,
                    clean_twin_available=(
                        _clean in delta_ts or _os.path.isdir(_os.path.join(build_args["root"], "videos", _clean))
                    ),
                    video_source=self._video_source,
                    clean_required=(str(Path(build_args["root"]).resolve()) in self._required_clean_video_roots),
                )

            log.info(f"Loading shard root={build_args['root']}")
            ds = LeRobotDataset(
                repo_id=build_args["repo_id"],
                root=build_args["root"],
                delta_timestamps=delta_ts,
                tolerance_s=build_args["tolerance_s"],
                force_cache_sync=build_args["force_cache_sync"],
                download_videos=build_args["download_videos"],
                video_backend=build_args["video_backend"],
                revision=build_args["revision"],
                episodes=None,
            )
            # LeRobot otherwise decodes one current frame for *every* video feature,
            # including cameras/clean twins omitted from delta_timestamps. Keep only
            # the explicitly requested video keys in the live metadata view.
            requested_video_keys = set(delta_ts).intersection(ds.meta.video_keys)
            ds.meta.info["features"] = retain_requested_video_features(ds.meta.info["features"], requested_video_keys)
            self._datasets[ds_idx] = ds
            self._loaded_lru[ds_idx] = None

        return ds

    # -- index resolution ----------------------------------------------------

    def _resolve_index(self, idx: int) -> tuple[int, int, int, int]:
        """Map a flat global index to the source dataset, row, episode, and frame.

        Multiple datasets are concatenated into a single virtual sequence.
        Each episode contributes a contiguous *span* of valid frames, and
        ``_episode_cum_ends[i]`` stores the running total of valid frames
        through the *i*-th span.  For example, with two episodes of lengths
        5 and 3 the cum-ends are ``[5, 8]``, so global index 6 falls in the
        second span at offset 1.

        The lookup is O(log N) via :func:`bisect_right`.

        Returns:
            dataset_idx: Which source dataset this sample belongs to.
            row_idx: Row index *within* that dataset's LeRobot table.
            episode_id: The episode ID for this sample.
            frame_offset: Frame offset from the start of the episode span
                (0-based).

        Pure index math -- no I/O or dataset access.  Higher-level helpers
        like :meth:`_fetch_sample` build on this.
        """
        # Support negative indexing (e.g. -1 → last sample).
        if idx < 0:
            idx += self._num_valid_indices
        if idx < 0 or idx >= self._num_valid_indices:
            raise IndexError(f"{self.__class__.__name__} index {idx} out of range for size {self._num_valid_indices}")

        # _episode_cum_ends is a monotonically increasing list where entry i
        # holds the cumulative number of valid frames up to and including the
        # i-th episode span.  bisect_right finds the first span whose
        # cumulative end is strictly greater than idx, i.e. the span that
        # contains idx.
        #
        # Example: cum_ends = [5, 8, 20]
        #   idx=0  -> span_idx=0  (first span,  frames 0..4)
        #   idx=4  -> span_idx=0
        #   idx=5  -> span_idx=1  (second span, frames 5..7)
        #   idx=8  -> span_idx=2  (third span,  frames 8..19)
        span_idx = bisect_right(self._episode_cum_ends, idx)

        # The global index where this span begins is the previous span's
        # cumulative end (or 0 for the very first span).  The frame_offset
        # is how far idx is into this particular episode.
        span_start = 0 if span_idx == 0 else self._episode_cum_ends[span_idx - 1]
        frame_offset = idx - span_start

        # _episode_records[span_idx] stores (dataset_idx, row_start, valid_len,
        # episode_id).  row_start is the absolute row in the LeRobot table
        # where this episode begins.  With sample_stride=k, consecutive
        # valid indices map to rows k apart inside the episode, so the
        # effective row is row_start + frame_offset * sample_stride.
        dataset_idx, row_start, _, episode_id = self._episode_records[span_idx]
        row_idx = row_start + frame_offset * self._sample_stride
        episode_frame_offset = self._episode_frame_bases[span_idx] + frame_offset
        return dataset_idx, row_idx, episode_id, episode_frame_offset

    def _validate_mode_weights(self, weights: list[float] | None) -> list[float] | None:
        """Validate joint_mode_weights against _JOINT_MODES; None means uniform."""
        if weights is None:
            return None  # random.choices(..., weights=None) is uniform
        weights = [float(w) for w in weights]
        if len(weights) != len(self._JOINT_MODES):
            raise ValueError(
                f"joint_mode_weights must have {len(self._JOINT_MODES)} entries in the "
                f"order {self._JOINT_MODES}, got {weights!r}"
            )
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"joint_mode_weights must be non-negative with a positive sum, got {weights!r}")
        return weights

    def _maybe_log_mode_counts(self) -> None:
        """Periodically log the realized FD/ID/policy split (per worker)."""
        total = sum(self._mode_counts.values())
        if total == 0 or total % self._mode_log_every:
            return
        frac = {m: round(c / total, 3) for m, c in self._mode_counts.items()}
        log.info(f"[WORKER {_os.getpid()}] joint-mode realized split over {total} samples: {frac}")

    @staticmethod
    def _apply_outcome_gate(
        modes: tuple[str, ...], weights: list[float] | None, outcome: str | None
    ) -> list[float] | None:
        """Zero the ``policy`` sampling weight for ``failure`` episodes.

        A failed trajectory must not train the policy expert (it would imitate
        failing behaviour), but it is still valid supervision for the world model
        (``forward_dynamics``), ``inverse_dynamics`` and ``cross_view_inpainting``.
        ``recovery`` and ``success`` roll normally (recovering from a bad state IS
        a policy skill worth imitating). ``outcome=None`` (demonstration data with
        no outcome label) is a no-op, so demonstration-only mixtures are unaffected.
        """
        if outcome != "failure":
            return weights
        base = list(weights) if weights is not None else [1.0] * len(modes)
        return [0.0 if m == "policy" else w for m, w in zip(modes, base)]

    def _choose_mode(self, outcome: str | None = None) -> str:
        """Resolve the active mode for one sample request.

        For ``mode == "joint"`` we sample one joint sub-mode per request from
        :attr:`_JOINT_MODES`. Sampling frequencies come from ``joint_mode_weights``
        (default uniform); the SC3-Eval FD-heavy recipe is ``[0.8, 0.1, 0.1]``.
        ``outcome`` (per-episode failure/success/recovery label) gates the policy
        expert off for failures via :meth:`_apply_outcome_gate`.
        """
        if self._mode != "joint":
            # cosmos-framework's inference tools name the policy mode "wam".
            return MODE_ALIASES.get(self._mode, self._mode)
        weights = self._apply_outcome_gate(self._JOINT_MODES, self._joint_mode_weights, outcome)
        mode = random.choices(self._JOINT_MODES, weights=weights, k=1)[0]
        self._mode_counts[mode] += 1
        self._maybe_log_mode_counts()
        return mode

    # -- universal bad-sample memory (all loaders) --------------------------
    def _registered_shard_root(self, dataset_idx: int) -> str | None:
        """Resolve the worker-local index through its actual registered sources."""
        if dataset_idx is None or dataset_idx < 0:
            return None
        try:
            return self._dataset_build_args[dataset_idx]["root"]
        except (IndexError, AttributeError, TypeError, KeyError):
            return None

    def _shard_key(self, dataset_idx: int) -> str | None:
        """Stable per-shard key = last two path components of the shard root
        (mount- and dataset_idx-independent), matching the committed known-bad key space."""
        root = self._registered_shard_root(dataset_idx)
        if root is None:
            return None
        parts = str(root).rstrip("/").split("/")
        return "/".join(parts[-2:]) if len(parts) >= 2 else str(root)

    def _mark_bad(self, dataset_idx: int, row_idx: int) -> None:
        """Remember ``row_idx`` of this shard as bad for the rest of this run (per worker)."""
        key = self._shard_key(dataset_idx)
        if key is None:
            return
        self._session_bad_by_shard_key.setdefault(key, set()).add(int(row_idx))

    def _is_remembered_bad(self, dataset_idx: int, row_idx: int) -> bool:
        """True iff this (shard, row) already failed this run -> skip it without reloading."""
        mem = getattr(self, "_session_bad_by_shard_key", None)
        if not mem:
            return False
        key = self._shard_key(dataset_idx)
        return key is not None and int(row_idx) in mem.get(key, ())

    def _remember_bad_sample(self, idx: int, exc: BaseException | None = None) -> None:
        """Base fallback for loaders WITHOUT their own ``_log_bad_sample`` (e.g. CMR):
        resolve + remember so the window is never resampled this run. Loaders that also
        want the JSONL audit override ``_log_bad_sample`` (dvrk/suturebot)."""
        if exc is not None and not is_persistent_sample_failure(exc):
            return
        try:
            dataset_idx, row_idx, _, _ = self._resolve_index(idx)
        except Exception:
            return
        self._mark_bad(dataset_idx, row_idx)

    @staticmethod
    def _get_tabular_item(ds: LeRobotDataset, row_idx: int) -> dict[str, Any]:
        """Fetch actions/state/task without opening any video decoder.

        This is intentionally a short-lived metadata view on the worker-local
        ``LeRobotDataset``. DataLoader workers call ``__getitem__`` serially, and the
        ``finally`` block restores the live-video view before returning or raising.
        Video keys must also be removed from ``delta_indices``; otherwise LeRobot's
        tabular query would try to read the absent video column from Arrow.
        """
        original_features = ds.meta.info["features"]
        original_delta_indices = ds.delta_indices
        video_keys = {key for key, value in original_features.items() if value.get("dtype") == "video"}
        ds.meta.info["features"] = retain_requested_video_features(original_features, set())
        if original_delta_indices is not None:
            ds.delta_indices = {key: value for key, value in original_delta_indices.items() if key not in video_keys}
        try:
            return ds[row_idx]
        finally:
            ds.meta.info["features"] = original_features
            ds.delta_indices = original_delta_indices

    def _fetch_sample(
        self,
        idx: int,
        *,
        decode_video: bool = True,
        mode_override: str | None = None,
    ) -> tuple[str, int, int, dict[str, Any]]:
        """Resolve index, pick a mode, and load the sample from the dataset.

        Returns ``(mode, dataset_idx, row_idx, sample_dict)``.
        """
        # Resolve first so we know the episode: a failure/fail episode gates the policy
        # expert off (outcome='failure'); demonstration data (no failure split) -> None -> no-op.
        dataset_idx, row_idx, episode_id, frame_offset = self._resolve_index(idx)
        # Universal in-session skip: if this (shard, row) already failed this run, don't
        # pay to reload it — raise a tolerance error so the subclass retry loop resamples.
        if self._is_remembered_bad(dataset_idx, row_idx):
            raise AssertionError("tolerance: session-remembered-bad sample")
        # Camera-motion window reject (data-driven, off unless the shard has a mask):
        # this window spans the actual future query horizon in source frames. If it
        # overlaps a pan past the threshold, resample another window from the SAME
        # (good) episode. Pure index math — no decode — so effectively free. Remembered
        # bad so the same idx is never re-tried this run.
        _cam = self._camera_motion_segments.get(dataset_idx)
        if _cam:
            _segs = _cam.get(episode_id)
            if _segs and should_reject_window(
                _segs,
                frame_offset * self._sample_stride,
                self._dataset_build_args[dataset_idx]["future_frame_offset"] + 1,
                self._camera_overlap_max,
            ):
                self._mark_bad(dataset_idx, row_idx)
                raise AssertionError("tolerance: camera panning within window")
        outcome = None
        if getattr(self, "_failure_episodes", None) and episode_id in self._failure_episodes.get(dataset_idx, ()):
            outcome = "failure"
        mode = mode_override if mode_override is not None else self._choose_mode(outcome)

        try:
            dataset = self._get_dataset(dataset_idx)
            sample = dataset[row_idx] if decode_video else self._get_tabular_item(dataset, row_idx)
        except Exception as exc:
            # Undecodable/missing window -> remember so we never retry it this run.
            if is_persistent_sample_failure(exc):
                self._mark_bad(dataset_idx, row_idx)
            raise

        # Mode-conditional clean video (dots only for policy): swap the clean twin in
        # for forward/inverse dynamics; policy/CVI keep the annotated frames. No-op
        # unless the shard exposes a "<VIDEO_KEY>_clean" video (added by the repack).
        _vkey = getattr(self, "VIDEO_KEY", None)
        if decode_video and _vkey is not None and not (self._skip_video_loading or self._latent_cache_active):
            apply_mode_conditional_video(
                sample,
                mode,
                _vkey,
                f"{_vkey}_clean",
                video_source=self._video_source,
            )

        if self._skip_video_loading or self._latent_cache_active:
            sample = defaultdict(lambda: None, sample)
        elif decode_video and self._duplicate_frame_max_fraction is not None:
            # Reject frozen / re-encoded windows (near-identical consecutive frames
            # even after the 10 Hz downsample). Raised inside the subclass retry
            # loop's try -> resamples another window. Uses the instance's single
            # VIDEO_KEY when present (all dVRK/CMR loaders define one).
            vkey = getattr(self, "VIDEO_KEY", None)
            vid = sample.get(vkey) if vkey is not None else None
            if (
                vid is not None
                and frame_duplicate_fraction(vid, self._duplicate_frame_eps) > self._duplicate_frame_max_fraction
            ):
                self._mark_bad(dataset_idx, row_idx)
                raise AssertionError("tolerance: duplicate frames within window (frozen/re-encoded)")

        # Mode-conditional coarse dropout: occlude coarse regions for the action-prediction
        # experts (IDM/policy) so they don't over-rely on any fixed region; NEVER for
        # forward_dynamics (video = generation target) or CVI. After the frozen-video check
        # (so it sees real pixels); live path only (no-op under latent cache / skip video).
        if (
            self._coarse_dropout is not None
            and mode in _COARSE_DROPOUT_MODES
            and not (self._skip_video_loading or self._latent_cache_active)
        ):
            _cdk = getattr(self, "VIDEO_KEY", None)
            _cdv = sample.get(_cdk) if _cdk is not None else None
            if _cdv is not None:
                sample[_cdk] = apply_coarse_dropout(_cdv, self._coarse_dropout)

        return mode, dataset_idx, row_idx, sample

    # -- action normalization ------------------------------------------------

    def _normalizer_filename(self) -> str:
        """Bundled stats filename for this dataset instance.

        Default convention (matches ``compute_action_stats.py`` output):
        ``<embodiment_type>[_<pose_convention>][_<rotation_format>].json``.

        Pose/rotation suffixes are appended only when the instance actually
        has them (SE(3) pose datasets like Bridge / DROID).  Joint-space
        datasets — where both are ``None`` — resolve to just
        ``<embodiment_type>.json``.

        Subclasses may override when the bundled filename uses a different
        scheme (e.g. UMI's ``uva_umi_single_task_normalizer.json``).
        """
        if not self._embodiment_type:
            raise RuntimeError(
                f"{self.__class__.__name__}: embodiment_type is not set; cannot resolve normalizer filename."
            )
        parts = [self._embodiment_type]
        if self._pose_convention:
            parts.append(self._pose_convention)
        if self._rotation_format:
            parts.append(self._rotation_format)
        return "_".join(parts) + ".json"

    def _normalizer_path(self) -> Path:
        """Full path to the bundled stats JSON for this dataset."""
        return self._NORMALIZERS_DIR / self._normalizer_filename()

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        """Lazy-load action normalization stats (once per worker process).

        Raises :class:`FileNotFoundError` if the stats file is missing.  This
        is intentional — silently falling back to identity normalization when
        the user asked for ``quantile`` / ``quantile_rot`` / ``meanstd`` /
        ``minmax`` would be a training bug.
        """
        if self._norm_stats is not None:
            return self._norm_stats
        stats_key = "global_raw" if self._action_normalization == "quantile_rot" else "global"
        raw = load_action_stats(str(self._normalizer_path()), stats_key=stats_key)
        self._norm_stats = {}
        for key, value in raw.items():
            self._norm_stats[key] = torch.from_numpy(value).float()  # [D]
        return self._norm_stats

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalization, or return the raw action.

        - ``action_normalization=None`` → pass-through (used by viewer / debug)
        - ``"quantile"``   → ``2·(x − q01) / (q99 − q01) − 1`` clamped to [-1, 1]
        - ``"quantile_rot"`` → same as ``"quantile"``, but using ``global_raw``
          stats so rotation dimensions are normalized too.
        - ``"meanstd"``    → ``(x − mean) / std``
        - ``"minmax"``     → ``2·(x − min) / (max − min) − 1`` clamped to [-1, 1]
        """
        if self._action_normalization is None:
            return action
        method = "quantile" if self._action_normalization == "quantile_rot" else self._action_normalization
        normalized_action = normalize_action(
            action,
            method,
            self._load_norm_stats(),
        )  # [T,D]
        return normalized_action

    # -- video formatting ----------------------------------------------------

    def _convert_video(self, video_tchw: torch.Tensor | None) -> torch.Tensor | None:
        """Convert LeRobot ``(T,C,H,W)`` float video to Action ``(C,T,H,W)`` uint8.

        Args:
            video_tchw: Raw floating-point video tensor in ``[0, 1]`` with
                LeRobot layout, or ``None``.  # [T,C,H,W] | None

        Returns:
            Action-formatted video tensor, or ``None``.  # [C,T,H,W] | None
        """
        if self._skip_video_loading or video_tchw is None:
            return None
        if video_tchw.ndim != 4:
            raise ValueError(
                f"{self.__class__.__name__}._convert_video expected video with shape [T,C,H,W], "
                f"got ndim={video_tchw.ndim}"
            )
        if not torch.is_floating_point(video_tchw):
            raise TypeError(
                f"{self.__class__.__name__}._convert_video expected floating-point video in [0, 1], "
                f"got dtype={video_tchw.dtype}"
            )
        video_min = video_tchw.amin()  # []
        video_max = video_tchw.amax()  # []
        if video_min.item() < 0.0 or video_max.item() > 1.0:
            raise ValueError(
                f"{self.__class__.__name__}._convert_video expected floating-point video in [0, 1], "
                f"got range=[{video_min.item():.6f}, {video_max.item():.6f}]"
            )
        formatted_video = (video_tchw * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)  # [C,T,H,W]
        if self._split == "train" and (self._image_aug_strength > 0.0 or self._image_aug_pixel_drop > 0.0):
            formatted_video = self._apply_train_image_aug(
                formatted_video, self._image_aug_strength, self._image_aug_pixel_drop
            )
        return formatted_video

    @staticmethod
    def _apply_train_image_aug(video_cthw: torch.Tensor, strength: float, pixel_drop: float) -> torch.Tensor:
        """Per-clip TRAIN image aug on a uint8 ``[C,T,H,W]`` RGB video.

        Color jitter (brightness/contrast/saturation/hue) uses the SAME factors
        across all T frames (no intra-clip flicker); optional light pixel dropout
        applies a per-clip binary mask zeroing ``pixel_drop`` fraction of the
        spatial grid (consistent across frames). NB the generative modes (FD/CVI)
        also treat the video as the target, so keep ``pixel_drop`` small.
        """
        import torchvision.transforms.functional as TF

        v = video_cthw.permute(1, 0, 2, 3).float().div_(255.0)  # [T,C,H,W] in [0,1]
        if strength > 0.0:
            s = float(strength)
            v = TF.adjust_brightness(v, random.uniform(1.0 - s, 1.0 + s))
            v = TF.adjust_contrast(v, random.uniform(1.0 - s, 1.0 + s))
            v = TF.adjust_saturation(v, random.uniform(1.0 - s, 1.0 + s))
            v = TF.adjust_hue(v, random.uniform(-s / 4.0, s / 4.0))
        if pixel_drop > 0.0:
            _, _, h, w = v.shape
            keep = (torch.rand(h, w, device=v.device) >= float(pixel_drop)).to(v.dtype)  # [H,W] 1=keep
            v = v * keep  # broadcast over T,C
        return v.clamp_(0.0, 1.0).mul_(255.0).to(torch.uint8).permute(1, 0, 2, 3)  # [C,T,H,W]

    # -- result building -----------------------------------------------------

    def _build_action_spec(self) -> ActionSpec | None:
        """Subclass override: declare this dataset's action layout.

        Called once per instance — the result is cached by ``self.action_spec``.
        Return ``None`` to skip spec-driven idle detection; in that case
        ``_compute_idle_frames`` will log a one-time warning and return
        ``None`` for every sample.
        """
        return None

    @cached_property
    def action_spec(self) -> ActionSpec | None:
        """Cached :class:`ActionSpec` from ``_build_action_spec``.

        Returns ``None`` when the subclass did not declare one; idle detection
        is then skipped (with a one-time warning) until the subclass overrides
        ``_build_action_spec``.
        """
        return self._build_action_spec()

    @cached_property
    def action_names(self) -> list[str] | None:
        spec = self.action_spec
        return spec.names if spec is not None else None

    # Idle-detection thresholds. Defined as **velocities** (per second) so the
    # same numeric value means the same physical motion across datasets with
    # different sampling rates; converted to per-frame at call time using
    # ``self._fps`` via :meth:`_resolve_idle_thresholds`.
    #
    # Defaults:
    #   - ``idle_eps_t_per_sec``           = 5 mm/s   (≈ 1 mm/frame at 5 Hz)
    #   - ``idle_eps_r_per_sec``           = 1.5°/s   (geodesic, rotation-format aware)
    #   - ``idle_eps_g``                   = 1e-2     unit gripper Δ (no fps)
    #   - ``idle_joint_threshold_per_sec`` = 5e-3 rad/s
    #   - ``idle_min_streak``              = 3        require ≥ 3 consecutive
    #
    # Subclasses can either override the ``*_per_sec`` attributes (preferred —
    # keeps the velocity semantics) or set the corresponding ``idle_eps_*`` /
    # ``idle_joint_threshold`` attribute to a non-``None`` value to bypass the
    # per-fps conversion entirely (raw per-frame override).
    idle_eps_t_per_sec: float = 5e-3
    idle_eps_r_per_sec: float = math.radians(1.5)
    idle_eps_g: float = 1e-2
    idle_joint_threshold_per_sec: float = 5e-3
    idle_min_streak: int = 3

    # Optional per-frame overrides. ``None`` (default) → use the ``*_per_sec``
    # attribute / fps conversion above.
    idle_eps_t: float | None = None
    idle_eps_r: float | None = None
    idle_joint_threshold: float | None = None

    def _resolve_idle_thresholds(self) -> tuple[float, float, float, float]:
        """Resolve per-frame idle thresholds for this dataset instance.

        Returns ``(eps_t, eps_r, eps_g, joint_threshold)`` in raw per-frame
        units. Honours direct per-frame overrides if the subclass sets the
        non-``_per_sec`` attribute; otherwise scales the ``_per_sec`` values
        by ``self._fps``.
        """
        fps = float(self._fps) if self._fps else 1.0
        eps_t = self.idle_eps_t if self.idle_eps_t is not None else self.idle_eps_t_per_sec / fps
        eps_r = self.idle_eps_r if self.idle_eps_r is not None else self.idle_eps_r_per_sec / fps
        joint_thr = (
            self.idle_joint_threshold
            if self.idle_joint_threshold is not None
            else self.idle_joint_threshold_per_sec / fps
        )
        return float(eps_t), float(eps_r), float(self.idle_eps_g), float(joint_thr)

    def _compute_idle_frames(self, raw_action: torch.Tensor) -> torch.Tensor | None:
        """Count idle frames in the *raw* (un-normalized) action chunk.

        Requires ``self.action_spec`` to be declared via ``_build_action_spec``.
        Returns ``None`` when:
        - ``pose_convention`` is not ``"backward_framewise"`` (TODO: extend),
        - the subclass has not declared an ``ActionSpec`` (logs a one-time warning),
        - the action layout does not match the declared spec.

        Detection thresholds come from the ``idle_eps_*`` class attributes
        (overridable per dataset). Subclasses can also override this method
        outright, or pass an explicit ``idle_frames`` integer via
        ``**extras`` to :meth:`_build_result`.
        """

        # framewise -> ‖pos‖ from origin; anchored -> per-step displacement
        # (compute_idle_frames anchored=True diffs consecutive frames). Absolute /
        # None have no consistent zero-motion reference -> skip.
        _ANCHORED = ("backward_anchored", "backward_anchored_world_trans")
        if self._pose_convention not in ("backward_framewise",) + _ANCHORED:
            if not getattr(self, "_warned_pose_convention", False):
                log.warning(
                    f"Dataset {self.__class__.__name__}: pose_convention="
                    f"{self._pose_convention!r} has no framewise/anchored idle "
                    "semantics; skipping idle-frames detection."
                )
                self._warned_pose_convention = True
            return None
        anchored = self._pose_convention in _ANCHORED

        spec = self.action_spec
        if spec is None:
            if not getattr(self, "_warned_no_action_spec", False):
                log.warning(
                    f"Dataset {self.__class__.__name__} has no action spec defined; "
                    "skipping idle-frames detection. Override _build_action_spec() to enable it."
                )
                self._warned_no_action_spec = True
            return None

        eps_t, eps_r, eps_g, joint_thr = self._resolve_idle_thresholds()
        try:
            n = compute_idle_frames(
                raw_action,
                spec,
                eps_t=eps_t,
                eps_r=eps_r,
                eps_g=eps_g,
                joint_threshold=joint_thr,
                min_streak=self.idle_min_streak,
                anchored=anchored,
            )
        except (ValueError, TypeError) as e:
            if not getattr(self, "_warned_action_layout", False):
                log.warning(
                    f"Dataset {self.__class__.__name__}: action layout does "
                    f"not match the declared ActionSpec "
                    f"(action_dim={int(raw_action.shape[-1])}, "
                    f"spec.dim={spec.dim}); skipping idle-frames detection. "
                    f"Underlying error: {e}"
                )
                self._warned_action_layout = True
            return None
        return torch.tensor(n, dtype=torch.long)

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor | None,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        """Assemble the common return dict for ``__getitem__``.

        ``video`` is expected in raw LeRobot layout before final formatting.
        Subclasses may pass extra keys (e.g. ``initial_pose``) via ``**extras``.
        ``idle_frames`` is auto-computed from the raw (un-normalized) ``action``
        whenever the dataset's pose/rotation conventions allow it; subclasses
        can override by passing ``idle_frames`` (int or scalar tensor) via
        ``**extras``.
        """
        # Compute idle_frames from the raw action before normalization, unless
        # the subclass has provided one explicitly via ``**extras``.
        if "idle_frames" not in extras:
            idle_frames = self._compute_idle_frames(action)
            if idle_frames is not None:
                extras = {"idle_frames": idle_frames, **extras}

        normalized_action = self._normalize_action(action)  # [T,D]
        if self._skip_video_loading:
            result: dict[str, Any] = {"action": normalized_action}
            if "idle_frames" in extras:
                result["idle_frames"] = extras["idle_frames"]
            return result
        formatted_video = self._convert_video(video)  # [C,T,H,W] | None
        result = {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "conditioning_fps_action": torch.tensor(self._action_fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            **extras,
        }
        # Mono-FD sampling (see __init__): mark FD samples that should condition only
        # on the frame-0 condition tiles. The transform turns this into
        # SequencePlan.condition_views_first_frame; the packer builds the hybrid mask.
        if mode == "forward_dynamics" and self._fd_mono_condition_prob > 0.0:
            if self._split == "train" and random.random() < self._fd_mono_condition_prob:
                result["condition_views_first_frame"] = list(self._fd_mono_condition_views)
        return result

    # -- latent cache --------------------------------------------------------

    @staticmethod
    def _check_latent_cache_ready(latent_cache_root: str | None, latent_cache_name: str | None) -> bool:
        """Return True iff the latent cache is ready for this dataset.

        Reads the manifest at init time (cheap filesystem check) so that
        ``_get_dataset`` correctly decides whether to strip video keys.
        Falls back to live encoding (False) when:
          - ``latent_cache_root`` is None (no cache configured)
          - manifest.json doesn't exist yet (precompute still running)
          - this dataset's name isn't in the manifest yet (partially done precompute)
        Logs a one-time warning so operators know which datasets are falling back.
        """
        if latent_cache_root is None or latent_cache_name is None:
            return False
        from cosmos3_h_embodied.data._unsupported import read_manifest

        manifest = read_manifest(latent_cache_root)
        if manifest is None:
            log.warning(
                f"Latent cache configured for {latent_cache_name!r} but manifest not found "
                f"under {latent_cache_root!r} — precompute still in progress. "
                "Falling back to live video decode + VAE encode for this dataset."
            )
            return False
        if latent_cache_name not in manifest.get("datasets", {}):
            log.warning(
                f"Latent cache manifest found but {latent_cache_name!r} not in it yet "
                f"(precompute still running). Falling back to live encode for this dataset."
            )
            return False
        return True

    def _latent_cache_reader(self, root: str | None = None):
        """Lazily build a per-worker latent-cache reader (safe_open not fork-safe).

        One reader is cached per root so the aug-variant roots (see
        ``latent_cache_variant_roots``) each keep their own mmap LRU.
        """
        from cosmos3_h_embodied.data._unsupported import LatentCacheReader

        root = root or self._latent_cache_root
        if self._latent_readers is None:
            self._latent_readers = {}
        reader = self._latent_readers.get(root)
        if reader is None:
            reader = LatentCacheReader(
                root,
                self._latent_cache_name,
                expected_version_hash=self._latent_cache_version_hash,
            )
            self._latent_readers[root] = reader
        return reader

    def _pick_latent_cache_root(self) -> str:
        """Choose which precomputed cache root (original or a jittered variant) to read.

        Color jitter cannot be applied live when training off a latent cache (the
        cache is only valid because the pixel pipeline is deterministic), so the
        augmentation is precomputed as K variant roots. We sample one uniformly per
        access on the TRAIN split, giving K+1 discrete color renditions per window.
        Val/test always read the un-augmented original root so eval stays fixed.
        """
        if not self._latent_cache_variant_roots or self._split != "train":
            return self._latent_cache_root
        return random.choice([self._latent_cache_root, *self._latent_cache_variant_roots])

    def _apply_policy_view_dropout(self, latent: torch.Tensor, mode: str) -> torch.Tensor:
        """Zero wrist-tile latents on POLICY samples (partial-observation training).

        Deployment may lose a wrist camera, and we also want a policy that still
        works from the endoscope alone (multiview-trained -> mono-inference). The
        existing mono-conditioning path (``fd_mono_condition_prob``) only covers
        forward_dynamics, so policy samples never saw a degraded canvas and
        mono inference was out-of-distribution. Here we blank wrist tiles directly
        on the ``[C,T,H,W]`` VAE latent, which (unlike pixel masking) is compatible
        with the precomputed latent cache.

        Disjoint draws on the TRAIN split only:
          * ``policy_view_dropout_mono_prob``   -> drop BOTH wrist tiles (endoscope only)
          * ``policy_view_dropout_single_prob`` -> drop ONE wrist tile (random side)
          * otherwise                           -> full canvas, untouched

        The wrist tiles are the dVRK multiview canvas's ``psm1``/``psm2`` bands
        (``cross_view.DVRK_MULTIVIEW_TILES``); ``endoscope`` is never dropped.
        """
        if mode != "policy" or self._split != "train":
            return latent
        p_mono = self._policy_view_dropout_mono_prob
        p_single = self._policy_view_dropout_single_prob
        if p_mono <= 0.0 and p_single <= 0.0:
            return latent

        r = random.random()
        if r < p_mono:
            drop_names = ("psm1", "psm2")
        elif r < p_mono + p_single:
            drop_names = (random.choice(["psm1", "psm2"]),)
        else:
            return latent

        from cosmos3_h_embodied.data._unsupported import DVRK_MULTIVIEW_TILES, resolve_tiles

        latent_h, latent_w = latent.shape[-2:]
        try:
            # dict: name -> (r0, r1, c0, c1) latent bounds
            bounds = resolve_tiles(latent_h, latent_w, DVRK_MULTIVIEW_TILES)
        except Exception as e:  # non-multiview canvas (or geometry mismatch) -> no-op
            log.warning(
                f"{self.__class__.__name__}: policy view dropout skipped (tile resolve failed "
                f"on latent {latent_h}x{latent_w}): {e}",
                rank0_only=True,
            )
            return latent

        latent = latent.clone()
        for name in drop_names:
            r0, r1, c0, c1 = bounds[name]
            latent[..., r0:r1, c0:c1] = 0.0
        return latent

    def _inject_cached_latent(self, result: dict[str, Any], idx: int) -> dict[str, Any]:
        """Replace the (never-decoded) video with the precomputed VAE latent.

        Sets ``result["video"]`` to the cached latent ``[C,T,H,W]`` (already
        crop-removed + normalized), ``result["image_size"]`` to the cached
        ``[4]`` tensor, and ``result["vision_is_latent"]=True`` so the transform
        skips resize and the model forward skips normalize+encode+crop. Raises on
        a miss — a full per-window precompute is expected to cover every index.
        """
        root = self._pick_latent_cache_root()
        got = self._latent_cache_reader(root).get(idx)
        if got is None:
            raise KeyError(
                f"{self.__class__.__name__}: latent-cache miss for "
                f"{self._latent_cache_name!r} idx={idx} in root {root!r}. Rebuild "
                "precompute_latents.py so the cache covers every valid window."
            )
        latent, image_size = got
        # Partial-observation training for POLICY samples (wrist-tile dropout).
        # Done here, in latent space, so the precomputed cache is still usable.
        latent = self._apply_policy_view_dropout(latent, str(result.get("mode", "")))
        result["video"] = latent
        result["image_size"] = image_size
        result["vision_is_latent"] = torch.tensor(True)
        return result

    def _idle_frac_for_mode(self, mode: Any) -> float | None:
        """Resolve the idle-filter fraction for this window's sampled ``mode``.

        ``_idle_filter_frac`` is either a single float (all modes) or a per-mode
        mapping (e.g. {policy: 0.3, forward_dynamics: 0.5, inverse_dynamics: 0.5}).
        Returns ``None`` (=> no idle filtering) when unset or the mode is absent.
        """
        f = self._idle_filter_frac
        if f is None:
            return None
        if isinstance(f, dict):
            return f.get(str(mode))
        return f

    def _reject_if_idle(self, idx: int, result: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        """TRAIN-only idle-window rejection sampling.

        If this window's action chunk is more than ``self._idle_filter_frac`` idle
        (``idle_frames`` count / ``chunk_length``), resample a different window via
        bounded recursion — de-biasing the policy away from the "hold still" default
        that makes it freeze at end-of-task-looking states. No-op unless the filter
        is enabled, we're on the train split, and an idle count is available.
        """
        if self._idle_filter_frac is None or self._split != "train" or depth >= 8:
            return result
        frac = self._idle_frac_for_mode(result.get("mode"))
        if frac is None:
            return result
        idle = result.get("idle_frames")
        if idle is None:
            return result
        idle_n = int(idle.item()) if torch.is_tensor(idle) else int(idle)
        if idle_n / max(1, self._chunk_length) <= frac:
            return result
        new_idx = random.randrange(self._num_valid_indices)
        return self.__getitem__(new_idx, depth + 1)  # type: ignore[call-arg]

    def __len__(self) -> int:
        return self._num_valid_indices
