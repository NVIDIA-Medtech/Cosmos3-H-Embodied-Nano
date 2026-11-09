# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Portable native-rate CMR geometry, validity and video decoding.

Descriptors reference public-source files relative to a caller-owned data root.
Both v2.1 episode files and v3 multi-shard episodes use the same read path. No
private index, episode numbering convention, or source-tree import is required.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as functional
from scipy.spatial.transform import Rotation

from cosmos3_h_embodied.dreams.contract import STRIDE, reference

BUTTONS = (8, 9, 21, 22)
AXES = (11, 12, 24, 25)


def resolve(root: Path, relative: str) -> Path:
    """Resolve a portable path, rejecting absolute paths and directory traversal."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Expected a data-root-relative path: {relative}")
    return path


@lru_cache(maxsize=4)
def read_episode(root: str, files: tuple[str, ...], episode_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the complete episode across listed shards and validate native indexing."""
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for relative in files:
        table = pq.read_table(
            resolve(Path(root), relative),
            columns=["episode_index", "frame_index", "timestamp", "action", "observation.state"],
            use_threads=False,
        )
        ids = table["episode_index"].to_numpy()  # [N]
        keep = ids == episode_id  # [N]
        if not keep.any():
            continue
        a = np.asarray(table["action"].to_pylist(), dtype=np.float32)[keep]  # [L,100]
        s = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[keep]  # [L,100]
        t = np.asarray(table["timestamp"])[keep]  # [L]
        f = np.asarray(table["frame_index"])[keep]  # [L]
        parts.append((a, s, t, f))
    if not parts:
        raise ValueError(f"Episode {episode_id} absent from declared data shards")
    a, s, t, f = (np.concatenate(values) for values in zip(*parts, strict=True))  # [L,100],[L,100],[L],[L]
    if a.shape[1:] != (100,) or s.shape != a.shape or not np.array_equal(f, np.arange(len(f))):
        raise ValueError("Expected native 100D CMR arrays and complete contiguous episode-local frame indices")
    return a, s, t


def row_validity(action: np.ndarray, state: np.ndarray) -> np.ndarray:  # [L,100],[L,100] -> [L]
    """Validate native telemetry; camera magnitude is checked separately."""
    valid = np.isfinite(action).all(1) & np.isfinite(state).all(1)  # [L]
    valid &= (state[:, 16:18] >= 0.5).all(1)  # [L]
    # Native dumps use both Hook enum 8 and legacy Hook enum 11.
    valid &= (state[:, 22] == 0) & np.isin(state[:, 23], [8, 11])  # [L], Cadiere / Monopolar Hook
    valid &= np.isin(action[:, BUTTONS], [0, 1]).all(1)  # [L]
    for offset in (3, 16):
        valid &= np.linalg.norm(action[:, offset : offset + 4], axis=1) > 1e-8  # [L]
    return valid


def history_starts(
    action: np.ndarray, state: np.ndarray, times: np.ndarray, anchor: int
) -> np.ndarray:  # [L,100],[L,100],[L] -> [H]
    """Keep a contiguous valid suffix; never conceal camera activity by shortening H."""
    starts: list[int] = []
    for i in range(1, min(16, anchor // STRIDE) + 1):
        start = anchor - i * STRIDE
        stop = start + STRIDE + 1
        if not row_validity(action[start:stop], state[start:stop]).all():
            break
        if not np.equal(state[start:stop, [12, 13, 20, 21]], state[anchor, [12, 13, 20, 21]]).all():
            break
        interval = times[start:stop]  # [7]
        if not np.isfinite(interval).all() or np.any(np.abs(np.diff(interval) - 1 / 60) > 1e-4):
            break
        starts.append(start)
    return np.asarray(starts[::-1], dtype=np.int64)  # [H]


def window_reason(
    action: np.ndarray, state: np.ndarray, times: np.ndarray, anchor: int, future: int, threshold: float = 8000
) -> tuple[str | None, np.ndarray]:  # [L,100],[L,100],[L] -> reason,[H]
    """Return one rejection reason and true history for an inclusive native span."""
    empty = np.empty(0, dtype=np.int64)  # [0]
    end = anchor + future * STRIDE
    if anchor < 0 or end >= len(action):
        return "bounds", empty
    history = history_starts(action, state, times, anchor)  # [H]
    begin = int(history[0]) if len(history) else anchor
    span = slice(begin, end + 1)
    if not row_validity(action[span], state[span]).all():
        return "telemetry_or_engagement", history
    if not np.equal(state[span, [12, 13, 20, 21]], state[anchor, [12, 13, 20, 21]]).all():
        return "linkage_or_scaling", history
    if not np.isfinite(times[span]).all() or np.any(np.abs(np.diff(times[span]) - 1 / 60) > 1e-4):
        return "timestamps", history
    sticks = action[span][:, AXES].reshape(-1, 2, 2)  # [S,2,2]
    if np.any(np.linalg.norm(sticks, axis=-1) > threshold):
        return "camera", history
    return None, history


def encode(
    action: np.ndarray, state: np.ndarray, anchor: int, starts: np.ndarray
) -> np.ndarray:  # [L,100],[L,100],[T] -> [T,28]
    """Encode fixed-anchor poses, interval-start jaws/axes and interval-max buttons."""
    result = np.zeros((len(starts), 28), dtype=np.float32)  # [T,28]
    if not len(starts):
        return result
    if anchor < 0 or anchor >= len(state) or starts.min() < 0 or starts.max() + STRIDE >= len(state):
        raise ValueError("Action transition outside episode")
    span = slice(min(anchor, int(starts.min())), max(anchor, int(starts.max()) + STRIDE) + 1)
    if not row_validity(action[span], state[span]).all():
        raise ValueError("Invalid native telemetry or instrument engagement")
    if not np.equal(state[span, [12, 13, 20, 21]], state[anchor, [12, 13, 20, 21]]).all():
        raise ValueError("Linkage/scaling changed across transition")
    translation_scale, rotation_scale = state[anchor, [12, 13]]  # [2]
    translation_scale = translation_scale or 1.0
    for source, dest in ((0, 0), (13, 14)):
        endpoint = action[starts + STRIDE, source : source + 7]  # [T,7]
        origin = action[anchor, source : source + 7]  # [7]
        result[:, dest : dest + 3] = (endpoint[:, :3] - origin[:3]) * translation_scale  # [T,3]
        r0 = Rotation.from_quat(origin[3:]).as_matrix()  # [3,3]
        r1 = Rotation.from_quat(endpoint[:, 3:]).as_matrix()  # [T,3,3]
        relative = (r0.T @ r1).astype(np.float32)  # [T,3,3]
        scaled = Rotation.from_rotvec(
            Rotation.from_matrix(relative).as_rotvec() * rotation_scale
        ).as_matrix()  # [T,3,3]
        result[:, dest + 3 : dest + 9] = np.concatenate((scaled[:, :, 0], scaled[:, :, 1]), axis=-1)  # [T,6]
        result[:, dest + 9] = action[starts, source + 10]  # [T]
    rows = starts[:, None] + np.arange(STRIDE)  # [T,6]
    result[:, [10, 11, 24, 25]] = action[rows[:, :, None], np.asarray(BUTTONS)].max(1)  # [T,4]
    result[:, [12, 13, 26, 27]] = action[starts[:, None], np.asarray(AXES)]  # [T,4]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite encoded action")
    return result


def normalize(action: np.ndarray) -> np.ndarray:  # [T,28] -> [T,28]
    """Apply the shipped frozen reference affine transform exactly once."""
    stats = reference()["normalization"]
    lo, hi = (np.asarray(stats[k], dtype=np.float32) for k in ("q01", "q99"))  # [28],[28]
    return np.clip((action - (lo + hi) / 2) / (np.maximum(hi - lo, 1e-8) / 2), -1.5, 1.5)  # [T,28]


def decode(root: Path, episode: dict[str, Any], start: int, future: int) -> np.ndarray:  # -> [3,C+1,480,832]
    """PTS-align and letterbox video; fail instead of substituting a nearby frame."""
    first = float(episode["video_offset_seconds"]) + start / 60
    desired = first + np.arange(future + 1) / 10  # [C+1]
    frames: list[np.ndarray] = []
    with av.open(str(resolve(root, episode["video"]))) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(int(first / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame has no PTS")
            time = float(frame.pts * stream.time_base)
            if time < desired[len(frames)] - 0.5 / 60:
                continue
            if abs(time - desired[len(frames)]) > 0.5 / 60:
                raise ValueError(f"Video/telemetry timing mismatch: {time} vs {desired[len(frames)]}")
            rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float() / 255  # [3,H,W]
            height, width = rgb.shape[-2:]
            scale = min(480 / height, 832 / width)
            h2, w2 = round(height * scale), round(width * scale)
            rgb = functional.interpolate(rgb[None], size=(h2, w2), mode="bilinear", align_corners=False)[0]  # [3,H2,W2]
            rgb = functional.pad(
                rgb, ((832 - w2) // 2, 832 - w2 - (832 - w2) // 2, (480 - h2) // 2, 480 - h2 - (480 - h2) // 2)
            )  # [3,480,832]
            frames.append((rgb * 255).clamp(0, 255).to(torch.uint8).numpy())  # [3,480,832]
            if len(frames) == future + 1:
                break
    if len(frames) != future + 1:
        raise ValueError("Short decoded video")
    return np.stack(frames, axis=1)  # [3,C+1,480,832]
