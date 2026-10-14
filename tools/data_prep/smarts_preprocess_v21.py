# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Assemble smarts' exploded scalar-pose schema into a standard v2.1 LeRobot
dataset so it can be converted to v3.0 and loaded by the generalized dVRK loader.

smarts stores every pose field as a separate scalar column
(action.psm1.pose.position.x, ...orientation.w, ...gripper). It ALSO has a
non-contiguous episode layout (episode indices are gapped: 0..15, 100.., max
7507). LeRobot's ``convert_data`` renumbers episodes 0..N-1 in path order while
``episodes_stats``/``episodes.jsonl`` keep the real (gapped) index → a
"Number of episodes is not the same" mismatch. So we RENUMBER episodes to a
contiguous 0..N-1 (in ascending-real-index order) across data, episodes.jsonl,
and the video symlinks so all three stay aligned.

Per source leaf we write, at DST:
  - data/case-{new//chunks:03d}/episode_{new:06d}.parquet — assembled action(16)
    + observation.state(16) = [psm1 pos3+quat_xyzw4+gripper, psm2 same], with
    episode_index renumbered to ``new`` and a fresh global ``index``.
  - videos/case-{new//chunks:03d}/{video_key}/episode_{new:06d}.mp4 — SYMLINK to
    the source video for the original episode.
  - meta/: info.json (features -> 16-D arrays + video keys), modality.json
    (quat-7 bimanual), episodes.jsonl (renumbered), tasks.jsonl copied.
Pose is taken from ``observation.cartesian_state`` for BOTH blocks (present in
every leaf incl joint-only stitch-coldcut); gripper from action (action block)
and observation.state (state block). episodes_stats.jsonl is synthesized after.

    python tools/data_prep/smarts_preprocess_v21.py --src <leaf> --dst <out leaf>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ARMS = ("psm1", "psm2")


def _scalar(t, name):
    a = np.asarray(t.column(name).to_pylist(), dtype=np.float32)
    return a.reshape(-1, 1) if a.ndim == 1 else a[:, :1]


def _assemble(t, prefix_pose, gripper_col):
    blocks = []
    for arm in _ARMS:
        pos = np.concatenate([_scalar(t, f"{prefix_pose}.{arm}.pose.position.{c}") for c in "xyz"], 1)
        quat = np.concatenate([_scalar(t, f"{prefix_pose}.{arm}.pose.orientation.{c}") for c in "xyzw"], 1)
        grip = _scalar(t, gripper_col.format(arm=arm))
        blocks.append(np.concatenate([pos, quat, grip], 1))  # 8
    return np.concatenate(blocks, 1).astype(np.float32)  # 16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = args.src.rstrip("/"), args.dst.rstrip("/")
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)

    info = json.load(open(os.path.join(src, "meta", "info.json")))
    chunks = int(info.get("chunks_size", 1000))
    vid_feats = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    vid_tmpl = info["video_path"]  # videos/case-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4

    # source episodes.jsonl -> {orig_ei: episode dict} (for task + original chunk)
    src_eps = {}
    with open(os.path.join(src, "meta", "episodes.jsonl")) as fh:
        for line in fh:
            e = json.loads(line)
            src_eps[e["episode_index"]] = e

    # sort source episode parquets by real episode_index (ascending)
    pfs = sorted(glob.glob(os.path.join(src, "data", "*", "*.parquet")))
    pairs = []
    for p in pfs:
        ei = pq.read_table(p, columns=["episode_index"]).column(0)[0].as_py()
        ei = ei[0] if isinstance(ei, list) else ei
        pairs.append((ei, p))
    pairs.sort(key=lambda x: x[0])

    global_index = 0
    new_eps = []
    for new_i, (orig_ei, p) in enumerate(pairs):
        t = pq.read_table(p)
        n = t.num_rows
        action = _assemble(t, "observation.cartesian_state", "action.{arm}.gripper")
        state = _assemble(t, "observation.cartesian_state", "observation.state.{arm}.gripper")
        new_chunk = new_i // chunks
        # Scalarize the copied scalar columns: the smarts source stores
        # timestamp/frame_index/task_index as 1-element list<double>/list<int>,
        # but info.json + LeRobot expect scalar float32/int64 — copying them
        # as-is makes LeRobot's cast_table_to_schema throw "Couldn't cast
        # list<double> to float" on the first sample fetch (leaf unloadable).
        def _scalar(name, np_dtype):
            a = np.asarray(t.column(name).to_pylist())
            if a.ndim > 1:
                a = a[:, 0]
            return pa.array(a.astype(np_dtype))

        # Regularize timestamp + frame_index to uniform fps per episode. The smarts
        # SurgSync source is event-sampled (irregular real-time dt 0.1-1.3s), but the
        # video mp4 is regular fps and LeRobot's timestamp-tolerance check reads the
        # PARQUET timestamp when video is skipped (latent-cache mode) — irregular
        # parquet ts then rejects every window ("256 consecutive samples failed").
        # frame_index/fps matches the regular video frames, so both paths agree.
        _fps = float(info.get("fps", 10.0)) or 10.0
        cols = {
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
            "action": pa.array(list(action), type=pa.list_(pa.float32())),
            "timestamp": pa.array((np.arange(n) / _fps).astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, new_i, dtype=np.int64)),
            "task_index": _scalar("task_index", np.int64),
            "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
        }
        global_index += n
        outp = os.path.join(dst, f"data/case-{new_chunk:03d}/episode_{new_i:06d}.parquet")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        pq.write_table(pa.table(cols), outp)

        # renamed video symlinks (new index/chunk -> original file)
        orig = src_eps.get(orig_ei, {})
        orig_chunk = orig.get("episode_chunk", orig_ei // chunks)
        for vk in vid_feats:
            src_vid = os.path.join(src, vid_tmpl.format(episode_chunk=orig_chunk, video_key=vk, episode_index=orig_ei))
            dst_vid = os.path.join(dst, vid_tmpl.format(episode_chunk=new_chunk, video_key=vk, episode_index=new_i))
            os.makedirs(os.path.dirname(dst_vid), exist_ok=True)
            if os.path.exists(src_vid) and not os.path.lexists(dst_vid):
                os.symlink(src_vid, dst_vid)
        # LeRobot v2.1 episodes.jsonl uses the PLURAL "tasks": [<str>, ...] key; the
        # smarts source stores it as singular "task". Downstream _filter_valid_episodes
        # (and LeRobot's own meta loader) read "tasks" — writing only "task" makes every
        # episode look prompt-less and the whole leaf gets dropped as "empty prompt".
        # Emit the plural list (fall back to source "tasks" if already plural).
        _task_str = orig.get("task") or ""
        if not _task_str and orig.get("tasks"):
            _task_str = orig["tasks"][0]
        new_eps.append({"episode_index": new_i, "episode_chunk": new_chunk,
                        "task_index": int(orig.get("task_index", 0)),
                        "tasks": [_task_str] if _task_str else [],
                        "task": _task_str, "length": n})

    # meta: episodes.jsonl (renumbered), tasks copied, info features rewritten, modality
    with open(os.path.join(dst, "meta", "episodes.jsonl"), "w") as fh:
        for e in new_eps:
            fh.write(json.dumps(e) + "\n")
    if os.path.exists(os.path.join(src, "meta", "tasks.jsonl")):
        shutil.copy(os.path.join(src, "meta", "tasks.jsonl"), os.path.join(dst, "meta", "tasks.jsonl"))
    names16 = [f"{a}_{c}" for a in _ARMS for c in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")]
    newF = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": names16},
        "action": {"dtype": "float32", "shape": [16], "names": names16},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        **{k: info["features"][k] for k in vid_feats},
    }
    info["features"] = newF
    info["total_episodes"] = len(new_eps)
    info["total_frames"] = int(global_index)
    json.dump(info, open(os.path.join(dst, "meta", "info.json"), "w"), indent=4)
    blk = {"psm1_pose": {"start": 0, "end": 7}, "psm1_gripper": {"start": 7, "end": 8},
           "psm2_pose": {"start": 8, "end": 15}, "psm2_gripper": {"start": 15, "end": 16}}
    vid_mod = {k.split(".")[-1]: {"original_key": k} for k in vid_feats}
    json.dump({"state": dict(blk), "action": dict(blk), "video": vid_mod},
              open(os.path.join(dst, "meta", "modality.json"), "w"), indent=2)
    print(f"preprocessed {src} -> {dst}: {len(new_eps)} episodes (renumbered 0..{len(new_eps)-1}), "
          f"{global_index} frames, {len(vid_feats)} video keys")


if __name__ == "__main__":
    main()
