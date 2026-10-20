# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Simulate: roll out forward dynamics or the policy from a start frame, chunk by chunk.

- **Forward dynamics** (world model): the video for given actions. Each 2.4 s chunk (24 actions,
  24 frames at 10 Hz) is conditioned on the last generated frame of the previous chunk.
- **Policy**: the model predicts each chunk's actions and video from the last generated frame,
  the task text and its own previous actions (the policy acting in the model's imagined scene).

Actions are in the model's representation (see ``README_ACTION_SPACE.md``): per chunk, 24 rows
relative to the pose at the chunk's start, raw (normalized here with ``--normalizer``) or already
normalized (``--normalized``). The history of a chunk (up to 16 past actions, relative to the same
pose) is given for the first chunk (``--history``) and derived from the previous chunk's actions
for the next ones, which needs ``--normalizer``. Chunk ``k`` uses seed ``--seed + k``, as
``quant_eval.py``.

Inputs come from files (``--image``, ``--actions``, ``--history``, ``--caption``,
``--embodiment``), or from ``--inputs``, the ``inputs.npz`` that ``quant_eval.py --save-inputs``
writes for a recorded episode (with the recorded history of every chunk): replaying it
reproduces that rollout.

Writes ``rollout.npz`` (``video`` [T, H, W, 3] uint8 with the conditioning frame first;
``action`` [24 K, D] normalized, and ``action_raw`` with ``--normalizer``) and ``rollout.mp4``.

Example::

    python -m cosmos3_h_embodied.scripts.simulate --no-guardrails \\
        --checkpoint-path <dcp dir>/model --config-file cosmos3_h_embodied/configs/config.py \\
        --experiment openh_h16_task_prompt --num-steps 35 --guidance 6.0 --seed 0 \\
        --image frame.png --actions actions.npy --history history.npy \\
        --normalizer dvrk_hamlyn_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json \\
        --caption "<caption>" --embodiment hamlyn -o outputs/simulate
"""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_TRAINING": "1",
    }
)

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic
import torch
import torchvision.io
import tyro
from PIL import Image

from cosmos3_h_embodied.data.domain_utils import get_domain_id
from cosmos3_h_embodied.scripts.rollout_utils import (
    CHUNK,
    MAX_HISTORY_ACTIONS,
    RESOLUTION,
    QuantileDenormalizer,
    generate_chunk,
    make_policy_transform,
    make_raw_sample,
    reanchor_history,
)
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import SetupOverrides, tyro_cli
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.dataset_samples import MapDatasetSamples
from cosmos_framework.utils import log


class SimulateArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    setup: tyro.conf.OmitArgPrefixes[SetupOverrides] = OmniSetupOverrides.model_construct()
    """Model, sampling (``--num-steps``, ``--guidance``, ``--seed``) and output directory (``-o``)."""
    mode: Literal["forward_dynamics", "policy"] = "forward_dynamics"
    """``forward_dynamics``: video for the given actions; ``policy``: actions and video."""
    inputs: Path | None = None
    """``inputs.npz`` from ``quant_eval.py --save-inputs`` (replaces the inputs below)."""
    image: Path | None = None
    """Start frame (PNG or JPEG), the endoscope image as recorded."""
    actions: Path | None = None
    """Forward dynamics: ``.npy`` or ``.json``, ``[K, 24, D]`` or ``[24 K, D]``, per chunk relative to its start."""
    history: Path | None = None
    """``.npy`` or ``.json``, ``[nH, D]`` (``nH <= 16``, oldest first): the first chunk's action history."""
    normalized: bool = False
    """``--actions`` and ``--history`` are already normalized."""
    normalizer: str | None = None
    """Normalizer of the embodiment (file name in ``data/normalizers`` or a path)."""
    caption: str | None = None
    """Caption: robot and setting context + task, as in training."""
    embodiment: str | None = None
    """Embodiment name (e.g. ``hamlyn``; see ``data/domain_utils.py``)."""
    num_chunks: int = 3
    """Policy: chunks to roll out (forward dynamics: from the actions)."""
    save_video: bool = True
    """Write ``rollout.mp4``."""


def _load_array(path: Path) -> np.ndarray:
    data = np.load(path) if path.suffix == ".npy" else np.asarray(json.loads(path.read_text()))
    return np.asarray(data, dtype=np.float32)


def _load_inputs(args: SimulateArgs, normalizer: QuantileDenormalizer | None) -> dict:
    """``{"frame", "action" [K,24,D] | None, "history" [list of [nH,D]], "prompt" [K], "domain_id", "seed"}``."""
    if args.inputs is not None:
        npz = np.load(args.inputs)
        num_chunks = len(npz["action"]) if args.mode == "forward_dynamics" else args.num_chunks
        history = [h[MAX_HISTORY_ACTIONS - n :] for h, n in zip(npz["history"], npz["history_length"])]
        prompts = [str(p) for p in npz["prompt"]]
        return {
            "frame": torch.from_numpy(npz["frame"]).permute(2, 0, 1).contiguous(),
            "action": npz["action"] if args.mode == "forward_dynamics" else None,
            "history": history if args.mode == "forward_dynamics" else history[:1],
            "prompt": prompts + [prompts[-1]] * (num_chunks - len(prompts)),
            "domain_id": int(npz["domain_id"]),
            "seed": int(npz["seed"]),
        }
    if args.image is None or args.caption is None or args.embodiment is None:
        raise ValueError("give --inputs, or --image, --caption and --embodiment")
    with Image.open(args.image) as img:
        frame = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)

    def model_space(x: np.ndarray) -> np.ndarray:
        if args.normalized:
            return x
        if normalizer is None:
            raise ValueError("raw actions need --normalizer (or pass --normalized)")
        return normalizer.normalize(torch.from_numpy(x)).numpy()

    action = None
    if args.mode == "forward_dynamics":
        if args.actions is None:
            raise ValueError("forward dynamics needs --actions")
        action = _load_array(args.actions)
        if action.ndim == 2:
            if len(action) % CHUNK:
                raise ValueError(f"--actions has {len(action)} rows, not a multiple of {CHUNK}")
            action = action.reshape(-1, CHUNK, action.shape[-1])
        action = model_space(action)
    history = [model_space(_load_array(args.history))] if args.history is not None else [None]
    num_chunks = len(action) if action is not None else args.num_chunks
    return {
        "frame": frame.contiguous(),
        "action": action,
        "history": history,
        "prompt": [args.caption] * num_chunks,
        "domain_id": get_domain_id(args.embodiment),
        "seed": None,
    }


def _next_history(action: np.ndarray, normalizer: QuantileDenormalizer | None) -> np.ndarray:
    """Normalized history of the next chunk from this chunk's normalized actions."""
    if normalizer is None:
        raise ValueError("deriving the next chunk's history needs --normalizer")
    raw = normalizer.denormalize(torch.from_numpy(np.asarray(action, dtype=np.float32))).numpy()
    return normalizer.normalize(torch.from_numpy(reanchor_history(raw))).numpy()


def simulate(args: SimulateArgs) -> None:
    setup = args.setup.build_setup()
    init_output_dir(setup.output_dir)
    register_checkpoints()
    normalizer = QuantileDenormalizer(args.normalizer) if args.normalizer else None
    inputs = _load_inputs(args, normalizer)
    num_chunks = len(inputs["prompt"])
    if inputs["action"] is not None:
        dim = inputs["action"].shape[-1]
    elif inputs["history"][0] is not None:
        dim = inputs["history"][0].shape[-1]
    elif normalizer is not None:
        dim = normalizer.dim
    else:
        raise ValueError("cannot infer the action width: give --normalizer")
    if normalizer is not None and normalizer.dim != dim:
        raise ValueError(f"the normalizer has {normalizer.dim} dimensions, the actions {dim}")
    seed = setup.sample_overrides.seed
    seed = seed if seed is not None else (inputs["seed"] if inputs["seed"] is not None else 0)

    pipe = setup.get_inference_cls().create(setup)
    max_action_dim = int(getattr(pipe.model.config, "max_action_dim", 64))
    # The datasets' sample construction (transform, caption metadata, sampling settings).
    samples = MapDatasetSamples(
        None,
        [args.mode],
        [],
        make_policy_transform(max_action_dim),
        RESOLUTION,
        "",
        setup.sample_overrides.model_dump(exclude_none=True),
    )
    frames, actions = [], []
    condition: torch.Tensor | None = None
    history = inputs["history"][0]
    for k in range(num_chunks):
        chunk_action = inputs["action"][k] if inputs["action"] is not None else None
        raw = make_raw_sample(
            inputs["frame"],
            inputs["prompt"][k],
            inputs["domain_id"],
            dim,
            mode=args.mode,
            action=None if chunk_action is None else torch.from_numpy(np.asarray(chunk_action)),
            history=None if history is None else torch.from_numpy(np.asarray(history)),
        )
        sample_args, data_batch = samples._make_sample_from_raw(raw, 0, args.mode)
        sample_args = sample_args.model_copy(
            update={"name": f"chunk{k}", "output_dir": setup.output_dir / f"chunk{k}", "seed": seed + k}
        ).build_sample(model_config=pipe.model_config)
        video, predicted = generate_chunk(pipe, sample_args, data_batch, condition)
        if args.mode == "policy":
            if predicted is None:
                raise RuntimeError("the model returned no actions")
            chunk_action = predicted[-CHUNK:, :dim].numpy()
        frames.append(video if k == 0 else video[:, 1:])
        actions.append(np.asarray(chunk_action, dtype=np.float32))
        condition = video[:, -1:]
        log.info(f"[simulate] chunk {k + 1}/{num_chunks} done (seed {seed + k})")
        if k + 1 < num_chunks:
            given = inputs["history"]
            history = given[k + 1] if len(given) > k + 1 else _next_history(chunk_action, normalizer)

    video = torch.cat(frames, dim=1).permute(1, 2, 3, 0).numpy()  # [T,H,W,3]
    action = np.concatenate(actions)
    out = {"video": video, "action": action}
    if normalizer is not None:
        out["action_raw"] = normalizer.denormalize(torch.from_numpy(action)).numpy()
    np.savez_compressed(setup.output_dir / "rollout.npz", **out)
    if args.save_video:
        clip = torch.from_numpy(video)
        pad_h, pad_w = clip.shape[1] % 2, clip.shape[2] % 2  # h264 needs even sizes
        if pad_h or pad_w:
            clip = torch.nn.functional.pad(clip.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode="replicate")
            clip = clip.permute(0, 2, 3, 1)
        torchvision.io.write_video(str(setup.output_dir / "rollout.mp4"), clip, fps=10, video_codec="h264", options={"crf": "18"})
    log.success(f"[simulate] {video.shape[0]} frames, {len(action)} actions -> {setup.output_dir}")


def main() -> None:
    simulate(tyro_cli(SimulateArgs, description=__doc__))


if __name__ == "__main__":
    main()
