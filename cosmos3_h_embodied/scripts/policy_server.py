# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HTTP policy server for the Open-H H16 checkpoint (policy mode with action history).

Endpoints:

- ``GET /``: health check.
- ``GET /info``: model and sampling settings.
- ``POST /predict``: policy inference.

  Request (JSON)::

      {
        "image": "<base64 PNG, current endoscope frame>",
        "prompt": "<caption text, e.g. caption context + task, as in training>",
        "domain_name": "<embodiment, e.g. hamlyn>",        # optional, server default otherwise
        "image_size": <int>,                                  # optional, see below
        "history_action": [[...], ...],                       # optional, [nH, raw_dim], nH <= 16
        "guidance": <float>, "num_steps": <int>, "seed": <int>,  # optional overrides
        "return_video": <bool>                                # optional, default true
      }

  ``history_action`` holds the last ``nH`` (at most 16) actions in the model's
  *normalized* space, oldest first, exactly as the training dataset produces them
  (``backward_anchored_world_trans`` + rot6d, normalized with the family's normalizer).
  They are prepended to the action sequence as clean conditioning rows, as in training.
  ``image_size`` (an int) first resizes the frame to that height, keeping the aspect
  ratio; the frame is then resized and reflection-padded to the training resolution
  bucket by the training transform.

  Response: ``{"action": [[...], ...] (T x raw_dim, denormalized), "video": [<base64 PNG>, ...]}``.
  With ``--no-denormalize`` actions are returned in the normalized space.

The batch is built by :func:`build_policy_batch` with the training transform
(:class:`~cosmos3_h_embodied.data.transforms.ActionTransformPipeline`), so resizing, padding,
caption metadata, action padding and the sequence plan match training.

Example::

    PYTHONPATH=. python -m cosmos3_h_embodied.scripts.policy_server \\
        --checkpoint-path <dcp dir>/model --config-file <exported config.yaml> \\
        --normalizer dvrk_hamlyn_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json \\
        --domain-name hamlyn --port 8000
"""

from __future__ import annotations

from cosmos_framework.inference.common.init import init_script

init_script()

import base64
import binascii
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import torch
import torchvision.transforms.functional as transforms_F
import tyro
from PIL import Image

from cosmos3_h_embodied.data.action_normalization import load_action_stats
from cosmos3_h_embodied.data.domain_utils import get_domain_id
from cosmos3_h_embodied.data.transforms import ActionTransformPipeline, remove_reflection_padding
from cosmos_framework.utils import log

# Training recipe values (H16 task-prompt recipe).
DEFAULT_FPS = 10
DEFAULT_ACTION_CHUNK_SIZE = 24
DEFAULT_MAX_ACTION_DIM = 64
MAX_HISTORY_ACTIONS = 16
NORMALIZERS_DIR = Path(__file__).resolve().parents[1] / "data" / "normalizers"


# ---------------------------------------------------------------------------
# Request -> batch (no model, no GPU)
# ---------------------------------------------------------------------------


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


def decode_base64_png(image_b64: str) -> torch.Tensor:
    """Base64 PNG (optionally a ``data:`` URL) -> RGB uint8 tensor ``[3, H, W]``."""
    if "," in image_b64 and image_b64[:64].lower().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_b64 + "=" * (-len(image_b64) % 4))
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 image: {e}") from e
    with Image.open(io.BytesIO(raw)) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]


def build_policy_batch(
    req: dict[str, Any],
    *,
    transform: ActionTransformPipeline,
    raw_action_dim: int,
    fps: int = DEFAULT_FPS,
    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE,
    default_domain_name: str | None = None,
    image: torch.Tensor | None = None,
    image_device: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the model data batch for one policy request.

    Args:
        req: The request (see the module docstring).
        transform: Transform from :func:`make_policy_transform`.
        raw_action_dim: Unpadded action width of the embodiment.
        fps: Video/action rate of the recipe (10 Hz).
        action_chunk_size: Number of future action rows to generate.
        default_domain_name: Embodiment used when the request has no ``domain_name``.
        image: Already decoded frame ``[3, H, W]`` uint8 (otherwise decoded from ``req["image"]``).
        image_device: Device of the batch's ``image_size`` tensor.

    Returns:
        The batch, and metadata (``num_history_actions``, ``image_size``, ``prompt``).
    """
    prompt = req.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("'prompt' must be a string")
    domain_name = req.get("domain_name", default_domain_name)
    if not isinstance(domain_name, str):
        raise ValueError("'domain_name' must be a string (or set a server default)")
    if image is None:
        if not isinstance(req.get("image"), str):
            raise ValueError("'image' must be a base64 string")
        image = decode_base64_png(req["image"])

    image_size = req.get("image_size")
    if image_size is not None:
        if not (isinstance(image_size, int) and image_size > 0):
            raise ValueError("'image_size' must be a positive int")
        img_h, img_w = image.shape[-2:]
        if img_h != image_size:
            new_w = int(round(img_w * image_size / img_h))
            image = transforms_F.resize(
                image, size=[image_size, new_w], interpolation=transforms_F.InterpolationMode.BICUBIC, antialias=True
            ).contiguous()

    num_frames = action_chunk_size + 1
    sample: dict[str, Any] = {
        "video": image.unsqueeze(1).repeat(1, num_frames, 1, 1),  # [3,T,H,W]
        "action": torch.zeros((action_chunk_size, raw_action_dim), dtype=torch.float32),  # [T,D_raw]
        "ai_caption": prompt,
        "mode": "policy",
        "conditioning_fps": torch.tensor(fps, dtype=torch.long),
        "domain_id": torch.tensor(get_domain_id(domain_name), dtype=torch.long),
    }
    history = req.get("history_action")
    if history is not None:
        history_t = torch.as_tensor(history, dtype=torch.float32)
        if history_t.numel() == 0:
            history_t = history_t.reshape(0, raw_action_dim)
        if history_t.ndim != 2 or history_t.shape[1] != raw_action_dim:
            raise ValueError(f"'history_action' must be [nH, {raw_action_dim}], got {tuple(history_t.shape)}")
        if history_t.shape[0] > MAX_HISTORY_ACTIONS:
            raise ValueError(f"at most {MAX_HISTORY_ACTIONS} history actions, got {history_t.shape[0]}")
        if not torch.isfinite(history_t).all():
            raise ValueError("'history_action' must be finite")
        sample["history_action"] = history_t
    num_history = 0 if history is None else int(sample["history_action"].shape[0])

    out = transform(sample, resolution=None)
    batch: dict[str, Any] = {
        "video": [[out["video"]]],
        "raw_action_dim": [out["raw_action_dim"]],
        "action_processing_record": [out["action_processing_record"]],
        "action": [[out["action"]]],
        "mode": ["policy"],
        "ai_caption": [out["ai_caption"]],
        "prompt": [out["ai_caption"]],
        "conditioning_fps": [out["conditioning_fps"]],
        "image_size": out["image_size"].unsqueeze(0).to(device=image_device),
        "domain_id": [out["domain_id"]],
        "sequence_plan": [out["sequence_plan"]],
    }
    return batch, {"num_history_actions": num_history, "image_size": out["image_size"], "prompt": out["ai_caption"]}


class QuantileDenormalizer:
    """Inverse of the training ``quantile`` normalization (``global`` q01/q99 stats)."""

    def __init__(self, normalizer: str | Path) -> None:
        path = Path(normalizer)
        if not path.is_file():
            path = NORMALIZERS_DIR / str(normalizer)
        stats = load_action_stats(str(path), stats_key="global")
        self.path = path
        self.q01 = torch.as_tensor(stats["q01"], dtype=torch.float32)  # [D]
        self.range = (torch.as_tensor(stats["q99"], dtype=torch.float32) - self.q01).clamp(min=1e-8)  # [D]

    @property
    def dim(self) -> int:
        return int(self.q01.shape[0])

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1.0) / 2.0 * self.range.to(action.device) + self.q01.to(action.device)  # [...,D]

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        return 2.0 * (action - self.q01.to(action.device)) / self.range.to(action.device) - 1.0  # [...,D]


# ---------------------------------------------------------------------------
# Model service
# ---------------------------------------------------------------------------


class ServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint_path: str
    """Model checkpoint (DCP directory with ``*.distcp``, e.g. ``<converted>/model``)."""
    config_file: str
    """Training config: exported YAML, or ``cosmos3_h_embodied/configs/config.py`` with ``--experiment``."""
    experiment: str = ""
    """Experiment name when ``--config-file`` is a Python config module."""
    normalizer: str
    """Normalizer of the served embodiment (file name in the package's normalizers directory, or a path)."""
    domain_name: str | None = None
    """Default embodiment name for requests without ``domain_name``."""
    output_dir: Path = Path("outputs/policy_server")
    """Directory for the inference pipeline's config snapshot and logs."""
    seed: int = 0
    guidance: float = 1.0
    num_steps: int = 35
    fps: int = DEFAULT_FPS
    action_chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
    denormalize: bool = True
    """Return actions denormalized with ``--normalizer`` (else in the normalized model space)."""
    host: str = "0.0.0.0"
    port: int = 8000
    http_400_on_error: bool = False
    """Return HTTP 400 on errors (default: HTTP 200 with an empty action list)."""


class PolicyService:
    def __init__(self, args: ServerArgs) -> None:
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.scripts.action_policy_server_utils import (
            disable_runtime_ema_for_frozen_config,
            maybe_init_distributed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        maybe_init_distributed()
        overrides = OmniSetupOverrides.model_validate(
            dict(checkpoint_path=args.checkpoint_path, config_file=args.config_file, experiment=args.experiment)
        )
        overrides.output_dir = args.output_dir
        overrides.guardrails = False
        setup_args = disable_runtime_ema_for_frozen_config(overrides.build_setup())
        init_output_dir(setup_args.output_dir)
        self.pipe = OmniInference.create(setup_args)
        self.model = self.pipe.model
        self.model.eval()
        self.args = args
        self.normalizer = QuantileDenormalizer(args.normalizer)
        self.raw_action_dim = self.normalizer.dim
        self.max_action_dim = int(getattr(self.model.config, "max_action_dim", DEFAULT_MAX_ACTION_DIM))
        self.transform = make_policy_transform(self.max_action_dim)
        self._lock = threading.Lock()

    def info(self) -> dict[str, Any]:
        a = self.args
        return {
            "checkpoint": a.checkpoint_path,
            "config_file": a.config_file,
            "experiment": a.experiment,
            "normalizer": str(self.normalizer.path),
            "raw_action_dim": self.raw_action_dim,
            "max_history_actions": MAX_HISTORY_ACTIONS,
            "domain_name": a.domain_name,
            "fps": a.fps,
            "action_chunk_size": a.action_chunk_size,
            "seed": a.seed,
            "guidance": a.guidance,
            "num_steps": a.num_steps,
            "denormalize": a.denormalize,
        }

    def predict(self, req: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        batch, meta = build_policy_batch(
            req,
            transform=self.transform,
            raw_action_dim=self.raw_action_dim,
            fps=self.args.fps,
            action_chunk_size=self.args.action_chunk_size,
            default_domain_name=self.args.domain_name,
            image_device="cuda",
        )
        guidance = float(req.get("guidance", self.args.guidance))
        num_steps = int(req.get("num_steps", self.args.num_steps))
        seed = int(req.get("seed", self.args.seed))
        video_b64: list[str] = []
        with self._lock, torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                batch, guidance=guidance, seed=[seed], num_steps=num_steps, has_negative_prompt=False
            )
            action = samples["action"][0].float().squeeze(0)  # [nH+T,D] or [T,D]
            if req.get("return_video", True):
                video = self.model.decode(samples["vision"][0]).squeeze(0)  # [C,T,H,W]
                video = remove_reflection_padding(video, meta["image_size"])
                frames = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)  # [C,T,H,W]
                for t in range(frames.shape[1]):
                    buf = io.BytesIO()
                    Image.fromarray(frames[:, t].permute(1, 2, 0).cpu().numpy()).save(buf, format="PNG")
                    video_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        n_hist = meta["num_history_actions"]
        if n_hist and action.shape[0] == self.args.action_chunk_size + n_hist:
            action = action[n_hist:]  # generated rows only
        action = action[:, : self.raw_action_dim]
        if self.args.denormalize:
            action = self.normalizer.denormalize(action)
        log.info(
            f"[policy-server] history={n_hist} steps={num_steps} guidance={guidance} "
            f"ms={(time.monotonic() - t0) * 1000:.0f} prompt={meta['prompt']!r}"
        )
        return {"action": action.cpu().tolist(), "video": video_b64}


class _Handler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer  # type: ignore[assignment]

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        service: PolicyService = self.server.service  # type: ignore[attr-defined]
        if self.path == "/":
            self._send(200, {"status": "ok"})
        elif self.path == "/info":
            self._send(200, service.info())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        service: PolicyService = self.server.service  # type: ignore[attr-defined]
        if self.path not in ("/", "/predict"):
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("JSON body must be an object")
            self._send(200, service.predict(req))
        except Exception as e:  # noqa: BLE001 - report to the client
            traceback.print_exc()
            self._send(400 if service.args.http_400_on_error else 200, {"action": [], "error": str(e)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def main() -> None:
    args = tyro.cli(ServerArgs, description=__doc__)
    service = PolicyService(args)
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.service = service  # type: ignore[attr-defined]
    log.info(f"[policy-server] serving on http://{args.host}:{args.port} (GET /info, POST /predict)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
