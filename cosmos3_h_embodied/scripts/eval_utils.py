# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-sample metrics for action evaluation (video PSNR, action MSE on predicted rows)."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.utils import log
from cosmos3_h_embodied.data.transforms import remove_reflection_padding
from cosmos_framework.inference.metrics import compute_action_mse, compute_psnr, compute_ssim
from cosmos_framework.inference.common.args import SampleOutputs
from cosmos_framework.inference.vision import read_media_frames

VIDEO_MODES = {"forward_dynamics"}
ACTION_MODES = {"inverse_dynamics"}
# "wam" is cosmos-framework's name for the policy mode.
BOTH_MODES = {"policy", "wam"}
ALL_MODES = VIDEO_MODES | ACTION_MODES | BOTH_MODES
_LPIPS_MODEL = None


def _compute_lpips(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Mean AlexNet LPIPS over video frames (needs the ``lpips`` package)."""
    global _LPIPS_MODEL

    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").eval().to(device)
    gt_frames = gt.permute(1, 0, 2, 3).float().div(127.5).sub(1.0)
    pred_frames = pred.permute(1, 0, 2, 3).float().div(127.5).sub(1.0)
    values = []
    with torch.no_grad():
        for start in range(0, len(gt_frames), 16):
            values.append(
                _LPIPS_MODEL(
                    gt_frames[start : start + 16].to(device),
                    pred_frames[start : start + 16].to(device),
                )
                .detach()
                .float()
                .cpu()
            )
    return float(torch.cat(values).mean().item())


def extract_gt_video(data_batch: dict) -> torch.Tensor | None:
    """Snapshot the GT video as (C, T, H, W) uint8, trimmed to its content region if padded.

    Must be called BEFORE the inference pipeline runs — the model normalizes
    `data_batch["video"]` in place from uint8 [0, 255] to float [-1, 1].
    """
    video = data_batch.get("video")
    if video is None:
        return None
    gt_video = video[0].detach().clone()
    image_size = data_batch.get("image_size")
    if image_size is not None:
        gt_video = remove_reflection_padding(gt_video, image_size[0])
    return gt_video


def extract_gt_action(data_batch: dict) -> torch.Tensor | None:
    """Snapshot the GT action as a (T, D) float32 tensor, or None when absent."""
    action = data_batch.get("action", [None])[0]
    if action is None:
        return None

    raw_action_dim = data_batch.get("raw_action_dim", [None])[0]
    if raw_action_dim is not None:
        # If raw_action_dim is provided, it indicates that the GT action has been padded to a larger size.
        # We trim the action to its original dimension before returning it.
        raw_action_dim = int(raw_action_dim.item())  # remove batch dim and convert to int
        assert action.shape[-1] >= raw_action_dim, (
            f"invalid raw_action_dim={raw_action_dim} for action with shape {action.shape}"
        )
        action = action[..., :raw_action_dim]

    return action.detach().clone().float()


def extract_action_prediction_mask(data_batch: dict) -> torch.Tensor | None:
    """Snapshot which action rows are predictions rather than supplied context."""
    action = data_batch.get("action", [None])[0]
    if action is None:
        return None
    plans = data_batch.get("sequence_plan")
    if not plans:
        raise ValueError("action evaluation requires a sequence plan")
    plan = plans[0] if isinstance(plans, list) else plans
    mask = torch.ones(action.shape[0], dtype=torch.bool)
    conditioned = plan.condition_frame_indexes_action
    if conditioned:
        indexes = torch.as_tensor(conditioned, dtype=torch.long)
        if (indexes < 0).any() or (indexes >= len(mask)).any():
            raise ValueError("action conditioning index outside target trajectory")
        mask[indexes] = False
    return mask


def _parse_mode_from_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 2:
        raise ValueError(f"unexpected sample name: {name!r}")
    mode = parts[-2]
    if mode not in ALL_MODES:
        raise ValueError(f"unexpected mode {mode!r} in sample name {name!r}; expected one of {sorted(ALL_MODES)}")
    return mode


def derive_match_key_and_group(pred_path: Path, predictions_dir: Path) -> tuple[str, str]:
    """Path → ``(match_key, group)``. Used by vision eval to pair predictions with GT.

    For ``inference.py``-style outputs (basename ``vision.*``), ``match_key`` is the
    parent directory name and ``group`` is the path between *predictions_dir* and
    that directory. Otherwise ``match_key`` is the filename stem.

    Examples (with ``predictions_dir=/root``):
        ``/root/t2v/episode_0/vision.mp4`` → ``("episode_0", "t2v")``
        ``/root/sub/foo.mp4`` → ``("foo", "sub")``
    """
    pred_path = pred_path.resolve()
    predictions_dir = predictions_dir.resolve()
    if not pred_path.is_relative_to(predictions_dir):
        raise ValueError(f"pred_path {pred_path} is not under predictions_dir {predictions_dir}")
    rel = pred_path.relative_to(predictions_dir)
    parts = rel.parts
    if pred_path.name.startswith("vision."):
        if len(parts) < 2:
            raise ValueError(f"expected <group>/<key>/vision.* under predictions_dir, got rel={rel}")
        match_key = parts[-2]
        group_parts = parts[:-2]
    else:
        match_key = pred_path.stem
        group_parts = parts[:-1]
    group = "/".join(group_parts)
    return match_key, group


def compute_video_metrics(
    gt_video_cthw_uint8: torch.Tensor,
    pred_path: Path,
    mode: str,
    full_metrics: bool = False,
) -> dict[str, float]:
    """Compute per-clip PSNR. Temporal-mismatch policy depends on *mode*:

    - ``"vision"``: lenient. If ``T_gt != T_pred`` (after the H/W top-left crop), trim both to
      ``min(T_gt, T_pred)`` from the start (VFM generation is aligned to ``4k+1`` frames —
      latent temporal factor 4 + 1 conditioning frame — while the on-disk GT keeps the raw
      clip length, so small T deltas (e.g. GT=96, pred=93) are expected and treating them as
      hard errors loses the entire eval) and log an info line.
    - any other mode (``forward_dynamics``, ``policy``, ...): strict. Pred T is fixed by
      the action chunk size; a mismatch indicates a real bug.

    Spatial (H/W) mismatch always errors — the existing top-left crop of pred to GT's
    H/W stays in place; mismatches that survive the crop indicate a config bug rather
    than an SFT-style trim.
    """
    # +1 so an over-long prediction surfaces as a shape mismatch instead of silent truncation.
    pred, _ = read_media_frames(pred_path, max_frames=gt_video_cthw_uint8.shape[1] + 1)
    # Match GT's spatial dims (top-left crop, mirroring remove_reflection_padding's convention)
    # so a reflection-padded GT trimmed to its content region can be compared against the
    # padded mp4 saved to disk.
    pred = pred[..., : gt_video_cthw_uint8.shape[-2], : gt_video_cthw_uint8.shape[-1]]
    gt = gt_video_cthw_uint8
    # Symmetric crop: if pred is *smaller* than GT (VAE latent-downsample factor 16 rounds
    # H/W down when the source isn't a multiple of 16, e.g. GT H=468 → pred H=464),
    # crop GT down to pred's H/W so PSNR is computed over the overlap region.
    if pred.shape[-2] < gt.shape[-2] or pred.shape[-1] < gt.shape[-1]:
        gt = gt[..., : pred.shape[-2], : pred.shape[-1]]

    if pred.shape != gt.shape:
        # Spatial mismatch after both crops is a hard error.
        if pred.shape[-2:] != gt.shape[-2:]:
            raise ValueError(f"video spatial mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)} ({pred_path})")
        # Temporal mismatch: lenient for vision eval, strict otherwise.
        if mode == "vision":
            min_t = min(gt.shape[1], pred.shape[1])
            log.info(
                f"vision frame-count mismatch trimmed to {min_t} (gt T={gt.shape[1]}, pred T={pred.shape[1]}, "
                f"{pred_path}); likely due to generation aligned to 4k+1 frames "
                "(latent temporal factor 4 + 1 conditioning frame), while the on-disk GT "
                "keeps the raw clip length."
            )
            gt = gt[:, :min_t]
            pred = pred[:, :min_t]
        else:
            raise ValueError(f"video shape mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)} ({pred_path})")

    metrics = {"psnr": compute_psnr(gt, pred)}
    if full_metrics:
        metrics["ssim"] = compute_ssim(gt, pred)
        metrics["lpips"] = _compute_lpips(gt, pred)
    return metrics


def _compute_action_metrics(gt_action_td: torch.Tensor, pred_action_list: list) -> dict[str, Any]:
    pred = torch.tensor(pred_action_list, dtype=torch.float32)
    if pred.shape != gt_action_td.shape:
        raise ValueError(f"action shape mismatch: gt {tuple(gt_action_td.shape)} vs pred {tuple(pred.shape)}")
    return {"action_mse": compute_action_mse(gt_action_td, pred)}


def _pad_video_for_yuv420p(video_cthw: torch.Tensor) -> torch.Tensor:
    """Replicate-pad the visualization copy to even H/W for H.264 yuv420p.

    Metrics continue to use the unmodified content crop. The one-pixel pad is
    only for the optional browser-compatible GT MP4 sidecar.
    """
    pad_h = video_cthw.shape[-2] % 2
    pad_w = video_cthw.shape[-1] % 2
    if not pad_h and not pad_w:
        return video_cthw
    return torch.nn.functional.pad(video_cthw, (0, pad_w, 0, pad_h), mode="replicate")


def compute_sample_metrics(
    name: str,
    gt_video_cthw: torch.Tensor | None,
    gt_action_td: torch.Tensor | None,
    sample_output: SampleOutputs,
    sample_dir: Path,
    vision_extension: str,
    action_prediction_mask: torch.Tensor | None = None,
    full_metrics: bool = False,
) -> dict[str, Any]:
    """Compute metrics for a single sample, dispatched by the mode parsed from `name`."""
    mode = _parse_mode_from_name(name)
    out: dict[str, Any] = {"mode": mode, "name": sample_dir.name}
    if mode in VIDEO_MODES | BOTH_MODES:
        if gt_video_cthw is None:
            raise ValueError(f"mode={mode!r} requires GT video but data_batch had none")
        out.update(
            compute_video_metrics(gt_video_cthw, sample_dir / f"vision{vision_extension}", mode, full_metrics)
        )
        # Dump the GT clip next to the prediction so viewers can render pred vs GT side by side.
        # Best-effort, as the action dump below.
        try:
            gt_path = sample_dir / "vision_gt.mp4"
            if not gt_path.exists():
                import torchvision.io as _tvio

                gt_for_video = _pad_video_for_yuv420p(gt_video_cthw)
                _tvio.write_video(
                    str(gt_path),
                    gt_for_video.permute(1, 2, 3, 0).contiguous().cpu(),  # CTHW -> THWC uint8
                    fps=10,
                    video_codec="h264",
                    options={"crf": "20", "pix_fmt": "yuv420p"},
                )
        except Exception as _exc:  # diagnostic only — don't fail eval over visualization data
            log.warning(f"failed to dump vision_gt.mp4 for {name}: {_exc}")
    if mode in ACTION_MODES | BOTH_MODES:
        pred_action = sample_output.outputs[0].content.get("action") if sample_output.outputs else None
        if pred_action is None:
            raise ValueError(f"mode={mode!r} requires predicted action but content has none")
        if gt_action_td is None:
            raise ValueError(f"mode={mode!r} requires GT action but data_batch had none")
        if action_prediction_mask is not None:
            pred_tensor = torch.as_tensor(pred_action, dtype=torch.float32)
            if pred_tensor.shape != gt_action_td.shape:
                raise ValueError("full action prediction/target shape mismatch before masking")
            mask = action_prediction_mask.cpu().bool()
            if mask.shape != (gt_action_td.shape[0],) or not mask.any():
                raise ValueError("action scoring requires a nonempty prediction mask")
            gt_action_td = gt_action_td[mask.to(gt_action_td.device)]
            pred_action = pred_tensor[mask].tolist()
            (sample_dir / "action_scoring.json").write_text(
                json.dumps(
                    {
                        "scope": "predicted_rows_valid_channels",
                        "predicted_rows": torch.where(mask)[0].tolist(),
                        "conditioning_rows_excluded": int((~mask).sum()),
                        "raw_action_dim": int(gt_action_td.shape[-1]),
                    },
                    indent=2,
                )
                + "\n"
            )
        out.update(_compute_action_metrics(gt_action_td, pred_action))
        # Dump pred+GT trajectories so downstream viewers can render side-by-side.
        try:
            (sample_dir / "pred_action.json").write_text(json.dumps(pred_action))
            (sample_dir / "gt_action.json").write_text(json.dumps(gt_action_td.detach().cpu().tolist()))
        except Exception as _exc:  # diagnostic only — don't fail eval over visualization data
            log.warning(f"failed to dump action trajectories for {name}: {_exc}")
    return out


def aggregate_metrics(output_dir: Path) -> dict[str, Any]:
    """Walk `output_dir` for per-sample `metrics.json` files; emit per-mode/metric summary.

    Each scalar metric is summarised as ``{mean, count}``.
    """
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in output_dir.rglob("metrics.json"):
        try:
            m = json.loads(f.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        mode = m.pop("mode", None)
        m.pop("name", None)
        if mode is None:
            continue
        for k, v in m.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    totals[mode][f"{k}/{sub_k}"].append(float(sub_v))
            else:
                totals[mode][k].append(float(v))
    return {
        mode: {metric: {"mean": float(np.mean(vals)), "count": len(vals)} for metric, vals in metrics.items()}
        for mode, metrics in totals.items()
    }
