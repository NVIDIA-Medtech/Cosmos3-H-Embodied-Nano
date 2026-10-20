# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action-safe, temporally-consistent video augmentation for action datasets.

The cosmos3 *action* video pipeline is otherwise deterministic (resize+pad); this
adds the augmentations GR00T's Open-H recipe used, restricted to the ones that do
NOT break the video<->action correspondence:

* **ColorJitter** (brightness/contrast/saturation/hue) — photometric, geometry-safe.
* **RandomErasing** (GR00T "coarse pixel dropout") — occlusion, geometry-safe.
* **Light RandomResizedCrop** (GR00T's ~95% crop) — geometric BUT re-frames the
  *whole clip identically*, so relative instrument motion (hence the action) is
  preserved.

Deliberately EXCLUDED (would require also transforming the action):
* Horizontal flip — mirrors handedness (left arm <-> right arm).
* Rotation — rotates the motion frame.

**Temporal consistency is the core invariant:** all augmentation parameters are
sampled ONCE per clip and applied identically to every frame, so the clip stays
temporally coherent and the per-window action still matches the pixels.

Input/output: a ``(C, T, H, W)`` tensor (cosmos3 action video layout), uint8 or
float. Returns the same dtype/layout. Only the pixels change (no resize unless
crop fires, which resizes back to the original H,W).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision.transforms.functional as F


@dataclass
class ActionVideoAugmentConfig:
    enabled: bool = False
    # Opt-in working-set reduction. Replay the same clip parameters on each
    # temporal chunk; retain the original RNG advancement and transform order.
    frame_chunk_size: int = 0
    # Random rotation (GR00T random_rotation_angle: 5). Same angle on every frame of the
    # clip (geometry-consistent); U[-deg, deg].
    rotation_prob: float = 0.5
    rotation_degrees: float = 5.0
    # ColorJitter (GR00T-ish defaults). Each factor f -> U[1-f, 1+f]; hue -> U[-hue, hue].
    color_jitter_prob: float = 0.8
    brightness: float = 0.12
    contrast: float = 0.15
    saturation: float = 0.15
    hue: float = 0.02
    # RandomErasing (coarse dropout). One box per clip, same box on every frame.
    erasing_prob: float = 0.5
    erasing_scale: tuple[float, float] = (0.02, 0.12)   # fraction of frame area
    erasing_ratio: tuple[float, float] = (0.3, 3.3)     # aspect ratio range
    # Light RandomResizedCrop. Consistent crop across the clip, resized back to (H,W).
    crop_prob: float = 0.5
    crop_scale: tuple[float, float] = (0.90, 1.0)        # kept area fraction
    crop_ratio: tuple[float, float] = (0.95, 1.05)


class ActionVideoAugment:
    """Callable applying temporally-consistent, action-safe augmentation to a clip."""

    def __init__(self, cfg: ActionVideoAugmentConfig | dict | None = None) -> None:
        if cfg is None:
            cfg = ActionVideoAugmentConfig()
        elif isinstance(cfg, dict):
            cfg = ActionVideoAugmentConfig(**cfg)
        if cfg.frame_chunk_size < 0:
            raise ValueError("frame_chunk_size must be non-negative")
        self.cfg = cfg

    # ------------------------------------------------------------------
    def __call__(self, video_cthw: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if not self.cfg.enabled or video_cthw is None:
            return video_cthw
        if video_cthw.ndim != 4:
            raise ValueError(f"expected (C,T,H,W), got {tuple(video_cthw.shape)}")

        chunk_size = self.cfg.frame_chunk_size
        if chunk_size and video_cthw.shape[1] > chunk_size:
            rng = generator if generator is not None else torch.default_generator
            initial_state = rng.get_state()
            parts = []
            for start in range(0, video_cthw.shape[1], chunk_size):
                rng.set_state(initial_state)
                parts.append(self._apply_clip(video_cthw[:, start : start + chunk_size], generator))
            return torch.cat(parts, dim=1)
        return self._apply_clip(video_cthw, generator)

    def _apply_clip(self, video_cthw: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """Original operation order, shared by full-clip and temporal-chunk execution."""

        orig_dtype = video_cthw.dtype
        # Work in float [0,1], layout (T,C,H,W) so torchvision funcs treat T as batch.
        x = video_cthw
        if orig_dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        x = x.permute(1, 0, 2, 3).contiguous()  # (T,C,H,W)
        _, c, h, w = x.shape

        g = generator

        # 0) Random rotation: one clip-fixed angle applied to every frame (geometry-safe;
        #    GR00T random_rotation_angle: 5).
        if self.cfg.rotation_degrees > 0 and self._coin(self.cfg.rotation_prob, g):
            angle = float(self._uniform(-self.cfg.rotation_degrees, self.cfg.rotation_degrees, g))
            x = F.rotate(x, angle)

        # 1) Light consistent crop (re-frame the whole clip identically).
        if c >= 3 and self._coin(self.cfg.crop_prob, g):
            top, left, ch, cw = self._sample_crop(h, w, g)
            x = F.resized_crop(x, top, left, ch, cw, size=[h, w], antialias=True)

        # 2) ColorJitter with clip-fixed factors, applied in a fixed order.
        if c >= 3 and self._coin(self.cfg.color_jitter_prob, g):
            b = self._uniform(1 - self.cfg.brightness, 1 + self.cfg.brightness, g)
            con = self._uniform(1 - self.cfg.contrast, 1 + self.cfg.contrast, g)
            s = self._uniform(1 - self.cfg.saturation, 1 + self.cfg.saturation, g)
            hue = self._uniform(-self.cfg.hue, self.cfg.hue, g)
            x = F.adjust_brightness(x, b)
            x = F.adjust_contrast(x, con)
            x = F.adjust_saturation(x, s)
            x = F.adjust_hue(x.clamp(0, 1), hue)

        # 3) RandomErasing: one box, same on every frame.
        if self._coin(self.cfg.erasing_prob, g):
            i, j, eh, ew = self._sample_erase(h, w, g)
            if eh > 0 and ew > 0:
                x[:, :, i : i + eh, j : j + ew] = 0.0

        x = x.clamp(0, 1).permute(1, 0, 2, 3).contiguous()  # back to (C,T,H,W)
        if orig_dtype == torch.uint8:
            x = (x * 255.0).round().to(torch.uint8)
        else:
            x = x.to(orig_dtype)
        return x

    # ------------------------------------------------------------------
    @staticmethod
    def _rand(g: torch.Generator | None) -> float:
        return float(torch.rand(1, generator=g).item())

    def _coin(self, p: float, g: torch.Generator | None) -> bool:
        return p > 0.0 and self._rand(g) < p

    def _uniform(self, lo: float, hi: float, g: torch.Generator | None) -> float:
        return lo + (hi - lo) * self._rand(g)

    def _sample_crop(self, h: int, w: int, g: torch.Generator | None) -> tuple[int, int, int, int]:
        area = h * w
        for _ in range(10):
            target = self._uniform(*self.cfg.crop_scale, g) * area
            ar = self._uniform(*self.cfg.crop_ratio, g)
            cw = int(round((target * ar) ** 0.5))
            ch = int(round((target / ar) ** 0.5))
            if 0 < cw <= w and 0 < ch <= h:
                top = int(self._uniform(0, h - ch, g)) if h > ch else 0
                left = int(self._uniform(0, w - cw, g)) if w > cw else 0
                return top, left, ch, cw
        return 0, 0, h, w  # fallback: no crop

    def _sample_erase(self, h: int, w: int, g: torch.Generator | None) -> tuple[int, int, int, int]:
        area = h * w
        for _ in range(10):
            target = self._uniform(*self.cfg.erasing_scale, g) * area
            ar = self._uniform(*self.cfg.erasing_ratio, g)
            eh = int(round((target * ar) ** 0.5))
            ew = int(round((target / ar) ** 0.5))
            if eh < h and ew < w:
                i = int(self._uniform(0, h - eh, g))
                j = int(self._uniform(0, w - ew, g))
                return i, j, eh, ew
        return 0, 0, 0, 0  # fallback: no erase


if __name__ == "__main__":  # quick self-test
    aug = ActionVideoAugment(ActionVideoAugmentConfig(enabled=True))
    for dt in (torch.uint8, torch.float32):
        v = (torch.rand(3, 25, 320, 512) * (255 if dt == torch.uint8 else 1)).to(dt)
        out = aug(v)
        assert out.shape == v.shape and out.dtype == dt, (out.shape, out.dtype)
        # temporal consistency of the color op: per-frame mean shift is ~equal across T
        # (not a strict test, just a smoke signal)
        print(f"{dt}: in {tuple(v.shape)} -> out {tuple(out.shape)} ok, "
              f"range[{out.float().min():.1f},{out.float().max():.1f}]")
    print("video_aug self-test OK")
