# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""OmniMoT model with action-only active-row loss normalization."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.model.generator import omni_mot_model as _cf_omni_mot_model
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class HistoryOmniMoTModel(OmniMoTModel):
    """``OmniMoTModel`` whose action loss is averaged over the noised action rows only.

    With action history, a sample carries clean history rows (conditioning) in front of
    the noised future rows. CF averages the action flow-matching loss over all rows
    (``normalize_loss_by_active=False``), so at 16 history rows and 24 future rows the
    action loss is scaled by 24/40. With ``normalize_action_loss_by_active=True`` the
    action loss is ``sum / (noised rows * action dims)``, independent of history length,
    while the vision loss keeps CF's mean normalization. This is the normalization the
    released H16 checkpoint was trained with.

    Args:
        config: The CF model config.
        normalize_action_loss_by_active: Normalize the action loss by the noised rows.
            Has no effect when ``normalize_loss_by_active`` is already set globally.
    """

    def __init__(self, config: OmniMoTModelConfig, normalize_action_loss_by_active: bool = True) -> None:
        self.normalize_action_loss_by_active = bool(normalize_action_loss_by_active)
        super().__init__(config)

    def _compute_losses(self, *args: Any, **kwargs: Any) -> Any:
        if not self.normalize_action_loss_by_active:
            return super()._compute_losses(*args, **kwargs)
        with _active_normalized_action_loss(self.rectified_flow_action):
            return super()._compute_losses(*args, **kwargs)


@contextlib.contextmanager
def _active_normalized_action_loss(rectified_flow_action: Any) -> Iterator[None]:
    """Within the block, CF's action flow-matching loss call uses ``normalize_by_active=True``.

    ``OmniMoTModel._compute_losses`` calls the module-level ``compute_flow_matching_loss``
    once per modality; the action call is the one that passes ``raw_action_dim`` together
    with the action rectified flow. Only that call is changed.
    """
    original = _cf_omni_mot_model.compute_flow_matching_loss

    def compute_flow_matching_loss(*args: Any, **kwargs: Any) -> Any:
        if "raw_action_dim" in kwargs and kwargs.get("rectified_flow") is rectified_flow_action:
            kwargs["normalize_by_active"] = True
        return original(*args, **kwargs)

    _cf_omni_mot_model.compute_flow_matching_loss = compute_flow_matching_loss
    try:
        yield
    finally:
        _cf_omni_mot_model.compute_flow_matching_loss = original
