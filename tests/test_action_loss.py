# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Active-row normalization of the action loss (``HistoryOmniMoTModel``)."""

from __future__ import annotations

import inspect
import re

import cosmos_framework.model.generator.omni_mot_model as cf_omni_mot_model

from cosmos3_h_embodied.model.omni_mot_model import _active_normalized_action_loss


def test_cf_compute_losses_calls_action_loss_as_expected() -> None:
    """Guard: the wrapper relies on how cosmos-framework's ``_compute_losses`` calls the action loss."""
    source = inspect.getsource(cf_omni_mot_model.OmniMoTModel._compute_losses)
    calls = re.findall(r"compute_flow_matching_loss\((.*?)\n\s*\)", source, flags=re.S)
    action_calls = [c for c in calls if "raw_action_dim=" in c]
    assert len(action_calls) == 1, "expected exactly one action loss call passing raw_action_dim"
    assert "rectified_flow=self.rectified_flow_action" in action_calls[0]
    assert "normalize_by_active=normalize_by_active" in action_calls[0]


def test_only_the_action_loss_is_normalized_by_active_rows(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cf_omni_mot_model, "compute_flow_matching_loss", lambda **kw: calls.append(kw))
    action_rf, vision_rf = object(), object()
    cf = cf_omni_mot_model
    with _active_normalized_action_loss(action_rf):
        cf.compute_flow_matching_loss(rectified_flow=vision_rf, normalize_by_active=False)
        cf.compute_flow_matching_loss(rectified_flow=action_rf, raw_action_dim=[20], normalize_by_active=False)
    cf.compute_flow_matching_loss(rectified_flow=action_rf, raw_action_dim=[20], normalize_by_active=False)
    assert [c["normalize_by_active"] for c in calls] == [False, True, False]
