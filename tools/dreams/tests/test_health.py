# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Smoke gates must reject, rather than sanitize, nonfinite or missing gradients."""

import pytest
import torch
from cosmos3_h_embodied.dreams.training import SmokeHealthCallback
from cosmos_framework.model.generator.distillation.optimizer import PhaseOptimizer


def test_finite_gradient_gate() -> None:
    model = torch.nn.Linear(3, 2)
    model(torch.ones(1, 3)).sum().backward()
    gate = SmokeHealthCallback()
    gate.on_before_optimizer_step(model, None, None, None)
    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite gradients"):
        gate.on_before_optimizer_step(model, None, None, None)


def test_missing_gradients_are_not_a_training_pass() -> None:
    with pytest.raises(RuntimeError, match="no gradients"):
        SmokeHealthCallback().on_before_optimizer_step(torch.nn.Linear(3, 2), None, None, None)


class CriticPhaseModel(torch.nn.Linear):
    def get_optimizer_key(self, iteration: int) -> str:
        return "critic"


def test_sf_critic_gradients_outside_registered_model() -> None:
    model = CriticPhaseModel(3, 2)
    critic = torch.nn.Linear(3, 2)  # Intentionally not a registered model submodule.
    critic(torch.ones(1, 3)).sum().backward()
    optimizer = PhaseOptimizer({"critic": torch.optim.SGD(critic.parameters(), lr=0.01)})
    gate = SmokeHealthCallback()
    gate.on_before_optimizer_step(model, optimizer, None, None)
    critic.weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        gate.on_before_optimizer_step(model, optimizer, None, None)
