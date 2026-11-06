# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint semantics and explicit regular-net SF initialization."""

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from cosmos_framework.model.generator.distillation.self_forcing_dmd import SelfForcingDMDModel
from cosmos_framework.utils.callback import Callback

from cosmos3_h_embodied.dreams.contract import file_sha256, require_contract, stamp, write_json


class FailureMemoryCallback(Callback):
    """Best-effort local memory evidence even when the first training step fails."""

    iteration: int = 0

    def on_training_step_start(self, model: Any, data_batch: dict[str, Any], iteration: int = 0) -> None:
        self.iteration = iteration

    def on_exception(self, error: BaseException) -> None:
        try:
            row = {"iteration": self.iteration, "error_type": type(error).__name__, "error": str(error)}
            if torch.cuda.is_initialized():
                row.update(
                    device=torch.cuda.current_device(),
                    gpu=torch.cuda.get_device_name(),
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),
                    max_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            write_json(
                Path(self.config.job.path_local) / f"failure_rank{int(os.environ.get('RANK', '0')):02d}.json", row
            )
        except Exception as diagnostic_error:
            print(f"Could not record failure memory: {diagnostic_error}", flush=True)


class RegularNetSelfForcing(SelfForcingDMDModel):
    """Load student, fixed teacher and fresh critic from regular net, never EMA."""

    def _load_checkpoint_to_net(
        self, net: torch.nn.Module, ckpt_path: str, prefix: str = "net", credential_path: str | None = None
    ) -> None:
        require_contract(ckpt_path)
        super()._load_checkpoint_to_net(net, ckpt_path, prefix="net", credential_path=credential_path)


class ActionContractCallback(Callback):
    """Guard incoming semantics and stamp only successfully saved checkpoints."""

    config: Any
    trainer: Any

    def __init__(self, manifest: str, parent: str, teacher: str | None = None) -> None:
        super().__init__()
        self.manifest = manifest
        self.parent = parent
        self.teacher = teacher
        require_contract(parent)
        if teacher is not None:
            require_contract(teacher)

    def on_load_checkpoint_start(self, model: Any) -> None:
        if self.config.checkpoint.load_path:
            require_contract(self.config.checkpoint.load_path)

    def on_load_checkpoint_end(self, model: Any, iteration: int = 0, checkpoint_path: str | None = None) -> None:
        if checkpoint_path:
            receipt = require_contract(checkpoint_path)
            if self.config.checkpoint.load_training_state:
                provenance = receipt["provenance"]
                if provenance.get("manifest_sha256") != file_sha256(Path(self.manifest)):
                    raise ValueError("Same-stage resume requires the identical prepared manifest")
                if provenance.get("world_size") != int(os.environ.get("WORLD_SIZE", "1")):
                    raise ValueError("Same-stage resume requires the identical world size")
                if provenance.get("stage") != self.config.job.name:
                    raise ValueError("Full training-state resume cannot cross stages")
                if (
                    provenance.get("parent") != self.parent
                    or provenance.get("scoring_teacher_and_fresh_critic") != self.teacher
                ):
                    raise ValueError("Same-stage resume requires identical parent and scoring teacher")

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        root = Path(self.config.job.path_local) / "checkpoints" / f"iter_{iteration:09d}"
        if not (root / "model/.metadata").is_file():
            raise RuntimeError(f"Checkpoint success callback found no model metadata: {root}")
        stamp(
            root,
            {
                "iteration": iteration,
                "stage": self.config.job.name,
                "manifest_sha256": file_sha256(Path(self.manifest)),
                "parent": self.parent,
                "scoring_teacher_and_fresh_critic": self.teacher,
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "regular_net_only": True,
                "ema_enabled": False,
            },
        )


class SmokeHealthCallback(Callback):
    """Fail the smoke on nonfinite gradients and record per-rank runtime evidence."""

    started: float = 0.0
    config: Any

    def on_training_step_start(self, model: Any, data_batch: dict[str, Any], iteration: int = 0) -> None:
        self.started = time.monotonic()

    def on_before_optimizer_step(
        self, model: Any, optimizer: Any, scheduler: Any, grad_scaler: Any, iteration: int = 0
    ) -> None:
        finite = []
        # SF's critic is intentionally outside model.parameters(); inspect the
        # actual phase optimizer rather than accidentally checking only the student.
        parameters = (
            optimizer.parameters_for_key(model.get_optimizer_key(iteration))
            if hasattr(optimizer, "parameters_for_key")
            else model.parameters()
        )
        for parameter in parameters:
            if parameter.grad is not None:
                gradient = parameter.grad
                local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                finite.append(torch.isfinite(local).all())
        if not finite:
            raise RuntimeError("Smoke optimizer step has no gradients")
        flag = torch.stack(finite).all().to(torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        if not bool(flag):
            raise FloatingPointError(f"Nonfinite gradients at iteration {iteration}")

    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        local_loss = loss.to_local() if hasattr(loss, "to_local") else loss
        if not bool(torch.isfinite(local_loss).all()):
            raise FloatingPointError(f"Nonfinite loss at iteration {iteration}")
        root = Path(self.config.job.path_local)
        root.mkdir(parents=True, exist_ok=True)
        row = {
            "iteration": iteration,
            "loss": float(local_loss.detach().float().mean()),
            "seconds": time.monotonic() - self.started,
            "gpu": torch.cuda.get_device_name(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "delivered_draw": str(data_batch.get("_action_sample_index")),
            "source_key": str(data_batch.get("__key__")),
        }
        with (root / f"smoke_health_rank{int(os.environ.get('RANK', '0')):02d}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
