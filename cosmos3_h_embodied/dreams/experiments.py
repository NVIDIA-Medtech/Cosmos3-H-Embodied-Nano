# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Six concrete public-framework registrations for the exp15-style recipe.

Use COSMOS_DREAMS_SPEC to select a caller-owned JSON run specification. Config
composition is side-effect free; data/checkpoint validation happens at launch.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any

from cosmos_framework.configs.simulation.common.bimanual_profiles import make_bimanual_experiment
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from hydra.core.config_store import ConfigStore

from cosmos3_h_embodied.callbacks.action_dataloader_state import ActionDataLoaderStateCallback
from cosmos3_h_embodied.configs.experiment_h16 import h16_model_config
from cosmos3_h_embodied.dreams.contract import reference
from cosmos3_h_embodied.dreams.loader import make_loader
from cosmos3_h_embodied.dreams.training import (
    ActionContractCallback,
    FailureMemoryCallback,
    RegularNetSelfForcing,
    SmokeHealthCallback,
)

STAGES = ("c24_a", "c24_b", "c72", "tf", "sf4", "sf2")


def merge(target: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def experiment(stage: str, spec: dict[str, Any]) -> LazyDict:
    """Resolve the shipped numeric reference into runnable framework settings."""
    ref = reference()
    recipe = next(s for s in ref["stages"] if s["name"] == stage)
    sf = stage in ("sf4", "sf2")
    node = make_bimanual_experiment(3 if sf else 1 if stage == "tf" else 0)
    cfg = h16_model_config()
    merge(cfg, ref["common_model_config"])
    merge(cfg, recipe["model_config"])
    # The public framework replaced this old boolean with an explicit mode.
    averaging = cfg["rectified_flow_training_config"].pop("sample_level_loss_averaging", False)
    if averaging:
        raise ValueError("This recipe requires the original local-item loss reduction")
    cfg["rectified_flow_training_config"]["loss_reduction_mode"] = "local_item_mean"
    cfg["ema"] = {"enabled": False}
    cfg.update(
        action_gen=True, sound_gen=False, action_schema="native", action_global_asinh=False, action_metric_gripper=False
    )
    cfg["parallelism"].update(
        data_parallel_shard_degree=spec["world_size"],
        context_parallel_shard_degree=1,
        fsdp_reduce_dtype=recipe["fsdp_reduce_dtype"] or "bfloat16",
    )
    cfg["compile"] = {"enabled": False}
    cfg["tokenizer"].update(vae_path=spec["vae"], bucket_name="", object_store_credential_path_pretrained=None)
    cfg["vlm_config"]["pretrained_weights"] = {"enabled": False, "backbone_path": "", "credentials_path": ""}
    cfg["vlm_config"]["use_system_prompt"] = False
    cfg["vlm_config"]["tokenizer"].update(config_variant="hf", pretrained_model_name=spec["text_tokenizer"])
    parent = spec["parents"][stage]
    teacher = spec["teacher"] if sf else None
    if sf:
        cfg.update(
            vlm_config_teacher=copy.deepcopy(cfg["vlm_config"]),
            vlm_config_fake_score=copy.deepcopy(cfg["vlm_config"]),
            teacher_load_from={"load_path": teacher + "/model", "credentials": None},
            student_load_from={"load_path": parent + "/model", "credentials": None},
        )
    steps = spec["steps"].get(stage, recipe["max_iter"])
    manifest = spec["manifests"][stage]
    node["job"] = {"project": "cosmos3_h_dreams", "group": spec["run_name"], "name": stage, "wandb_mode": "disabled"}
    node["model"] = {"config": cfg}
    if sf:
        node["model"]["_target_"] = RegularNetSelfForcing
    # Override the bimanual schema/EMA/default backbone; preserve only framework
    # infrastructure selections (model/trainer/checkpointer/optimizer classes).
    for item in node["defaults"]:
        if isinstance(item, dict) and "override /vlm_config" in item:
            item["override /vlm_config"] = None
    node["optimizer"] = copy.deepcopy(ref["training"]["sf_optimizer" if sf else "teacher_tf_optimizer"])
    node["scheduler"] = (
        {} if sf else dict(f_max=[1.0], f_min=[1.0], warm_up_steps=[0], cycle_lengths=[steps], verbosity_interval=0)
    )
    node["trainer"].update(
        max_iter=steps,
        logging_iter=1,
        grad_accum_iter=1,
        run_validation=False,
        run_validation_on_start=False,
        save_zero_checkpoint=False,
        seed=28003,
        distributed_parallelism="fsdp",
        grad_scaler_args={"enabled": False},
    )
    node["trainer"]["callbacks"] = {
        "failure_memory": L(FailureMemoryCallback)(),
        "action_dataloader_state": L(ActionDataLoaderStateCallback)(),
        "action_contract": L(ActionContractCallback)(manifest=manifest, parent=parent, teacher=teacher),
        "grad_clip": {"clip_norm": 1.0, "force_finite": False},
        "manual_gc": {"every_n": 5},
        "skip_nan_step": {"max_consecutive_nan": 0},
    }
    if spec.get("purpose") == "integration_only_not_convergence_or_reproduction":
        # Insert before clipping so qualification cannot mask nonfinite gradients.
        node["trainer"]["callbacks"] = {"smoke_health": L(SmokeHealthCallback)(), **node["trainer"]["callbacks"]}
    node["checkpoint"] = dict(
        load_path=spec.get("resume", {}).get(stage) or ("" if sf else parent),
        load_training_state=bool(spec.get("resume", {}).get(stage)),
        strict_resume=True,
        load_ema_to_reg=False,
        keys_not_to_resume=[],
        keys_to_skip_loading=[],
        save_iter=spec.get("save_every", {}).get(stage, steps),
        broadcast_via_filesystem=True,
        type={"disable_async": True},
    )
    node["dataloader_train"] = L(make_loader)(
        manifest=manifest,
        data_root=spec["data_root"],
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        split="train",
    )
    node["dataloader_val"] = None
    node["model_parallel"] = {"cpu_offloading_activations": False}
    node["upload_reproducible_setup"] = False
    return LazyDict(node, flags={"allow_objects": True})


def register() -> None:
    spec_path = os.environ.get("COSMOS_DREAMS_SPEC")
    if spec_path is None:
        raise ValueError("Set COSMOS_DREAMS_SPEC to the run.json written by the smoke driver")
    spec = json.loads(Path(spec_path).read_text())
    for stage in STAGES:
        ConfigStore.instance().store(
            group="experiment", package="_global_", name=f"chole_{stage}", node=experiment(stage, spec)
        )


register()
