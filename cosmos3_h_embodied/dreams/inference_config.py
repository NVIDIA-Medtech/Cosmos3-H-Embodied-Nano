# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Generate reloadable inference configurations without loading or training weights."""

import argparse
import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import attrs
from cosmos_framework.inference.common.distillation_export import (
    resolve_student_base_model,
    sanitize_student_model_config,
)
from cosmos_framework.inference.common.config import structure_config
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils.config import load_config
from cosmos_framework.utils.lazy_config import LazyConfig, instantiate
from cosmos_framework.utils.lazy_config.registry import convert_target_to_string
from omegaconf import DictConfig, OmegaConf

from cosmos3_h_embodied.dreams.contract import file_sha256, write_json


def validate_model_config(model: dict):
    """Use the native inference deserializer and construct only architecture config."""
    structured = structure_config(model, DictConfig)
    # This invokes Qwen3VLMoTConfig.from_json_file, not the network constructor.
    # Merely parsing YAML would miss the inherited-classmethod serialization bug.
    architecture = instantiate(structured.config.vlm_config.model_instance.config)
    # These properties materialize the concrete HF configuration. Constructing
    # the wrapper alone also succeeds for the broken abstract _MoTConfigBase.
    _ = architecture.full_config
    _ = architecture.text_config
    if architecture.include_visual:
        _ = architecture.vision_config
    return structured.config


def save_model_config(config: Path, output: Path) -> None:
    """Project a serialized training configuration onto one regular inference model.

    Works for bidirectional teachers, causal TF and fixed-step SF students. Does
    not instantiate a network or load checkpoints, optimizer, teacher or critic.
    """
    resolved = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    model = copy.deepcopy(resolved["model"])
    base_model, base_config = resolve_student_base_model(model, default_base_model=OmniMoTModel)
    sanitize_student_model_config(
        model,
        base_model_target=convert_target_to_string(base_model),
        base_config_type=convert_target_to_string(base_config),
        base_config_field_names={f.name for f in attrs.fields(base_config)},
    )
    model["config"]["parallelism"].update(data_parallel_shard_degree=1, context_parallel_shard_degree=1)
    # Instantiate only the configuration (including the architecture JSON), not the model.
    validate_model_config(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"model": model}), output)


def compose(spec: Path, stage: str, output: Path) -> None:
    os.environ["COSMOS_DREAMS_SPEC"] = str(spec.resolve())
    cfg = load_config(
        "cosmos_framework/configs/simulation/config.py",
        ["--", f"experiment=chole_{stage}"],
        experiment_module="cosmos3_h_embodied.dreams.experiments",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".inference-config-", dir=output.parent) as temporary:
        serialized = Path(temporary) / "training.yaml"
        LazyConfig.save_yaml(cfg, str(serialized))
        save_model_config(serialized, output)
    write_json(
        output.with_suffix(".receipt.json"),
        {
            "stage": stage,
            "spec_sha256": file_sha256(spec),
            "config_sha256": file_sha256(output),
            "weights_loaded": False,
            "config_instantiation": "passed",
            "inference_gpu_count": 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--stage", choices=["c24_a", "c24_b", "c72", "tf", "sf4", "sf2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compose(args.spec, args.stage, args.output)


if __name__ == "__main__":
    main()
