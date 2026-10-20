# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the model of an exported config on the meta device and compare its network keys with a checkpoint.

Usage: network_keys.py <exported config.yaml> <released .pt>  (prints JSON)
"""

from cosmos_framework.inference.common.init import init_script

init_script(training=True, env={"COSMOS_TRAINING": "1", "COSMOS_DEVICE": "meta"})

import contextlib
import json
import sys
from pathlib import Path

import hydra
import omegaconf
import torch

import cosmos_framework.model.generator.omni_mot_model as cf_omni_mot_model
from cosmos_framework.inference.common.config import ROOT_DIR, deserialize_config_dict, structure_config

overrides = [
    "model.config.load_vision_tokenizer=false",
    "model.config.ema.enabled=false",
    "model.config.compile.enabled=false",
    "model.config.parallelism.data_parallel_shard_degree=1",
]
cfg = omegaconf.OmegaConf.merge(deserialize_config_dict(Path(sys.argv[1])), omegaconf.OmegaConf.from_dotlist(overrides))
omegaconf.OmegaConf.resolve(cfg)
config = structure_config(cfg)
# FSDP wrapping does not change parameter names and cannot run on the meta device.
cf_omni_mot_model.parallelize_vfm_network = lambda net, **kwargs: net
with contextlib.chdir(ROOT_DIR.parent):
    model = hydra.utils.instantiate(config.model)
net = {k: list(v.shape) for k, v in model.net.state_dict().items()}
if sys.argv[2].endswith(".safetensors"):
    from safetensors import safe_open

    with safe_open(sys.argv[2], "pt") as f:
        ckpt = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
else:
    ckpt = {k: list(v.shape) for k, v in torch.load(sys.argv[2], map_location="cpu", mmap=True, weights_only=True).items()}
print(json.dumps({
    "model_class": type(model).__name__,
    "only_model": sorted(set(net) - set(ckpt)),
    "only_checkpoint": sorted(set(ckpt) - set(net)),
    "shape_mismatches": sorted(k for k in set(net) & set(ckpt) if net[k] != ckpt[k]),
    "num_tensors": len(net),
}))
