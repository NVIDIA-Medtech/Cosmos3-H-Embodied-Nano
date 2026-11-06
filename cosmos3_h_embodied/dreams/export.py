# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Export regular student tensors and a causal inference-only configuration."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from cosmos3_h_embodied.dreams.contract import contract_id, file_sha256, require_contract, stamp, write_json
from cosmos3_h_embodied.dreams.inference_config import save_model_config


def _export_into(source: Path, config: Path, output: Path) -> None:
    """Export net.* only; exclude the fixed teacher, critic, EMA and optimizer."""
    require_contract(source)
    torch.set_num_threads(1)
    reader = dcp.FileSystemReader(source / "model")
    meta = reader.read_metadata()
    selected = {key: value for key, value in meta.state_dict_metadata.items() if key.startswith("net.")}
    if not selected or any(not isinstance(m, TensorStorageMetadata) for m in selected.values()):
        raise ValueError("Expected tensor-only regular-net DCP")
    state = {
        key: torch.empty(value.size, dtype=value.properties.dtype) for key, value in selected.items()
    }  # dict[tensor]
    dcp.load(state, storage_reader=reader, no_dist=True)
    result = {}
    for key in list(state):
        tensor = state.pop(key)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite export: {key}")
        result[key.removeprefix("net.")] = (
            tensor.to(torch.bfloat16).contiguous() if tensor.is_floating_point() else tensor.contiguous()
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result, output / "model_bf16.pt")
    save_file(
        result, output / "model_bf16.safetensors", metadata={"contract_sha256": contract_id(), "weights": "regular_net"}
    )
    # The public native loader accepts HF bundles or DCP, not bare .pt files.
    # Re-read the exported payload to exercise the actual portable file on inference.
    restored = torch.load(output / "model_bf16.pt", map_location="cpu", weights_only=True, mmap=True)
    for key in result:
        torch.testing.assert_close(restored[key], result[key], rtol=0, atol=0)
    dcp.save({f"net.{k}": v for k, v in restored.items()}, checkpoint_id=output / "reload.dcp/model", no_dist=True)
    stamp(output / "reload.dcp", {"export_sha256": file_sha256(output / "model_bf16.pt"), "roundtrip": "exact"})
    # Use the released framework's projection helper so SF-only teacher/critic
    # fields cannot instantiate extra networks in downstream inference.
    save_model_config(config, output / "inference.yaml")
    write_json(output / "action_contract.json", json.loads((source / "action_contract.json").read_text()))
    write_json(
        output / "export.json",
        {
            "checkpoint_metadata_sha256": file_sha256(source / "model/.metadata"),
            "regular_weights": True,
            "dtype": "bfloat16",
            "tensor_count": len(result),
            "files": {
                name: file_sha256(output / name)
                for name in ("model_bf16.pt", "model_bf16.safetensors", "inference.yaml")
            },
            "history": "One anchored H0..H16 prefix per rollout; subsequent supplied actions enter causal context",
            "dependencies": "Pinned patched cosmos-framework and caller-owned tokenizer/VAE paths in inference.yaml",
        },
    )


def export(source: Path, config: Path, output: Path) -> None:
    """Publish a complete export atomically; failed attempts cannot poison a retry."""
    if output.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-export-", dir=output.parent) as temporary:
        staging = Path(temporary)
        _export_into(source, config, staging)
        staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
