# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned CMR action contracts and checkpoint provenance guards."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA = "cmr28-anchored-h16-interval-buttons-v1"
DOMAIN = 25
STRIDE = 6
OLD_TO_NEW = (*range(10), *range(14, 24))


def canonical(value: Any) -> bytes:
    """Serialize a receipt deterministically, independently of whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    """Hash a file without loading checkpoint/video payloads into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@lru_cache(maxsize=1)
def reference() -> dict[str, Any]:
    """Read the shipped numerical recipe, never a private prepared dataset."""
    return json.loads((Path(__file__).parent.parent / "exp15_distillation_reference.json").read_text())


def contract_id() -> str:
    """Bind schema, geometry and normalization to one semantic identity."""
    spec = reference()
    return hashlib.sha256(
        canonical({"schema": SCHEMA, "action": spec["action"], "normalization": spec["normalization"]})
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write a complete JSON artifact atomically in its destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stamp(root: Path, provenance: dict[str, Any]) -> None:
    """Attach semantic identity; callers must verify weights before stamping."""
    write_json(
        root / "action_contract.json", {"schema": SCHEMA, "contract_sha256": contract_id(), "provenance": provenance}
    )


def require_contract(root: str | Path) -> dict[str, Any]:
    """Reject absent/mismatched receipts even when padded weight shapes agree."""
    root = Path(root)
    if root.name == "model":
        root = root.parent
    receipt = json.loads((root / "action_contract.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("contract_sha256") != contract_id():
        raise ValueError(f"CMR28 semantic checkpoint contract mismatch: {root}")
    if not (root / "model/.metadata").is_file():
        raise ValueError(f"Incomplete model checkpoint: {root}")
    return receipt
