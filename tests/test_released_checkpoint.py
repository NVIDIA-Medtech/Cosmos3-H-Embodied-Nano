# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The model built from the public config has exactly the released checkpoint's tensors.

Needs the released weights: ``H16_CHECKPOINT=<model_bf16.pt> pytest tests/test_released_checkpoint.py``.
Builds the model on the meta device (no GPU, no weights loaded into memory).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = os.environ.get("H16_CHECKPOINT")


@pytest.mark.skipif(not CHECKPOINT or not Path(CHECKPOINT).is_file(), reason="set H16_CHECKPOINT to the released .pt")
def test_network_matches_released_checkpoint(tmp_path) -> None:
    env = dict(os.environ, OPENH_DATA_ROOT="/DATA", PYTHONPATH=str(REPO))
    config = tmp_path / "h16.yaml"
    subprocess.run(
        [sys.executable, "-m", "cosmos_framework.scripts.export_config", "-o", str(config),
         "--config-file", "cosmos3_h_embodied/configs/config.py", "--experiment", "openh_h16_task_prompt"],
        cwd=REPO, env=env, check=True, capture_output=True,
    )  # fmt: skip
    env.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29561")
    out = subprocess.run(
        [sys.executable, str(REPO / "tests" / "scripts" / "network_keys.py"), str(config), CHECKPOINT],
        cwd=REPO, env=env, check=True, capture_output=True, text=True,
    )  # fmt: skip
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["model_class"] == "HistoryOmniMoTModel"
    assert result["only_model"] == [] and result["only_checkpoint"] == [] and result["shape_mismatches"] == []
    assert result["num_tensors"] == 809
