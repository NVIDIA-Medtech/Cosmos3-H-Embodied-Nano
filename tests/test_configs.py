# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The registered experiments load, describe the documented mixtures, and accept per-experiment changes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NORMALIZERS = REPO / "cosmos3_h_embodied" / "data" / "normalizers"

_LOAD = """
import json
from cosmos_framework.inference.common.config import load_config
from cosmos3_h_embodied.configs.experiment_h16 import EVAL_DATASET_OVERRIDES, register_experiment

register_experiment("test_fd", families=["dvrk_hamlyn"], shard_across_workers=False, mode="forward_dynamics",
                    quaternion_order="wxyz", num_workers=2)
out = {}
for name, overrides in (("openh_h16_task_prompt", []), ("openh_h16_task_prompt_reference", []),
                        ("openh_h16_task_prompt", list(EVAL_DATASET_OVERRIDES)), ("test_fd", [])):
    c = load_config("cosmos3_h_embodied/configs/config.py", name, overrides=overrides)
    d = c.dataloader_train.dataloaders.action_data.dataloader.dataset
    out[name + ("/eval" if overrides else "")] = dict(
        families={e.name: dict(roots=list(e.dataset.roots), normalizer=e.dataset.normalizer_filename,
                               mode=e.dataset.mode, quaternion_order=e.dataset.get("quaternion_order"))
                  for e in d.list_of_datasets},
        video_augment=bool(d.video_augment), cuda_families=len(d.video_augment_cuda_families or []),
        shard_across_workers=d.shard_across_workers,
        num_workers=c.dataloader_train.dataloaders.action_data.dataloader.num_workers,
        model=c.model._target_.__name__ if hasattr(c.model._target_, "__name__") else str(c.model._target_),
    )
print("RESULT " + json.dumps(out))
"""


@pytest.fixture(scope="module")
def experiments() -> dict:
    env = dict(os.environ, OPENH_DATA_ROOT="/DATA", PYTHONPATH=str(REPO))
    res = subprocess.run([sys.executable, "-c", _LOAD], cwd=REPO, env=env, check=True, capture_output=True, text=True)
    return json.loads(next(line for line in res.stdout.splitlines() if line.startswith("RESULT "))[len("RESULT ") :])


def test_public_experiment_uses_the_hugging_face_mixture(experiments) -> None:
    exp = experiments["openh_h16_task_prompt"]
    assert len(exp["families"]) == 11
    assert all(r.startswith("/DATA/") for f in exp["families"].values() for r in f["roots"])
    assert exp["model"].endswith("HistoryOmniMoTModel")
    assert exp["video_augment"] and exp["cuda_families"] == 11


def test_reference_experiment_is_the_training_mixture(experiments) -> None:
    exp = experiments["openh_h16_task_prompt_reference"]
    assert len(exp["families"]) == 15
    assert sum(len(f["roots"]) for f in exp["families"].values()) == 55
    assert {n: f["quaternion_order"] for n, f in exp["families"].items() if f["quaternion_order"] == "wxyz"} == {
        "dvrk_ucsd": "wxyz",
        "dvrk_hamlyn": "wxyz",
    }


def test_every_family_has_its_normalizer(experiments) -> None:
    for name, family in experiments["openh_h16_task_prompt_reference"]["families"].items():
        assert (NORMALIZERS / family["normalizer"]).is_file(), name


def test_eval_overrides_disable_video_augmentation(experiments) -> None:
    """Evaluation on val/full conditions on and scores clean frames (``scripts/eval.py`` applies these overrides)."""
    exp = experiments["openh_h16_task_prompt/eval"]
    assert not exp["video_augment"] and exp["cuda_families"] == 0


def test_register_experiment_overrides_family_settings(experiments) -> None:
    exp = experiments["test_fd"]
    assert list(exp["families"]) == ["dvrk_hamlyn"]
    assert exp["families"]["dvrk_hamlyn"]["mode"] == "forward_dynamics"
    assert exp["families"]["dvrk_hamlyn"]["quaternion_order"] == "wxyz"  # also set by the family: no conflict
    assert exp["shard_across_workers"] is False and exp["num_workers"] == 2
