# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public workflows must not depend on a previous training run or private files."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos3_h_embodied.dreams import prepare, research, smoke
from cosmos3_h_embodied.dreams.contract import contract_id, file_sha256, stamp, write_json


def manifest(path, *, split="train"):
    write_json(
        path,
        {
            "contract_sha256": contract_id(),
            "review_status": "visually_reviewed_two_instruments",
            "episodes": [{"recording_id": "recording_a", "split": split}],
        },
    )


def research_spec(tmp_path):
    for directory in ("a", "b"):
        for future in (24, 72):
            manifest(tmp_path / directory / f"c{future}.json")
    path = research.initialize_research(tmp_path, 8, "research_run", tmp_path / "a", tmp_path / "b")
    return path, json.loads(path.read_text())


def fake_checkpoint(path, stage, spec):
    (path / "model").mkdir(parents=True)
    (path / "model/.metadata").write_text("test metadata")
    key = "c24_a" if stage == "initializer" else stage
    stamp(
        path,
        {
            "stage": stage,
            "world_size": spec["world_size"],
            "parent": spec["parents"][key],
            "manifest_sha256": file_sha256(Path(spec["manifests"][key])),
        },
    )


def test_research_budgets_and_explicit_handoff(tmp_path):
    _, spec = research_spec(tmp_path)
    assert list(spec["steps"].values()) == [2000, 2000, 4400, 1000, 300, 300]
    assert spec["manifests"]["c24_a"] != spec["manifests"]["c24_b"]
    research.validate_handoff(spec, "c24_a")
    with pytest.raises(ValueError, match="Evaluate and select"):
        research.validate_handoff(spec, "c24_b")


def test_selection_requires_validation_and_preserves_provenance(tmp_path):
    path, spec = research_spec(tmp_path)
    source = smoke.checkpoint(tmp_path, "research_run", "c24_a", 200)
    fake_checkpoint(source, "c24_a", spec)
    results = tmp_path / "review/results.json"
    write_json(results, {"cases": [{"mode": "recorded"}]})
    run = dict(
        status="completed",
        modes=["recorded"],
        split="test",
        checkpoint=str(source / "model"),
        checkpoint_metadata_sha256=file_sha256(source / "model/.metadata"),
    )
    write_json(results.parent / "evaluation.json", run)
    with pytest.raises(ValueError, match="validation results"):
        research.select(tmp_path, "c24_a", source, results, "reviewed")
    run["split"] = "val"
    write_json(results.parent / "evaluation.json", run)
    research.select(tmp_path, "c24_a", source, results, "Reviewed fixed validation windows")
    updated = json.loads(path.read_text())
    assert updated["parents"]["c24_b"] == str(source)
    research.validate_handoff(updated, "c24_b")
    results.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        research.validate_handoff(updated, "c24_b")


def test_research_driver_keeps_periodic_save_interval(tmp_path, monkeypatch):
    _, spec = research_spec(tmp_path)
    fake_checkpoint(Path(spec["parents"]["c24_a"]), "initializer", spec)
    captured = []

    def execute(command, log, **kwargs):
        active = json.loads((tmp_path / "active_run.json").read_text())
        captured.append(active)
        final = smoke.checkpoint(tmp_path, "research_run", "c24_a", 2000)
        fake_checkpoint(final, "c24_a", spec)

    monkeypatch.setattr(smoke, "execute", execute)
    smoke.run(tmp_path, stages=["c24_a"])
    assert len(captured) == 1
    assert captured[0]["steps"]["c24_a"] == 2000
    assert captured[0]["save_every"]["c24_a"] == 200


def test_smoke_keeps_resume_probe(tmp_path, monkeypatch):
    path = smoke.initialize(tmp_path, 8, "public_smoke")
    spec = json.loads(path.read_text())
    for file in set(spec["manifests"].values()):
        manifest(Path(file))
    fake_checkpoint(Path(spec["parents"]["c24_a"]), "initializer", spec)
    stops = []

    def execute(command, log, **kwargs):
        active = json.loads((tmp_path / "active_run.json").read_text())
        stop = active["steps"]["c24_a"]
        stops.append(stop)
        fake_checkpoint(smoke.checkpoint(tmp_path, "public_smoke", "c24_a", stop), "c24_a", spec)

    monkeypatch.setattr(smoke, "execute", execute)
    smoke.run(tmp_path, stages=["c24_a"])
    assert stops == [1, 2]


def test_inference_profile_cannot_train(tmp_path):
    smoke.initialize(tmp_path, 1, "released_evaluation", "inference")
    with pytest.raises(ValueError, match="cannot launch training"):
        smoke.run(tmp_path)


def test_preparation_does_not_replace_manifests(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "c24.json").write_text("existing")
    with pytest.raises(FileExistsError):
        prepare.prepare_to_directory(tmp_path, tmp_path / "selection.json", output)
    assert (output / "c24.json").read_text() == "existing"


@pytest.mark.parametrize("model_set, expected", [("sf4", 1), ("sf2", 1), ("students", 2)])
def test_inference_download_excludes_initializer(tmp_path, monkeypatch, model_set, expected):
    downloads = []

    def get(**kw):
        downloads.append(kw)
        path = Path(kw["local_dir"]) / kw["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        return str(path)

    def snapshot(**kw):
        path = Path(kw["local_dir"]) / "text_tokenizer/tokenizer.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    monkeypatch.setattr(prepare, "hf_hub_download", get)
    monkeypatch.setattr(prepare, "snapshot_download", snapshot)
    prepare.download_assets(tmp_path, model_set=model_set)
    assert not any("h16_50000" in d["filename"] for d in downloads)
    assert sum("exp15" in d["filename"] for d in downloads) == expected


def test_failure_memory_records_first_step_without_cuda(tmp_path, monkeypatch):
    import torch
    from cosmos3_h_embodied.dreams.training import FailureMemoryCallback

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "0")
    callback = FailureMemoryCallback()
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    callback.on_training_step_start(None, {}, iteration=1)
    callback.on_exception(RuntimeError("out of memory"))
    row = json.loads((tmp_path / "failure_rank00.json").read_text())
    assert row["iteration"] == 1 and row["error"] == "out of memory"


def test_released_conversion_reuses_only_verified_output(tmp_path):
    import torch
    from safetensors.torch import save_file
    from cosmos3_h_embodied.dreams.released import convert

    pin = prepare.assets()["students"]
    filename = next(f for f in pin["files"] if "_4-step_" in f)
    source = tmp_path / "assets/students" / filename
    source.parent.mkdir(parents=True)
    save_file({"tiny_weight": torch.ones(2, dtype=torch.bfloat16)}, source)
    write_json(
        tmp_path / "assets/receipt.json",
        {
            "students": {
                "revision": pin["revision"],
                "repo": pin["repository"],
                "files": [{"path": str(source.relative_to(tmp_path)), "sha256": file_sha256(source)}],
            }
        },
    )
    output = tmp_path / "converted"
    convert(tmp_path, 4, output)
    convert(tmp_path, 4, output)
    (output / "model/.metadata").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Converted file changed"):
        convert(tmp_path, 4, output)


@pytest.mark.parametrize(
    "stage, causal, steps",
    [
        ("c24_a", False, None),
        ("c24_b", False, None),
        ("c72", False, None),
        ("tf", True, None),
        ("sf4", True, 4),
        ("sf2", True, 2),
    ],
)
def test_config_only_roundtrip_without_weights(tmp_path, stage, causal, steps):
    from cosmos3_h_embodied.dreams.inference_config import compose, validate_model_config
    from omegaconf import OmegaConf

    spec = smoke.initialize(tmp_path, 1, "released_evaluation", "inference")
    output = tmp_path / f"{stage}.yaml"
    compose(spec, stage, output)
    model = OmegaConf.load(output).model
    cfg = validate_model_config(OmegaConf.to_container(model, resolve=True))
    assert cfg["_type"].rsplit(".", 1)[-1] in ("OmniMoTModelConfig", "OmniMoTCausalModelConfig")
    assert cfg.tokenizer.vae_path.startswith(str(tmp_path))
    assert cfg.video_temporal_causal is causal
    assert cfg.parallelism.data_parallel_shard_degree == 1
    assert "fake_score" not in str(model)
    assert "teacher_load_from" not in model.config
    assert (len(cfg.fixed_step_sampler_config.t_list) if cfg.fixed_step_sampler_config else None) == steps
    if stage == "sf4":
        invalid = OmegaConf.to_container(model, resolve=True)
        architecture = invalid["config"]["vlm_config"]["model_instance"]["config"]["base_config"]
        assert "Qwen3VLMoTConfig.from_json_file" in architecture["_target_"]
        architecture["_target_"] = architecture["_target_"].replace("Qwen3VLMoTConfig", "_MoTConfigBase")
        with pytest.raises(ValueError, match="_full_config_cls"):
            validate_model_config(invalid)


def test_prepare_failure_does_not_publish_half_a_selection(tmp_path, monkeypatch):
    def fail(root, selection, output, **kwargs):
        write_json(output / "c24.json", {})
        raise ValueError("C72 has no test samples")

    monkeypatch.setattr(prepare, "prepare", fail)
    with pytest.raises(ValueError, match="C72"):
        prepare.prepare_to_directory(tmp_path, tmp_path / "selection.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_dataset_size_does_not_download(tmp_path, monkeypatch):
    class API:
        def list_repo_tree(self, *args, **kwargs):
            assert kwargs["revision"] == prepare.assets()["dataset"]["revision"]
            return [SimpleNamespace(path="folder"), SimpleNamespace(path="file", size=123)]

    monkeypatch.setattr(prepare, "HfApi", API)
    monkeypatch.setattr(prepare, "snapshot_download", lambda **kw: pytest.fail("size query downloaded files"))
    prepare.download_dataset(tmp_path, size_only=True)
    assert json.loads((tmp_path / "reports/dataset_inventory.json").read_text())["bytes"] == 123
