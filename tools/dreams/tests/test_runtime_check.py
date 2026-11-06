# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Missing media dependencies must fail before a costly training run."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cosmos3_h_embodied.dreams import runtime_check


def test_media_records_both_versions(monkeypatch):
    def run(command, **kwargs):
        assert command == ["ffmpeg", "-version"]
        assert kwargs["check"] and kwargs["timeout"] == 30
        return SimpleNamespace(stdout="ffmpeg version test\nconfiguration: test")

    monkeypatch.setattr(runtime_check.subprocess, "run", run)
    monkeypatch.setattr(runtime_check, "load_torchcodec", lambda: SimpleNamespace(__version__="test"))
    assert runtime_check.check_media() == {"ffmpeg": "ffmpeg version test", "torchcodec": "test"}


@pytest.mark.parametrize("error", [FileNotFoundError("ffmpeg"), subprocess.CalledProcessError(1, "ffmpeg")])
def test_missing_ffmpeg_is_actionable(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(runtime_check.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="system packages") as caught:
        runtime_check.check_media()
    assert caught.value.__cause__ is error


def test_executable_does_not_hide_missing_shared_libraries(monkeypatch):
    monkeypatch.setattr(runtime_check.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="ffmpeg version test"))
    error = RuntimeError("libavutil.so.60: cannot open shared object file")

    def fail():
        raise error

    monkeypatch.setattr(runtime_check, "load_torchcodec", fail)
    with pytest.raises(RuntimeError, match="FFmpeg shared libraries") as caught:
        runtime_check.check_media()
    assert caught.value.__cause__ is error


def test_media_failure_is_recorded_before_gpu_check(tmp_path, monkeypatch):
    output = tmp_path / "devices.json"
    monkeypatch.setattr(sys, "argv", ["runtime_check", "--gpus", "8", "--output", str(output)])

    def fail():
        raise RuntimeError("media dependency missing")

    def unexpected():
        pytest.fail("GPU operations must not run after media failure")

    monkeypatch.setattr(runtime_check, "check_media", fail)
    monkeypatch.setattr(runtime_check.torch.cuda, "is_available", unexpected)
    with pytest.raises(RuntimeError, match="media dependency missing"):
        runtime_check.main()
    assert json.loads(output.read_text()) == {
        "status": "failed", "phase": "media", "error": "media dependency missing"
    }
