# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf kube show — render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _minimal_cr() -> dict:
    """Minimal valid AIPerfJob CR dict."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "test-job"},
        "spec": {
            "image": "nvcr.io/nvidia/aiperf:latest",
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 32, "osl": 16},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "kind": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 1,
                    }
                ],
            },
        },
    }


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc, sort_keys=False))
    return path


def test_show_module_importable() -> None:
    """The show module must be importable and expose an `app` attribute."""
    from aiperf.cli_commands.kube import show

    assert hasattr(show, "app"), "show.app (cyclopts App) must be defined"


def test_show_registered_in_kube_app() -> None:
    """The `show` subcommand must be wired into `aiperf kube`."""
    from aiperf.cli_commands.kube._app import app

    # cyclopts App iteration yields registered command names as strings
    # (alongside flags like --help). We only care that "show" is registered.
    command_names = set(app)
    assert "show" in command_names


def _run_show(path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Invoke the show command's default callable directly and return stdout."""
    from aiperf.cli_commands.kube.show import show as show_cmd

    show_cmd(path=path)
    return capsys.readouterr().out


def test_show_renders_jinja_templates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`{{ a * b }}` inside phases must resolve to an int; variables section is preserved."""
    doc = _minimal_cr()
    doc["spec"]["variables"] = {
        "concurrency_per_gpu": 2,
        "deployment_gpu_count": 16,
    }
    doc["spec"]["benchmark"]["phases"][0]["concurrency"] = (
        "{{ concurrency_per_gpu * deployment_gpu_count }}"
    )
    doc["spec"]["benchmark"]["phases"][0]["requests"] = (
        "{{ concurrency_per_gpu * deployment_gpu_count * 10 }}"
    )
    path = _write(tmp_path / "job.yaml", doc)

    out = _run_show(path, capsys)
    rendered = yaml.safe_load(out)

    phase = rendered["spec"]["benchmark"]["phases"][0]
    assert phase["name"] == "default"
    assert phase["concurrency"] == 32
    assert phase["requests"] == 320
    assert rendered["spec"]["variables"] == {
        "concurrency_per_gpu": 2,
        "deployment_gpu_count": 16,
    }
    assert "{{" not in out


def test_show_passes_through_non_benchmark_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """metadata and non-benchmark spec.* keys must appear unchanged."""
    doc = _minimal_cr()
    doc["spec"]["image"] = "custom-image:v1"
    doc["spec"]["connectionsPerWorker"] = 200
    doc["spec"]["podTemplate"] = {
        "imagePullSecrets": [{"name": "mysecret"}],
        "env": [{"name": "X", "value": "y"}],
    }
    path = _write(tmp_path / "job.yaml", doc)

    out = _run_show(path, capsys)
    rendered = yaml.safe_load(out)

    assert rendered["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
    assert rendered["kind"] == "AIPerfJob"
    assert rendered["metadata"]["name"] == "test-job"
    assert rendered["spec"]["image"] == "custom-image:v1"
    assert rendered["spec"]["connectionsPerWorker"] == 200
    assert rendered["spec"]["podTemplate"]["imagePullSecrets"] == [{"name": "mysecret"}]
    assert rendered["spec"]["podTemplate"]["env"] == [{"name": "X", "value": "y"}]


def test_show_resolves_env_var_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """${VAR:default} must resolve to `default` when VAR is unset."""
    monkeypatch.delenv("AIPERF_TEST_MODEL", raising=False)

    doc = _minimal_cr()
    doc["spec"]["benchmark"]["models"] = ["${AIPERF_TEST_MODEL:fallback-model}"]
    path = _write(tmp_path / "job.yaml", doc)

    out = _run_show(path, capsys)
    rendered = yaml.safe_load(out)

    # AIPerfConfig normalises string/list forms to {"items": [{"name": ...}]}.
    items = rendered["spec"]["benchmark"]["models"]["items"]
    assert items[0]["name"] == "fallback-model"


def test_show_missing_file_exits_nonzero(tmp_path: Path) -> None:
    from aiperf.cli_commands.kube.show import show as show_cmd

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=tmp_path / "does-not-exist.yaml")
    assert exc_info.value.code != 0


def test_show_wrong_kind_exits_nonzero(tmp_path: Path) -> None:
    from aiperf.cli_commands.kube.show import show as show_cmd

    doc = _minimal_cr()
    doc["kind"] = "Pod"
    path = _write(tmp_path / "job.yaml", doc)

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=path)
    assert exc_info.value.code != 0


def test_show_missing_benchmark_exits_nonzero(tmp_path: Path) -> None:
    from aiperf.cli_commands.kube.show import show as show_cmd

    doc = _minimal_cr()
    del doc["spec"]["benchmark"]
    path = _write(tmp_path / "job.yaml", doc)

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=path)
    assert exc_info.value.code != 0


def test_show_invalid_benchmark_exits_nonzero(tmp_path: Path) -> None:
    """First phase with seamless=True is a known AIPerfConfig invariant violation."""
    from aiperf.cli_commands.kube.show import show as show_cmd

    doc = _minimal_cr()
    # Seamless is only valid on non-first phases; AIPerfConfig rejects it here.
    doc["spec"]["benchmark"]["phases"][0]["seamless"] = True
    path = _write(tmp_path / "job.yaml", doc)

    with pytest.raises(SystemExit) as exc_info:
        show_cmd(path=path)
    assert exc_info.value.code != 0
