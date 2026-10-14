# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes runs reject remote dataset references before anything is scheduled."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import kopf
import pytest
import yaml
from pytest import param

from aiperf.cli_commands.kube import generate as generate_cmd
from aiperf.cli_commands.kube import profile as profile_cmd
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions
from aiperf.config.loader import ConfigurationError
from aiperf.kubernetes.spec_converter import validate_job_spec, validate_sweep_spec
from aiperf.operator.handlers.create import _validate_spec
from aiperf.operator.status import StatusBuilder

_MESSAGE = (
    r"Remote dataset references \(hf://, s3://, https://\) are not yet supported "
    r"for Kubernetes runs"
)
_PRESIGNED = "https://h.example/t.jsonl?X-Amz-Signature=topsecret"

_BENCHMARK: dict[str, object] = {
    "models": ["m"],
    "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
    "datasets": [
        {
            "name": "main",
            "type": "file",
            "path": "hf://org/repo/t.jsonl",
            "format": "mooncake_trace",
        }
    ],
    "phases": [{"name": "profiling", "type": "concurrency", "requests": 8, "concurrency": 2}],
}  # fmt: skip


def _benchmark(path: str) -> dict[str, object]:
    benchmark = copy.deepcopy(_BENCHMARK)
    benchmark["datasets"][0]["path"] = path  # type: ignore[index]
    return benchmark


def _job_spec(path: str) -> dict[str, object]:
    return {"image": "aiperf:ci", "benchmark": _benchmark(path)}


def _kube_options() -> KubeOptions:
    return KubeOptions(image="aiperf:ci", name="remote-job")


@pytest.mark.parametrize(
    "path",
    [
        param("hf://org/repo/t.jsonl", id="hf"),
        param("s3://bucket/traces/t.jsonl", id="s3"),
        param(_PRESIGNED, id="presigned-https"),
    ],
)  # fmt: skip
def test_validate_job_spec_rejects_remote_dataset(path: str) -> None:
    with pytest.raises(ConfigurationError, match=_MESSAGE) as exc:
        validate_job_spec(_job_spec(path))
    assert "Download the file and use a local path" in str(exc.value)
    assert "topsecret" not in str(exc.value)


def test_validate_sweep_spec_rejects_remote_dataset() -> None:
    spec = {
        **_job_spec("s3://bucket/t.jsonl"),
        "sweep": {"type": "grid", "parameters": {"phases.profiling.concurrency": [1, 2]}},
    }  # fmt: skip
    with pytest.raises(ConfigurationError, match=_MESSAGE):
        validate_sweep_spec(spec)


def test_validate_job_spec_accepts_local_dataset(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"timestamp": 0, "input_length": 4, "output_length": 2}\n')
    validate_job_spec(_job_spec(str(trace)))


def test_operator_marks_remote_dataset_spec_invalid() -> None:
    patch_obj = MagicMock()
    patch_obj.status = {}
    status = StatusBuilder(patch_obj)
    with (
        patch("aiperf.operator.handlers.create.events") as events,
        pytest.raises(kopf.PermanentError, match=_MESSAGE),
    ):
        _validate_spec(_job_spec(_PRESIGNED), {}, status)
    events.spec_invalid.assert_called_once()
    assert "topsecret" not in events.spec_invalid.call_args.args[1]


@pytest.mark.parametrize(
    "resolve",
    [
        param(profile_cmd._resolve_spec_and_name, id="kube-profile"),
        param(generate_cmd._resolve_spec_and_name, id="kube-generate"),
    ],
)  # fmt: skip
def test_kube_cli_rejects_remote_input_file(resolve) -> None:
    cli_config = CLIConfig(
        model_names=["m"],
        urls=["http://localhost:8000"],
        input_file="hf://org/repo/t.jsonl",
        custom_dataset_type="mooncake_trace",
    )
    with pytest.raises(ConfigurationError, match=_MESSAGE):
        resolve(cli_config, _kube_options())


@pytest.mark.parametrize(
    "resolve",
    [
        param(profile_cmd._resolve_spec_and_name, id="kube-profile"),
        param(generate_cmd._resolve_spec_and_name, id="kube-generate"),
    ],
)  # fmt: skip
def test_kube_cli_rejects_remote_dataset_in_aiperfjob_cr(resolve, tmp_path: Path) -> None:  # fmt: skip
    cr = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "remote-job"},
        "spec": _job_spec("hf://org/repo/t.jsonl"),
    }
    config_file = tmp_path / "job.yaml"
    config_file.write_text(yaml.safe_dump(cr, sort_keys=False))
    with pytest.raises(ConfigurationError, match=_MESSAGE):
        resolve(CLIConfig(config_file=config_file), _kube_options())
