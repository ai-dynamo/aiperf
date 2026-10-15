# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes runs accept remote dataset references and pin them on the CLI host."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
import yaml
from pytest import param

from aiperf.cli_commands.kube import generate as generate_cmd
from aiperf.cli_commands.kube import profile as profile_cmd
from aiperf.cli_commands.kube import sweep as sweep_cmd
from aiperf.config import BenchmarkConfig
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.config.flags import CLIConfig
from aiperf.config.kube import KubeOptions
from aiperf.config.loader import ConfigurationError
from aiperf.dataset.remote.errors import (
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
)
from aiperf.dataset.remote.models import ProbeResult
from aiperf.dataset.remote.preflight import reset_probe_cache
from aiperf.kubernetes.remote_datasets import pin_remote_datasets_for_kubernetes
from aiperf.kubernetes.spec_converter import validate_job_spec, validate_sweep_spec
from aiperf.operator.handlers.create import _validate_spec
from aiperf.operator.status import StatusBuilder
from tests.unit.conftest import make_run_from_cli

SHA = "c" * 40
HF_REF = "hf://org/repo/t.jsonl"
S3_REF = "s3://bucket/traces/t.jsonl"
HTTPS_REF = "https://h.example/t.jsonl"
_UNPINNED_WARNING = "the dataset-manager pod will pin it"

_BENCHMARK: dict[str, object] = {
    "models": ["m"],
    "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
    "datasets": [
        {"name": "main", "type": "file", "path": HF_REF, "format": "mooncake_trace"}
    ],
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 8, "concurrency": 2}
    ],
}

_RESOLVERS = [
    param(profile_cmd._resolve_spec_and_name, id="kube-profile"),
    param(generate_cmd._resolve_spec_and_name, id="kube-generate"),
]


def _benchmark(path: object) -> dict[str, object]:
    benchmark = copy.deepcopy(_BENCHMARK)
    benchmark["datasets"][0]["path"] = path  # type: ignore[index]
    return benchmark


def _job_spec(path: object) -> dict[str, object]:
    return {"image": "aiperf:ci", "benchmark": _benchmark(path)}


def _kube_options(**overrides: object) -> KubeOptions:
    return KubeOptions(image="aiperf:ci", name="remote-job", **overrides)


def _probe_for(ref: RemoteRef) -> ProbeResult:
    return ProbeResult(
        ref=ref.with_version(SHA),
        pinned_ref=f"{ref.display}@{SHA}",
        data_relpath="t.jsonl",
        data_is_dir=False,
        filename="t.jsonl",
        manifest=None,
        manifest_sha256=None,
        needs_auth=False,
    )


def _access_error(kind: RemoteErrorKind, ref: str = HF_REF) -> RemoteSourceError:
    return RemoteSourceError(
        ref=ref,
        kind=kind,
        stage=RemoteStage.ACCESS,
        what="the source refused this machine",
        hint="Check credentials",
    )


def _remote_benchmark(ref: str = HF_REF, **cli: object) -> BenchmarkConfig:
    return make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            input_file=ref,
            custom_dataset_type="mooncake_trace",
            request_count=4,
            **cli,
        )
    ).cfg


def _host_cannot_read() -> AbstractContextManager[AsyncMock]:
    """Every probe from this machine fails with ``AUTH_MISSING``."""
    return patch(
        "aiperf.dataset.remote.service.probe_ref",
        AsyncMock(side_effect=_access_error(RemoteErrorKind.AUTH_MISSING)),
    )


@pytest.fixture(autouse=True)
def _fresh_probe_cache() -> Iterator[None]:
    reset_probe_cache()
    yield
    reset_probe_cache()


@pytest.fixture
def remote_access() -> Iterator[AsyncMock]:
    """Pin every reference to ``SHA``; its first record has no timestamps."""
    with (
        patch(
            "aiperf.dataset.remote.service.probe_ref",
            AsyncMock(side_effect=_probe_for),
        ) as probe,
        patch(
            "aiperf.dataset.remote.preflight.remote_first_record_has_timestamp",
            AsyncMock(return_value=False),
        ),
    ):
        yield probe


@pytest.mark.parametrize(
    "path",
    [
        param(HF_REF, id="hf"),
        param(S3_REF, id="s3"),
        param(HTTPS_REF, id="https"),
    ],
)  # fmt: skip
def test_validate_job_spec_remote_dataset_is_accepted(path: str) -> None:
    validated = validate_job_spec(_job_spec(path))
    assert validated.benchmark.datasets[0].path == RemoteRef(ref=path)


def test_validate_job_spec_pinned_remote_dataset_keeps_version() -> None:
    validated = validate_job_spec(_job_spec({"ref": HF_REF, "resolved_version": SHA}))
    assert validated.benchmark.datasets[0].path.resolved_version == SHA


def test_validate_sweep_spec_remote_dataset_is_accepted() -> None:
    spec = {
        **_job_spec("s3://bucket/t.jsonl"),
        "sweep": {"type": "grid", "parameters": {"phases.profiling.concurrency": [1, 2]}},
    }  # fmt: skip
    validated = validate_sweep_spec(spec)
    assert validated.benchmark.datasets[0].path == RemoteRef(ref="s3://bucket/t.jsonl")


def test_validate_job_spec_remote_fixed_schedule_marker_is_rejected() -> None:
    spec = _job_spec(HF_REF)
    spec["benchmark"]["datasets"][0]["remoteFixedScheduleCheck"] = {"promoted": True}  # type: ignore[index]
    with pytest.raises(ConfigurationError, match="remoteFixedScheduleCheck"):
        validate_job_spec(spec)


def test_operator_validate_spec_remote_dataset_is_accepted() -> None:
    patch_obj = MagicMock()
    patch_obj.status = {}
    status = StatusBuilder(patch_obj)
    with patch("aiperf.operator.handlers.create.events") as events:
        validated = _validate_spec(_job_spec(HF_REF), {}, status)
    events.spec_valid.assert_called_once()
    assert validated.benchmark.datasets[0].path == RemoteRef(ref=HF_REF)


@pytest.mark.parametrize("resolve", _RESOLVERS)  # fmt: skip
def test_resolve_spec_and_name_remote_input_file_is_pinned_on_host(
    resolve, remote_access: AsyncMock
) -> None:
    cli_config = CLIConfig(
        model_names=["m"],
        urls=["http://localhost:8000"],
        input_file=HF_REF,
        custom_dataset_type="mooncake_trace",
    )
    spec, _, _ = resolve(cli_config, _kube_options())
    dataset = spec["benchmark"]["datasets"][0]
    assert dataset["path"] == {"ref": HF_REF, "resolved_version": SHA}
    assert "remoteFixedScheduleCheck" not in dataset
    assert spec["benchmark"]["phases"][0]["type"] == "concurrency"
    remote_access.assert_awaited_once()


@pytest.mark.parametrize("resolve", _RESOLVERS)  # fmt: skip
def test_resolve_spec_and_name_remote_dataset_in_aiperfjob_cr_is_pinned_on_host(
    resolve, remote_access: AsyncMock, tmp_path: Path
) -> None:
    cr = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "remote-job"},
        "spec": _job_spec(HF_REF),
    }
    config_file = tmp_path / "job.yaml"
    config_file.write_text(yaml.safe_dump(cr, sort_keys=False))
    spec, _, _ = resolve(CLIConfig(config_file=config_file), _kube_options())
    assert spec["benchmark"]["datasets"][0]["path"] == {
        "ref": HF_REF,
        "resolved_version": SHA,
    }


def _write_sweep_yaml(tmp_path: Path) -> Path:
    config_file = tmp_path / "sweep.yaml"
    body = {
        **_benchmark(HF_REF),
        "sweep": {"type": "grid", "parameters": {"phases.profiling.concurrency": [1, 2]}},
    }  # fmt: skip
    config_file.write_text(yaml.safe_dump(body, sort_keys=False))
    return config_file


def test_build_sweep_cr_dict_remote_dataset_is_pinned_once(
    remote_access: AsyncMock, tmp_path: Path
) -> None:
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=_write_sweep_yaml(tmp_path),
        kube_options=_kube_options(),
        multi_run_trials=None,
        cooldown_seconds=0.0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["spec"]["benchmark"]["datasets"][0]["path"] == {
        "ref": HF_REF,
        "resolved_version": SHA,
    }
    remote_access.assert_awaited_once()


@pytest.mark.parametrize(
    "ref,kind",
    [
        param(HF_REF, RemoteErrorKind.AUTH_MISSING, id="hf-auth-missing"),
        param(HF_REF, RemoteErrorKind.AUTH_DENIED, id="hf-auth-denied"),
        param(HF_REF, RemoteErrorKind.GATED, id="hf-gated"),
        param(S3_REF, RemoteErrorKind.AUTH_MISSING, id="s3-auth-missing"),
        param(S3_REF, RemoteErrorKind.AUTH_DENIED, id="s3-auth-denied"),
    ],
)  # fmt: skip
def test_pin_remote_datasets_for_kubernetes_unreadable_source_stays_unpinned_and_warns(
    ref: str, kind: RemoteErrorKind
) -> None:
    benchmark = _remote_benchmark(ref, disable_auto_fixed_schedule=True)
    with (
        patch(
            "aiperf.dataset.remote.service.probe_ref",
            AsyncMock(side_effect=_access_error(kind, ref)),
        ),
        patch("aiperf.kubernetes.console.print_warning") as warn,
    ):
        pin_remote_datasets_for_kubernetes(benchmark)
    assert benchmark.get_default_dataset().path.resolved_version is None
    message = warn.call_args.args[0]
    assert _UNPINNED_WARNING in message
    assert str(kind) in message


@pytest.mark.parametrize(
    "ref,kind",
    [
        param(HTTPS_REF, RemoteErrorKind.AUTH_MISSING, id="https-auth-missing"),
        param(HTTPS_REF, RemoteErrorKind.AUTH_DENIED, id="https-auth-denied"),
        param("http://h.example/t.jsonl", RemoteErrorKind.AUTH_DENIED, id="http-auth-denied"),
    ],
)  # fmt: skip
def test_pin_remote_datasets_for_kubernetes_http_auth_failure_raises(
    ref: str, kind: RemoteErrorKind
) -> None:
    """The pod would fetch the same URL, so its credentials cannot do better."""
    benchmark = _remote_benchmark(ref, disable_auto_fixed_schedule=True)
    with (
        patch(
            "aiperf.dataset.remote.service.probe_ref",
            AsyncMock(side_effect=_access_error(kind, ref)),
        ),
        patch("aiperf.kubernetes.console.print_warning") as warn,
        pytest.raises(RemoteSourceError) as exc_info,
    ):
        pin_remote_datasets_for_kubernetes(benchmark)
    assert exc_info.value.kind == kind
    warn.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    [
        param(RemoteErrorKind.NOT_FOUND, id="not-found"),
        param(RemoteErrorKind.MANIFEST_INVALID, id="manifest-invalid"),
        param(RemoteErrorKind.TRANSIENT, id="transient"),
    ],
)  # fmt: skip
def test_pin_remote_datasets_for_kubernetes_other_failure_raises(
    kind: RemoteErrorKind,
) -> None:
    benchmark = _remote_benchmark(disable_auto_fixed_schedule=True)
    with (
        patch(
            "aiperf.dataset.remote.service.probe_ref",
            AsyncMock(side_effect=_access_error(kind)),
        ),
        pytest.raises(RemoteSourceError) as exc_info,
    ):
        pin_remote_datasets_for_kubernetes(benchmark)
    assert exc_info.value.kind == kind


def test_pin_remote_datasets_for_kubernetes_unreadable_with_deferred_promotion_raises() -> (
    None
):
    benchmark = _remote_benchmark()
    assert benchmark.get_default_dataset().remote_fixed_schedule_check is not None
    with (
        patch(
            "aiperf.dataset.remote.service.probe_ref",
            AsyncMock(side_effect=_access_error(RemoteErrorKind.AUTH_MISSING)),
        ),
        pytest.raises(ConfigurationError, match="--fixed-schedule") as exc_info,
    ):
        pin_remote_datasets_for_kubernetes(benchmark)
    assert "--no-fixed-schedule" in str(exc_info.value)


def _unpinned_cli_config() -> CLIConfig:
    return CLIConfig(
        model_names=["m"],
        urls=["http://localhost:8000"],
        input_file=HF_REF,
        custom_dataset_type="mooncake_trace",
        disable_auto_fixed_schedule=True,
    )


@pytest.mark.asyncio
async def test_generate_unpinned_warning_keeps_yaml_stdout_parseable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        _host_cannot_read(),
        patch("aiperf.cli_commands.kube.generate._print_memory_estimate"),
    ):
        await generate_cmd.generate(
            cli_config=_unpinned_cli_config(),
            kube_options=_kube_options(namespace="bench-ns"),
            operator=True,
        )
    captured = capsys.readouterr()
    cr = yaml.safe_load(captured.out)
    assert cr["kind"] == "AIPerfJob"
    assert cr["spec"]["benchmark"]["datasets"][0]["path"] == HF_REF
    assert _UNPINNED_WARNING not in captured.out
    assert _UNPINNED_WARNING in " ".join(captured.err.split())


@pytest.mark.asyncio
async def test_sweep_dry_run_unpinned_warning_keeps_json_stdout_parseable(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with _host_cannot_read():
        await sweep_cmd.sweep(
            cli_config=CLIConfig(config_file=_write_sweep_yaml(tmp_path)),
            kube_options=_kube_options(namespace="bench-ns"),
            dry_run=True,
        )
    captured = capsys.readouterr()
    cr = orjson.loads(captured.out)
    assert cr["kind"] == "AIPerfSweep"
    assert cr["spec"]["benchmark"]["datasets"][0]["path"] == HF_REF
    assert _UNPINNED_WARNING in " ".join(captured.err.split())


@pytest.mark.asyncio
async def test_profile_dry_run_unpinned_warning_keeps_json_stdout_parseable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        _host_cannot_read(),
        patch("aiperf.cli_commands.kube.profile._print_memory_estimate"),
    ):
        await profile_cmd.profile(
            cli_config=_unpinned_cli_config(),
            kube_options=_kube_options(namespace="bench-ns"),
            dry_run=True,
            operator=True,
        )
    captured = capsys.readouterr()
    cr = orjson.loads(captured.out)
    assert cr["kind"] == "AIPerfJob"
    assert cr["spec"]["benchmark"]["datasets"][0]["path"] == HF_REF
    assert _UNPINNED_WARNING in " ".join(captured.err.split())
