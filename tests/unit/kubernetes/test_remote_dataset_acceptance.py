# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes runs accept remote dataset references and pin them on the CLI host."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import kopf
import orjson
import pytest
import yaml
from pytest import param

from aiperf.cli_commands.kube import generate as generate_cmd
from aiperf.cli_commands.kube import profile as profile_cmd
from aiperf.cli_commands.kube import sweep as sweep_cmd
from aiperf.common.redact import REDACTED_VALUE
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
_CHECK_WARNING = "needs a Hugging Face token"

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


def _needs_auth_unknown() -> AbstractContextManager[AsyncMock]:
    """The anonymous probe cannot tell, so the pod-credential check warns."""
    return patch(
        "aiperf.dataset.remote.anonymous.anonymous_needs_auth",
        AsyncMock(return_value=None),
    )


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


@pytest.fixture(autouse=True)
def _public_sources() -> Iterator[None]:
    with patch(
        "aiperf.dataset.remote.anonymous.anonymous_needs_auth",
        AsyncMock(return_value=False),
    ):
        yield


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
async def test_generate_unpinned_and_check_warnings_keep_yaml_stdout_parseable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        _host_cannot_read(),
        _needs_auth_unknown(),
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
    assert _CHECK_WARNING not in captured.out
    assert _CHECK_WARNING in " ".join(captured.err.split())


@pytest.mark.asyncio
async def test_sweep_dry_run_unpinned_and_check_warnings_keep_json_stdout_parseable(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with _host_cannot_read(), _needs_auth_unknown():
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
    assert _CHECK_WARNING not in captured.out
    assert _CHECK_WARNING in " ".join(captured.err.split())


@pytest.mark.asyncio
async def test_profile_dry_run_unpinned_and_check_warnings_keep_json_stdout_parseable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        _host_cannot_read(),
        _needs_auth_unknown(),
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
    assert _CHECK_WARNING not in captured.out
    assert _CHECK_WARNING in " ".join(captured.err.split())


_PRESIGNED = (
    "https://h.example/t.jsonl?X-Amz-Date=20261015T000000Z&X-Amz-Expires=3600"
    "&X-Amz-Signature=topsecret"
)


def test_validate_job_spec_unredacted_presigned_dataset_is_rejected() -> None:
    with pytest.raises(
        ConfigurationError, match="AIPERF_INJECTED_DATASET_URLS"
    ) as exc_info:
        validate_job_spec(_job_spec(_PRESIGNED))
    assert "topsecret" not in str(exc_info.value)


def test_validate_job_spec_redacted_presigned_dataset_is_accepted() -> None:
    redacted = _PRESIGNED.replace("topsecret", REDACTED_VALUE)
    validated = validate_job_spec(_job_spec(redacted))
    assert validated.benchmark.datasets[0].path.ref == redacted


def test_operator_validate_spec_unredacted_presigned_dataset_fails_without_echoing_it() -> (
    None
):
    patch_obj = MagicMock()
    patch_obj.status = {}
    status = StatusBuilder(patch_obj)
    with (
        patch("aiperf.operator.handlers.create.events") as events,
        pytest.raises(kopf.PermanentError, match="AIPERF_INJECTED_DATASET_URLS"),
    ):
        _validate_spec(_job_spec(_PRESIGNED), {}, status)
    events.spec_invalid.assert_called_once()
    assert "topsecret" not in events.spec_invalid.call_args.args[1]
    assert "topsecret" not in patch_obj.status["error"]


def test_kubernetes_deployment_configmap_never_holds_the_signature() -> None:
    from aiperf.config import AIPerfConfig
    from aiperf.config.deployment import DeploymentConfig
    from aiperf.kubernetes.resources import KubernetesDeployment

    config = AIPerfConfig(benchmark=_benchmark(_PRESIGNED))
    deployment = KubernetesDeployment(
        job_id="job", namespace="ns", config=config, deployment=DeploymentConfig()
    )
    run_config = deployment.get_configmap_spec().data["run_config.json"]
    assert "topsecret" not in run_config
    assert f"X-Amz-Signature={REDACTED_VALUE}" in run_config


_PRESIGNED_NO_EXPIRY = "https://h.example/t.jsonl?X-Amz-Signature=topsecret"


@pytest.mark.parametrize("resolve", _RESOLVERS)  # fmt: skip
def test_resolve_spec_and_name_presigned_input_file_keeps_signature_out_of_cr(
    resolve, remote_access: AsyncMock
) -> None:
    cli_config = CLIConfig(
        model_names=["m"],
        urls=["http://localhost:8000"],
        input_file=_PRESIGNED_NO_EXPIRY,
        custom_dataset_type="mooncake_trace",
    )
    kube_options = _kube_options(
        env_from_secrets={"AIPERF_INJECTED_DATASET_URLS": "dataset-urls/urls"}
    )
    spec, _, _ = resolve(cli_config, kube_options)
    assert b"topsecret" not in orjson.dumps(spec)
    assert spec["benchmark"]["datasets"][0]["path"] == {
        "ref": _PRESIGNED_NO_EXPIRY.replace("topsecret", REDACTED_VALUE),
        "resolved_version": SHA,
    }
    remote_access.assert_awaited_once()


@pytest.mark.parametrize("resolve", _RESOLVERS)  # fmt: skip
def test_resolve_spec_and_name_presigned_input_file_without_secret_env_is_rejected(
    resolve, remote_access: AsyncMock
) -> None:
    cli_config = CLIConfig(
        model_names=["m"],
        urls=["http://localhost:8000"],
        input_file=_PRESIGNED_NO_EXPIRY,
        custom_dataset_type="mooncake_trace",
    )
    with pytest.raises(ValueError, match="AIPERF_INJECTED_DATASET_URLS") as exc_info:
        resolve(cli_config, _kube_options())
    assert "topsecret" not in str(exc_info.value)


_REDACTED_NO_EXPIRY = _PRESIGNED_NO_EXPIRY.replace("topsecret", REDACTED_VALUE)
_URLS_SECRET = {"AIPERF_INJECTED_DATASET_URLS": "dataset-urls/urls"}


def _write_job_cr(tmp_path: Path, spec: dict[str, object]) -> Path:
    cr = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "remote-job"},
        "spec": spec,
    }
    config_file = tmp_path / "job.yaml"
    config_file.write_text(yaml.safe_dump(cr, sort_keys=False))
    return config_file


def _host_denied(ref: str) -> AbstractContextManager[AsyncMock]:
    """This machine is refused, as S3 refuses a URL whose signature is ``<redacted>``."""
    return patch(
        "aiperf.dataset.remote.service.probe_ref",
        AsyncMock(side_effect=_access_error(RemoteErrorKind.AUTH_DENIED, ref)),
    )


def test_resolve_spec_and_name_generated_cr_with_redacted_presigned_ref_resubmits_without_network(
    remote_access: AsyncMock, tmp_path: Path
) -> None:
    cli_config = CLIConfig(
        model_names=["m"],
        urls=["http://localhost:8000"],
        input_file=_PRESIGNED_NO_EXPIRY,
        custom_dataset_type="mooncake_trace",
    )
    kube_options = _kube_options(env_from_secrets=_URLS_SECRET)
    generated, _, _ = generate_cmd._resolve_spec_and_name(cli_config, kube_options)
    config_file = _write_job_cr(tmp_path, generated)
    with (
        _host_denied(_REDACTED_NO_EXPIRY) as probe,
        patch("aiperf.kubernetes.console.print_warning") as warn,
    ):
        spec, _, _ = profile_cmd._resolve_spec_and_name(
            CLIConfig(config_file=config_file), kube_options
        )
    probe.assert_not_awaited()
    warn.assert_not_called()
    assert spec["benchmark"]["datasets"][0]["path"] == {
        "ref": _REDACTED_NO_EXPIRY,
        "resolved_version": SHA,
    }


def test_resolve_spec_and_name_unpinned_redacted_cr_warns_and_submits_unpinned(
    tmp_path: Path,
) -> None:
    config_file = _write_job_cr(tmp_path, _job_spec(_REDACTED_NO_EXPIRY))
    with (
        _host_denied(_REDACTED_NO_EXPIRY) as probe,
        patch("aiperf.kubernetes.console.print_warning") as warn,
    ):
        spec, _, _ = profile_cmd._resolve_spec_and_name(
            CLIConfig(config_file=config_file),
            _kube_options(env_from_secrets=_URLS_SECRET),
        )
    probe.assert_not_awaited()
    assert spec["benchmark"]["datasets"][0]["path"] == _REDACTED_NO_EXPIRY
    assert _UNPINNED_WARNING in warn.call_args_list[0].args[0]


def test_pin_remote_datasets_for_kubernetes_redacted_ref_with_deferred_promotion_raises() -> (
    None
):
    benchmark = _remote_benchmark(_REDACTED_NO_EXPIRY)
    assert benchmark.get_default_dataset().remote_fixed_schedule_check is not None
    with (
        _host_denied(_REDACTED_NO_EXPIRY) as probe,
        pytest.raises(ConfigurationError, match="--fixed-schedule"),
    ):
        pin_remote_datasets_for_kubernetes(benchmark)
    probe.assert_not_awaited()


class TransportError(Exception):
    """Stands in for httpx.TransportError, which the HF resolver matches by name."""


class _HFConnectError(TransportError):
    """Stands in for httpx.ConnectError (a DNS or connect failure)."""


def _hf_failure(status: int | None) -> RemoteSourceError:
    from huggingface_hub.errors import HfHubHTTPError

    from aiperf.dataset.remote import hf

    error = (
        _HFConnectError("[Errno 8] nodename nor servname provided")
        if status is None
        else HfHubHTTPError(f"http {status}", response=MagicMock(status_code=status))
    )
    mapped = hf._translate(
        RemoteRef(ref=HF_REF), error, RemoteStage.ACCESS, has_token=False
    )
    assert mapped is not None
    return mapped


def _s3_failure(status: int | None) -> RemoteSourceError:
    from botocore.exceptions import ClientError, EndpointConnectionError

    from aiperf.dataset.remote import s3

    error = (
        EndpointConnectionError(endpoint_url="https://bucket.s3.amazonaws.com")
        if status is None
        else ClientError(
            {"Error": {"Code": "ServiceUnavailable"}, "ResponseMetadata": {"HTTPStatusCode": status}},
            "HeadObject",
        )
    )  # fmt: skip
    mapped = s3._classify(RemoteRef(ref=S3_REF), error, RemoteStage.ACCESS)
    assert mapped is not None
    return mapped


def _url_failure(ref: str, status: int | None) -> RemoteSourceError:
    from aiperf.dataset.remote import http_fetch

    if status is None:
        return http_fetch._transport_error(
            TimeoutError("connect timed out"), ref_display=ref, stage=RemoteStage.ACCESS
        )
    mapped = http_fetch.status_error(status, ref_display=ref, stage=RemoteStage.ACCESS)
    assert mapped is not None
    return mapped


_HTTP_REF = "http://h.example/t.jsonl"
_HOST_FAILURES = [
    param(HF_REF, lambda status: _hf_failure(status), id="hf"),
    param(S3_REF, lambda status: _s3_failure(status), id="s3"),
    param(HTTPS_REF, lambda status: _url_failure(HTTPS_REF, status), id="https"),
    param(_HTTP_REF, lambda status: _url_failure(_HTTP_REF, status), id="http"),
]


@pytest.fixture
def online(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reach the resolvers: tests/unit/conftest.py forces HF offline for the session."""
    from aiperf.common.environment import Environment

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(Environment.DATASET, "REMOTE_OFFLINE", False)


def _resolver_failing_with(
    error: RemoteSourceError,
) -> AbstractContextManager[MagicMock]:
    resolver = MagicMock()
    resolver.probe = AsyncMock(side_effect=error)
    return patch("aiperf.dataset.remote.service.get_resolver", return_value=resolver)


@pytest.mark.usefixtures("online")
@pytest.mark.parametrize("ref,failure", _HOST_FAILURES)  # fmt: skip
def test_pin_remote_datasets_for_kubernetes_connection_failure_stays_unpinned_and_warns(
    ref: str, failure
) -> None:
    """This machine cannot reach the source; a cluster-only network may."""
    from aiperf.dataset.remote.service import RETRY_ATTEMPTS

    benchmark = _remote_benchmark(ref, disable_auto_fixed_schedule=True)
    error = failure(None)
    assert error.connection_failed
    with (
        _resolver_failing_with(error) as get_resolver,
        patch("aiperf.kubernetes.console.print_warning") as warn,
    ):
        pin_remote_datasets_for_kubernetes(benchmark)
    assert get_resolver.return_value.probe.await_count == RETRY_ATTEMPTS
    assert benchmark.get_default_dataset().path.resolved_version is None
    message = warn.call_args.args[0]
    assert _UNPINNED_WARNING in message
    assert "could not connect" in message


@pytest.mark.usefixtures("online")
@pytest.mark.parametrize("ref,failure", _HOST_FAILURES)  # fmt: skip
def test_pin_remote_datasets_for_kubernetes_http_503_raises_after_retries(
    ref: str, failure
) -> None:
    """The source answered and is struggling; the pod would get the same answer."""
    from aiperf.dataset.remote.service import RETRY_ATTEMPTS

    benchmark = _remote_benchmark(ref, disable_auto_fixed_schedule=True)
    with (
        _resolver_failing_with(failure(503)) as get_resolver,
        patch("aiperf.kubernetes.console.print_warning") as warn,
        pytest.raises(RemoteSourceError) as exc_info,
    ):
        pin_remote_datasets_for_kubernetes(benchmark)
    assert exc_info.value.kind == RemoteErrorKind.TRANSIENT
    assert get_resolver.return_value.probe.await_count == RETRY_ATTEMPTS
    warn.assert_not_called()


@pytest.mark.usefixtures("online")
def test_pin_remote_datasets_for_kubernetes_connection_failure_with_deferred_promotion_raises() -> (
    None
):
    benchmark = _remote_benchmark(S3_REF)
    with (
        _resolver_failing_with(_s3_failure(None)),
        pytest.raises(ConfigurationError, match="--fixed-schedule"),
    ):
        pin_remote_datasets_for_kubernetes(benchmark)


@pytest.mark.usefixtures("online")
def test_pin_remote_datasets_for_kubernetes_refused_connection_on_real_http_path_warns() -> (
    None
):
    """End to end through aiohttp: nothing listens on the discard port."""
    ref = "http://127.0.0.1:9/t.jsonl"
    benchmark = _remote_benchmark(ref, disable_auto_fixed_schedule=True)
    with patch("aiperf.kubernetes.console.print_warning") as warn:
        pin_remote_datasets_for_kubernetes(benchmark)
    assert benchmark.get_default_dataset().path.resolved_version is None
    assert _UNPINNED_WARNING in warn.call_args.args[0]
