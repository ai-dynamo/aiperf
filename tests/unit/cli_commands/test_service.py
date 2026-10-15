# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for service CLI command."""

from __future__ import annotations

import inspect
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import orjson
import pytest
from pytest import param
from rich.console import Console

import aiperf.cli_utils as cli_utils
from aiperf.cli_commands.service import app, service
from aiperf.common.environment import Environment
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError, RemoteStage
from aiperf.kubernetes import dataset_startup
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_bootstrap() -> Generator[MagicMock, None, None]:
    """Mock bootstrap_and_run_service."""
    # Patched at source; works because service() uses lazy imports inside the function body.
    with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as mock:
        yield mock


@pytest.fixture(autouse=True)
def termination_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every startup failure away from the real ``/dev/termination-log``."""
    log = tmp_path / "termination-log"
    monkeypatch.setattr(dataset_startup, "TERMINATION_LOG_PATH", log)
    return log


@pytest.fixture
def service_type() -> MagicMock:
    """Create a mock ServiceType."""
    return MagicMock()


def _write_benchmark_run(
    path: Path,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    urls: list[str] | None = None,
) -> Path:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {
                "urls": urls or ["http://server/v1"],
                "type": "chat",
                "api_key": api_key,
                "headers": headers or {},
            },
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 1,
                }
            ],
        }
    )
    run = BenchmarkRun(benchmark_id="job", cfg=cfg, artifact_dir=path.parent)
    path.write_bytes(orjson.dumps(run.model_dump(mode="json", exclude_none=True)))
    return path


@pytest.fixture
def benchmark_run_file(tmp_path: Path) -> Path:
    """Write a valid serialized run for service bootstrap tests."""
    return _write_benchmark_run(tmp_path / "run.json")


@pytest.fixture(autouse=True)
def _reset_health_settings() -> Generator[None, None, None]:
    """Reset Environment.SERVICE health settings after each test."""
    original_enabled = Environment.SERVICE.HEALTH_ENABLED
    original_host = Environment.SERVICE.HEALTH_HOST
    original_port = Environment.SERVICE.HEALTH_PORT
    yield
    Environment.SERVICE.HEALTH_ENABLED = original_enabled
    Environment.SERVICE.HEALTH_HOST = original_host
    Environment.SERVICE.HEALTH_PORT = original_port


class TestServiceCommand:
    """Tests for service() CLI function."""

    def test_requires_serialized_benchmark_run(self) -> None:
        """The service command has no per-process CLI config fallback."""
        signature = inspect.signature(service)

        assert "cli_config" not in signature.parameters
        assert (
            signature.parameters["benchmark_run_file"].default
            is inspect.Parameter.empty
        )

    def test_kubernetes_service_args_parse(self) -> None:
        """The operator's serialized-run invocation remains accepted by Cyclopts."""
        command, _bound, _ignored = app.parse_args(
            [
                "--type",
                "worker",
                "--benchmark-run",
                "/etc/aiperf/run_config.json",
            ]
        )

        assert command is service

    def test_forwards_all_arguments(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that service_id is forwarded to bootstrap."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            service_id="worker-1",
        )

        mock_bootstrap.assert_called_once()
        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["service_type"] is service_type
        assert call_kwargs["service_id"] == "worker-1"
        assert "run" in call_kwargs

    def test_default_optional_arguments(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that optional arguments default to None."""
        service(service_type=service_type, benchmark_run_file=benchmark_run_file)

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["service_id"] is None

    def test_health_port_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that health_port sets Environment.SERVICE health settings."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_port=9090,
        )

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_PORT == 9090

    def test_health_host_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that health_host sets Environment.SERVICE health settings."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host="0.0.0.0",
        )

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"

    def test_health_host_and_port_set_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that both health_host and health_port set Environment.SERVICE health settings."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host="0.0.0.0",
            health_port=8081,
        )

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"
        assert Environment.SERVICE.HEALTH_PORT == 8081

    def test_none_health_args_do_not_modify_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that None health args leave Environment.SERVICE unchanged."""
        original_enabled = Environment.SERVICE.HEALTH_ENABLED
        original_host = Environment.SERVICE.HEALTH_HOST
        original_port = Environment.SERVICE.HEALTH_PORT

        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host=None,
            health_port=None,
        )

        assert original_enabled == Environment.SERVICE.HEALTH_ENABLED
        assert original_host == Environment.SERVICE.HEALTH_HOST
        assert original_port == Environment.SERVICE.HEALTH_PORT

    def test_health_args_not_passed_to_bootstrap(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that health args are not forwarded to bootstrap_and_run_service."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host="0.0.0.0",
            health_port=8080,
        )

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert "health_host" not in call_kwargs
        assert "health_port" not in call_kwargs

    def test_benchmark_run_rehydrates_all_secret_backed_endpoint_fields(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        run_file = _write_benchmark_run(
            tmp_path / "run.json",
            api_key="serialized-api-key",
            headers={"Authorization": "Bearer serialized-header"},
            urls=["http://user:password@server/v1"],
        )
        monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
        monkeypatch.setenv(
            "AIPERF_INJECTED_HEADERS",
            '{"Authorization":"Bearer secret-header"}',
        )
        monkeypatch.setenv(
            "AIPERF_INJECTED_ENDPOINT_URLS",
            '["http://real-user:real-password@server/v1"]',
        )

        service(service_type=service_type, benchmark_run_file=run_file)

        endpoint = mock_bootstrap.call_args.kwargs["run"].cfg.endpoint
        assert endpoint.api_key == "secret-api-key"
        assert endpoint.headers["Authorization"] == "Bearer secret-header"
        assert endpoint.urls == ["http://real-user:real-password@server/v1"]
        assert "OPENAI_API_KEY" not in __import__("os").environ
        assert "AIPERF_INJECTED_HEADERS" not in __import__("os").environ
        assert "AIPERF_INJECTED_ENDPOINT_URLS" not in __import__("os").environ

    def test_private_api_key_injection_precedes_openai_alias(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        run_file = _write_benchmark_run(tmp_path / "run.json", api_key="serialized")
        monkeypatch.setenv("AIPERF_INJECTED_API_KEY", "private-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "compatibility-secret")

        service(service_type=service_type, benchmark_run_file=run_file)

        endpoint = mock_bootstrap.call_args.kwargs["run"].cfg.endpoint
        assert endpoint.api_key == "private-secret"

    def test_benchmark_run_rejects_unresolved_redacted_credentials(
        self,
        tmp_path: Path,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        run_file = _write_benchmark_run(tmp_path / "run.json", api_key="serialized")

        with pytest.raises(SystemExit):
            service(service_type=service_type, benchmark_run_file=run_file)

        mock_bootstrap.assert_not_called()

    def test_unreadable_benchmark_run_fails_before_bootstrap(
        self,
        tmp_path: Path,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Missing or unsafe run files cannot fall back to local CLI config."""
        with pytest.raises(SystemExit):
            service(
                service_type=service_type,
                benchmark_run_file=tmp_path / "missing.json",
            )

        mock_bootstrap.assert_not_called()


class TestDatasetPreflightGate:
    """Only `dataset_manager` may run the public-dataset preflight.

    It is the sole container that composes the dataset, so its configure step
    is what `AIPERF_DATASET_CONFIGURATION_TIMEOUT` bounds. Letting every other
    pod run the preflight would have each of them separately download the same
    multi-GB resolution into a private filesystem it never reads.
    """

    def test_dataset_manager_runs_the_preflight(
        self, mock_bootstrap: MagicMock, benchmark_run_file: Path
    ) -> None:
        from aiperf.plugin.enums import ServiceType

        with patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run"
        ) as mock_preflight:
            service(
                service_type=ServiceType.DATASET_MANAGER,
                benchmark_run_file=benchmark_run_file,
            )

        mock_preflight.assert_called_once()
        assert "run" in mock_bootstrap.call_args.kwargs

    @pytest.mark.parametrize(
        "service_type_name",
        ["WORKER", "TIMING_MANAGER", "RECORDS_MANAGER", "SYSTEM_CONTROLLER"],
    )  # fmt: skip
    def test_other_services_skip_the_preflight(
        self,
        mock_bootstrap: MagicMock,
        benchmark_run_file: Path,
        service_type_name: str,
    ) -> None:
        from aiperf.plugin.enums import ServiceType

        with patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run"
        ) as mock_preflight:
            service(
                service_type=getattr(ServiceType, service_type_name),
                benchmark_run_file=benchmark_run_file,
            )

        mock_preflight.assert_not_called()
        assert "run" in mock_bootstrap.call_args.kwargs


_RAW_DATASET_URL = "https://bucket.s3.amazonaws.com/t.jsonl?X-Amz-Signature=topsecret"


def _write_remote_dataset_run(path: Path) -> Path:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://server/v1"], "type": "chat"},
            "datasets": [
                {
                    "name": "main",
                    "type": "file",
                    "path": {"ref": _RAW_DATASET_URL, "resolved_version": "etag:abc"},
                    "format": "mooncake_trace",
                }
            ],
            "phases": [
                {"name": "profiling", "type": "concurrency", "requests": 4, "concurrency": 1}
            ],
        }
    )  # fmt: skip
    run = BenchmarkRun(benchmark_id="job", cfg=cfg, artifact_dir=path.parent)
    path.write_bytes(orjson.dumps(run.model_dump(mode="json", exclude_none=True)))
    return path


class TestDatasetUrlRehydration:
    """Presigned dataset URLs reach only the dataset-manager process."""

    def test_dataset_manager_rehydrates_presigned_dataset_before_preflight(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
    ) -> None:
        from aiperf.plugin.enums import ServiceType

        run_file = _write_remote_dataset_run(tmp_path / "run.json")
        assert b"topsecret" not in run_file.read_bytes()
        monkeypatch.setenv(
            "AIPERF_INJECTED_DATASET_URLS", orjson.dumps([_RAW_DATASET_URL]).decode()
        )
        seen: list[str] = []
        with patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run",
            side_effect=lambda run: seen.append(run.cfg.datasets[0].path.ref),
        ):
            service(
                service_type=ServiceType.DATASET_MANAGER, benchmark_run_file=run_file
            )

        assert seen == [_RAW_DATASET_URL]
        assert os.getenv("AIPERF_INJECTED_DATASET_URLS") is None
        mock_bootstrap.assert_called_once()
        path = mock_bootstrap.call_args.kwargs["run"].cfg.datasets[0].path
        assert path.ref == _RAW_DATASET_URL
        assert path.canonical == RemoteRef(ref=_RAW_DATASET_URL).canonical
        assert path.resolved_version == "etag:abc"

    def test_dataset_manager_without_injected_url_fails_before_bootstrap(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
        termination_log: Path,
    ) -> None:
        from aiperf.plugin.enums import ServiceType

        run_file = _write_remote_dataset_run(tmp_path / "run.json")
        monkeypatch.delenv("AIPERF_INJECTED_DATASET_URLS", raising=False)
        buffer = io.StringIO()
        monkeypatch.setattr(
            cli_utils, "console", Console(file=buffer, force_terminal=False, width=200)
        )
        with pytest.raises(SystemExit):
            service(
                service_type=ServiceType.DATASET_MANAGER, benchmark_run_file=run_file
            )
        mock_bootstrap.assert_not_called()
        output = buffer.getvalue()
        assert "AIPERF_INJECTED_DATASET_URLS has no URL for it" in " ".join(
            output.replace("│", " ").split()
        )
        assert "topsecret" not in output
        assert "has no URL for it" in termination_log.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "service_type",
        [
            param(ServiceType.SYSTEM_CONTROLLER, id="system-controller"),
            param(ServiceType.TIMING_MANAGER, id="timing-manager"),
            param(ServiceType.RECORDS_MANAGER, id="records-manager"),
            param(ServiceType.API, id="api"),
            param(ServiceType.WORKER, id="worker"),
        ],
    )  # fmt: skip
    def test_other_services_drop_injected_dataset_urls(
        self,
        service_type: ServiceType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
    ) -> None:
        run_file = _write_remote_dataset_run(tmp_path / "run.json")
        monkeypatch.setenv(
            "AIPERF_INJECTED_DATASET_URLS", orjson.dumps([_RAW_DATASET_URL]).decode()
        )
        service(service_type=service_type, benchmark_run_file=run_file)

        assert os.getenv("AIPERF_INJECTED_DATASET_URLS") is None
        run = mock_bootstrap.call_args.kwargs["run"]
        assert "topsecret" not in run.cfg.datasets[0].path.ref

    def test_dataset_manager_failure_writes_termination_message_and_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
        termination_log: Path,
    ) -> None:
        run_file = _write_remote_dataset_run(tmp_path / "run.json")
        monkeypatch.setenv(
            "AIPERF_INJECTED_DATASET_URLS", orjson.dumps([_RAW_DATASET_URL]).decode()
        )
        error = RemoteSourceError(
            ref=_RAW_DATASET_URL,
            kind=RemoteErrorKind.GATED,
            stage=RemoteStage.POD,
            what="it is gated",
            hint="Accept its terms",
        )
        with (
            patch(
                "aiperf.cli_runner._preflight.preflight_datasets_for_run",
                side_effect=error,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            service(
                service_type=ServiceType.DATASET_MANAGER, benchmark_run_file=run_file
            )
        assert exc_info.value.code == 1
        message = termination_log.read_text(encoding="utf-8")
        assert message.startswith("[dataset-manager pod] Cannot use ")
        assert "topsecret" not in message
        mock_bootstrap.assert_not_called()
