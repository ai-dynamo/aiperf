# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MLflow post-run data exporter."""

from __future__ import annotations

import asyncio
import builtins
import sys
import types
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.mlflow_data_exporter import MLflowDataExporter
from aiperf.plugin.enums import EndpointType


def _write_artifact(path: Path, content: str = "test") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _install_fake_mlflow_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install fake mlflow modules into sys.modules and return call state."""
    state: dict[str, Any] = {
        "tracking_uris": [],
        "experiments": [],
        "run_names": [],
        "run_ids": [],
        "log_batch_calls": [],
        "artifacts": [],
        "artifact_contents": {},
        # run_id -> pre-existing run_name (simulates a live-streaming run that
        # MLflow already auto-named before the deferred exporter starts).
        "live_run_names": {},
    }
    default_run_id = "run-123"

    class FakeMetric:
        def __init__(self, key: str, value: float, timestamp: int, step: int) -> None:
            self.key = key
            self.value = value
            self.timestamp = timestamp
            self.step = step

    class FakeParam:
        def __init__(self, key: str, value: str) -> None:
            self.key = key
            self.value = value

    class FakeRunTag:
        def __init__(self, key: str, value: str) -> None:
            self.key = key
            self.value = value

    class FakeMlflowClient:
        def log_batch(
            self,
            *,
            run_id: str,
            metrics: list[FakeMetric],
            params: list[FakeParam],
            tags: list[FakeRunTag],
        ) -> None:
            state["log_batch_calls"].append(
                {
                    "run_id": run_id,
                    "metrics": metrics,
                    "params": params,
                    "tags": tags,
                }
            )

    class FakeRunContext:
        def __init__(self, run_id: str, run_name: str | None) -> None:
            self._run_id = run_id
            self._run_name = run_name

        def __enter__(self) -> Any:
            info = types.SimpleNamespace(run_id=self._run_id, run_name=self._run_name)
            return types.SimpleNamespace(info=info)

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    mlflow_module = types.ModuleType("mlflow")
    entities_module = types.ModuleType("mlflow.entities")
    tracking_module = types.ModuleType("mlflow.tracking")

    def set_tracking_uri(uri: str) -> None:
        state["tracking_uris"].append(uri)

    def set_experiment(name: str) -> None:
        state["experiments"].append(name)

    def start_run(
        run_name: str | None = None, run_id: str | None = None, **kwargs: Any
    ) -> FakeRunContext:
        state["run_names"].append(run_name)
        state["run_ids"].append(run_id)
        selected_run_id = run_id or default_run_id
        # Mimic MLflow: when an existing run is reused via run_id, the stored
        # run_name is whatever it was assigned at creation time (tracked via
        # `state["live_run_names"]` if the test installed one).
        if run_id is not None:
            selected_run_name = state.get("live_run_names", {}).get(run_id)
        else:
            selected_run_name = run_name
        return FakeRunContext(selected_run_id, selected_run_name)

    def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        state["artifacts"].append((local_path, artifact_path))
        state["artifact_contents"][local_path] = Path(local_path).read_text(
            encoding="utf-8"
        )

    mlflow_module.set_tracking_uri = set_tracking_uri  # type: ignore[attr-defined]
    mlflow_module.set_experiment = set_experiment  # type: ignore[attr-defined]
    mlflow_module.start_run = start_run  # type: ignore[attr-defined]
    mlflow_module.log_artifact = log_artifact  # type: ignore[attr-defined]
    mlflow_module.entities = entities_module  # type: ignore[attr-defined]
    mlflow_module.tracking = tracking_module  # type: ignore[attr-defined]

    entities_module.Metric = FakeMetric  # type: ignore[attr-defined]
    entities_module.Param = FakeParam  # type: ignore[attr-defined]
    entities_module.RunTag = FakeRunTag  # type: ignore[attr-defined]
    tracking_module.MlflowClient = FakeMlflowClient  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities_module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_module)
    return state


@pytest.fixture
def sample_results() -> ProfileResults:
    return ProfileResults(
        records=[
            MetricResult(
                tag="request_throughput",
                header="Request Throughput",
                unit="req/s",
                avg=42.5,
            ),
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=None,
            ),
        ],
        total_expected=12,
        completed=10,
        start_ns=0,
        end_ns=1,
        was_cancelled=False,
        error_summary=[],
    )


@pytest.fixture
def mlflow_user_config(tmp_path: Path) -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            type=EndpointType.CHAT,
            model_names=["test-model"],
            urls=["http://localhost:8000"],
        ),
        output=OutputConfig(artifact_directory=tmp_path),
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment="aiperf-tests",
        mlflow_run_name="nightly-run",
        mlflow_tags=[("team", "perf"), ("env", "ci")],
    )


class TestMLflowDataExporter:
    def test_disabled_without_tracking_uri(
        self, tmp_path: Path, sample_results: ProfileResults
    ) -> None:
        config = ExporterConfig(
            results=sample_results,
            user_config=UserConfig(
                endpoint=EndpointConfig(
                    type=EndpointType.CHAT,
                    model_names=["test-model"],
                ),
                output=OutputConfig(artifact_directory=tmp_path),
            ),
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        with pytest.raises(
            DataExporterDisabled,
            match="set --mlflow-tracking-uri to enable",
        ):
            MLflowDataExporter(config)

    def test_disabled_without_results(self, mlflow_user_config: UserConfig) -> None:
        config = ExporterConfig(
            results=None,
            user_config=mlflow_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        with pytest.raises(DataExporterDisabled, match="no profile results"):
            MLflowDataExporter(config)

    @pytest.mark.asyncio
    async def test_export_uploads_batch_and_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_results: ProfileResults,
        mlflow_user_config: UserConfig,
    ) -> None:
        _write_artifact(tmp_path / "profile_export_aiperf.json")
        _write_artifact(tmp_path / "profile_export_aiperf_timeslices.json")
        _write_artifact(tmp_path / "summary.csv")
        _write_artifact(tmp_path / "plots" / "request_throughput.png")
        _write_artifact(tmp_path / "plots" / "custom" / "panel.html")

        state = _install_fake_mlflow_modules(monkeypatch)
        config = ExporterConfig(
            results=sample_results,
            user_config=mlflow_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )

        exporter = MLflowDataExporter(config)
        await asyncio.to_thread(exporter._export_sync)

        assert state["tracking_uris"] == ["http://mlflow:5000"]
        assert state["experiments"] == ["aiperf-tests"]
        assert state["run_names"] == ["nightly-run"]
        assert len(state["log_batch_calls"]) == 1

        batch = state["log_batch_calls"][0]
        assert batch["run_id"] == "run-123"

        metric_map = {metric.key: metric.value for metric in batch["metrics"]}
        assert metric_map["request_throughput"] == 42.5
        assert metric_map["aiperf.completed_requests"] == 10.0
        assert metric_map["aiperf.total_expected_requests"] == 12.0
        assert "time_to_first_token" not in metric_map

        param_map = {param.key: param.value for param in batch["params"]}
        assert param_map["endpoint.type"] == "chat"
        assert param_map["endpoint.models"] == "test-model"
        assert param_map["endpoint.urls"] == "http://localhost:8000"

        tag_map = {tag.key: tag.value for tag in batch["tags"]}
        assert tag_map["team"] == "perf"
        assert tag_map["env"] == "ci"
        assert tag_map["aiperf.was_cancelled"] == "false"
        assert "aiperf.version" in tag_map
        assert tag_map["benchmark_id"]

        uploaded = [
            (Path(local_path).relative_to(tmp_path).as_posix(), artifact_path)
            for local_path, artifact_path in state["artifacts"]
        ]
        assert (
            "profile_export_aiperf.json",
            "exports",
        ) in uploaded
        assert (
            "profile_export_aiperf_timeslices.json",
            "exports",
        ) in uploaded
        assert ("summary.csv", "exports") in uploaded
        assert ("plots/request_throughput.png", "plots") in uploaded
        assert ("plots/custom/panel.html", "plots/custom") in uploaded

        # Verify de-duplication across overlapping glob patterns.
        assert (
            sum(
                1
                for local_path, _ in uploaded
                if local_path == "profile_export_aiperf_timeslices.json"
            )
            == 1
        )

        metadata = orjson.loads(
            (tmp_path / "mlflow_export.json").read_text(encoding="utf-8")
        )
        assert metadata["run_id"] == "run-123"
        assert metadata["run_name"] == "nightly-run"
        assert metadata["tracking_uri"] == "http://mlflow:5000"
        assert metadata["experiment"] == "aiperf-tests"
        assert "request_throughput" in metadata["metric_keys"]
        assert set(metadata["uploaded_artifacts"]) == {
            "profile_export_aiperf.json",
            "profile_export_aiperf_timeslices.json",
            "summary.csv",
            "plots/request_throughput.png",
            "plots/custom/panel.html",
            "mlflow_export.json",
        }

    @pytest.mark.asyncio
    async def test_export_respects_custom_artifact_globs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_results: ProfileResults,
    ) -> None:
        _write_artifact(tmp_path / "profile_export_aiperf.json")
        _write_artifact(tmp_path / "plots" / "latency.png")

        state = _install_fake_mlflow_modules(monkeypatch)
        user_config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                model_names=["test-model"],
            ),
            output=OutputConfig(artifact_directory=tmp_path),
            mlflow_tracking_uri="http://mlflow:5000",
            mlflow_experiment="aiperf-tests",
            mlflow_artifact_globs=["plots/**/*.png"],
        )
        config = ExporterConfig(
            results=sample_results,
            user_config=user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(config)
        await asyncio.to_thread(exporter._export_sync)

        uploaded = [
            Path(local_path).relative_to(tmp_path).as_posix()
            for local_path, _ in state["artifacts"]
        ]
        assert "plots/latency.png" in uploaded
        assert "mlflow_export.json" in uploaded

    @pytest.mark.asyncio
    async def test_export_reuses_live_streaming_run_when_metadata_matches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_results: ProfileResults,
        mlflow_user_config: UserConfig,
    ) -> None:
        _write_artifact(tmp_path / "profile_export_aiperf.json")
        live_run_id = "live-run-555"
        benchmark_id = mlflow_user_config.benchmark_id
        assert benchmark_id is not None
        metadata = {
            "tracking_uri": "http://mlflow:5000",
            "experiment": "aiperf-tests",
            "run_id": live_run_id,
            "run_name": "live-stream-run",
            "benchmark_id": benchmark_id,
            "live_streaming": True,
        }
        (tmp_path / "mlflow_export.json").write_bytes(orjson.dumps(metadata))

        state = _install_fake_mlflow_modules(monkeypatch)
        # Simulate the live-streaming fanout having let MLflow auto-name the
        # run (the user did not pass --mlflow-run-name). The deferred
        # exporter must propagate that MLflow-assigned name into the final
        # mlflow_export.json, not the stale name from the live metadata.
        mlflow_assigned_name = "bustling-kit-384"
        state["live_run_names"][live_run_id] = mlflow_assigned_name

        config = ExporterConfig(
            results=sample_results,
            user_config=mlflow_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(config)
        await asyncio.to_thread(exporter._export_sync)

        assert state["run_ids"] == [live_run_id]
        assert state["run_names"] == [None]
        assert state["log_batch_calls"][0]["run_id"] == live_run_id
        uploaded = [
            (Path(local_path).relative_to(tmp_path).as_posix(), artifact_path)
            for local_path, artifact_path in state["artifacts"]
        ]
        assert ("mlflow_export.json", "exports") in uploaded

        written_metadata = orjson.loads(
            (tmp_path / "mlflow_export.json").read_text(encoding="utf-8")
        )
        assert written_metadata["run_id"] == live_run_id
        assert written_metadata["reused_live_run"] is True
        # Regression: the final metadata must use the MLflow-assigned run name
        # (not the stale placeholder written by the live-streaming fanout).
        assert written_metadata["run_name"] == mlflow_assigned_name
        uploaded_metadata = orjson.loads(
            state["artifact_contents"][str(tmp_path / "mlflow_export.json")]
        )
        assert uploaded_metadata == written_metadata

    def test_upload_artifacts_to_run_supports_plot_only_upload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _write_artifact(tmp_path / "ttft_over_time.png")
        _write_artifact(tmp_path / "custom" / "panel.html")

        state = _install_fake_mlflow_modules(monkeypatch)
        uploaded = MLflowDataExporter.upload_artifacts_to_run(
            tracking_uri="http://mlflow:5000",
            run_id="existing-run-789",
            artifact_directory=tmp_path,
            artifact_files=[
                tmp_path / "ttft_over_time.png",
                tmp_path / "custom" / "panel.html",
            ],
        )

        assert state["tracking_uris"] == ["http://mlflow:5000"]
        assert state["run_ids"] == ["existing-run-789"]
        assert uploaded == ["ttft_over_time.png", "custom/panel.html"]
        assert (
            "ttft_over_time.png",
            "plots",
        ) in [
            (Path(local_path).relative_to(tmp_path).as_posix(), artifact_path)
            for local_path, artifact_path in state["artifacts"]
        ]

    @pytest.mark.asyncio
    async def test_export_raises_when_mlflow_dependency_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_results: ProfileResults,
        mlflow_user_config: UserConfig,
    ) -> None:
        for module_name in ("mlflow", "mlflow.entities", "mlflow.tracking"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

        original_import = builtins.__import__

        def _raise_for_mlflow(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "mlflow" or name.startswith("mlflow."):
                raise ImportError("mlflow intentionally unavailable in test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise_for_mlflow)

        exporter = MLflowDataExporter(
            ExporterConfig(
                results=sample_results,
                user_config=mlflow_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )
        )
        with pytest.raises(RuntimeError, match="optional MLflow dependency"):
            await asyncio.to_thread(exporter._export_sync)

    @pytest.mark.asyncio
    async def test_export_subprocess_terminates_on_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_results: ProfileResults,
        mlflow_user_config: UserConfig,
    ) -> None:
        """Regression: ``export()`` must terminate a hung subprocess on timeout.

        Simulates a wedged MLflow call by stubbing the subprocess entrypoint
        with a sleeper at module scope (spawn can't pickle local closures).
        With a 1s timeout the outer ``export()`` returns quickly (not 30s
        default) and the worker subprocess is no longer alive.
        """
        import time as time_module

        from aiperf.common import environment as env_module

        monkeypatch.setattr(
            "aiperf.exporters.mlflow_export_subprocess.run_export_in_subprocess",
            _hang_forever_subprocess_entry,
        )
        monkeypatch.setattr(
            env_module.Environment.MLFLOW, "EXPORT_TIMEOUT_SECONDS", 1.0
        )

        config = ExporterConfig(
            results=sample_results,
            user_config=mlflow_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(config)

        start = time_module.monotonic()
        await exporter.export()
        elapsed = time_module.monotonic() - start

        # Must return within a small multiple of the 1s timeout, not 30s.
        assert elapsed < 10.0, f"export() did not honor timeout: elapsed={elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_export_subprocess_silent_crash_warns_with_exitcode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_results: ProfileResults,
        mlflow_user_config: UserConfig,
    ) -> None:
        """Regression: if the spawn child dies before writing to the queue
        (spawn bootstrap failure, SIGKILL/OOM, native crash), the parent must
        warn using ``process.exitcode`` instead of returning silently.
        """
        from aiperf.exporters import mlflow_export_subprocess

        monkeypatch.setattr(
            "aiperf.exporters.mlflow_export_subprocess.run_export_in_subprocess",
            _silent_crash_subprocess_entry,
        )

        warnings: list[str] = []

        def _collect(message: str) -> None:
            warnings.append(message)

        config = ExporterConfig(
            results=sample_results,
            user_config=mlflow_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )
        await mlflow_export_subprocess.export_with_timeout(
            config, export_timeout=10.0, warn=_collect
        )

        assert any(
            "exited with non-zero status" in msg and "exitcode=137" in msg
            for msg in warnings
        ), f"expected exitcode warning, got {warnings!r}"


def _hang_forever_subprocess_entry(exporter_config: Any, result_queue: Any) -> None:
    """Module-level subprocess stub (must be picklable for spawn context)."""
    import time as _time

    _time.sleep(30)
    result_queue.put(None)


def _silent_crash_subprocess_entry(exporter_config: Any, result_queue: Any) -> None:
    """Module-level subprocess stub that exits non-zero without writing to the queue.

    Simulates a spawn bootstrap failure, SIGKILL/OOM, or native crash where
    ``run_export_in_subprocess`` never reaches its ``try/except`` block and the
    queue therefore carries no error message.
    """
    import os

    os._exit(137)
