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
from pytest import param

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.config import (
    ArtifactsConfig,
    BenchmarkConfig,
    EndpointConfig,
    MLflowConfig,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.mlflow_data_exporter import MLflowDataExporter
from aiperf.plugin.enums import EndpointType


def _write_artifact(path: Path, content: str = "test") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_mlflow_cfg(
    tmp_path: Path,
    *,
    tracking_uri: str | None = "http://mlflow:5000",
    experiment: str = "aiperf-tests",
    run_name: str | None = None,
    tags: str | None = None,
    artifact_globs: list[str] | None = None,
    profiling: dict[str, Any] | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        model="test-model",
        endpoint=EndpointConfig(
            urls=["http://localhost:8000"],
            type=EndpointType.CHAT,
        ),
        dataset={"type": "synthetic"},
        profiling=profiling or {"type": "concurrency", "requests": 1, "concurrency": 1},
        artifacts=ArtifactsConfig(dir=tmp_path),
        mlflow=MLflowConfig(
            tracking_uri=tracking_uri,
            experiment=experiment,
            run_name=run_name,
            tags=tags,
            artifact_globs=artifact_globs,
        ),
    )


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
                count=10,
                sum=425.0,
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
def mlflow_cfg(tmp_path: Path) -> BenchmarkConfig:
    return _make_mlflow_cfg(
        tmp_path,
        run_name="nightly-run",
        tags="team:perf,env:ci",
    )


class TestMLflowDataExporter:
    @pytest.mark.parametrize(
        ("tag", "expected_key"),
        [
            param("http_req_waiting", "system/http_req_waiting", id="http-timing"),
            param("http_req_connection_reused", "system/http_req_connection_reused", id="http-counter"),
            param("credit_drop_latency", "system/credit_drop_latency", id="internal"),
            param("inter_chunk_latency", "system/inter_chunk_latency", id="stream-diagnostic"),
            param("error_isl", "system/error_isl", id="error-input"),
            param("total_error_isl", "system/total_error_isl", id="total-error-input"),
            param("num_images", "system/num_images", id="image-count"),
            param("video_peak_memory", "system/video_peak_memory", id="video-memory"),
            param("usage_prompt_cache_read_tokens", "system/usage_prompt_cache_read_tokens", id="usage-cache"),
            param("total_usage_completion_tokens", "system/total_usage_completion_tokens", id="total-usage"),
            param("overall_usage_prompt_cache_read_pct", "system/overall_usage_prompt_cache_read_pct", id="cache-percent"),
            param("usage_prompt_tokens_diff_pct", "system/usage_prompt_tokens_diff_pct", id="usage-discrepancy"),
            param("requested_osl", "system/requested_osl", id="output-target"),
            param("osl_mismatch_count", "system/osl_mismatch_count", id="output-mismatch"),
            param("nvidia_total_gpu_power", "system/nvidia_total_gpu_power", id="nvidia-power"),
            param("amd_energy_per_user", "system/amd_energy_per_user", id="amd-energy"),
            param("nvidia_energy_delay_product", "system/nvidia_energy_delay_product", id="nvidia-efficiency"),
            param("amd_goodput_per_watt", "system/amd_goodput_per_watt", id="amd-efficiency"),
            param("time_to_first_token", "time_to_first_token", id="primary-latency"),
            param("request_throughput", "request_throughput", id="primary-throughput"),
            param("error_request_count", "error_request_count", id="primary-error-count"),
            param("benchmark_duration", "benchmark_duration", id="hidden-primary"),
            param("min_request_timestamp", "min_request_timestamp", id="internal-primary-start"),
            param("max_response_timestamp", "max_response_timestamp", id="internal-primary-end"),
            param("accuracy.correct", "accuracy.correct", id="model-quality"),
            param("video_inference_time", "video_inference_time", id="video-latency"),
            param("custom_latency", "custom_latency", id="custom"),
            param("http_req_custom", "http_req_custom", id="custom-http-prefix"),
            param("usage_custom", "usage_custom", id="custom-usage-prefix"),
            param("nvidia_custom_power", "nvidia_custom_power", id="custom-vendor-prefix"),
            param("system/http_req_waiting", "system/http_req_waiting", id="already-prefixed"),
        ],
    )  # fmt: skip
    def test_build_metric_payload_uses_system_namespace_for_diagnostics(
        self,
        tag: str,
        expected_key: str,
        sample_results: ProfileResults,
        mlflow_cfg: BenchmarkConfig,
    ) -> None:
        sample_results.records = [MetricResult(tag=tag, header=tag, unit="ms", avg=2.5)]
        exporter = MLflowDataExporter(
            ExporterConfig(
                results=sample_results, cfg=mlflow_cfg, telemetry_results=None
            )
        )

        assert exporter._build_metric_payload() == {
            expected_key: 2.5,
            "aiperf.completed_requests": 10.0,
            "aiperf.total_expected_requests": 12.0,
        }

    @pytest.mark.asyncio
    async def test_export_categorizes_metrics_without_changing_records(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_results: ProfileResults,
        mlflow_cfg: BenchmarkConfig,
    ) -> None:
        sample_results.records = [
            MetricResult(
                tag="http_req_waiting",
                header="HTTP waiting",
                unit="ms",
                avg=2.5,
                p95=4.0,
                count=2,
                sum=5.0,
            ),
            MetricResult(
                tag="time_to_first_token",
                header="TTFT",
                unit="ms",
                avg=12.0,
            ),
            MetricResult(
                tag="usage_completion_tokens",
                header="Usage",
                unit="tokens",
                avg=3.0,
            ),
            MetricResult(
                tag="amd_total_gpu_energy",
                header="Energy",
                unit="J",
                avg=9.0,
            ),
        ]
        original_results = sample_results.model_dump()
        state = _install_fake_mlflow_modules(monkeypatch)
        exporter = MLflowDataExporter(
            ExporterConfig(
                results=sample_results, cfg=mlflow_cfg, telemetry_results=None
            )
        )

        await asyncio.to_thread(exporter._export_sync)

        assert len(state["log_batch_calls"]) == 1
        metrics = state["log_batch_calls"][0]["metrics"]
        expected = {
            "system/http_req_waiting": 2.5,
            "system/http_req_waiting.p95": 4.0,
            "system/http_req_waiting.count": 2.0,
            "system/http_req_waiting.sum": 5.0,
            "time_to_first_token": 12.0,
            "system/usage_completion_tokens": 3.0,
            "system/amd_total_gpu_energy": 9.0,
            "aiperf.completed_requests": 10.0,
            "aiperf.total_expected_requests": 12.0,
        }
        assert len(metrics) == len(expected)
        assert {metric.key: metric.value for metric in metrics} == expected
        assert sample_results.model_dump() == original_results

    @pytest.mark.parametrize(
        ("field", "reverse"),
        [
            param("avg", False, id="avg-generated-first"),
            param("avg", True, id="avg-prefixed-first"),
            param("p95", False, id="p95-generated-first"),
            param("p95", True, id="p95-prefixed-first"),
        ],
    )
    def test_build_metric_payload_rejects_normalized_key_collisions(
        self,
        field: str,
        reverse: bool,
        sample_results: ProfileResults,
        mlflow_cfg: BenchmarkConfig,
    ) -> None:
        values = {field: 2.5}
        records = [
            MetricResult(
                tag="http_req_waiting",
                header="HTTP waiting",
                unit="ms",
                **values,
            ),
            MetricResult(
                tag="system/http_req_waiting",
                header="Already prefixed HTTP waiting",
                unit="ms",
                **values,
            ),
        ]
        sample_results.records = list(reversed(records)) if reverse else records
        exporter = MLflowDataExporter(
            ExporterConfig(
                results=sample_results, cfg=mlflow_cfg, telemetry_results=None
            )
        )

        expected_key = (
            "system/http_req_waiting"
            if field == "avg"
            else f"system/http_req_waiting.{field}"
        )
        with pytest.raises(
            ValueError,
            match=rf"Duplicate MLflow metric key '{expected_key}'",
        ):
            exporter._build_metric_payload()

    def test_disabled_without_tracking_uri(
        self, tmp_path: Path, sample_results: ProfileResults
    ) -> None:
        config = ExporterConfig(
            results=sample_results,
            cfg=_make_mlflow_cfg(tmp_path, tracking_uri=None),
            telemetry_results=None,
        )
        with pytest.raises(
            DataExporterDisabled,
            match="set --mlflow-tracking-uri to enable",
        ):
            MLflowDataExporter(config)

    def test_disabled_without_results(self, mlflow_cfg: BenchmarkConfig) -> None:
        config = ExporterConfig(
            results=None,
            cfg=mlflow_cfg,
            telemetry_results=None,
        )
        with pytest.raises(DataExporterDisabled, match="no profile results"):
            MLflowDataExporter(config)

    @pytest.mark.parametrize(
        ("profiling", "expected_rate"),
        [
            param({"type": "poisson", "rate": 5.0, "requests": 10}, "5.0", id="poisson"),
            param({"type": "constant", "rate": 2.5, "requests": 5}, "2.5", id="constant"),
            param({"type": "concurrency", "requests": 1, "concurrency": 1}, None, id="concurrency_has_no_rate"),
        ],
    )  # fmt: skip
    def test_build_param_payload_rate_phase_emits_request_rate(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
        profiling: dict[str, Any],
        expected_rate: str | None,
    ) -> None:
        config = ExporterConfig(
            results=sample_results,
            cfg=_make_mlflow_cfg(tmp_path, profiling=profiling),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(config)

        params = exporter._build_param_payload()

        assert params["timing.mode"] == profiling["type"]
        if expected_rate is None:
            assert "loadgen.request_rate" not in params
        else:
            assert params["loadgen.request_rate"] == expected_rate

    def test_build_param_payload_stray_rate_attr_on_non_rate_phase_omits_rate(
        self,
        tmp_path: Path,
        sample_results: ProfileResults,
    ) -> None:
        """The exporter must read the rate via get_phase_rate (isinstance-gated),
        not attribute probing: a rate-shaped attribute on a concurrency phase
        must not emit loadgen.request_rate."""
        config = ExporterConfig(
            results=sample_results,
            cfg=_make_mlflow_cfg(
                tmp_path,
                profiling={"type": "concurrency", "requests": 1, "concurrency": 1},
            ),
            telemetry_results=None,
        )
        exporter = MLflowDataExporter(config)
        phase = exporter._cfg.get_profiling_phases()[0]
        object.__setattr__(phase, "rate", 7.5)

        params = exporter._build_param_payload()

        assert "loadgen.request_rate" not in params

    @pytest.mark.asyncio
    async def test_export_uploads_batch_and_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        sample_results: ProfileResults,
        mlflow_cfg: BenchmarkConfig,
    ) -> None:
        _write_artifact(tmp_path / "profile_export_aiperf.json")
        _write_artifact(tmp_path / "profile_export_aiperf_timeslices.json")
        _write_artifact(tmp_path / "summary.csv")
        _write_artifact(tmp_path / "plots" / "request_throughput.png")
        _write_artifact(tmp_path / "plots" / "custom" / "panel.html")

        state = _install_fake_mlflow_modules(monkeypatch)
        config = ExporterConfig(
            results=sample_results,
            cfg=mlflow_cfg,
            telemetry_results=None,
            run=types.SimpleNamespace(benchmark_id="bench-upload-001"),
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
        assert metric_map["request_throughput.count"] == 10.0
        assert metric_map["request_throughput.sum"] == 425.0
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
        cfg = _make_mlflow_cfg(
            tmp_path,
            artifact_globs=["plots/**/*.png"],
        )
        config = ExporterConfig(
            results=sample_results,
            cfg=cfg,
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
        mlflow_cfg: BenchmarkConfig,
    ) -> None:
        _write_artifact(tmp_path / "profile_export_aiperf.json")
        live_run_id = "live-run-555"
        benchmark_id = "bench-live-555"
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
            cfg=mlflow_cfg,
            telemetry_results=None,
            run=types.SimpleNamespace(benchmark_id=benchmark_id),
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
        mlflow_cfg: BenchmarkConfig,
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
                cfg=mlflow_cfg,
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
        mlflow_cfg: BenchmarkConfig,
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
            cfg=mlflow_cfg,
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
        mlflow_cfg: BenchmarkConfig,
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
            cfg=mlflow_cfg,
            telemetry_results=None,
        )
        # Generous ceiling: the child exits fast (os._exit right after import),
        # so join() returns as soon as it dies -- but under `pytest -n auto` load
        # the spawned child's re-import can exceed a tight 10s window, tripping the
        # timeout path and masking the exitcode warning this test asserts on.
        await mlflow_export_subprocess.export_with_timeout(
            config, export_timeout=60.0, warn=_collect
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
