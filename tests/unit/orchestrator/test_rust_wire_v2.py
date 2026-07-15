# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-v2 BenchmarkRun wire dump stays structural."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.rust_wire import (
    RustWireError,
    build_authored_run_request,
    dump_benchmark_run,
)


def _run(
    artifact_target: Path,
    *,
    dataset: dict | None = None,
    transport: dict | None = None,
    endpoint_url: str = "http://127.0.0.1:8000",
    endpoint_type: str = "chat",
) -> BenchmarkRun:
    benchmark: dict = {
        "models": ["mock-model"],
        "endpoint": {
            "urls": [endpoint_url],
            "type": endpoint_type,
            "streaming": True,
        },
        "dataset": dataset
        or {
            "type": "synthetic",
            "entries": 2,
            "prompts": {"isl": 8, "osl": 2},
        },
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 2,
                "concurrency": 1,
            }
        ],
        "tokenizer": {"name": "builtin"},
        "runtime": {"workers": 3},
        "gpu_telemetry": {"enabled": False},
        "server_metrics": {"enabled": False},
        "artifacts": {"dir": str(artifact_target)},
    }
    if transport is not None:
        benchmark["transport"] = transport
    config = AIPerfConfig.model_validate({"benchmark": benchmark})
    return BenchmarkRun(
        benchmark_id="authored-v2",
        cfg=config.benchmark,
        artifact_dir=artifact_target,
        label="cell",
        trial=2,
        random_seed=17,
    )


@pytest.mark.parametrize("operation", ["validate", "execute"])
def test_v2_envelope_is_benchmark_run_shaped(tmp_path: Path, operation: str) -> None:
    run = _run(tmp_path / "artifacts")

    request = build_authored_run_request(run, operation=operation)

    assert request["protocol_version"] == 2
    assert request["operation"] == operation
    assert "expected_distribution_id" not in request
    authored = request["run"]
    assert authored["benchmark_id"] == "authored-v2"
    assert authored["label"] == "cell"
    assert authored["trial"] == 2
    assert authored["random_seed"] == 17
    assert authored["artifact_dir"] == str(tmp_path / "artifacts")
    assert "identity" not in authored
    assert "artifact_target" not in authored
    assert "resources" not in authored
    assert "workload" not in authored.get("cfg", {})
    assert authored["cfg"]["transport"]["type"] == "http"
    assert authored["cfg"]["endpoint"]["type"] == "chat"
    assert authored["cfg"]["datasets"][0]["type"] == "synthetic"
    assert "logging" not in authored["cfg"]
    assert "wandb" not in authored["cfg"]
    assert "otel" not in authored["cfg"]
    assert "mlflow" not in authored["cfg"]
    assert "resolved" in authored


def test_export_genai_perf_enabled_by_default_json_summary(tmp_path: Path) -> None:
    # Default summary is ["json"]; the native v1 summary sink is the sole emitter
    # of profile_export_aiperf.{json,csv} and enables on that signal.
    run = _run(tmp_path / "artifacts")
    genai_perf = dump_benchmark_run(run)["cfg"]["export"]["genai_perf"]
    assert genai_perf["enabled"] is True


def test_export_genai_perf_disabled_without_json_summary(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    run.cfg.artifacts.summary = False
    export = dump_benchmark_run(run)["cfg"]["export"]
    assert export["genai_perf"] == {"enabled": False}


def test_export_genai_perf_projects_frontend_metadata(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    # The native v1 summary sink is enabled by the default "json" summary; the
    # frontend projection is then included.
    run.cfg.artifacts.summary = ["json"]

    genai_perf = dump_benchmark_run(run)["cfg"]["export"]["genai_perf"]

    assert genai_perf["enabled"] is True
    # header_map carries the registered display header for known metric tags,
    # exactly as native_report._metric_result derives it.
    assert genai_perf["header_map"]["request_latency"] == "Request Latency"
    # filtered / scalar tag sets are sorted registered subsets.
    assert genai_perf["filtered_tags"] == sorted(genai_perf["filtered_tags"])
    assert "request_throughput" in genai_perf["scalar_tags"]
    # Envelope carries the frontend-owned JSON values verbatim.
    envelope = genai_perf["envelope"]
    assert envelope["benchmark_id"] == "authored-v2"
    assert envelope["aiperf_version"]
    assert isinstance(envelope["input_config"], dict)
    assert envelope["run_info"]["benchmark_id"] == "authored-v2"
    assert "start_time" not in envelope and "end_time" not in envelope


def test_export_timeslice_absent_without_slice_duration(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    export = dump_benchmark_run(run)["cfg"]["export"]
    assert "timeslice" not in export


def test_export_timeslice_projects_input_config_and_registry(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    run.cfg.artifacts.slice_duration = 1.0

    timeslice = dump_benchmark_run(run)["cfg"]["export"]["timeslice"]

    # Both files are produced whenever the run configures a slice duration.
    assert timeslice["json"] is True
    assert timeslice["csv"] is True
    # input_config is the exact BenchmarkConfig dump the Python timeslice
    # exporter wraps after the timeslices array.
    assert isinstance(timeslice["input_config"], dict)
    assert timeslice["input_config"]["endpoint"]["type"] == "chat"
    # The registry-derived metric identity mirrors the genai-perf v1 sink.
    assert timeslice["header_map"]["request_latency"] == "Request Latency"
    assert timeslice["filtered_tags"] == sorted(timeslice["filtered_tags"])
    assert "request_throughput" in timeslice["scalar_tags"]


def test_export_server_metrics_absent_when_disabled(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    export = dump_benchmark_run(run)["cfg"]["export"]
    assert "server_metrics" not in export


def test_export_server_metrics_projects_envelope(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    run.cfg.server_metrics.enabled = True
    run.cfg.server_metrics.urls = ["http://127.0.0.1:8000/vllm/metrics"]

    server_metrics = dump_benchmark_run(run)["cfg"]["export"]["server_metrics"]

    # json/csv are the default server-metrics formats.
    assert server_metrics["json"] is True
    assert server_metrics["csv"] is True
    # Envelope: package version, run identity, and the exact input_config dump.
    assert server_metrics["aiperf_version"]
    assert server_metrics["benchmark_id"] == "authored-v2"
    assert isinstance(server_metrics["input_config"], dict)
    assert server_metrics["input_config"]["endpoint"]["type"] == "chat"


def test_export_server_metrics_omits_input_config_when_csv_only(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    from aiperf.common.enums import ServerMetricsFormat

    run.cfg.server_metrics.enabled = True
    run.cfg.server_metrics.urls = ["http://127.0.0.1:8000/vllm/metrics"]
    run.cfg.server_metrics.formats = [ServerMetricsFormat.CSV]

    server_metrics = dump_benchmark_run(run)["cfg"]["export"]["server_metrics"]

    assert server_metrics["json"] is False
    assert server_metrics["csv"] is True
    # input_config is only needed by the JSON exporter.
    assert "input_config" not in server_metrics
    assert server_metrics["benchmark_id"] == "authored-v2"


def _network_run(artifact_target: Path) -> BenchmarkRun:
    """A run with OTel/MLflow/W&B all configured, for the network-export gate."""
    run = _run(artifact_target)
    # Already normalized by the config validator's BeforeValidator on the real
    # load path (verified end-to-end in the live run); set here directly since
    # attribute assignment bypasses validation.
    run.cfg.otel.metrics_url = "http://127.0.0.1:4318/v1/metrics"
    run.cfg.otel.gen_ai_provider = "openai"
    run.cfg.otel.custom_resource_attributes = {"deployment.environment": "e2e"}
    run.cfg.mlflow.tracking_uri = "file:///tmp/mlruns"
    run.cfg.mlflow.experiment = "e2e"
    run.cfg.wandb.project = "e2e"
    run.cfg.wandb.tags = ["netclose"]
    return run


def test_native_export_disabled_returns_python_to_all_sinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiperf.common.environment import Environment
    from aiperf.orchestrator.rust_wire import _export, _live_streaming

    monkeypatch.setattr(Environment.RUNTIME, "NATIVE_EXPORT", False)
    run = _network_run(tmp_path / "artifacts")

    export = _export(run)
    # With the native plane disabled the runner drives no export sinks (an empty
    # block decodes to all-disabled defaults) so it writes only native-v2.json;
    # the legacy Python ExporterManager + streaming sidecar are the single
    # emitter for every artifact including the network destinations.
    assert export == {}
    assert _live_streaming(run) is not None


def test_network_export_projected_and_python_suppressed_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from aiperf.common.environment import Environment
    from aiperf.orchestrator.rust_wire import _export, _live_streaming

    monkeypatch.setattr(Environment.RUNTIME, "NATIVE_EXPORT", True)
    run = _network_run(tmp_path / "artifacts")

    export = _export(run)

    otel = export["otel"]
    assert otel["enabled"] is True
    # metrics_url is normalized to the OTLP/HTTP metrics path by the config validator.
    assert otel["endpoint"].endswith("/v1/metrics")
    assert otel["provider"] == "openai"
    attrs = otel["resource_attributes"]
    assert attrs["aiperf.benchmark.id"] == "authored-v2"
    assert attrs["aiperf.endpoint.type"] == "chat"
    assert attrs["aiperf.model.name"] == "mock-model"
    assert attrs["deployment.environment"] == "e2e"
    # service.name is set by the sink, not projected.
    assert "service.name" not in attrs

    mlflow = export["mlflow"]
    assert mlflow["enabled"] is True
    assert mlflow["tracking_uri"] == "file:///tmp/mlruns"
    assert mlflow["experiment"] == "e2e"
    assert mlflow["benchmark_id"] == "authored-v2"
    assert mlflow["params"]["endpoint.type"] == "chat"
    assert mlflow["params"]["loadgen.concurrency"] == "1"

    wandb = export["wandb"]
    assert wandb["project"] == "e2e"
    assert wandb["tags"] == ["netclose"]
    assert wandb["benchmark_id"] == "authored-v2"
    # config_json is the serialized redacted config object (a JSON string).
    parsed = json.loads(wandb["config_json"])
    assert parsed["endpoint"]["type"] == "chat"

    # By default the Python streaming sidecar is suppressed (single emitter).
    assert _live_streaming(run) is None


def test_dump_strips_python_only_cfg_sections(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    dumped = dump_benchmark_run(run)
    assert set(dumped["cfg"]["runtime"]) <= {"workers", "workers_max", "workers_min"}
    assert dumped["cfg"]["runtime"]["workers"] == 3


def test_invalid_operation_rejected(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    with pytest.raises(RustWireError, match="operation"):
        build_authored_run_request(run, operation="prepare")  # type: ignore[arg-type]


def test_graph_dataset_stays_on_cfg(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "artifacts",
        dataset={
            "type": "file",
            "path": str(tmp_path / "trace.jsonl"),
            "format": "dag_jsonl",
        },
    )
    authored = build_authored_run_request(run, operation="validate")["run"]
    assert authored["cfg"]["datasets"][0]["format"] == "dag_jsonl"
    assert "workload" not in authored


def test_records_parquet_absent_by_default(tmp_path: Path) -> None:
    # Default records format is ["jsonl"]: JSONL path present, no parquet sidecar.
    run = _run(tmp_path / "artifacts")
    artifacts = dump_benchmark_run(run)["cfg"]["artifacts"]
    assert artifacts["records_path"] == "profile_export.jsonl"
    assert "records_parquet_path" not in artifacts


def test_records_parquet_projects_relative_path_when_in_formats(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    run.cfg.artifacts.records = ["jsonl", "parquet"]
    artifacts = dump_benchmark_run(run)["cfg"]["artifacts"]
    assert artifacts["records_path"] == "profile_export.jsonl"
    assert artifacts["records_parquet_path"] == "profile_export.parquet"


def test_records_parquet_only_omits_jsonl(tmp_path: Path) -> None:
    # ["parquet"] alone selects the columnar sidecar without the JSONL.
    run = _run(tmp_path / "artifacts")
    run.cfg.artifacts.records = ["parquet"]
    artifacts = dump_benchmark_run(run)["cfg"]["artifacts"]
    assert "records_path" not in artifacts
    assert artifacts["records_parquet_path"] == "profile_export.parquet"


def test_records_csv_projects_relative_path_when_in_formats(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    run.cfg.artifacts.records = ["jsonl", "csv"]
    artifacts = dump_benchmark_run(run)["cfg"]["artifacts"]
    assert artifacts["records_path"] == "profile_export.jsonl"
    assert artifacts["records_csv_path"] == "profile_export_records.csv"


def test_records_all_three_formats_project_together(tmp_path: Path) -> None:
    run = _run(tmp_path / "artifacts")
    run.cfg.artifacts.records = ["jsonl", "csv", "parquet"]
    artifacts = dump_benchmark_run(run)["cfg"]["artifacts"]
    assert artifacts["records_path"] == "profile_export.jsonl"
    assert artifacts["records_csv_path"] == "profile_export_records.csv"
    assert artifacts["records_parquet_path"] == "profile_export.parquet"


def test_records_csv_stripped_for_dynosim(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "artifacts",
        transport={
            "type": "dynosim_offline",
            "topology": "single",
            "engine": {"block_size": 16},
            "required_features": ["dynamo-full"],
        },
        endpoint_type="dynosim",
        endpoint_url="dynosim://offline",
    )
    run.cfg.artifacts.records = ["jsonl", "csv"]
    artifacts = dump_benchmark_run(run)["cfg"]["artifacts"]
    assert "records_csv_path" not in artifacts


def test_records_parquet_stripped_for_dynosim(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "artifacts",
        transport={
            "type": "dynosim_offline",
            "topology": "single",
            "engine": {"block_size": 16},
            "required_features": ["dynamo-full"],
        },
        endpoint_type="dynosim",
        endpoint_url="dynosim://offline",
    )
    run.cfg.artifacts.records = ["jsonl", "parquet"]
    artifacts = dump_benchmark_run(run)["cfg"]["artifacts"]
    assert "records_parquet_path" not in artifacts


def test_dynosim_transport_inline(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "artifacts",
        transport={
            "type": "dynosim_offline",
            "topology": "single",
            "engine": {"block_size": 16},
            "required_features": ["dynamo-full"],
        },
        endpoint_type="dynosim",
        endpoint_url="dynosim://offline",
    )
    authored = build_authored_run_request(run, operation="execute")["run"]
    transport = authored["cfg"]["transport"]
    assert transport["type"] == "dynosim_offline"
    assert "config" not in transport
    assert transport["topology"] == "single"


def _write_burst_gpt_csv(path: Path) -> Path:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Timestamp", "Model", "Request tokens", "Response tokens"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "Timestamp": 0.0,
                "Model": "ChatGPT",
                "Request tokens": 472,
                "Response tokens": 18,
            }
        )
    return path


def test_unset_file_format_is_structurally_detected(tmp_path: Path) -> None:
    """A BurstGPT CSV without ``--custom-dataset-type`` ships the detected loader.

    Regresses the auto-detect projection bug: the FileDataset default of
    ``single_turn`` was shipped verbatim, so the runner tried to JSON-parse the
    CSV header and failed with "expected value at line 1 column 1". The
    projection now runs structural ``can_load`` detection when the user did not
    set a format, and ships ``burst_gpt`` (the runner-native alias) instead.
    """
    csv_path = _write_burst_gpt_csv(tmp_path / "burst_gpt.csv")
    run = _run(
        tmp_path / "artifacts",
        dataset={"type": "file", "path": str(csv_path)},
    )
    authored = build_authored_run_request(run, operation="execute")["run"]
    assert authored["cfg"]["datasets"][0]["format"] == "burst_gpt"


def test_explicit_file_format_bypasses_detection(tmp_path: Path) -> None:
    """An explicit ``format`` is shipped unchanged, without structural probing.

    The projection must not second-guess an author who selected the format:
    even though this JSONL content would structurally detect as ``single_turn``,
    the explicit selection is preserved verbatim.
    """
    jsonl_path = tmp_path / "prompts.jsonl"
    jsonl_path.write_text('{"text": "hi"}\n', encoding="utf-8")
    run = _run(
        tmp_path / "artifacts",
        dataset={
            "type": "file",
            "path": str(jsonl_path),
            "format": "single_turn",
        },
    )
    authored = build_authored_run_request(run, operation="execute")["run"]
    assert authored["cfg"]["datasets"][0]["format"] == "single_turn"
