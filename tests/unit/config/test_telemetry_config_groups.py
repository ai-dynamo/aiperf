# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from aiperf.config import load_config_from_string
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import EndpointType


def test_benchmark_otel_and_mlflow_groups_load_without_flat_forwarders() -> None:
    config = load_config_from_string(
        textwrap.dedent("""\
            benchmark:
              models: [llama]
              endpoint:
                urls: ["http://x:8000/v1/chat/completions"]
              datasets:
                - name: main
                  type: synthetic
              phases:
                - name: profiling
                  type: concurrency
                  requests: 10
                  concurrency: 1
              artifacts:
                dir: ./artifacts
                export_outputs_json: true
              otel:
                metrics_url: http://localhost:4318
                stream_metrics_enabled: true
                stream_timing_enabled: false
                custom_resource_attributes:
                  deployment.environment: local
                gen_ai_provider: vllm
              mlflow:
                tracking_uri: http://localhost:5000
                experiment: my-experiment
                run_name: my-run
                tags:
                  team: perf
                  env: ci
                parent_run_id: null
                artifact_globs:
                  - "*.json"
                  - "*.csv"
            """),
    )

    benchmark = config.benchmark
    assert benchmark.artifacts.export_outputs_json is True
    assert benchmark.otel.metrics_url == "http://localhost:4318"
    assert benchmark.otel.stream_timing_enabled is False
    assert benchmark.otel.custom_resource_attributes == {
        "deployment.environment": "local"
    }
    assert benchmark.otel.gen_ai_provider == "vllm"
    assert benchmark.mlflow.tracking_uri == "http://localhost:5000"
    assert benchmark.mlflow.experiment == "my-experiment"
    assert benchmark.mlflow.run_name == "my-run"
    assert benchmark.mlflow.tags == {"team": "perf", "env": "ci"}
    assert benchmark.mlflow.tags_dict == {"team": "perf", "env": "ci"}
    assert benchmark.mlflow.parent_run_id is None
    assert benchmark.mlflow.resolved_artifact_globs == ["*.json", "*.csv"]

    removed_flat_fields = [
        "output",
        "mlflow_enabled",
        "mlflow_tracking_uri",
        "mlflow_experiment",
        "mlflow_run_name",
        "mlflow_tags_dict",
        "mlflow_parent_run_id",
        "mlflow_resolved_artifact_globs",
        "otel_metrics_url",
        "otel_stream_metrics_enabled",
        "otel_stream_timing_enabled",
        "otel_custom_resource_attributes",
        "gen_ai_provider",
        "stream",
        "benchmark_id",
    ]
    for field_name in removed_flat_fields:
        assert not hasattr(benchmark, field_name), field_name


def test_cli_telemetry_flags_populate_first_class_groups() -> None:
    config = convert_cli_to_aiperf(
        CLIConfig(
            model_names=["llama"],
            endpoint_type=EndpointType.CHAT,
            otel_url="http://localhost:4318",
            stream="timing",
            gen_ai_provider="vllm",
            accuracy_benchmark="mmlu",
            accuracy_tasks="abstract_algebra,anatomy",
            accuracy_n_shots=16,
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment="my-experiment",
            mlflow_run_name="my-run",
            mlflow_tags="team:perf",
            mlflow_parent_run_id="parent-1",
            mlflow_artifact_globs=["*.json", "*.csv"],
        )
    )

    benchmark = config.benchmark
    assert benchmark.otel.metrics_url == "http://localhost:4318/v1/metrics"
    assert benchmark.otel.stream_metrics_enabled is False
    assert benchmark.otel.stream_timing_enabled is True
    assert benchmark.otel.gen_ai_provider == "vllm"
    assert benchmark.accuracy.tasks == ["abstract_algebra", "anatomy"]
    assert benchmark.accuracy.n_shots == 16
    assert benchmark.mlflow.tracking_uri == "http://localhost:5000"
    assert benchmark.mlflow.experiment == "my-experiment"
    assert benchmark.mlflow.run_name == "my-run"
    assert benchmark.mlflow.tags_dict == {"team": "perf"}
    assert benchmark.mlflow.parent_run_id == "parent-1"
    assert benchmark.mlflow.resolved_artifact_globs == ["*.json", "*.csv"]


def test_legacy_artifacts_telemetry_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="otel_metrics_url"):
        load_config_from_string(
            textwrap.dedent("""\
                benchmark:
                  models: [llama]
                  endpoint:
                    urls: ["http://x:8000/v1/chat/completions"]
                  datasets:
                    - name: main
                      type: synthetic
                  phases:
                    - name: profiling
                      type: concurrency
                      requests: 10
                      concurrency: 1
                  artifacts:
                    otel_metrics_url: http://localhost:4318
                """),
        )
