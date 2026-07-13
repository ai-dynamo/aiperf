# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-v2 authored projection stays structural and side-effect free."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from pydantic import TypeAdapter, ValidationError

from aiperf.common.environment import Environment
from aiperf.config import (
    AgenticProviderConfig,
    AIPerfConfig,
    BenchmarkRun,
    RunnerTransportConfig,
    RunnerWorkloadConfig,
)
from aiperf.config.execution import OpenTransport
from aiperf.orchestrator import rust_executor
from aiperf.orchestrator.runner_installation import RunnerInstallation
from aiperf.orchestrator.rust_wire import (
    RustWireError,
    build_authored_run_request,
)

_TRANSPORT_ADAPTER = TypeAdapter(RunnerTransportConfig)

_DISTRIBUTION_A = "blake3:" + "a" * 64
_DISTRIBUTION_B = "blake3:" + "b" * 64


def _run(
    artifact_target: Path,
    *,
    dataset: dict | None = None,
    accuracy: dict | None = None,
    transport: dict | None = None,
    workload: dict | None = None,
    endpoint_url: str = "http://127.0.0.1:8000",
    endpoint_type: str = "future_endpoint",
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
    if accuracy is not None:
        benchmark["accuracy"] = accuracy
    if transport is not None:
        benchmark["transport"] = transport
    if workload is not None:
        benchmark["workload"] = workload
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
def test_v2_envelope_and_default_scheduled_projection(
    tmp_path: Path, operation: str
) -> None:
    run = _run(tmp_path / "not-created")

    request = build_authored_run_request(
        run,
        operation=operation,
        expected_distribution_id=_DISTRIBUTION_A,
    )

    assert request["protocol_version"] == 2
    assert request["operation"] == operation
    assert request["expected_distribution_id"] == _DISTRIBUTION_A
    authored = request["run"]
    assert authored["identity"] == {
        "benchmark_id": "authored-v2",
        "label": "cell",
        "trial": 2,
        "random_seed": 17,
    }
    assert authored["artifact_target"] == str(tmp_path / "not-created")
    assert authored["transport"] == {"type": "http", "config": {}}
    assert authored["workload"]["type"] == "scheduled"
    assert authored["workload"]["config"]["worker_count"] == 1
    assert "name" not in authored["workload"]["config"]["dataset"]
    assert authored["workload"]["config"]["tokenizer"]["name"] == "builtin"
    assert authored["workload"]["config"]["phases"] == [
        {
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": False,
            "seamless": False,
            "requests": 2,
            "concurrency": 1,
        }
    ]
    assert authored["resources"]["endpoints"]["profiles"][0]["id"] == "default"
    assert (
        authored["resources"]["endpoints"]["profiles"][0]["type"] == "future_endpoint"
    )
    profile = authored["resources"]["endpoints"]["profiles"][0]
    assert {
        key: profile[key]
        for key in (
            "http2",
            "ssl_verify",
            "connection_limit",
            "keepalive_timeout",
        )
    } == {
        "http2": False,
        "ssl_verify": True,
        "connection_limit": 2500,
        "keepalive_timeout": 300.0,
    }


def test_v2_projects_resolved_worker_lower_bound(tmp_path: Path) -> None:
    run = _run(tmp_path / "worker-placement")
    run.cfg.runtime.workers_min = 3

    request = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )

    assert request["run"]["workload"]["config"]["worker_count"] == 3


def test_v2_projects_authored_http_policy_without_legacy_conversion(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path / "http-policy")
    run.cfg.endpoint.http2 = True
    run.cfg.endpoint.ssl_verify = False
    run.cfg.endpoint.connection_limit = 7
    run.cfg.endpoint.keepalive_timeout = 0.25

    profile = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["endpoints"]["profiles"][0]

    assert profile["http2"] is True
    assert profile["ssl_verify"] is False
    assert profile["connection_limit"] == 7
    assert profile["keepalive_timeout"] == 0.25
    assert "endpoint" not in profile


def test_normalized_unauthored_tokenizer_uses_builtin_for_placeholder_model(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path / "not-created")
    assert run.cfg.tokenizer is not None
    run.cfg.tokenizer.name = None

    tokenizer = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]["config"]["tokenizer"]

    assert tokenizer["name"] == "builtin"
    assert tokenizer["revision"] == "main"
    assert tokenizer["trust_remote_code"] is False


def test_online_grpc_is_preserved_only_in_the_authored_v2_projection(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path / "grpc-v2-only",
        transport={"type": "grpc"},
        endpoint_url="grpc://127.0.0.1:8001",
        endpoint_type="kserve_v2_infer",
    )

    request = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )

    assert request["protocol_version"] == 2
    assert request["run"]["transport"] == {"type": "grpc", "config": {}}
    assert request["run"]["resources"]["endpoints"]["profiles"][0]["urls"] == [
        "grpc://127.0.0.1:8001"
    ]


@pytest.mark.parametrize(
    ("backend", "url", "message"),
    [
        ("http", "grpc://127.0.0.1:8001", "grpc"),
        ("grpc", "http://127.0.0.1:8000", "requires grpc"),
    ],
)
def test_builtin_online_backend_url_families_fail_closed(
    tmp_path: Path, backend: str, url: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _run(
            tmp_path / backend,
            transport={"type": backend},
            endpoint_url=url,
            endpoint_type="kserve_v2_infer",
        )


def test_dag_jsonl_rows_enter_graph_workload_once_without_conversion(
    tmp_path: Path,
) -> None:
    authored_rows = [
        {
            "session_id": "root",
            "turns": [
                {
                    "timestamp": 17,
                    "messages": [{"role": "user", "content": "root"}],
                    "forks": ["child"],
                    "opaque_extension": {"preserve": [1, 2, 3]},
                }
            ],
        },
        {
            "session_id": "child",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "child"}],
                    "spawns": [{"children": ["grandchild"]}],
                }
            ],
        },
    ]
    run = _run(
        tmp_path / "graph-target",
        dataset={
            "type": "file",
            "format": "dag_jsonl",
            "records": authored_rows,
        },
    )

    workload = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]

    assert workload["type"] == "graph"
    assert workload["config"]["dataset"]["format"] == "dag_jsonl"
    assert workload["config"]["dataset"]["records"] == authored_rows
    assert "conversation" not in workload["config"]["dataset"]
    assert "graph_ir" not in workload["config"]["dataset"]


@pytest.mark.parametrize("format_name", ["weka_trace", "dynamo_trace"])
def test_recorded_graph_sources_enter_the_native_graph_workload_untouched(
    tmp_path: Path, format_name: str
) -> None:
    source: dict[str, object]
    if format_name == "weka_trace":
        source = {
            "records": [
                {
                    "id": "trace",
                    "models": ["m"],
                    "block_size": 16,
                    "hash_id_scope": "global",
                    "requests": [],
                }
            ]
        }
    else:
        source = {"path": str(tmp_path / "capture-prefix")}
    run = _run(
        tmp_path / f"{format_name}-target",
        dataset={
            "type": "file",
            "format": format_name,
            **source,
            "synthesis": {
                "idle_gap_cap_seconds": None,
                "corpus": "sonnet",
            },
        },
    )

    workload = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]

    assert workload["type"] == "graph"
    dataset = workload["config"]["dataset"]
    assert dataset["format"] == format_name
    assert dataset["synthesis"]["idle_gap_cap_seconds"] is None
    assert dataset["synthesis"]["corpus"] == "sonnet"
    if format_name == "weka_trace":
        assert dataset["records"] == source["records"]
    else:
        assert dataset["path"] == str((tmp_path / "capture-prefix").resolve())


def test_projection_never_reads_resolved_or_performs_resolver_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "never-created"
    missing_dataset = tmp_path / "missing-dag.jsonl"
    run = _run(
        target,
        dataset={
            "type": "file",
            "format": "dag_jsonl",
            "path": str(missing_dataset),
        },
    )
    run.cfg.tokenizer.name = "remote/not-cached"

    class _ResolvedAccessIsABug:
        def __getattribute__(self, name: str):
            raise AssertionError(f"v2 projection read BenchmarkRun.resolved.{name}")

    run.__dict__["resolved"] = _ResolvedAccessIsABug()
    monkeypatch.setattr(
        "aiperf.config.resolution.resolvers.build_default_resolver_chain",
        lambda: pytest.fail("v2 authored projection invoked resolver chain"),
    )
    assert not target.exists()
    assert not missing_dataset.exists()

    projected = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )

    assert projected["run"]["workload"]["config"]["dataset"]["path"] == str(
        missing_dataset
    )
    assert projected["run"]["workload"]["config"]["tokenizer"]["name"] == (
        "remote/not-cached"
    )
    assert not target.exists()
    assert not missing_dataset.exists()


def test_open_backend_workload_and_agentic_provider_payloads_survive(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path,
        transport={
            "type": " acme_remote_backend ",
            "placement": {"fabric": "zmq", "nodes": 4},
        },
        workload={
            "type": " acme_agentic ",
            "config": {
                "provider": {
                    "type": "future_provider",
                    "config": {"canonical_option": True},
                },
                "extension_option": [1, 2],
            },
        },
    )

    authored = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]

    assert authored["transport"] == {
        "type": "acme_remote_backend",
        "config": {"placement": {"fabric": "zmq", "nodes": 4}},
    }
    assert authored["workload"]["type"] == "acme_agentic"
    assert authored["workload"]["config"]["provider"] == {
        "type": "future_provider",
        "config": {"canonical_option": True},
    }
    assert authored["workload"]["config"]["extension_option"] == [1, 2]

    provider = AgenticProviderConfig(
        type=" out_of_tree_provider ", config={"opaque": "value"}
    )
    assert provider.type == "out_of_tree_provider"


def test_default_static_accuracy_selection(tmp_path: Path) -> None:
    run = _run(tmp_path, accuracy={"benchmark": "mmlu"})

    workload = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]

    assert workload["type"] == "static_accuracy"
    assert workload["config"]["accuracy"]["benchmark"] == "mmlu"
    assert Path(workload["config"]["accuracy"]["python_executable"]).is_absolute()
    assert workload["config"]["accuracy"]["worker_module"] == ("aiperf.accuracy.worker")


def test_v2_dataset_and_tokenizer_projection_is_native_shaped_but_unresolved(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path,
        dataset={
            "type": "synthetic",
            "entries": 1,
            "turns": 2,
            "turn_delay": {"mean": 25, "stddev": 3},
            "prompts": {"isl": 8, "osl": 2},
        },
    )
    run.cfg.tokenizer = None

    config = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]["config"]

    assert config["dataset"]["turn_delay_ms"] == {"mean": 25.0, "stddev": 3.0}
    assert "turn_delay" not in config["dataset"]
    assert "name" not in config["dataset"]
    assert config["tokenizer"] == {
        "name": "builtin",
        "revision": "main",
        "trust_remote_code": False,
        "apply_chat_template": False,
    }


def test_v2_file_dataset_projects_directly_to_selected_native_adapter(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path / "not-created",
        dataset={
            "type": "file",
            "records": [{"timestamp": 0.0, "input_length": 8, "output_length": 2}],
            "format": "burst_gpt_trace",
            "entries": 1,
            "sampling": "shuffle",
            "random_seed": 23,
            "inter_turn_delay_cap_seconds": 0.5,
            "osl": 7,
        },
    )

    dataset = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]["config"]["dataset"]

    assert dataset == {
        "type": "file",
        "format": "burst_gpt",
        "sampling": "shuffle",
        "options": {"inter_turn_delay_cap_seconds": 0.5},
        "entries": 1,
        "random_seed": 23,
        "osl": {"value": 7.0},
        "records": [{"timestamp": 0.0, "input_length": 8, "output_length": 2}],
    }
    assert not run.artifact_dir.exists()


def test_v2_user_files_are_rendered_and_serialized_once_without_writing(
    tmp_path: Path,
) -> None:
    from aiperf.config.user_files import UserFile

    target = tmp_path / "not-created"
    run = _run(target)
    run.variables = {"count": 7, "owner": "Ada"}
    run.cfg.artifacts.user_files = [
        UserFile(
            path="meta/run.json",
            format="json",
            content={"count": "{{ count }}", "owner": "{{ owner }}"},
        ),
        UserFile(path="notes.txt", content="hello {{ owner }}\n"),
    ]

    files = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["artifacts"]["user_files"]

    assert files == [
        {
            "path": "meta/run.json",
            "format": "json",
            "content": '{\n  "count": 7,\n  "owner": "Ada"\n}',
        },
        {"path": "notes.txt", "format": "text", "content": "hello Ada\n"},
    ]
    assert not target.exists()


def test_v2_public_dataset_is_expanded_once_without_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiperf.dataset.loader.sharegpt import ShareGPTLoader

    source_url = "https://datasets.example.test/sharegpt.json"
    monkeypatch.setattr(ShareGPTLoader, "url", source_url)
    run = _run(
        tmp_path / "not-created",
        dataset={
            "type": "public",
            "dataset": "sharegpt",
            "entries": 7,
            "sampling": "shuffle",
        },
    )

    dataset = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]["config"]["dataset"]

    assert dataset == {
        "type": "public",
        "name": "sharegpt",
        "format": "sharegpt",
        "source": {"type": "url", "url": source_url},
        "sampling": "shuffle",
        "options": {"max_conversations": 7},
        "entries": 7,
    }
    assert not run.artifact_dir.exists()


def test_v2_public_weka_dataset_projects_revision_pinned_native_graph_source(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path / "not-created",
        dataset={
            "type": "public",
            "dataset": "weka_cc_traces_062126",
            "entries": 3,
            "sampling": "sequential",
        },
    )

    workload = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["workload"]

    assert workload["type"] == "graph"
    assert workload["config"]["dataset"] == {
        "type": "public",
        "name": "weka_cc_traces_062126",
        "format": "weka_trace",
        "source": {
            "type": "hugging_face",
            "dataset": "semianalysisai/cc-traces-weka-062126",
            "subset": "default",
            "split": "train",
            "revision": "23f152f6f0f9399a85901b89a6458def0ef16729",
        },
        "sampling": "sequential",
        "options": {"max_conversations": 3},
        "entries": 3,
    }
    assert not run.artifact_dir.exists()


def test_v2_sidecars_are_direct_protocol_neutral_native_inputs(tmp_path: Path) -> None:
    run = _run(tmp_path / "not-created")
    run.cfg.gpu_telemetry.enabled = True
    run.cfg.gpu_telemetry.urls = ["http://gpu-node:9400"]
    run.cfg.network_latency.enabled = True
    run.cfg.network_latency.mean_ms = 2.5
    run.cfg.server_metrics.enabled = True
    run.cfg.server_metrics.urls = ["metrics-node:9000"]
    run.cfg.server_metrics.formats = ["jsonl", "parquet"]
    run.cfg.otel.metrics_url = "http://otel:4318"
    run.cfg.mlflow.tracking_uri = "http://mlflow:5000"

    sidecars = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["sidecars"]

    assert set(sidecars) == {
        "gpu_telemetry",
        "network_latency",
        "server_metrics",
        "live_streaming",
    }
    assert sidecars["gpu_telemetry"]["sources"][-1] == {
        "type": "dcgm",
        "url": "http://gpu-node:9400/metrics",
    }
    assert sidecars["network_latency"] == {"mean_rtt_ns": 2_500_000}
    assert sidecars["server_metrics"]["formats"] == ["jsonl", "parquet"]
    assert sidecars["live_streaming"]["otel"]["metrics_url"] == (
        "http://otel:4318/v1/metrics"
    )
    assert not run.artifact_dir.exists()


def test_v2_content_server_environment_projects_without_touching_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path / "not-created")
    content_dir = tmp_path / "content-does-not-exist"
    monkeypatch.setattr(Environment.CONTENT_SERVER, "ENABLED", True)
    monkeypatch.setattr(Environment.CONTENT_SERVER, "HOST", "0.0.0.0")
    monkeypatch.setattr(Environment.CONTENT_SERVER, "PORT", 8090)
    monkeypatch.setattr(Environment.CONTENT_SERVER, "CONTENT_DIR", str(content_dir))
    monkeypatch.setattr(Environment.CONTENT_SERVER, "MAX_TRACKED_RECORDS", 321)

    sidecar = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["sidecars"]["content_server"]

    assert sidecar == {
        "host": "0.0.0.0",
        "port": 8090,
        "content_dir": str(content_dir.absolute()),
        "max_tracked_records": 321,
    }
    assert not content_dir.exists()


def test_v2_enabled_content_server_without_directory_omits_externalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Environment.CONTENT_SERVER, "ENABLED", True)
    monkeypatch.setattr(Environment.CONTENT_SERVER, "CONTENT_DIR", "")

    sidecar = build_authored_run_request(
        _run(tmp_path / "not-created"),
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["sidecars"]["content_server"]

    assert "content_dir" not in sidecar


def test_v2_content_server_relative_directory_is_authored_as_absolute_without_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Environment.CONTENT_SERVER, "ENABLED", True)
    monkeypatch.setattr(Environment.CONTENT_SERVER, "CONTENT_DIR", "generated-media")

    sidecar = build_authored_run_request(
        _run(tmp_path / "not-created"),
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["sidecars"]["content_server"]

    expected = tmp_path / "generated-media"
    assert sidecar["content_dir"] == str(expected)
    assert not expected.exists()


def test_v2_custom_gpu_source_never_reads_resolved_state(tmp_path: Path) -> None:
    run = _run(tmp_path / "not-created")
    metrics_file = tmp_path / "not-read.csv"
    run.cfg.gpu_telemetry.enabled = True
    run.cfg.gpu_telemetry.urls = ["http://gpu-node:9400"]
    run.cfg.gpu_telemetry.metrics_file = metrics_file

    class _ResolvedAccessIsABug:
        def __getattribute__(self, name: str):
            raise AssertionError(
                f"v2 sidecar projection read BenchmarkRun.resolved.{name}"
            )

    run.__dict__["resolved"] = _ResolvedAccessIsABug()
    sidecar = build_authored_run_request(
        run,
        operation="execute",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["sidecars"]["gpu_telemetry"]

    assert "custom_metrics" not in sidecar
    assert all(source["type"] == "python" for source in sidecar["sources"])
    assert sidecar["sources"][-1]["metrics_file"] == str(metrics_file.absolute())
    assert not metrics_file.exists()
    assert not run.artifact_dir.exists()


def test_component_ids_are_open_strings_not_enums() -> None:
    # An unmodelled transport ID routes to the open fallback variant, which keeps
    # the identity an open (stripped) string and passes its own keys through.
    transport = _TRANSPORT_ADAPTER.validate_python(
        {"type": " future_backend ", "a": 1}
    )
    assert isinstance(transport, OpenTransport)
    assert transport.type == "future_backend"
    assert transport.model_extra == {"a": 1}
    open_schema = OpenTransport.model_json_schema()
    assert open_schema["properties"]["type"]["type"] == "string"
    assert "enum" not in open_schema["properties"]["type"]

    workload = RunnerWorkloadConfig(type="future_workload", config={"b": 2})
    assert workload.type == "future_workload"
    workload_schema = RunnerWorkloadConfig.model_json_schema()
    assert workload_schema["properties"]["type"]["type"] == "string"
    assert "enum" not in workload_schema["properties"]["type"]

    for model in (RunnerWorkloadConfig, AgenticProviderConfig):
        with pytest.raises(ValidationError, match="at least 1|non-empty string"):
            model(type="   ")
    # A blank transport ID has no factory identity to strip to; the open variant
    # rejects it just like the named-component models above.
    with pytest.raises(ValidationError, match="at least 1|non-empty string"):
        OpenTransport(type="   ")


def test_runner_installation_uses_only_advertised_distribution_identity(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    installation = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={
            "protocol_versions": [1, 2],
            "distribution_id": _DISTRIBUTION_A,
        },
    )

    request = installation.project_authored_request(run, operation="validate")

    assert request["expected_distribution_id"] == _DISTRIBUTION_A

    missing_identity = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={"protocol_versions": [1, 2]},
    )
    with pytest.raises(RuntimeError, match="without distribution_id|not invent"):
        missing_identity.project_authored_request(run, operation="validate")


def test_executor_loads_only_the_advertised_v2_pair_adapter(tmp_path: Path) -> None:
    run = _run(tmp_path / "authored-only")
    installation = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={
            "protocol_versions": [1, 2],
            "distribution_id": _DISTRIBUTION_A,
            "supported_pairs": [["http", "scheduled"]],
            "endpoint_types": ["future_endpoint"],
        },
    )
    executor = rust_executor.RustSubprocessExecutor(
        base_dir=tmp_path,
        installation=installation,
    )
    request = executor._request_for_run(run)

    assert request["protocol_version"] == 2
    assert request["run"]["transport"]["type"] == "http"
    assert request["run"]["workload"]["type"] == "scheduled"
    assert not hasattr(rust_executor.RustSubprocessExecutor, "_resolve_run")
    assert not hasattr(rust_executor, "build_run_request")
    assert not hasattr(rust_executor, "validate_v1_selection")
    assert not run.artifact_dir.exists()


def test_unsupported_pair_fails_at_exact_v2_image_preflight(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "must-not-exist",
        transport={"type": "dynamo_offline"},
    )
    installation = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={
            "protocol_versions": [1, 2],
            "distribution_id": _DISTRIBUTION_A,
            "supported_pairs": [["http", "scheduled"]],
        },
    )
    executor = rust_executor.RustSubprocessExecutor(
        base_dir=tmp_path,
        installation=installation,
    )

    with pytest.raises(
        RuntimeError,
        match=r"executable protocol-v2 pair \('dynamo_offline', 'scheduled'\)",
    ):
        executor._request_for_run(run)

    assert not run.artifact_dir.exists()


def test_v2_terminal_is_bound_to_negotiated_distribution() -> None:
    distribution_id = _DISTRIBUTION_B
    terminal = rust_executor._parse_terminal(
        orjson.dumps(
            {
                "protocol_version": 2,
                "event": "run_terminal",
                "distribution_id": distribution_id,
                "benchmark_id": "bound-runner",
                "success": False,
                "report_path": None,
                "stage": "validation",
                "errors": [{"code": "invalid", "message": "bad pair"}],
                "provenance": {},
            }
        )
        + b"\n",
        SimpleNamespace(benchmark_id="bound-runner"),
        protocol_version=2,
        distribution_id=distribution_id,
    )

    assert terminal["stage"] == "validation"
    with pytest.raises(ValueError, match="distribution_id"):
        rust_executor._parse_terminal(
            orjson.dumps({**terminal, "distribution_id": "blake3:wrong"}) + b"\n",
            SimpleNamespace(benchmark_id="bound-runner"),
            protocol_version=2,
            distribution_id=distribution_id,
        )


def test_executor_never_reinterprets_an_unsupported_pair_as_v1(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "not-created",
        transport={"type": "future_backend", "node": "remote"},
    )
    installation = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={
            "protocol_versions": [2],
            "distribution_id": _DISTRIBUTION_A,
            "supported_pairs": [["http", "scheduled"]],
            "endpoint_types": ["future_endpoint"],
        },
    )
    executor = rust_executor.RustSubprocessExecutor(
        base_dir=tmp_path,
        installation=installation,
    )
    result = executor.execute_sync(run)

    assert result.success is False
    assert "executable protocol-v2 pair ('future_backend', 'scheduled')" in (
        result.error or ""
    )
    assert not run.artifact_dir.exists()


def test_readiness_is_authored_in_v2_and_rejected_by_v1(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.cfg.endpoint.wait_for_model_timeout = 90.0
    run.cfg.endpoint.wait_for_model_interval = 1.25
    run.cfg.endpoint.wait_for_model_mode = "both"

    endpoint = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id=_DISTRIBUTION_A,
    )["run"]["resources"]["endpoints"]["profiles"][0]

    assert endpoint["wait_for_model_timeout"] == 90.0
    assert endpoint["wait_for_model_interval"] == 1.25
    assert endpoint["wait_for_model_mode"] == "both"


@pytest.mark.parametrize(
    ("operation", "distribution_id"),
    [
        ("prepare", "blake3:x"),
        ("validate", ""),
        ("execute", "   "),
        ("execute", " blake3:x "),
    ],
)
def test_v2_envelope_rejects_unknown_operation_or_missing_identity(
    tmp_path: Path, operation: str, distribution_id: str
) -> None:
    with pytest.raises(RustWireError):
        build_authored_run_request(
            _run(tmp_path),
            operation=operation,
            expected_distribution_id=distribution_id,
        )


def test_dynosim_offline_transport_projects_minimal_typed_config_snake_case(
    tmp_path: Path,
) -> None:
    """Authoring ``transport.type: dynosim_offline`` projects the exact runner schema.

    Mirrors the offline replay path proven byte-exact at 50k: a minimal
    ``{engine: {block_size}}`` config projects to snake_case with only the
    authored fields (the runner fills every other serde default). The clock rides
    on the transport id, so there is no ``replay_mode`` field.
    """
    run = _run(
        tmp_path,
        transport={
            "type": "dynosim_offline",
            "engine": {"block_size": 16},
        },
        dataset={
            "type": "file",
            "format": "mooncake_trace",
            "sampling": "sequential",
            "path": "trace.jsonl",
        },
    )
    authored = build_authored_run_request(
        run, operation="execute", expected_distribution_id=_DISTRIBUTION_A
    )["run"]
    assert authored["transport"] == {
        "type": "dynosim_offline",
        "config": {"engine": {"block_size": 16}},
    }
    assert "replay_mode" not in authored["transport"]["config"]
    # Trace + concurrency reuse the shared scheduled surface, not the transport.
    assert authored["workload"]["type"] == "scheduled"
    assert authored["workload"]["config"]["dataset"]["format"] == "mooncake_trace"


def test_dynosim_online_transport_normalizes_camelcase_and_sorts_features(
    tmp_path: Path,
) -> None:
    """camelCase authoring dumps snake_case; required_features is deterministic.

    The wall-clock axis is the ``dynosim_online`` transport id, not a config field.
    """
    run = _run(
        tmp_path,
        transport={
            "type": "dynosim_online",
            "topology": "disaggregated",
            "prefillWorkers": 2,
            "decodeWorkers": 4,
            "routerMode": "kv",
            "router": {"opaque": True},
            "sla": {"ttftMs": 500},
            "requiredFeatures": [
                "dynamo-kvbm-offload",
                "dynamo-aic-forward-pass",
            ],
            "artifacts": {"reportJson": "dynamo/report.json"},
        },
    )
    authored = build_authored_run_request(
        run, operation="validate", expected_distribution_id=_DISTRIBUTION_A
    )["run"]
    assert authored["transport"]["type"] == "dynosim_online"
    config = authored["transport"]["config"]
    assert "replay_mode" not in config
    assert config["topology"] == "disaggregated"
    assert config["prefill_workers"] == 2
    assert config["decode_workers"] == 4
    assert config["router_mode"] == "kv"
    assert config["router"] == {"opaque": True}
    assert config["sla"] == {"ttft_ms": 500.0}
    assert config["artifacts"] == {"report_json": "dynamo/report.json"}
    # BTreeSet on the runner side, but the wire must be deterministic → sorted.
    assert config["required_features"] == [
        "dynamo-aic-forward-pass",
        "dynamo-kvbm-offload",
    ]


def test_dynosim_transport_not_constrained_to_http_url_scheme(
    tmp_path: Path,
) -> None:
    """dynosim is native protocol-v2: it does not impose the http scheme."""
    # A grpc:// endpoint URL would be rejected under transport.type='http', but
    # dynosim's in-process replay never dials it, so validation must pass.
    for transport_type in ("dynosim_offline", "dynosim_online"):
        _run(
            tmp_path / transport_type,
            transport={"type": transport_type},
            endpoint_url="grpc://127.0.0.1:8001",
        )


def test_dynosim_transport_defaults_endpoint_dialect_to_dynosim(
    tmp_path: Path,
) -> None:
    """When transport is dynosim_* and endpoint.type is unset it defaults to dynosim."""
    config = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {"urls": ["http://unused.invalid"]},
                "dataset": {
                    "type": "synthetic",
                    "entries": 2,
                    "prompts": {"isl": 8, "osl": 2},
                },
                "phases": [
                    {"name": "profiling", "type": "concurrency", "requests": 2, "concurrency": 1}
                ],
                "transport": {"type": "dynosim_offline"},
            }
        }
    )
    assert str(config.benchmark.endpoint.type) == "dynosim"


def test_typed_dynosim_config_rejected_under_non_dynosim_transport() -> None:
    """dynosim-only fields cannot be smuggled onto a non-dynosim transport.

    Inline authoring makes this a structural guarantee: the ``http`` / ``grpc``
    variants forbid extras, so a dynosim field (e.g. ``workers``) on a non-dynosim
    transport is rejected as an unknown key rather than a runtime type check.
    """
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _TRANSPORT_ADAPTER.validate_python({"type": "http", "workers": 1})
