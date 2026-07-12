# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-v2 authored projection stays structural and side-effect free."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from pydantic import ValidationError

from aiperf.config import (
    AgenticProviderConfig,
    AIPerfConfig,
    BenchmarkRun,
    RunnerBackendConfig,
    RunnerWorkloadConfig,
)
from aiperf.orchestrator import rust_executor
from aiperf.orchestrator.runner_installation import RunnerInstallation
from aiperf.orchestrator.rust_wire import (
    RustWireError,
    build_authored_run_request,
    build_run_request,
)


def _run(
    artifact_target: Path,
    *,
    dataset: dict | None = None,
    accuracy: dict | None = None,
    backend: dict | None = None,
    workload: dict | None = None,
) -> BenchmarkRun:
    benchmark: dict = {
        "models": ["mock-model"],
        "endpoint": {
            "urls": ["http://127.0.0.1:8000"],
            "type": "future_endpoint",
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
    if backend is not None:
        benchmark["backend"] = backend
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
        expected_distribution_id="blake3:runner-image",
    )

    assert request["protocol_version"] == 2
    assert request["operation"] == operation
    assert request["expected_distribution_id"] == "blake3:runner-image"
    authored = request["run"]
    assert authored["identity"] == {
        "benchmark_id": "authored-v2",
        "label": "cell",
        "trial": 2,
        "random_seed": 17,
    }
    assert authored["artifact_target"] == str(tmp_path / "not-created")
    assert authored["backend"] == {"type": "online_http", "config": {}}
    assert authored["workload"]["type"] == "scheduled"
    assert authored["workload"]["config"]["worker_count"] == 1
    assert authored["endpoints"]["profiles"][0]["id"] == "default"
    assert authored["endpoints"]["profiles"][0]["type"] == "future_endpoint"


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
        expected_distribution_id="blake3:graph",
    )["run"]["workload"]

    assert workload["type"] == "graph"
    assert workload["config"]["dataset"]["format"] == "dag_jsonl"
    assert workload["config"]["dataset"]["records"] == authored_rows
    assert "conversation" not in workload["config"]["dataset"]
    assert "graph_ir" not in workload["config"]["dataset"]


def test_v1_dag_jsonl_skips_legacy_dataset_timing_and_zmq_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authored_rows = [
        {
            "session_id": "root",
            "turns": [{"messages": [{"role": "user", "content": "root"}]}],
        }
    ]
    run = _run(
        tmp_path / "graph-target",
        dataset={
            "type": "file",
            "format": "dag_jsonl",
            "records": authored_rows,
        },
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail("legacy resolver touched a runner-owned dag_jsonl program")

    monkeypatch.setattr(
        "aiperf.config.resolution.resolvers.DatasetResolver.resolve",
        fail_if_called,
    )
    monkeypatch.setattr(
        "aiperf.config.resolution.resolvers.TimingResolver.resolve",
        fail_if_called,
    )
    monkeypatch.setattr(
        "aiperf.config.resolution.resolvers.CommConfigResolver.resolve",
        fail_if_called,
    )

    rust_executor.RustSubprocessExecutor._resolve_run(run)
    dataset = build_run_request(run)["run"]["dataset"]

    assert dataset["format"] == "dag_jsonl"
    assert dataset["sampling"] == "sequential"
    assert dataset["records"] == authored_rows


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
        expected_distribution_id="blake3:no-side-effects",
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
        backend={
            "type": " acme_remote_backend ",
            "config": {"placement": {"fabric": "zmq", "nodes": 4}},
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
        expected_distribution_id="blake3:custom",
    )["run"]

    assert authored["backend"] == {
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
        expected_distribution_id="blake3:accuracy",
    )["run"]["workload"]

    assert workload["type"] == "static_accuracy"
    assert workload["config"]["accuracy"]["benchmark"] == "mmlu"


def test_component_ids_are_open_strings_not_enums() -> None:
    backend = RunnerBackendConfig(type="future_backend", config={"a": 1})
    workload = RunnerWorkloadConfig(type="future_workload", config={"b": 2})

    assert backend.type == "future_backend"
    assert workload.type == "future_workload"
    backend_schema = RunnerBackendConfig.model_json_schema()
    workload_schema = RunnerWorkloadConfig.model_json_schema()
    assert backend_schema["properties"]["type"]["type"] == "string"
    assert workload_schema["properties"]["type"]["type"] == "string"
    assert "enum" not in backend_schema["properties"]["type"]
    assert "enum" not in workload_schema["properties"]["type"]

    for model in (RunnerBackendConfig, RunnerWorkloadConfig, AgenticProviderConfig):
        with pytest.raises(ValidationError, match="at least 1|non-empty string"):
            model(type="   ")


def test_runner_installation_uses_only_advertised_distribution_identity(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    installation = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={
            "protocol_versions": [1, 2],
            "distribution_id": "blake3:advertised",
        },
    )

    request = installation.project_authored_request(run, operation="validate")

    assert request["expected_distribution_id"] == "blake3:advertised"

    missing_identity = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={"protocol_versions": [1, 2]},
    )
    with pytest.raises(RuntimeError, match="without distribution_id|not invent"):
        missing_identity.project_authored_request(run, operation="validate")


def test_executor_selects_advertised_v2_pair_without_legacy_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path / "authored-only")
    installation = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={
            "protocol_versions": [1, 2],
            "distribution_id": "blake3:" + "a" * 64,
            "supported_pairs": [["online_http", "scheduled"]],
            "endpoint_types": ["future_endpoint"],
        },
    )
    executor = rust_executor.RustSubprocessExecutor(
        base_dir=tmp_path,
        installation=installation,
    )
    monkeypatch.setattr(
        executor,
        "_resolve_run",
        lambda _run: pytest.fail("an executable v2 pair entered legacy resolution"),
    )

    request = executor._request_for_run(run)

    assert request["protocol_version"] == 2
    assert request["run"]["backend"]["type"] == "online_http"
    assert request["run"]["workload"]["type"] == "scheduled"
    assert not run.artifact_dir.exists()


def test_v2_terminal_is_bound_to_negotiated_distribution() -> None:
    distribution_id = "blake3:" + "b" * 64
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
            orjson.dumps({**terminal, "distribution_id": "blake3:wrong"})
            + b"\n",
            SimpleNamespace(benchmark_id="bound-runner"),
            protocol_version=2,
            distribution_id=distribution_id,
        )


def test_v1_projection_remains_the_execution_compatibility_path(tmp_path: Path) -> None:
    request = build_run_request(_run(tmp_path))

    assert request["protocol_version"] == 1
    assert "operation" not in request
    assert request["run"]["endpoint"]["type"] == "future_endpoint"


@pytest.mark.parametrize(
    "selection",
    [
        {"backend": {"type": "future_backend", "config": {}}},
        {"workload": {"type": "future_workload", "config": {}}},
    ],
)
def test_v1_fails_closed_for_v2_only_selections(
    tmp_path: Path, selection: dict
) -> None:
    with pytest.raises(RustWireError, match="protocol v1"):
        build_run_request(_run(tmp_path, **selection))


def test_v1_selection_failure_precedes_resolver_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(
        tmp_path / "not-created",
        backend={"type": "future_backend", "config": {"node": "remote"}},
    )
    installation = RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities={"endpoint_types": ["future_endpoint"]},
    )
    executor = rust_executor.RustSubprocessExecutor(
        base_dir=tmp_path,
        installation=installation,
    )
    monkeypatch.setattr(
        executor,
        "_resolve_run",
        lambda _run: pytest.fail("v2-only selection reached resolver side effects"),
    )

    result = executor.execute_sync(run)

    assert result.success is False
    assert "protocol v1" in (result.error or "")
    assert not run.artifact_dir.exists()


def test_readiness_is_authored_in_v2_and_rejected_by_v1(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.cfg.endpoint.wait_for_model_timeout = 90.0
    run.cfg.endpoint.wait_for_model_interval = 1.25
    run.cfg.endpoint.wait_for_model_mode = "both"

    endpoint = build_authored_run_request(
        run,
        operation="validate",
        expected_distribution_id="blake3:readiness",
    )["run"]["endpoints"]["profiles"][0]

    assert endpoint["wait_for_model_timeout"] == 90.0
    assert endpoint["wait_for_model_interval"] == 1.25
    assert endpoint["wait_for_model_mode"] == "both"
    with pytest.raises(RustWireError, match="protocol v1 cannot honor.*readiness"):
        build_run_request(run)


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
