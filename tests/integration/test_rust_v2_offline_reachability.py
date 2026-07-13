# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Product reachability proofs for the feature-bearing offline runner pairs."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.runner_installation import RunnerInstallation
from aiperf.orchestrator.rust_executor import RustSubprocessExecutor


def _runner_binary() -> Path:
    default = Path(__file__).resolve().parents[2] / "target/debug/aiperf-runner"
    return Path(os.environ.get("AIPERF_RUNNER_BIN", default))


@pytest.fixture(scope="module")
def offline_installation() -> RunnerInstallation:
    """Resolve and identity-check the exact feature-built runner image once."""
    installation = RunnerInstallation.resolve(_runner_binary())
    assert installation.supports_pair("dynamo_offline", "scheduled")
    assert installation.supports_pair("dynamo_offline", "graph")
    return installation


def _base_benchmark(artifact_dir: Path) -> dict[str, Any]:
    return {
        "models": ["mock-model"],
        "endpoint": {
            "urls": ["http://127.0.0.1:9"],
            "type": "chat",
            "streaming": True,
        },
        "transport": {
            "type": "dynamo_offline",
            "artifacts": {
                "report_json": "dynamo/report.json",
                "per_request_jsonl": "dynamo/requests.jsonl",
            },
        },
        "tokenizer": {"name": "builtin"},
        "runtime": {"workers": 1, "ui": "none"},
        "gpu_telemetry": {"enabled": False},
        "server_metrics": {"enabled": False},
        "network_latency": {"enabled": False},
        "artifacts": {
            "dir": str(artifact_dir),
            "records": False,
            "raw": False,
            "trace": False,
            "export_outputs_json": False,
        },
    }


def _scheduled_run(artifact_dir: Path) -> BenchmarkRun:
    benchmark = _base_benchmark(artifact_dir)
    benchmark["transport"].update(
        {
            "topology": "aggregated",
            "workers": 2,
            "router_mode": "kv",
            "sla": {"e2e_ms": 1000.0},
        }
    )
    benchmark.update(
        {
            "dataset": {
                "type": "synthetic",
                "entries": 4,
                "prompts": {"isl": 8, "osl": 2},
            },
            "phases": [
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 1,
                },
                {
                    "name": "profiling",
                    "type": "constant",
                    "requests": 4,
                    "rate": 100.0,
                    "concurrency": 2,
                    "prefill_concurrency": 2,
                },
            ],
        }
    )
    config = AIPerfConfig.model_validate({"benchmark": benchmark})
    return BenchmarkRun(
        benchmark_id="python-v2-offline-scheduled",
        cfg=config.benchmark,
        artifact_dir=artifact_dir,
        label="offline-scheduled",
        random_seed=41,
    )


def _graph_rows() -> list[dict[str, Any]]:
    return [
        {
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "root"}],
                    "forks": [{"child": "child", "background": True}],
                    "max_tokens": 2,
                },
                {
                    "messages": [{"role": "user", "content": "joined"}],
                    "max_tokens": 2,
                },
            ],
        },
        {
            "session_id": "child",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "child"}],
                    "max_tokens": 2,
                }
            ],
        },
    ]


def _graph_run(artifact_dir: Path) -> BenchmarkRun:
    benchmark = _base_benchmark(artifact_dir)
    benchmark.update(
        {
            "dataset": {
                "type": "file",
                "format": "dag_jsonl",
                "records": _graph_rows(),
            },
            "phases": [
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "requests": 3,
                    "concurrency": 1,
                },
                {
                    "name": "profiling",
                    "type": "constant",
                    "requests": 3,
                    "duration": 1.0,
                    "rate": 100.0,
                    "concurrency": 2,
                    "prefill_concurrency": 2,
                    "seamless": True,
                    "grace_period": 0.01,
                    "rate_ramp": {"duration": 0.001, "strategy": "linear"},
                    "cancellation": {"rate": 0.0, "delay": 0.001},
                }
            ],
        }
    )
    config = AIPerfConfig.model_validate({"benchmark": benchmark})
    return BenchmarkRun(
        benchmark_id="python-v2-offline-graph",
        cfg=config.benchmark,
        artifact_dir=artifact_dir,
        label="offline-graph",
        random_seed=17,
    )


def _execute_v2(
    monkeypatch: pytest.MonkeyPatch,
    installation: RunnerInstallation,
    run: BenchmarkRun,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[bytes], Any]:
    """Execute through the product's protocol-v2-only subprocess boundary."""
    original_execute = RunnerInstallation.execute
    captured: list[tuple[dict[str, Any], subprocess.CompletedProcess[bytes]]] = []

    def recording_execute(
        selected: RunnerInstallation, request: dict[str, Any]
    ) -> subprocess.CompletedProcess[bytes]:
        completed = original_execute(selected, request)
        captured.append((copy.deepcopy(request), completed))
        return completed

    monkeypatch.setattr(RunnerInstallation, "execute", recording_execute)
    result = RustSubprocessExecutor(
        run.artifact_dir,
        installation=installation,
    ).execute_sync(run)

    assert len(captured) == 1
    request, completed = captured[0]
    return request, completed, result


def _terminal(completed: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, completed.stderr.decode(errors="replace")
    terminal = orjson.loads(lines[0])
    assert isinstance(terminal, dict)
    return terminal


def _assert_product_result(
    *,
    run: BenchmarkRun,
    request: dict[str, Any],
    completed: subprocess.CompletedProcess[bytes],
    result: Any,
    installation: RunnerInstallation,
    workload: str,
    mode: str,
    request_count: int,
    parity_shared_fields: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert request["protocol_version"] == 2
    assert request["operation"] == "execute"
    assert request["expected_distribution_id"] == installation.distribution_id
    assert request["run"]["transport"]["type"] == "dynamo_offline"
    assert request["run"]["workload"]["type"] == workload

    terminal = _terminal(completed)
    assert completed.returncode == 0, terminal
    assert terminal["success"] is True
    assert terminal["distribution_id"] == installation.distribution_id
    assert terminal["provenance"]["transport"] == "dynamo_offline"
    assert terminal["provenance"]["workload"] == workload
    assert terminal["provenance"]["parity_shared_fields"] == parity_shared_fields

    native_path = run.artifact_dir / "native-v2.json"
    assert Path(terminal["report_path"]) == native_path
    assert native_path.is_file()
    assert result.success, result.error
    assert result.artifacts_path == run.artifact_dir
    assert result.summary_metrics["request_count"].avg == float(request_count)

    native = orjson.loads(native_path.read_bytes())
    assert native["schema_version"] == "2.0"
    assert native["run"]["mode"] == mode
    assert (run.artifact_dir / "profile_export_aiperf.json").is_file()
    return terminal, native


def test_python_config_v2_reaches_offline_scheduled_pair_without_v1_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline_installation: RunnerInstallation,
) -> None:
    artifact_dir = tmp_path / "scheduled"
    run = _scheduled_run(artifact_dir)

    request, completed, result = _execute_v2(
        monkeypatch,
        offline_installation,
        run,
    )
    terminal, native = _assert_product_result(
        run=run,
        request=request,
        completed=completed,
        result=result,
        installation=offline_installation,
        workload="scheduled",
        mode="offline:scheduled",
        request_count=4,
        parity_shared_fields="77",
    )

    assert terminal["provenance"]["phase_count"] == "2"
    assert terminal["provenance"]["topology"] == "aggregated"
    assert terminal["provenance"]["router"] == "kv"
    assert native["metrics"]["goodput"]
    dynamo = orjson.loads((artifact_dir / "dynamo/report.json").read_bytes())
    assert dynamo["num_requests"] == 6
    assert len((artifact_dir / "dynamo/requests.jsonl").read_text().splitlines()) == 6


def test_python_config_v2_reaches_direct_graph_adapter_without_dual_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline_installation: RunnerInstallation,
) -> None:
    artifact_dir = tmp_path / "graph"
    run = _graph_run(artifact_dir)

    request, completed, result = _execute_v2(
        monkeypatch,
        offline_installation,
        run,
    )
    terminal, native = _assert_product_result(
        run=run,
        request=request,
        completed=completed,
        result=result,
        installation=offline_installation,
        workload="graph",
        mode="offline:graph",
        request_count=3,
        parity_shared_fields="74",
    )

    dataset = request["run"]["workload"]["config"]["dataset"]
    assert dataset["format"] == "dag_jsonl"
    assert dataset["records"] == _graph_rows()
    assert "graph_ir" not in dataset
    assert "conversation" not in dataset
    assert terminal["provenance"]["phase_count"] == "2"
    assert native["run"]["graph"]["phase_count"] == 2
    assert native["run"]["graph"]["outcome"]["admitted"] == 2
    assert native["run"]["graph"]["outcome"]["completed"] == 2
    assert native["warmup_metrics"]
    dynamo = orjson.loads((artifact_dir / "dynamo/report.json").read_bytes())
    assert dynamo["completed_requests"] == 6
    assert len((artifact_dir / "dynamo/requests.jsonl").read_text().splitlines()) == 6


def test_online_only_capability_image_rejects_offline_before_legacy_resolution(
    tmp_path: Path,
    offline_installation: RunnerInstallation,
) -> None:
    """A runner without the pair cannot reinterpret an offline run as v1 HTTP."""
    capabilities = copy.deepcopy(offline_installation.capabilities)
    capabilities["supported_pairs"] = [
        pair for pair in capabilities["supported_pairs"] if pair[0] == "http"
    ]
    capabilities["transports"] = [
        backend
        for backend in capabilities["transports"]
        if backend["id"] == "http"
    ]
    online_only = RunnerInstallation(
        binary=offline_installation.binary,
        capabilities=capabilities,
    )
    run = _scheduled_run(tmp_path / "must-not-exist")

    executor = RustSubprocessExecutor(tmp_path, installation=online_only)

    result = executor.execute_sync(run)

    assert result.success is False
    assert "executable protocol-v2 pair ('dynamo_offline', 'scheduled')" in (
        result.error or ""
    )
    assert not run.artifact_dir.exists()
