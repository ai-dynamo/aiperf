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
def test_v2_envelope_is_benchmark_run_shaped(
    tmp_path: Path, operation: str
) -> None:
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
