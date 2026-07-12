# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Rust-runner capability negotiation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from aiperf.orchestrator import rust_executor


def _completed(payload: object, *, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["aiperf-runner", "--capabilities"],
        returncode=returncode,
        stdout=orjson.dumps(payload) + b"\n",
        stderr=b"runner diagnostic" if returncode else b"",
    )


def test_capabilities_accept_matching_protocol_and_report_schema(monkeypatch) -> None:
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "endpoint_types": ["chat"],
        "dataset_types": ["synthetic"],
        "phase_types": ["concurrency"],
        "phase_features": [],
        "run_features": [],
        "telemetry_source_types": [],
        "runner_version": "0.0.0",
    }
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    assert rust_executor._load_capabilities(Path("runner")) == response


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("protocol_versions", [2], "does not support protocol 1"),
        ("report_schema_version", "3.0", "report schema '3.0' is incompatible"),
        ("event", "something_else", "unknown capability response"),
    ],
)
def test_capabilities_reject_incompatible_runner(
    monkeypatch, field: str, value: object, match: str
) -> None:
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "endpoint_types": [],
        "dataset_types": [],
        "phase_types": [],
        "phase_features": [],
        "run_features": [],
        "telemetry_source_types": [],
    }
    response[field] = value
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    with pytest.raises((RuntimeError, ValueError), match=match):
        rust_executor._load_capabilities(Path("runner"))


def test_capabilities_surface_process_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _completed({}, returncode=2),
    )

    with pytest.raises(RuntimeError, match="exit 2.*runner diagnostic"):
        rust_executor._load_capabilities(Path("runner"))


def test_missing_terminal_surfaces_exit_and_redacted_stderr() -> None:
    with pytest.raises(ValueError) as raised:
        rust_executor._parse_terminal(
            b"",
            SimpleNamespace(benchmark_id="run"),
            returncode=-6,
            stderr=b"Authorization: Bearer runner-secret\nstack overflow",
        )

    message = str(raised.value)
    assert "child exit code -6" in message
    assert "stack overflow" in message
    assert "runner-secret" not in message
    assert "<redacted>" in message


@pytest.mark.parametrize(
    "field",
    [
        "endpoint_types",
        "dataset_types",
        "phase_types",
        "phase_features",
        "run_features",
        "telemetry_source_types",
    ],
)
def test_capabilities_require_every_typed_feature_inventory(
    monkeypatch, field: str
) -> None:
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "endpoint_types": [],
        "dataset_types": [],
        "phase_types": [],
        "phase_features": [],
        "run_features": [],
        "telemetry_source_types": [],
    }
    response.pop(field)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    with pytest.raises(ValueError, match=field):
        rust_executor._load_capabilities(Path("runner"))


def test_request_capabilities_cover_every_nested_native_variant() -> None:
    capabilities = {
        "endpoint_types": ["chat"],
        "dataset_types": ["synthetic"],
        "phase_types": ["concurrency"],
        "phase_features": ["adaptive_scale", "ramps", "request_cancellation"],
        "run_features": [
            "gpu_telemetry",
            "outputs_json",
            "python_accuracy_evaluator",
            "raw_records",
            "http_transport_policy",
            "network_latency",
        ],
        "telemetry_source_types": ["dcgm"],
    }
    request = {
        "run": {
            "endpoint": {"type": "chat", "timeout_seconds": 10.0},
            "dataset": {"type": "synthetic"},
            "phases": [
                {
                    "type": "concurrency",
                    "adaptive_scale": {},
                    "concurrency_ramp": {},
                    "cancellation": {},
                }
            ],
            "artifacts": {
                "outputs_path": "outputs.json",
                "raw_path": "profile_export_raw.jsonl",
            },
            "accuracy": {},
            "gpu_telemetry": {"sources": [{"type": "dcgm"}]},
            "network_latency": {"mean_rtt_ns": 2_500_000},
        }
    }

    rust_executor._require_request_capabilities(capabilities, request)

    for field, value in (
        ("endpoint_types", "chat"),
        ("dataset_types", "synthetic"),
        ("phase_types", "concurrency"),
        ("phase_features", "adaptive_scale"),
        ("phase_features", "ramps"),
        ("phase_features", "request_cancellation"),
        ("run_features", "outputs_json"),
        ("run_features", "python_accuracy_evaluator"),
        ("run_features", "gpu_telemetry"),
        ("run_features", "raw_records"),
        ("run_features", "http_transport_policy"),
        ("run_features", "network_latency"),
        ("telemetry_source_types", "dcgm"),
    ):
        narrowed = {name: list(values) for name, values in capabilities.items()}
        narrowed[field].remove(value)
        with pytest.raises(RuntimeError, match=rf"{field}\.{value}"):
            rust_executor._require_request_capabilities(narrowed, request)
