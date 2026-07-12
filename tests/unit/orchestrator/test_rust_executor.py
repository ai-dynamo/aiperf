# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Rust-runner capability negotiation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from aiperf.orchestrator import runner_installation, rust_executor

_TEST_DISTRIBUTION_ID = "blake3:" + "a" * 64


def _completed(payload: object, *, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["aiperf-runner", "--capabilities"],
        returncode=returncode,
        stdout=orjson.dumps(payload) + b"\n",
        stderr=b"runner diagnostic" if returncode else b"",
    )


def _capabilities(*endpoint_types: str) -> dict[str, object]:
    return {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "distribution_id": _TEST_DISTRIBUTION_ID,
        "endpoint_types": list(endpoint_types),
        "dataset_types": ["synthetic"],
        "phase_types": ["concurrency"],
        "phase_features": [],
        "run_features": ["http_transport_policy", "thread_per_core_execution"],
        "telemetry_source_types": [],
        "server_metrics_formats": [],
        "runner_version": "0.0.0",
    }


def test_capabilities_accept_matching_protocol_and_report_schema(monkeypatch) -> None:
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "distribution_id": _TEST_DISTRIBUTION_ID,
        "endpoint_types": ["chat"],
        "dataset_types": ["synthetic"],
        "phase_types": ["concurrency"],
        "phase_features": [],
        "run_features": [],
        "telemetry_source_types": [],
        "server_metrics_formats": [],
        "runner_version": "0.0.0",
    }
    monkeypatch.setattr(
        runner_installation,
        "_runner_distribution_id",
        lambda _binary: _TEST_DISTRIBUTION_ID,
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    assert runner_installation._load_capabilities(Path("runner")) == response


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
        "distribution_id": _TEST_DISTRIBUTION_ID,
        "endpoint_types": [],
        "dataset_types": [],
        "phase_types": [],
        "phase_features": [],
        "run_features": [],
        "telemetry_source_types": [],
        "server_metrics_formats": [],
    }
    response[field] = value
    monkeypatch.setattr(
        runner_installation,
        "_runner_distribution_id",
        lambda _binary: _TEST_DISTRIBUTION_ID,
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    with pytest.raises((RuntimeError, ValueError), match=match):
        runner_installation._load_capabilities(Path("runner"))


def test_capabilities_surface_process_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_installation,
        "_runner_distribution_id",
        lambda _binary: _TEST_DISTRIBUTION_ID,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["runner", "--capabilities"],
            returncode=2,
            stdout=b"",
            stderr=b"x-api-key: runner-secret\nrunner diagnostic",
        ),
    )

    with pytest.raises(RuntimeError, match="exit 2") as raised:
        runner_installation._load_capabilities(Path("runner"))
    assert "runner diagnostic" in str(raised.value)
    assert "runner-secret" not in str(raised.value)
    assert "<redacted>" in str(raised.value)


def test_distribution_identity_algorithm_is_pinned(tmp_path: Path) -> None:
    binary = tmp_path / "aiperf-runner"
    binary.write_bytes(b"runner-image\0with-binary-bytes")

    assert runner_installation._runner_distribution_id(binary) == (
        "blake3:3cef527b3dd7185b4ab8590b425b730bd84d695f5c9e1a97302780b7056bf2e9"
    )


def test_protocol_v2_validation_response_is_bound_to_image_and_exit() -> None:
    payload = {
        "protocol_version": 2,
        "event": "run_validation",
        "distribution_id": _TEST_DISTRIBUTION_ID,
        "benchmark_id": "validate-me",
        "success": True,
        "completeness": "static",
        "deferred_checks": [
            {
                "code": "dataset_load",
                "path": "run.workload",
                "reason": "requires execution preparation",
            }
        ],
    }

    response = runner_installation._parse_validation_response(
        orjson.dumps(payload) + b"\n",
        benchmark_id="validate-me",
        distribution_id=_TEST_DISTRIBUTION_ID,
        returncode=0,
    )

    assert response["success"] is True
    with pytest.raises(ValueError, match="exit code"):
        runner_installation._parse_validation_response(
            orjson.dumps(payload) + b"\n",
            benchmark_id="validate-me",
            distribution_id=_TEST_DISTRIBUTION_ID,
            returncode=1,
        )


def test_protocol_v2_validation_failure_requires_typed_errors() -> None:
    payload = {
        "protocol_version": 2,
        "event": "run_validation",
        "distribution_id": _TEST_DISTRIBUTION_ID,
        "benchmark_id": "validate-me",
        "success": False,
        "completeness": "static",
        "errors": [],
    }

    with pytest.raises(ValueError, match="must contain errors"):
        runner_installation._parse_validation_response(
            orjson.dumps(payload) + b"\n",
            benchmark_id="validate-me",
            distribution_id=_TEST_DISTRIBUTION_ID,
            returncode=1,
        )


def test_capabilities_reject_distribution_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "aiperf-runner"
    binary.write_bytes(b"selected-runner-image")
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "distribution_id": "blake3:" + "0" * 64,
        "endpoint_types": [],
        "dataset_types": [],
        "phase_types": [],
        "phase_features": [],
        "run_features": [],
        "telemetry_source_types": [],
        "server_metrics_formats": [],
    }
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    with pytest.raises(RuntimeError, match="does not match.*exact selected"):
        runner_installation._load_capabilities(binary)


def test_installation_rejects_binary_replacement_before_execution(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "aiperf-runner"
    binary.write_bytes(b"negotiated-runner-image")
    installation = runner_installation.RunnerInstallation(
        binary=binary,
        capabilities={
            "distribution_id": runner_installation._runner_distribution_id(binary)
        },
    )
    binary.write_bytes(b"replacement-runner-image")

    with pytest.raises(RuntimeError, match="no longer matches|was replaced"):
        installation.verify_distribution_identity()


def test_runner_installation_accepts_runner_owned_endpoint_ids() -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_capabilities("chat", "messages", "acme_chat"),
    )

    installation.preflight_endpoint("messages")
    installation.preflight_endpoint("acme_chat")


def test_runner_installation_rejects_unavailable_endpoint_clearly() -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_capabilities("chat", "messages"),
    )

    with pytest.raises(RuntimeError, match="not compiled.*chat, messages") as raised:
        installation.preflight_endpoint("acme_chat")

    message = str(raised.value)
    assert "AIPERF_RUNNER_BIN" in message
    assert "plugin" not in message.lower()


def test_runner_installation_preflights_every_fixed_plan_endpoint() -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_capabilities("chat", "messages"),
    )
    plan = SimpleNamespace(
        configs=[
            SimpleNamespace(
                endpoint=SimpleNamespace(type="chat"), endpoint_profiles={}
            ),
            SimpleNamespace(
                endpoint=SimpleNamespace(type="future_compiled_endpoint"),
                endpoint_profiles={},
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="future_compiled_endpoint"):
        installation.preflight_plan(plan)


def test_executor_rejects_endpoint_before_resolution_without_secret_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_capabilities("chat", "messages"),
    )
    executor = rust_executor.RustSubprocessExecutor(
        base_dir=tmp_path,
        installation=installation,
    )
    monkeypatch.setattr(
        executor,
        "_resolve_run",
        lambda _run: pytest.fail("config resolution must not run"),
    )
    run = SimpleNamespace(
        cfg=SimpleNamespace(
            endpoint=SimpleNamespace(type="acme_chat", api_key="sk-never-log-me")
        ),
        label="custom-endpoint",
        trial=0,
        artifact_dir=tmp_path,
    )

    result = executor.execute_sync(run)

    assert result.success is False
    assert "acme_chat" in result.error
    assert "available endpoints: chat, messages" in result.error
    assert "sk-never-log-me" not in result.error


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


def test_terminal_failure_redacts_runner_error_and_stderr(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(
        args=["runner"],
        returncode=1,
        stdout=b"",
        stderr=b"x-api-key: stderr-secret",
    )

    result = rust_executor._failure(
        completed,
        {"error": "Authorization: Bearer terminal-secret"},
        SimpleNamespace(label="failed", artifact_dir=tmp_path),
    )

    assert result.success is False
    assert "terminal-secret" not in result.error
    assert "stderr-secret" not in result.error
    assert result.error.count("<redacted>") == 2


@pytest.mark.parametrize(
    "field",
    [
        "endpoint_types",
        "dataset_types",
        "phase_types",
        "phase_features",
        "run_features",
        "telemetry_source_types",
        "server_metrics_formats",
    ],
)
def test_capabilities_require_every_typed_feature_inventory(
    monkeypatch, field: str
) -> None:
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "distribution_id": _TEST_DISTRIBUTION_ID,
        "endpoint_types": [],
        "dataset_types": [],
        "phase_types": [],
        "phase_features": [],
        "run_features": [],
        "telemetry_source_types": [],
        "server_metrics_formats": [],
    }
    response.pop(field)
    monkeypatch.setattr(
        runner_installation,
        "_runner_distribution_id",
        lambda _binary: _TEST_DISTRIBUTION_ID,
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    with pytest.raises(ValueError, match=field):
        runner_installation._load_capabilities(Path("runner"))


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
            "thread_per_core_execution",
            "network_latency",
            "server_metrics",
            "python_live_streaming",
        ],
        "telemetry_source_types": ["dcgm"],
        "server_metrics_formats": ["json", "csv", "jsonl", "parquet"],
    }
    request = {
        "run": {
            "workers": 2,
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
            "server_metrics": {"formats": ["json", "parquet"]},
            "live_streaming": {},
        }
    }

    runner_installation._require_request_capabilities(capabilities, request)

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
        ("run_features", "thread_per_core_execution"),
        ("run_features", "network_latency"),
        ("run_features", "server_metrics"),
        ("run_features", "python_live_streaming"),
        ("telemetry_source_types", "dcgm"),
        ("server_metrics_formats", "json"),
        ("server_metrics_formats", "parquet"),
    ):
        narrowed = {name: list(values) for name, values in capabilities.items()}
        narrowed[field].remove(value)
        with pytest.raises(RuntimeError, match=rf"{field}\.{value}"):
            runner_installation._require_request_capabilities(narrowed, request)


def test_v2_request_capabilities_read_endpoint_profiles_from_resources() -> None:
    capabilities = {
        "supported_pairs": [["online_grpc", "scheduled"]],
        "endpoint_types": ["kserve_v2_infer"],
    }
    request = {
        "run": {
            "backend": {"type": "online_grpc"},
            "workload": {"type": "scheduled"},
            "resources": {
                "endpoints": {
                    "profiles": [{"id": "default", "type": "kserve_v2_infer"}]
                }
            },
        }
    }

    runner_installation._require_v2_request_capabilities(capabilities, request)


def test_v2_request_capabilities_defer_resource_presence_to_workload_registry() -> None:
    capabilities = {
        "supported_pairs": [["telemetry_archive", "telemetry_watch"]],
        "endpoint_types": [],
    }
    request = {
        "run": {
            "backend": {"type": "telemetry_archive"},
            "workload": {"type": "telemetry_watch"},
            "resources": {},
        }
    }

    runner_installation._require_v2_request_capabilities(capabilities, request)
