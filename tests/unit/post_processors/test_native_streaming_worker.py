# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError

from aiperf.rust_shims.live_streaming_worker import (
    _ACTIVATION_CONTROL_ADAPTER,
    _WORKER_EVENT_ADAPTER,
    ActivateEvent,
    InitializeEvent,
    PhaseStatsEvent,
    _build_run,
    _phase_data,
    main,
)


def _initialize(tmp_path: Path) -> InitializeEvent:
    return InitializeEvent.model_validate(
        {
            "protocol_version": 1,
            "event": "initialize",
            "benchmark_id": "native-stream-test",
            "config": {
                "models": ["mock-model"],
                "endpoint_type": "chat",
                "endpoint_urls": ["http://127.0.0.1:8000/v1/chat/completions"],
                "streaming": True,
                "artifact_dir": str(tmp_path),
                "otel": {
                    "metrics_url": "http://127.0.0.1:4318",
                    "custom_resource_attributes": {"team": "inference"},
                    "gen_ai_provider": "nvidia",
                },
                "mlflow": {
                    "tracking_uri": "http://127.0.0.1:5000",
                    "experiment": "native",
                    "run_name": "wire",
                    "tags": {"source": "rust"},
                    "parent_run_id": "parent",
                    "artifact_globs": ["*.json"],
                },
            },
        }
    )


def _phase_event() -> PhaseStatsEvent:
    return PhaseStatsEvent.model_validate(
        {
            "protocol_version": 1,
            "event": "phase_stats",
            "observed_at_ns": 3_500_000_000,
            "stats": {
                "phase_id": "profiling",
                "kind": "profiling",
                "state": "started",
                "start_ns": 500_000_000,
                "sent_end_ns": None,
                "requests_end_ns": None,
                "total_expected_requests": 8,
                "expected_num_sessions": None,
                "expected_duration_ns": 10_000_000_000,
                "grace_period": {"kind": "finite", "duration_ns": 2_000_000_000},
                "requests_sent": 5,
                "requests_completed": 3,
                "requests_cancelled": 1,
                "request_errors": 1,
                "sent_sessions": 4,
                "completed_sessions": 2,
                "cancelled_sessions": 1,
                "total_session_turns": 7,
                "in_flight_requests": 1,
                "in_flight_sessions": 1,
                "in_flight_prefills": 0,
                "pending_branch_work": 0,
                "stuck_session_slots_released": 0,
                "stuck_prefill_slots_released": 0,
                "final_requests_sent": None,
                "final_requests_completed": None,
                "final_requests_cancelled": None,
                "final_request_errors": None,
                "final_sent_sessions": None,
                "final_completed_sessions": None,
                "final_cancelled_sessions": None,
                "timeout_triggered": False,
                "grace_period_timeout_triggered": False,
                "cancel_drain_timeout_triggered": False,
                "forced_completion": False,
                "was_cancelled": False,
                "completion_reason": None,
            },
        }
    )


def test_builds_real_config_v2_run_for_the_canonical_processor(tmp_path: Path) -> None:
    run = _build_run(_initialize(tmp_path))

    assert run.benchmark_id == "native-stream-test"
    assert run.cfg.get_model_names() == ["mock-model"]
    assert str(run.cfg.endpoint.type) == "chat"
    assert run.cfg.otel.metrics_url == "http://127.0.0.1:4318/v1/metrics"
    assert run.cfg.otel.custom_resource_attributes == {"team": "inference"}
    assert run.cfg.mlflow.tracking_uri == "http://127.0.0.1:5000"
    assert run.cfg.mlflow.tags_dict == {"source": "rust"}
    assert run.artifact_dir == tmp_path


def test_native_phase_snapshot_preserves_exact_counters_and_clock_elapsed() -> None:
    stats = _phase_data(_phase_event())

    assert stats.requests_sent == 5
    assert stats.requests_completed == 3
    assert stats.requests_cancelled == 1
    assert stats.total_session_turns == 7
    assert stats.in_flight_requests == 1
    assert stats.in_flight_sessions == 1
    assert stats.expected_duration_sec == 10.0
    assert stats.expected_grace_period_sec == 2.0
    assert stats.requests_elapsed_time == 3.0


def test_worker_event_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _WORKER_EVENT_ADAPTER.validate_python(
            {
                "protocol_version": 1,
                "event": "shutdown",
                "dropped_events": 0,
                "silently_ignored": True,
            }
        )


def test_activation_is_a_distinct_strict_control_barrier() -> None:
    activation = _ACTIVATION_CONTROL_ADAPTER.validate_python(
        {"protocol_version": 1, "event": "activate"}
    )
    assert isinstance(activation, ActivateEvent)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        _ACTIVATION_CONTROL_ADAPTER.validate_python(
            {
                "protocol_version": 1,
                "event": "activate",
                "artifact_created": True,
            }
        )


def test_worker_rejects_shim_arguments() -> None:
    assert main(["unexpected"]) == 2


def test_worker_prepares_before_artifact_creation_and_activates_afterward(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    initialize = _initialize(artifact_dir).model_dump(mode="json")
    initialize["config"]["otel"]["metrics_url"] = None
    initialize["config"]["mlflow"]["tracking_uri"] = None
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "aiperf.rust_shims",
            "live-streaming",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    def exchange(event: dict[str, object]) -> dict[str, object]:
        process.stdin.write(orjson.dumps(event) + b"\n")
        process.stdin.flush()
        response = process.stdout.readline()
        assert response, process.stderr.read().decode(errors="replace")
        parsed = orjson.loads(response)
        assert isinstance(parsed, dict)
        return parsed

    prepared = exchange(initialize)
    assert prepared["event"] == "prepared"
    assert prepared["active"] is False
    assert not artifact_dir.exists()

    artifact_dir.mkdir()
    ready = exchange({"protocol_version": 1, "event": "activate"})
    assert ready["event"] == "ready"
    assert ready["active"] is False
    terminal = exchange(
        {"protocol_version": 1, "event": "shutdown", "dropped_events": 0}
    )
    assert terminal["event"] == "terminal"
    assert terminal["success"] is True

    process.stdin.close()
    assert process.wait(timeout=10) == 0, process.stderr.read().decode(errors="replace")
