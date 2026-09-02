# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from aiperf.common.scenario.base import (
    EmptyTracePoolError,
    ScenarioLockError,
    ScenarioSpec,
    ScenarioViolation,
    TrajectoryWarmupFailedError,
)
from aiperf.plugin.enums import TimingMode


def test_scenario_spec_is_frozen() -> None:
    spec = ScenarioSpec(
        name="test",
        timing_mode=TimingMode.REQUEST_RATE,
        require_ignore_eos=True,
        require_use_think_time_only=True,
        forbid_input_truncation=True,
        require_loader="weka_trace",
        min_benchmark_duration_seconds=900,
        inter_turn_delay_cap_seconds=60.0,
    )
    with pytest.raises(ValidationError):
        spec.name = "mutated"


def test_scenario_violation_carries_flag_and_values() -> None:
    v = ScenarioViolation(
        flag="--timing-mode",
        current_value="request_rate",
        required_value="agentic_replay",
        message="scenario requires agentic_replay",
    )
    assert v.flag == "--timing-mode"
    assert "agentic_replay" in str(v)


def test_scenario_lock_error_lists_all_violations() -> None:
    violations = [
        ScenarioViolation(flag="--a", current_value=1, required_value=2, message="a"),
        ScenarioViolation(flag="--b", current_value=3, required_value=4, message="b"),
    ]
    err = ScenarioLockError(violations)
    assert "--a" in str(err)
    assert "--b" in str(err)


def test_empty_trace_pool_error_is_runtime_error() -> None:
    err = EmptyTracePoolError("loader returned 0 traces")
    assert isinstance(err, RuntimeError)


def test_trajectory_warmup_failed_error_lists_trace_ids() -> None:
    err = TrajectoryWarmupFailedError(["trace_a", "trace_b"])
    assert "trace_a" in str(err)
    assert "trace_b" in str(err)
