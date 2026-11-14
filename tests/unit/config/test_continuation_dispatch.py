# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Continuation policy survives CLI, YAML, and timing configuration resolution."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.config.flags import CLIConfig
from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.config.phases import PhaseConfig
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import (
    CreditPhaseConfig,
    RequestCancellationConfig,
    _build_phase_config,
)


@pytest.mark.parametrize("phase_type", ["constant", "poisson", "gamma"])
@pytest.mark.parametrize("kind", ["warmup", "profiling"])
def test_direct_phase_reaches_timing_config(phase_type: str, kind: str) -> None:
    phase = TypeAdapter(PhaseConfig).validate_python(
        {
            "name": kind,
            "type": phase_type,
            "rate": 4,
            "requests": 9,
            "continuationDispatch": "direct",
        }
    )
    timing = _build_phase_config(
        phase,
        default_cancellation=RequestCancellationConfig(),
        phase_index=0,
        profiling_index=0 if kind == "profiling" else None,
    )
    assert timing.continuation_dispatch == "direct"
    assert timing.request_rate == 4


@pytest.mark.parametrize(
    "phase",
    [
        param({"type": "concurrency"}, id="concurrency"),
        param({"type": "fixed_schedule"}, id="fixed-schedule"),
        param({"type": "user_centric", "rate": 4, "users": 1}, id="user-centric"),
        param(
            {
                "type": "poisson",
                "rate": 4,
                "duration": 10,
                "adaptive_scale": True,
                "adaptive_sustain_duration": 1,
                "adaptive_control_variable": "request_rate",
                "sla": [{"metric_tag": "request_latency", "op": "le", "threshold": 1000}],
            },
            id="adaptive",
        ),
        param(
            {"type": "poisson", "rate": 4, "timing_mode": TimingMode.AGENTIC_REPLAY},
            id="agentic-replay",
        ),
    ],
)  # fmt: skip
def test_direct_phase_rejects_incompatible_scheduling(phase: dict) -> None:
    with pytest.raises(ValidationError, match="continuation"):
        TypeAdapter(PhaseConfig).validate_python(
            {
                "name": "profiling",
                "requests": 9,
                "continuation_dispatch": "direct",
                **phase,
            }
        )


def test_cli_policy_defaults_to_rate_slot() -> None:
    phase = TypeAdapter(PhaseConfig).validate_python(
        {
            "name": "profiling",
            **build_profiling(CLIConfig(request_rate=4, request_count=9)),
        }
    )
    assert phase.continuation_dispatch == "rate_slot"


def test_cli_direct_policy_requires_request_rate() -> None:
    with pytest.raises(ValueError, match="--continuation-dispatch"):
        build_profiling(CLIConfig(request_count=9, continuation_dispatch="direct"))


def test_cli_direct_policy_routes() -> None:
    result = build_profiling(
        CLIConfig(request_rate=4, request_count=9, continuation_dispatch="direct")
    )
    assert result["continuation_dispatch"] == "direct"


def test_cli_policy_defers_scenario_compatibility_to_scenario_resolution() -> None:
    result = build_profiling(
        CLIConfig(
            request_rate=4,
            request_count=9,
            continuation_dispatch="direct",
            scenario="agentx",
            unsafe_override=True,
        )
    )
    assert result["continuation_dispatch"] == "direct"


@pytest.mark.parametrize("policy", ["rate_slot", "direct"])
def test_config_flag_overrides_only_profiling_policy(
    tmp_path: Path, policy: str
) -> None:
    config_path = tmp_path / "rate.yaml"
    config_path.write_text(
        """schemaVersion: '2.0'
benchmark:
  model: test-model
  endpoint: {url: 'http://localhost:8000'}
  dataset: {type: synthetic}
  phases:
    - {name: warmup, type: constant, rate: 4, requests: 3}
    - {name: profiling, type: poisson, rate: 4, requests: 9, continuationDispatch: direct}
"""
    )
    config = resolve_config(CLIConfig(continuation_dispatch=policy), config_path)
    warmup, profiling = config.benchmark.phases
    assert warmup.continuation_dispatch == "rate_slot"
    assert profiling.continuation_dispatch == policy


@pytest.mark.parametrize("phase_type", ["constant", "poisson", "gamma"])
@pytest.mark.parametrize("policy", [None, "rate_slot", "direct"])
@pytest.mark.parametrize("sweep", [False, True])
def test_templated_phase_policy_matches_executed_plan(
    tmp_path: Path, phase_type: str, policy: str | None, sweep: bool
) -> None:
    """CLI policy overrides reach both rendered validation and sweep execution."""
    config_path = tmp_path / "templated-rate.yaml"
    config_path.write_text(
        f"""variables:
  ptype: {phase_type}
benchmark:
  model: test-model
  endpoint: {{url: 'http://localhost:8000'}}
  dataset: {{type: synthetic}}
  phases:
    - {{name: warmup, type: constant, rate: 4, requests: 3}}
    - name: profiling
      type: "{{{{ ptype }}}}"
      rate: 4
      requests: 9
      continuationDispatch: rate_slot
"""
        + (
            "sweep:\n  type: grid\n  parameters:\n"
            "    phases.profiling.requests: [9, 12]\n"
            if sweep
            else ""
        ),
        encoding="utf-8",
    )
    cli = CLIConfig() if policy is None else CLIConfig(continuation_dispatch=policy)
    config = resolve_config(cli, config_path)
    plan = build_benchmark_plan(config)
    expected_policy = policy or "rate_slot"

    assert config.benchmark.phases[1].type == phase_type
    assert config.benchmark.phases[1].continuation_dispatch == expected_policy
    assert len(plan.configs) == (2 if sweep else 1)
    assert {run.phases[1].requests for run in plan.configs} == (
        {9, 12} if sweep else {9}
    )
    for run in plan.configs:
        warmup, profiling = run.phases
        assert warmup.continuation_dispatch == "rate_slot"
        assert profiling.type == phase_type
        assert profiling.continuation_dispatch == expected_policy


@pytest.mark.parametrize("policy", ["rate_slot", "direct"])
@pytest.mark.parametrize(
    "phase_type, fields",
    [
        param("concurrency", "concurrency: 1", id="concurrency"),
        param("fixed_schedule", "", id="fixed-schedule"),
        param("user_centric", "rate: 4\n      users: 1", id="user-centric"),
    ],
)  # fmt: skip
def test_templated_incompatible_phase_policy_names_flag(
    tmp_path: Path, policy: str, phase_type: str, fields: str
) -> None:
    """Rendered incompatible phases still reject the explicitly supplied flag."""
    config_path = tmp_path / "incompatible.yaml"
    config_path.write_text(
        f"""variables:
  ptype: {phase_type}
benchmark:
  model: test-model
  endpoint: {{url: 'http://localhost:8000'}}
  dataset: {{type: synthetic}}
  phases:
    - name: profiling
      type: "{{{{ ptype }}}}"
      requests: 9
      {fields}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="--continuation-dispatch"):
        resolve_config(CLIConfig(continuation_dispatch=policy), config_path)


@pytest.mark.parametrize(
    "overrides",
    [
        param({"timing_mode": TimingMode.USER_CENTRIC_RATE}, id="user-centric"),
        param({"timing_mode": TimingMode.AGENTIC_REPLAY}, id="scenario-override"),
        param({"timing_mode": TimingMode.ADAPTIVE_SCALE}, id="adaptive"),
        param({"request_rate": None}, id="unlimited-rate"),
    ],
)  # fmt: skip
def test_resolved_timing_rejects_direct_policy_in_other_modes(overrides: dict) -> None:
    values = {
        "phase": CreditPhase.PROFILING,
        "timing_mode": TimingMode.REQUEST_RATE,
        "request_rate": 4,
        "arrival_pattern": ArrivalPattern.CONSTANT,
        "continuation_dispatch": "direct",
        **overrides,
    }
    with pytest.raises(ValidationError, match="continuation-dispatch"):
        CreditPhaseConfig(**values)
