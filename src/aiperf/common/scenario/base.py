# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario config-lock base models (adapted for the ajc/rust tree).

Ported from ``ajc/aiperf-graph-ir:src/aiperf/common/scenario/base.py``.

Adaptations for ajc/rust (see ``docs``/report for the full rationale):

* ``timing_mode`` is a plain ``str`` documented marker instead of a
  ``TimingMode`` enum value. ajc/rust has no ``TimingMode.GRAPH_IR``; a graph
  (weka) workload is selected by dataset FORMAT (``weka_trace``), so the
  validator DETECTS a weka workload rather than matching a per-phase timing
  mode. The field survives only as a human-facing marker of intent.
* ``require_cache_bust`` is a plain ``str | None`` documented marker instead of
  a ``CacheBustTarget`` enum. ajc/rust has no ``endpoint.cache_bust`` knob, so
  the corresponding lock is a documented skip (see
  ``validator._apply_require_cache_bust``). The field is kept so the spec still
  records the contract intent.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from aiperf.common.models import AIPerfBaseModel


class ScenarioSpec(AIPerfBaseModel):
    """Frozen declaration of a benchmark scenario's invariants."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str = Field(description="Scenario identifier, e.g. 'inferencex-agentx-mvp'.")
    timing_mode: str = Field(
        description="Documented marker for the required workload class "
        "(e.g. 'graph_ir'). On ajc/rust a graph workload is selected by dataset "
        "format (weka_trace); the validator detects a weka graph workload rather "
        "than matching a per-phase timing mode, so this field is informational.",
    )
    require_ignore_eos: bool = Field(
        description="Inject ignore_eos=true into endpoint.extra; error on explicit false."
    )
    require_streaming: bool = Field(
        default=False,
        description=(
            "Force --streaming=true (auto-enabled when unset; error on explicit "
            "--no-streaming). Streaming is required for the per-token latency "
            "metrics (TTFT, ITL) that are core to this benchmark; without it a "
            "run would silently report no first-token signal."
        ),
    )
    forbid_input_truncation: bool = Field(
        description=(
            "Reject client-side input-length truncation. Currently checks "
            "`--synthesis-max-isl` (which drops traces whose input length "
            "exceeds the cap)."
        )
    )
    require_loader: str | tuple[str, ...] = Field(
        description=(
            "Required loader plugin name (e.g. 'weka_trace'), or a tuple of "
            "equivalent loader names. The detected loader must match any one "
            "of them — useful when several loader plugins produce byte-identical "
            "data (e.g. file-based vs HF-hosted variants)."
        )
    )
    min_benchmark_duration_seconds: int = Field(
        gt=0, description="Floor on --benchmark-duration in seconds."
    )
    default_benchmark_duration_seconds: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Value auto-filled into --benchmark-duration when the user leaves "
            "it unset. Explicit user values are honored (subject to the "
            "min_benchmark_duration_seconds floor). None disables auto-fill."
        ),
    )
    default_trajectory_start_min_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Trajectory-start (t*) window lower bound the scenario runs with, "
            "living at cfg.trajectory_start_min_ratio "
            "(--trajectory-start-min-ratio). apply_trajectory_ratios "
            "AUTO-APPLIES this value onto the run config when the field is "
            "unset; a user-explicit value differing from this raises "
            "ScenarioLockError naming --trajectory-start-min-ratio "
            "(downgradable to a warning via --unsafe-override). None disables "
            "the check."
        ),
    )
    default_trajectory_start_max_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Trajectory-start (t*) window upper bound the scenario runs with, "
            "living at cfg.trajectory_start_max_ratio "
            "(--trajectory-start-max-ratio). apply_trajectory_ratios "
            "AUTO-APPLIES this value onto the run config when the field is "
            "unset; a user-explicit value differing from this raises "
            "ScenarioLockError naming --trajectory-start-max-ratio "
            "(downgradable to a warning via --unsafe-override). None disables "
            "the check."
        ),
    )
    trace_idle_gap_cap_seconds: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Hard ceiling (seconds) for idle gaps within each root trace. For "
            "recorded graph replay (weka, dynamo), parent + subagent "
            "request-start timestamps are compressed "
            "per-trace before per-turn delays are derived."
        ),
    )
    require_cache_bust: str | None = Field(
        default=None,
        description=(
            "Documented marker for the required first-turn cache-bust target. "
            "ajc/rust has no endpoint.cache_bust knob, so the corresponding lock "
            "is a documented skip (validator._apply_require_cache_bust). Kept so "
            "the spec still records the contract intent."
        ),
    )


class ScenarioViolation(AIPerfBaseModel):
    """A single conflict between user config and a locked scenario invariant."""

    flag: str = Field(
        description="The user-facing flag or config field that conflicts."
    )
    current_value: Any = Field(description="The value the user provided.")
    required_value: Any = Field(description="The value the scenario requires.")
    message: str = Field(description="Human-readable explanation of the conflict.")

    def __str__(self) -> str:
        return (
            f"{self.flag}: got {self.current_value!r}, "
            f"required {self.required_value!r} ({self.message})"
        )


class ScenarioOutcome(AIPerfBaseModel):
    """Result of applying a scenario lock against a resolved ``BenchmarkRun``.

    Produced by ``aiperf.common.scenario.apply_scenario`` and stored on
    ``run.resolved.scenario_outcome``. ``submission_valid`` is ``None`` when no
    scenario is set, ``True`` when all invariants are satisfied (after
    auto-fills), and ``False`` under ``--unsafe-override`` when violations were
    downgraded to warnings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str | None = Field(
        default=None,
        description="The applied scenario name, or None when no --scenario was set.",
    )
    applied_locks: list[str] = Field(
        default_factory=list,
        description="Short tags for each invariant lock that was applied "
        "(auto-filled or validated), e.g. 'timing_mode', 'streaming'. "
        "Order reflects application order.",
    )
    violations: list[ScenarioViolation] = Field(
        default_factory=list,
        description="All scenario invariant conflicts collected in one pass. "
        "Non-empty only under --unsafe-override (otherwise a ScenarioLockError "
        "is raised).",
    )
    submission_valid: bool | None = Field(
        default=None,
        description="True when the scenario lock is satisfied, False under "
        "--unsafe-override with violations, None when no scenario is set.",
    )
    submission_invalid_reasons: list[str] = Field(
        default_factory=list,
        description="Short tags explaining why submission_valid is False "
        "(e.g. 'unsafe_override').",
    )


class ScenarioLockError(ValueError):
    """Raised when a scenario lock is violated and --unsafe-override is not set."""

    def __init__(self, violations: list[ScenarioViolation]) -> None:
        self.violations = violations
        joined = "\n  - ".join(str(v) for v in violations)
        super().__init__(
            f"Scenario invariants violated ({len(violations)} conflict"
            f"{'s' if len(violations) != 1 else ''}):\n  - {joined}\n"
            "Pass --unsafe-override to convert to warnings (run will be marked submission_valid=false)."
        )


class TrajectoryWarmupFailedError(RuntimeError):
    """Raised when WARMUP has terminal failures across trajectories and PROFILING cannot honestly start."""

    def __init__(self, failed_trace_ids: list[str]) -> None:
        self.failed_trace_ids = failed_trace_ids
        super().__init__(
            f"Trajectory warmup failed for {len(failed_trace_ids)} trace(s): "
            f"{', '.join(failed_trace_ids)}. Run aborted to preserve metrics integrity."
        )


class UnknownScenarioError(ValueError):
    """Raised when --scenario references a name not in the registry."""
