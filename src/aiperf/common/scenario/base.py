# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from aiperf.common.enums import CacheBustTarget
from aiperf.common.models import AIPerfBaseModel
from aiperf.plugin.enums import TimingMode


class ScenarioSpec(AIPerfBaseModel):
    """Frozen declaration of a benchmark scenario's invariants."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str = Field(description="Scenario identifier, e.g. 'inferencex-agentx-mvp'.")
    timing_mode: TimingMode = Field(
        description="Required timing mode for this scenario."
    )
    require_ignore_eos: bool = Field(
        description="Inject ignore_eos=true into extra_inputs; error on explicit false."
    )
    require_use_think_time_only: bool = Field(
        default=False,
        description="Force --use-think-time-only=true to exclude response time from inter-turn delays.",
    )
    require_use_end_to_start_delays: bool = Field(
        default=False,
        description="Force --use-end-to-start-delays=true so inter-turn delays are the "
        "end-to-start idle gap (t_curr - (t_prev + api_time_prev)), not the "
        "start-to-start delta. Prevents per-stream clock drift that fabricates "
        "cross-stream concurrency on completion-gated replay.",
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
    forbid_ignore_trace_delays: bool = Field(
        default=False,
        description=(
            "Reject --ignore-trace-delays. The scenario replays recorded trace "
            "timing; --ignore-trace-delays nulls every per-turn timestamp/delay "
            "in the loader, dispatching all turns back-to-back and falsifying the "
            "workload while the run would otherwise still report "
            "submission_valid=true."
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
        description="Floor on --benchmark-duration in seconds."
    )
    default_benchmark_duration_seconds: int | None = Field(
        default=None,
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
            "Value auto-filled into --trajectory-start-min-ratio when the user "
            "leaves it unset. Explicit user values are honored. None disables "
            "auto-fill."
        ),
    )
    default_trajectory_start_max_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Value auto-filled into --trajectory-start-max-ratio when the user "
            "leaves it unset. Explicit user values are honored. None disables "
            "auto-fill."
        ),
    )
    inter_turn_delay_cap_seconds: float | None = Field(
        default=None,
        description="Hard ceiling for trace inter-turn delays in seconds. None disables.",
    )
    trace_idle_gap_cap_seconds: float | None = Field(
        default=None,
        description=(
            "Hard ceiling (seconds) for idle gaps within each root trace. For "
            "Weka, parent + subagent request-start timestamps are compressed "
            "per-trace before per-turn delays are derived. Takes precedence over "
            "inter_turn_delay_cap_seconds and supersedes use_think_time_only."
        ),
    )
    require_cache_bust: CacheBustTarget | None = Field(
        default=None,
        description=(
            "When set, prompt.cache_bust.target must equal this value. "
            "Mismatch is rejected unless --unsafe-override is also set "
            "(which stamps submission_valid=false)."
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


class EmptyTracePoolError(RuntimeError):
    """Raised when the loader produces 0 valid traces and the scenario requires a non-empty pool."""


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
