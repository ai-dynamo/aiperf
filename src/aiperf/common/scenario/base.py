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
        description="Force --use-think-time-only=true to exclude response time from inter-turn delays."
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
    inter_turn_delay_cap_seconds: float = Field(
        description="Hard ceiling for trace inter-turn delays in seconds."
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


class InsufficientTrajectoriesError(RuntimeError):
    """Raised when AGENTIC_REPLAY concurrency exceeds the usable trajectory count.

    Each AGENTIC_REPLAY profiling lane is anchored to a distinct trajectory
    built at startup; when ``concurrency`` exceeds the number of usable
    trajectories (pool size minus traces skipped for being too short to split
    into a warmup + profiling turn), the run cannot honour the requested
    concurrency and is rejected up front instead of silently capping the
    effective load.
    """

    def __init__(
        self, concurrency: int, usable_trajectories: int, pool_size: int
    ) -> None:
        self.concurrency = concurrency
        self.usable_trajectories = usable_trajectories
        self.pool_size = pool_size
        super().__init__(
            f"AGENTIC_REPLAY concurrency {concurrency} exceeds usable trajectory "
            f"count {usable_trajectories} (raw pool size {pool_size}; traces with "
            f"fewer than 2 turns are skipped because warmup+profiling needs at "
            f"least one turn each). Each lane is pinned to a distinct trajectory, "
            f"so the run cannot reach the requested concurrency. Lower "
            f"--concurrency to at most {usable_trajectories}, or use a larger "
            f"trace corpus."
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
