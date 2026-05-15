# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-run trial mechanics and convergence configuration.

`MultiRunConfig` describes how many trials to run within a single sweep
variation, and the (optional) early-stop criterion for those trials.
Sweep-vs-no-sweep, BO-vs-grid, and inter-variation scheduling all live
on `SweepConfig` (`aiperf.config.sweep`) — not here.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from aiperf.common.enums import ConvergenceStat
from aiperf.config.base import BaseConfig
from aiperf.plugin.enums import ConvergenceCriterionType


class ConvergenceConfig(BaseConfig):
    """Adaptive trial-stopping criterion.

    Presence of this object on `MultiRunConfig.convergence` enables
    adaptive stopping: trials run until the criterion fires (or
    `MultiRunConfig.num_runs` is reached, whichever comes first).
    """

    model_config = ConfigDict(extra="forbid")

    metric: Annotated[
        str,
        Field(
            description="Metric tag to evaluate convergence on (e.g., 'ttft').",
        ),
    ]
    stat: Annotated[
        ConvergenceStat,
        Field(
            default=ConvergenceStat.AVG,
            description="Statistic on the metric to inspect (avg/p50/p90/p95/p99).",
        ),
    ]
    mode: Annotated[
        ConvergenceCriterionType,
        Field(
            default=ConvergenceCriterionType.CI_WIDTH,
            description="Convergence criterion plugin name (ci_width|cv|distribution).",
        ),
    ]
    threshold: Annotated[
        float | None,
        Field(
            gt=0,
            lt=1,
            default=None,
            description="Criterion threshold (interpretation depends on `mode`). "
            "When None (default), each criterion uses its own algorithm-specific "
            "default: ci_width=0.10, cv=0.05, distribution=0.05 (KS p-value).",
        ),
    ]
    min_runs: Annotated[
        int,
        Field(
            ge=2,
            default=2,
            description="Minimum trials before the criterion is checked.",
        ),
    ]


class MultiRunConfig(BaseConfig):
    """Trial mechanics within a single sweep variation.

    When `num_runs > 1` (or `convergence` is set), AIPerf runs multiple
    trials per variation and aggregates statistics. When `num_runs == 1`
    and `convergence is None`, exactly one trial runs per variation.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    num_runs: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            default=1,
            description="Upper bound on trials per variation. When `convergence` is set, "
            "trials may stop earlier; this is the hard ceiling. Cap matches "
            "`BenchmarkPlan.trials` so programmatic plan assembly cannot "
            "exceed what user-facing config validation allows.",
        ),
    ]
    cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            le=86400,
            default=0.0,
            description="Cooldown between trials within a single variation. "
            "Capped at 24h to surface typos like `1e18` at config-load time "
            "rather than blocking the orchestrator inside ``asyncio.sleep``.",
        ),
    ]
    confidence_level: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            default=0.95,
            description="Confidence level for aggregate statistics (0.90, 0.95, 0.99).",
        ),
    ]
    set_consistent_seed: Annotated[
        bool,
        Field(
            default=True,
            description="Auto-set a random seed when none is given for cross-trial consistency.",
        ),
    ]
    vary_seed_per_trial: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "When True, derive a distinct seed for each trial of a variation "
                "via SHA-256 over `(envelope_seed, variation.label, trial)`. "
                "When False (default), all trials of a variation share the "
                "variation's seed. Off gives pure-runtime "
                "variance for confidence statistics; on captures end-to-end "
                "variance including per-trial workload sampling."
            ),
        ),
    ]
    disable_warmup_after_first: Annotated[
        bool,
        Field(
            default=True,
            description="Skip warmup on trials 2..N of a variation (steady-state measurement).",
        ),
    ]
    convergence: Annotated[
        ConvergenceConfig | None,
        Field(
            default=None,
            description="Optional adaptive early-stop. Absent = fixed-N (run `num_runs` trials).",
        ),
    ]

    @model_validator(mode="after")
    def _check_convergence_min_runs_le_num_runs(self) -> Self:
        if self.convergence is not None and self.convergence.min_runs > self.num_runs:
            raise ValueError(
                f"convergence.min_runs ({self.convergence.min_runs}) must be <= "
                f"num_runs ({self.num_runs}). Either lower min_runs or raise num_runs."
            )
        return self
