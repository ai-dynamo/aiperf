# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SteadyStateConfig - configuration for steady-state windowing.

Field-level bounds live on the Fields; the cross-field manual-window
contract (start_pct/end_pct set together, start_pct < end_pct) is enforced
by the ``model_validator`` below, so a half-set or inverted window fails at
config validation rather than silently degrading to automatic detection.
"""

from typing import Annotated, Self

from pydantic import Field, model_validator

from aiperf.config.base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, Groups


class SteadyStateConfig(BaseConfig):
    """Configuration for steady-state detection and windowed metric computation.

    When enabled, AIPerf detects the steady-state region of a benchmark run by
    analyzing concurrency curves, then re-computes metrics only over that window.
    This excludes ramp-up and ramp-down artifacts from the results.
    """

    _CLI_GROUP = Groups.OUTPUT

    enabled: Annotated[
        bool,
        Field(
            description="Enable steady-state metric computation. When enabled, AIPerf detects the steady-state "
            "region of a benchmark run and reports windowed metrics that exclude ramp-up and ramp-down periods.",
        ),
        CLIParameter(
            name="--steady-state",
            group=_CLI_GROUP,
        ),
    ] = False

    start_pct: Annotated[
        float | None,
        Field(
            ge=0.0,
            lt=100.0,
            description="Manual override: start of steady-state window as a percentage of total benchmark duration. "
            "Must be used together with --steady-state-end-pct. Overrides automatic detection.",
        ),
        CLIParameter(
            name="--steady-state-start-pct",
            group=_CLI_GROUP,
        ),
    ] = None

    end_pct: Annotated[
        float | None,
        Field(
            gt=0.0,
            le=100.0,
            description="Manual override: end of steady-state window as a percentage of total benchmark duration. "
            "Must be used together with --steady-state-start-pct. Overrides automatic detection.",
        ),
        CLIParameter(
            name="--steady-state-end-pct",
            group=_CLI_GROUP,
        ),
    ] = None

    min_window_pct: Annotated[
        float,
        Field(
            gt=0.0,
            le=100.0,
            description="Minimum steady-state window size as a percentage of total benchmark duration. "
            "If the detected window is smaller than this, AIPerf falls back to the full duration.",
        ),
        CLIParameter(
            name="--steady-state-min-window-pct",
            group=_CLI_GROUP,
        ),
    ] = 10.0

    bootstrap_iterations: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Number of bootstrap iterations for confidence intervals on boundaries. "
            "Set to 50+ to enable. Increases summarize time proportionally.",
        ),
        CLIParameter(
            name="--steady-state-bootstrap-iterations",
            group=_CLI_GROUP,
        ),
    ] = None

    @model_validator(mode="after")
    def _validate_manual_window(self) -> Self:
        """Enforce the manual-window override contract.

        ``start_pct``/``end_pct`` must be set together (a half-set override
        would silently fall back to automatic detection), and the window must
        be non-empty (``start_pct < end_pct``).
        """
        if (self.start_pct is None) != (self.end_pct is None):
            set_flag, unset_flag = (
                ("--steady-state-start-pct", "--steady-state-end-pct")
                if self.start_pct is not None
                else ("--steady-state-end-pct", "--steady-state-start-pct")
            )
            raise ValueError(
                f"steady-state manual window: {set_flag} was set without "
                f"{unset_flag}; the manual override requires both bounds. "
                f"Set both, or neither to use automatic detection."
            )
        if (
            self.start_pct is not None
            and self.end_pct is not None
            and self.start_pct >= self.end_pct
        ):
            raise ValueError(
                f"steady-state manual window: start_pct ({self.start_pct}) must "
                f"be < end_pct ({self.end_pct}); an empty or inverted window "
                f"selects no samples."
            )
        return self
