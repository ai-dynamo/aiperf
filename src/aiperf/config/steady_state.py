# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SteadyStateConfig - configuration for steady-state windowing.

CLI-input shim with no validators. Cross-field validation (start_pct/end_pct
paired, start_pct < end_pct, implicit-enable when both set) lives in the
v1->v2 converter at ``aiperf.config.v1._converter_runtime.validate_steady_state``,
per the v1 hard rule that AIPerfConfig is the single validation gate.
"""

from typing import Annotated

from pydantic import Field

from aiperf.config._base import BaseConfig
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
