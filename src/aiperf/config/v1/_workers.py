# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 WorkersConfig - CLI-only worker-pool input DTO.

Validator-free per the v1 contract. Carries CLI flag annotations and Pydantic
field metadata only; AIPerfConfig (or the v1->v2 converter) owns resolution.
"""

from typing import Annotated

from pydantic import Field

from aiperf.config._base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, DisableCLI, Groups


class WorkersConfig(BaseConfig):
    """Worker configuration."""

    _CLI_GROUP = Groups.WORKERS

    min: Annotated[
        int | None,
        Field(
            description=(
                "Minimum number of worker processes to maintain in the pool. Workers handle request execution and API communication. "
                "Currently not configurable via CLI - reserved for future dynamic scaling features."
            ),
        ),
        DisableCLI(reason="Not currently supported"),
    ] = None

    max: Annotated[
        int | None,
        Field(
            description=(
                "Maximum number of workers to create. If not specified, the number of"
                " workers will be determined by the formula `min(concurrency, (num CPUs * 0.75) - 1)`, "
                " with a default max cap of 32. Any value provided will still be capped by"
                " the concurrency value (if specified), but not by the max cap."
            ),
        ),
        CLIParameter(
            name=("--workers-max", "--max-workers"),
            group=_CLI_GROUP,
        ),
    ] = None
