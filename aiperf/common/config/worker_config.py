# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from cyclopts import Parameter
from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import WorkersDefaults
from aiperf.common.config.groups import Groups


class WorkersConfig(BaseConfig):
    """Worker configuration."""

    _CLI_GROUP = Groups.WORKERS

    min: Annotated[
        int | None,
        Field(
            description="Minimum number of workers to maintain",
        ),
        Parameter(
            name=("--workers-min", "--min-workers"),
            group=_CLI_GROUP,
        ),
    ] = WorkersDefaults.MIN

    max: Annotated[
        int | None,
        Field(
            description="Maximum number of workers to create. If not specified, the number of"
            " workers will be determined by the smaller of (concurrency + 1) and (num CPUs - 1).",
        ),
        Parameter(
            name=("--workers-max", "--max-workers"),
            group=_CLI_GROUP,
        ),
    ] = WorkersDefaults.MAX

    health_check_interval: Annotated[
        float,
        Field(
            description="Interval in seconds to for workers to publish their health status.",
        ),
        Parameter(
            name=("--workers-health-check-interval"),
            group=_CLI_GROUP,
        ),
    ] = WorkersDefaults.HEALTH_CHECK_INTERVAL
