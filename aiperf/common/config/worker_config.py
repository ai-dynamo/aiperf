#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from typing import Annotated

import cyclopts
from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import WorkerDefaults


class WorkersConfig(BaseConfig):
    """Worker configuration."""

    _GROUP_NAME = "Workers"

    min: Annotated[
        int | None,
        Field(
            description="Minimum number of workers to maintain",
        ),
        cyclopts.Parameter(
            name=("--workers-min", "--min-workers"),
            group=_GROUP_NAME,
        ),
    ] = WorkerDefaults.MIN

    max: Annotated[
        int | None,
        Field(
            description="Maximum number of workers to create. If not specified, the number of"
            " workers will be determined by the smaller of (concurrency + 1) and (num CPUs - 1).",
        ),
        cyclopts.Parameter(
            name=("--workers-max", "--max-workers"),
            group=_GROUP_NAME,
        ),
    ] = WorkerDefaults.MAX

    health_check_interval_seconds: Annotated[
        float,
        Field(
            description="Interval in seconds to for workers to publish their health status.",
        ),
        cyclopts.Parameter(
            name=("--workers-health-check-interval-seconds"),
            group=_GROUP_NAME,
        ),
    ] = WorkerDefaults.HEALTH_CHECK_INTERVAL_SECONDS
