# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
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
        DisableCLI(reason="Not currently supported"),
    ] = WorkersDefaults.MIN

    max: Annotated[
        int | None,
        Field(
            description="Maximum number of workers to create. If not specified, the number of"
            " workers will be determined by the smaller of (concurrency + 1) and (num CPUs - 1).",
        ),
        CLIParameter(
            name=("--workers-max", "--max-workers"),
            group=_CLI_GROUP,
        ),
    ] = WorkersDefaults.MAX
