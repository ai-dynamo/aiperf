# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import SystemDefaults
from aiperf.common.config.groups import Groups


class SystemControllerConfig(BaseSettings):
    """Configuration for the System Controller. This is used to configure the System Controller."""

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_SYSTEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    _CLI_GROUP = Groups.SYSTEM_CONTROLLER

    node_controllers: Annotated[
        int,
        Field(
            ge=0,
            description="The number of node controllers to expect in the system. If not specified, the system will not expect any node controllers.",
        ),
        CLIParameter(
            name=("--node-controllers", "--nodes"),
            group=_CLI_GROUP,
        ),
    ] = SystemDefaults.NODE_CONTROLLERS
