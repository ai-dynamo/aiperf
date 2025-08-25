# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.groups import Groups


class NodeConfig(BaseSettings):
    """Configuration for the Node Controller. This is used to configure the Node Controller."""

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_NODE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    _CLI_GROUP = Groups.NODE_CONTROLLER

    node_id: Annotated[
        str,
        Field(
            ...,
            description="The ID of the node. This is used to identify the node in the system.",
        ),
        CLIParameter(
            name=("--node-id", "-n"),
            group=_CLI_GROUP,
        ),
    ]
