# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from cyclopts import Parameter
from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import UIDefaults
from aiperf.common.config.groups import Groups
from aiperf.common.enums import AIPerfUIType


class UIConfig(BaseConfig):
    """Configuration for the UI."""

    _CLI_GROUP = Groups.UI

    type: Annotated[
        AIPerfUIType,
        Field(
            description="Type of UI to use",
        ),
        Parameter(
            name=("--ui-type", "--ui"),
            group=_CLI_GROUP,
        ),
    ] = UIDefaults.UI_TYPE

    min_update_percent: Annotated[
        float,
        Field(
            description="Minimum percentage difference from the last update to trigger a UI update (for non-dashboard UIs).",
        ),
        Parameter(
            name=("--ui-min-update-percent"),
            group=_CLI_GROUP,
        ),
    ] = UIDefaults.MIN_UPDATE_PERCENT
