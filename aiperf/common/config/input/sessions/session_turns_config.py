#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import SessionTurnsDefaults


class SessionTurnsConfig(BaseConfig):
    """
    A configuration class for defining session turns related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=SessionTurnsDefaults.MEAN,
            ge=0,
            description="The mean number of turns in a session",
        ),
    ]
    stddev: Annotated[
        float,
        Field(
            default=SessionTurnsDefaults.STDDEV,
            ge=0,
            description="The standard deviation of the number of turns in a session",
        ),
    ]
