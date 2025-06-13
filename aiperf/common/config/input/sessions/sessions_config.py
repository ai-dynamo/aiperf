#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import (
    SessionsDefaults,
    SessionTurnDelayDefaults,
    SessionTurnsDefaults,
)


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


class SessionTurnDelayConfig(BaseConfig):
    """
    A configuration class for defining session turn delay related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=SessionTurnDelayDefaults.MEAN,
            ge=0,
            description="The mean delay (in ms) between turns in a session",
        ),
    ]
    stddev: Annotated[
        float,
        Field(
            default=SessionTurnDelayDefaults.STDDEV,
            ge=0,
            description="The standard deviation (in ms) of the delay between turns in a session",
        ),
    ]
    ratio: Annotated[
        float,
        Field(
            default=SessionTurnDelayDefaults.RATIO,
            ge=0,
            description="A ratio to scale multi-turn delays when using a payload file",
        ),
    ]


class SessionsConfig(BaseConfig):
    """
    A configuration class for defining sessions related settings.
    """

    num: Annotated[
        int,
        Field(
            default=SessionsDefaults.NUM,
            ge=0,
            description="The number of sessions to simulate",
        ),
    ]
    turns: SessionTurnsConfig = SessionTurnsConfig()
    turn_delay: SessionTurnDelayConfig = SessionTurnDelayConfig()
