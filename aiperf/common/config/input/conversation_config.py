#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import (
    ConversationDefaults,
    TurnDefaults,
    TurnDelayDefaults,
)


class TurnDelayConfig(BaseConfig):
    """
    A configuration class for defining turn delay related settings.
    """

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="The mean delay between turns within a conversation in milliseconds.",
        ),
    ] = TurnDelayDefaults.MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of the delay between turns \
            \nwithin a conversation in milliseconds.",
        ),
    ] = TurnDelayDefaults.STDDEV

    ratio: Annotated[
        float,
        Field(
            ge=0,
            description="A ratio to scale multi-turn delays.",
        ),
    ] = TurnDelayDefaults.RATIO


class TurnConfig(BaseConfig):
    """
    A configuration class for defining turn related settings in a conversation.
    """

    mean: Annotated[
        int,
        Field(
            ge=0,
            description="The mean number of turns within a conversation.",
        ),
    ] = TurnDefaults.MEAN

    stddev: Annotated[
        int,
        Field(
            ge=0,
            description="The standard deviation of the number of turns within a conversation.",
        ),
    ] = TurnDefaults.STDDEV

    delay: TurnDelayConfig = TurnDelayConfig()


class ConversationConfig(BaseConfig):
    """
    A configuration class for defining conversations related settings.
    """

    num: Annotated[
        int,
        Field(
            ge=1,
            description="The total number of independent conversations.",
        ),
    ] = ConversationDefaults.NUM

    turn: TurnConfig = TurnConfig()
