# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
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
        cyclopts.Parameter(
            name=(
                "--conversation-turn-delay-mean",
                "--session-turn-delay-mean",  # GenAI-Perf
            ),
        ),
    ] = TurnDelayDefaults.MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of the delay between turns \n"
            "within a conversation in milliseconds.",
        ),
        cyclopts.Parameter(
            name=(
                "--conversation-turn-delay-stddev",
                "--session-turn-delay-stddev",  # GenAI-Perf
            ),
        ),
    ] = TurnDelayDefaults.STDDEV

    ratio: Annotated[
        float,
        Field(
            ge=0,
            description="A ratio to scale multi-turn delays.",
        ),
        cyclopts.Parameter(
            name=(
                "--conversation-turn-delay-ratio",
                "--session-delay-ratio",  # GenAI-Perf
            ),
        ),
    ] = TurnDelayDefaults.RATIO


class TurnConfig(BaseConfig):
    """
    A configuration class for defining turn related settings in a conversation.
    """

    mean: Annotated[
        int,
        Field(
            ge=1,
            description="The mean number of turns within a conversation.",
        ),
        cyclopts.Parameter(
            name=(
                "--conversation-turn-mean",
                "--session-turns-mean",  # GenAI-Perf
            ),
        ),
    ] = TurnDefaults.MEAN

    stddev: Annotated[
        int,
        Field(
            ge=0,
            description="The standard deviation of the number of turns within a conversation.",
        ),
        cyclopts.Parameter(
            name=(
                "--conversation-turn-stddev",
                "--session-turns-stddev",  # GenAI-Perf
            ),
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
            description="The total number of unique conversations to generate.\n"
            "Each conversation represents a single request session between client and server.\n"
            "Supported on synthetic mode only and conversations will be reused until benchmarking is complete.",
        ),
        cyclopts.Parameter(
            name=(
                "--conversation-num",
                "--num-conversations",
                "--num-sessions",  # GenAI-Perf
            ),
        ),
    ] = ConversationDefaults.NUM

    turn: TurnConfig = TurnConfig()
