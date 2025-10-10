# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import (
    ConversationDefaults,
    TurnDefaults,
    TurnDelayDefaults,
)
from aiperf.common.config.groups import Groups


class TurnDelayConfig(BaseConfig):
    """
    A configuration class for defining turn delay related settings.
    """

    _CLI_GROUP = Groups.CONVERSATION_INPUT

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="The mean delay between turns within a conversation in milliseconds.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-mean",
                "--session-turn-delay-mean",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = TurnDelayDefaults.MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of the delay between turns \n"
            "within a conversation in milliseconds.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-stddev",
                "--session-turn-delay-stddev",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = TurnDelayDefaults.STDDEV

    ratio: Annotated[
        float,
        Field(
            ge=0,
            description="A ratio to scale multi-turn delays.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-delay-ratio",
                "--session-delay-ratio",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = TurnDelayDefaults.RATIO


class TurnConfig(BaseConfig):
    """
    A configuration class for defining turn related settings in a conversation.
    """

    _CLI_GROUP = Groups.CONVERSATION_INPUT

    mean: Annotated[
        int,
        Field(
            ge=1,
            description="The mean number of turns within a conversation.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-mean",
                "--session-turns-mean",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = TurnDefaults.MEAN

    stddev: Annotated[
        int,
        Field(
            ge=0,
            description="The standard deviation of the number of turns within a conversation.",
        ),
        CLIParameter(
            name=(
                "--conversation-turn-stddev",
                "--session-turns-stddev",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = TurnDefaults.STDDEV

    delay: TurnDelayConfig = TurnDelayConfig()


class ConversationConfig(BaseConfig):
    """
    A configuration class for defining conversations related settings.
    """

    _CLI_GROUP = Groups.CONVERSATION_INPUT

    num: Annotated[
        int,
        Field(
            ge=1,
            description="The total number of unique conversations to generate.\n"
            "Each conversation represents a single request session between client and server.\n"
            "Supported on synthetic mode and the custom random_pool dataset. The number of conversations \n"
            "will be used to determine the number of entries in both the custom random_pool and synthetic \n"
            "datasets and will be reused until benchmarking is complete.",
        ),
        CLIParameter(
            name=(
                "--conversation-num",
                "--num-conversations",
                "--num-sessions",  # GenAI-Perf
                "--num-dataset-entries",  # GenAI-Perf,
                "--num-prompts",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = ConversationDefaults.NUM

    turn: TurnConfig = TurnConfig()
