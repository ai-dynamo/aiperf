# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import (
    ConversationDefaults,
    PromptDefaults,
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
            description="Mean delay in milliseconds between consecutive turns within a multi-turn conversation. Simulates user think time between "
            "receiving a response and sending the next message. Delays follow normal distribution around this mean (±`--conversation-turn-delay-stddev`). "
            "Only applies to multi-turn conversations (`--conversation-turn-mean` > 1). Set to 0 for back-to-back turns.",
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
            description="Standard deviation for turn delays in milliseconds. Creates variability in user think time between conversation turns. "
            "Delays follow normal distribution. Set to 0 for deterministic delays. "
            "Models realistic human interaction patterns with variable response times.",
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
            description="Multiplier for scaling all turn delays within conversations. Applied after mean/stddev calculation: "
            "`actual_delay = calculated_delay × ratio`. Use to proportionally adjust timing without changing distribution shape. "
            "Values < 1 speed up conversations, > 1 slow them down. Set to 0 to eliminate delays entirely.",
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
            description="Mean number of request-response turns per conversation. Each turn consists of a user message and model response. "
            "Turn counts follow normal distribution around this mean (±`--conversation-turn-stddev`). Set to 1 for single-turn interactions. "
            "Multi-turn conversations enable testing of context retention and conversation history handling.",
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
            description="Standard deviation for number of turns per conversation. Creates variability in conversation lengths, simulating "
            "diverse interaction patterns (quick questions vs. extended dialogues). Turn counts follow normal distribution. "
            "Set to 0 for uniform conversation lengths.",
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
        int | None,
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
            ),
            group=_CLI_GROUP,
        ),
    ] = ConversationDefaults.NUM

    num_dataset_entries: Annotated[
        int,
        Field(
            ge=1,
            description="Total number of unique entries to generate for the dataset. Each entry represents one user message that can be "
            "used as a turn in conversations. Entries are reused across conversations and turns according to `--dataset-sampling-strategy`. "
            "Higher values provide more diversity.",
        ),
        CLIParameter(
            name=(
                "--num-dataset-entries",  # GenAI-Perf
                "--num-prompts",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = PromptDefaults.NUM

    turn: TurnConfig = TurnConfig()
