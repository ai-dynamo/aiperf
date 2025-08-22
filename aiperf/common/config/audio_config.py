# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import AudioDefaults
from aiperf.common.config.config_validators import parse_str_or_list_of_positive_values
from aiperf.common.config.groups import Groups
from aiperf.common.enums import AudioFormat


class AudioLengthConfig(BaseConfig):
    """
    A configuration class for defining audio length related settings.
    """

    _CLI_GROUP = Groups.AUDIO_INPUT

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="The mean length of the audio in seconds.",
        ),
        CLIParameter(
            name=(
                "--audio-length-mean",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.LENGTH_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of the length of the audio in seconds.",
        ),
        CLIParameter(
            name=(
                "--audio-length-stddev",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.LENGTH_STDDEV


class AudioConfig(BaseConfig):
    """
    A configuration class for defining audio related settings.
    """

    _CLI_GROUP = Groups.AUDIO_INPUT

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="The batch size of audio requests AIPerf should send.\n"
            "This is currently supported with the OpenAI `chat` endpoint type",
        ),
        CLIParameter(
            name=(
                "--audio-batch-size",
                "--batch-size-audio",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.BATCH_SIZE

    length: AudioLengthConfig = AudioLengthConfig()

    format: Annotated[
        AudioFormat,
        Field(
            description="The format of the audio files (wav or mp3).",
        ),
        CLIParameter(
            name=(
                "--audio-format",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.FORMAT

    depths: Annotated[
        list[int],
        Field(
            min_length=1,
            description="A list of audio bit depths to randomly select from in bits.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=(
                "--audio-depths",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.DEPTHS

    sample_rates: Annotated[
        list[float],
        Field(
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.\n"
            "Common sample rates are 16, 44.1, 48, 96, etc.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        CLIParameter(
            name=(
                "--audio-sample-rates",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.SAMPLE_RATES

    num_channels: Annotated[
        int,
        Field(
            ge=1,
            le=2,
            description="The number of audio channels to use for the audio data generation.",
        ),
        CLIParameter(
            name=(
                "--audio-num-channels",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = AudioDefaults.NUM_CHANNELS
