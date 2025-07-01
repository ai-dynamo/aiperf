# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import AudioDefaults
from aiperf.common.config.config_validators import parse_str_or_list_of_positive_values
from aiperf.common.enums import AudioFormat


class AudioLengthConfig(BaseConfig):
    """
    A configuration class for defining audio length related settings.
    """

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="The mean length of the audio in seconds.",
        ),
        cyclopts.Parameter(
            name=("--audio-length-mean"),
        ),
    ] = AudioDefaults.LENGTH_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of the length of the audio in seconds.",
        ),
        cyclopts.Parameter(
            name=("--audio-length-stddev"),
        ),
    ] = AudioDefaults.LENGTH_STDDEV


class AudioConfig(BaseConfig):
    """
    A configuration class for defining audio related settings.
    """

    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="The batch size of audio requests GenAI-Perf should send.\
            \nThis is currently supported with the OpenAI `multimodal` endpoint type",
        ),
        cyclopts.Parameter(
            name=("--audio-batch-size"),
        ),
    ] = AudioDefaults.BATCH_SIZE

    length: AudioLengthConfig = AudioLengthConfig()

    format: Annotated[
        AudioFormat,
        Field(
            description="The format of the audio files (wav or mp3).",
        ),
        cyclopts.Parameter(
            name=("--audio-format"),
        ),
    ] = AudioDefaults.FORMAT

    depths: Annotated[
        list[int],
        Field(
            min_length=1,
            description="A list of audio bit depths to randomly select from in bits.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        cyclopts.Parameter(
            name=("--audio-depths"),
        ),
    ] = AudioDefaults.DEPTHS

    sample_rates: Annotated[
        list[float],
        Field(
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.\
            \nCommon sample rates are 16, 44.1, 48, 96, etc.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
        cyclopts.Parameter(
            name=("--audio-sample-rates"),
        ),
    ] = AudioDefaults.SAMPLE_RATES

    num_channels: Annotated[
        int,
        Field(
            ge=1,
            le=2,
            description="The number of audio channels to use for the audio data generation.",
        ),
        cyclopts.Parameter(
            name=("--audio-num-channels"),
        ),
    ] = AudioDefaults.NUM_CHANNELS
