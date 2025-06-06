#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import AudioDefaults
from aiperf.common.config.config_validators import (
    parse_str_or_list_of_positive_values,
)
from aiperf.common.enums import AudioFormat


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
    ] = AudioDefaults.BATCH_SIZE

    # length = ConfigAudioLength()

    format: Annotated[
        AudioFormat,
        Field(
            description="The format of the audio files (wav or mp3).",
        ),
    ] = AudioDefaults.FORMAT

    depths: Annotated[
        list[int],
        Field(
            min_length=1,
            description="A list of audio bit depths to randomly select from in bits.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
    ] = AudioDefaults.DEPTHS

    sample_rates: Annotated[
        list[float],
        Field(
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
    ] = AudioDefaults.SAMPLE_RATES

    num_channels: Annotated[
        int,
        Field(
            ge=1,
            le=2,
            description="The number of audio channels to use for the audio data generation.",
        ),
    ] = AudioDefaults.NUM_CHANNELS
