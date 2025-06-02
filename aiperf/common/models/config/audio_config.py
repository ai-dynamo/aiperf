#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Annotated

from pydantic import BeforeValidator, Field

from aiperf.common.enums import AudioFormat
from aiperf.common.models.config.base_config import BaseConfig
from aiperf.common.models.config.config_defaults import AudioDefaults
from aiperf.common.models.config.config_validators import (
    parse_str_or_list_of_positive_values,
)


class AudioConfig(BaseConfig):
    """
    A configuration class for defining input related settings.
    """

    batch_size: Annotated[
        int,
        Field(
            default=AudioDefaults.BATCH_SIZE,
            ge=0,
            description="The batch size of audio requests GenAI-Perf should send.\
            \nThis is currently supported with the OpenAI `multimodal` endpoint type",
        ),
    ]

    # length = ConfigAudioLength()

    format: Annotated[
        AudioFormat,
        Field(
            default=AudioDefaults.FORMAT,
            description="The format of the audio files (wav or mp3).",
        ),
    ]
    depths: Annotated[
        list[int],
        Field(
            default=AudioDefaults.DEPTHS,
            min_length=1,
            description="A list of audio bit depths to randomly select from in bits.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
    ]
    sample_rates: Annotated[
        list[float],
        Field(
            default=AudioDefaults.SAMPLE_RATES,
            min_length=1,
            description="A list of audio sample rates to randomly select from in kHz.",
        ),
        BeforeValidator(parse_str_or_list_of_positive_values),
    ]
    num_channels: Annotated[
        int,
        Field(
            default=AudioDefaults.NUM_CHANNELS,
            ge=1,
            le=2,
            description="The number of audio channels to use for the audio data generation.",
        ),
    ]
