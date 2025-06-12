#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import AudioDefaults


class AudioLengthConfig(BaseConfig):
    """
    A configuration class for defining audio length related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=AudioDefaults.LENGTH_MEAN,
            ge=0,
            description="The mean length of the audio in seconds.",
        ),
    ]

    stddev: Annotated[
        float,
        Field(
            default=AudioDefaults.LENGTH_STDDEV,
            ge=0,
            description="The standard deviation of the length of the audio in seconds.",
        ),
    ]
