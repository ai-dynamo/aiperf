#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import ImageDefaults


class ImageHeightConfig(BaseConfig):
    """
    A configuration class for defining image height related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=ImageDefaults.HEIGHT_MEAN,
            ge=0,
            description="The mean height of images when generating synthetic image data.",
        ),
    ]
    stddev: Annotated[
        float,
        Field(
            default=ImageDefaults.HEIGHT_STDDEV,
            ge=0,
            description="The standard deviation of height of images when generating synthetic image data.",
        ),
    ]
