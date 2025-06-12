#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import ImageDefaults


class ImageWidthConfig(BaseConfig):
    """
    A configuration class for defining image width related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=ImageDefaults.WIDTH_MEAN,
            ge=0,
            description="The mean width of images when generating synthetic image data.",
        ),
    ]
    stddev: Annotated[
        float,
        Field(
            default=ImageDefaults.WIDTH_STDDEV,
            ge=0,
            description="The standard deviation of width of images when generating synthetic image data.",
        ),
    ]
