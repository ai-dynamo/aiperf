#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import ImageDefaults
from aiperf.common.enums import ImageFormat


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


class ImageConfig(BaseConfig):
    """
    A configuration class for defining image related settings.
    """

    width: ImageWidthConfig = ImageWidthConfig()
    height: ImageHeightConfig = ImageHeightConfig()
    batch_size: Annotated[
        int,
        Field(
            default=ImageDefaults.BATCH_SIZE,
            ge=0,
            description="The image batch size of the requests GenAI-Perf should send.\
            \nThis is currently supported with the image retrieval endpoint type.",
        ),
    ]
    format: Annotated[
        ImageFormat,
        Field(
            default=ImageDefaults.FORMAT,
            description="The compression format of the images.",
        ),
    ]
