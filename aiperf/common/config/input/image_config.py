# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
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
            ge=0,
            description="The mean height of images when generating synthetic image data.",
        ),
        cyclopts.Parameter(
            name=(
                "--image-height-mean",  # GenAI-Perf
            ),
        ),
    ] = ImageDefaults.HEIGHT_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of height of images when generating synthetic image data.",
        ),
        cyclopts.Parameter(
            name=(
                "--image-height-stddev",  # GenAI-Perf
            ),
        ),
    ] = ImageDefaults.HEIGHT_STDDEV


class ImageWidthConfig(BaseConfig):
    """
    A configuration class for defining image width related settings.
    """

    mean: Annotated[
        float,
        Field(
            ge=0,
            description="The mean width of images when generating synthetic image data.",
        ),
        cyclopts.Parameter(
            name=(
                "--image-width-mean",  # GenAI-Perf
            ),
        ),
    ] = ImageDefaults.WIDTH_MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of width of images when generating synthetic image data.",
        ),
        cyclopts.Parameter(
            name=(
                "--image-width-stddev",  # GenAI-Perf
            ),
        ),
    ] = ImageDefaults.WIDTH_STDDEV


class ImageConfig(BaseConfig):
    """
    A configuration class for defining image related settings.
    """

    width: ImageWidthConfig = ImageWidthConfig()
    height: ImageHeightConfig = ImageHeightConfig()
    batch_size: Annotated[
        int,
        Field(
            ge=0,
            description="The image batch size of the requests AIPerf should send.\n"
            "This is currently supported with the image retrieval endpoint type.",
        ),
        cyclopts.Parameter(
            name=(
                "--image-batch-size",
                "--batch-size-image",  # GenAI-Perf
            ),
        ),
    ] = ImageDefaults.BATCH_SIZE

    format: Annotated[
        ImageFormat,
        Field(
            description="The compression format of the images.",
        ),
        cyclopts.Parameter(
            name=(
                "--image-format",  # GenAI-Perf
            ),
        ),
    ] = ImageDefaults.FORMAT
