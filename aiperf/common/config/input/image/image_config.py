#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import ImageDefaults
from aiperf.common.config.input.image.image_height_config import ImageHeightConfig
from aiperf.common.config.input.image.image_width_config import ImageWidthConfig
from aiperf.common.enums import ImageFormat


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
