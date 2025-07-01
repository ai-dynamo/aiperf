# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import random
from pathlib import Path

from PIL import Image

from aiperf.common.config import ImageConfig
from aiperf.common.enums import ImageFormat
from aiperf.services.dataset import utils
from aiperf.services.dataset.generator.base import BaseGenerator


class ImageGenerator(BaseGenerator):
    """A class that generates images from source images.

    This class provides methods to create synthetic images by resizing
    source images (located in the 'assets/source_images' directory)
    to specified dimensions and converting them to a chosen image format (e.g., PNG, JPEG).
    The dimensions can be randomized based on mean and standard deviation values.
    """

    def __init__(self, config: ImageConfig):
        super().__init__()
        self.config = config

    def generate(self, *args, **kwargs) -> str:
        """Generate an image with the configured parameters.

        Returns:
            A base64 encoded string of the generated image.
        """
        image_format = self.config.format
        if image_format == ImageFormat.RANDOM:
            image_format = random.choice(
                [f for f in ImageFormat if f != ImageFormat.RANDOM]
            )

        width = utils.sample_positive_normal_integer(
            self.config.width.mean, self.config.width.stddev
        )
        height = utils.sample_positive_normal_integer(
            self.config.height.mean, self.config.height.stddev
        )

        self.logger.debug(
            "Generating image with width=%d, height=%d",
            width,
            height,
        )

        image = self._sample_source_image()
        image = image.resize(size=(width, height))
        base64_image = utils.encode_image(image, image_format)
        return f"data:image/{image_format.name.lower()};base64,{base64_image}"

    def _sample_source_image(self):
        """Sample one image among the source images.

        Returns:
            A PIL Image object randomly selected from the source images.
        """
        filepath = Path(__file__).parent.resolve() / "assets" / "source_images" / "*"
        filenames = glob.glob(str(filepath))
        if not filenames:
            raise ValueError(f"No source images found in '{filepath}'")
        return Image.open(random.choice(filenames))
