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

import glob
import random
from pathlib import Path

from PIL import Image

from aiperf.common.enums import ImageFormat
from aiperf.services.dataset.generator import utils


class ImageGenerator:
    """A class that generates images from source images.

    This class provides methods to create synthetic images by resizing
    source images (located in the 'assets/source_images' directory)
    to specified dimensions and converting them to a chosen image format (e.g., PNG, JPEG).
    The dimensions can be randomized based on mean and standard deviation values.
    """

    @classmethod
    def create_synthetic_image(
        cls,
        image_width_mean: int,
        image_width_stddev: int,
        image_height_mean: int,
        image_height_stddev: int,
        image_format: ImageFormat | None = None,
    ) -> str:
        """Generate an image with the provided parameters.

        Args:
            image_width_mean: The mean width of the image.
            image_width_stddev: The standard deviation of the image width.
            image_height_mean: The mean height of the image.
            image_height_stddev: The standard deviation of the image height.
            image_format: The format of the image.

        Returns:
            A base64 encoded string of the generated image.
        """
        if image_format is None:
            image_format = random.choice(list(ImageFormat))
        width = cls._sample_random_positive_integer(
            image_width_mean, image_width_stddev
        )
        height = cls._sample_random_positive_integer(
            image_height_mean, image_height_stddev
        )

        image = cls._sample_source_image()
        image = image.resize(size=(width, height))
        base64_image = utils.encode_image(image, image_format.name)

        return f"data:image/{image_format.name.lower()};base64,{base64_image}"

    @classmethod
    def _sample_source_image(cls):
        """Sample one image among the source images.

        Returns:
            A PIL Image object randomly selected from the source images.
        """
        filepath = Path(__file__).parent.resolve() / "assets" / "source_images" / "*"
        filenames = glob.glob(str(filepath))
        if not filenames:
            raise ValueError(f"No source images found in '{filepath}'")
        return Image.open(random.choice(filenames))

    @classmethod
    def _sample_random_positive_integer(cls, mean: int, stddev: int) -> int:
        """Sample a random positive integer from a Gaussian distribution.

        Args:
            mean: The mean of the Gaussian distribution.
            stddev: The standard deviation of the Gaussian distribution.

        Returns:
            A positive integer sampled from the distribution. If the sampled
            number is zero, it returns 1.
        """
        n = int(abs(random.gauss(mean, stddev)))
        return n if n != 0 else 1  # avoid zero
