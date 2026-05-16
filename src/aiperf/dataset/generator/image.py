# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from aiperf.common import random_generator as rng
from aiperf.common.enums import ImageFormat
from aiperf.dataset import utils
from aiperf.dataset.generator.base import BaseGenerator

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class ImageGenerator(BaseGenerator):
    """A class that generates images from source images.

    This class provides methods to create synthetic images by resizing
    source images (located in the 'assets/source_images' directory)
    to specified dimensions and converting them to a chosen image format (e.g., PNG, JPEG).
    The dimensions can be randomized based on mean and standard deviation values.
    """

    def __init__(self, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        # Extract image config from dataset config
        dataset_config = run.cfg.get_default_dataset()
        self.image_config = getattr(dataset_config, "images", None)

        # Separate RNGs for independent concerns
        self._dimensions_rng = rng.derive("dataset.image.dimensions")
        self._format_rng = rng.derive("dataset.image.format")
        self._source_rng = rng.derive("dataset.image.source")

        self.run = run

        # Pre-load source images into memory for fast sampling
        source_images_dir = Path(__file__).parent.resolve() / "assets" / "source_images"
        image_paths = sorted(glob.glob(str(source_images_dir / "*")))
        if not image_paths:
            raise ValueError(
                f"No source images found in '{source_images_dir}'. "
                "Please ensure the source_images directory contains at least one image file."
            )

        self._source_images = []
        for path in image_paths:
            with Image.open(path) as img:
                self._source_images.append(img.copy())
        self.debug(
            lambda: f"Pre-loaded {len(self._source_images)} source images into memory"
        )

    def generate(self, *args, **kwargs) -> str:
        """Generate an image with the configured parameters.

        Returns:
            A base64 encoded string of the generated image.
        """
        if self.image_config is None:
            raise ValueError("Image config is not available in dataset config")

        image_format = self.image_config.format
        if image_format == ImageFormat.RANDOM:
            formats = [f for f in ImageFormat if f != ImageFormat.RANDOM]
            image_format = self._format_rng.choice(formats)

        width = self.image_config.width.sample_int(self._dimensions_rng)
        height = self.image_config.height.sample_int(self._dimensions_rng)

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
        """Sample one image among the pre-loaded source images.

        Returns:
            A PIL Image object randomly selected from the source images.
            Returns a copy to prevent accidental mutation of cached images.
        """
        return self._source_rng.choice(self._source_images).copy()
