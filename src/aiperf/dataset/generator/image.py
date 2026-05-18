# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from aiperf.common import random_generator as rng
from aiperf.common.enums import ImageFormat, ImageSource
from aiperf.config.dataset.content import ImageConfig
from aiperf.dataset import utils
from aiperf.dataset.generator.base import BaseGenerator


class ImageGenerator(BaseGenerator):
    """A class that generates images from source images.

    This class provides methods to create synthetic images by resizing
    source images to specified dimensions and converting them to a chosen
    image format (e.g., PNG, JPEG). The dimensions can be randomized based
    on mean and standard deviation values.

    Supports three source modes:
    - ASSETS: loads images from the bundled 'assets/source_images' directory
    - NOISE: generates random noise images on the fly
    - PATH: loads images from the given directory (e.g. `./source_images`)
    """

    def __init__(self, config: ImageConfig | None, **kwargs):
        super().__init__(**kwargs)
        self.config = config if config is not None else ImageConfig()

        if not self.config.images_enabled():
            self.debug(lambda: "Images are disabled, skipping image generation")
            return

        # Separate RNGs for independent concerns
        self._dimensions_rng = rng.derive("dataset.image.dimensions")
        self._format_rng = rng.derive("dataset.image.format")

        if self.config.source == ImageSource.ASSETS:
            self._source_rng = rng.derive("dataset.image.source")
            source_images_dir = (
                Path(__file__).parent.resolve() / "assets" / "source_images"
            )
            self._source_images = self._load_source_images_from_disk(source_images_dir)
            self._create_source_image = self._create_from_source_images
        elif self.config.source == ImageSource.NOISE:
            self._noise_rng = rng.derive("dataset.image.noise")
            self._create_source_image = self._create_from_noise
        elif isinstance(self.config.source, Path):
            self._source_rng = rng.derive("dataset.image.source")
            self._source_images = self._load_source_images_from_disk(self.config.source)
            self._create_source_image = self._create_from_source_images
        else:
            raise ValueError(f"Invalid source: {self.config.source}")

    def _load_source_images_from_disk(self, source_path: Path) -> list[Image.Image]:
        """Load source images from the given directory."""
        if not source_path.exists():
            raise FileNotFoundError(f"The directory '{source_path}' does not exist.")
        if not source_path.is_dir():
            raise NotADirectoryError(f"The path '{source_path}' is not a directory.")

        image_paths = sorted(glob.glob(str(source_path / "*")))
        images: list[Image.Image] = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    images.append(img.copy())
            except (UnidentifiedImageError, OSError) as e:
                self.debug(
                    lambda p=path, exc=e: f"Skipping non-image file '{p}': {exc}"
                )
        if not images:
            raise ValueError(
                f"No source images found in '{source_path}'. "
                "Please ensure the directory contains at least one image file."
            )
        self.debug(lambda: f"Pre-loaded {len(images)} source images from disk")
        return images

    def generate(self, *args, **kwargs) -> str:
        """Generate an image with the configured parameters.

        Returns:
            A base64 encoded string of the generated image.
        """
        image_format = self.config.format
        if image_format == ImageFormat.RANDOM:
            formats = [f for f in ImageFormat if f != ImageFormat.RANDOM]
            image_format = self._format_rng.choice(formats)

        width_dist = self.config.width
        height_dist = self.config.height
        width = self._dimensions_rng.sample_positive_normal_integer(
            int(width_dist.expected_value), int(getattr(width_dist, "stddev", 0) or 0)
        )
        height = self._dimensions_rng.sample_positive_normal_integer(
            int(height_dist.expected_value), int(getattr(height_dist, "stddev", 0) or 0)
        )

        image = self._create_source_image(width, height)
        self.debug(
            lambda: f"Generated image from {self.config.source} with width={width}, height={height}"
        )
        base64_image = utils.encode_image(image, image_format)
        return f"data:image/{image_format.name.lower()};base64,{base64_image}"

    def _create_from_source_images(self, width: int, height: int) -> Image.Image:
        """Sample one pre-loaded source image and resize to target dimensions."""
        image = self._source_rng.choice(self._source_images).copy()
        return image.resize(size=(width, height))

    def _create_from_noise(self, width: int, height: int) -> Image.Image:
        """Generate a random noise image at the target dimensions."""
        pixels = self._noise_rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(pixels)
