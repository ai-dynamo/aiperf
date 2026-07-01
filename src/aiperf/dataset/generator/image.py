# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from aiperf.common import random_generator as rng
from aiperf.common.enums import (
    ImageFormat,
    ImageSource,
    ImageSourceSamplingStrategy,
)
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
    - ASSETS: indexes images from the bundled 'assets/source_images' directory
    - NOISE: generates random noise images on the fly
    - PATH: indexes images from the given directory (e.g. `./source_images`)
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
            self._configure_source_image_paths(source_images_dir)
            self._create_source_image = self._create_from_source_images
        elif self.config.source == ImageSource.NOISE:
            self._noise_rng = rng.derive("dataset.image.noise")
            self._create_source_image = self._create_from_noise
        elif isinstance(self.config.source, Path):
            self._source_rng = rng.derive("dataset.image.source")
            self._configure_source_image_paths(self.config.source)
            self._create_source_image = self._create_from_source_images
        else:
            raise ValueError(f"Invalid source: {self.config.source}")

    def _configure_source_image_paths(self, source_path: Path) -> None:
        self._source_images_dir = source_path
        self._source_image_paths = self._load_source_image_paths_from_disk(source_path)
        self._available_source_image_indexes = list(
            range(len(self._source_image_paths))
        )
        self._available_source_image_index_set = set(
            self._available_source_image_indexes
        )
        self._source_image_indexes: list[int] = []
        self._source_image_index = 0

    def _load_source_image_paths_from_disk(self, source_path: Path) -> list[Path]:
        """Index candidate source-image paths from the given directory."""
        if not source_path.exists():
            raise FileNotFoundError(f"The directory '{source_path}' does not exist.")
        if not source_path.is_dir():
            raise NotADirectoryError(f"The path '{source_path}' is not a directory.")

        supported_extensions = {ext.lower() for ext in Image.registered_extensions()}
        image_paths = [
            Path(path)
            for path in sorted(glob.glob(str(source_path / "*")))
            if Path(path).suffix.lower() in supported_extensions
        ]
        if not image_paths:
            raise ValueError(
                f"No source images found in '{source_path}'. "
                "Please ensure the directory contains at least one image file."
            )
        self.debug(lambda: f"Indexed {len(image_paths)} source image paths from disk")
        return image_paths

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
        """Open one sampled source image and resize to target dimensions."""
        while self._available_source_image_indexes:
            index, path = self._next_source_image_path()
            try:
                with Image.open(path) as image:
                    return image.resize(size=(width, height))
            except (UnidentifiedImageError, OSError) as e:
                self._retire_unreadable_source_image(index, path, e)

        raise ValueError(
            f"No readable source images found in '{self._source_images_dir}'. "
            "Please ensure the directory contains at least one readable image file."
        )

    def _next_source_image_path(self) -> tuple[int, Path]:
        if (
            self.config.source_sampling
            == ImageSourceSamplingStrategy.RANDOM_WITH_REPLACEMENT
        ):
            index = self._source_rng.choice(self._available_source_image_indexes)
            return index, self._source_image_paths[index]

        if self.config.source_sampling == ImageSourceSamplingStrategy.SHUFFLE_CYCLE:
            if not self._source_image_indexes:
                self._source_image_indexes = list(self._available_source_image_indexes)
                self._source_rng.shuffle(self._source_image_indexes)
            index = self._source_image_indexes.pop()
            return index, self._source_image_paths[index]

        if self.config.source_sampling == ImageSourceSamplingStrategy.SEQUENTIAL_CYCLE:
            for _ in range(len(self._source_image_paths)):
                index = self._source_image_index
                self._source_image_index = (self._source_image_index + 1) % len(
                    self._source_image_paths
                )
                if index in self._available_source_image_index_set:
                    return index, self._source_image_paths[index]

        raise ValueError(f"Invalid source sampling: {self.config.source_sampling}")

    def _retire_unreadable_source_image(
        self, index: int, path: Path, exc: UnidentifiedImageError | OSError
    ) -> None:
        self.debug(lambda: f"Skipping unreadable image file '{path}': {exc}")
        self._available_source_image_index_set.discard(index)
        self._available_source_image_indexes = [
            i for i in self._available_source_image_indexes if i != index
        ]
        self._source_image_indexes = [
            i for i in self._source_image_indexes if i != index
        ]

    def _create_from_noise(self, width: int, height: int) -> Image.Image:
        """Generate a random noise image at the target dimensions."""
        pixels = self._noise_rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(pixels)
