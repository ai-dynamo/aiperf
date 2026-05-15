# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.groups import Groups
from aiperf.common.enums import ImageFormat, ImageSource

_logger = AIPerfLogger(__name__)


def _parse_image_source(value: object) -> object:
    if isinstance(value, ImageSource):
        return value
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str):
        try:
            return ImageSource(value)
        except ValueError:
            return Path(value).expanduser()
    return value


class ImageHeightConfig(BaseConfig):
    """
    A configuration class for defining image height related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Mean height in pixels for synthetically generated images. Image heights follow a normal distribution "
            "around this mean (±`--image-height-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. "
            "Combined with `--image-width-mean` and `--image-source` to determine generated image dimensions and content.",
        ),
        CLIParameter(
            name=(
                "--image-height-mean",  # GenAI-Perf
            ),
            group=Groups.IMAGE_INPUT,
        ),
    ]

    stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Standard deviation for synthetic image heights in pixels. Creates variability in vertical resolution when > 0, "
            "simulating mixed-resolution image inputs. Heights follow normal distribution. "
            "Set to 0 for uniform image heights.",
        ),
        CLIParameter(
            name=(
                "--image-height-stddev",  # GenAI-Perf
            ),
            group=Groups.IMAGE_INPUT,
        ),
    ]


class ImageWidthConfig(BaseConfig):
    """
    A configuration class for defining image width related settings.
    """

    mean: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Mean width in pixels for synthetically generated images. Image widths follow a normal distribution "
            "around this mean (±`--image-width-stddev`). Used when `--image-batch-size` > 0 for multimodal vision benchmarking. "
            "Combined with `--image-height-mean` and `--image-source` to determine generated image dimensions and content.",
        ),
        CLIParameter(
            name=(
                "--image-width-mean",  # GenAI-Perf
            ),
            group=Groups.IMAGE_INPUT,
        ),
    ]

    stddev: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            description="Standard deviation for synthetic image widths in pixels. Creates variability in horizontal resolution when > 0, "
            "simulating mixed-resolution image inputs. Widths follow normal distribution. "
            "Set to 0 for uniform image widths.",
        ),
        CLIParameter(
            name=(
                "--image-width-stddev",  # GenAI-Perf
            ),
            group=Groups.IMAGE_INPUT,
        ),
    ]


class ImageConfig(BaseConfig):
    """
    A configuration class for defining image related settings.
    """

    width: Annotated[
        ImageWidthConfig,
        Field(
            default_factory=ImageWidthConfig,
            description="Width distribution in pixels for synthetic images (mean and stddev).",
        ),
    ]
    height: Annotated[
        ImageHeightConfig,
        Field(
            default_factory=ImageHeightConfig,
            description="Height distribution in pixels for synthetic images (mean and stddev).",
        ),
    ]
    batch_size: Annotated[
        int,
        Field(
            default=1,
            ge=0,
            description="Number of images to include in each multimodal request. Supported with `chat` endpoint type for vision-language models. "
            "Each image is generated according to `--image-source` (random noise by default, or sampled/resized from a directory of source images). "
            "Set to 0 to disable image inputs. Higher batch sizes test multi-image understanding and increase request payload size.",
        ),
        CLIParameter(
            name=(
                "--image-batch-size",
                "--batch-size-image",  # GenAI-Perf
            ),
            group=Groups.IMAGE_INPUT,
        ),
    ]

    format: Annotated[
        ImageFormat,
        Field(
            default=ImageFormat.PNG,
            description="Image file format for generated images. Choose `png` for lossless compression (larger files, best quality), "
            "`jpeg` for lossy compression (smaller files, good quality), or `random` to randomly select between PNG and JPEG for each image. "
            "Format affects file size in multimodal requests and encoding overhead.",
        ),
        CLIParameter(
            name=(
                "--image-format",  # GenAI-Perf
            ),
            group=Groups.IMAGE_INPUT,
        ),
    ]

    source: Annotated[
        ImageSource | Path,
        BeforeValidator(_parse_image_source),
        Field(
            default=ImageSource.NOISE,
            description="Source image generation mode (default `noise`). "
            "`noise` generates random noise images on the fly at the requested dimensions — no files on disk required, "
            "and the pool is effectively unbounded so servers cannot dedupe on identical inputs. "
            "`assets` loads images from the built-in `assets/source_images` directory (ships with a bundled set of source images) "
            "and resizes them to the requested dimensions. "
            "A path to a directory loads images from the given directory (e.g. `--image-source ./source_images`). "
            "Note: random-noise images are roughly incompressible, so payload bytes are larger than equivalent natural images.",
        ),
        CLIParameter(
            name=("--image-source",),
            group=Groups.IMAGE_INPUT,
        ),
    ]

    @model_validator(mode="after")
    def _validate_image_options(self) -> Self:
        """Validate the image options.

        Flag configs where the user supplied non-default width/height parameters but did
        not enable images. `batch_size=0` is treated as an explicit disable and always
        allowed, and default-valued fields (e.g., from a round-tripped config file) do
        not trip the check.
        """
        if self.images_enabled() or self.batch_size == 0:
            return self
        if (
            "batch_size" in self.model_fields_set
            and self.batch_size != type(self).model_fields["batch_size"].default
        ):
            _logger.warning(
                "--image-batch-size was set, but image generation is disabled because "
                "--image-width-mean and --image-height-mean are not both positive."
            )
        if (
            self.width.mean != 0.0
            or self.width.stddev != 0.0
            or self.height.mean != 0.0
            or self.height.stddev != 0.0
        ):
            raise ValueError(
                "Image generation is disabled but image dimension options were provided. Please set `--image-batch-size`, `--image-width-mean`, and `--image-height-mean` to enable image generation."
            )
        return self

    def images_enabled(self) -> bool:
        """Check if images are enabled."""
        return self.width.mean > 0 and self.height.mean > 0 and self.batch_size > 0
