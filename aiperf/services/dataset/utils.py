#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import base64
import json
import math
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aiperf.common.enums import ImageFormat


def check_file_exists(filename: Path) -> None:
    """Verifies that the file exists.

    Args:
        filename : The file path to verify.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filename.exists():
        raise FileNotFoundError(f"The file '{filename}' does not exist.")


def open_image(filename: str) -> Image:
    """Opens an image file.

    Args:
        filename : The file path to open.

    Returns:
        The opened PIL Image object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    check_file_exists(Path(filename))
    img = Image.open(filename)

    if img.format is None:
        raise RuntimeError(f"Failed to determine image format of '{filename}'.")

    if img.format.upper() not in list(ImageFormat):
        raise RuntimeError(
            f"'{img.format}' is not one of the supported image formats: "
            f"{', '.join(ImageFormat)}"
        )
    return img


def encode_image(img: Image, format: str) -> str:
    """Encodes an image into base64 encoded string.

    Args:
        img: The PIL Image object to encode.
        format: The image format to use (e.g., "JPEG", "PNG").

    Returns:
        A base64 encoded string representation of the image.
    """
    # JPEG does not support P or RGBA mode (commonly used for PNG) so it needs
    # to be converted to RGB before an image can be saved as JPEG format.
    if format == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")

    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_json_str(json_str: str, func: Callable = lambda x: x) -> dict[str, Any]:
    """Deserializes JSON encoded string into Python object.

    Args:
        json_str: JSON encoded string
        func: A function that takes deserialized JSON object. This can be used to
            run validation checks on the object. Defaults to identity function.

    Returns:
        The deserialized JSON object.

    Raises:
        RuntimeError: If the JSON string is invalid.
    """
    try:
        # TODO: Consider using orjson for faster JSON parsing
        return func(json.loads(json_str))
    except json.JSONDecodeError as e:
        snippet = json_str[:200] + ("..." if len(json_str) > 200 else "")
        raise RuntimeError(f"Failed to parse JSON string: '{snippet}'") from e


def sample_normal(
    mean: float, stddev: float, lower: float = -np.inf, upper: float = np.inf
) -> int:
    """Sample from a normal distribution with support for bounds using rejection sampling.

    Args:
        mean: The mean of the normal distribution.
        stddev: The standard deviation of the normal distribution.
        lower: The lower bound of the distribution.
        upper: The upper bound of the distribution.

    Returns:
        An integer sampled from the distribution.
    """
    while True:
        n = np.random.normal(mean, stddev)
        if lower <= n <= upper:
            return n


def sample_positive_normal(mean: float, stddev: float) -> float:
    """Sample from a normal distribution ensuring positive values
    without distorting the distribution.

    Args:
        mean: Mean value for the normal distribution
        stddev: Standard deviation for the normal distribution

    Returns:
        A positive sample from the normal distribution

    Raises:
        ValueError: If mean is less than 0
    """
    if mean < 0:
        raise ValueError(f"Mean value ({mean}) should be greater than 0")
    return sample_normal(mean, stddev, lower=0)


def sample_positive_normal_integer(mean: float, stddev: float) -> int:
    """Sample a random positive integer from a normal distribution.

    Args:
        mean: The mean of the normal distribution.
        stddev: The standard deviation of the normal distribution.

    Returns:
        A positive integer sampled from the distribution. If the sampled
        number is less than 1, it returns 1.
    """
    return math.ceil(sample_positive_normal(mean, stddev))
