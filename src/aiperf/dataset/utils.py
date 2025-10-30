# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from pathlib import Path

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
    # Use explicit compression settings to ensure deterministic output across platforms
    # (macOS and Linux may have different library versions that produce different output)
    if format == "PNG":
        # PNG: Explicit compress_level and disable optimize to ensure consistent zlib compression
        img.save(buffer, format=format, compress_level=6, optimize=False)
    elif format == "JPEG":
        # JPEG: Explicit quality and subsampling to ensure consistent libjpeg output
        img.save(buffer, format=format, quality=85, subsampling=0)
    else:
        img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
