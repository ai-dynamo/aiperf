#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

from PIL import Image


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
