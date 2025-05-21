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

    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
