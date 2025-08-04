# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CaseInsensitiveStrEnum


class MediaType(CaseInsensitiveStrEnum):
    """The various types of media (e.g. text, image, audio)."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
