# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base import CaseInsensitiveStrEnum


class Modality(CaseInsensitiveStrEnum):
    """Modality of the model. Can be used to determine the type of data to send to the model in
    conjunction with the ModelSelectionStrategy.MODALITY_AWARE."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting the model to use for the request."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    MODALITY_AWARE = "modality_aware"
