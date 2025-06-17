# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CaseInsensitiveStrEnum


class ComposerType(CaseInsensitiveStrEnum):
    """
    The type of composer to use for the dataset.
    """

    SYNTHETIC = "synthetic"
    CUSTOM = "custom"
    PUBLIC_DATASET = "public_dataset"
