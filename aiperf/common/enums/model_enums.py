# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting the model to use for the request."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
