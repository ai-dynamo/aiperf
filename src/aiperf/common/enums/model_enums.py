# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting the model to use for the request."""

    ROUND_ROBIN = "round_robin"
    """Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models)."""

    RANDOM = "random"
    """Randomly select a model for each prompt using uniform distribution."""
