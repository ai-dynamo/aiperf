# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum
from aiperf.common.enums.enums import ConnectionReuseStrategy as ConnectionReuseStrategy
from aiperf.common.enums.enums import ConversationContextMode as ConversationContextMode
from aiperf.common.enums.enums import DatasetType as DatasetType
from aiperf.common.enums.enums import ModelSelectionStrategy as ModelSelectionStrategy
from aiperf.common.enums.enums import PromptSource as PromptSource

CreditPhase = str
"""Type alias for credit phase names. Phases are arbitrary strings (e.g. 'warmup', 'main', 'cooldown')."""


class OslMode(CaseInsensitiveStrEnum):
    """Defines how output sequence length is handled in composed datasets."""

    FILL = "fill"
    """Only apply OSL if the source record lacks it."""

    OVERRIDE = "override"
    """Always use OSL from augmentation config."""


class SweepType(CaseInsensitiveStrEnum):
    """Defines the sweep strategy for parameter exploration."""

    GRID = "grid"
    """All combinations of variable values (Cartesian product)."""

    SCENARIOS = "scenarios"
    """Hand-picked configurations merged with base."""

    SEQUENTIAL = "sequential"
    """Ordered parameter sets applied one at a time."""
