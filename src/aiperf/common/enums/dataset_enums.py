# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum
from aiperf.common.enums.enums import ConnectionReuseStrategy as ConnectionReuseStrategy
from aiperf.common.enums.enums import ConversationContextMode as ConversationContextMode
from aiperf.common.enums.enums import DatasetType as DatasetType
from aiperf.common.enums.enums import ModelSelectionStrategy as ModelSelectionStrategy

CreditPhase = str
"""Type alias for credit phase names. Phases are arbitrary strings (e.g. 'warmup', 'main', 'cooldown')."""


class SweepType(CaseInsensitiveStrEnum):
    """Defines the sweep strategy for parameter exploration."""

    GRID = "grid"
    """All combinations of variable values (Cartesian product)."""

    ZIP = "zip"
    """Element-wise pairing of variable values."""

    SCENARIOS = "scenarios"
    """Hand-picked configurations merged with base."""

    ADAPTIVE_SEARCH = "adaptive_search"
    """Planner-driven adaptive outer-loop search."""

    SOBOL = "sobol"
    """Sobol quasi-random sampling over dimensions."""

    LATIN_HYPERCUBE = "latin_hypercube"
    """Latin hypercube sampling over dimensions."""
