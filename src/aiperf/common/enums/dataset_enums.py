# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AssistantResponseMode(CaseInsensitiveStrEnum):
    """Source of assistant responses in Mooncake message conversations."""

    RECORDED = "recorded"
    """Replay each row's complete authored history."""

    LIVE = "live"
    """Accumulate message deltas and responses generated during the run."""


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
