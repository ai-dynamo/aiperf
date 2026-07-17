# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared observation store for multi-tier SLO boundary search."""

from __future__ import annotations

from collections import defaultdict

from aiperf.orchestrator.models import RunResult


class SharedObservationStore:
    """Stores observations indexed by concurrency, shared across all tiers."""

    def __init__(self) -> None:
        self._data: dict[int, list[list[RunResult]]] = defaultdict(list)

    def store(self, concurrency: int, results: list[RunResult]) -> None:
        """Append a probe's results at the given concurrency level."""
        self._data[concurrency].append(results)

    def get(self, concurrency: int) -> list[list[RunResult]]:
        """Return all stored probe results for a concurrency level."""
        return self._data[concurrency]

    def concurrency_levels(self) -> list[int]:
        """Return sorted list of all probed concurrency values."""
        return sorted(self._data)
