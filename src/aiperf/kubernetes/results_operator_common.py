# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared download primitives for the operator results-server clients.

A leaf module: ``results_operator`` imports ``results_operator_sweeps``, so
anything both need lives here rather than in either of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["_JobDownloadOutcome", "_is_refused_name"]


@dataclass(frozen=True, slots=True)
class _JobDownloadOutcome:
    """Result of downloading every advertised file for one run."""

    downloaded: list[tuple[str, int]]
    """(display name, size in bytes) for each file that landed on disk."""

    failed: list[str]
    """Display names the server advertised but did not deliver."""

    @property
    def complete(self) -> bool:
        """True when every advertised file was retrieved."""
        return not self.failed


def _is_refused_name(display_name: str) -> bool:
    """True when we decline to write this name, regardless of the server.

    Dot-files (the results-ready marker among them), absolute paths and
    parent traversals are refused by policy. They are advertised in listings
    but never downloaded, so they are skips rather than failures.
    """
    normalized = Path(display_name)
    leaf = normalized.name
    return (
        not leaf
        or leaf.startswith(".")
        or normalized.is_absolute()
        or ".." in normalized.parts
    )
