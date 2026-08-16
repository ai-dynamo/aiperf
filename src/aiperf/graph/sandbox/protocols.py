# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool-execution sandbox seam.

One sandbox per trace instance. The executor opens it before the trace runs,
`ToolNode` dispatch calls `run` per recorded command, and it closes after the
trace finishes -- all outside the measured window except `run` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True, frozen=True)
class ToolResult:
    """One executed command's outcome."""

    stdout: str
    """Combined stdout/stderr text the command produced."""
    returncode: int
    """Process exit status; -1 when the command timed out."""
    duration_s: float
    """Measured wall-clock execution time, the quantity the benchmark reports."""
    timed_out: bool
    """Whether the per-command ceiling fired."""


class ToolSandbox(Protocol):
    """Executes recorded commands somewhere isolated from the harness."""

    async def open(self) -> None:
        """Start the session. Called outside the measured window."""
        ...

    async def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        """Execute one recorded command and return its outcome.

        Must not raise on command failure -- a nonzero exit or a timeout is a
        recorded outcome, not a harness error. Raise only when the session
        itself is unusable.
        """
        ...

    async def close(self) -> None:
        """Tear the session down. Idempotent; called outside the measured window."""
        ...
