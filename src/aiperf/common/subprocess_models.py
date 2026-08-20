# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclass models used by ``SubprocessManager``.

Kept in a separate module so the main manager file stays under the
``file-size`` ergonomic limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.types import ServiceTypeT

if IS_WINDOWS:
    from multiprocessing.context import SpawnProcess

    ForkProcess = SpawnProcess
    ForkServerProcess = SpawnProcess
else:
    from multiprocessing.context import ForkProcess, ForkServerProcess, SpawnProcess


@dataclass(slots=True)
class SubprocessInfo:
    """Information about a subprocess managed by ``SubprocessManager``."""

    service_type: ServiceTypeT
    """Type of service running in the process."""

    service_id: str
    """ID of the service running in the process."""

    process: Process | SpawnProcess | ForkProcess | ForkServerProcess | None = None
    """The underlying multiprocessing process instance."""

    @property
    def exitcode(self) -> int | None:
        """Exit code of the process, or None if still running or no process."""
        return self.process.exitcode if self.process else None

    @property
    def pid(self) -> int | None:
        """PID of the process, or None if no process."""
        return self.process.pid if self.process else None
