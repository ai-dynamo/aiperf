# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RunExecutor interface for MultiRunOrchestrator.

A RunExecutor takes a fully-built BenchmarkRun and executes it to produce
a RunResult. Current implementation: LocalSubprocessExecutor (forks a
subprocess). A K8sChildJobExecutor that creates an AIPerfJob CR in the
sweep-controller pod is planned but not yet implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun
    from aiperf.orchestrator.models import RunResult


__all__ = ["RunExecutor"]


class RunExecutor(ABC):
    """Executes a single (variation, trial) BenchmarkRun and returns a RunResult."""

    @abstractmethod
    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Execute one benchmark run.

        Implementations:
            - LocalSubprocessExecutor: fork subprocess of aiperf.orchestrator.subprocess_runner
            - K8sChildJobExecutor (planned): create AIPerfJob CR via kubernetes_asyncio, watch to terminal

        Args:
            run: Fully-built BenchmarkRun (config + variation + trial + label).

        Returns:
            RunResult with success/error and summary metrics.
        """

    @abstractmethod
    def derive_id(self, plan: BenchmarkPlan, var_idx: int, trial: int) -> str:
        """Derive a stable benchmark_id for this run.

        For local: random uuid hex is fine.
        For K8s: deterministic from (sweep_name, var_idx, trial) so a restarted
        sweep-controller pod sees the same name and resumes idempotently.
        """
