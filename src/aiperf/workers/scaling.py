# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone worker scaling calculations.

Used by both SystemController (to pre-compute counts at startup) and
WorkerManager (to determine initial pool size). No service imports.
"""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment

if TYPE_CHECKING:
    from aiperf.config import AIPerfConfig


def calculate_worker_count(config: AIPerfConfig) -> int:
    """Calculate the number of workers to spawn.

    Applies the same logic as WorkerManager.__init__:
    1. Default to 75% of CPU cores - 1, capped at MAX_WORKERS_CAP
    2. Cap by concurrency if set
    3. Floor at min workers (default 1)
    """
    cpu_count = multiprocessing.cpu_count()
    max_workers = config.runtime.workers

    # Extract max concurrency across all phases
    concurrency: int | None = None
    for phase in config.phases.values():
        if phase.concurrency is not None:
            concurrency = max(concurrency or 0, phase.concurrency)

    if max_workers is None:
        max_workers = max(
            1,
            min(
                int(cpu_count * Environment.WORKER.CPU_UTILIZATION_FACTOR) - 1,
                Environment.WORKER.MAX_WORKERS_CAP,
            ),
        )

    if concurrency and concurrency < max_workers:
        max_workers = concurrency

    max_workers = max(max_workers, config.runtime.workers_min or 1)
    return max_workers


def calculate_record_processor_count(num_workers: int) -> int:
    """Calculate the number of record processors to scale with workers."""
    return max(1, num_workers // Environment.RECORD.PROCESSOR_SCALE_FACTOR)
