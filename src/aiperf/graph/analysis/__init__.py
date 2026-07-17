# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native static trace analysis over the async-dataflow graph engine.

A pure, side-effect-free elaboration of a parsed graph + one trace into an
ordered firing timeline. Reuses the dataflow ``Scheduler`` (adjacency) and the
channel model's ``producers_per_channel`` fan-in primitive rather than a
separate analysis scheduler. The only ordering index is a
parallel-readiness ``cohort`` frontier counter, which is a derived view, not
an execution barrier.
"""

from __future__ import annotations

from aiperf.graph.analysis.snapshot import (
    compute_snapshot,
    trace_duration_us,
)

__all__ = [
    "compute_snapshot",
    "trace_duration_us",
]
