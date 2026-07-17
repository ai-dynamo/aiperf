# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared idle-gap warp entry defaults for recorded-trace adapters (weka, dynamo).

Recorded traces replay their inter-request delays through the shared
:class:`~aiperf.dataset.graph.segment_ir.trie_content.ActiveIdleWarp`. The run
path resolves the cap from the dataset's ``synthesis.idle_gap_cap_seconds``
(``--synthesis-idle-gap-cap``) and forwards it via
``GraphParseContext.idle_gap_cap_seconds``; the default below is only the
adapter-entry fallback for direct callers with no run config (CLI tooling,
tests). Both adapters share ONE sentinel + default so a dynamo replay warps
exactly like a weka replay under the same knob.
"""

from __future__ import annotations

from typing import Any

# Default per-trace idle-gap cap (seconds) when the entry kwarg is left at the
# sentinel -- the value the recorded weka workloads were captured/replayed
# against. ``workload_detect._GRAPH_IDLE_GAP_CAP_DEFAULT`` and the
# ``synthesis.idle_gap_cap_seconds`` config default mirror it for the run path.
DEFAULT_IDLE_GAP_CAP_SECONDS = 60.0

# Sentinel for adapter-entry ``idle_gap_cap_seconds`` kwargs: left in place the
# cap resolves to :data:`DEFAULT_IDLE_GAP_CAP_SECONDS`; pass an explicit float
# (or ``None`` to disable warping) to override. A sentinel -- not the default as
# the literal -- lets callers pass ``None`` to mean "disable" without it
# colliding with "use the default".
IDLE_GAP_CAP_USE_DEFAULT: Any = object()


def resolve_idle_gap_cap(value: float | None | Any) -> float | None:
    """Resolve an entry kwarg to a concrete cap (sentinel -> the 60s default)."""
    if value is IDLE_GAP_CAP_USE_DEFAULT:
        return DEFAULT_IDLE_GAP_CAP_SECONDS
    return value
