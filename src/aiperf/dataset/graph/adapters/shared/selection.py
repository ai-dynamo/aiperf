# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-only filter-then-cap trace selection shared by the recorded adapters.

The graph plane historically ignored ``--num-dataset-entries`` and
``--max-context-length`` (ai-dynamo/aiperf#1106): every recorded trace was
built and dispatch silently cloned traces to fill lanes. This module is the ONE
selection primitive the graph-plane loaders call to honor those knobs BEFORE
the expensive build:

* FILTER: drop a candidate whose peak context (input + output tokens on its
  largest single request, computed schema-only by the
  :mod:`~aiperf.dataset.graph.adapters.shared.peak_context` helpers) exceeds
  ``max_context_length``.
* CAP: keep the FIRST ``num_dataset_entries`` eligible candidates (in the
  adapter's deterministic scan order).

"Filter THEN cap" -- the cap is applied to the ELIGIBLE set, never to the raw
prefix (which would drop below N whenever a rejected trace fell in the first N).
Scanning stops as soon as the cap is reached, so on a capped load only the first
N-eligible-plus-rejected-so-far candidates are ever examined; the returned
:class:`SelectionStats` reflect only what was scanned.
"""

from __future__ import annotations

from aiperf.common.aiperf_logger import AIPerfLogger

# The selection primitive itself is schema-only and identical to the trace
# loader's, so the graph plane REUSES it rather than keeping a second copy in
# sync; only the summary log line below is graph-specific.
from aiperf.dataset.loader.selection import SelectionStats, filter_then_cap

_logger = AIPerfLogger(__name__)


def log_selection_summary(
    stats: SelectionStats,
    *,
    source: str,
    num_dataset_entries: int | None,
    max_context_length: int | None,
) -> None:
    """Emit the once-per-build filter-then-cap summary (mirrors the trace loader).

    The graph counterpart of
    :meth:`~aiperf.dataset.loader.base_trace_loader.BaseTraceLoader._log_filtering_summary`:
    called at the PARENT-side finalize point where the single
    :class:`SelectionStats` is produced (once per build, before any worker
    fan-out), so it logs exactly once no matter how the build parallelizes.
    Every adapter finalize point routes through this ONE helper so the summary
    text is uniform across the graph-plane loaders. When no selection knob
    is set the selection scan never runs, so this is never called (byte-identical
    quiet path).
    """
    _logger.info(
        lambda: (
            f"graph trace selection [{source}]: scanned {stats.scanned:,}, "
            f"rejected_by_maxctx {stats.rejected_by_maxctx:,} "
            f"(--max-context-length={max_context_length}), "
            f"eligible {stats.eligible:,}, loaded {stats.loaded:,} "
            f"(--num-dataset-entries={num_dataset_entries}), "
            f"largest_observed_peak_context {stats.largest_observed:,}"
        )
    )


__all__ = [
    "SelectionStats",
    "filter_then_cap",
    "log_selection_summary",
]
