# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure, aiperf-import-free time-model transforms for Baseten trace replay.

These functions rewrite per-event replay timestamps for the ``baseten_trace``
loader. They are deliberately dependency-free (stdlib only) so they unit-test
exhaustively and stay portable to other consumers. They never see ``hash_ids`` or prompt content, so KV-cache fidelity
is structurally preserved — unlike the Synthesizer's speedup path, which
rewrites ``hash_ids``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["reflow_idle_gaps"]


def reflow_idle_gaps(timestamps_ms: Sequence[float], cap_ms: float | None) -> list[int]:
    """Collapse global idle gaps in a replay schedule.

    Given per-event normalized timestamps (ms, in any positional order), return
    a new list in the **same positional order** where — walking events in time
    order — any gap to the next event larger than ``cap_ms`` is shortened to
    ``cap_ms``. Ordering and relative spacing up to the cap are preserved; pure
    dead air (stretches with no events) is removed so fixed-schedule replay of a
    sparse, sampled trace does not idle.

    The earliest event keeps its original value: trimming the *leading* offset
    is the origin policy's concern, not this function's.

    ``cap_ms`` of ``None`` disables the reflow (identity transform); a
    non-positive cap raises ``ValueError`` (the CLI enforces ``> 0``), and a
    sub-millisecond cap rounds up to 1 ms since timestamps are integer ms.
    Ties (equal timestamps) are preserved as zero-length gaps and keep their
    original relative order (stable sort).
    """
    if cap_ms is not None and cap_ms <= 0:
        raise ValueError(f"cap_ms must be positive or None, got {cap_ms}")
    values = [int(t) for t in timestamps_ms]
    n = len(values)
    if cap_ms is None or n <= 1:
        return values

    cap = math.ceil(cap_ms)
    # Stable order by (timestamp, original index) so ties keep input order.
    order = sorted(range(n), key=lambda i: (values[i], i))
    out = [0] * n
    first = order[0]
    out[first] = values[first]
    prev_old = values[first]
    prev_new = values[first]
    for i in order[1:]:
        gap = values[i] - prev_old  # >= 0 because we walk in sorted order
        prev_new += min(gap, cap)
        out[i] = prev_new
        prev_old = values[i]
    return out
