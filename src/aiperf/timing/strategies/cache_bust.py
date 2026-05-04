# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-conversation cache-bust marker builder.

Same (benchmark_id, recycle_pass, trajectory_index, trace_id) always yields
the same digest - reproducible across reruns. Position controls whitespace
placement, not the digest itself.

Adding ``trace_id`` to the four-dimensional digest input ensures every
(recycle_pass, lane, trace) combination is unique by construction. Without
``trace_id``, two different traces landing on the same ``(recycle_pass, lane)``
tuple at different points in time would produce the same marker — empirically
a 33% collision rate at MVP scale.
"""

import hashlib
from typing import Protocol

from aiperf.common.enums import CacheBustTarget

_DIGEST_LEN = 12  # 12 hex chars = 48 bits, ample for in-run uniqueness

_MARKER_TOKEN_SAMPLES = 8


class _EncodeOnly(Protocol):
    def encode(self, text: str, **kwargs) -> list[int]: ...


def build_cache_bust_marker(
    benchmark_id: str,
    recycle_pass: int,
    trajectory_index: int,
    trace_id: str,
    *,
    target: CacheBustTarget,
) -> str | None:
    """Render the marker text for the given inputs and target position.

    The digest tuple is intentionally phase-agnostic. Spec requires
    "warmup-coherent" markers: a trajectory's warmup turn ``k_i`` and its
    first profiling turn ``k_i+1`` must share the same marker so warmup
    KV-cache work transfers to profiling. Adding phase to the digest
    would defeat that — keep it out.

    Returns ``None`` when target is NONE so callers can unconditionally pass
    the result through into ``Credit.cache_bust_marker: str | None``. Returning
    ``""`` would introduce a third "no marker" value distinct from ``None``.
    """
    if target == CacheBustTarget.NONE:
        return None

    unique_str = f"{benchmark_id}:{recycle_pass}:{trajectory_index}:{trace_id}"
    digest = hashlib.sha256(unique_str.encode()).hexdigest()[:_DIGEST_LEN]
    rid = f"[rid:{digest}]"

    if target in (CacheBustTarget.SYSTEM_PREFIX, CacheBustTarget.FIRST_TURN_PREFIX):
        return f"{rid}\n\n"
    return f"\n\n{rid}"


def estimate_marker_token_cost(
    target: CacheBustTarget,
    tokenizer: _EncodeOnly,
    samples: int = _MARKER_TOKEN_SAMPLES,
) -> int:
    """Average token count of the cache-bust marker for a given target.

    Tokenizes ``samples`` distinct markers and rounds the mean to an int.
    Returns 0 for ``CacheBustTarget.NONE``. The 12-hex digest dominates
    the variance, so a handful of samples is enough.
    """
    if target == CacheBustTarget.NONE:
        return 0

    total = 0
    for i in range(samples):
        marker = build_cache_bust_marker(
            benchmark_id="estimator",
            recycle_pass=i,
            trajectory_index=i,
            trace_id=f"estimator-{i}",
            target=target,
        )
        total += len(tokenizer.encode(marker))
    return round(total / samples)
