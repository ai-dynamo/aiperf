# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-conversation cache-bust marker builder.

Same (benchmark_id, recycle_pass, trajectory_index, trace_id) always yields
the same digest - reproducible across reruns. Position controls whitespace
placement, not the digest itself.

``trace_id`` is part of the digest because ``recycle_pass`` counts per trace:
without it, a lane recycling from one trace to another would repeat the same
``(recycle_pass, lane)`` tuple and reuse the marker, letting the server's
prefix cache stay warm across the boilerplate different plays share — the
exact cross-play warming the marker exists to defeat.
"""

import hashlib
from typing import Protocol

from aiperf.common.enums import CacheBustTarget

_DIGEST_LEN = 12  # 12 hex chars = 48 bits, ample for in-run uniqueness

_MARKER_TOKEN_SAMPLES = 8

_SUFFIX_SEP = "::"
_UNSET = object()

WARMUP_ISOLATION_MARKER = "[warmup]\n\n"


def base_trace_id(conversation_id: str) -> str:
    """Strip any descendant suffix (``::sa:``/``::fa:``/``:sN``) to the root trace id.

    Every member of a trajectory tree — the depth-0 main session and its subagent
    (``::sa:``) / flat-agent (``::fa:``) descendants — shares one base trace id, so
    keying the marker digest on it lets any member compute the same value.
    """
    return conversation_id.split(_SUFFIX_SEP, 1)[0]


def resolve_tree_marker(
    ledger,
    root_correlation_id: str,
    *,
    benchmark_id: str,
    trajectory_index: int,
    conversation_id: str,
    target: CacheBustTarget,
) -> str | None:
    """Resolve the cache-bust marker for a trajectory TREE, idempotently.

    The marker is a property of the tree (``root_correlation_id``): the first member
    to resolve mints it (digesting the base trace id + tree lane, bumping
    ``recycle_pass`` once); every other member — main turns, subagents, flat agents,
    at any depth and in any dispatch order — reuses the stored value. Because the
    ledger survives the WARMUP -> PROFILING boundary, a tree that continues across
    phases keeps its marker, while fresh trees (recycles, new lanes) mint distinct
    ones.

    ``ledger`` is duck-typed: it needs ``session_marker`` (dict keyed by
    ``root_correlation_id``) and ``recycle_pass`` (dict keyed by base trace id).
    Returns ``None`` when cache-bust is disabled, recording the ``None`` so callers
    can look it up unconditionally.
    """
    existing = ledger.session_marker.get(root_correlation_id, _UNSET)
    if existing is not _UNSET:
        return existing
    if target == CacheBustTarget.NONE:
        ledger.session_marker[root_correlation_id] = None
        return None
    base = base_trace_id(conversation_id)
    new_pass = ledger.recycle_pass.get(base, -1) + 1
    ledger.recycle_pass[base] = new_pass
    marker = build_cache_bust_marker(
        benchmark_id, new_pass, trajectory_index, base, target=target
    )
    ledger.session_marker[root_correlation_id] = marker
    return marker


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
    if target in (
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
        CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
    ):
        return WARMUP_ISOLATION_MARKER

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
    if target in (
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
        CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
    ):
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
