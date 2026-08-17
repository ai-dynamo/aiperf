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
from typing import Any, Protocol

from aiperf.common.enums import CacheBustTarget

_DIGEST_LEN = 12  # 12 hex chars = 48 bits, ample for in-run uniqueness

_MARKER_TOKEN_SAMPLES = 8

_SUFFIX_SEP = "::"
_UNSET = object()

WARMUP_ISOLATION_MARKER = "[warmup]\n\n"
WARMUP_ISOLATION_TARGETS = (
    CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
    CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
)


def base_trace_id(conversation_id: str) -> str:
    """Strip any ``::``-delimited descendant suffix (``::sa:``/``::fa:``) to the root trace id.

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

    Two paths depending on ``target``:

    (a) RID-digest targets (``SYSTEM_PREFIX``, ``SYSTEM_SUFFIX``,
        ``FIRST_TURN_PREFIX``, ``FIRST_TURN_SUFFIX``): the digest tuple is
        intentionally phase-agnostic. Spec requires "warmup-coherent" markers:
        a trajectory's warmup turn ``k_i`` and its first profiling turn
        ``k_i+1`` must share the same marker so warmup KV-cache work transfers
        to profiling. Adding phase to the digest would defeat that — keep it out.

    (b) ``WARMUP_ISOLATION_*`` targets: return the constant
        ``WARMUP_ISOLATION_MARKER`` string. No digest is computed. The
        phase-aware gate (emit during WARMUP, suppress during PROFILING) lives
        in ``CreditIssuer._issue_credit_internal``, not here.

    Returns ``None`` when target is NONE so callers can unconditionally pass
    the result through into ``Credit.cache_bust_marker: str | None``. Returning
    ``""`` would introduce a third "no marker" value distinct from ``None``.
    """
    if target == CacheBustTarget.NONE:
        return None
    if target in WARMUP_ISOLATION_TARGETS:
        return WARMUP_ISOLATION_MARKER

    unique_str = f"{benchmark_id}:{recycle_pass}:{trajectory_index}:{trace_id}"
    digest = hashlib.sha256(unique_str.encode()).hexdigest()[:_DIGEST_LEN]
    rid = f"[rid:{digest}]"

    if target in (CacheBustTarget.SYSTEM_PREFIX, CacheBustTarget.FIRST_TURN_PREFIX):
        return f"{rid}\n\n"
    return f"\n\n{rid}"


def build_trace_instance_marker(
    benchmark_id: str,
    trace_instance_id: str,
    *,
    target: CacheBustTarget,
) -> str | None:
    """Mint the per-TRACE-INSTANCE cache-bust marker for the agent-graph replay path.

    Matches the per-trajectory-TREE scoping of :func:`resolve_tree_marker` (which
    keys the marker on ``root_correlation_id`` and shares ONE value across every
    member of the tree). On the graph path the trajectory-tree analog is the trace
    INSTANCE id (``credit.trace_id``, ``{template}::{nonce}``, e.g.
    ``t-1::3f2a...``): every dispatch of one
    instance carries the same ``trace_id`` (the adapter pins ``credit.trace_id``
    to the root instance for nested dispatches too), so digesting it yields ONE
    marker shared across all of the instance's turns/dispatches. Distinct
    instances (``t-1::3f2a...`` vs ``t-2::9c17...``) get distinct markers, and a
    RECYCLED template (a new session slot -- the same ``t-1`` template with a
    fresh nonce) is a fresh instance id -> a fresh marker, mirroring the
    per-recycle ``recycle_pass`` bump.

    The recycle pass is already distinguished by the instance id (each recycle
    mints a fresh ``::{nonce}``),
    so the digest tuple's ``recycle_pass`` / ``trajectory_index`` slots are pinned
    to ``0`` and the instance id fills the ``trace_id`` slot. The rendered marker
    is byte-identical to :func:`build_cache_bust_marker` -- a ``[rid:<12hex>]\\n\\n``
    prefix whose digest is ``sha256(f"{benchmark_id}:0:0:{trace_instance_id}")[:12]``.

    Args:
        benchmark_id: Run-scoped salt so two runs mint different markers.
        trace_instance_id: The trace INSTANCE id (``credit.trace_id``); shared by
            every dispatch of the instance, distinct per instance, regenerated on
            recycle.
        target: Injection position. ``NONE`` returns ``None`` (no marker);
            ``FIRST_TURN_PREFIX`` renders the prefix marker.

    Returns:
        The rendered marker text, or ``None`` when ``target`` is ``NONE``.
    """
    return build_cache_bust_marker(
        benchmark_id,
        0,
        0,
        trace_instance_id,
        target=target,
    )


def _content_has_marker_prefix(content: Any, marker: str) -> bool:
    """Whether ``content`` already begins with ``marker`` (idempotency guard).

    Mirrors the agentx worker's marker-at-edge check for the prefix case: a plain
    string is checked with ``startswith``; an OpenAI multimodal list-of-parts is
    checked against a leading ``{"type": "text", "text": ...}`` part. Any other
    content shape is treated as "no marker present".
    """
    if isinstance(content, str):
        return content.startswith(marker)
    if isinstance(content, list) and content:
        return content[0] == {"type": "text", "text": marker.strip()}
    return False


def inject_marker_at_first_user_message(
    messages: list[dict[str, Any]], marker: str | None
) -> None:
    """Prepend ``marker`` to the first ``role == "user"`` message, in place.

    Implements the ``FIRST_TURN_PREFIX`` target for the graph path: walk the wire
    ``messages`` forward, find the first user-role message, and prepend the marker
    to its content. Plain-string content becomes ``marker + content``; OpenAI
    multimodal list content gets a leading ``{"type": "text", "text":
    marker.strip()}`` part. The injection is idempotent (re-stamping the same
    marker is a no-op) and stamps ONLY the first user message. No-op when
    ``marker`` is falsy (``NONE`` target) or no user-role message exists.

    Args:
        messages: The materialized wire ``messages`` list (mutated in place).
        marker: The rendered marker text, or ``None`` to stamp nothing.
    """
    if not marker:
        return
    for idx, msg in enumerate(messages):
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        content = msg.get("content", "")
        if _content_has_marker_prefix(content, marker):
            return
        if isinstance(content, str):
            messages[idx] = {**msg, "content": marker + content}
            return
        if isinstance(content, list):
            marker_part = {"type": "text", "text": marker.strip()}
            messages[idx] = {**msg, "content": [marker_part, *content]}
            return
        return


def estimate_marker_token_cost(
    target: CacheBustTarget,
    tokenizer: _EncodeOnly,
    samples: int = _MARKER_TOKEN_SAMPLES,
) -> int:
    """Average token count of the cache-bust marker for a given target.

    Tokenizes ``samples`` distinct markers and rounds the mean to an int.
    Returns 0 for ``CacheBustTarget.NONE`` and for ``WARMUP_ISOLATION_*``
    targets (their marker is only injected during WARMUP; PROFILING credits
    carry ``None``, so there is no profiling-phase token cost to estimate).
    The 12-hex digest dominates the variance for RID targets, so a handful
    of samples is enough.
    """
    if target == CacheBustTarget.NONE:
        return 0
    if target in WARMUP_ISOLATION_TARGETS:
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
