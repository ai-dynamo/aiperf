# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-session cache-bust marker builder for graph snapshot replay.

Same ``(benchmark_id, recycle_pass, lane_index, trace_id)`` always yields the
same digest - reproducible across reruns. Adding ``trace_id`` to the digest
input ensures every ``(recycle_pass, lane, trace)`` combination is unique by
construction: without it, two different traces landing on the same
``(recycle_pass, lane)`` tuple would collide on one marker.

Byte-compatible with the agentx ``timing/strategies/cache_bust.py`` digest
(``sha256(f"{benchmark_id}:{recycle_pass}:{trajectory_index}:{trace_id}")[:12]``,
wrapped as ``[rid:<digest>]\\n\\n`` for the prefix target). The graph snapshot
path uses only the ``FIRST_TURN_PREFIX`` / ``NONE`` variants.
"""

from __future__ import annotations

import hashlib
from typing import Any

from aiperf.common.enums import CacheBustTarget

_DIGEST_LEN = 12  # 12 hex chars = 48 bits, ample for in-run uniqueness


def build_cache_bust_marker(
    benchmark_id: str,
    recycle_pass: int,
    lane_index: int,
    trace_id: str,
    *,
    target: CacheBustTarget,
) -> str | None:
    """Render the marker text for the given inputs and target position.

    The digest tuple is intentionally phase-agnostic. A lane's WARMUP priming
    turn and its first PROFILING turn must share the same marker so the warmup
    KV-cache work transfers to profiling; adding phase to the digest would
    defeat that - keep it out.

    Returns ``None`` when target is ``NONE`` so callers can treat "no marker"
    uniformly -- ``inject_marker_at_first_user_message`` no-ops on a falsy
    marker (returning ``""`` would introduce a third "no marker" value
    distinct from ``None``).

    Args:
        benchmark_id: Run-scoped salt (the source's ``random_seed`` stringified)
            so two runs with different seeds mint different markers.
        recycle_pass: Never-restarting per-lane pass counter; rotates the digest
            on every recycle so a recycled lane can never collide with a warmed
            digest from an earlier pass.
        lane_index: Absolute lane ordinal; decorrelates concurrent lanes that
            sampled the same trace.
        trace_id: The lane's sampled conversation/trace id; makes the digest
            unique per trace within a ``(recycle_pass, lane)`` slot.
        target: Injection position. ``NONE`` returns ``None``;
            ``FIRST_TURN_PREFIX`` wraps the digest with a trailing blank line.
    """
    if target == CacheBustTarget.NONE:
        return None

    unique_str = f"{benchmark_id}:{recycle_pass}:{lane_index}:{trace_id}"
    digest = hashlib.sha256(unique_str.encode()).hexdigest()[:_DIGEST_LEN]
    rid = f"[rid:{digest}]"
    return f"{rid}\n\n"


def build_trace_instance_marker(
    benchmark_id: str,
    trace_instance_id: str,
    *,
    target: CacheBustTarget,
) -> str | None:
    """Mint the per-TRACE-INSTANCE cache-bust marker for the graph-IR replay path.

    Matches agentx's per-trajectory-TREE scoping (``cache_bust.py::resolve_tree_marker``
    keys the marker on ``root_correlation_id`` and shares ONE value across every
    member of the tree -- main turns, subagents, flat agents). On our graph path
    the trajectory-tree analog is the trace INSTANCE id (``credit.trace_id``, e.g.
    ``t-1#0``): every dispatch of one instance carries the same ``trace_id`` (the
    adapter pins ``credit.trace_id`` to the root instance for nested/subagent
    dispatches too), so digesting it yields ONE marker shared across all of the
    instance's turns/dispatches. Distinct instances (``t-1#0`` vs ``t-2#0``) get
    distinct markers, and a RECYCLED template (a new session slot, e.g. ``t-1#1``)
    is a fresh instance id -> a fresh marker, mirroring agentx's per-recycle
    ``recycle_pass`` bump.

    The recycle ordinal is already baked into the instance id (the ``#N`` suffix),
    so the digest tuple's ``recycle_pass`` / ``lane_index`` slots are pinned to
    ``0`` and the instance id fills the ``trace_id`` slot. The rendered marker is
    byte-identical to :func:`build_cache_bust_marker` -- a ``[rid:<12hex>]\\n\\n``
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
        recycle_pass=0,
        lane_index=0,
        trace_id=trace_instance_id,
        target=target,
    )


def _content_has_marker_prefix(content: Any, marker: str) -> bool:
    """Whether ``content`` already begins with ``marker`` (idempotency guard).

    Mirrors agentx ``worker.py::_content_has_marker_at_edge`` for the prefix
    case: a plain string is checked with ``startswith``; an OpenAI multimodal
    list-of-parts is checked against a leading ``{"type": "text", "text": ...}``
    part. Any other content shape is treated as "no marker present".
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

    Mirrors agentx ``worker.py::_inject_marker_into_first_user_turn`` for the
    ``FIRST_TURN_PREFIX`` target: walk the wire ``messages`` forward, find the
    first user-role message, and prepend the marker to its content. Plain-string
    content becomes ``marker + content``; OpenAI multimodal list content gets a
    leading ``{"type": "text", "text": marker.strip()}`` part. The injection is
    idempotent (re-stamping the same marker is a no-op) and stamps ONLY the first
    user message. No-op when ``marker`` is falsy (``NONE`` target) or no
    user-role message exists.

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


__all__ = [
    "build_cache_bust_marker",
    "build_trace_instance_marker",
    "inject_marker_at_first_user_message",
]
