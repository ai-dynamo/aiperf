# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side materialization of a graph-IR node's request.

On a graph credit the worker rebuilds the node's request from its OWN per-node
envelope in the shared unified segment store (``GraphSegmentUnifiedClient``) --
NO per-worker session, NO co-location, any worker serves any node. Each envelope
self-describes the node's full message list as interned int ``handles``
(materialized via ``materialize_handles``); a slot-carrying node also carries an
``items`` assembly program that splices predecessor responses from the
worker-local dynamic pool (see ``_assemble_items``) to reconstruct the full
user/assistant alternation. The build interns the whole conversation prefix per
node, so there is no ancestor concatenation at dispatch. Warmup reuses the
profiling envelope and applies the warmup 1-token output cap at materialization
time. After the messages are built the node's own ``dispatch_overrides``,
run-level model fallback, warmup cap, and ``stream`` mode are layered on. No
tokenizer, no synthesis, no catalog at dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.enums import CacheBustTarget
from aiperf.common.environment import Environment
from aiperf.timing.strategies.cache_bust import (
    build_trace_instance_marker,
    inject_marker_at_first_user_message,
)

if TYPE_CHECKING:
    from aiperf.common.models import EndpointInfo
    from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient

# The node's recorded output cap is carried under this dispatch-override key; the
# worker maps it to the wire token field per the endpoint convention (see
# ``endpoints/openai_chat.py::format_payload``).
_MAX_OUTPUT_TOKENS_KEY = "max_output_tokens"
_LEGACY_TOKEN_FIELD = "max_tokens"
_MODERN_TOKEN_FIELD = "max_completion_tokens"
_PROFILING_VARIANT = "profiling"
_WARMUP_VARIANT = "warmup"

# Dynamo session-identity headers stamped verbatim at build time
# (``trie_lowering.py::_node_meta``). These carry the RECORDED session ids and
# must be uniquified per replay instance; ``x-dynamo-session-final`` is a
# per-turn flag, not an identity, and is forwarded untouched.
_DYNAMO_SESSION_ID_HEADERS = ("x-dynamo-session-id", "x-dynamo-parent-session-id")


def encode_overrides_inner(overrides: dict | None) -> bytes:
    """Outer body fields (max_tokens/model/stream/extra/...) as the inner JSON to
    splice after the messages array. orjson handles all value typing/escaping."""
    if not overrides:
        return b""
    return orjson.dumps(overrides)[1:-1]  # strip the { }


def read_node_envelope(
    client: GraphSegmentUnifiedClient,
    trace_id: str,
    node_ordinal: int,
    phase_variant: str = "profiling",
) -> dict[str, Any] | None:
    """Fetch + decode one node's manifest envelope (variant-aliased), or None.

    The single pre-read seam: the worker calls this ONCE per graph credit --
    to route bytes-vs-dict and read the dynamic-content envelope flags
    (``capture``, ``items``) -- and passes the decoded envelope into the
    materialize functions so the manifest is never fetched or decoded twice.
    """
    lookup_variant = (
        _PROFILING_VARIANT if phase_variant == _WARMUP_VARIANT else phase_variant
    )
    raw = client.get_node_envelope(trace_id, node_ordinal, lookup_variant)
    if raw is None:
        return None
    return orjson.loads(raw)


def _assemble_items(
    client: GraphSegmentUnifiedClient,
    items: list[dict[str, Any]],
    slot_resolver: Callable[[int], Any],
) -> list[dict[str, Any]]:
    """Assemble a slot-carrying node's messages from its ``items`` program.

    ``{"h": handle}`` appends the interned static message; ``{"s": {"src"}}``
    appends the producer's pooled reply as an assistant message — the verbatim
    recorded assistant message (``orjson.loads(message_json)``, ``tool_calls``
    preserved) when the capture is structured, else ``{"role": "assistant",
    "content": text}`` — or nothing on FAILED/EMPTY (deliberate omission);
    ``{"m": {"role", "parts"}}`` emits one composed message whose content
    concatenates static text parts and pooled slot texts (FAILED/EMPTY
    substitute the empty string, so the role and static instruction survive a
    failed producer). A MISSING pool value is :class:`GraphPoolMissingError` —
    broken stickiness or backstop eviction is a loud trace error, never a
    default.
    """
    from aiperf.graph.dynamic_pool import (
        GraphCapturedReply,
        GraphPoolMissingError,
        GraphPoolSentinel,
    )

    def _slot_value(src: int) -> GraphCapturedReply | None:
        value = slot_resolver(src)
        if value is None:
            raise GraphPoolMissingError(src)
        if isinstance(value, GraphPoolSentinel):
            return None
        return value

    messages: list[dict[str, Any]] = []
    for token in items:
        if "h" in token:
            messages.extend(client.materialize_handles([token["h"]]))
        elif "s" in token:
            reply = _slot_value(token["s"]["src"])
            if reply is None:
                pass  # FAILED/EMPTY producer: deliberate omission
            elif reply.message_json:
                messages.append(orjson.loads(reply.message_json))
            else:
                messages.append({"role": "assistant", "content": reply.text})
        elif "m" in token:
            parts: list[str] = []
            for part in token["m"]["parts"]:
                if "t" in part:
                    parts.append(part["t"])
                else:
                    reply = _slot_value(part["sv"])
                    parts.append(reply.text if reply is not None else "")
            messages.append({"role": token["m"]["role"], "content": "".join(parts)})
        else:
            raise ValueError(f"unknown assembly item {token!r}")
    if Environment.GRAPH.MERGE_CONSECUTIVE_USER:
        messages = _merge_consecutive_user(messages)
    return messages


def _merge_consecutive_user(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse consecutive user-role messages into one (contents newline-joined).

    Consecutive user turns arise in a reconstructed conversation two ways:
    omitting a FAILED/EMPTY producer's assistant turn leaves its user turn
    adjacent to the next one, and a user-role init/delta boundary can place two
    authored user turns back to back. Some chat APIs reject consecutive
    same-role messages. Gated by ``AIPERF_GRAPH_MERGE_CONSECUTIVE_USER``. Only
    user turns merge (adjacent assistants are intended); non-string content is
    left untouched.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        if (
            merged
            and message.get("role") == "user"
            and merged[-1].get("role") == "user"
            and isinstance(message.get("content"), str)
            and isinstance(merged[-1].get("content"), str)
        ):
            merged[-1] = {
                "role": "user",
                "content": f"{merged[-1]['content']}\n{message['content']}",
            }
        else:
            merged.append(message)
    return merged


def materialize_graph_request_unified(
    client: GraphSegmentUnifiedClient,
    trace_id: str,
    node_ordinal: int,
    phase_variant: str = "profiling",
    *,
    use_legacy_max_tokens: bool = False,
    envelope: dict[str, Any] | None = None,
    slot_resolver: Callable[[int], Any] | None = None,
    default_model: str | None = None,
) -> dict[str, Any] | None:
    """Rebuild an INTERNED (A2) node's request payload from int-handle manifests.

    Reads the node's envelope from the unified store and ONLY proceeds when the
    envelope carries ``handles`` (the int-handle path), materializing ``messages``
    via :meth:`GraphSegmentUnifiedClient.materialize_handles` (or via
    :func:`_assemble_items` when the envelope carries a slot-splice ``items``
    program). ``dispatch_overrides`` / warmup / stream handling layer on after,
    stamping ``stream`` only when the envelope recorded one.

    Returns:
        ``{"messages": [...], **dispatch_overrides}`` -- plus ``"stream": bool``
        only when the envelope recorded one -- the materialized request payload
        pieces -- or ``None`` when the node's envelope is absent OR carries no
        ``handles`` (every interned-store envelope carries ``handles``, so a
        ``None`` is a genuine miss).
    """
    if envelope is None:
        envelope = read_node_envelope(client, trace_id, node_ordinal, phase_variant)
    if envelope is None:
        return None
    handles = envelope.get("handles")
    if handles is None:
        return None

    items = envelope.get("items")
    if items is not None and slot_resolver is not None:
        messages = _assemble_items(client, items, slot_resolver)
    else:
        messages = client.materialize_handles(handles)
    request: dict[str, Any] = {"messages": messages}
    _apply_dispatch_overrides(
        request,
        envelope.get("dispatch_overrides") or {},
        use_legacy_max_tokens=use_legacy_max_tokens,
    )
    # Run-level model fallback: a node with no per-node ``dispatch_overrides``
    # model (native graphs -- weka/dynamo stamp their recorded model) needs the
    # run ``--model`` in the wire body, mirroring the linear path's
    # ``turn.model or primary_model_name``. The endpoint's ``format_payload`` is
    # bypassed for graph credits, so nothing else adds it.
    if default_model is not None:
        request.setdefault("model", default_model)
    if phase_variant == _WARMUP_VARIANT:
        request.pop(_MODERN_TOKEN_FIELD, None)
        request[_LEGACY_TOKEN_FIELD] = Environment.GRAPH.WARMUP_MAX_OUTPUT_TOKENS
    env_stream = envelope.get("stream")
    if env_stream is not None:
        request["stream"] = bool(env_stream)
    return request


def materialize_graph_request_unified_bytes(
    client: GraphSegmentUnifiedClient,
    trace_id: str,
    node_ordinal: int,
    phase_variant: str = "profiling",
    *,
    use_legacy_max_tokens: bool = False,
    endpoint: EndpointInfo,
    envelope: dict[str, Any] | None = None,
    default_model: str | None = None,
) -> tuple[bytes, str | None, bool] | None:
    """Build a pre-serialized body from INTERNED (A2) int-handle manifests.

    The contiguous-bytes sibling of :func:`materialize_graph_request_unified`
    (no per-segment ``orjson.loads``, no ``orjson.dumps`` of the messages
    array): reads the SAME envelope, ONLY proceeds when it carries ``handles``,
    and folds every outer field into the overrides tail at build time (the node's
    ``dispatch_overrides`` with the mapped token cap, the warmup cap, and the
    run-level options :func:`apply_run_level_payload_options` adds), then builds
    the body via :meth:`GraphSegmentUnifiedClient.build_request_body_handles`.

    Because the body is built once and cannot be mutated after, the cache-bust
    path is NOT handled here: it mutates message content (which bytes cannot
    do), so the caller takes this path only when
    ``endpoint.cache_bust == CacheBustTarget.NONE``. The overrides tail is built
    with the SAME helpers + order as the dict path; parity is parsed-JSON
    equality (``orjson.loads(body) == dict_path_payload``), not raw-byte
    key-order parity.

    Returns:
        ``(body, model, effective_stream)`` -- the contiguous
        ``{"messages":[...],<overrides>}`` body bytes; the node's per-node model
        (``dispatch_overrides["model"]`` as folded, or ``None``) surfaced for
        parity checks against the dict path's ``payload.get("model")`` -- the
        worker deliberately does NOT stamp ``Turn.model`` with it (the recorded
        model rides only the body bytes; ``record.model_name`` falls back to the
        run ``--model`` for tokenizer selection); and the FINAL stamped wire
        ``stream`` mode, so the caller can carry the recorded per-node override
        onto ``RequestInfo`` for the transport (``effective_streaming``). Or
        ``None`` when the node's envelope is absent OR carries no ``handles``.
    """
    if envelope is None:
        envelope = read_node_envelope(client, trace_id, node_ordinal, phase_variant)
    if envelope is None:
        return None
    handles = envelope.get("handles")
    if handles is None:
        return None

    overrides: dict[str, Any] = {}
    _apply_dispatch_overrides(
        overrides,
        envelope.get("dispatch_overrides") or {},
        use_legacy_max_tokens=use_legacy_max_tokens,
    )
    # Run-level model fallback (see the dict path): fold the run ``--model`` into
    # the body when the node carries none, so native graph requests are not
    # rejected for a missing ``model`` field.
    if default_model is not None:
        overrides.setdefault("model", default_model)
    if phase_variant == _WARMUP_VARIANT:
        overrides.pop(_MODERN_TOKEN_FIELD, None)
        overrides[_LEGACY_TOKEN_FIELD] = Environment.GRAPH.WARMUP_MAX_OUTPUT_TOKENS
    env_stream = envelope.get("stream")
    stream_override = bool(env_stream) if env_stream is not None else None
    apply_run_level_payload_options(
        overrides,
        endpoint,
        stream_override=stream_override,
        skip_endpoint_extra=bool(envelope.get("endpoint_extra_applied")),
    )

    body = client.build_request_body_handles(handles, encode_overrides_inner(overrides))
    return body, overrides.get("model"), bool(overrides["stream"])


def _apply_dispatch_overrides(
    request: dict[str, Any],
    dispatch_overrides: dict[str, Any],
    *,
    use_legacy_max_tokens: bool,
) -> None:
    """Merge ``dispatch_overrides`` into ``request``, mapping the token cap.

    ``max_output_tokens`` becomes ``max_tokens`` (legacy) or
    ``max_completion_tokens`` (modern); all other keys (``model`` and any
    provider-specific tunables) pass through verbatim.
    """
    for key, value in dispatch_overrides.items():
        if key == _MAX_OUTPUT_TOKENS_KEY:
            token_field = (
                _LEGACY_TOKEN_FIELD if use_legacy_max_tokens else _MODERN_TOKEN_FIELD
            )
            request[token_field] = value
        else:
            request[key] = value


def stamp_cache_bust_marker(
    payload: dict[str, Any],
    *,
    benchmark_id: str,
    trace_instance_id: str,
    target: CacheBustTarget,
) -> None:
    """Stamp the FIRST_TURN_PREFIX cache-bust marker on the wire payload.

    The marker is PER-TRACE-INSTANCE: every dispatch of one trace instance
    (``credit.trace_id``, e.g. ``t-1#0`` -- including its nested/subagent
    dispatches, which share the same ``trace_id``) carries the SAME marker, so
    the instance's own conversation prefix stays consistent and prefix-caches
    WITHIN the instance. Distinct instances get distinct markers (cross-instance
    bust), and a recycled template (a fresh instance id like ``t-1#1``) mints a
    fresh marker. Because the marker is deterministic from ``(benchmark_id,
    trace_instance_id)``, the stateless worker needs no shared ledger to agree
    across dispatches.

    The marker is prepended to the first ``role == "user"`` message of the
    materialized ``payload["messages"]`` -- the FIRST user turn only, idempotent,
    in the ``[rid:<12hex>]\\n\\n`` prefix format of agentx's cache-bust path
    (``inject_marker_at_first_user_message`` mirrors agentx
    ``worker.py::_inject_marker_into_first_user_turn``).

    No-op (payload byte-identical) when ``target`` is ``NONE`` -- the default --
    so the verbatim-replay path is unchanged unless the run passes ``--cache-bust``.

    Args:
        payload: The materialized request payload (mutated in place); its
            ``messages`` list is the wire prefix.
        benchmark_id: Run-scoped salt so two runs mint different markers.
        trace_instance_id: The trace INSTANCE id (``credit.trace_id``); shared by
            every dispatch of the instance, distinct per instance, fresh on recycle.
        target: Cache-bust mode; ``NONE`` stamps nothing.
    """
    if target == CacheBustTarget.NONE:
        return
    marker = build_trace_instance_marker(benchmark_id, trace_instance_id, target=target)
    messages = payload.get("messages")
    if marker is None or not isinstance(messages, list):
        return
    inject_marker_at_first_user_message(messages, marker)


def uniquify_dynamo_session_headers(
    extra_headers: dict[str, str] | None,
    *,
    trace_instance_id: str | None,
    phase_variant: str,
) -> dict[str, str] | None:
    """Uniquify recorded dynamo session ids per trace REPLAY INSTANCE.

    Build-time lowering stamps the RECORDED ``x-dynamo-session-id`` /
    ``x-dynamo-parent-session-id`` verbatim into the node envelope, but the
    graph replay strategy runs multiple concurrent instances of the SAME trace
    (lanes/recycles wrap the corpus). Forwarded verbatim, every instance would
    share one server-side session: affinity collapses to one bucket and the
    first instance's ``x-dynamo-session-final`` evicts KV under its
    still-running siblings. Mirror of :func:`stamp_cache_bust_marker`'s
    placement -- the worker is the only party that knows the instance id.

    Both identity headers get the SAME deterministic ``#<phase>.<suffix>``
    suffix derived from ``trace_instance_id`` (``t-1#0.0`` -> ``0.0``) and
    ``phase_variant``, so parent-child linkage WITHIN an instance is preserved
    while distinct instances -- and a warmup instance vs the profiling
    instance of the same ``(lane, pass)`` slot -- never collide.
    ``x-dynamo-session-final`` is forwarded untouched: each instance dispatches
    its own copy of the session's last turn, closing only its own session.

    No-op (input returned as-is) when there are no headers, no dynamo identity
    header, or no ``#`` instance suffix on ``trace_instance_id`` -- plain
    (non-instanced) replay is unaffected.

    Args:
        extra_headers: The node envelope's per-node HTTP headers, or ``None``.
        trace_instance_id: The credit's trace INSTANCE id
            (``{template}#{lane}.{pass}``); ``None`` / suffix-less ids disable
            the transform.
        phase_variant: The credit's graph phase variant (``"profiling"`` /
            ``"warmup"``); folded into the suffix so warmup and profiling
            instances of one slot open distinct sessions.

    Returns:
        A NEW dict with the identity headers suffixed, or the input unchanged.
    """
    if not extra_headers or not trace_instance_id:
        return extra_headers
    _, sep, instance_nonce = trace_instance_id.partition("::")
    if not sep:
        return extra_headers
    if not any(header in extra_headers for header in _DYNAMO_SESSION_ID_HEADERS):
        return extra_headers
    tag = f"::{phase_variant}-{instance_nonce}"
    transformed = dict(extra_headers)
    for header in _DYNAMO_SESSION_ID_HEADERS:
        value = transformed.get(header)
        if value is not None:
            transformed[header] = f"{value}{tag}"
    return transformed


def strip_dynamo_session_headers(
    extra_headers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Drop RECORDED dynamo session-identity headers from a node envelope.

    When an active ``--session-routing`` plugin owns session identity, the
    build-time recorded ``x-dynamo-session-id`` / ``-parent-session-id`` /
    ``-session-final`` headers are stale replay artifacts: forwarding them
    alongside the plugin's live headers would put two (case-distinct,
    conflicting) identities on the wire. Non-identity recorded headers are
    preserved. Returns the input unchanged when nothing needs stripping.
    """
    if not extra_headers:
        return extra_headers
    identity = (*_DYNAMO_SESSION_ID_HEADERS, "x-dynamo-session-final")
    if not any(header in extra_headers for header in identity):
        return extra_headers
    stripped = {k: v for k, v in extra_headers.items() if k not in identity}
    return stripped or None


def apply_run_level_payload_options(
    payload: dict[str, Any],
    endpoint: EndpointInfo,
    stream_override: bool | None = None,
    *,
    skip_endpoint_extra: bool = False,
) -> None:
    """Layer run-level endpoint concerns onto a materialized graph payload.

    The materialized payload carries the node's own per-node
    ``dispatch_overrides`` (``model``, the mapped token cap). This layers the
    run-level concerns the verbatim ``raw_payload`` path would otherwise drop
    (the chat endpoint's ``format_payload`` is bypassed for graph credits):

    - ``stream`` is stamped from the RECORDED per-node ``stream_override`` when
      the caller supplies one (weka ``"n"``/``"s"``, dynamo ``ttft_ms``); a
      ``None`` override falls back to the GLOBAL ``endpoint.streaming`` setting.
      The recorded per-node mode wins for graph credits so a recorded ``n``-type
      turn stays non-streaming inside an otherwise-streaming run; the transport
      picks the matching wire mode per-request (``effective_streaming``).
    - ``endpoint.extra`` (the user's ``--extra-inputs`` vendor tunables) is
      merged with the USER winning over any per-node key.
      SKIPPED when ``skip_endpoint_extra`` is set: an adapter that already folded
      the run's ``--extra-inputs`` into the node's ``dispatch_overrides`` at parse
      owns those keys, so re-merging here would clobber the adapter-owned values.
    - ``stream_options.include_usage = True`` is forced when the FINAL stamped
      ``stream`` is on AND ``endpoint.use_server_token_count`` is set, so
      server-side token-count metrics still get usage on graph credits (it keys
      on ``payload.get("stream")`` AFTER the stamp above). Any author-supplied
      ``stream_options`` keys are preserved.

    Args:
        payload: The materialized request payload (mutated in place).
        endpoint: The run's endpoint info carrying the run-level options.
        stream_override: Recorded per-node wire mode for graph credits; ``None``
            follows the global ``endpoint.streaming``.
        skip_endpoint_extra: When set, the ``endpoint.extra`` merge is skipped
            (adapter-owned extras precedence); the ``stream`` stamp and
            ``include_usage`` forcing are unaffected.
    """
    payload["stream"] = (
        stream_override if stream_override is not None else bool(endpoint.streaming)
    )

    if not skip_endpoint_extra:
        for key, value in endpoint.extra or []:
            payload[key] = value

    if payload.get("stream") and endpoint.use_server_token_count:
        _ensure_include_usage(payload)


def _ensure_include_usage(payload: dict[str, Any]) -> None:
    """Force ``stream_options.include_usage = True``, preserving author keys.

    Mirrors ``ChatEndpoint._ensure_include_usage`` so graph credits get the
    same usage opt-in the linear path applies in ``format_payload``.
    """
    stream_options = payload.get("stream_options")
    if not isinstance(stream_options, dict):
        payload["stream_options"] = {"include_usage": True}
        return
    if "include_usage" not in stream_options:
        stream_options["include_usage"] = True


__all__ = [
    "encode_overrides_inner",
    "materialize_graph_request_unified",
    "materialize_graph_request_unified_bytes",
    "apply_run_level_payload_options",
    "stamp_cache_bust_marker",
    "strip_dynamo_session_headers",
    "uniquify_dynamo_session_headers",
]
