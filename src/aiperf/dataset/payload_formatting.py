# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helper for pre-formatting conversation payloads at dataset-load time.

When eligible (single-turn / random_pool / turn-local response-baked multi-turn
with no FORK), the composer walks every conversation once and stamps
``turn.raw_payload`` with ``endpoint.format_payload(...)``. The mmap backing
store then picks ``PAYLOAD_BYTES`` and workers ride the bytes mmap fast path
with zero format_payload work at dispatch time.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    CreditPhase,
    RequestContentType,
)
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import RequestInfo
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


class _SupportsInfo(Protocol):
    """Minimal logger surface used by :func:`preformat_payloads`.

    Any object exposing a callable ``info(msg)`` works — both
    ``AIPerfLoggerMixin`` instances (which accept str or a zero-arg lambda)
    and stdlib ``logging.Logger`` instances satisfy this protocol.
    """

    def info(self, msg: Any) -> None: ...


def is_preformat_eligible(conversation: Conversation) -> bool:
    """Return True if every turn's wire payload can be determined at config time.

    See module docstring for the full contract. False whenever live state
    (worker-captured assistant responses, FORK seeding) would change the
    wire payload at dispatch.

    A ``conversation.context_mode`` of ``None`` is treated as the global
    default ``DELTAS_WITHOUT_RESPONSES`` (matching how every other consumer
    resolves the field).
    """
    # FORK seeds the child from the parent's live turn_list — can't pre-encode.
    # Both directions matter: a parent that declares a FORK branch will have
    # its own turns sent verbatim (so the parent ITSELF could pre-format), but
    # the FORK child it spawns inherits the parent's accumulated context at
    # runtime and re-formats off that. Since dataset-format selection requires
    # ALL turns to have raw_payload (or none), refusing both parent and child
    # keeps the dataset on the CONVERSATION slow path uniformly.
    if any(b.mode == ConversationBranchMode.FORK for b in conversation.branches):
        return False

    # FORK children (non-root with a parent_conversation_id) are seeded from
    # the parent's live turn_list at dispatch — pre-encoding their first turn
    # would freeze an empty seed.
    if not conversation.is_root and conversation.parent_conversation_id is not None:
        return False

    # DELTAS_WITHOUT_RESPONSES folds live assistant captures into the next
    # turn's payload. DELTAS_WITH_RESPONSES still accumulates prior dataset
    # turns via ContentSession.advance_turn(), while pre-formatting is
    # turn-local. Neither mode can safely pre-encode multi-turn payloads.
    # None resolves to DELTAS_WITHOUT_RESPONSES elsewhere; mirror that here.
    effective_mode = (
        conversation.context_mode
        if conversation.context_mode is not None
        else ConversationContextMode.DELTAS_WITHOUT_RESPONSES
    )
    return not (
        effective_mode
        in {
            ConversationContextMode.DELTAS_WITHOUT_RESPONSES,
            ConversationContextMode.DELTAS_WITH_RESPONSES,
        }
        and len(conversation.turns) > 1
    )


def format_conversation_payloads(
    conversations: list[Conversation], model_endpoint: ModelEndpointInfo
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Walk eligible conversations + yield ``(session_id, turn_index, payload)``.

    Skips ineligible conversations silently (caller decides whether to ignore
    the whole conversation or fall back to dispatch-time formatting). Turns
    that already carry a ``raw_payload`` (e.g. populated by the raw_payload
    or inputs_json loaders) are also skipped — they're already in the exact
    shape the worker will send, so re-running the endpoint formatter would
    discard authoring intent and risk a different wire shape.
    Raises ``NotImplementedError`` if the endpoint plugin can't format any
    payload (caller should treat as a global "skip pre-format" signal).
    """
    EndpointClass = plugins.get_class(PluginType.ENDPOINT, model_endpoint.endpoint.type)
    endpoint = EndpointClass(model_endpoint=model_endpoint)

    for conv in conversations:
        if not is_preformat_eligible(conv):
            continue
        for i, turn in enumerate(conv.turns):
            if turn.raw_payload is not None:
                continue
            request_info = RequestInfo(
                model_endpoint=model_endpoint,
                turns=[turn],
                turn_index=i,
                credit_num=i,
                credit_phase=CreditPhase.PROFILING,
                x_request_id="",
                x_correlation_id="",
                conversation_id=conv.session_id,
                system_message=conv.system_message,
                user_context_message=conv.user_context_message,
            )
            yield conv.session_id, i, endpoint.format_payload(request_info)


def preformat_payloads(
    conversations: list[Conversation],
    model_endpoint: ModelEndpointInfo,
    *,
    logger: _SupportsInfo,
) -> None:
    """Stamp ``turn.raw_payload`` on every eligible turn for the mmap fast path.

    Walks every eligible conversation once (see :func:`is_preformat_eligible`)
    and writes the result of ``endpoint.format_payload(...)`` onto
    ``turn.raw_payload`` in-place. ``_select_mmap_format`` then picks
    ``PAYLOAD_BYTES`` for the dataset and workers ride the bytes mmap fast
    path with zero ``format_payload`` work at dispatch time.

    Turns already carrying ``raw_payload`` (populated by the raw_payload /
    inputs_json loaders) are skipped — they're already in the exact shape
    the worker will send. Falls back gracefully (info-logs + returns) when
    the endpoint plugin raises ``NotImplementedError`` (e.g. ``RawEndpoint``
    over un-pre-encoded data) so the slow CONVERSATION path continues
    unchanged.

    Args:
        conversations: The conversations to walk in-place.
        model_endpoint: Endpoint info used to construct the formatter.
        logger: Anything exposing ``info(msg)`` — info messages are emitted
            under the caller's name (composer or DatasetManager).
    """
    if (
        model_endpoint.endpoint.request_content_type
        == RequestContentType.MULTIPART_FORM_DATA
    ):
        logger.info(
            "Skipping payload pre-formatting for multipart/form-data endpoint "
            "(transport must build FormData at dispatch time)."
        )
        return

    turn_lookup: dict[tuple[str, int], Turn] = {
        (conv.session_id, i): turn
        for conv in conversations
        for i, turn in enumerate(conv.turns)
    }
    count = 0
    try:
        for session_id, turn_idx, payload in format_conversation_payloads(
            conversations, model_endpoint
        ):
            turn_lookup[(session_id, turn_idx)].raw_payload = payload
            count += 1
    except NotImplementedError:
        logger.info(
            "Skipping payload pre-formatting "
            "(endpoint does not support format_payload — falling back to "
            "dispatch-time formatting)."
        )
        return

    if count:
        logger.info(
            f"Pre-formatted {count} payloads at config time "
            "(PAYLOAD_BYTES mmap fast path enabled)."
        )
