# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.enums import (
    CacheBustTarget,
    CommAddress,
    CommandType,
    ConversationBranchMode,
    MemoryMapFormat,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.event_loop_monitor import EventLoopMonitor
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import (
    background_task,
    on_command,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    DatasetConfiguredNotification,
    ErrorMessage,
    InferenceResultsMessage,
    WorkerHealthMessage,
)
from aiperf.common.messages.dataset_messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
)
from aiperf.common.mixins import ProcessHealthMixin
from aiperf.common.models import (
    Conversation,
    ErrorDetails,
    MemoryMapClientMetadata,
    ModelEndpointInfo,
    ParsedResponse,
    ProcessHealth,
    RecordContext,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.models.dataset_models import GraphSegmentClientMetadata
from aiperf.common.models.record_models import find_last_non_empty_usage
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
    StreamingDealerClientProtocol,
    StreamingPushClientProtocol,
)
from aiperf.common.scenario.context_overflow import is_context_overflow_response
from aiperf.config.adaptive_scale_phase import (
    sla_filters_require_first_token_observation,
)
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
    GraphTraceEnd,
    RouterToWorkerMessage,
    WorkerReady,
    WorkerShutdown,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient
from aiperf.dataset.memory_map_utils import (
    apply_max_tokens_to_wire_payload,
    turn_from_payload_turn,
)
from aiperf.dataset.protocols import DatasetClientStoreProtocol
from aiperf.graph.dynamic_pool import GraphPoolMissingError
from aiperf.graph.errors import GraphErrorCode, format_graph_error
from aiperf.graph.ids import template_id
from aiperf.graph.worker_materialize import (
    apply_run_level_payload_options,
    materialize_graph_request_unified,
    materialize_graph_request_unified_bytes,
    read_node_envelope,
    stamp_cache_bust_marker,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.payload_retention import resolve_strip_record_payload_bytes
from aiperf.workers.inference_client import InferenceClient
from aiperf.workers.session_manager import UserSession, UserSessionManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.graph.dynamic_pool import GraphCapturedReply, GraphPoolSentinel
    from aiperf.transports.base_transports import FirstTokenCallback


def _mint_x_request_id(credit: Credit) -> str:
    """Fresh per-dispatch request id (``X-Request-ID``), minted per credit.

    Linear credits keep the legacy opaque ``uuid4``. Graph credits mint the
    recoverable-plus-nonce form ``{node_id}::{nonce}``: the node id
    (``{scope}:{turn}``, the trajectory coordinate) is derived from the
    credit's own ``conversation_id`` (``{trace}[::{scope}]``) and
    ``turn_index``, and the nonce keeps the id fresh per dispatch. Safe to
    enrich because ``x_request_id`` is write-only (never read back, matched,
    or deduped) -- it exists so an export row identifies its template node
    without any side lookup.
    """
    if credit.node_ordinal is None or not credit.conversation_id:
        return str(uuid.uuid4())
    conversation = credit.conversation_id
    _, sep, child_scope = conversation.partition("::")
    scope = child_scope if sep else conversation
    return f"{scope}:{credit.turn_index}::{uuid.uuid4().hex}"


def _phase_needs_first_token_callback(phase) -> bool:
    if phase.prefill_concurrency is not None:
        return True
    return bool(
        getattr(phase, "adaptive_scale", False)
        and sla_filters_require_first_token_observation(phase.sla)
    )


_logger = AIPerfLogger(__name__)


def _error_message(error: str | ErrorDetails | None) -> str | None:
    """Extract the human error text for overflow substring matching.

    ``credit_context.error`` usually holds an ``ErrorDetails``; read its
    ``.message`` rather than the model's ``__str__`` repr (which embeds
    ``code=``/``type=``/``cause=`` noise and would silently break substring
    matching if the repr ever changed). Used only for classification -- the
    wire ``CreditReturn.error`` still carries the full ``str(ErrorDetails)`` so
    no diagnostic detail (code/type/cause) is dropped downstream.
    """
    if error is None:
        return None
    return error.message if isinstance(error, ErrorDetails) else str(error)


def _is_terminal_context_overflow(
    credit: Credit, credit_context: CreditContext
) -> bool:
    """Whether this credit return ends the conversation via context overflow.

    A context-overflow error on a non-final, non-cancelled turn is terminal:
    agentic_replay recycles the lane and issues no further (final/cancel) credit
    for the session, so the worker must treat it as a terminal disposition.
    Mirrors the strategy's own ``terminal_overflow`` classification, applied to
    the same error body the worker returns on the credit.
    """
    if credit.is_final_turn or credit_context.cancelled:
        return False
    body = _error_message(credit_context.error)
    return body is not None and is_context_overflow_response(body=body)


def _apply_cache_bust_to_system_message(
    system_message: str | None, marker: str, target: CacheBustTarget
) -> str | None:
    """Apply marker to the structured system_message string.

    Returns the modified string, or `None` if the input was None — the caller
    is then expected to fall back to mutating raw_messages.
    """
    if not marker or target == CacheBustTarget.NONE or system_message is None:
        return system_message
    if target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
    ):
        return marker + system_message
    if target == CacheBustTarget.SYSTEM_SUFFIX:
        return system_message + marker
    return system_message


def _content_has_marker_at_edge(
    content: object, marker: str, *, is_prefix: bool
) -> bool:
    """Whether ``content`` already carries ``marker`` at the prefix/suffix edge.

    The injection helpers run once per credit and several paths mutate a turn
    object shared across the session's turns (delta-mode ``turn_list[0]``, and
    the unconditional every-credit first-user mark used for seeded resumes), so
    re-injecting the constant per-session marker must not stack it. This check
    is exact (the per-session marker is constant; a fresh recycled play sees
    pristine content and injects its own marker). Handles plain-string and
    OpenAI multimodal list-of-parts content.
    """
    if isinstance(content, str):
        return content.startswith(marker) if is_prefix else content.endswith(marker)
    if isinstance(content, list) and content:
        marker_part = {"type": "text", "text": marker.strip()}
        return content[0 if is_prefix else -1] == marker_part
    return False


def _inject_marker_into_raw_messages(
    raw_messages: list[dict], marker: str, *, is_prefix: bool
) -> None:
    """Mutate the first system-role message's content in-place.

    No-op when raw_messages is empty or the first message is not a system role.
    For multimodal content (``content`` is a list of parts), the marker is
    inserted as a new ``{"type": "text", "text": marker}`` part at the start
    (prefix) or end (suffix) of the parts list. Idempotent via
    :func:`_content_has_marker_at_edge`.
    """
    if not raw_messages or not marker:
        return
    first = raw_messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return
    content = first.get("content", "")
    if _content_has_marker_at_edge(content, marker, is_prefix=is_prefix):
        return
    if isinstance(content, str):
        raw_messages[0] = {
            **first,
            "content": (marker + content) if is_prefix else (content + marker),
        }
        return
    if isinstance(content, list):
        marker_part = {"type": "text", "text": marker.strip()}
        new_content = [marker_part, *content] if is_prefix else [*content, marker_part]
        raw_messages[0] = {**first, "content": new_content}
        return
    _logger.warning(
        f"cache-bust: cannot inject marker into raw_messages[0].content of "
        f"type {type(content).__name__}; marker dropped"
    )


def _inject_marker_into_first_user_turn(
    raw_messages: list[dict], marker: str, *, is_prefix: bool
) -> None:
    """Mutate the first user-role message's content in-place.

    No-op when raw_messages is empty. For multimodal content (``content`` is
    a list of parts), the marker is inserted as a new
    ``{"type": "text", "text": marker}`` part at the start (prefix) or end
    (suffix) of the parts list. Idempotent via
    :func:`_content_has_marker_at_edge` — FIRST_TURN_* injection runs every
    credit (to mark seeded turn 0 on mid-trajectory resumes), so repeated calls
    on the same shared turn must not stack the marker.
    """
    if not raw_messages or not marker:
        return
    for idx, msg in enumerate(raw_messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if _content_has_marker_at_edge(content, marker, is_prefix=is_prefix):
                return
            if isinstance(content, str):
                raw_messages[idx] = {
                    **msg,
                    "content": (marker + content) if is_prefix else (content + marker),
                }
                return
            if isinstance(content, list):
                marker_part = {"type": "text", "text": marker.strip()}
                new_content = (
                    [marker_part, *content] if is_prefix else [*content, marker_part]
                )
                raw_messages[idx] = {**msg, "content": new_content}
                return
            _logger.warning(
                f"cache-bust: cannot inject marker into first user-turn content "
                f"of type {type(content).__name__}; marker dropped"
            )
            return


def _find_effective_raw_system(turn_list: list[Turn]) -> list[dict] | None:
    """Return the ``raw_system`` block list the endpoint will actually send.

    Mirrors ``base_endpoint._latest_turn_attr``: walks ``turn_list`` from the
    end and returns the first non-None ``raw_system``. Only ``MessagesEndpoint``
    reads this field, and there it OUTRANKS both the conversation-level
    ``system_message`` string and any system-role ``raw_messages`` entry — so a
    ``SYSTEM_*`` marker written to either of those would be silently dropped
    from the wire whenever a turn carries ``raw_system``.
    """
    for turn in reversed(turn_list):
        if turn.raw_system is not None:
            return turn.raw_system
    return None


def _inject_marker_into_raw_system(
    raw_system: list[dict], marker: str, *, is_prefix: bool
) -> None:
    """Insert the marker as a text block at the edge of a ``raw_system`` list.

    ``raw_system`` is a list of vendor content blocks (not role/content
    messages), so the marker becomes its own ``{"type": "text", ...}`` block
    rather than being spliced into an existing block's text — that leaves any
    neighbouring block's ``cache_control`` breakpoint attached to the same
    bytes it was authored against. Idempotent via
    :func:`_content_has_marker_at_edge`.
    """
    if not marker:
        return
    if _content_has_marker_at_edge(raw_system, marker, is_prefix=is_prefix):
        return
    marker_block = {"type": "text", "text": marker.strip()}
    if is_prefix:
        raw_system.insert(0, marker_block)
    else:
        raw_system.append(marker_block)


def _find_first_system_message(turn_list: list[Turn]) -> list[dict] | None:
    """Return the raw_messages list whose first dict has ``role == "system"``, or None.

    Walks ``turn_list`` forward and returns the first ``raw_messages`` whose
    leading dict is a system-role message. Used by cache-bust system-target
    injection so it works for both single-turn message-array mode (system
    lives in ``turn_list[-1]``, which is also ``turn_list[0]``) and
    accumulating delta mode (system in ``turn_list[0]``, deltas in
    ``turn_list[1..]``).
    """
    for turn in turn_list:
        raw = turn.raw_messages
        if raw and isinstance(raw[0], dict) and raw[0].get("role") == "system":
            return raw
    return None


def _find_first_user_turn(turn_list: list[Turn]) -> Turn | None:
    """Return the first turn whose payload carries the conversation's initial
    user message, or None.

    Walks ``turn_list`` forward. A turn qualifies when it has any
    ``raw_messages`` entry with ``role == "user"``, or when ``texts`` is
    non-empty (synthetic-Turn path). If no turn matches but at least one turn
    has neither ``raw_messages`` nor ``texts`` (truly empty synthetic Turn,
    e.g. before any prompt has been generated), returns that first empty
    turn so a marker-only-text seed path still resolves.
    """
    empty_synthetic: Turn | None = None
    for turn in turn_list:
        if turn.raw_messages:
            for msg in turn.raw_messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return turn
        elif turn.texts:
            return turn
        elif empty_synthetic is None:
            empty_synthetic = turn
    return empty_synthetic


def _inject_marker_into_first_user_text(
    turn: Turn, marker: str, *, is_prefix: bool
) -> None:
    """Mutate the first ``Text.contents[0]`` on a structured Turn (synthetic-Turn path).

    Used as a fallback when ``Turn.raw_messages`` is None and the endpoint
    formatter would synthesise the user message from ``Turn.texts``. If the
    Turn has no ``texts`` entries, prepends one whose content is the marker
    alone (becomes the entire turn body — fine because there was nothing else
    to merge with).
    """
    if not marker:
        return
    # Seed with the full marker (including ``\n\n``) so
    # ``_content_has_marker_at_edge`` idempotency matches the string path.
    # Multimodal list-of-parts injection still uses ``marker.strip()`` when
    # comparing/inserting text parts.
    if not turn.texts:
        turn.texts = [Text(contents=[marker])]
        return
    first = turn.texts[0]
    if not first.contents:
        first.contents = [marker]
        return
    existing = first.contents[0]
    if _content_has_marker_at_edge(existing, marker, is_prefix=is_prefix):
        return
    first.contents[0] = (marker + existing) if is_prefix else (existing + marker)


def _inject_marker_at_first_user(
    turn_list: list[Turn], marker: str, *, is_prefix: bool
) -> None:
    """Inject ``marker`` at the first user turn (raw_messages or texts).

    Wraps the lookup + dispatch shared by SYSTEM_* fallback (sub-path 3
    in :func:`_apply_cache_bust`) and the FIRST_TURN_* path. No-op when
    there is no user-bearing turn at all.
    """
    user_turn = _find_first_user_turn(turn_list)
    if user_turn is None:
        return
    if user_turn.raw_messages:
        _inject_marker_into_first_user_turn(
            user_turn.raw_messages, marker, is_prefix=is_prefix
        )
    else:
        _inject_marker_into_first_user_text(user_turn, marker, is_prefix=is_prefix)


def _apply_cache_bust(
    session: UserSession,
    credit: Credit,
    system_message: str | None,
) -> str | None:
    """Dispatch cache-bust marker injection for a single credit.

    Mutates the appropriate turn's ``raw_messages`` (or ``texts``) in-place
    when the marker attaches to the trace's pre-rendered messages. Returns
    the (possibly modified) ``system_message`` string for the caller to
    forward into request building.

    The system / first-user lookups walk ``turn_list`` forward rather than
    indexing ``[-1]``, so this works under both ``MESSAGE_ARRAY_WITH_RESPONSES``
    (single-turn ``turn_list``) and ``DELTAS_WITH_RESPONSES`` (accumulating
    ``turn_list`` where the system role lives in ``turn_list[0]`` and later
    deltas start with the prior assistant response).

    Injection targets the *effective wire prefix* (see
    :func:`_effective_prefix_turns`): the slice of ``turn_list`` that
    ``build_messages`` actually emits. A ``reset_context`` turn makes
    ``build_messages`` discard every prior turn, so the effective prefix begins
    at the last such turn — which may sit mid-history (seeded on a resume,
    never dispatched as the current turn), not just at ``turn_list[-1]``.
    Marking the discarded turn 0 instead would leave the real prefix unmarked
    and let recycled plays warm the server's cache on identical post-reset
    bytes. ``SYSTEM_*`` targets are handled in
    :func:`_apply_system_target_cache_bust`.
    """
    marker = credit.cache_bust_marker
    target = credit.cache_bust_target

    if not marker or target == CacheBustTarget.NONE:
        return system_message

    # FORK children share the parent's KV cache by design: they seed turn_list
    # from the parent (the SAME Turn objects) and must send the parent's exact
    # prefix to hit its cache. The parent already injected its marker into those
    # shared turns, so the child inherits it for free — re-busting here would
    # diverge the child's prefix from the parent's (cache miss) AND mutate the
    # parent's shared, read-only Turn objects (stacking markers). So cache-bust
    # is a no-op for FORK children. SPAWN children start fresh (no shared turns)
    # and root sessions own their prefix, so both are busted normally.
    if (
        session.parent_correlation_id is not None
        and session.branch_mode == ConversationBranchMode.FORK
    ):
        return system_message

    is_prefix = target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
        CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
    )
    prefix_turns = _effective_prefix_turns(session)

    if target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.SYSTEM_SUFFIX,
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
    ):
        return _apply_system_target_cache_bust(
            prefix_turns,
            all_turns=session.turn_list,
            system_message=system_message,
            marker=marker,
            target=target,
            is_prefix=is_prefix,
        )

    # Mark the effective prefix's opening user turn every credit (idempotent).
    # Unconditional rather than turn_index==0-gated so a seeded mid-trajectory
    # resume (turn_list back-filled with turns 0..k_i at a credit whose
    # turn_index > 0) still marks the true wire prefix.
    _inject_marker_at_first_user(prefix_turns, marker, is_prefix=is_prefix)
    return system_message


def _apply_system_target_cache_bust(
    prefix_turns: list[Turn],
    *,
    all_turns: list[Turn],
    system_message: str | None,
    marker: str,
    target: CacheBustTarget,
    is_prefix: bool,
) -> str | None:
    """Inject a ``SYSTEM_PREFIX`` / ``SYSTEM_SUFFIX`` / ``WARMUP_ISOLATION_SYSTEM`` marker for one credit.

    ``prefix_turns`` is the effective wire prefix slice (see
    :func:`_effective_prefix_turns`). Four sub-paths, ordered to match how the
    endpoint resolves the system field on the wire — marking a lower-precedence
    carrier would leave the bytes that actually ship unchanged:
      1. ``raw_system`` present on any turn: it outranks both ``system_message``
         and system-role ``raw_messages`` in ``MessagesEndpoint``, so the marker
         goes there. Resolved over ``all_turns`` (not the prefix slice) because
         ``_latest_turn_attr`` ignores ``reset_context``.
      2. Conversation-level ``system_message`` present: marker applied every
         turn (string mutation re-applied per credit). Unaffected by
         ``reset_context`` — the ``system_message`` rides on ``RequestInfo`` and
         is re-emitted every turn independent of ``build_messages``' reset.
      3. ``raw_messages`` first dict has ``role=="system"``: marker injected
         into the first system message of the prefix slice.
      4. No system anywhere -> first-user-turn fallback: marker injected
         into the first user turn every credit (idempotent), matching
         ``FIRST_TURN_*`` semantics so a seeded mid-trajectory resume still
         marks the prefix.

    Returns the (possibly modified) ``system_message``.
    """
    raw_system_blocks = _find_effective_raw_system(all_turns)
    if raw_system_blocks is not None:
        _inject_marker_into_raw_system(raw_system_blocks, marker, is_prefix=is_prefix)
        return system_message
    if system_message is not None:
        return _apply_cache_bust_to_system_message(system_message, marker, target)
    raw_system = _find_first_system_message(prefix_turns)
    if raw_system is not None:
        _inject_marker_into_raw_messages(raw_system, marker, is_prefix=is_prefix)
    else:
        _inject_marker_at_first_user(prefix_turns, marker, is_prefix=is_prefix)
    return system_message


def _effective_prefix_turns(session: UserSession) -> list[Turn]:
    """The ``turn_list`` slice that forms the wire prefix for cache-bust.

    ``base_endpoint.build_messages`` restarts the message array at every
    ``reset_context`` turn that carries ``raw_messages`` (discarding everything
    before it), so the effective prefix begins at the *last* such turn in
    ``turn_list`` — not turn 0, and not merely ``turn_list[-1]``: a reset can
    sit mid-history (e.g. seeded into a mid-trajectory resume, where it is never
    dispatched as the current turn). Returns the slice from that turn to the
    end, or the whole ``turn_list`` when there is no reset.
    """
    turns = session.turn_list
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].reset_context and turns[i].raw_messages is not None:
            return turns[i:]
    return turns


class Worker(BaseComponentService, ProcessHealthMixin):
    """Worker processes credits from the TimingManager and makes API calls to inference servers.

    Responsibilities:
    - Receives credits via DEALER socket from StickyCreditRouter
    - Processes individual turns (1 credit = 1 turn) with session caching for sticky routing
    - Manages conversation state and assistant responses across turns
    - Sends inference results to RecordProcessor for metric calculation
    - Reports health and task statistics to WorkerManager

    Architecture:

      ┌────────────────────┐
      │ StickyCreditRouter │
      │   (ROUTER socket)  │
      └────┬──────────▲────┘
           │          │
        Credit   CreditReturn
           │          │
           ▼          │  ┌─── RequestRecord ──▶ RecordProcessor
      ┌────────────────────┐
      │  Worker (DEALER)   │
      │                    │
      │ 1. Check cache     │
      │ 2. Advance session │
      │ 3. Build request   │
      └────┬──────────▲────┘
           │          │
           ▼          │
      ┌────────────────────┐
      │  InferenceClient   │
      │  (HTTP/streaming)  │
      └────┬──────────▲────┘
           │          │
           ▼          │
      ┌────────────────────┐
      │  Inference Server  │
      │   (vLLM, TRT-LLM)  │
      └────────────────────┘

    Credit Flow (All Modes):
    ═══════════════════════════════════════════════════════════════════════════
    1. Credit arrives with x_correlation_id (shared across all turns)
    2. Check session cache:
       - Cache HIT:  Reuse session → Sticky routing working!
       - Cache MISS: Fetch conversation → Create & cache session
    3. Advance session to credit.turn_index
    4. Process single turn, return credit immediately
    5. If final_turn: Evict session from cache

    Example timeline for 3-turn conversation:
    T1: credit[turn=0, x_corr=ABC] → cache MISS → fetch & cache session → return
    T2: credit[turn=1, x_corr=ABC] → cache HIT  → reuse session → return
    T3: credit[turn=2, x_corr=ABC] → cache HIT  → reuse session → evict → return
        └─▶ Same worker processes all turns (StickyCreditRouter sticky routing)

    Session Lifecycle:
    - First turn: Create session from DatasetManager, cache by x_correlation_id
    - Subsequent turns: Retrieve from cache, advance to turn_index
    - Final turn: Process and evict from cache
    - StickyCreditRouter ensures all turns route to same worker for cache hits
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        self.debug(lambda: f"Worker process __init__ (pid: {self._process.pid})")

        self.event_loop_monitor = EventLoopMonitor(self.service_id)

        self.task_stats: WorkerTaskStats = WorkerTaskStats()

        self.credit_tasks: dict[int, asyncio.Task] = {}
        # Worker-local dynamic-content pool for graph traces (captured
        # assistant responses). Engaged only when a
        # node envelope carries `capture` -- inert for recorded corpora.
        from aiperf.graph.dynamic_pool import GraphDynamicPool

        self._graph_dynamic_pool = GraphDynamicPool(
            max_bytes=Environment.GRAPH.DYNAMIC_POOL_MAX_BYTES
        )

        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )

        self.model_endpoint = ModelEndpointInfo.from_run(self.run)

        self.inference_client: InferenceClient = InferenceClient(
            model_endpoint=self.model_endpoint,
            service_id=self.service_id,
            strip_record_payload_bytes=resolve_strip_record_payload_bytes(
                self.run.cfg, self.model_endpoint
            ),
        )
        self.attach_child_lifecycle(self.inference_client)
        self.debug(
            lambda: (
                f"Created inference client for {self.model_endpoint.endpoint.type}, "
                f"class: {self.inference_client.__class__.__name__}"
            ),
        )

        # Identity must be unique - ZMQ ROUTER uses it to address messages to specific
        # DEALERs. The sticky router tracks workers by this identity.
        self.credit_dealer_client: StreamingDealerClientProtocol = (
            self.comms.create_streaming_dealer_client(
                address=CommAddress.CREDIT_ROUTER,
                identity=self.service_id,
                bind=False,
            )
        )
        self.credit_dealer_client.register_receiver(self._on_credit_message)

        # Dual-channel returns: CreditReturn/FirstToken go out a dedicated typed
        # PUSH -> router PULL fan-in instead of back on the bidirectional credit
        # DEALER, so the dispatch DEALER is receive-only (no shared-FD send/recv
        # contention). WorkerReady/WorkerShutdown still go on the DEALER so the
        # ROUTER registers/tracks identity. Returns carry worker_id in-message
        # since PUSH/PULL has no ZMQ envelope identity.
        self.credit_return_push_client: StreamingPushClientProtocol = (
            self.comms.create_streaming_push_client(
                CommAddress.CREDIT_RETURN,
                bind=False,
            )
        )

        self.memory_usage_before_profiling: float | None = None

        self.session_manager: UserSessionManager = UserSessionManager()

        # Dataset client for direct data access (eliminates DatasetManager bottleneck)
        # Initialized when DatasetConfiguredNotification is received via factory
        self._dataset_client: DatasetClientStoreProtocol | None = None
        self._dataset_configured_event = asyncio.Event()
        # True when the mmap dataset ships pre-encoded per-turn payload bytes
        # (PAYLOAD_BYTES format); enables the verbatim-replay fast path.
        self._is_payload_bytes: bool = False

        # Caches a FAILED store open (missing/corrupt files) so the worker does
        # not re-attempt the doomed open on every graph credit (F4): a single
        # fatal, actionable error is surfaced per credit instead of a silent
        # retry-loop that looks like a hang.
        self._graph_store_open_error: ErrorDetails | None = None
        # Worker-side reader for the graph unified segment store (content pool +
        # per-node manifests) -- the SOLE graph store shape; every graph build
        # writes one. Opened lazily on the first graph credit from the SAME
        # (base_path, benchmark_id) the build-time GraphSegmentUnifiedBackingStore
        # wrote to; ``_graph_unified_open_attempted`` guards a single open.
        self._graph_unified_client: GraphSegmentUnifiedClient | None = None
        self._graph_unified_open_attempted = False
        # Why the unified open failed, when the failure is actionable (an
        # on-disk store REJECTED as pre-v3 by the A2-strict reader, vs simply
        # absent). Folded into the fatal GraphStoreUnavailable message so
        # the operator sees "re-parse required" instead of a misleading
        # "neither store exists".
        self._graph_unified_open_failure: str | None = None
        # Graph store location from the dataset broadcast (the graph-typed
        # DatasetConfiguredNotification). The worker opens the unified store
        # from THIS, never from env conventions -- absence is a recorded failure
        # feeding the GraphStoreUnavailable fatal, not a temp-dir fallback.
        self._graph_client_metadata: GraphSegmentClientMetadata | None = None

        # Detecting first token requires parsing each SSE chunk, so only enable
        # FirstToken messages when a downstream consumer needs them. Graph credits
        # additionally opt in per-credit via ``credit.first_token_event`` (checked
        # at dispatch) for post-TTFT first-token anchoring even when no phase-level
        # consumer is active. ``_phase_needs_first_token_callback`` already covers
        # prefill-concurrency limiting.
        self._prefill_concurrency_enabled: bool = any(
            _phase_needs_first_token_callback(phase) for phase in self.run.cfg.phases
        )

        # One-shot warning gate so cache-bust diagnostics don't spam logs at
        # high concurrency — the misconfiguration is the same for every credit.
        self._cache_bust_warning_shown: bool = False

        # Only used as a fallback when dataset client is not initialized
        # or was not available when the credit was dropped. Must be created here
        # so it can be attached to the worker lifecycle.
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                address=CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
                bind=False,
            )
        )

    @on_start
    async def _send_worker_ready_message(self) -> None:
        """Send WorkerReady to announce presence."""
        await self.credit_dealer_client.send(WorkerReady(worker_id=self.service_id))

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(self, msg: DatasetConfiguredNotification) -> None:
        """Initialize dataset client when configuration is received.

        Uses factory pattern to dynamically create the appropriate client.
        The factory auto-extracts client_type from client_metadata, leveraging
        the discriminated union pattern for type-safe routing. This allows new
        storage backends (S3, Redis, etc.) to work without modifying Worker code.
        """
        ClientStoreClass = plugins.get_class(
            PluginType.DATASET_CLIENT_STORE, msg.client_metadata.client_type
        )
        self._dataset_client = ClientStoreClass(client_metadata=msg.client_metadata)
        await self._dataset_client.initialize()
        # Graph runs carry the unified-store location on the broadcast; capture
        # it so ``_graph_unified_reader`` opens exactly what the DatasetManager
        # built (no env re-derivation). Non-graph broadcasts leave this None.
        self._graph_client_metadata = (
            msg.client_metadata
            if isinstance(msg.client_metadata, GraphSegmentClientMetadata)
            else None
        )
        self.session_manager.set_default_context_mode(msg.metadata.default_context_mode)
        if isinstance(msg.client_metadata, MemoryMapClientMetadata):
            self._is_payload_bytes = (
                msg.client_metadata.format == MemoryMapFormat.PAYLOAD_BYTES
            )
            if (
                self._is_payload_bytes
                and self.run.cfg.get_cache_bust_target() != CacheBustTarget.NONE
            ):
                raise RuntimeError(
                    "cache-bust is incompatible with PAYLOAD_BYTES fast path; "
                    "loader should have skipped preformat "
                    "(see DatasetManager._preformat_payloads)"
                )
        self._dataset_configured_event.set()
        self.debug(
            lambda: (
                f"Dataset client initialized: type={msg.client_metadata.client_type}"
            )
        )

    @on_stop
    async def _send_worker_shutdown_message(self) -> None:
        """Send WorkerShutdown to announce shutdown."""
        try:
            await self.credit_dealer_client.send(
                WorkerShutdown(worker_id=self.service_id)
            )
            self.debug(
                lambda: (
                    f"Sent WorkerShutdown for graceful disconnect ({self.service_id})"
                )
            )
        except Exception as e:
            self.warning(
                f"Failed to send shutdown message (already disconnected?): {e!r}"
            )

    @background_task(
        immediate=False,
        interval=Environment.WORKER.HEALTH_CHECK_INTERVAL,
    )
    async def _health_check_task(self) -> None:
        """Task to report the health of the worker to the worker manager."""
        health = await asyncio.to_thread(self.get_process_health)
        await self.publish(self.create_health_message(health))

    def create_health_message(self, health: ProcessHealth) -> WorkerHealthMessage:
        return WorkerHealthMessage(
            service_id=self.service_id,
            health=health,
            task_stats=self.task_stats,
        )

    async def _on_credit_message(self, message: RouterToWorkerMessage) -> None:
        """Handle incoming credit message from TimingManager via StickyCreditRouter."""
        match message:
            case Credit():
                self._schedule_credit_drop_task(message)
            case CancelCredits():
                await self._on_cancel_credits_message(message)
            case GraphTraceEnd():
                # Sticky-lifecycle close: evict the trace's dynamic pool entry
                # (deferred while its credits are still in flight -- their
                # capture writes may land after this message on cancelled
                # paths).
                self._graph_dynamic_pool.trace_end(message.trace_id)
            case _:
                self.warning(
                    f"Unknown credit message type: {message.__class__.__name__}"
                )

    def _schedule_credit_drop_task(self, credit: Credit) -> None:
        """Schedule a task to handle the credit drop message from TimingManager via StickyCreditRouter.

        This method creates the credit context outside the task so it's available to the done callback.
        This simply schedules the task to be executed asynchronously and adds a done callback to
        ensure the credit is returned. It does not wait for it to actually execute.
        """
        drop_perf_ns = time.perf_counter_ns()
        credit_context = CreditContext(
            credit=credit,
            drop_perf_ns=drop_perf_ns,
        )

        task = self.execute_async(self._on_credit_drop_message_task(credit_context))
        self.credit_tasks[credit.id] = task
        task.add_done_callback(
            lambda t, ctx=credit_context: self._on_credit_drop_message_task_done(t, ctx)
        )

    def _on_credit_drop_message_task_done(
        self, task: asyncio.Task, credit_context: CreditContext
    ) -> None:
        """Handle credit task completion - ensure credit is ALWAYS returned.

        This callback runs when a credit task finishes, whether it completed normally,
        was cancelled, or errored. For cancelled tasks that never started executing,
        the finally block never runs, so we must return the credit here.
        """
        credit_id = credit_context.credit.id

        # Always remove from tracking dict when task completes
        self.credit_tasks.pop(credit_id, None)

        # The finally block handles normal/error returns. This callback only needs
        # to return credits for tasks that were cancelled before they started executing.
        if credit_context.returned:
            # Clear references explicitly since GC is disabled during profiling
            credit_context.credit = None
            credit_context.error = None
            return

        # Credit was NOT returned - this means the task was cancelled before it started
        # or failed in some way that prevented the finally block from sending the return
        self.debug(
            lambda id=credit_id: (
                f"Credit {id} task done but NOT returned! "
                f"Task likely was cancelled before finally block could execute. Returning now."
            )
        )

        # Update credit_context with cancellation status
        credit_context.cancelled = credit_context.cancelled or task.cancelled()

        # Build and send return message (synchronous context, need to schedule send)
        credit_return = CreditReturn(
            credit=credit_context.credit,
            cancelled=credit_context.cancelled,
            first_token_sent=credit_context.first_token_sent,
            error=str(credit_context.error) if credit_context.error else None,
            request_latency_ns=credit_context.request_latency_ns,
            inter_token_latency_ns=credit_context.inter_token_latency_ns,
            output_sequence_length=credit_context.output_sequence_length,
            worker_id=self.service_id,
        )
        self.execute_async(self.credit_return_push_client.send(credit_return))
        credit_context.returned = True

        # Explicitly clear references to help refcounting (GC is disabled on workers)
        credit_context.credit = None
        credit_context.error = None

    async def _on_cancel_credits_message(self, message: CancelCredits) -> None:
        """Handle incoming cancel credits message from TimingManager via StickyCreditRouter."""
        self.debug(
            lambda: f"Received cancel credits message: credit_ids={message.credit_ids}"
        )
        for credit_id in message.credit_ids:
            if task := self.credit_tasks.get(credit_id):
                task.cancel()
            else:
                self.debug(
                    lambda id=credit_id: (
                        f"Task for credit {id} not found (already completed?)"
                    )
                )

    async def _on_credit_drop_message_task(self, credit_context: CreditContext) -> None:
        """Handle incoming credit from TimingManager via StickyCreditRouter.

        Flow:
        1. Process single turn:
           - Check session cache by x_correlation_id
           - If cache miss: Fetch conversation and create session
           - Advance session to turn_index
           - Send request to inference server
        2. ALWAYS return credit in finally block, regardless of success/failure

        Credit return is guaranteed via finally block to ensure accurate concurrency tracking.
        For tasks cancelled before they start, the done callback handles the return.
        """
        try:
            if not self.inference_client:
                raise NotInitializedError("Inference server client not initialized.")
            # Defensive: no_request credits are normally short-circuited at the
            # router and never reach a worker; this guarantees no HTTP even if one
            # ever arrives here. The finally block emits a normal CreditReturn.
            if credit_context.credit.no_request:
                return
            await self._process_credit(credit_context)
        except Exception as e:
            self.exception(
                f"Error occurred while processing credit {credit_context.credit.id}: {e!r}"
            )
        except asyncio.CancelledError:
            self.debug(lambda: f"Credit {credit_context.credit.id} cancelled")
            credit_context.cancelled = True
        finally:
            # Lockstep: a credit returned as completed MUST be accompanied by a
            # record, or the RecordsManager completion barrier (which has no
            # timeout) hangs waiting for it. If processing failed before any
            # record was emitted, forward an error record now. Cancelled credits
            # are excluded from the barrier target, so they need no record.
            #
            # The emit MUST NOT abort the credit return below: if it raised, the
            # finally would exit early, ``returned`` would stay False, and the
            # done callback would return the credit as completed anyway (no
            # record emitted) -- the exact lockstep break this guards against.
            # Contain the failure and always proceed to return the credit.
            #
            # A no_request virtual orchestrator credit legitimately produces no
            # record (it sends no HTTP request) and is excluded from the billable
            # request count, so — like a cancelled credit — it must be exempt from
            # the lockstep failure-record emit, else it returns with a spurious
            # error and the records barrier waits for a record that never comes.
            if (
                not credit_context.cancelled
                and not credit_context.record_emitted
                and not credit_context.credit.no_request
            ):
                try:
                    await self._emit_credit_failure_record(credit_context)
                except Exception as e:
                    self.exception(
                        f"Failed to emit lockstep failure record for credit "
                        f"{credit_context.credit.id}: {e!r}"
                    )
            # ALWAYS return the credit here to ensure accurate tracking
            credit_return = CreditReturn(
                credit=credit_context.credit,
                cancelled=credit_context.cancelled,
                first_token_sent=credit_context.first_token_sent,
                error=str(credit_context.error) if credit_context.error else None,
                request_latency_ns=credit_context.request_latency_ns,
                inter_token_latency_ns=credit_context.inter_token_latency_ns,
                output_sequence_length=credit_context.output_sequence_length,
                worker_id=self.service_id,
            )
            await self.credit_return_push_client.send(credit_return)
            # Mark as returned AFTER send succeeds
            # If send fails/cancelled, done callback will retry
            # The router does NOT de-duplicate returns (a second one would
            # double-count), so only the done callback -- which checks
            # ``returned`` -- may resend.
            credit_context.returned = True
            # Note: Don't null credit_context.credit here - done callback needs
            # credit.id for cleanup. Done callback handles all reference clearing.

    async def _emit_credit_failure_record(self, credit_context: CreditContext) -> None:
        """Forward an error record for a credit whose processing failed before
        any record was emitted.

        Mirrors the conversation-retrieval error path so the records-side count
        stays in lockstep with the credit the worker is about to return as
        completed; without it the RecordsManager completion barrier (no timeout)
        hangs waiting for a record that never arrives.
        """
        credit = credit_context.credit
        err = credit_context.error
        if isinstance(err, ErrorDetails):
            error = err
        elif isinstance(err, str) and err:
            error = ErrorDetails(message=err, type="CreditProcessingError", code=500)
        else:
            error = ErrorDetails(
                message="Credit processing failed before a record was produced",
                type="CreditProcessingError",
                code=500,
            )
        await self._send_inference_result_message(
            RequestRecord(
                request_info=RecordContext(
                    conversation_id=credit.conversation_id,
                    turn_index=credit.turn_index,
                    credit_num=credit.id,
                    credit_phase=credit.phase,
                    phase_index=credit.phase_index,
                    profiling_index=credit.profiling_index,
                    phase_name=credit.phase_name,
                    phase_kind=credit.phase_kind,
                    x_request_id=str(uuid.uuid4()),
                    x_correlation_id=credit.x_correlation_id,
                    agent_depth=credit.agent_depth,
                    parent_correlation_id=credit.parent_correlation_id,
                ),
                model_name=self.model_endpoint.primary_model_name,
                timestamp_ns=time.time_ns(),
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=error,
            )
        )
        # Surface the error on the credit so the subsequent CreditReturn carries
        # it and increment_returned(..., errored=...) counts it. Without this the
        # forwarded error record would not be reflected in the phase-complete
        # request_errors log line.
        credit_context.error = error
        credit_context.record_emitted = True

    async def _process_credit(self, credit_context: CreditContext) -> None:
        """Process a credit (1 credit = 1 request).

        Orchestrates error handling and session eviction for both paths:
        - **Payload bytes fast path**: pre-encoded bytes from mmap, bypasses
          session/conversation deserialization entirely.
        - **Normal path**: session-based conversation handling with turn
          accumulation and response storage.

        Credit return is guaranteed by the caller (_on_credit_drop_message_task).
        """
        x_correlation_id = credit_context.credit.x_correlation_id
        credit = credit_context.credit
        x_request_id = _mint_x_request_id(credit)

        first_token_callback = self._make_first_token_callback(credit_context)

        # Agent-graph credits carry a trace_id/node_ordinal: the worker rebuilds the
        # node's request from the shared graph store mmap (D1) instead of taking
        # the linear session-cache path. Any worker can serve any node.
        if credit.trace_id is not None:
            await self._process_graph_credit(
                credit_context, x_request_id, first_token_callback
            )
            return

        try:
            if await self._try_payload_bytes_fast_path(
                credit_context, x_request_id, first_token_callback
            ):
                return

            # Normal path: session-based conversation handling (cache-bust
            # marker injection, DAG fork/spawn seeding, response accumulation).
            await self._process_credit_with_session(
                credit_context, x_request_id, x_correlation_id, first_token_callback
            )

        except asyncio.CancelledError:
            credit_context.cancelled = True
            raise
        except Exception as e:
            credit_context.error = ErrorDetails.from_exception(e)
            self.exception(f"Error processing credit: {e!r}")
        finally:
            self._handle_terminal_disposition(credit, credit_context, x_correlation_id)

    def _make_first_token_callback(
        self, credit_context: CreditContext
    ) -> FirstTokenCallback | None:
        """Build first-token callback when prefill concurrency limiting is active.

        Detecting first token requires parsing each SSE chunk, so this overhead
        is skipped when the orchestrator doesn't need TTFT events for slot management.

        Graph credits additionally opt in per-credit via
        ``credit.first_token_event`` (post-TTFT first-token anchoring) even when
        no phase-level consumer is active.

        Returns:
            Callback that sends FirstToken to the router on meaningful content,
            or None when neither consumer needs it.
        """
        credit = credit_context.credit
        if not self._prefill_concurrency_enabled and not credit.first_token_event:
            return None

        async def on_first_token(ttft_ns: int, message: SSEMessage) -> bool:
            parsed = self.inference_client.endpoint.parse_response(message)
            if parsed is None or parsed.data is None:
                return False

            await self.credit_return_push_client.send(
                FirstToken(
                    credit_id=credit.id,
                    phase=credit.phase,
                    phase_index=credit.phase_index,
                    ttft_ns=ttft_ns,
                    trace_id=credit.trace_id,
                    x_correlation_id=credit.x_correlation_id,
                    turn_index=credit.turn_index,
                )
            )
            credit_context.first_token_sent = True
            return True

        return on_first_token

    async def _process_credit_with_session(
        self,
        credit_context: CreditContext,
        x_request_id: str,
        x_correlation_id: str,
        first_token_callback: FirstTokenCallback | None,
    ) -> None:
        """Normal credit path: session-based conversation handling.

        Flow:
        1. Check session cache using x_correlation_id:
           - Cache hit: Reuse session (enables conversation caching on inference server)
           - Cache miss: Retrieve conversation from DatasetManager, create new session
        2. Advance session to current turn index
        3. Build RequestInfo from session state and send request
        4. Store assistant response in session for multi-turn accumulation

        Session Lifecycle:
        - First turn: Session created and cached under x_correlation_id
        - Subsequent turns: Retrieved from cache (sticky routing ensures same worker)
        - Final turn: Evicted by caller (_process_credit) in its finally block
        """
        credit = credit_context.credit
        session = self.session_manager.get(x_correlation_id)
        if session is None:
            _conversation = await self._retrieve_conversation_for_session(
                credit_context=credit_context,
            )
            session = self.session_manager.create_and_store(
                x_correlation_id,
                _conversation,
                credit.num_turns,
                url_index=credit.url_index,
                parent_correlation_id=credit.parent_correlation_id,
                branch_mode=credit.branch_mode,
            )
            self._pin_parent_if_fork_child(credit, x_correlation_id)
            self._seed_from_parent_if_fork_child(credit, x_correlation_id)

        session.advance_turn(credit.turn_index)

        system_message = _apply_cache_bust(
            session,
            credit,
            session.conversation.system_message,
        )
        self._maybe_warn_cache_bust_silent_drop(session, credit)

        request_info = self._create_request_info(
            session=session,
            credit_context=credit_context,
            x_request_id=x_request_id,
            system_message=system_message,
            user_context_message=session.conversation.user_context_message,
        )
        record: RequestRecord = await self._execute_request(
            credit_context, request_info, first_token_callback
        )

        self._finalize_session_response(session, credit_context, record)

    async def _try_payload_bytes_fast_path(
        self,
        credit_context: CreditContext,
        x_request_id: str,
        first_token_callback: FirstTokenCallback | None,
    ) -> bool:
        """Serve the credit from pre-encoded mmap payload bytes when possible.

        Payload bytes fast path: stream pre-encoded bytes from the mmap
        verbatim, bypassing session/conversation deserialization and
        per-request payload formatting entirely. Skipped for DAG descendants
        (agent_depth > 0), whose turn accumulation needs session state, and for
        credits carrying a ``max_tokens_override``: the override cannot be
        reflected in the pre-encoded bytes, so serving them verbatim would send
        the baked-in cap while the enriched metric turn reports the override,
        yielding a spurious OSL mismatch. Those defer to the session path, which
        formats the payload with the override applied.

        Returns:
            True when the request was fully handled (sent + result emitted);
            False when the caller should take the normal session path.
        """
        credit = credit_context.credit
        if (
            not self._is_payload_bytes
            or self._dataset_client is None
            or credit.agent_depth != 0
            or credit.max_tokens_override is not None
        ):
            return False
        payload_turn = await self._dataset_client.get_payload_turn(
            credit.conversation_id, credit.turn_index
        )
        if payload_turn is None:
            return False

        self.task_stats.total += 1
        request_info = self._create_request_info(
            credit_context=credit_context,
            x_request_id=x_request_id,
            payload_bytes=payload_turn.payload_bytes,
            # Scalars only — payload_bytes carries the wire body; enrichment
            # reads max_tokens / timestamp off this turn for OSL-mismatch and
            # schedule-lag metrics.
            turns=[
                Turn(
                    role="user",
                    max_tokens=payload_turn.max_tokens,
                    timestamp=payload_turn.timestamp,
                )
            ],
        )
        record = await self.inference_client.send_request(
            request_info, first_token_callback=first_token_callback
        )
        self._populate_response_metrics(credit_context, record)
        await self._send_inference_result_message(record)
        credit_context.record_emitted = True
        if record.error is not None:
            credit_context.error = record.error
        return True

    def _process_responses_for_record(
        self,
        record: RequestRecord,
        *,
        capture_assistant_turn: bool,
    ) -> tuple[list[ParsedResponse], Turn | None]:
        """Process responses while preserving protocol-only third-party endpoints.

        The fallback supports ``EndpointProtocol`` implementations that do not
        inherit ``BaseEndpoint`` and therefore lack its optional replay helpers.
        """
        endpoint = self.inference_client.endpoint
        process_responses = getattr(endpoint, "process_responses", None)
        if callable(process_responses):
            return process_responses(
                record, capture_assistant_turn=capture_assistant_turn
            )

        parsed_responses = endpoint.extract_response_data(record)
        build_assistant_turn = getattr(endpoint, "build_assistant_turn", None)
        assistant_turn = (
            build_assistant_turn(record)
            if capture_assistant_turn and callable(build_assistant_turn)
            else None
        )
        return parsed_responses, assistant_turn

    def _finalize_session_response(
        self,
        session: UserSession,
        credit_context: CreditContext,
        record: RequestRecord,
    ) -> None:
        """Store the assistant turn (when retained) and populate metrics from a
        single response-processing pass, shared with the payload-bytes path."""
        parsed_responses, assistant_turn = self._process_responses_for_record(
            record,
            capture_assistant_turn=session.should_store_response(),
        )
        if assistant_turn is not None:
            session.store_response(assistant_turn)
        self._populate_response_metrics(credit_context, record, parsed_responses)

    def _populate_response_metrics(
        self,
        credit_context: CreditContext,
        record: RequestRecord,
        parsed_responses: list[ParsedResponse] | None = None,
    ) -> None:
        """Derive latency/OSL/ITL from ``record`` onto ``credit_context``.

        Shared by the normal session path and the payload-bytes fast path so
        both emit identical CreditReturn timing fields from the same record.
        """
        if parsed_responses is None:
            parsed_responses = self.inference_client.endpoint.extract_response_data(
                record
            )
        content_perf_ns = self._content_response_perf_ns_for_record(
            record, parsed_responses
        )
        credit_context.request_latency_ns = self._request_latency_ns_for_record(
            record, content_perf_ns
        )
        credit_context.output_sequence_length = (
            self._output_sequence_length_for_responses(parsed_responses)
        )
        credit_context.inter_token_latency_ns = self._inter_token_latency_ns_for_record(
            record,
            content_perf_ns,
            parsed_responses,
            credit_context.output_sequence_length,
        )

    def _content_response_perf_ns_for_record(
        self,
        record: RequestRecord,
        parsed_responses: list[ParsedResponse] | None = None,
    ) -> list[int]:
        """Return perf timestamps for parsed responses with meaningful content."""
        if parsed_responses is None:
            parsed_responses = self.inference_client.endpoint.extract_response_data(
                record
            )
        return [parsed.perf_ns for parsed in parsed_responses if parsed.data]

    def _request_latency_ns_for_record(
        self, record: RequestRecord, content_perf_ns: list[int] | None = None
    ) -> int | None:
        """Return the same latency sample used by RequestLatencyMetric."""
        if content_perf_ns is None:
            content_perf_ns = self._content_response_perf_ns_for_record(record)
        if not content_perf_ns:
            return None
        final_response_perf_ns = content_perf_ns[-1]
        if final_response_perf_ns < record.start_perf_ns:
            return None
        return final_response_perf_ns - record.start_perf_ns

    def _output_sequence_length_for_responses(
        self, parsed_responses: list[ParsedResponse]
    ) -> int | None:
        usage = find_last_non_empty_usage(parsed_responses)
        if usage is None or usage.completion_tokens is None:
            return None
        return usage.completion_tokens

    def _inter_token_latency_ns_for_record(
        self,
        record: RequestRecord,
        content_perf_ns: list[int] | None = None,
        parsed_responses: list[ParsedResponse] | None = None,
        output_sequence_length: int | None = None,
    ) -> float | None:
        """Return ITL using the records-pipeline output sequence formula."""
        if parsed_responses is None:
            parsed_responses = self.inference_client.endpoint.extract_response_data(
                record
            )
        if content_perf_ns is None:
            content_perf_ns = self._content_response_perf_ns_for_record(
                record, parsed_responses
            )
        if output_sequence_length is None:
            output_sequence_length = self._output_sequence_length_for_responses(
                parsed_responses
            )
        if len(content_perf_ns) < 2 or output_sequence_length is None:
            return None
        if output_sequence_length < 2:
            return None
        request_latency_ns = content_perf_ns[-1] - record.start_perf_ns
        ttft_ns = content_perf_ns[0] - record.start_perf_ns
        if request_latency_ns < 0 or ttft_ns < 0:
            return None
        return (request_latency_ns - ttft_ns) / (output_sequence_length - 1)

    def _pin_parent_if_fork_child(self, credit: Credit, x_correlation_id: str) -> None:
        """FORK child seed: pin the parent so its session stays resident in
        the cache until every FORK child has joined. FORK children
        sticky-route to the parent's worker, so the parent's session
        lives on this same SessionManager.
        """
        if (
            credit.parent_correlation_id is None
            or credit.branch_mode != ConversationBranchMode.FORK
        ):
            return
        try:
            self.session_manager.pin_for_fork_child(credit.parent_correlation_id)
        except KeyError:
            # Parent already evicted — child arrived too late to pin; let
            # the request proceed without seed context rather than failing.
            self.warning(
                f"FORK child {x_correlation_id!r} arrived after parent "
                f"{credit.parent_correlation_id!r} was evicted; not pinning"
            )

    def _seed_from_parent_if_fork_child(
        self, credit: Credit, x_correlation_id: str
    ) -> None:
        """Copy the parent session's accumulated ``turn_list`` into the
        freshly-created FORK child session.

        Companion to ``_pin_parent_if_fork_child``: pinning keeps the
        parent resident, this seeds the child with the parent's context
        so the request-builder prepends parent prompt + captured
        responses before the child's own messages. SPAWN-mode children
        skip this — fresh-context is the whole point of SPAWN.
        """
        if (
            credit.parent_correlation_id is None
            or credit.branch_mode != ConversationBranchMode.FORK
        ):
            return
        self.session_manager.seed_from_parent(
            x_correlation_id, credit.parent_correlation_id
        )

    def _resolve_graph_session_headers(
        self, envelope: dict[str, Any], credit: Credit
    ) -> dict[str, str] | None:
        """Resolve recorded dynamo session-identity headers for one dispatch.

        Recorded dynamo session ids are stamped verbatim at build time. They are
        suffixed per replay instance so concurrent instances of one trace never
        share -- or session-final-evict -- a server session.
        """
        from aiperf.graph.worker_materialize import uniquify_dynamo_session_headers

        return uniquify_dynamo_session_headers(
            envelope.get("extra_headers"),
            trace_instance_id=credit.trace_id,
            phase=credit.phase,
        )

    async def _process_graph_credit(
        self,
        credit_context: CreditContext,
        x_request_id: str,
        first_token_callback: Callable | None,
    ) -> None:
        """Process an agent-graph credit by materializing its request from the mmap.

        Reads the credit's ``(trace_id, node_ordinal)`` and
        rebuilds the node's request payload from the unified interned segment
        store (:func:`materialize_graph_request_unified` /
        :func:`materialize_graph_request_unified_bytes`) -- the sole graph
        store shape. It layers the
        run-level endpoint options the verbatim path would otherwise drop
        (``endpoint.extra`` / ``stream_options.include_usage``) while keeping
        the per-node ``dispatch_overrides`` / ``stream`` winning, wraps it as a
        single-turn ``raw_payload``, sends it, and forwards the result.

        A missing/corrupt store sets a fatal ``credit_context.error`` (not a
        silent swallow + per-credit retry); a missing node ordinal sets a clean
        ``GraphEnvelopeMissing`` error. Every pre-dispatch failure ALSO emits a
        synthetic error ``RequestRecord`` (:meth:`_send_graph_error_record`):
        the RecordsManager completion barrier counts success+error RECORDS
        against the credit-side ``final_requests_completed``, so an errored
        credit that produced no record would starve the barrier and hang the
        run at "please wait for the results". A catch-all around the body
        extends the same guarantee to UNANTICIPATED raisers (corrupt envelope
        bytes, materialize bugs, transport raises): the error is attributed on
        the context (mirroring the session path) and a record is emitted unless
        the dispatch path already sent one.
        """
        credit = credit_context.credit
        segment_store = self._graph_store_reader(credit_context)
        if segment_store is None:
            # Store absent/corrupt: error already set on the context. Do NOT
            # send a garbage request; the run surfaces the fatal config error.
            await self._fail_graph_credit(credit_context, x_request_id)
            return
        # ``credit.trace_id`` is the per-recycle INSTANCE id
        # (``{template}::{nonce}``, e.g. ``t-1::3f2a...``):
        # it rotates the cache-bust marker per recycle pass. The unified store
        # is keyed by the BASE/template id, so strip the ``::{nonce}`` instance
        # suffix before reading -- every recycle instance of one template reads
        # the same build-time manifests.
        base_trace_id = template_id(credit.trace_id)
        endpoint = self.model_endpoint.endpoint

        # Catch-all boundary: any exception escaping the body must still
        # attribute the error on the context (else the CreditReturn reports
        # success) AND emit a record (else the RecordsManager barrier starves
        # and the run hangs). ``credit_context.record_emitted`` tracks every
        # emission inside the region so a failure AFTER a record landed never
        # double-emits -- and, because it lives on the context rather than a
        # local, the task-level lockstep guard in
        # ``_on_credit_drop_message_task`` also sees it and does not append a
        # spurious ``CreditProcessingError`` record to a successful dispatch.
        credit_context.record_emitted = False
        try:
            # Pre-read the node envelope ONCE: routes bytes-vs-dict, carries the
            # `capture` flag (and assembly `items`), and is handed to the
            # materialize functions so the manifest is never decoded twice.
            envelope = read_node_envelope(
                segment_store, base_trace_id, credit.node_ordinal, credit.phase
            )
            if envelope is None:
                self._set_graph_envelope_missing(credit_context, base_trace_id)
                await self._fail_graph_credit(credit_context, x_request_id)
                return
            capture = bool(envelope.get("capture"))
            has_items = envelope.get("items") is not None
            extra_headers = self._resolve_graph_session_headers(envelope, credit)

            # In-flight bracket for the dynamic pool: a GraphTraceEnd arriving
            # while this credit is mid-processing defers eviction until the
            # bracket closes (a cancelled dispatch's FAILED write must land in a
            # live entry, not re-create an evicted one).
            pool = self._graph_dynamic_pool
            pool.credit_started(credit.trace_id)
            try:
                # Unified (interned A2) store: every node carries int ``handles``,
                # so a ``None`` from the unified fns IS a genuine miss. When no
                # content mutation is needed (cache-bust prepends a marker to the
                # first user message, which a pre-serialized body cannot do), build
                # the request body once from content-pool slices and send it
                # verbatim; otherwise take the unified dict path. Slot-carrying
                # nodes (``items``) always take the dict path: their messages are
                # composed per request from the dynamic pool.
                if endpoint.cache_bust == CacheBustTarget.NONE and not has_items:
                    built = materialize_graph_request_unified_bytes(
                        segment_store,
                        base_trace_id,
                        credit.node_ordinal,
                        credit.phase,
                        use_legacy_max_tokens=endpoint.use_legacy_max_tokens,
                        endpoint=endpoint,
                        envelope=envelope,
                        default_model=self.model_endpoint.primary_model_name,
                    )
                    if built is None:
                        self._set_graph_envelope_missing(credit_context, base_trace_id)
                        await self._fail_graph_credit(credit_context, x_request_id)
                        return
                    body, _model = built
                    request_info = self._build_graph_request_info(
                        credit_context,
                        None,
                        x_request_id,
                        raw_payload_bytes=body,
                        extra_headers=extra_headers,
                    )
                else:
                    try:
                        payload = materialize_graph_request_unified(
                            segment_store,
                            base_trace_id,
                            credit.node_ordinal,
                            credit.phase,
                            use_legacy_max_tokens=endpoint.use_legacy_max_tokens,
                            envelope=envelope,
                            default_model=self.model_endpoint.primary_model_name,
                            slot_resolver=(
                                (lambda ordinal: pool.get(credit.trace_id, ordinal))
                                if has_items
                                else None
                            ),
                        )
                    except GraphPoolMissingError as e:
                        # Broken stickiness (worker death re-route) or backstop
                        # eviction: a loud trace error, never a silent omission.
                        # The dispatch adapter sniffs this prefix.
                        await self._fail_graph_credit(
                            credit_context,
                            x_request_id,
                            error=format_graph_error(
                                GraphErrorCode.POOL_MISSING,
                                f"no pooled value for trace={credit.trace_id} "
                                f"src_ordinal={e.src_ordinal}; the producing "
                                f"node's reply was evicted or its worker was "
                                f"re-routed",
                            ),
                        )
                        return
                    if payload is None:
                        self._set_graph_envelope_missing(credit_context, base_trace_id)
                        await self._fail_graph_credit(credit_context, x_request_id)
                        return
                    apply_run_level_payload_options(
                        payload,
                        endpoint,
                        skip_endpoint_extra=bool(
                            envelope.get("endpoint_extra_applied")
                        ),
                    )
                    stamp_cache_bust_marker(
                        payload,
                        benchmark_id=self.run.benchmark_id,
                        trace_instance_id=credit.trace_id,
                        target=endpoint.cache_bust,
                    )
                    request_info = self._build_graph_request_info(
                        credit_context,
                        payload,
                        x_request_id,
                        extra_headers=extra_headers,
                    )
                await self._dispatch_graph_request(
                    request_info,
                    credit_context,
                    first_token_callback,
                    capture=capture,
                )
                credit_context.record_emitted = True
            finally:
                pool.credit_finished(credit.trace_id)
        except Exception as e:
            credit_context.error = ErrorDetails.from_exception(e)
            self.exception(
                f"Error processing graph credit {credit.id} "
                f"(trace={credit.trace_id!r} node_ordinal={credit.node_ordinal}): {e!r}"
            )
            if not credit_context.record_emitted:
                # Route through _fail_graph_credit, not _send_graph_error_record:
                # only the former sets record_emitted, and without it the
                # task-level lockstep guard emits a SECOND record for this credit.
                await self._fail_graph_credit(credit_context, x_request_id)
        return

    async def _dispatch_graph_request(
        self,
        request_info: RequestInfo,
        credit_context: CreditContext,
        first_token_callback: Callable | None,
        *,
        capture: bool,
    ) -> None:
        """Send one materialized graph request; capture the response if flagged.

        The pool write happens strictly BEFORE the CreditReturn (the caller's
        task finally), so a successor credit -- issued only after this credit
        resolves and sticky-routed to this worker -- always observes the
        entry. Every exit stores a value for captured nodes: a structured
        :class:`GraphCapturedReply` on success (text plus, for tool_calls
        replies, the verbatim assistant message JSON), ``EMPTY`` on a
        successful response with no replayable content, ``FAILED`` on a
        dispatch error, cancellation, or capture-extraction failure.
        """
        from aiperf.graph.dynamic_pool import GraphPoolSentinel

        credit = credit_context.credit
        pool_key = (credit.trace_id, credit.node_ordinal)
        self.task_stats.total += 1
        try:
            record: RequestRecord = await self.inference_client.send_request(
                request_info, first_token_callback=first_token_callback
            )
        except BaseException:
            if capture:
                self._graph_dynamic_pool.put(*pool_key, GraphPoolSentinel.FAILED)
            raise
        if capture:
            self._graph_dynamic_pool.put(
                *pool_key, self._graph_capture_value(credit_context, record)
            )
        await self._send_inference_result_message(record)
        if record.error is not None:
            credit_context.error = record.error
        return

    def _graph_capture_value(
        self, credit_context: CreditContext, record: RequestRecord
    ) -> GraphCapturedReply | GraphPoolSentinel:
        """Extract the pool value for a captured node's response record.

        Plain-text replies capture as a text-only :class:`GraphCapturedReply`;
        replies whose endpoint returned ``raw_messages`` (chat ``tool_calls`` /
        structured content) also carry the verbatim orjson-serialized
        assistant message so downstream splices reproduce the legacy
        child-seed rendering byte-for-byte.
        """
        from aiperf.graph.dynamic_pool import GraphCapturedReply, GraphPoolSentinel

        if record.error is not None:
            return GraphPoolSentinel.FAILED
        try:
            turn = self.inference_client.endpoint.build_assistant_turn(record)
            if turn is None:
                return GraphPoolSentinel.EMPTY
            if turn.raw_messages:
                # A single-message capture cannot faithfully carry more than one
                # assistant message (openai_responses can legitimately produce
                # several); truncating to raw_messages[0] would silently drop the
                # rest, so fail loudly instead.
                if len(turn.raw_messages) > 1:
                    credit_context.error = format_graph_error(
                        GraphErrorCode.CAPTURE_FAILED,
                        f"multi-entry raw_messages ({len(turn.raw_messages)} "
                        "entries) is not representable in a single-message "
                        "capture; single-message endpoints only",
                    )
                    return GraphPoolSentinel.FAILED
                message = turn.raw_messages[0]
                content = message.get("content")
                return GraphCapturedReply(
                    text=content if isinstance(content, str) else "",
                    message_json=orjson.dumps(message).decode(),
                )
            text = "".join(
                content for t in turn.texts or [] for content in (t.contents or [])
            )
        except Exception as e:
            credit = credit_context.credit
            credit_context.error = format_graph_error(
                GraphErrorCode.CAPTURE_FAILED,
                f"could not extract the pool value from the reply for "
                f"trace={credit.trace_id} node_ordinal={credit.node_ordinal}: {e!r}",
            )
            return GraphPoolSentinel.FAILED
        if not text:
            return GraphPoolSentinel.EMPTY
        return GraphCapturedReply(text=text)

    def _build_graph_request_info(
        self,
        credit_context: CreditContext,
        payload: dict[str, Any] | None,
        x_request_id: str,
        *,
        raw_payload_bytes: bytes | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> RequestInfo:
        """Wrap a materialized graph payload as a single-turn ``RequestInfo``.

        Two shapes, selected by which arg is set:

        - **dict path** (``payload``): a ``raw_payload`` carrying ``messages``,
          per-node ``dispatch_overrides``, ``stream``, and -- when ``--cache-bust``
          is set -- the stamped first-user marker.
        - **bytes path** (``raw_payload_bytes``): a pre-serialized body built once
          from mmap slices; ``payload`` is ``None``.

        ``Turn.model`` is deliberately left unset on both shapes: the recorded
        per-node model rides only the wire body (sent verbatim), while
        ``record.model_name`` falls back to the run ``--model`` in
        ``_finalize_request_record`` so tokenizer selection behaves like plain
        aiperf -- recorded deployment ids (e.g. ``dynamo/org/model-fp8``) are
        usually not resolvable tokenizer repos.

        Either way the chat endpoint's ``format_payload`` is bypassed and the body
        is sent verbatim.
        """
        credit = credit_context.credit
        turn = Turn(
            role="user",
            raw_payload=payload,
            raw_payload_bytes=raw_payload_bytes,
            # Per-node HTTP headers from the envelope (dynamo session
            # identity: x-dynamo-session-id / -parent-session-id /
            # -session-final). The transport merges the last turn's
            # extra_headers into the request headers; body is untouched.
            extra_headers=extra_headers,
        )
        return RequestInfo(
            model_endpoint=self.model_endpoint,
            credit_num=credit.id,
            credit_phase=credit.phase,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=credit.x_correlation_id,
            conversation_id=credit.conversation_id,
            turn_index=credit.turn_index,
            turns=[turn],
            drop_perf_ns=credit_context.drop_perf_ns,
            credit_issued_ns=credit.issued_at_ns,
            is_final_turn=credit.is_final_turn,
            agent_depth=credit.agent_depth,
            parent_correlation_id=credit.parent_correlation_id,
            # The graph adapter mints real per-trajectory num_turns
            # (is_final_turn = the recorded session-final fact) and stamps the
            # instance's root trajectory corr.
            root_correlation_id=credit.effective_root_correlation_id,
            is_parent_final=credit.is_parent_final,
            is_tree_final=credit.is_tree_final,
            url_index=credit.url_index,
        )

    def _graph_unified_reader(self) -> GraphSegmentUnifiedClient | None:
        """Return the lazily-opened unified store client, or ``None`` on failure.

        Cached across credits: the doomed/successful open is attempted exactly
        once (``_graph_unified_open_attempted``) and reused. The unified client
        carries both the addressing face (``get_node_envelope``) and the content
        face (``materialize_handles`` / ``build_request_body_handles``);
        ``_graph_store_reader`` returns this ONE
        client when the store exists on disk. A failed open is cached as
        ``None``; the A2-strict ``ValueError`` (an on-disk pre-v3 store rejected
        with "re-parse required") additionally remembers its reason so the
        fatal error path reports it instead of claiming no store exists.
        """
        if self._graph_unified_open_attempted:
            return self._graph_unified_client
        self._graph_unified_open_attempted = True
        meta = self._graph_client_metadata
        if meta is None:
            # A graph credit arrived but the dataset broadcast was not
            # graph-typed: the store location is unknown by contract (no env
            # re-derivation). Feeds the existing GraphStoreUnavailable fatal.
            self._graph_unified_open_failure = (
                "dataset broadcast did not carry GraphSegmentClientMetadata "
                "(or no dataset-configured broadcast was received); the "
                "DatasetManager must build the graph store and broadcast its "
                "location before workers can serve graph credits"
            )
            return None
        try:
            self._graph_unified_client = GraphSegmentUnifiedClient(
                base_path=meta.store_base_path, benchmark_id=meta.benchmark_id
            ).open()
        except ValueError as e:
            # A2-strict rejection: the store EXISTS on disk but is a legacy
            # pre-v3 shape. Unlike a missing store this is actionable, so keep
            # the reason for the GraphStoreUnavailable fatal.
            self._graph_unified_open_failure = str(e)
            self.warning(lambda e=e: f"unified store rejected: {e}")
            self._graph_unified_client = None
        except Exception as e:
            self.debug(lambda e=e: f"unified store not opened: {e!r}")
            self._graph_unified_client = None
        return self._graph_unified_client

    def _graph_store_reader(
        self, credit_context: CreditContext
    ) -> GraphSegmentUnifiedClient | None:
        """Return the worker's lazily-opened unified store client, or ``None``.

        Resolves the store from the graph-typed dataset broadcast
        (:class:`GraphSegmentClientMetadata.store_base_path` +
        ``benchmark_id``), so it reads exactly what the build-time
        :class:`GraphSegmentUnifiedBackingStore` wrote. There is NO env
        fallback: a graph credit whose broadcast carried no graph store
        location is a recorded failure, not a temp-dir guess.

        When the store is absent or corrupt, rather than letting the failure
        propagate to the broad ``_on_credit_drop_message_task`` ``except``
        (which logs but does NOT attribute the error -- F4), this caches a
        fatal, actionable
        :class:`ErrorDetails`, sets it on the credit context, and returns
        ``None`` so the caller skips the send. Attributing the error here is
        NOT sufficient on its own: the caller must still emit a synthetic
        error record (:meth:`_send_graph_error_record`), or the RecordsManager
        completion barrier starves and the run hangs at end of phase. The
        doomed open is attempted exactly ONCE; every later credit reuses the
        cached error.
        """
        unified = self._graph_unified_reader()
        if unified is not None:
            return unified
        if self._graph_store_open_error is None:
            meta = self._graph_client_metadata
            if meta is None:
                # No graph store location arrived on the broadcast; the failure
                # string already explains the missing GraphSegmentClientMetadata.
                message = f"No graph store could be opened: {self._graph_unified_open_failure}"
            else:
                unified_failure = (
                    f" The store WAS found but rejected: "
                    f"{self._graph_unified_open_failure}."
                    if self._graph_unified_open_failure is not None
                    else ""
                )
                message = (
                    f"No graph store could be opened under {meta.store_base_path}: "
                    f"the unified segment store "
                    f"(aiperf_graph_segments_{meta.benchmark_id}) could not be "
                    f"opened.{unified_failure} The store location arrives on the "
                    f"dataset broadcast (GraphSegmentClientMetadata.store_base_path); "
                    f"ensure the DatasetManager built the graph store and that its "
                    f"base path is on a shared filesystem visible to every worker."
                )
            self._graph_store_open_error = ErrorDetails(
                type="GraphStoreUnavailable",
                message=message,
            )
            self.error(lambda c=self._graph_store_open_error: c.message)
        credit_context.error = self._graph_store_open_error
        return None

    def _set_graph_envelope_missing(
        self, credit_context: CreditContext, base_trace_id: str
    ) -> None:
        """Set a clean ``GraphEnvelopeMissing`` error for an unaddressable node."""
        credit = credit_context.credit
        credit_context.error = ErrorDetails(
            type="GraphEnvelopeMissing",
            message=(
                f"No graph-store envelope for trace={base_trace_id!r} "
                f"(instance={credit.trace_id!r}) "
                f"node_ordinal={credit.node_ordinal} "
                f"phase={credit.phase!r}"
            ),
        )
        self.warning(lambda c=credit_context.error: c.message)

    async def _fail_graph_credit(
        self,
        credit_context: CreditContext,
        x_request_id: str,
        error: str | ErrorDetails | None = None,
    ) -> None:
        """Attribute a pre-dispatch graph failure and emit its synthetic record.

        The RecordsManager completion barrier counts success+error RECORDS
        against the credit-side ``final_requests_completed``, so an errored
        credit that produced no record starves the barrier and hangs the run at
        "please wait for the results". Binding the emission and the
        ``record_emitted`` flag together here keeps that invariant in one place
        instead of repeating it at every exit.

        ``error=None`` preserves an error the caller already attributed.
        """
        if error is not None:
            credit_context.error = error
        await self._send_graph_error_record(credit_context, x_request_id)
        credit_context.record_emitted = True

    async def _send_graph_error_record(
        self, credit_context: CreditContext, x_request_id: str
    ) -> None:
        """Emit a synthetic error ``RequestRecord`` for a pre-dispatch graph failure.

        Pre-dispatch failures (missing store, missing envelope, missing pool
        entry) never reach the inference client, so no ``RequestRecord`` would
        otherwise flow to the RecordProcessor. The RecordsManager's completion
        barrier counts success+error RECORDS against the credit-side
        ``final_requests_completed`` (``records_tracker``), so a credit that
        returns with an error but no record starves the barrier and the run
        hangs at "please wait for the results". Mirror of the session path's
        synthetic error record for a failed conversation fetch
        (:meth:`_request_conversation_from_dataset_manager`). The
        ``CreditReturn`` semantics are untouched -- the caller's finally still
        returns the credit with ``credit_context.error`` attached.
        """
        credit = credit_context.credit
        error = credit_context.error
        if not isinstance(error, ErrorDetails):
            error = ErrorDetails(
                type="GraphPreDispatchError",
                message=str(error) if error else "pre-dispatch graph credit failure",
            )
        now_perf_ns = time.perf_counter_ns()
        await self._send_inference_result_message(
            RequestRecord(
                request_info=RequestInfo(
                    model_endpoint=self.model_endpoint,
                    conversation_id=credit.conversation_id,
                    turn_index=credit.turn_index,
                    turns=[],
                    credit_num=credit.id,
                    credit_phase=credit.phase,
                    x_request_id=x_request_id,
                    x_correlation_id=credit.x_correlation_id,
                    drop_perf_ns=credit_context.drop_perf_ns,
                ),
                model_name=self.model_endpoint.primary_model_name,
                timestamp_ns=time.time_ns(),
                start_perf_ns=now_perf_ns,
                end_perf_ns=now_perf_ns,
                error=error,
            )
        )

    def _handle_terminal_disposition(
        self, credit: Credit, credit_context: CreditContext, x_correlation_id: str
    ) -> None:
        """Evict on final turn, cancellation, or terminal context overflow.

        Context overflow on a non-final, non-cancelled turn is terminal for
        agentic_replay (lane recycled, no final/cancel credit follows), so it
        must take the same eviction path.
        """
        if (
            credit.is_final_turn
            or credit_context.cancelled
            or _is_terminal_context_overflow(credit, credit_context)
        ):
            self._release_and_evict_for_terminal(credit, x_correlation_id)

    def _release_and_evict_for_terminal(
        self, credit: Credit, x_correlation_id: str
    ) -> None:
        """Release the parent pin (if FORK child) then evict this session.

        FORK parents whose terminal turn declared forks (``has_forks``)
        defer eviction: children arrive on the orchestrator's intercept
        path AFTER this credit return runs, so ``evict_if_unpinned``
        cannot find any pin to honor here. Setting ``pending_fork_eviction``
        signals ``release_fork_child`` to auto-evict the moment the last
        child joins.

        Non-FORK and non-parent sessions evict immediately.
        """
        if (
            credit.parent_correlation_id is not None
            and credit.branch_mode == ConversationBranchMode.FORK
        ):
            self.session_manager.release_fork_child(credit.parent_correlation_id)
        cur_session = self.session_manager.get(x_correlation_id)
        if cur_session is not None and cur_session.is_fork_parent:
            if credit.has_forks:
                cur_session.pending_fork_eviction = True
            self.session_manager.evict_if_unpinned(x_correlation_id)
        else:
            self.session_manager.evict(x_correlation_id)

    def _maybe_warn_cache_bust_silent_drop(
        self,
        session: UserSession,
        credit: Credit,
    ) -> None:
        """Emit a one-shot warning if cache-bust was requested but had nowhere
        to land on this credit (an empty ``session.turn_list``).

        Rate-limited to once per worker via ``self._cache_bust_warning_shown`` —
        the misconfiguration is identical for every credit, so a single
        actionable line beats N-thousand duplicates at scale.
        """
        if self._cache_bust_warning_shown:
            return
        target = credit.cache_bust_target
        marker = credit.cache_bust_marker
        if not marker or target == CacheBustTarget.NONE:
            return
        if not session.turn_list:
            self._cache_bust_warning_shown = True
            self.warning(
                f"cache-bust target={target.value} requested but session.turn_list "
                f"is empty — marker NOT injected (further occurrences suppressed)."
            )

    async def _execute_request(
        self,
        credit_context: CreditContext,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None,
    ) -> RequestRecord:
        """Send request, record result, and propagate errors to credit context.

        Metrics/response storage are handled once by the caller
        (``_finalize_session_response``) from a single response-processing pass.
        """
        self.task_stats.total += 1
        record = await self.inference_client.send_request(
            request_info, first_token_callback=first_token_callback
        )
        await self._send_inference_result_message(record)
        credit_context.record_emitted = True
        if record.error is not None:
            credit_context.error = record.error
        return record

    def _create_request_info(
        self,
        *,
        x_request_id: str,
        credit_context: CreditContext,
        session: UserSession | None = None,
        system_message: str | None = None,
        user_context_message: str | None = None,
        payload_bytes: bytes | None = None,
        turns: list[Turn] | None = None,
    ) -> RequestInfo:
        """Create RequestInfo for inference request.

        When ``session`` is provided (normal path), conversation state comes from
        the session. When omitted (raw payload fast path), fields are taken
        directly from the credit.

        When ``session`` is provided (normal path), conversation state comes
        from the session. When omitted (payload-bytes fast path), fields are
        taken directly from the credit.

        Args:
            x_request_id: Unique ID for this request (X-Request-ID header)
            credit_context: Context with credit metadata (num, phase, timestamps)
            session: Session with conversation history (None for raw payload path)
            system_message: Optional shared system message to prepend to first turn
            user_context_message: Optional per-conversation user context message
            payload_bytes: Pre-encoded payload bytes from mmap (raw payload path)
            turns: Explicit turns list (raw payload fast path with image metadata).
                   Takes precedence over session-derived turns when provided.

        Returns:
            RequestInfo with all data needed to send inference request
        """
        credit = credit_context.credit
        if turns is None:
            turns = session.turn_list if session else []
        if credit.max_tokens_override is not None and turns:
            last_turn = turns[-1]
            update: dict[str, Any] = {"max_tokens": credit.max_tokens_override}
            # A verbatim raw_payload dict is shipped as-is by inference_client,
            # so Turn.max_tokens alone is ignored on the wire; rewrite the wire
            # cap too or the recorded (baked-in) value would be sent instead.
            if isinstance(last_turn.raw_payload, dict):
                update["raw_payload"] = apply_max_tokens_to_wire_payload(
                    last_turn.raw_payload, credit.max_tokens_override
                )
            turns = [*turns[:-1], last_turn.model_copy(update=update)]
        source_turn = turns[-1] if turns else None
        return RequestInfo(
            model_endpoint=self.model_endpoint,
            credit_num=credit.id,
            credit_phase=credit.phase,
            phase_index=credit.phase_index,
            profiling_index=credit.profiling_index,
            phase_name=credit.phase_name,
            phase_kind=credit.phase_kind,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=session.x_correlation_id
            if session
            else credit.x_correlation_id,
            conversation_id=session.conversation.session_id
            if session
            else credit.conversation_id,
            turn_index=session.turn_index if session else credit.turn_index,
            source_trace_id=source_turn.source_trace_id if source_turn else None,
            source_outer_idx=source_turn.source_outer_idx if source_turn else None,
            source_inner_idx=source_turn.source_inner_idx if source_turn else None,
            source_kind=source_turn.source_kind if source_turn else None,
            turns=turns,
            drop_perf_ns=credit_context.drop_perf_ns,
            credit_issued_ns=credit.issued_at_ns,
            system_message=system_message,
            user_context_message=user_context_message,
            is_final_turn=credit.is_final_turn,
            url_index=session.url_index if session else credit.url_index,
            payload_bytes=payload_bytes,
            agent_depth=credit.agent_depth,
            parent_correlation_id=credit.parent_correlation_id,
            root_correlation_id=credit.effective_root_correlation_id,
            is_parent_final=credit.is_parent_final,
            is_tree_final=credit.is_tree_final,
            cache_bust_marker=credit.cache_bust_marker,
            cache_bust_target=credit.cache_bust_target
            if credit.cache_bust_marker is not None
            else None,
        )

    async def _retrieve_conversation(
        self,
        *,
        conversation_id: str,
        credit_context: CreditContext,
    ) -> Conversation:
        """Retrieve conversation from dataset client.

        The dataset client is initialized via factory when DatasetConfiguredNotification
        is received. The client type (mmap, S3, etc.) is transparent to this method.

        Args:
            conversation_id: ID of conversation to retrieve (from dataset)
            credit_context: Credit context

        Returns:
            Conversation object with turns and metadata

        Raises:
            asyncio.CancelledError: If stopping while no dataset client is available
            ValueError: If the DatasetManager fallback returns an error response
        """
        if self._dataset_client is not None:
            return await self._dataset_client.get_conversation(conversation_id)
        elif self.stop_requested:
            raise asyncio.CancelledError("Stop requested while retrieving conversation")

        return await self._request_conversation_from_dataset_manager(
            conversation_id, credit_context
        )

    async def _retrieve_conversation_for_session(
        self,
        *,
        credit_context: CreditContext,
    ) -> Conversation:
        """Retrieve a Conversation suitable for session-based processing.

        In the PAYLOAD_BYTES memory-map format the client's ``get_conversation``
        path raises because the full authoring shape is not persisted — only
        the per-turn payload bytes. For session-mode processing we reconstruct
        a minimal ``Conversation`` from per-turn payload bytes so
        ``session_manager`` can still advance turns.
        """
        conversation_id = credit_context.credit.conversation_id
        num_turns = credit_context.credit.num_turns

        if self._is_payload_bytes and self._dataset_client is not None:
            turns: list[Turn] = []
            for turn_index in range(num_turns):
                payload_turn = await self._dataset_client.get_payload_turn(
                    conversation_id, turn_index
                )
                if payload_turn is None:
                    turns.append(Turn(role="user"))
                    continue
                turns.append(turn_from_payload_turn(payload_turn))
            return Conversation(
                session_id=conversation_id,
                turns=turns,
                context_mode=self.session_manager.default_context_mode,
            )

        return await self._retrieve_conversation(
            conversation_id=conversation_id,
            credit_context=credit_context,
        )

    async def _request_conversation_from_dataset_manager(
        self, conversation_id: str, credit_context: CreditContext
    ) -> Conversation:
        """Fallback: Request from DatasetManager via ZMQ"""
        conversation_response: (
            ConversationResponseMessage | ErrorMessage
        ) = await self.conversation_request_client.request(
            ConversationRequestMessage(
                service_id=self.service_id,
                conversation_id=conversation_id,
                credit_phase=credit_context.credit.phase,
            )
        )
        if self.is_trace_enabled:
            self.trace(f"Received response message: {conversation_response}")

        # Check for error in conversation response
        if isinstance(conversation_response, ErrorMessage):
            error = conversation_response.error
            await self._send_inference_result_message(
                RequestRecord(
                    request_info=RecordContext(
                        conversation_id=conversation_id,
                        turn_index=0,
                        credit_num=credit_context.credit.id,
                        credit_phase=credit_context.credit.phase,
                        phase_index=credit_context.credit.phase_index,
                        profiling_index=credit_context.credit.profiling_index,
                        phase_name=credit_context.credit.phase_name,
                        phase_kind=credit_context.credit.phase_kind,
                        x_request_id=str(uuid.uuid4()),
                        x_correlation_id=credit_context.credit.x_correlation_id,
                        agent_depth=credit_context.credit.agent_depth,
                        parent_correlation_id=credit_context.credit.parent_correlation_id,
                    ),
                    model_name=self.model_endpoint.primary_model_name,
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    error=error,
                )
            )
            credit_context.record_emitted = True
            raise ValueError(f"Failed to retrieve conversation response: {error}")

        return conversation_response.conversation

    async def _send_inference_result_message(self, record: RequestRecord) -> None:
        """Send RequestRecord to RecordProcessor for metric calculation.

        All records (success and error) flow through this method to ensure consistent
        metric calculation and error tracking.

        Flow:
        1. Update task statistics (success/failure counts; ``total`` is bumped at dispatch)
        2. Wrap record in InferenceResultsMessage
        3. Push to RecordProcessor via PUSH socket (fire-and-forget)

        Note: Uses execute_async() to avoid blocking on network I/O.
        """
        # All records will flow through here to be sent to the inference results push client.
        self.task_stats.task_finished(record.valid)

        msg = InferenceResultsMessage(
            service_id=self.service_id,
            record=record,
        )
        self.execute_async(self.inference_results_push_client.push(msg))

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _on_profile_configure_command(self, message: CommandMessage) -> None:
        """Configure the worker."""
        self.debug("Waiting for dataset to be configured before starting profiling")
        await asyncio.wait_for(
            self._dataset_configured_event.wait(),
            timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
        )
        if self.is_debug_enabled:
            health = await asyncio.to_thread(self.get_process_health)
            memory_usage = health.memory_usage / BYTES_PER_MIB
            self.memory_usage_before_profiling = memory_usage
            self.debug(f"Memory usage before profiling: {memory_usage:.2f} MiB")

        self.event_loop_monitor.start()

    @on_stop
    async def _worker_stop(self) -> None:
        # Clean up dataset client resources using protocol lifecycle
        if self._dataset_client is not None:
            dataset_client = self._dataset_client
            self._dataset_client = None
            await dataset_client.stop()
            self.debug("Dataset client stopped")

        if self._graph_unified_client is not None:
            self._graph_unified_client.close()
            self._graph_unified_client = None
            self.debug("Graph unified store client closed")

        self.event_loop_monitor.stop()


def main() -> None:
    """Main entry point for the worker."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.WORKER)


if __name__ == "__main__":
    main()
