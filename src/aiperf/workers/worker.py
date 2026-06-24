# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
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
    ProcessHealth,
    ReasoningResponseData,
    RecordContext,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
    StreamingDealerClientProtocol,
)
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
    RouterToWorkerMessage,
    WorkerReady,
    WorkerShutdown,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.protocols import DatasetClientStoreProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.payload_retention import resolve_strip_record_payload_bytes
from aiperf.workers.inference_client import InferenceClient
from aiperf.workers.session_manager import UserSession, UserSessionManager

if TYPE_CHECKING:
    from aiperf.transports.base_transports import FirstTokenCallback


_logger = AIPerfLogger(__name__)


def _apply_cache_bust_to_system_message(
    system_message: str | None, marker: str, target: CacheBustTarget
) -> str | None:
    """Apply marker to the structured system_message string.

    Returns the modified string, or `None` if the input was None — the caller
    is then expected to fall back to mutating raw_messages.
    """
    if not marker or target == CacheBustTarget.NONE or system_message is None:
        return system_message
    if target == CacheBustTarget.SYSTEM_PREFIX:
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
    if not turn.texts:
        turn.texts = [Text(contents=[marker.strip()])]
        return
    first = turn.texts[0]
    if not first.contents:
        first.contents = [marker.strip()]
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
    )
    prefix_turns = _effective_prefix_turns(session)

    if target in (CacheBustTarget.SYSTEM_PREFIX, CacheBustTarget.SYSTEM_SUFFIX):
        return _apply_system_target_cache_bust(
            prefix_turns,
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
    system_message: str | None,
    marker: str,
    target: CacheBustTarget,
    is_prefix: bool,
) -> str | None:
    """Inject a ``SYSTEM_PREFIX`` / ``SYSTEM_SUFFIX`` marker for one credit.

    ``prefix_turns`` is the effective wire prefix slice (see
    :func:`_effective_prefix_turns`). Three sub-paths:
      1. Conversation-level ``system_message`` present: marker applied every
         turn (string mutation re-applied per credit). Unaffected by
         ``reset_context`` — the ``system_message`` rides on ``RequestInfo`` and
         is re-emitted every turn independent of ``build_messages``' reset.
      2. ``raw_messages`` first dict has ``role=="system"``: marker injected
         into the first system message of the prefix slice.
      3. No system in the slice -> first-user-turn fallback: marker injected
         into the first user turn every credit (idempotent), matching
         ``FIRST_TURN_*`` semantics so a seeded mid-trajectory resume still
         marks the prefix.

    Returns the (possibly modified) ``system_message``.
    """
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
        if turns[i].reset_context and turns[i].raw_messages:
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
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        self.debug(lambda: f"Worker process __init__ (pid: {self._process.pid})")

        self.event_loop_monitor = EventLoopMonitor(self.service_id)

        self.task_stats: WorkerTaskStats = WorkerTaskStats()

        self.credit_tasks: dict[int, asyncio.Task] = {}

        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )

        self.model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)

        self.inference_client: InferenceClient = InferenceClient(
            model_endpoint=self.model_endpoint,
            service_id=self.service_id,
            strip_record_payload_bytes=resolve_strip_record_payload_bytes(
                self.user_config, self.model_endpoint
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

        self.memory_usage_before_profiling: float | None = None

        self.session_manager: UserSessionManager = UserSessionManager()

        # Dataset client for direct data access (eliminates DatasetManager bottleneck)
        # Initialized when DatasetConfiguredNotification is received via factory
        self._dataset_client: DatasetClientStoreProtocol | None = None
        self._dataset_configured_event = asyncio.Event()
        self._is_payload_bytes: bool = False

        # Only send FirstToken messages when prefill concurrency limiting is active.
        # Detecting first token requires parsing each SSE chunk, so skip this overhead
        # when the orchestrator doesn't need TTFT events for slot management.
        self._prefill_concurrency_enabled: bool = (
            self.user_config.loadgen.prefill_concurrency is not None
            or self.user_config.loadgen.warmup_prefill_concurrency is not None
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
        self.session_manager.set_default_context_mode(msg.metadata.default_context_mode)
        if isinstance(msg.client_metadata, MemoryMapClientMetadata):
            self._is_payload_bytes = (
                msg.client_metadata.format == MemoryMapFormat.PAYLOAD_BYTES
            )
            if (
                self._is_payload_bytes
                and self.user_config.input.prompt.cache_bust.target
                != CacheBustTarget.NONE
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
        )
        self.execute_async(self.credit_dealer_client.send(credit_return))
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
            if not credit_context.cancelled and not credit_context.record_emitted:
                try:
                    await self._emit_credit_failure_record(credit_context)
                except Exception as e:  # noqa: BLE001
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
            )
            await self.credit_dealer_client.send(credit_return)
            # Mark as returned AFTER send succeeds
            # If send fails/cancelled, done callback will retry
            # Router idempotency guard handles duplicates
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
        x_request_id = str(uuid.uuid4())
        x_correlation_id = credit_context.credit.x_correlation_id
        first_token_callback = self._make_first_token_callback(credit_context)

        try:
            # Payload bytes fast path: bypass session/conversation deserialization.
            # Skipped for DAG descendants (agent_depth > 0) so their turn_list
            # goes through session_manager — FORK children need parent-seeded
            # accumulation and all multi-turn children need session state.
            # (FORK *parents* are kept off this fast path upstream: the dataset
            # manager refuses to preformat any DAG dataset that declares
            # branches, so a FORK parent's conversation is served structured —
            # with branches — and its session is created and pinned here.)
            context_mode_requires_session = credit_context.credit.agent_depth > 0
            if (
                self._is_payload_bytes
                and self._dataset_client is not None
                and not context_mode_requires_session
            ):
                conversation_id = credit_context.credit.conversation_id
                turn_index = credit_context.credit.turn_index
                payload_bytes = await self._dataset_client.get_payload_bytes(
                    conversation_id, turn_index
                )
                if payload_bytes is not None:
                    # The canonical wire payload is ``payload_bytes`` — it's
                    # stashed on request_info and consumed verbatim by the
                    # transport. Record-side consumers derive media counts
                    # from the endpoint's ``extract_payload_inputs`` over
                    # ``payload_bytes``; nothing reads ``turn.images``
                    # downstream of this fast path.
                    turns: list[Turn] = [Turn(role="user")]
                    request_info = self._create_request_info(
                        x_request_id=x_request_id,
                        credit_context=credit_context,
                        payload_bytes=payload_bytes,
                        turns=turns,
                    )
                    await self._execute_request(
                        credit_context, request_info, first_token_callback
                    )
                    return

            # Normal path: session-based conversation handling.
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
            if credit_context.credit.is_final_turn or credit_context.cancelled:
                self.session_manager.evict(x_correlation_id)

    def _make_first_token_callback(
        self, credit_context: CreditContext
    ) -> FirstTokenCallback | None:
        """Build first-token callback when prefill concurrency limiting is active.

        Detecting first token requires parsing each SSE chunk, so this overhead
        is skipped when the orchestrator doesn't need TTFT events for slot management.

        Returns:
            Callback that sends FirstToken to the router on meaningful content,
            or None when prefill concurrency is disabled.
        """
        if not self._prefill_concurrency_enabled:
            return None

        credit = credit_context.credit

        async def on_first_token(ttft_ns: int, message: SSEMessage) -> bool:
            parsed = self.inference_client.endpoint.parse_response(message)
            if parsed is None or parsed.data is None:
                return False

            await self.credit_dealer_client.send(
                FirstToken(
                    credit_id=credit.id,
                    phase=credit.phase,
                    ttft_ns=ttft_ns,
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
        session = self.session_manager.get(x_correlation_id)
        if session is None:
            _conversation = await self._retrieve_conversation_for_session(
                credit_context=credit_context,
            )
            session = self.session_manager.create_and_store(
                x_correlation_id,
                _conversation,
                credit_context.credit.num_turns,
                url_index=credit_context.credit.url_index,
                parent_correlation_id=credit_context.credit.parent_correlation_id,
                branch_mode=credit_context.credit.branch_mode,
            )

        session.advance_turn(credit_context.credit.turn_index)

        system_message = _apply_cache_bust(
            session,
            credit_context.credit,
            session.conversation.system_message,
        )
        self._maybe_warn_cache_bust_silent_drop(session, credit_context.credit)

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

        if session.should_store_response() and (
            resp_turn := await self._process_response(record)
        ):
            session.store_response(resp_turn)

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
        """Send request, record result, and propagate errors to credit context."""
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
            turns = [
                *turns[:-1],
                turns[-1].model_copy(update={"max_tokens": credit.max_tokens_override}),
            ]
        return RequestInfo(
            model_endpoint=self.model_endpoint,
            credit_num=credit.id,
            credit_phase=credit.phase,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=session.x_correlation_id
            if session
            else credit.x_correlation_id,
            conversation_id=session.conversation.session_id
            if session
            else credit.conversation_id,
            turn_index=session.turn_index if session else credit.turn_index,
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
            RuntimeError: If dataset client not initialized
            KeyError: If conversation_id not found in dataset
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
                payload_bytes = await self._dataset_client.get_payload_bytes(
                    conversation_id, turn_index
                )
                raw_payload = orjson.loads(payload_bytes) if payload_bytes else None
                turns.append(Turn(role="user", raw_payload=raw_payload))
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

    async def _process_response(self, record: RequestRecord) -> Turn | None:
        """Extract assistant response from RequestRecord and convert to Turn for session.

        Flow:
        1. Use endpoint to parse responses into structured data
        2. Extract text content from all responses
        3. If text present: Create Turn with role="assistant"
        4. If no text: Return None (error response or no content)

        Args:
            record: RequestRecord with raw responses from inference server

        Returns:
            Turn object for storing in session, or None if no content
        """
        resp = self.inference_client.endpoint.extract_response_data(record)
        # Skip reasoning responses in multi-turn conversations
        output_texts = []
        for response in resp:
            if not response.data:
                continue
            if isinstance(response.data, ReasoningResponseData):
                if response.data.content:
                    output_texts.append(response.data.content)
            else:
                output_texts.append(response.data.get_text())
        resp_text = "".join(output_texts)

        return (
            Turn(role="assistant", texts=[Text(contents=[resp_text])])
            if resp_text
            else None
        )

    async def _send_inference_result_message(self, record: RequestRecord) -> None:
        """Send RequestRecord to RecordProcessor for metric calculation.

        All records (success and error) flow through this method to ensure consistent
        metric calculation and error tracking.

        Flow:
        1. Update task statistics (total and success/failure counts)
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

        self.event_loop_monitor.stop()


def main() -> None:
    """Main entry point for the worker."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.WORKER)


if __name__ == "__main__":
    main()
