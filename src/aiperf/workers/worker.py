# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
import uuid

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.enums import CommAddress, CommandType, MessageType
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
    ModelEndpointInfo,
    ReasoningResponseData,
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
from aiperf.workers.inference_client import InferenceClient
from aiperf.workers.session_manager import UserSession, UserSessionManager


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
        self.event_loop_monitor.set_callback(self._on_event_loop_blocked_callback)

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
        )
        self.attach_child_lifecycle(self.inference_client)
        self.debug(
            lambda: f"Created inference client for {self.model_endpoint.endpoint.type}, "
            f"class: {self.inference_client.__class__.__name__}",
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

        # Only send FirstToken messages when prefill concurrency limiting is active.
        # Detecting first token requires parsing each SSE chunk, so skip this overhead
        # when the orchestrator doesn't need TTFT events for slot management.
        self._prefill_concurrency_enabled: bool = (
            self.user_config.loadgen.prefill_concurrency is not None
            or self.user_config.loadgen.warmup_prefill_concurrency is not None
        )

        # Only used as a fallback when dataset client is not initialized
        # or was not available when the credit was dropped. Must be created here
        # so it can be attached to the worker lifecycle.
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                address=CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
                bind=False,
            )
        )

    async def _on_event_loop_blocked_callback(self, delta_ms: float) -> None:
        """Callback to handle event loop blocking.

        This callback is called when the event loop is blocked for longer than the threshold.
        """
        self.warning(f"Event loop is blocked for {delta_ms:.2f}ms")

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
        self._dataset_configured_event.set()
        self.debug(
            lambda: f"Dataset client initialized: type={msg.client_metadata.client_type}"
        )

    @on_stop
    async def _send_worker_shutdown_message(self) -> None:
        """Send WorkerShutdown to announce shutdown."""
        try:
            await self.credit_dealer_client.send(
                WorkerShutdown(worker_id=self.service_id)
            )
            self.debug(
                lambda: f"Sent WorkerShutdown for graceful disconnect ({self.service_id})"
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
        await self.publish(self.create_health_message())

    def create_health_message(self) -> WorkerHealthMessage:
        return WorkerHealthMessage(
            service_id=self.service_id,
            health=self.get_process_health(),
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
            lambda id=credit_id: f"Credit {id} task done but NOT returned! "
            f"Task likely was cancelled before finally block could execute. Returning now."
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
                    lambda id=credit_id: f"Task for credit {id} not found (already completed?)"
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

    async def _process_credit(self, credit_context: CreditContext) -> None:
        """Process a credit (1 credit = 1 request).

        Flow:
        1. Generate UUID for x_request_id (X-Request-ID header)
        2. Check session cache using x_correlation_id:
           - Cache hit: Reuse session (enables conversation caching on inference server)
           - Cache miss: Retrieve conversation from DatasetManager, create new session
        3. Advance session to current turn index
        4. Process the turn (send request, collect response)
        5. On error: Set error in pre-created result
        6. Finally: Evict session from cache if this is the final turn

        Session Lifecycle:
        - First turn: Session created and cached under x_correlation_id
        - Subsequent turns: Session retrieved from cache (sticky routing ensures same worker)
        - Final turn: Session evicted from cache to free memory
        """
        x_request_id = str(uuid.uuid4())
        x_correlation_id = credit_context.credit.x_correlation_id
        credit = credit_context.credit

        # First token callback - only needed when prefill concurrency is enabled
        # Sends FirstToken to router for prefill concurrency slot release
        # Returns True when meaningful content is found to stop looking for first token
        first_token_callback = None
        if self._prefill_concurrency_enabled:

            async def first_token_callback(ttft_ns: int, message: SSEMessage) -> bool:
                # Use endpoint to check if message has meaningful content
                parsed = self.inference_client.endpoint.parse_response(message)
                if parsed is None or parsed.data is None:
                    return False  # Keep looking for meaningful content

                # Meaningful content found - send FirstToken to router
                await self.credit_dealer_client.send(
                    FirstToken(
                        credit_id=credit.id,
                        phase=credit.phase,
                        ttft_ns=ttft_ns,
                    )
                )
                # Track that FirstToken was sent so CreditReturn can report it
                credit_context.first_token_sent = True
                return True  # Stop looking, first token found

        try:
            session = self.session_manager.get(x_correlation_id)
            if session is None:
                _conversation = await self._retrieve_conversation(
                    conversation_id=credit_context.credit.conversation_id,
                    credit_context=credit_context,
                )
                # Store url_index from first turn so all turns hit the same backend
                session = self.session_manager.create_and_store(
                    x_correlation_id,
                    _conversation,
                    credit_context.credit.num_turns,
                    url_index=credit_context.credit.url_index,
                )

            session.advance_turn(credit_context.credit.turn_index)

            self.task_stats.total += 1
            request_info: RequestInfo = self._create_request_info(
                session=session,
                credit_context=credit_context,
                x_request_id=x_request_id,
                system_message=session.conversation.system_message,
                user_context_message=session.conversation.user_context_message,
            )
            record: RequestRecord = await self.inference_client.send_request(
                request_info, first_token_callback=first_token_callback
            )
            await self._send_inference_result_message(record)

            # Copy request-level errors to credit context for CreditReturn tracking
            if record.error is not None:
                credit_context.error = record.error

            if resp_turn := await self._process_response(record):
                session.store_response(resp_turn)

        except asyncio.CancelledError:
            # Mark cancelled before re-raising so finally can evict session
            credit_context.cancelled = True
            raise
        except Exception as e:
            credit_context.error = ErrorDetails.from_exception(e)
            self.exception(f"Error processing credit: {e!r}")
        finally:
            # Evict session on final turn OR if cancelled (no retry expected)
            if credit_context.credit.is_final_turn or credit_context.cancelled:
                self.session_manager.evict(x_correlation_id)

    def _create_request_info(
        self,
        *,
        x_request_id: str,
        session: UserSession,
        credit_context: CreditContext,
        system_message: str | None = None,
        user_context_message: str | None = None,
    ) -> RequestInfo:
        """Create RequestInfo for inference request with session state and credit metadata.

        Consolidates all information needed by InferenceClient and endpoints to:
        - Format the request payload (model, parameters, conversation history)
        - Set HTTP headers (X-Request-ID, X-Correlation-ID, auth)
        - Track request timing (drop_perf_ns for credit drop latency)
        - Handle cancellation (cancel_after_ns if specified)

        Args:
            x_request_id: Unique ID for this request (X-Request-ID header)
            session: Session containing conversation history and current turn index
            credit_context: Context with credit metadata (num, phase, timestamps)
            system_message: Optional shared system message to prepend to first turn
            user_context_message: Optional per-conversation user context message

        Returns:
            RequestInfo with all data needed to send inference request
        """
        credit = credit_context.credit
        return RequestInfo(
            model_endpoint=self.model_endpoint,
            credit_num=credit.id,
            credit_phase=credit.phase,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=session.x_correlation_id,
            conversation_id=session.conversation.session_id,
            turn_index=session.turn_index,
            turns=session.turn_list,
            drop_perf_ns=credit_context.drop_perf_ns,
            credit_issued_ns=credit.issued_at_ns,
            system_message=system_message,
            user_context_message=user_context_message,
            is_final_turn=credit.is_final_turn,
            # Use session's url_index to ensure all turns hit the same backend
            url_index=session.url_index,
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
                    request_info=RequestInfo(
                        model_endpoint=self.model_endpoint,
                        conversation_id=conversation_id,
                        turn_index=0,
                        turns=[],
                        credit_num=credit_context.credit.id,
                        credit_phase=credit_context.credit.phase,
                        x_request_id=str(uuid.uuid4()),
                        x_correlation_id=credit_context.credit.x_correlation_id,
                        drop_perf_ns=credit_context.drop_perf_ns,
                    ),
                    model_name=self.model_endpoint.primary_model_name,
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    error=error,
                )
            )
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
            memory_usage = self.get_process_health().memory_usage / BYTES_PER_MIB
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
