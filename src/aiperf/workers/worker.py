# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.base_component_service import BaseComponentService

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
from aiperf.common.channel_codecs import RAW_INFERENCE_CODEC
from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.control_structs import Command
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    MessageType,
    WorkerStartupState,
)
from aiperf.common.environment import Environment
from aiperf.common.event_loop_monitor import EventLoopMonitor
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import (
    Hook,
    background_task,
    on_command,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.inference_wire import (
    build_inference_results_wire_message,
    encode_inference_results_wire_message,
)
from aiperf.common.memory_profiler import MemoryProfiler
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    ErrorMessage,
    WorkerHealthMessage,
    WorkerStartupStateMessage,
)
from aiperf.common.messages.dataset_messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
)
from aiperf.common.mixins import ProcessHealthMixin
from aiperf.common.models import (
    Conversation,
    DatasetClientMetadata,
    DatasetMetadata,
    ErrorDetails,
    MemoryMapClientMetadata,
    ProcessHealth,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetReady,
    GroupDatasetStateQuery,
    GroupDatasetStateSnapshot,
    GroupManagerToPeerMessage,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupPeerHello,
    GroupPeerShutdown,
    GroupWorkerHealth,
    GroupWorkerStartupState,
)
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
    StreamingDealerClientProtocol,
)
from aiperf.credit.messages import (
    CancelCredits,
    CreditChannelMessage,
    CreditReturn,
    FirstToken,
    InFlightReconciliation,
    InFlightReport,
    TimePong,
    WorkerConnected,
    WorkerDispatchable,
    WorkerShutdown,
    WorkerUndispatchable,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.protocols import DatasetClientStoreProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker
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

        self.event_loop_monitor = EventLoopMonitor(
            self.service_id,
            artifact_dir=self.run.cfg.output.artifact_directory,
        )

        self.task_stats: WorkerTaskStats = WorkerTaskStats()

        self.credit_tasks: dict[int, asyncio.Task] = {}

        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
                codec=RAW_INFERENCE_CODEC,
            )
        )

        self.inference_client: InferenceClient = InferenceClient(
            run=self.run,
            service_id=self.service_id,
        )
        self.attach_child_lifecycle(self.inference_client)
        self.debug(
            lambda: (
                f"Created inference client for {self.run.cfg.endpoint.type}, "
                f"class: {self.inference_client.__class__.__name__}"
            ),
        )

        # Credit channel (Router -> Worker): receive-only, gets Credit and CancelCredits.
        # Identity must be unique - ZMQ ROUTER uses it to address messages.
        self.credit_dealer_client: StreamingDealerClientProtocol = (
            self.comms.create_streaming_dealer_client(
                address=CommAddress.CREDIT_ROUTER,
                identity=self.service_id,
                bind=False,
                decode_type=CreditChannelMessage,
            )
        )
        self.credit_dealer_client.register_receiver(self._on_credit_message)

        # Return channel (Worker -> Router): send-only. CreditReturn, FirstToken,
        # WorkerConnected, WorkerDispatchable, WorkerShutdown, TimePing.
        # No incoming messages.
        self.return_dealer_client: StreamingDealerClientProtocol = (
            self.comms.create_streaming_dealer_client(
                address=CommAddress.CREDIT_RETURN_ROUTER,
                identity=self.service_id,
                bind=False,
            )
        )

        self.pod_lifecycle_dealer_client: StreamingDealerClientProtocol | None = None
        if self._is_group_managed_mode():
            self.pod_lifecycle_dealer_client = (
                self.comms.create_streaming_dealer_client(
                    address=CommAddress.GROUP_LIFECYCLE,
                    identity=self.service_id,
                    bind=False,
                    decode_type=GroupManagerToPeerMessage,
                )
            )
            self.pod_lifecycle_dealer_client.register_receiver(
                self._on_pod_lifecycle_message
            )

        self.memory_usage_before_profiling: float | None = None
        self._pod_index = os.environ.get("AIPERF_POD_INDEX")

        self.session_manager: UserSessionManager = UserSessionManager()

        self.clock_offset_tracker = ClockOffsetTracker(logger_name=self.service_id)

        # Memory profiler for debugging memory growth (enabled via AIPERF_DEV_MEMORY_PROFILE_ENABLED)
        self._memory_profiler = MemoryProfiler(service_id=self.service_id)

        # Dataset client for direct data access (eliminates DatasetManager bottleneck)
        # Initialized when DatasetConfiguredNotification is received via factory.
        # In Kubernetes mode (network client type), initialization is deferred until
        # WorkerGroupManager downloads the dataset and sends GroupDatasetReady.
        self._dataset_client: DatasetClientStoreProtocol | None = None
        self._dataset_configured_event = asyncio.Event()
        self._latest_pod_dataset_state: GroupDatasetStateSnapshot | None = None
        self._dataset_state_retry_task: asyncio.Task[None] | None = None
        self._worker_ready_event = asyncio.Event()
        self._worker_ready_lock = asyncio.Lock()
        self._startup_state: WorkerStartupState | None = None

        # Only send FirstToken messages when prefill concurrency limiting is active.
        # Detecting first token requires parsing each SSE chunk, so skip this overhead
        # when the orchestrator doesn't need TTFT events for slot management.
        # Check all phases for prefill_concurrency settings
        self._prefill_concurrency_enabled = any(
            phase.prefill_concurrency is not None
            for phase in self.run.cfg.phases.values()
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

    @on_start
    async def _send_worker_ready_message(self) -> None:
        """Announce connectivity, then become dispatchable when startup gates clear."""
        await self._publish_startup_state(WorkerStartupState.STARTING)
        await self.return_dealer_client.send(WorkerConnected(worker_id=self.service_id))
        if self._is_group_managed_mode():
            if self.pod_lifecycle_dealer_client is not None:
                await self.pod_lifecycle_dealer_client.send(
                    GroupPeerHello(
                        service_id=self.service_id,
                        service_type=str(self.service_type),
                        pod_index=self._pod_index,
                    )
                )
            await self._publish_startup_state(WorkerStartupState.WAITING_FOR_DATASET)
            self._ensure_group_dataset_state_retry()
            await self._complete_group_startup_flow()
            self.debug(
                "Group-managed mode: deferring WorkerDispatchable until group-local dataset state is ready"
            )
            return
        await self._publish_startup_state(WorkerStartupState.ROUTER_PROBING)
        await self._measure_baseline_rtt()
        await self._mark_worker_ready()

    def _is_kubernetes_mode(self) -> bool:
        """Check if running in Kubernetes mode."""
        return self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES

    def _is_group_managed_mode(self) -> bool:
        """Check if WorkerGroupManager owns this worker's startup lifecycle."""
        return self.run.cfg.runtime.uses_worker_group_manager

    def _uses_controller_control_channel(self) -> bool:
        """Group-managed workers stay off the controller control channel."""
        return not self._is_group_managed_mode()

    def _uses_global_message_bus_probe(self) -> bool:
        """Group-managed workers should not block startup on global PUB/SUB probes."""
        return not self._is_group_managed_mode()

    def _should_subscribe_to_message_type(
        self, _hook: Hook, message_type: MessageType
    ) -> bool:
        """Group-managed workers should not consume global dataset broadcasts."""
        if self._is_group_managed_mode():
            return message_type != MessageType.DATASET_CONFIGURED_NOTIFICATION
        return True

    async def _measure_baseline_rtt(self) -> None:
        """Measure baseline RTT on the credit channel before announcing readiness."""
        await self.clock_offset_tracker.measure_baseline_rtt(
            send_ping=self.return_dealer_client.send,
        )

    async def _on_pod_lifecycle_message(
        self, message: GroupManagerToPeerMessage
    ) -> None:
        """Handle group-local lifecycle messages from WorkerGroupManager."""
        if isinstance(message, GroupDatasetReady):
            await self._on_dataset_ready(message)
        elif isinstance(message, GroupDatasetStateSnapshot):
            self._latest_pod_dataset_state = message
            await self._complete_group_startup_flow(message)
        elif isinstance(message, GroupPeerCommand):
            await self._handle_pod_peer_command(message)

    async def _handle_pod_peer_command(self, message: GroupPeerCommand) -> None:
        """Handle group-local lifecycle commands from WorkerGroupManager."""
        if self.pod_lifecycle_dealer_client is None:
            return
        if message.command == str(CommandType.PROFILE_CONFIGURE):
            await self._configure_for_profiling()
        elif message.command == str(CommandType.SHUTDOWN):
            await self.stop()
        else:
            self.warning(f"Unknown group-local command: {message.command}")
            return
        await self.pod_lifecycle_dealer_client.send(
            GroupPeerCommandAck(cid=message.cid, service_id=self.service_id)
        )

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(self, msg: DatasetConfiguredNotification) -> None:
        """Initialize dataset client when configuration is received.

        Local-mode workers initialize directly from the dataset broadcast.
        Group-managed workers rely on group-local dataset readiness and should not
        subscribe to or consume this global broadcast during startup.
        """
        if not self._matches_current_benchmark(msg.client_metadata):
            self.warning(
                "Ignoring dataset configuration for a different benchmark: "
                f"{msg.client_metadata.data_file_path}"
            )
            return

        if self._dataset_configured_event.is_set():
            self.debug("Dataset already initialized, ignoring rebroadcast")
            return

        if self._is_group_managed_mode():
            self.debug(
                "Ignoring global dataset configuration in group-managed mode; "
                "waiting for group-local dataset readiness"
            )
            return

        await self._initialize_dataset_client(msg.client_metadata, msg.metadata)

    async def _on_dataset_ready(self, msg: GroupDatasetReady) -> None:
        """Handle group-local dataset readiness from WorkerGroupManager."""
        if not self._matches_current_download(msg):
            self.debug(
                lambda: (
                    "Ignoring downloaded dataset for a different pod or benchmark: "
                    f"service={msg.service_id}, pod_index={msg.pod_index}, "
                    f"path={msg.data_file_path}"
                )
            )
            return

        async with self._worker_ready_lock:
            if (
                self._dataset_configured_event.is_set()
                or self._worker_ready_event.is_set()
            ):
                self.debug("Dataset already initialized, ignoring download rebroadcast")
                return

            if not msg.success:
                self._ensure_group_dataset_state_retry()
                return

            await self._initialize_dataset_client(
                MemoryMapClientMetadata(
                    data_file_path=Path(msg.data_file_path),
                    index_file_path=Path(msg.index_file_path),
                    conversation_count=msg.conversation_count,
                    total_size_bytes=msg.total_size_bytes,
                )
            )
            await self._mark_worker_ready_locked()

    async def _query_pod_dataset_state(self) -> GroupDatasetStateSnapshot | None:
        """Fetch the current group-local dataset state from WorkerGroupManager."""
        if self.pod_lifecycle_dealer_client is None or not hasattr(
            self.pod_lifecycle_dealer_client, "request"
        ):
            return None

        try:
            response = await self.pod_lifecycle_dealer_client.request(
                GroupDatasetStateQuery(
                    rid=uuid.uuid4().hex,
                    service_id=self.service_id,
                ),
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return None
        if not isinstance(response, GroupDatasetStateSnapshot):
            return None
        self._latest_pod_dataset_state = response
        return response

    def _ensure_group_dataset_state_retry(self) -> None:
        """Start a retry loop for group-local dataset state if one is not already running."""
        if not self._is_group_managed_mode() or self._worker_ready_event.is_set():
            return
        if (
            self._dataset_state_retry_task is None
            or self._dataset_state_retry_task.done()
        ):
            self._dataset_state_retry_task = self.execute_async(
                self._retry_group_dataset_state_until_ready()
            )

    async def _retry_group_dataset_state_until_ready(self) -> None:
        """Poll group-local dataset state until this worker becomes dispatchable."""
        while not self.stop_requested and not self._worker_ready_event.is_set():
            await self._complete_group_startup_flow()
            if self._worker_ready_event.is_set():
                return
            await asyncio.sleep(1.0)

    async def _complete_group_startup_flow(
        self,
        snapshot: GroupDatasetStateSnapshot | None = None,
    ) -> None:
        """Make the worker dispatchable once group-local dataset state is ready."""
        if not self._is_group_managed_mode() or self._worker_ready_event.is_set():
            return

        async with self._worker_ready_lock:
            if self._worker_ready_event.is_set():
                return

            if self._dataset_configured_event.is_set():
                await self._mark_worker_ready_locked()
                return

            current_snapshot = snapshot or await self._query_pod_dataset_state()
            if current_snapshot is None or not current_snapshot.ready:
                self._ensure_group_dataset_state_retry()
                return

            if not self._dataset_configured_event.is_set():
                await self._initialize_dataset_client(
                    MemoryMapClientMetadata(
                        data_file_path=Path(current_snapshot.data_file_path),
                        index_file_path=Path(current_snapshot.index_file_path),
                        conversation_count=current_snapshot.conversation_count,
                        total_size_bytes=current_snapshot.total_size_bytes,
                    ),
                )
            if current_snapshot.default_context_mode is not None:
                self.session_manager.set_default_context_mode(
                    current_snapshot.default_context_mode
                )
            await self._mark_worker_ready_locked()

    async def _mark_worker_ready(self) -> None:
        """Send the ready transition exactly once."""
        async with self._worker_ready_lock:
            await self._mark_worker_ready_locked()

    async def _mark_worker_ready_locked(self) -> None:
        """Send the ready transition exactly once while holding the ready lock."""
        if self._worker_ready_event.is_set():
            return
        await self.return_dealer_client.send(
            WorkerDispatchable(worker_id=self.service_id)
        )
        await self._publish_startup_state(WorkerStartupState.READY)
        self._worker_ready_event.set()
        retry_task = self._dataset_state_retry_task
        if retry_task is not None and retry_task is not asyncio.current_task():
            retry_task.cancel()

    def _matches_current_benchmark(
        self, client_metadata: DatasetClientMetadata
    ) -> bool:
        """Check whether dataset client metadata belongs to the current benchmark dataset."""
        if not isinstance(client_metadata, MemoryMapClientMetadata):
            return True
        return self.run.cfg.artifacts.benchmark_id in str(
            client_metadata.data_file_path
        )

    def _matches_current_download(self, msg: GroupDatasetReady) -> bool:
        """Check whether a group-local dataset-ready notification belongs to this worker's pod."""
        benchmark_id = self.run.cfg.artifacts.benchmark_id
        if benchmark_id not in msg.data_file_path:
            return False
        if not self._is_kubernetes_mode():
            return True
        return msg.pod_index is not None and msg.pod_index == self._pod_index

    async def _initialize_dataset_client(
        self,
        client_metadata: DatasetClientMetadata,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> None:
        """Initialize the dataset client from metadata.

        Args:
            client_metadata: The client metadata with paths/config for dataset access.
            dataset_metadata: Dataset structure metadata (conversations, context mode).
        """
        ClientStoreClass = plugins.get_class(
            PluginType.DATASET_CLIENT_STORE, client_metadata.client_type
        )
        self._dataset_client = ClientStoreClass(client_metadata=client_metadata)
        await self._dataset_client.initialize()
        if dataset_metadata is not None:
            self.session_manager.set_default_context_mode(
                dataset_metadata.default_context_mode
            )
        self._dataset_configured_event.set()
        self.debug(
            lambda: f"Dataset client initialized: type={client_metadata.client_type}"
        )

    @on_stop
    async def _send_worker_shutdown_message(self) -> None:
        """Send WorkerShutdown to announce shutdown."""
        try:
            await self._publish_startup_state(WorkerStartupState.SHUTTING_DOWN)
            retry_task = self._dataset_state_retry_task
            if retry_task is not None and not retry_task.done():
                retry_task.cancel()
            if self._is_kubernetes_mode():
                await self.return_dealer_client.send(
                    WorkerUndispatchable(worker_id=self.service_id, reason="shutdown")
                )
            if self.pod_lifecycle_dealer_client is not None:
                await self.pod_lifecycle_dealer_client.send(
                    GroupPeerShutdown(
                        service_id=self.service_id,
                        service_type=str(self.service_type),
                    )
                )
            await self.return_dealer_client.send(
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
        if (
            self._is_group_managed_mode()
            and self.pod_lifecycle_dealer_client is not None
        ):
            await self.pod_lifecycle_dealer_client.send(
                self.create_pod_worker_health(health)
            )
            return
        await self.publish(self.create_health_message(health))

    def create_health_message(self, health: ProcessHealth) -> WorkerHealthMessage:
        return WorkerHealthMessage(
            service_id=self.service_id,
            health=health,
            task_stats=self.task_stats,
        )

    def create_pod_worker_health(self, health: ProcessHealth) -> GroupWorkerHealth:
        """Build the group-local msgspec health snapshot."""
        io_counters = (
            tuple(health.io_counters) if health.io_counters is not None else None
        )
        cpu_times = tuple(health.cpu_times) if health.cpu_times is not None else None
        num_ctx_switches = (
            tuple(health.num_ctx_switches)
            if health.num_ctx_switches is not None
            else None
        )
        return GroupWorkerHealth(
            service_id=self.service_id,
            pid=health.pid,
            create_time=health.create_time,
            uptime=health.uptime,
            cpu_usage=health.cpu_usage,
            memory_usage=health.memory_usage,
            pss_memory=health.pss_memory,
            io_counters=io_counters,
            cpu_times=cpu_times,
            num_ctx_switches=num_ctx_switches,
            num_threads=health.num_threads,
            task_total=self.task_stats.total,
            task_failed=self.task_stats.failed,
            task_completed=self.task_stats.completed,
        )

    async def _publish_startup_state(self, state: WorkerStartupState) -> None:
        """Publish a worker startup-state transition if it changed."""
        if self._startup_state == state:
            return
        self._startup_state = state
        if (
            self._is_group_managed_mode()
            and self.pod_lifecycle_dealer_client is not None
        ):
            await self.pod_lifecycle_dealer_client.send(
                GroupWorkerStartupState(
                    service_id=self.service_id,
                    startup_state=str(state),
                    request_ns=time.time_ns(),
                )
            )
            return
        await self.publish(
            WorkerStartupStateMessage(
                service_id=self.service_id,
                startup_state=state,
            )
        )

    async def _on_credit_message(self, message: CreditChannelMessage) -> None:
        """Handle incoming messages on the credit channel (Router -> Worker)."""
        with self.event_loop_monitor.activity(
            f"credit msg={message.__class__.__name__}"
        ):
            match message:
                case Credit():
                    self._schedule_credit_drop_task(message)
                case CancelCredits():
                    await self._on_cancel_credits_message(message)
                case TimePong():
                    self.clock_offset_tracker.handle_pong(message)
                case InFlightReconciliation():
                    await self._on_reconciliation(message)
                case _:
                    self.warning(
                        f"Unknown credit channel message: {message.__class__.__name__}"
                    )

    async def _on_reconciliation(self, message: InFlightReconciliation) -> None:
        """Respond to router's reconciliation request with current in-flight credits."""
        await self.return_dealer_client.send(
            InFlightReport(credit_ids=frozenset(self.credit_tasks.keys()))
        )

    def _schedule_credit_drop_task(self, credit: Credit) -> None:
        """Schedule a task to handle the credit drop message from TimingManager via StickyCreditRouter.

        This method creates the credit context outside the task so it's available to the done callback.
        This simply schedules the task to be executed asynchronously and adds a done callback to
        ensure the credit is returned. It does not wait for it to actually execute.
        """
        drop_perf_ns = time.perf_counter_ns()
        credit_received_ns = self.clock_offset_tracker._clock.now_ns()
        self.clock_offset_tracker.update(credit.issued_at_ns)
        credit_context = CreditContext(
            credit=credit,
            drop_perf_ns=drop_perf_ns,
            credit_received_ns=credit_received_ns,
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
        self.execute_async(self.return_dealer_client.send(credit_return))
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
        credit_id = credit_context.credit.id
        try:
            if not self.inference_client:
                raise NotInitializedError("Inference server client not initialized.")
            with (
                self.event_loop_monitor.activity(f"credit id={credit_id} processing"),
                self._memory_profiler.track("process_credit"),
            ):
                await self._process_credit(credit_context)
            self._memory_profiler.on_request_complete()
        except asyncio.CancelledError:
            self.debug(lambda: f"Credit {credit_id} cancelled")
            credit_context.cancelled = True
        except Exception as e:
            self.exception(f"Error occurred while processing credit {credit_id}: {e!r}")
        finally:
            # ALWAYS return the credit here to ensure accurate tracking
            credit_return = CreditReturn(
                credit=credit_context.credit,
                cancelled=credit_context.cancelled,
                first_token_sent=credit_context.first_token_sent,
                error=str(credit_context.error) if credit_context.error else None,
            )
            with self.event_loop_monitor.activity(
                f"credit id={credit_id} sending CreditReturn"
            ):
                await self.return_dealer_client.send(credit_return)
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
                await self.return_dealer_client.send(
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
            # Store clock offset for cross-machine timestamp alignment.
            # Do NOT overwrite timestamp_ns — it was set at record creation
            # (pre-request) and serves as the wall-clock anchor for all
            # exported timestamps. Overwriting it post-request would shift
            # every exported timestamp forward by the request latency.
            record.clock_offset_ns = self.clock_offset_tracker.offset_ns
            await self._send_inference_result_message(record)

            # Copy request-level errors to credit context for CreditReturn tracking
            if record.error is not None:
                credit_context.error = record.error

            if session.should_store_response() and (
                resp_turn := await self._process_response(record)
            ):
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
            config=self.run.cfg,
            credit_num=credit.id,
            session_num=credit.session_num,
            credit_phase=credit.phase,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=session.x_correlation_id,
            conversation_id=session.conversation.session_id,
            turn_index=session.turn_index,
            turns=session.turn_list,
            drop_perf_ns=credit_context.drop_perf_ns,
            credit_issued_ns=credit.issued_at_ns,
            credit_received_ns=credit_context.credit_received_ns,
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
                        config=self.run.cfg,
                        conversation_id=conversation_id,
                        turn_index=0,
                        turns=[],
                        credit_num=credit_context.credit.id,
                        session_num=credit_context.credit.session_num,
                        credit_phase=credit_context.credit.phase,
                        x_request_id=str(uuid.uuid4()),
                        x_correlation_id=credit_context.credit.x_correlation_id,
                        drop_perf_ns=credit_context.drop_perf_ns,
                    ),
                    model_name=self.run.cfg.get_model_names()[0],
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    error=error,
                    clock_offset_ns=self.clock_offset_tracker.offset_ns,
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

        Offloaded to a thread because extract_response_data parses every SSE
        message (JSON decode + string ops) synchronously.  For long streaming
        responses this can block the event loop for 10ms+.

        Args:
            record: RequestRecord with raw responses from inference server

        Returns:
            Turn object for storing in session, or None if no content
        """
        return await asyncio.to_thread(self._process_response_sync, record)

    def _process_response_sync(self, record: RequestRecord) -> Turn | None:
        """Synchronous response processing — runs in a thread pool."""
        resp = self.inference_client.endpoint.extract_response_data(record)
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

    def _build_inference_wire_message(self, record: RequestRecord):
        """Build the msgspec worker->record-processor wire payload."""
        include_raw_export_fields = self.run.cfg.artifacts.raw
        raw_payload = None
        if include_raw_export_fields and record.request_info is not None:
            raw_payload = self.inference_client.endpoint.format_payload(
                record.request_info
            )
        return build_inference_results_wire_message(
            service_id=self.service_id,
            record=record,
            raw_payload=raw_payload,
            include_request_headers=include_raw_export_fields,
            include_status=include_raw_export_fields,
            include_trace_data=self.run.cfg.artifacts.trace,
        )

    def _serialize_inference_wire(self, record: RequestRecord) -> bytes:
        """Serialize the msgspec worker->record-processor wire payload."""
        return encode_inference_results_wire_message(
            self._build_inference_wire_message(record)
        )

    async def _send_inference_result_message(self, record: RequestRecord) -> None:
        """Send RequestRecord to RecordProcessor for metric calculation.

        All records (success and error) flow through this method to ensure consistent
        metric calculation and error tracking.

        Flow:
        1. Update task statistics (total and success/failure counts)
        2. Project record into the msgspec wire message
        3. Serialize in thread pool (keep heavy encoding off the event loop)
        4. Push pre-serialized bytes to RecordProcessor via PUSH socket

        Note: Serialization is awaited so callers can safely mutate ``record``
        afterwards (e.g. ``extract_response_data`` nulls out responses).
        The ZMQ push is fire-and-forget to avoid blocking on network I/O.
        """
        self.task_stats.task_finished(record.valid)

        wire_data = await asyncio.to_thread(self._serialize_inference_wire, record)
        self.execute_async(self.inference_results_push_client.push_raw(wire_data))

    async def _configure_for_profiling(self) -> None:
        """Wait for startup gates, then enable profiling-time instrumentation."""
        self.debug("Waiting for dataset to be configured before starting profiling")
        await asyncio.wait_for(
            self._dataset_configured_event.wait(),
            timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
        )
        self.debug("Waiting for WorkerDispatchable to be acknowledged before profiling")
        await asyncio.wait_for(
            self._worker_ready_event.wait(),
            timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
        )
        if self.is_debug_enabled:
            health = await asyncio.to_thread(self.get_process_health)
            memory_usage = health.memory_usage / BYTES_PER_MIB
            self.memory_usage_before_profiling = memory_usage
            pss = await asyncio.to_thread(self.get_pss_memory)
            pss_mib = pss / BYTES_PER_MIB if pss is not None else None
            self.debug(
                f"Memory before profiling: RSS={memory_usage:.2f} MiB, "
                f"PSS={pss_mib:.2f} MiB"
                if pss_mib is not None
                else f"Memory before profiling: RSS={memory_usage:.2f} MiB (PSS unavailable)"
            )

        self.event_loop_monitor.start()

        # Wire monitor into sub_client for message-level activity tracking
        if hasattr(self, "sub_client"):
            self.sub_client.event_loop_monitor = self.event_loop_monitor

        # Start memory profiler if enabled via environment
        self._memory_profiler.start()

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _on_profile_configure_command(self, message: Command) -> None:
        """Configure the worker."""
        await self._configure_for_profiling()

    @on_stop
    async def _worker_stop(self) -> None:
        # Stop memory profiler and log final stats
        self._memory_profiler.stop()

        if self.is_debug_enabled:
            health = await asyncio.to_thread(self.get_process_health)
            rss_mib = health.memory_usage / BYTES_PER_MIB
            pss = await asyncio.to_thread(self.get_pss_memory)
            pss_mib = pss / BYTES_PER_MIB if pss is not None else None
            before = self.memory_usage_before_profiling
            self.debug(
                f"Memory after profiling: RSS={rss_mib:.2f} MiB, "
                + (
                    f"PSS={pss_mib:.2f} MiB"
                    if pss_mib is not None
                    else "PSS=unavailable"
                )
                + (
                    f" (RSS delta={rss_mib - before:+.2f} MiB)"
                    if before is not None
                    else ""
                )
            )

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
