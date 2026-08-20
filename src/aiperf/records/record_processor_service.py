# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import os
from typing import TYPE_CHECKING, Any

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    ExportLevel,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import (
    on_command,
    on_message,
    on_pull_message,
    on_start,
    on_stop,
)
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    FinalizeArtifactsCommand,
    InferenceResultsMessage,
    ProfileCompleteCommand,
    ProfileConfigureCommand,
    RecordsMessage,
)
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    MetricRecordMetadata,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.common.pod_lifecycle_structs import (
    GroupManagerToPeerMessage,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupPeerShutdown,
    GroupTokenizerReady,
    _send_group_peer_hello_with_retry,
)
from aiperf.common.protocols import PushClientProtocol, StreamingDealerClientProtocol
from aiperf.common.scenario import get_scenario
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import compute_time_ns
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.post_processors.protocols import (
    RecordObserverProtocol,
    RecordProcessorProtocol,
)
from aiperf.post_processors.record_observer_context import RecordObserverContext
from aiperf.records.dataset_gate import await_dataset_configured
from aiperf.records.inference_result_parser import InferenceResultParser

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class RecordProcessor(PullClientMixin, BaseComponentService):
    """RecordProcessor is responsible for processing the records and pushing them to the RecordsManager.
    This service is meant to be run in a distributed fashion, where the amount of record processors can be scaled
    based on the load of the system.
    """

    def __init__(
        self,
        run: "BenchmarkRun",
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            pull_client_address=CommAddress.RAW_INFERENCE_PROXY_BACKEND,
            pull_client_bind=False,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
            **kwargs,
        )
        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
        )
        # Group-local lifecycle channel to this pod's WorkerGroupManager.
        # The WGM counts registered record-processor peers to report
        # ready_record_processors, and waits for each to announce a clean
        # shutdown before uploading raw records. Without this registration the
        # count is permanently 0 and every Kubernetes run burns the full
        # RAW_RECORD_UPLOAD_TIMEOUT waiting for a shutdown that never arrives.
        #
        # Kubernetes only, matching Worker._is_group_managed_mode. A local
        # multiprocessing run has no WorkerGroupManager -- MultiProcessService
        # Manager spawns every service directly and never goes through
        # SubprocessManager's group-manager boundary -- so gating on
        # runtime.uses_worker_group_manager (True for MULTIPROCESSING) opened a
        # DEALER onto an unbound ipc:// endpoint and then retried GroupPeerHello
        # against it for up to GROUP_HELLO_TOTAL_TIMEOUT on every local run.
        self._pod_index: str | None = os.environ.get("AIPERF_POD_INDEX")
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
            # Without a receiver nothing answers GroupPeerCommand, and the WGM
            # blocks the full PROFILE_CONFIGURE_TIMEOUT waiting for an ack.
            self.pod_lifecycle_dealer_client.register_receiver(
                self._on_pod_lifecycle_message
            )
        self._tokenizer_bundles: dict[str, str] = {}
        self._tokenizer_ready: asyncio.Event = asyncio.Event()
        self.tokenizers: dict[str, Tokenizer] = {}
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.model_endpoint: ModelEndpointInfo = ModelEndpointInfo.from_run(self.run)
        self.inference_result_parser = InferenceResultParser(
            run=self.run,
        )
        # Cache: flag context-overflow records for the records-side "skip" path
        # (not dropped -- they still count toward total_records) when the active
        # scenario uses AGENTIC_REPLAY timing. The trajectory is already
        # terminated by the timing strategy via the separate CreditReturn path,
        # so the overflow event is intentionally tolerated and kept out of every
        # user-facing metric while the run still terminates cleanly.
        self._drop_agentic_overflow_records: bool = False
        scenario_name = self.run.cfg.scenario
        if scenario_name is not None:
            try:
                spec = get_scenario(scenario_name)
                self._drop_agentic_overflow_records = (
                    str(spec.timing_mode) == "agentic_replay"
                )
            except Exception:
                # Unknown scenario names are validated elsewhere; record
                # processing degrades to default error-emission behavior here.
                self._drop_agentic_overflow_records = False

        # DatasetConfiguredNotification (SUB) and inference results (PULL) arrive on
        # independent channels with no ordering guarantee. Gate record processing on
        # this event so processors are configured (e.g. accuracy ground truths) before
        # any record is graded.
        self._dataset_configured_event: asyncio.Event = asyncio.Event()

        # Stage 1 - PRODUCERS: parse a record and emit one typed result on the
        # record_type channel declared in plugins.yaml metadata. Grouped by that
        # declared channel (no runtime type-sniffing).
        self._producers: list[tuple[str, RecordProcessorProtocol]] = []
        for entry in plugins.iter_entries(PluginType.RECORD_PROCESSOR):
            try:
                ProducerClass = plugins.get_class(
                    PluginType.RECORD_PROCESSOR, entry.name
                )
                producer: RecordProcessorProtocol = ProducerClass(
                    run=self.run,
                    service_id=self.service_id,
                )
                record_type = entry.metadata["record_type"]
                self._producers.append((record_type, producer))
                self.attach_child_lifecycle(producer)
                self.debug(
                    f"Created record producer: {entry.name} ({record_type}): {producer.__class__.__name__}"
                )
            except PostProcessorDisabled:
                self.debug(
                    f"Record producer {entry.name} is disabled and will not be used"
                )
            except Exception as e:
                self.exception(f"Error creating record producer: {e!r}")
                raise

        # Stage 2 - OBSERVERS: view the produced results + the record and act
        # (e.g. write JSONL). They return nothing and emit no channel record.
        self._observers: list[RecordObserverProtocol] = []
        for entry in plugins.iter_entries(PluginType.RECORD_OBSERVER):
            try:
                ObserverClass = plugins.get_class(
                    PluginType.RECORD_OBSERVER, entry.name
                )
                observer: RecordObserverProtocol = ObserverClass(
                    run=self.run,
                    service_id=self.service_id,
                )
                self._observers.append(observer)
                self.attach_child_lifecycle(observer)
                self.debug(
                    f"Created record observer: {entry.name}: {observer.__class__.__name__}"
                )
            except PostProcessorDisabled:
                self.debug(
                    f"Record observer {entry.name} is disabled and will not be used"
                )
            except Exception as e:
                self.exception(f"Error creating record observer: {e!r}")
                raise

    def _is_group_managed_mode(self) -> bool:
        """Check if a WorkerGroupManager owns this processor's pod lifecycle."""
        return str(self.run.cfg.runtime.service_run_type).lower() == "kubernetes"

    @on_start
    async def _register_with_worker_group_manager(self) -> None:
        """Announce this record processor to the pod's WorkerGroupManager.

        The WGM derives ``ready_record_processors`` from its registered peers,
        so without this every pod reports 0 record processors regardless of
        health.
        """
        if self.pod_lifecycle_dealer_client is None:
            return
        try:
            await _send_group_peer_hello_with_retry(
                self.pod_lifecycle_dealer_client,
                service_id=self.service_id,
                service_type=str(self.service_type),
                pod_index=self._pod_index,
                logger=self,
            )
        except TimeoutError as e:
            # Pod accounting, not a functional dependency: an unacked hello
            # only means the WGM undercounts ready record processors. Records
            # still flow. Startup hooks fail fast now, so letting this
            # propagate would kill a processor that is otherwise healthy --
            # and it propagates in exactly the topologies that have no WGM to
            # ack in the first place.
            self.warning(
                f"WorkerGroupManager never acked this record processor's "
                f"registration ({e}); continuing. The pod will under-report "
                f"ready_record_processors."
            )

    @on_stop
    async def _notify_worker_group_manager_shutdown(self) -> None:
        """Tell the WorkerGroupManager this processor has finished flushing.

        The WGM blocks its raw-record upload on hearing this from every
        declared record processor. Without it the upload waits out the full
        RAW_RECORD_UPLOAD_TIMEOUT and then proceeds anyway, which both delays
        every run and risks uploading before local records are flushed.
        """
        if self.pod_lifecycle_dealer_client is None:
            return
        try:
            await self.pod_lifecycle_dealer_client.send(
                GroupPeerShutdown(
                    service_id=self.service_id,
                    service_type=str(self.service_type),
                )
            )
        except Exception as e:  # noqa: BLE001 - best-effort; the WGM may already have left the channel
            self.warning(
                f"Failed to send GroupPeerShutdown (peer already disconnected?): {e!r}"
            )

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        for _record_type, producer in self._producers:
            if hasattr(producer, "on_dataset_configured"):
                producer.on_dataset_configured(message.metadata)
        for observer in self._observers:
            if hasattr(observer, "on_dataset_configured"):
                observer.on_dataset_configured(message.metadata)
        self._dataset_configured_event.set()

    async def _on_pod_lifecycle_message(
        self, message: GroupManagerToPeerMessage
    ) -> None:
        """Dispatch group-local messages from the WorkerGroupManager."""
        if isinstance(message, GroupTokenizerReady):
            await self._on_tokenizer_ready(message)
        elif isinstance(message, GroupPeerCommand):
            await self._handle_pod_peer_command(message)

    async def _on_tokenizer_ready(self, message: GroupTokenizerReady) -> None:
        """Adopt the bundles the WorkerGroupManager downloaded for this pod."""
        if not message.success:
            # No tokenizer means this processor can never handle a record;
            # exit so kubelet restarts the pod alongside the WGM's own retry.
            self.error(
                f"Tokenizer download failed in WorkerGroupManager: "
                f"{message.error_message}; force-exiting {self.service_id}"
            )
            os._exit(1)
        self._tokenizer_bundles.update(message.bundles)
        self.inference_result_parser._tokenizer_bundles.update(message.bundles)
        self._tokenizer_ready.set()
        self.info(
            f"Tokenizer bundles ready: {sorted(self._tokenizer_bundles)} "
            f"(advertised by {message.service_id})"
        )

    async def _handle_pod_peer_command(self, message: GroupPeerCommand) -> None:
        """Run a group-local lifecycle command and acknowledge it.

        The WGM fans these out and blocks on an ack from every registered peer,
        bounded by PROFILE_CONFIGURE_TIMEOUT. An unanswered command stalls the
        whole pod for that timeout, twice per run.
        """
        if self.pod_lifecycle_dealer_client is None:
            return
        if message.command == str(CommandType.PROFILE_CONFIGURE):
            await self.inference_result_parser.configure()
        elif message.command == str(CommandType.FINALIZE_ARTIFACTS):
            await self._finalize_local_artifacts()
        elif message.command == str(CommandType.SHUTDOWN):
            # Ack before stopping: self.stop() tears down the comms children,
            # which includes this dealer socket, so an ack sent afterwards
            # never reaches the WGM and it blocks the full timeout.
            with contextlib.suppress(Exception):
                await self.pod_lifecycle_dealer_client.send(
                    GroupPeerCommandAck(cid=message.cid, service_id=self.service_id)
                )
            await self.stop()
            return
        elif message.command == str(CommandType.ABORT):
            # The WGM has failed its own lifecycle. Hard-exit so kubelet
            # restarts this container alongside it: a clean self.stop() would
            # exit 0 and leave the pod Ready with no record processors. The ack
            # is best-effort -- the WGM is already on its way out.
            self.error(
                f"Received ABORT from WorkerGroupManager; force-exiting "
                f"{self.service_id} so kubelet restarts this container"
            )
            with contextlib.suppress(Exception):
                await self.pod_lifecycle_dealer_client.send(
                    GroupPeerCommandAck(cid=message.cid, service_id=self.service_id)
                )
            os._exit(1)
        else:
            # No ack: acknowledging would tell the WGM the command succeeded.
            self.warning(f"Unknown group-local command: {message.command}")
            return
        await self.pod_lifecycle_dealer_client.send(
            GroupPeerCommandAck(cid=message.cid, service_id=self.service_id)
        )

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the tokenizers."""
        await self.inference_result_parser.configure()

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _profile_complete_command(
        self,
        message: ProfileCompleteCommand,  # noqa: ARG002
    ) -> None:
        """Finalize child record artifacts before result aggregation.

        RecordsManager sends PROFILE_COMPLETE after all records are processed
        but before exporting/aggregating results. Flushing children here ensures
        buffered writers drain to disk before the RawRecordAggregator reads them.

        Writers without a dedicated artifact finalizer are flushed in place.
        RawRecordWriterProcessor additionally closes its staging file so the
        local RawRecordAggregator can read and remove it on Windows.
        """
        await self._finalize_local_artifacts()

    @on_command(CommandType.FINALIZE_ARTIFACTS)
    async def _finalize_artifacts_command(
        self,
        message: FinalizeArtifactsCommand,  # noqa: ARG002
    ) -> None:
        """Acknowledge only after every local artifact writer is durable."""
        await self._finalize_local_artifacts()

    async def _finalize_local_artifacts(self) -> None:
        """Finalize every child writer, propagating any incomplete artifact."""
        children = []
        for child in self._children:
            finalizer = getattr(type(child), "finalize_artifact", None)
            finalize = (
                child.finalize_artifact
                if callable(finalizer)
                else getattr(child, "flush_buffer", None)
            )
            if finalize is not None:
                children.append((child, finalize))
        results = await asyncio.gather(
            *(finalize() for _child, finalize in children), return_exceptions=True
        )
        failures: list[Exception] = []
        for (child, _finalize), result in zip(children, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                failures.append(
                    RuntimeError(f"Failed to finalize child {child}: {result!r}")
                )
        if failures:
            raise ExceptionGroup(
                f"Failed to finalize {len(failures)} record artifact writer(s)",
                failures,
            )

    async def get_tokenizer(self, model: str) -> Tokenizer:
        """Get the tokenizer for a given model."""
        async with self.tokenizer_lock:
            if model not in self.tokenizers:
                tokenizer_config = self.run.cfg.tokenizer
                self.tokenizers[model] = await asyncio.to_thread(
                    Tokenizer.from_pretrained,
                    tokenizer_config.get_tokenizer_name_for_model(model),
                    trust_remote_code=tokenizer_config.trust_remote_code,
                    revision=tokenizer_config.revision,
                    resolve_alias=tokenizer_config.should_resolve_alias,
                )
            return self.tokenizers[model]

    def _create_metric_record_metadata(
        self,
        record: RequestRecord,
        worker_id: str,
        last_response_perf_ns: int | None = None,
    ) -> MetricRecordMetadata:
        """Create a metric record metadata based on a parsed response record."""

        # Controller frame: request_start/ack/end below are all derived from this
        # anchor and are exported and compared against credit_issued_ns, which the
        # controller stamped. Correcting once here converts the whole record's
        # exported timeline; record.timestamp_ns stays raw for provenance.
        start_time_ns = record.controller_timestamp_ns
        start_perf_ns = record.start_perf_ns

        end_perf_ns = (
            last_response_perf_ns or record.end_perf_ns or record.start_perf_ns
        )

        # Convert all timestamps from perf_ns to time_ns for the user
        request_end_ns = compute_time_ns(
            start_time_ns,
            start_perf_ns,
            end_perf_ns,
        )
        request_ack_ns = compute_time_ns(
            start_time_ns, start_perf_ns, record.recv_start_perf_ns
        )
        cancellation_time_ns = compute_time_ns(
            start_time_ns, start_perf_ns, record.cancellation_perf_ns
        )

        return MetricRecordMetadata(
            credit_issued_ns=record.request_info.credit_issued_ns,
            request_start_ns=start_time_ns,
            request_ack_ns=request_ack_ns,
            request_end_ns=request_end_ns,
            conversation_id=record.request_info.conversation_id,
            turn_index=record.request_info.turn_index,
            source_trace_id=record.request_info.source_trace_id,
            source_outer_idx=record.request_info.source_outer_idx,
            source_inner_idx=record.request_info.source_inner_idx,
            source_kind=record.request_info.source_kind,
            record_processor_id=self.service_id,
            benchmark_phase=record.request_info.credit_phase,
            phase_index=record.request_info.phase_index,
            profiling_index=record.request_info.profiling_index,
            phase_name=record.request_info.phase_name,
            phase_kind=record.request_info.phase_kind,
            x_request_id=record.request_info.x_request_id,
            x_correlation_id=record.request_info.x_correlation_id,
            session_num=record.request_info.credit_num,
            worker_id=worker_id,
            was_cancelled=cancellation_time_ns is not None,
            cancellation_time_ns=cancellation_time_ns,
            agent_depth=record.request_info.agent_depth,
            parent_correlation_id=record.request_info.parent_correlation_id,
            root_correlation_id=record.request_info.root_correlation_id,
        )

    @on_pull_message(MessageType.INFERENCE_RESULTS)
    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message.

        Lockstep contract: every received message forwards exactly one
        ``RecordsMessage``. The worker has already returned the credit as
        completed by the time the record arrives here, so a dropped record
        leaves the RecordsManager completion barrier (``success_records +
        error_records >= final_requests_completed``, which has no timeout)
        permanently short and hangs the run at end-of-phase. A parse/process
        failure is therefore forwarded as an error record instead of being
        allowed to escape the handler. The dataset-configured gate below is
        the one exception: its False path has already killed the service and
        aborted the run, so no barrier is left waiting.
        """
        if not await await_dataset_configured(self, self._dataset_configured_event):
            return
        record = message.record

        # Capture last response timestamp before parsing frees raw SSE data.
        last_response_perf_ns = (
            record.responses[-1].perf_ns if record.responses else None
        )

        try:
            await self._process_and_forward_record(
                message, record, last_response_perf_ns
            )
        except Exception as e:  # noqa: BLE001
            # Never drop the record: the worker already returned this credit as
            # completed, so forward an error record to keep the records-side
            # count in lockstep and let the completion barrier converge.
            self.exception(
                f"Failed to process inference record; forwarding as error: {e!r}"
            )
            # Last-resort guard: a failure inside the error-forward path must not
            # propagate out of the handler, or the timeout-less completion barrier
            # hangs the run (see docstring). Log and swallow.
            try:
                await self._forward_failed_record(
                    message, record, last_response_perf_ns, e
                )
            except Exception as forward_exc:  # noqa: BLE001
                self.exception(
                    f"Failed to forward error record; dropping to avoid escaping handler: {forward_exc!r}"
                )

    async def _process_and_forward_record(
        self,
        message: InferenceResultsMessage,
        record: RequestRecord,
        last_response_perf_ns: int | None,
    ) -> None:
        """Parse, produce, observe, and forward the records for a single request."""
        parsed_record = await self.inference_result_parser.parse_request_record(record)

        # Free raw SSE messages now that parsing extracted what it needs.
        # Skip when RAW export is active -- the raw writer needs them.
        if self.run.cfg.artifacts.export_level != ExportLevel.RAW:
            record.responses = None

        metadata = self._create_metric_record_metadata(
            record, message.service_id, last_response_perf_ns
        )

        # Flag context-overflow records for the records-side "skip" path when
        # the active scenario uses AGENTIC_REPLAY. RecordsManager will count
        # the record toward ``total_records`` (so the records-side counter
        # stays in lockstep with credit-side ``final_requests_completed``
        # and the completion barrier converges -- returning early here instead
        # would break that invariant in one direction only and hang the run at
        # end-of-phase) but skip the error tracker, accumulators, and stream
        # exporters so the overflow event doesn't show up in any user-facing
        # metric.
        if self._drop_agentic_overflow_records and getattr(
            record, "context_overflow", False
        ):
            metadata = metadata.model_copy(update={"context_overflow_skip": True})
            self.debug(
                lambda r=record: (
                    f"AGENTIC_REPLAY: flagging context-overflow record as "
                    f"metrics-skip (credit={r.request_info.credit_num} "
                    f"conv={r.request_info.conversation_id} "
                    f"turn={r.request_info.turn_index})"
                )
            )

        # Stage 1 - producers: run concurrently, group outputs by declared channel.
        by_type: dict[str, list[Any]] = {}
        producer_results = await asyncio.gather(
            *[
                producer.process_record(parsed_record, metadata)
                for _record_type, producer in self._producers
            ],
            return_exceptions=True,
        )
        for (record_type, _producer), result in zip(
            self._producers, producer_results, strict=True
        ):
            if isinstance(result, BaseException):
                self.error(f"Error in producer for {record_type}: {result!r}")
                continue
            if result is None:
                continue
            by_type.setdefault(record_type, []).append(result)

        # Snapshot the wire payload BEFORE observers run: ``produced`` is read-only
        # by contract, but a misbehaving observer that mutated ``by_type`` must not
        # be able to change what RecordsManager ingests.
        all_records = [record for records in by_type.values() for record in records]

        # Stage 2 - observers: view the produced results + the record and act.
        # Must run BEFORE _free_record_data so they can read the full parsed
        # record via ctx.record.
        ctx = RecordObserverContext(
            record=parsed_record,
            metadata=metadata,
            produced=by_type,
        )
        observer_results = await asyncio.gather(
            *[observer.observe(ctx) for observer in self._observers],
            return_exceptions=True,
        )
        for observer, result in zip(self._observers, observer_results, strict=True):
            if isinstance(result, BaseException):
                self.error(
                    f"Error in observer {observer.__class__.__name__}: {result!r}"
                )

        _trace_data, error = self._free_record_data(record, parsed_record)

        # Ship generically: ONE RecordsMessage per inference record carries the
        # request envelope (metadata + request-level error) plus every produced
        # typed record flattened into one list. Each record self-identifies via
        # its own serialized record_type field, so no per-type message class or
        # builder map is needed. Always pushed (even when no producer emitted a record)
        # to keep the RecordsManager completion barrier in lockstep with the
        # credit. The metric producer already put trace_data inside its
        # MetricRecordsData, so trace_data is not carried on the envelope.
        # The push is atomic (whole message serialized, then one NOBLOCK frame
        # send): a failure delivers nothing, so it propagates to the outer handler
        # which forwards exactly one error record -- no partial send, no
        # double-count, and the completion barrier stays in lockstep.
        await self.records_push_client.push(
            RecordsMessage(
                service_id=self.service_id,
                metadata=metadata,
                records=all_records,
                error=error,
            )
        )

    async def _forward_failed_record(
        self,
        message: InferenceResultsMessage,
        record: RequestRecord,
        last_response_perf_ns: int | None,
        exc: Exception,
    ) -> None:
        """Forward an error record after a parse/process failure so the
        records-side count stays in lockstep with the already-returned credit."""
        try:
            metadata = self._create_metric_record_metadata(
                record, message.service_id, last_response_perf_ns
            )
        except Exception as meta_exc:  # noqa: BLE001
            # Metadata creation itself can fail (e.g. request_info is None, which
            # was often the original failure cause). Fall back to a minimal record
            # built only from always-available fields so lockstep is preserved.
            self.exception(
                f"Failed to build metric record metadata for error record; using fallback: {meta_exc!r}"
            )
            if record.request_info is not None:
                session_num = record.request_info.credit_num
                benchmark_phase = record.request_info.credit_phase
                phase_index = record.request_info.phase_index
                profiling_index = record.request_info.profiling_index
                phase_name = record.request_info.phase_name
                phase_kind = record.request_info.phase_kind
            else:
                session_num = -1
                benchmark_phase = CreditPhase.PROFILING
                phase_index = None
                profiling_index = None
                phase_name = None
                phase_kind = None
            metadata = MetricRecordMetadata(
                session_num=session_num,
                request_start_ns=record.controller_timestamp_ns,
                request_end_ns=record.controller_timestamp_ns,
                worker_id=message.service_id,
                record_processor_id=self.service_id,
                benchmark_phase=benchmark_phase,
                phase_index=phase_index,
                profiling_index=profiling_index,
                phase_name=phase_name,
                phase_kind=phase_kind,
            )
        error = record.error or ErrorDetails.from_exception(exc)
        # The producers didn't run, so ship a RecordsMessage carrying a single
        # errored MetricRecordsData (empty metrics) so the accumulator still sees
        # the record and the records-tracker lockstep counts it.
        await self.records_push_client.push(
            RecordsMessage(
                service_id=self.service_id,
                metadata=metadata,
                records=[MetricRecordsData(metadata=metadata, metrics={}, error=error)],
                error=error,
            )
        )

    def _free_record_data(
        self, record: RequestRecord, parsed_record: ParsedResponseRecord
    ) -> tuple[BaseTraceData | None, ErrorDetails | None]:
        """Free large data structures from the record after all processors have run.

        All metrics and post-processors consume these fields during _process_record().
        The only data sent downstream is the typed records produced for this request
        (metadata, metrics, trace_data, error) -- so everything else can be released here.

        We assign None to fields typed as non-optional lists (responses) to let
        the GC reclaim the underlying objects. Using .clear() would keep the empty list
        alive, and reassigning [] would allocate a new object for no reason.
        """
        trace_data = record.trace_data
        error = record.error
        if self.run.cfg.artifacts.export_level != ExportLevel.RAW:
            record.responses = None
        record.trace_data = None
        record.request_headers = None
        parsed_record.responses = None
        return trace_data, error


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.RECORD_PROCESSOR)


if __name__ == "__main__":
    main()
