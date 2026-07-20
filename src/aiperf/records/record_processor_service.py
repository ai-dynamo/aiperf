# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
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
from aiperf.common.hooks import on_command, on_message, on_pull_message
from aiperf.common.messages import (
    DatasetConfiguredNotification,
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
from aiperf.common.protocols import PushClientProtocol
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
        self.tokenizers: dict[str, Tokenizer] = {}
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.model_endpoint: ModelEndpointInfo = ModelEndpointInfo.from_run(self.run)
        self.inference_result_parser = InferenceResultParser(
            run=self.run,
        )

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
        """Flush child record processors (e.g. RawRecordWriterProcessor buffers).

        RecordsManager sends PROFILE_COMPLETE after all records are processed
        but before exporting/aggregating results. Flushing children here ensures
        buffered writers drain to disk before the RawRecordAggregator reads them.

        We flush rather than stop: stop() runs the @on_stop hook chain inside
        the message-handler task, and when SystemController later broadcasts
        SHUTDOWN it cancels the in-flight handler task, leaving the writer
        wedged at STOPPING with the buffer un-flushed. flush_buffer() drains
        the buffer without tearing down the file handle, and the writer's
        normal _close_file hook handles teardown during service shutdown.
        """
        for child in self._children:
            flush = getattr(child, "flush_buffer", None)
            if flush is None:
                continue
            try:
                await flush()
            except Exception as e:  # noqa: BLE001
                self.error(f"Failed to flush child {child}: {e!r}")

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

        start_time_ns = record.timestamp_ns
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
            record_processor_id=self.service_id,
            benchmark_phase=record.request_info.credit_phase,
            x_request_id=record.request_info.x_request_id,
            x_correlation_id=record.request_info.x_correlation_id,
            session_num=record.request_info.credit_num,
            worker_id=worker_id,
            was_cancelled=cancellation_time_ns is not None,
            cancellation_time_ns=cancellation_time_ns,
            agent_depth=record.request_info.agent_depth,
            parent_correlation_id=record.request_info.parent_correlation_id,
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
            else:
                session_num = -1
                benchmark_phase = CreditPhase.PROFILING
            metadata = MetricRecordMetadata(
                session_num=session_num,
                request_start_ns=record.timestamp_ns,
                request_end_ns=record.timestamp_ns,
                worker_id=message.service_id,
                record_processor_id=self.service_id,
                benchmark_phase=benchmark_phase,
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

        We assign None to fields typed as non-optional lists (turns, responses) to let
        the GC reclaim the underlying objects. Using .clear() would keep the empty list
        alive, and reassigning [] would allocate a new object for no reason.
        """
        trace_data = record.trace_data
        error = record.error
        if self.run.cfg.artifacts.export_level != ExportLevel.RAW:
            record.responses = None
        record.turns = None
        record.trace_data = None
        record.request_headers = None
        if record.request_info:
            record.request_info.turns = None
            record.request_info.system_message = None
            record.request_info.user_context_message = None
        parsed_record.responses = None
        return trace_data, error


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.RECORD_PROCESSOR)


if __name__ == "__main__":
    main()
