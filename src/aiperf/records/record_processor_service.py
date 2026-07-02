# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import traceback

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommAddress, CommandType, ExportLevel, MessageType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_command, on_message, on_pull_message
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    InferenceResultsMessage,
    MetricRecordsMessage,
    ProfileCompleteCommand,
    ProfileConfigureCommand,
)
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
from aiperf.common.scenario import get_scenario
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import compute_time_ns
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.post_processors.protocols import RecordProcessorProtocol
from aiperf.records.inference_result_parser import InferenceResultParser


class RecordProcessor(PullClientMixin, BaseComponentService):
    """RecordProcessor is responsible for processing the records and pushing them to the RecordsManager.
    This service is meant to be run in a distributed fashion, where the amount of record processors can be scaled
    based on the load of the system.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            pull_client_address=CommAddress.RAW_INFERENCE_PROXY_BACKEND,
            pull_client_bind=False,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
        )
        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
        )
        self.tokenizers: dict[str, Tokenizer] = {}
        self.user_config: UserConfig = user_config
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.model_endpoint: ModelEndpointInfo = ModelEndpointInfo.from_user_config(
            user_config
        )
        self.inference_result_parser = InferenceResultParser(
            service_config=service_config,
            user_config=user_config,
        )
        # Cache: drop context-overflow records entirely (don't push as errors)
        # when the active scenario uses AGENTIC_REPLAY timing. The trajectory
        # is already terminated by the timing strategy via the separate
        # CreditReturn path, so emitting an error record would just double-
        # count an event we intentionally tolerate.
        self._drop_agentic_overflow_records: bool = False
        scenario_name = getattr(user_config, "scenario", None)
        if scenario_name is not None:
            try:
                spec = get_scenario(scenario_name)
                self._drop_agentic_overflow_records = (
                    str(spec.timing_mode) == "agentic_replay"
                )
            except Exception:  # noqa: BLE001
                # Unknown scenario names are validated elsewhere; record
                # processing degrades to default error-emission behavior here.
                self._drop_agentic_overflow_records = False

        self.records_processors: list[RecordProcessorProtocol] = []
        for entry in plugins.iter_entries(PluginType.RECORD_PROCESSOR):
            try:
                ProcessorClass = plugins.get_class(
                    PluginType.RECORD_PROCESSOR, entry.name
                )
                processor: RecordProcessorProtocol = ProcessorClass(
                    service_config=self.service_config,
                    user_config=self.user_config,
                    service_id=self.service_id,
                )
                self.records_processors.append(processor)
                self.attach_child_lifecycle(processor)
                self.debug(
                    f"Created record processor: {entry.name}: {processor.__class__.__name__}"
                )
            except PostProcessorDisabled:
                self.debug(
                    f"Record processor {entry.name} is disabled and will not be used"
                )
            except Exception as e:
                self.exception(f"Error creating record processor: {e!r}")
                raise

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        for processor in self.records_processors:
            if hasattr(processor, "on_dataset_configured"):
                processor.on_dataset_configured(message.metadata)

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
                tokenizer_config = self.user_config.tokenizer
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
            source_trace_id=record.request_info.source_trace_id,
            source_outer_idx=record.request_info.source_outer_idx,
            source_inner_idx=record.request_info.source_inner_idx,
            source_kind=record.request_info.source_kind,
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
            root_correlation_id=record.request_info.root_correlation_id,
        )

    @on_pull_message(MessageType.INFERENCE_RESULTS)
    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message.

        Lockstep contract: every received message forwards exactly one
        ``MetricRecordsMessage``. The worker has already returned the credit as
        completed by the time the record arrives here, so a dropped record
        leaves the RecordsManager completion barrier (``success_records +
        error_records >= final_requests_completed``, which has no timeout)
        permanently short and hangs the run at end-of-phase. A parse/process
        failure is therefore forwarded as an error record instead of being
        allowed to escape the handler.
        """
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
            await self._forward_failed_record(message, record, last_response_perf_ns, e)

    async def _process_and_forward_record(
        self,
        message: InferenceResultsMessage,
        record: RequestRecord,
        last_response_perf_ns: int | None,
    ) -> None:
        """Parse, process, and forward the metric record for a single request."""
        parsed_record = await self.inference_result_parser.parse_request_record(record)

        # Free raw SSE messages now that parsing extracted what it needs.
        # Skip when RAW export is active -- the raw writer needs them.
        if self.user_config.output.export_level != ExportLevel.RAW:
            record.responses = None

        metadata = self._create_metric_record_metadata(
            record, message.service_id, last_response_perf_ns
        )

        # Flag context-overflow records for the records-side "skip" path when
        # the active scenario uses AGENTIC_REPLAY. RecordsManager will count
        # the record toward ``total_records`` (so the records-side counter
        # stays in lockstep with credit-side ``final_requests_completed``
        # and the completion barrier converges -- a previous version of this
        # code returned early here, which broke that invariant in one
        # direction only and hung the run at end-of-phase) but skip the
        # error tracker, accumulators, and stream exporters so the overflow
        # event doesn't show up in any user-facing metric.
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
        raw_results = await self._process_record(parsed_record, metadata)

        trace_data, error = self._free_record_data(record, parsed_record)

        results = []
        for result in raw_results:
            if isinstance(result, BaseException):
                self.error(
                    f"Error processing record: {result!r}: {traceback.format_exception(result)}"
                )
            else:
                results.append(result)

        await self.records_push_client.push(
            MetricRecordsMessage(
                service_id=self.service_id,
                metadata=metadata,
                results=results,
                trace_data=trace_data,
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
        metadata = self._create_metric_record_metadata(
            record, message.service_id, last_response_perf_ns
        )
        await self.records_push_client.push(
            MetricRecordsMessage(
                service_id=self.service_id,
                metadata=metadata,
                results=[],
                trace_data=None,
                error=record.error or ErrorDetails.from_exception(exc),
            )
        )

    def _free_record_data(
        self, record: RequestRecord, parsed_record: ParsedResponseRecord
    ) -> tuple[BaseTraceData | None, ErrorDetails | None]:
        """Free large data structures from the record after all processors have run.

        All metrics and post-processors consume these fields during _process_record().
        The only data sent downstream in MetricRecordsMessage is metadata, results,
        trace_data, and error -- so everything else can be released here.

        We assign None to fields typed as non-optional lists (turns, responses) to let
        the GC reclaim the underlying objects. Using .clear() would keep the empty list
        alive, and reassigning [] would allocate a new object for no reason.
        """
        trace_data = record.trace_data
        error = record.error
        if self.user_config.output.export_level != ExportLevel.RAW:
            record.responses = None
        record.trace_data = None
        record.request_headers = None
        parsed_record.responses = None
        return trace_data, error

    async def _process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> list[MetricRecordDict | BaseException]:
        """Stream a record to the records processors."""
        tasks = [
            processor.process_record(record, metadata)
            for processor in self.records_processors
        ]
        results: list[MetricRecordDict | BaseException | None] = await asyncio.gather(
            *tasks, return_exceptions=True
        )
        return [result for result in results if result is not None]


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.RECORD_PROCESSOR)


if __name__ == "__main__":
    main()
