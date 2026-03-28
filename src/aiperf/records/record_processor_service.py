# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import traceback
from typing import TYPE_CHECKING

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.channel_codecs import RAW_INFERENCE_CODEC, RECORDS_CODEC
from aiperf.common.control_structs import Command
from aiperf.common.enums import CommAddress, CommandType, ExportLevel, MessageType
from aiperf.common.hooks import on_command, on_pull_message, on_start, on_stop
from aiperf.common.pod_lifecycle_structs import (
    PodManagerToPeerMessage,
    PodPeerHello,
    PodPeerShutdown,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.inference_wire import (
    InferenceResultsWireMessage,
    wire_message_to_request_record,
)
from aiperf.common.messages import (
    MetricRecordsMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    MetricRecordMetadata,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.common.protocols import (
    PushClientProtocol,
    StreamingDealerClientProtocol,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import compute_time_ns
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType
from aiperf.post_processors.protocols import RecordProcessorProtocol
from aiperf.records.inference_result_parser import InferenceResultParser


class RecordProcessor(PullClientMixin, BaseComponentService):
    """RecordProcessor is responsible for processing the records and pushing them to the RecordsManager.
    This service is meant to be run in a distributed fashion, where the amount of record processors can be scaled
    based on the load of the system.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            pull_client_address=CommAddress.RAW_INFERENCE_PROXY_BACKEND,
            pull_client_bind=False,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
            pull_client_codec=RAW_INFERENCE_CODEC,
            **kwargs,
        )
        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
            codec=RECORDS_CODEC,
        )
        self.tokenizers: dict[str, Tokenizer] = {}
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.inference_result_parser = InferenceResultParser(run=self.run)
        self._pod_index = os.environ.get("AIPERF_POD_INDEX")
        self.pod_lifecycle_dealer_client: StreamingDealerClientProtocol | None = None
        if self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES:
            self.pod_lifecycle_dealer_client = (
                self.comms.create_streaming_dealer_client(
                    address=CommAddress.POD_LIFECYCLE,
                    identity=self.service_id,
                    bind=False,
                    decode_type=PodManagerToPeerMessage,
                )
            )

        self.records_processors: list[RecordProcessorProtocol] = []
        for entry in plugins.iter_entries(PluginType.RECORD_PROCESSOR):
            try:
                ProcessorClass = plugins.get_class(
                    PluginType.RECORD_PROCESSOR, entry.name
                )
                processor: RecordProcessorProtocol = ProcessorClass(
                    run=self.run,
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

    @on_start
    async def _register_with_worker_pod_manager(self) -> None:
        """Register this record processor with the pod-local lifecycle router."""
        if self.pod_lifecycle_dealer_client is None:
            return
        await self.pod_lifecycle_dealer_client.send(
            PodPeerHello(
                service_id=self.service_id,
                service_type=str(self.service_type),
                pod_index=self._pod_index,
            )
        )

    @on_stop
    async def _notify_worker_pod_manager_shutdown(self) -> None:
        """Notify WorkerPodManager that local record flushing is complete."""
        if self.pod_lifecycle_dealer_client is None:
            return
        await self.pod_lifecycle_dealer_client.send(
            PodPeerShutdown(
                service_id=self.service_id,
                service_type=str(self.service_type),
            )
        )

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(self, message: Command) -> None:
        """Configure the tokenizers."""
        await self.inference_result_parser.configure()

    async def get_tokenizer(self, model: str) -> Tokenizer:
        """Get the tokenizer for a given model."""
        async with self.tokenizer_lock:
            if model not in self.tokenizers:
                resolved_names = self.run.resolved.tokenizer_names
                if resolved_names and model in resolved_names:
                    tokenizer_name = resolved_names[model]
                    resolve_alias = False
                else:
                    tokenizer_name = (
                        self.run.cfg.tokenizer.name if self.run.cfg.tokenizer else None
                    ) or model
                    resolve_alias = True
                self.tokenizers[model] = await asyncio.to_thread(
                    Tokenizer.from_pretrained,
                    tokenizer_name,
                    trust_remote_code=self.run.cfg.tokenizer.trust_remote_code
                    if self.run.cfg.tokenizer
                    else False,
                    revision=self.run.cfg.tokenizer.revision
                    if self.run.cfg.tokenizer
                    else "main",
                    resolve_alias=resolve_alias,
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
        # credit_received_ns is captured directly from the worker's MonotonicClock
        # at the instant the credit arrives — same clock domain as credit_issued_ns.
        credit_received_ns = (
            record.request_info.credit_received_ns if record.request_info else None
        )

        return MetricRecordMetadata(
            request_num=record.request_info.credit_num,
            credit_issued_ns=record.request_info.credit_issued_ns,
            credit_received_ns=credit_received_ns,
            request_start_ns=start_time_ns,
            request_ack_ns=request_ack_ns,
            request_end_ns=request_end_ns,
            conversation_id=record.request_info.conversation_id,
            turn_index=record.request_info.turn_index,
            record_processor_id=self.service_id,
            benchmark_phase=record.request_info.credit_phase,
            x_request_id=record.request_info.x_request_id,
            x_correlation_id=record.request_info.x_correlation_id,
            session_num=record.request_info.session_num
            if record.request_info.session_num is not None
            else record.request_info.credit_num,
            worker_id=worker_id,
            was_cancelled=cancellation_time_ns is not None,
            cancellation_time_ns=cancellation_time_ns,
            clock_offset_ns=record.clock_offset_ns,
        )

    @on_pull_message(MessageType.INFERENCE_RESULTS)
    async def _on_inference_results(self, message: InferenceResultsWireMessage) -> None:
        """Handle an inference results message."""
        worker_id, record = wire_message_to_request_record(
            config=self.run.cfg,
            message=message,
        )

        # Capture last response timestamp before parsing frees raw SSE data.
        last_response_perf_ns = (
            record.responses[-1].perf_ns if record.responses else None
        )

        parsed_record = await self.inference_result_parser.parse_request_record(record)

        # Free raw SSE messages now that parsing extracted what it needs.
        # Skip when RAW export is active -- the raw writer needs them.
        if self.run.cfg.output.export_level != ExportLevel.RAW:
            record.responses = None

        metadata = self._create_metric_record_metadata(
            record, worker_id, last_response_perf_ns
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
        if self.run.cfg.output.export_level != ExportLevel.RAW:
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
