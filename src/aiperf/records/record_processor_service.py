# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import traceback
from typing import TYPE_CHECKING

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    ExportLevel,
    MemoryMapFormat,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_command, on_message, on_pull_message, on_stop
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    InferenceResultsMessage,
    MetricRecordsMessage,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    MemoryMapClientMetadata,
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
from aiperf.dataset.protocols import DatasetClientStoreProtocol
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.post_processors.protocols import RecordProcessorProtocol
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

        self.records_processors: list[RecordProcessorProtocol] = []
        self._dataset_client: DatasetClientStoreProtocol | None = None
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

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        turn_metadata_by_conversation = {
            conv.conversation_id: tuple(conv.turns)
            for conv in message.metadata.conversations
        }

        client_metadata = message.client_metadata
        if (
            isinstance(client_metadata, MemoryMapClientMetadata)
            and client_metadata.format == MemoryMapFormat.PAYLOAD_BYTES
            and self._dataset_client is None
        ):
            ClientStoreClass = plugins.get_class(
                PluginType.DATASET_CLIENT_STORE, client_metadata.client_type
            )
            self._dataset_client = ClientStoreClass(client_metadata=client_metadata)
            await self._dataset_client.initialize()

        self.inference_result_parser.on_dataset_configured(
            turn_metadata_by_conversation=turn_metadata_by_conversation,
            dataset_client=self._dataset_client,
        )

        for processor in self.records_processors:
            if hasattr(processor, "on_dataset_configured"):
                processor.on_dataset_configured(message.metadata)

    @on_stop
    async def _stop_dataset_client(self) -> None:
        if self._dataset_client is not None:
            dataset_client = self._dataset_client
            self._dataset_client = None
            await dataset_client.stop()

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the tokenizers."""
        await self.inference_result_parser.configure()

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

        mi = record.metric_inputs
        return MetricRecordMetadata(
            credit_issued_ns=mi.credit_issued_ns if mi else None,
            request_start_ns=start_time_ns,
            request_ack_ns=request_ack_ns,
            request_end_ns=request_end_ns,
            conversation_id=mi.conversation_id if mi else None,
            turn_index=mi.turn_index if mi else None,
            record_processor_id=self.service_id,
            benchmark_phase=mi.credit_phase if mi else None,
            x_request_id=mi.x_request_id if mi else None,
            x_correlation_id=mi.x_correlation_id if mi else None,
            session_num=mi.credit_num if mi else None,
            worker_id=worker_id,
            was_cancelled=cancellation_time_ns is not None,
            cancellation_time_ns=cancellation_time_ns,
            agent_depth=mi.agent_depth if mi else 0,
            parent_correlation_id=mi.parent_correlation_id if mi else None,
        )

    @on_pull_message(MessageType.INFERENCE_RESULTS)
    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message."""
        record = message.record

        # Capture last response timestamp before parsing frees raw SSE data.
        last_response_perf_ns = (
            record.responses[-1].perf_ns if record.responses else None
        )

        parsed_record = await self.inference_result_parser.parse_request_record(record)

        metadata = self._create_metric_record_metadata(
            record, message.service_id, last_response_perf_ns
        )
        raw_results = await self._process_record(parsed_record, metadata)

        trace_data, error = self._free_record_data(record, parsed_record)

        metrics = self._merge_metric_results(raw_results)

        await self.records_push_client.push(
            MetricRecordsMessage(
                service_id=self.service_id,
                metadata=metadata,
                metrics=metrics,
                trace_data=trace_data,
                error=error,
            )
        )

    def _merge_metric_results(
        self, raw_results: list[MetricRecordDict | BaseException]
    ) -> MetricRecordDict:
        """Merge record processor metric dicts and log processor exceptions."""
        metrics: MetricRecordDict = {}
        for result in raw_results:
            if isinstance(result, BaseException):
                self.error(
                    f"Error processing record: {result!r}: {traceback.format_exception(result)}"
                )
                continue
            for tag, value in result.items():
                if tag in metrics:
                    self.warning(
                        f"Duplicate metric tag '{tag}' found in results. "
                        f"Overwriting previous value {metrics[tag]} with {value}."
                    )
                metrics[tag] = value
        return metrics

    def _free_record_data(
        self, record: RequestRecord, parsed_record: ParsedResponseRecord
    ) -> tuple[BaseTraceData | None, ErrorDetails | None]:
        """Free large data structures from the record after all processors have run.

        All metrics and post-processors consume these fields during _process_record().
        The only data sent downstream in MetricRecordsMessage is metadata, metrics,
        trace_data, and error -- so everything else can be released here.

        Drops the inline ``metric_inputs.payload_bytes`` blob (the largest per-record
        allocation on the CONVERSATION-format path) for non-RAW exports; RAW export
        needs nothing more from the record at this point either, but the raw writer
        has already serialized via ``parsed_record.payload_dict`` so freeing here is
        safe regardless.
        """
        trace_data = record.trace_data
        error = record.error
        if self.run.cfg.artifacts.export_level != ExportLevel.RAW:
            record.responses = None
        record.trace_data = None
        record.request_headers = None
        if record.metric_inputs is not None:
            # ``MetricInputs.payload_bytes`` is the largest per-record
            # allocation; clear it once metrics have been computed and any
            # downstream consumer (raw export) has materialized it.
            record.metric_inputs.payload_bytes = None
        parsed_record.responses = None
        parsed_record.payload_dict = None
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
