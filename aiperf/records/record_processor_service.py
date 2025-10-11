# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    MessageType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.factories import (
    RecordProcessorFactory,
    ServiceFactory,
)
from aiperf.common.hooks import on_command, on_pull_message
from aiperf.common.messages import (
    InferenceResultsMessage,
    MetricRecordsMessage,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.common.protocols import (
    PushClientProtocol,
    RecordProcessorProtocol,
    RequestClientProtocol,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import compute_time_ns
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.parsers.inference_result_parser import InferenceResultParser


@ServiceFactory.register(ServiceType.RECORD_PROCESSOR)
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
            pull_client_max_concurrency=Environment.PULL_CLIENT_MAX_CONCURRENCY,
        )
        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
        )
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
            )
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

        self.records_processors: list[RecordProcessorProtocol] = []
        for processor_type in RecordProcessorFactory.get_all_class_types():
            try:
                processor: RecordProcessorProtocol = (
                    RecordProcessorFactory.create_instance(
                        processor_type,
                        service_config=self.service_config,
                        user_config=self.user_config,
                    )
                )
                self.records_processors.append(processor)
                self.attach_child_lifecycle(processor)
                self.debug(
                    f"Created record processor: {processor_type}: {processor.__class__.__name__}"
                )
            except PostProcessorDisabled:
                self.debug(
                    f"Record processor {processor_type} is disabled and will not be used"
                )

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
                self.tokenizers[model] = Tokenizer.from_pretrained(
                    self.user_config.tokenizer.name or model,
                    trust_remote_code=self.user_config.tokenizer.trust_remote_code,
                    revision=self.user_config.tokenizer.revision,
                )
            return self.tokenizers[model]

    def _create_metric_record_metadata(
        self, record: ParsedResponseRecord, worker_id: str
    ) -> MetricRecordMetadata:
        """Create a metric record metadata based on a parsed response record."""

        start_time_ns = record.timestamp_ns
        start_perf_ns = record.start_perf_ns

        # Convert all timestamps from perf_ns to time_ns for the user
        request_end_ns = compute_time_ns(
            start_time_ns,
            start_perf_ns,
            record.responses[-1].perf_ns if record.responses else record.end_perf_ns,
        )
        request_ack_ns = compute_time_ns(
            start_time_ns, start_perf_ns, record.recv_start_perf_ns
        )
        cancellation_time_ns = compute_time_ns(
            start_time_ns, start_perf_ns, record.cancellation_perf_ns
        )

        return MetricRecordMetadata(
            request_start_ns=start_time_ns,
            request_ack_ns=request_ack_ns,
            request_end_ns=request_end_ns,
            conversation_id=record.conversation_id,
            turn_index=record.turn_index,
            record_processor_id=self.service_id,
            benchmark_phase=record.credit_phase,
            x_request_id=record.x_request_id,
            x_correlation_id=record.x_correlation_id,
            session_num=record.credit_num,
            worker_id=worker_id,
            was_cancelled=record.was_cancelled,
            cancellation_time_ns=cancellation_time_ns,
        )

    @on_pull_message(MessageType.INFERENCE_RESULTS)
    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message."""
        parsed_record = await self.inference_result_parser.parse_request_record(
            message.record
        )
        raw_results = await self._process_record(parsed_record)
        results = []
        for result in raw_results:
            if isinstance(result, BaseException):
                self.warning(f"Error processing record: {result}")
            else:
                results.append(result)

        await self.records_push_client.push(
            MetricRecordsMessage(
                service_id=self.service_id,
                metadata=self._create_metric_record_metadata(
                    message.record, message.service_id
                ),
                results=results,
                error=message.record.error,
            )
        )

    async def _process_record(
        self, record: ParsedResponseRecord
    ) -> list[MetricRecordDict | BaseException]:
        """Stream a record to the records processors."""
        tasks = [
            processor.process_record(record) for processor in self.records_processors
        ]
        results: list[MetricRecordDict | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )
        return results


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordProcessor)


if __name__ == "__main__":
    main()
