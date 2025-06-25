# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys
import time
from collections import deque

from aiperf.common.config import ServiceConfig
from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.messages import InferenceResultsMessage, Message
from aiperf.common.record_models import RequestRecord
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.tokenizer import Tokenizer
from aiperf.parsers import OpenAIResponseExtractor


@ServiceFactory.register(ServiceType.RECORDS_MANAGER)
class RecordsManager(BaseComponentService):
    """
    The RecordsManager service is primarily responsible for holding the
    results returned from the workers.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing records manager")

        self.records: deque[RequestRecord] = deque()
        self.error_records: deque[RequestRecord] = deque()

        # Track per-worker statistics
        self.worker_request_counts: dict[str, int] = {}
        self.worker_error_counts: dict[str, int] = {}

        self.start_time_ns: int = time.time_ns()
        self.end_time_ns: int | None = None

        self.extractor = OpenAIResponseExtractor()
        self.tokenizers: dict[str, Tokenizer] = {}

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.RECORDS_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize records manager-specific components."""
        self.logger.debug("Initializing records manager")
        # TODO: Implement records manager initialization

    @on_start
    async def _start(self) -> None:
        """Start the records manager."""
        self.logger.debug("Starting records manager")
        # TODO: Implement records manager start

    @on_stop
    async def _stop(self) -> None:
        """Stop the records manager."""
        self.logger.debug("Stopping records manager")
        # TODO: Implement records manager stop

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up records manager-specific components."""
        self.logger.debug("Cleaning up records manager")
        # TODO: Implement records manager cleanup

    @on_configure
    async def _configure(self, message: Message) -> None:
        """Configure the records manager."""
        self.logger.debug(f"Configuring records manager with message: {message}")
        # TODO: Implement records manager configuration

    async def _on_inference_results_internal(
        self, message: InferenceResultsMessage
    ) -> None:
        """Handle an inference results message."""
        record = message.record
        worker_id = message.service_id

        # Initialize worker counters if not seen before
        if worker_id not in self.worker_request_counts:
            self.worker_request_counts[worker_id] = 0
        if worker_id not in self.worker_error_counts:
            self.worker_error_counts[worker_id] = 0

        if record.has_error:
            self.logger.warning("Received error inference results: %s", record)
            self.error_records.append(record)
            self.worker_error_counts[worker_id] += 1

        elif record.valid:
            self.logger.debug(
                "Received inference results: %f milliseconds. %f milliseconds.",
                record.time_to_first_response_ns / NANOS_PER_MILLIS
                if record.time_to_first_response_ns
                else None,
                record.time_to_last_response_ns / NANOS_PER_MILLIS
                if record.time_to_last_response_ns
                else None,
            )
            self.records.append(record)
            self.worker_request_counts[worker_id] += 1

            # TODO: Job of post-processors

            tokenizer = self.get_tokenizer(record.request["model"])
            total_tokens = 0
            resp = self.extractor.extract_response_data(record)
            tokens = []
            for r in resp:
                if r.parsed_text is not None:
                    tokens.extend(r.parsed_text)
            if tokens:
                for t in tokens:
                    try:
                        total_tokens += len(tokenizer.encode(t))
                    except Exception as e:
                        self.logger.error("Error encoding token '%s': %s", t, e)
                        continue
            self.logger.debug(
                "Received %d tokens, %d responses, %d total tokens",
                len(tokens),
                len(resp),
                total_tokens,
            )
            # TODO: need to tokenize the output

        else:
            self.logger.warning("Received invalid inference results: %s", record)
            self.error_records.append(record)
            self.worker_error_counts[worker_id] += 1

    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message."""
        _ = asyncio.create_task(self._on_inference_results_internal(message))


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    sys.exit(main())
