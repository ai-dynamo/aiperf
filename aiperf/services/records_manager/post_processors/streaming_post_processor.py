# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import DEFAULT_STREAMING_MAX_QUEUE_SIZE
from aiperf.common.decorators import implements_protocol
from aiperf.common.hooks import background_task
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.protocols import StreamingPostProcessorProtocol


@implements_protocol(StreamingPostProcessorProtocol)
class BaseStreamingPostProcessor(MessageBusClientMixin, ABC):
    """
    BaseStreamingPostProcessor is a base class for all classes that wish to stream the incoming
    ParsedResponseRecords.
    """

    def __init__(
        self,
        service_id: str,
        service_config: ServiceConfig,
        user_config: UserConfig,
        max_queue_size: int = DEFAULT_STREAMING_MAX_QUEUE_SIZE,
        **kwargs,
    ) -> None:
        self.service_id = service_id
        self.user_config = user_config
        self.service_config = service_config
        super().__init__(
            user_config=user_config,
            service_config=service_config,
            **kwargs,
        )
        self.info(
            lambda: f"Created streaming post processor: {self.__class__.__name__} with max_queue_size: {max_queue_size:,}"
        )
        self.records_queue: asyncio.Queue[ParsedResponseRecord] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.cancellation_event = asyncio.Event()

    @background_task(immediate=True, interval=None)
    async def _stream_records_task(self) -> None:
        """Task that streams records from the queue to the post processor's stream_record method."""
        while not self.stop_requested and not self.cancellation_event.is_set():
            try:
                record = await self.records_queue.get()
                await self.stream_record(record)
                self.records_queue.task_done()
            except asyncio.CancelledError:
                break

        if self.cancellation_event.is_set():
            self.debug(
                lambda: f"Streaming post processor {self.__class__.__name__} task cancelled, draining queue"
            )
            # Drain the rest of the queue
            while not self.records_queue.empty():
                _ = self.records_queue.get_nowait()
                self.records_queue.task_done()

        self.debug(
            lambda: f"Streaming post processor {self.__class__.__name__} task completed"
        )

    @abstractmethod
    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Handle the incoming record. This method should be implemented by the subclass."""
        raise NotImplementedError(
            "BaseStreamingPostProcessor.stream_record method must be implemented by the subclass."
        )

    async def process_records(self, cancelled: bool) -> Any:
        """Handle the process records command. This method is called when the records manager receives
        a command to process the records, and can be handled by the subclass. The results will be
        returned by the records manager to the caller.
        """
        pass
