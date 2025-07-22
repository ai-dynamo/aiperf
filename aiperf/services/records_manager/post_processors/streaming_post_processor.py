# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from aiperf.common.comms import PubClientProtocol, SubClientProtocol
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.hooks import aiperf_task
from aiperf.common.messages.command_messages import CommandMessage
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ParsedResponseRecord

DEFAULT_MAX_QUEUE_SIZE = 100_000


class BaseStreamingPostProcessor(AIPerfLifecycleMixin, ABC):
    """
    BaseStreamingPostProcessor is a base class for all classes that wish to stream the incoming
    ParsedResponseRecords.
    """

    def __init__(
        self,
        pub_client: PubClientProtocol,
        sub_client: SubClientProtocol,
        service_id: str,
        service_config: ServiceConfig,
        user_config: UserConfig,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        **kwargs,
    ) -> None:
        self.service_id = service_id
        self.user_config = user_config
        self.service_config = service_config
        self.pub_client = pub_client
        self.sub_client = sub_client
        super().__init__(
            pub_client=pub_client,
            sub_client=sub_client,
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

    @aiperf_task
    async def _stream_records_task(self) -> None:
        """Task that streams records from the queue to the post processor's stream_record method."""
        while True:
            try:
                record = await self.records_queue.get()
                await self.stream_record(record)
                self.records_queue.task_done()
            except asyncio.CancelledError:
                break

    @abstractmethod
    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Handle the incoming record. This method should be implemented by the subclass."""
        raise NotImplementedError(
            "BaseStreamingPostProcessor.stream_record method must be implemented by the subclass."
        )

    async def on_process_records_command(self, message: CommandMessage) -> Any:
        """Handle the process records command. This method is called when the records manager receives
        a command to process the records, and can be handled by the subclass. The results will be
        returned by the records manager to the caller.
        """
        pass
