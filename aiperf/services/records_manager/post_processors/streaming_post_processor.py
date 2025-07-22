# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import abstractmethod

from aiperf.common.comms import PubClientProtocol, SubClientProtocol
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.hooks import aiperf_task
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.utils import yield_to_event_loop

DEFAULT_MAX_QUEUE_SIZE = 100_000


class StreamingPostProcessor(AIPerfLifecycleMixin):
    """
    StreamingPostProcessor is a base class for all classes that wish to stream the incoming
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
        self.debug(
            lambda: f"Initializing StreamingPostProcessor: {self.__class__.__name__} with max_queue_size: {max_queue_size}"
        )
        self.records_queue: asyncio.Queue[ParsedResponseRecord] = asyncio.Queue(
            maxsize=max_queue_size
        )

    @aiperf_task
    async def _stream_records_task(self) -> None:
        while True:
            try:
                record = await self.records_queue.get()
                self.execute_async(self.stream_record(record))
                await yield_to_event_loop()
            except asyncio.CancelledError:
                break

    @abstractmethod
    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Handle the incoming record. This method should be implemented by the subclass."""
        raise NotImplementedError(
            "StreamingPostProcessor.stream_record method must be implemented by the subclass."
        )
