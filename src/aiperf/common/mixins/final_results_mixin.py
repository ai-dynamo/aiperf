# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models.record_models import ProcessRecordsResult


class FinalResultsMixin(MessageBusClientMixin):
    """A mixin that stores final benchmark results when received.

    This mixin subscribes to ProcessRecordsResultMessage and stores the results
    for later retrieval via API endpoints. It also tracks whether the benchmark
    has completed.
    """

    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        super().__init__(service_config=service_config, **kwargs)
        self._final_results: ProcessRecordsResult | None = None
        self._benchmark_complete: bool = False

    @on_message(MessageType.PROCESS_RECORDS_RESULT)
    async def _on_process_records_result(
        self, message: ProcessRecordsResultMessage
    ) -> None:
        """Store final benchmark results when received.

        Lock-free because self._final_results is atomically replaced.
        Operations are atomic only when used in a single thread asyncio context.
        """
        self._final_results = message.results
        self._benchmark_complete = True
