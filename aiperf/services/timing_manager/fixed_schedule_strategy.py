# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.messages import CreditDropMessage, DatasetTimingResponse
from aiperf.services.timing_manager.credit_issuing_strategy import CreditIssuingStrategy


class FixedScheduleStrategy(CreditIssuingStrategy):
    """
    Class for fixed schedule credit issuing strategy.
    """

    def __init__(self, config, credit_drop_function):
        super().__init__(config, credit_drop_function)

        self._schedule: list[tuple[int, str]] = []

    async def initialize(self) -> None:
        pass

    async def _get_dataset_timing(self, message: DatasetTimingResponse) -> None:
        self._schedule = message.timing_data

    async def start(self) -> None:
        if not self._schedule:
            raise InvalidStateError("No schedule loaded, no credits will be dropped")

        start_time_ns = time.time_ns()

        timestamp_groups = defaultdict(list)

        for timestamp, conversation_id in self._schedule:
            timestamp_groups[timestamp].append((timestamp, conversation_id))

        schedule_unique_sorted = sorted(timestamp_groups.keys())

        for unique_timestamp in schedule_unique_sorted:
            wait_duration_ns = max(0, start_time_ns + unique_timestamp - time.time_ns())
            wait_duration_sec = wait_duration_ns / NANOS_PER_SECOND

            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            for _, conversation_id in timestamp_groups[unique_timestamp]:
                self.execute_async(
                    self.credit_drop_client.push(
                        CreditDropMessage(
                            service_id=self.service_id,
                            amount=1,
                            conversation_id=conversation_id,
                            credit_drop_ns=time.time_ns(),
                        ),
                    )
                )

        self.logger.info("Completed all scheduled credit drops")
