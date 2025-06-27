# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict

from aiperf.common.enums import Topic
from aiperf.common.messages import (
    CreditDropMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
)
from aiperf.services.timing_manager.credit_issuing_strategy import CreditIssuingStrategy


class FixedScheduleStrategy(CreditIssuingStrategy):
    """
    Class for fixed schedule credit issuing strategy.
    """

    def __init__(self, config, stop_event, comms, service_id):
        super().__init__(config, stop_event, comms, service_id)

        self._schedule: list[tuple[int, str]] = []

    async def initialize(self) -> None:
        await self.comms.register_pull_callback(
            message_type=DatasetTimingRequest, callback=self._get_dataset_timing
        )

    async def _get_dataset_timing(self, message: DatasetTimingResponse) -> None:
        self._schedule = message.timing_data

    async def start(self) -> None:
        if not self._schedule:
            self.logger.warning("No schedule loaded, no credits will be dropped")
            return
        if self.stop_event.is_set():
            self.logger.info("Stop event already set, not starting")
            return

        start_time_ns = time.time_ns()

        timestamp_groups = defaultdict(list)

        for timestamp, conversation_id in self._schedule:
            timestamp_groups[timestamp].append((timestamp, conversation_id))

        schedule_unique_sorted = sorted(timestamp_groups.keys())

        for unique_timestamp in schedule_unique_sorted:
            if self.stop_event.is_set():
                self.logger.info("Stop event detected, ending credit drops")
                break

            wait_duration_ns = max(0, start_time_ns + unique_timestamp - time.time_ns())
            wait_duration_sec = wait_duration_ns / 1_000_000_000

            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            if self.stop_event.is_set():
                self.logger.info("Stop event detected, ending credit drops")
                break

            for _, conversation_id in timestamp_groups[unique_timestamp]:
                asyncio.create_task(
                    self.comms.push(
                        topic=Topic.CREDIT_DROP,
                        message=CreditDropMessage(
                            service_id=self.service_id,
                            amount=1,
                            conversation_id=conversation_id,
                            credit_drop_ns=time.time_ns(),
                        ),
                    )
                )

        self.logger.info("Completed all scheduled credit drops")
