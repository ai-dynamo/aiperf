# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict
from collections.abc import Coroutine
from typing import Any

from aiperf.common.enums import Topic
from aiperf.common.messages import (
    CreditDropMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
)

# from aiperf.services.timing_manager import CreditDropInfo
from aiperf.services.timing_manager.credit_issuing_strategy import CreditIssuingStrategy


class FixedScheduleStrategy(CreditIssuingStrategy):
    """
    Class for fixed schedule credit issuing strategy.
    """

    def __init__(self, config, credit_drop_function):
        super().__init__(config, credit_drop_function)

        self._schedule: list[tuple[int, str]] = []

    async def initialize(self) -> None:
        await self.comms.register(
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

            tasks: set[Coroutine[Any, Any, None]] = set()

            for _, conversation_id in timestamp_groups[unique_timestamp]:
                # credit_drop_info = CreditDropInfo()
                # credit_drop_info.conversation_id = conversation_id
                # credit_drop_info.credit_drop_ns = time.time_ns()

                task = asyncio.create_task(
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
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        self.logger.info("Completed all scheduled credit drops")
