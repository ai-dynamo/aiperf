# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict

from aiperf.common.enums import TimingMode
from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.common.models.credit_models import CreditPhaseStats
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
)
from aiperf.services.timing_manager.credit_manager import CreditManagerProtocol


@CreditIssuingStrategyFactory.register(TimingMode.FIXED_SCHEDULE)
class FixedScheduleStrategy(CreditIssuingStrategy, AsyncTaskManagerMixin):
    """
    Class for fixed schedule credit issuing strategy.
    """

    def __init__(
        self,
        config: TimingManagerConfig,
        credit_manager: CreditManagerProtocol,
        schedule: list[tuple[int, str]],
    ):
        super().__init__(config=config, credit_manager=credit_manager)

        self._schedule: list[tuple[int, str]] = schedule

    async def _execute_single_phase(self, phase_stats: CreditPhaseStats) -> None:
        # TODO: Convert this code to work with the new CreditPhase logic and base classes

        if not self._schedule:
            self.warning("No schedule loaded, no credits will be dropped")
            return

        start_time_ns = time.time_ns()

        timestamp_groups = defaultdict(list)

        for timestamp, conversation_id in self._schedule:
            timestamp_groups[timestamp].append((timestamp, conversation_id))

        schedule_unique_sorted = sorted(timestamp_groups.keys())

        for unique_timestamp in schedule_unique_sorted:
            wait_duration_ns = max(0, start_time_ns + unique_timestamp - time.time_ns())
            wait_duration_sec = wait_duration_ns / 1_000_000_000

            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            for _, conversation_id in timestamp_groups[unique_timestamp]:
                self.execute_async(
                    self.credit_manager.drop_credit(
                        credit_phase=CreditPhase.PROFILING,
                        conversation_id=conversation_id,
                        # We already waited, so it can be sent ASAP
                        credit_drop_ns=None,
                    )
                )

        self.info("Completed all scheduled credit drops")
