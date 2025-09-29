# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections import defaultdict

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.factories import CreditIssuingStrategyFactory
from aiperf.common.models import CreditPhaseConfig, CreditPhaseStats
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
)
from aiperf.timing.credit_manager import CreditManagerProtocol


@CreditIssuingStrategyFactory.register(TimingMode.FIXED_SCHEDULE)
class FixedScheduleStrategy(CreditIssuingStrategy):
    """
    Class for fixed schedule credit issuing strategy.
    """

    def __init__(
        self,
        config: TimingManagerConfig,
        credit_manager: CreditManagerProtocol,
        schedule: list[tuple[int, str]],
    ):
        # NOTE: This all needs to be set before the super call, because the base class will call
        # _setup_profiling_phase_config() which uses it to set the total expected requests.
        self._schedule: list[tuple[int, str]] = schedule
        self._num_requests = len(self._schedule)
        self._auto_offset_timestamps = config.auto_offset_timestamps
        self._start_offset = config.fixed_schedule_start_offset
        self._end_offset = config.fixed_schedule_end_offset
        super().__init__(config=config, credit_manager=credit_manager)

    def _create_timestamp_groups(self) -> None:
        """
        Create a dictionary of timestamp groups, and filter the schedule to only include the requested subset.
        """
        if not self._schedule or self._num_requests == 0:
            raise ValueError(
                "No schedule loaded, unable to setup fixed schedule strategy"
            )
        # Group the schedule by timestamp
        self._timestamp_groups = defaultdict(list)
        for timestamp, conversation_id in self._schedule:
            self._timestamp_groups[timestamp].append(conversation_id)

        # Sort the timestamps, so we can drop credits in order
        self._sorted_timestamp_keys = sorted(self._timestamp_groups.keys())

        # Define the zero reference point for the schedule
        if self._auto_offset_timestamps:
            self._schedule_zero_ms = self._sorted_timestamp_keys[0]
        elif self._start_offset is not None:
            self._schedule_zero_ms = self._start_offset
        else:
            self._schedule_zero_ms = 0

    def _setup_profiling_phase_config(self) -> None:
        """
        Setup the profiling phase.

        Overrides the base implementation to set the total expected requests based on the number of requests in the schedule.
        """
        self._create_timestamp_groups()

        self.ordered_phase_configs.append(
            CreditPhaseConfig(
                type=CreditPhase.PROFILING,
                total_expected_requests=self._num_requests,
            )
        )

    def _perf_counter_ms(self) -> float:
        return time.perf_counter() * MILLIS_PER_SECOND

    async def _execute_single_phase(self, phase_stats: CreditPhaseStats) -> None:
        # This is used as a reference point for the wait duration calculation
        start_time_ms = self._perf_counter_ms()

        # Drop credits in order of the schedule
        for timestamp in self._sorted_timestamp_keys:
            # Lookup/cache the conversation IDs now, so that way they are ready to go
            conversation_ids = self._timestamp_groups[timestamp]

            # Calculate the wait duration for this timestamp
            # (timestamp - schedule_zero_ms) is the offset of the conversation(s) from the start of the schedule
            # (self._perf_counter_ms() - start_time_ms) is how much time has passed since we started dropping credits
            wait_duration_ms = (timestamp - self._schedule_zero_ms) - (
                self._perf_counter_ms() - start_time_ms
            )
            wait_duration_sec = wait_duration_ms / MILLIS_PER_SECOND

            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            # Drop credits asynchronously for all conversations at this timestamp
            for conversation_id in conversation_ids:
                should_cancel = self.cancellation_strategy.should_cancel_request()
                cancel_after_ns = self.cancellation_strategy.get_cancellation_delay_ns()

                await self.credit_manager.drop_credit(
                    credit_phase=CreditPhase.PROFILING,
                    conversation_id=conversation_id,
                    # We already waited, so it can be sent ASAP
                    credit_drop_ns=None,
                    should_cancel=should_cancel,
                    cancel_after_ns=cancel_after_ns,
                )
                phase_stats.sent += 1

        duration_sec = (self._perf_counter_ms() - start_time_ms) / MILLIS_PER_SECOND
        self.info(
            f"Sent all {self._num_requests:,} fixed schedule requests in {duration_sec:,.2f}s. Waiting for responses..."
        )
