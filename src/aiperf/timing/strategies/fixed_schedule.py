# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed schedule timing strategy for trace replay.

Replays conversation traces at precise timestamps from dataset metadata.
First turns sent by main loop at absolute timestamps, subsequent turns
dispatched using delay_ms or timestamp_ms from metadata.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, NamedTuple

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker


class ScheduleEntry(NamedTuple):
    """A single entry in the fixed schedule."""

    timestamp_ms: int | float
    turn: TurnToSend


class FixedScheduleStrategy(AIPerfLoggerMixin):
    """Timing strategy for replaying conversation traces with absolute timestamps.

    Sends first turns at precise timestamps from conversation metadata.
    Subsequent turns dispatched immediately or after calculated delay.

    This is a pure timing strategy - no lifecycle or orchestration concerns.
    The PhaseRunner handles all orchestration.
    """

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        conversation_source: ConversationSource,
        scheduler: LoopScheduler,
        credit_issuer: CreditIssuer,
        lifecycle: PhaseLifecycle,
        stop_checker: StopConditionChecker,
        **kwargs,
    ):
        """Initialize fixed schedule timing strategy with all dependencies."""
        super().__init__(logger_name="FixedScheduleTiming")
        self._config = config
        self._conversation_source = conversation_source
        self._scheduler = scheduler
        self._credit_issuer = credit_issuer
        self._lifecycle = lifecycle

        # Computed in setup_phase
        self._absolute_schedule: list[ScheduleEntry] = []
        self._schedule_zero_ms: float = 0.0

    def _timestamp_to_perf_sec(self, timestamp_ms: int | float) -> float:
        """Convert trace timestamp in milliseconds to perf counter seconds.

        Uses the offset from the schedule zero to calculate the target performance seconds.
        """
        target_offset_sec = (timestamp_ms - self._schedule_zero_ms) / MILLIS_PER_SECOND
        return self._lifecycle.started_at_perf_sec + target_offset_sec

    async def setup_phase(self) -> None:
        """Build absolute schedule from dataset metadata.

        Dataset is already filtered by loader (e.g., mooncake_trace._timestamp_within_offsets),
        so we just validate and build the schedule.
        """
        self._absolute_schedule = []  # Fresh schedule for each phase

        # Validate and build schedule
        for conv in self._conversation_source.dataset_metadata.conversations:
            if not conv.turns:
                continue

            # Validate first turn has timestamp (required for fixed schedule mode)
            if conv.turns[0].timestamp_ms is None:
                raise ValueError(
                    f"First turn of {conv.conversation_id} missing timestamp_ms"
                )

            self._absolute_schedule.append(
                ScheduleEntry(
                    timestamp_ms=conv.turns[0].timestamp_ms,
                    turn=TurnToSend(
                        conversation_id=conv.conversation_id,
                        x_correlation_id=str(uuid.uuid4()),
                        turn_index=0,
                        num_turns=len(conv.turns),
                    ),
                )
            )

        if not self._absolute_schedule:
            raise ValueError("No conversations with valid first-turn timestamps found")

        self._absolute_schedule.sort(key=lambda x: x.timestamp_ms)
        # Calculate schedule zero (dataset already filtered by loader)
        if self._config.auto_offset_timestamps:
            self._schedule_zero_ms = self._absolute_schedule[0].timestamp_ms
        elif self._config.fixed_schedule_start_offset is not None:
            self._schedule_zero_ms = float(self._config.fixed_schedule_start_offset)
        else:
            self._schedule_zero_ms = 0.0

        self.info(
            f"Built schedule with {len(self._absolute_schedule)} timestamps, "
            f"zero_ms={self._schedule_zero_ms:.0f}, "
            f"auto_offset={self._config.auto_offset_timestamps}"
        )

    async def execute_phase(self) -> None:
        """Execute absolute schedule: send first turns at precise timestamps.

        Note: Subsequent turns are handled by handle_credit_return.

        Raises:
            RuntimeError: If started_at_perf_ns is not set in the lifecycle
        """
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        for entry in self._absolute_schedule:
            self._scheduler.schedule_at_perf_sec(
                self._timestamp_to_perf_sec(entry.timestamp_ms),
                self._credit_issuer.issue_credit(entry.turn),
            )

    async def handle_credit_return(
        self,
        credit: Credit,
    ) -> None:
        """Handle credit return: dispatch next turn based on trace timing.

        Calculates delay from timestamp_ms or delay_ms metadata, then issues
        credit immediately (delay=0) or schedules for later (delay>0).
        """
        if credit.is_final_turn:
            return

        # This contains the delay_ms or timestamp_ms for the next turn
        next_meta = self._conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit)

        if next_meta.timestamp_ms is not None:
            self._scheduler.schedule_at_perf_sec(
                self._timestamp_to_perf_sec(next_meta.timestamp_ms),
                self._credit_issuer.issue_credit(turn),
            )
        elif next_meta.delay_ms is not None:
            self._scheduler.schedule_later(
                next_meta.delay_ms / MILLIS_PER_SECOND,
                self._credit_issuer.issue_credit(turn),
            )
        else:
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(turn),
            )
