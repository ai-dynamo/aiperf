# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import time

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.hooks import AIPerfLifecycleMixin, aiperf_auto_task
from aiperf.common.messages import CreditReturnMessage
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditManagerProtocol,
)


class ConcurrencyStrategy(CreditIssuingStrategy, AIPerfLifecycleMixin):
    """
    Class for concurrency credit issuing strategy.
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__(config=config, credit_manager=credit_manager)
        self._credit_lock = asyncio.Lock()

        self._total_credits = int(os.getenv("AIPERF_TOTAL_REQUESTS", 1000))
        self._credits_available = min(
            self._total_credits, int(os.getenv("AIPERF_CONCURRENCY", 10))
        )

        self._sent_credits = 0
        self._completed_credits = 0
        self._credit_event = asyncio.Event()
        self.start_time_ns = time.time_ns()

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        await self.run_and_wait_for_start()
        self.execute_async(self._issue_credit_drops())

    async def stop(self) -> None:
        await super().stop()
        await self.shutdown()

    async def _issue_credit_drops(self) -> None:
        """Issue credit drops to workers."""
        self.logger.debug("Issuing credit drops to workers")

        await asyncio.sleep(3)

        self.start_time_ns = time.time_ns()

        self.execute_async(
            self.credit_manager.publish_progress(
                self.start_time_ns, self._total_credits, self._completed_credits
            )
        )

        self.logger.info("TM: Issuing credit drops")
        while True:
            try:
                if not self._credits_available:
                    self.logger.debug("Waiting for credit event")
                    self._credit_event.clear()
                    await self._credit_event.wait()
                    self.logger.debug("Credit event received")
                    continue

                self.logger.debug(
                    "Issuing credit drop %s of %s",
                    self._sent_credits + 1,
                    self._total_credits,
                )

                credits_left = self._total_credits - self._sent_credits
                for _ in range(min(self._credits_available, credits_left)):
                    self.execute_async(
                        self.credit_manager.drop_credit(
                            # TODO: Do we need to pass conversation_id?
                            amount=1,
                            conversation_id=None,
                            credit_drop_ns=None,
                        )
                    )
                    self._sent_credits += 1
                    self._credits_available -= 1

                if self._sent_credits >= self._total_credits:
                    self.logger.debug("All credits sent, stopping credit drop task")
                    break

            except asyncio.CancelledError:
                self.logger.debug("Credit drop task cancelled")
                break
            except Exception as e:
                self.logger.error("Exception issuing credit drop: %s", e)
                await asyncio.sleep(0.1)

    async def on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message."""

        self.logger.debug("Processing credit return: %s", message)
        async with self._credit_lock:
            self._credits_available += message.amount
            self._completed_credits += message.amount

        self.logger.debug(
            "Processing credit return: %s (completed credits: %s of %s) (%.2f requests/s)",
            message.amount,
            self._completed_credits,
            self._total_credits,
            self._completed_credits
            / (time.time_ns() - self.start_time_ns)
            * NANOS_PER_SECOND,
        )

        if self._completed_credits >= self._total_credits:
            self.logger.debug(
                "All credits completed, stopping credit drop task after %.2f seconds (%.2f requests/s)",
                (time.time_ns() - self.start_time_ns) / NANOS_PER_SECOND,
                self._total_credits
                / ((time.time_ns() - self.start_time_ns) / NANOS_PER_SECOND),
            )

            self.execute_async(self.credit_manager.publish_credits_complete(False))

        self._credit_event.set()

    @aiperf_auto_task(interval=1)
    async def _report_progress_task(self) -> None:
        """Report the progress."""
        self.execute_async(
            self.credit_manager.publish_progress(
                self.start_time_ns, self._total_credits, self._completed_credits
            )
        )
