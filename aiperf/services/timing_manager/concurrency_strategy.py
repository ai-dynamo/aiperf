# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditManagerProtocol,
)


class ConcurrencyStrategy(CreditIssuingStrategy, AsyncTaskManagerMixin):
    """
    Class for concurrency credit issuing strategy.
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__(config=config, credit_manager=credit_manager)

        self._total_credits = int(os.getenv("AIPERF_TOTAL_REQUESTS", 1000))
        self._concurrency = min(
            self._total_credits, int(os.getenv("AIPERF_CONCURRENCY", 10))
        )

        self._sent_credits = 0
        self._completed_credits = 0
        self.start_time_ns = 0
        self._semaphore = asyncio.Semaphore(value=self._concurrency)

        self.logger.info(
            "TM: Concurrency Strategy initialized with total_credits=%s, concurrency=%s",
            self._total_credits,
            self._concurrency,
        )

    async def start(self) -> None:
        self.start_time_ns = time.time_ns()
        self.execute_async(self._progress_report_loop())
        self.execute_async(self._credit_drop_loop())

    async def _credit_drop_loop(self) -> None:
        """Issue credit drops to workers."""

        await asyncio.sleep(1.5)  # TODO: HACK: Wait for the system to be ready

        self.logger.info(
            "TM: Issuing credit drops %s total credits, %s concurrency",
            self._total_credits,
            self._concurrency,
        )

        while self._sent_credits < self._total_credits:
            await self._semaphore.acquire()
            self.execute_async(
                self.credit_manager.drop_credit(
                    # TODO: Allow concurrency to be used with trace datasets
                    conversation_id=None,
                    credit_drop_ns=None,
                )
            )
            self._sent_credits += 1

        self.logger.info("TM: All credits sent, stopping credit drop loop")

    async def on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message."""

        self._semaphore.release()
        self._completed_credits += 1

        if self.logger.isEnabledFor(logging.DEBUG):
            per_sec = (
                self._completed_credits
                / ((time.time_ns() - self.start_time_ns) / NANOS_PER_SECOND)
                if (time.time_ns() - self.start_time_ns) > 0
                else 0
            )
            self.logger.debug(
                "Processing credit return: (completed credits: %s of %s) (%.2f requests/s)",
                self._completed_credits,
                self._total_credits,
                per_sec,
            )

        if self._completed_credits >= self._total_credits:
            self.execute_async(self.credit_manager.publish_credits_complete(False))

            if self.logger.isEnabledFor(logging.DEBUG):
                per_sec = (
                    self._completed_credits
                    / ((time.time_ns() - self.start_time_ns) / NANOS_PER_SECOND)
                    if (time.time_ns() - self.start_time_ns) > 0
                    else 0
                )
                self.logger.debug(
                    "All credits completed, stopping credit drop task after %.2f seconds (%.2f requests/s)",
                    (time.time_ns() - self.start_time_ns) / NANOS_PER_SECOND,
                    per_sec,
                )

    async def _progress_report_loop(self) -> None:
        """Report the progress at a fixed interval."""
        while True:
            try:
                await self.credit_manager.publish_progress(
                    self.start_time_ns, self._total_credits, self._completed_credits
                )
            except asyncio.CancelledError:
                self.logger.debug("TM: Progress reporting loop cancelled")
                break
            except Exception as e:
                self.logger.error("TM: Error publishing progress: %s", e)

            if self._completed_credits >= self._total_credits:
                self.logger.debug(
                    "TM: All credits completed, stopping progress reporting loop"
                )
                break

            await asyncio.sleep(1)  # TODO: Make this configurable
