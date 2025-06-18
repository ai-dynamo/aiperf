# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import json
import sys
import time

from aiperf.common.comms.client_enums import ClientType, PullClientType, PushClientType
from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType, ServiceState, ServiceType, Topic
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_cleanup, on_configure, on_init, on_start, on_stop
from aiperf.common.messages import (
    CommandMessage,
    CreditDropMessage,
    CreditReturnMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
    Message,
)
from aiperf.common.service.base_component_service import BaseComponentService


@ServiceFactory.register(ServiceType.TIMING_MANAGER)
class TimingManager(BaseComponentService):
    """
    The TimingManager service is responsible to generate the schedule and issuing
    timing credits for requests.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self._credit_lock = asyncio.Lock()
        self._credits_available = 1
        self.logger.debug("Initializing timing manager")
        self._credit_drop_task: asyncio.Task | None = None
        self.dataset_timing_response: DatasetTimingResponse | None = None

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.TIMING_MANAGER

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service."""
        return [
            *(super().required_clients or []),
            PullClientType.CREDIT_RETURN,
            PushClientType.CREDIT_DROP,
        ]

    schedule: list[int] = []

    @on_init
    async def _initialize(self) -> None:
        """Initialize timing manager-specific components."""
        self.logger.debug("Initializing timing manager")
        # TODO: Implement timing manager initialization

    @on_configure
    async def _configure(self, message: Message) -> None:
        """Configure the timing manager."""
        self.logger.debug("Configuring timing manager with message: %s", message)

        if (
            not isinstance(message, CommandMessage)
            or not hasattr(message, "data")
            or not hasattr(message.data, "input_file_path")
        ):
            self.logger.error(
                "Invalid configuration message received. Expected CommandMessage with 'data.input_file_path'."
            )
            return

        try:
            # Initialize the schedule as an empty list
            self.schedule = []

            # Safely get the input_file_path attribute if it exists
            input_file_path = getattr(
                getattr(message, "data", None), "input_file_path", None
            )
            if not isinstance(input_file_path, str):
                self.logger.error(
                    "input_file_path is missing or not a string in configuration message."
                )
                return

            # Read the JSONL file line by line with explicit encoding
            with open(input_file_path, encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    # Skip empty lines
                    if not line:
                        continue

                    # Parse each line as JSON
                    json_obj = json.loads(line)

                    # Extract the timestamp and append to schedule
                    if "timestamp" in json_obj:
                        self.schedule.append(json_obj["timestamp"])

            self.logger.info("Loaded schedule with %d timestamps", len(self.schedule))
            self.logger.debug("Schedule: %s", self.schedule)
        except (OSError, FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(
                "Error loading schedule from %s: %s",
                getattr(message, "input_file_path", None),
                str(e),
            )
        except Exception as e:
            self.logger.error("Unexpected error configuring timing manager: %s", str(e))

    @on_start
    async def _start(self) -> None:
        """Start the timing manager."""
        self.logger.debug("Starting timing manager")

        # Record the start time in nanoseconds
        self._start_time_ns = time.time_ns()
        self.logger.info("Recording start time: %d", self._start_time_ns)

        # Setup credit return handling
        await self.comms.pull(
            topic=Topic.CREDIT_RETURN,
            callback=self._on_credit_return,
        )
        await self.set_state(ServiceState.RUNNING)
        await asyncio.sleep(1.5)

        self.logger.debug("TM: Requesting dataset timing information")
        self.dataset_timing_response = await self.comms.request(
            topic=Topic.DATASET_TIMING,
            message=DatasetTimingRequest(
                service_id=self.service_id,
            ),
        )
        self.logger.debug(
            "TM: Received dataset timing response: %s",
            self.dataset_timing_response,
        )

        # Start the credit dropping task
        self._credit_drop_task = asyncio.create_task(self._issue_credit_drops())

    @on_stop
    async def _stop(self) -> None:
        """Stop the timing manager."""
        self.logger.debug("Stopping timing manager")
        # TODO: Implement timing manager stop
        if self._credit_drop_task and not self._credit_drop_task.done():
            self._credit_drop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._credit_drop_task
            self._credit_drop_task = None

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up timing manager-specific components."""
        self.logger.debug("Cleaning up timing manager")
        # TODO: Implement timing manager cleanup

    async def _issue_credit_drops(self) -> None:
        """Issue credit drops according to the schedule."""
        self.logger.debug(
            "Starting credit drops with schedule of %d timestamps", len(self.schedule)
        )

        if not self.schedule:
            self.logger.warning("No schedule loaded, no credits will be dropped")
            return

        try:
            for timestamp in self.schedule:
                # Check if we should stop
                if self.stop_event.is_set():
                    self.logger.info("Stop event detected, ending credit drops")
                    break

                # Calculate when this credit should be dropped
                target_time_ns = self._start_time_ns + timestamp
                current_time_ns = time.time_ns()

                # Calculate how long to wait in seconds
                wait_time_sec = max(
                    0, (target_time_ns - current_time_ns) / 1_000_000_000
                )

                if wait_time_sec > 0:
                    self.logger.debug(
                        "Waiting %.6f seconds until next credit drop", wait_time_sec
                    )
                    # Wait until it's time to drop the credit
                    await asyncio.sleep(wait_time_sec)

                # Check again if we should stop (after waiting)
                if self.stop_event.is_set():
                    self.logger.info(
                        "Stop event detected after waiting, ending credit drops"
                    )
                    break

                # Acquire lock and drop credit
                async with self._credit_lock:
                    if self._credits_available <= 0:
                        self.logger.warning(
                            "No credits available, skipping scheduled credit drop"
                        )
                        continue

                    self.logger.info(
                        "Dropping credit at timestamp %d ns from start", timestamp
                    )
                    self._credits_available -= 1

                # Send the credit drop message
                await self.comms.push(
                    topic=Topic.CREDIT_DROP,
                    message=CreditDropMessage(
                        service_id=self.service_id,
                        amount=1,
                        credit_drop_ns=time.time_ns(),
                    ),
                )

            self.logger.info("Completed all scheduled credit drops")

        except asyncio.CancelledError:
            self.logger.debug("Credit drop task cancelled")
        except Exception as e:
            self.logger.error("Exception in credit drop scheduler: %s", e)

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message.

        Args:
            message: The credit return message received from the pull request
        """
        self.logger.debug("Processing credit return: %s", message)
        async with self._credit_lock:
            self._credits_available += message.amount


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    sys.exit(main())
