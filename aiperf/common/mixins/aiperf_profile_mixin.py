# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging

from aiperf.common.hooks import AIPerfHook, supports_hooks
from aiperf.common.messages import Message
from aiperf.common.mixins.hooks_mixin import HooksMixin


@supports_hooks(
    AIPerfHook.ON_PROFILE_CONFIGURE,
    AIPerfHook.ON_PROFILE_START,
    AIPerfHook.ON_PROFILE_STOP,
)
class AIPerfProfileMixin(HooksMixin):
    """Mixin to add profile-related hook support to a class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__class__.__name__)
        self.profile_started_event: asyncio.Event = asyncio.Event()
        self.profile_stopped_event: asyncio.Event = asyncio.Event()
        self.request_profile_stop_event: asyncio.Event = asyncio.Event()
        self.profile_configured_event: asyncio.Event = asyncio.Event()

    async def configure_profile(self, message: Message):
        """Configure the profile."""
        await self.run_hooks(AIPerfHook.ON_PROFILE_CONFIGURE, message)
        self.profile_configured_event.set()

    async def run_profile(self):
        """Run the profile."""
        # Run all the start hooks and set the start_event
        await self.run_hooks_async(AIPerfHook.ON_PROFILE_START)
        self.profile_started_event.set()

        while not self.request_profile_stop_event.is_set():
            try:
                # Wait forever until the stop_requested event is set
                await self.request_profile_stop_event.wait()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(
                    "Unhandled exception in while profile is running: %s", e
                )
                continue

        try:
            # Run all the stop hooks
            await self.run_hooks_async(AIPerfHook.ON_PROFILE_STOP)
        except Exception as e:
            self.logger.exception(
                "Unhandled exception in while profile is running: %s", e
            )

    async def stop_profile(self):
        """Request the profile to stop."""
        self.request_profile_stop_event.set()

    async def wait_for_profile_configured(self):
        """Wait for the profile to be configured."""
        await self.profile_configured_event.wait()

    async def wait_for_profile_started(self):
        """Wait for the profile to start."""
        await self.profile_started_event.wait()

    async def wait_for_profile_stopped(self):
        """Wait for the profile to stop."""
        await self.profile_stopped_event.wait()
