# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TimingManager aborts cleanly on DatasetConfigurationFailedNotification.

Without these handlers, a malformed dataset hangs the run for the
full DATASET.CONFIGURATION_TIMEOUT (300s) before raising; with them,
the run aborts immediately when the DatasetManager publishes the
failure notification.
"""

import asyncio

import pytest

from aiperf.common.messages import DatasetConfigurationFailedNotification
from aiperf.timing.manager import TimingManager


class TestDatasetFailureHandling:
    @pytest.mark.asyncio
    async def test_wait_for_dataset_or_failure_returns_on_failure_event(self) -> None:
        """The wait coroutine returns as soon as the failure event fires."""
        mgr = TimingManager.__new__(TimingManager)
        mgr._dataset_configured_event = asyncio.Event()
        mgr._dataset_failed_event = asyncio.Event()
        mgr._dataset_failure_reason = None

        async def fail_soon() -> None:
            await asyncio.sleep(0)
            mgr._dataset_failed_event.set()

        asyncio.create_task(fail_soon())

        await asyncio.wait_for(mgr._wait_for_dataset_or_failure(), timeout=2.0)
        assert mgr._dataset_failed_event.is_set()
        assert not mgr._dataset_configured_event.is_set()

    @pytest.mark.asyncio
    async def test_wait_for_dataset_or_failure_returns_on_configured_event(
        self,
    ) -> None:
        """And it also returns on the success path."""
        mgr = TimingManager.__new__(TimingManager)
        mgr._dataset_configured_event = asyncio.Event()
        mgr._dataset_failed_event = asyncio.Event()
        mgr._dataset_failure_reason = None

        async def configure_soon() -> None:
            await asyncio.sleep(0)
            mgr._dataset_configured_event.set()

        asyncio.create_task(configure_soon())

        await asyncio.wait_for(mgr._wait_for_dataset_or_failure(), timeout=2.0)
        assert mgr._dataset_configured_event.is_set()
        assert not mgr._dataset_failed_event.is_set()

    @pytest.mark.asyncio
    async def test_on_dataset_configuration_failed_sets_failure_event(self) -> None:
        """The notification handler sets the failure event and stores the reason."""
        mgr = TimingManager.__new__(TimingManager)
        mgr._dataset_configured_event = asyncio.Event()
        mgr._dataset_failed_event = asyncio.Event()
        mgr._dataset_failure_reason = None
        # Patch out logger access — we instantiated via __new__
        mgr.error = lambda *a, **k: None  # type: ignore[method-assign]

        msg = DatasetConfigurationFailedNotification(
            service_id="ds-mgr-1", error="bad config"
        )
        await mgr._on_dataset_configuration_failed(msg)

        assert mgr._dataset_failed_event.is_set()
        assert mgr._dataset_failure_reason == "bad config"
