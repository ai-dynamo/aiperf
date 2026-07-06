# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for fail-fast propagation when DatasetManager._profile_configure_command raises.

A bug in dataset configuration (e.g., AttributeError on a prompt generator)
must NOT translate into a 300s hang. Two pieces have to cooperate:

1. DatasetManager publishes DatasetConfigurationFailedNotification before
   re-raising, so the fan-out broadcast reaches peer services that block on
   DATASET_CONFIGURED_NOTIFICATION.

2. TimingManager._profile_configure_command waits on EITHER the success or
   failure event and raises immediately on failure, instead of blocking the
   full DATASET.CONFIGURATION_TIMEOUT.

Both directions are exercised here.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.environment import Environment
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.messages import (
    DatasetConfigurationFailedNotification,
    ProfileConfigureCommand,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.timing.manager import TimingManager
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def base_run() -> BenchmarkRun:
    return make_run_from_cli(CLIConfig(model_names=["test-model"]))


@pytest.fixture
def timing_run() -> BenchmarkRun:
    return make_run_from_cli(CLIConfig(model_names=["test-model"]))


class TestDatasetManagerPublishesFailureNotification:
    """DatasetManager must publish DatasetConfigurationFailedNotification when
    its PROFILE_CONFIGURE handler raises, so peers can break their waits."""

    @pytest.mark.asyncio
    async def test_failure_in_configure_publishes_notification_and_reraises(
        self, base_run
    ) -> None:
        dataset_manager = DatasetManager(run=base_run, service_id="test_service")
        await dataset_manager.initialize()

        published: list = []

        async def capture_publish(msg):
            published.append(msg)

        dataset_manager.publish = AsyncMock(side_effect=capture_publish)

        sentinel = RuntimeError("synthetic prompt generator exploded")

        async def raise_sentinel(*args, **kwargs):
            raise sentinel

        # Force the inner configure path to fail; the outer wrapper must still
        # publish the failure notification before re-raising.
        with (
            patch.object(
                dataset_manager, "_do_profile_configure", side_effect=raise_sentinel
            ),
            pytest.raises(RuntimeError, match="synthetic prompt generator exploded"),
        ):
            await asyncio.wait_for(
                dataset_manager._profile_configure_command(
                    ProfileConfigureCommand(service_id="test_service")
                ),
                timeout=5.0,
            )

        failure_notes = [
            m
            for m in published
            if isinstance(m, DatasetConfigurationFailedNotification)
        ]
        assert len(failure_notes) == 1, (
            f"expected exactly one failure notification, got {published}"
        )
        assert "synthetic prompt generator exploded" in failure_notes[0].error
        assert failure_notes[0].service_id == dataset_manager.service_id

    @pytest.mark.asyncio
    async def test_publish_failure_still_reraises_original_error(
        self, base_run
    ) -> None:
        """If the failure-notification publish itself crashes (e.g. the bus is
        already torn down), the ORIGINAL configure error must still propagate."""
        dataset_manager = DatasetManager(run=base_run, service_id="test_service")
        await dataset_manager.initialize()

        dataset_manager.publish = AsyncMock(side_effect=ConnectionError("bus is gone"))

        async def raise_sentinel(*args, **kwargs):
            raise RuntimeError("composer exploded")

        with (
            patch.object(
                dataset_manager, "_do_profile_configure", side_effect=raise_sentinel
            ),
            pytest.raises(RuntimeError, match="composer exploded"),
        ):
            await asyncio.wait_for(
                dataset_manager._profile_configure_command(
                    ProfileConfigureCommand(service_id="test_service")
                ),
                timeout=5.0,
            )

        dataset_manager.publish.assert_awaited_once()


class TestTimingManagerAbortsOnDatasetFailure:
    """TimingManager._profile_configure_command must abort within milliseconds
    of receiving DatasetConfigurationFailedNotification, instead of blocking
    on the 300s DATASET.CONFIGURATION_TIMEOUT envelope."""

    @pytest.fixture
    def timing_manager(self, timing_run) -> TimingManager:
        return TimingManager(
            run=timing_run,
            service_id="test-timing-manager",
        )

    @pytest.mark.asyncio
    async def test_failure_notification_aborts_configure_wait(
        self, timing_manager
    ) -> None:
        configure_task = asyncio.create_task(
            timing_manager._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-system-controller", config={}
                )
            )
        )

        # Ensure the configure task has entered the wait state before we
        # publish the failure notification.
        await asyncio.sleep(0.05)
        assert not configure_task.done()

        await timing_manager._on_dataset_configuration_failed(
            DatasetConfigurationFailedNotification(
                service_id="test-dataset-manager",
                error="RuntimeError: synthetic prompt generator exploded",
            )
        )

        with pytest.raises(InvalidStateError, match="Dataset configuration failed"):
            await asyncio.wait_for(configure_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_failure_notification_received_before_configure_aborts_immediately(
        self, timing_manager
    ) -> None:
        # If the failure notification arrives BEFORE the configure command
        # (e.g., because DatasetManager errored before the controller
        # broadcast PROFILE_CONFIGURE was processed by the timing manager),
        # the configure call should still raise immediately.
        await timing_manager._on_dataset_configuration_failed(
            DatasetConfigurationFailedNotification(
                service_id="test-dataset-manager",
                error="RuntimeError: pre-broadcast failure",
            )
        )

        with pytest.raises(InvalidStateError, match="pre-broadcast failure"):
            await asyncio.wait_for(
                timing_manager._profile_configure_command(
                    ProfileConfigureCommand.model_construct(
                        service_id="test-system-controller", config={}
                    )
                ),
                timeout=2.0,
            )

    @pytest.mark.asyncio
    async def test_dataset_configuration_timeout_still_enforced(
        self, timing_manager
    ) -> None:
        # When NEITHER event fires, the existing 300s envelope still applies.
        # Use a reduced timeout to keep this test fast.
        with (
            patch.object(Environment.DATASET, "CONFIGURATION_TIMEOUT", 0.1),
            pytest.raises(asyncio.TimeoutError),
        ):
            await timing_manager._profile_configure_command(
                ProfileConfigureCommand.model_construct(
                    service_id="test-system-controller", config={}
                )
            )
