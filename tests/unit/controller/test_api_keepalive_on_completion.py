# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The API must survive the controller long enough to serve final results.

Two defects compounded here. BenchmarkCompleteMessage had no publisher, so
/api/results never left "running" and POST /api/shutdown answered 409 forever
-- and that endpoint is the operator's graceful-exit handshake, so every
completion fell through to a hard pod delete. Separately, the grace window that
keeps the API listener alive was gated on _is_api_service_alive(), which is
unconditionally False under Kubernetes because KubernetesServiceManager
inherits an always-empty multi_process_info from MultiProcessServiceManager.
"""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.messages import BenchmarkCompleteMessage
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


def _controller(*, mp_info, api_registered=True):
    ctrl = SystemController.__new__(SystemController)
    ctrl.service_manager = MagicMock()
    api = MagicMock()
    api.registration_status = (
        ServiceRegistrationStatus.REGISTERED
        if api_registered
        else ServiceRegistrationStatus.UNREGISTERED
    )
    api.state = LifecycleState.RUNNING
    ctrl.service_manager.service_map = {ServiceType.API: [api]}
    ctrl.service_manager.multi_process_info = mp_info
    return ctrl


class TestApiLivenessUnderKubernetes:
    def test_registered_api_counts_as_alive_with_no_local_processes(self):
        """K8s runs the API as its own pod; there is no local process to poll."""
        ctrl = _controller(mp_info=[])
        assert ctrl._is_api_service_alive() is True

    def test_unregistered_api_is_not_alive(self):
        ctrl = _controller(mp_info=[], api_registered=False)
        assert ctrl._is_api_service_alive() is False

    def test_local_process_check_still_applies_when_present(self):
        """Multiprocess mode keeps the authoritative is_alive() cross-check."""
        rec = MagicMock()
        rec.service_type = ServiceType.API
        rec.process = MagicMock(is_alive=lambda: False)
        ctrl = _controller(mp_info=[rec])
        assert ctrl._is_api_service_alive() is False

    def test_live_local_process_is_alive(self):
        rec = MagicMock()
        rec.service_type = ServiceType.API
        rec.process = MagicMock(is_alive=lambda: True)
        ctrl = _controller(mp_info=[rec])
        assert ctrl._is_api_service_alive() is True


class TestBenchmarkCompleteIsPublished:
    @pytest.mark.asyncio
    async def test_completion_is_announced_when_the_api_is_enabled(self):
        """Without this the results endpoint never completes and shutdown 409s."""
        ctrl = SystemController.__new__(SystemController)
        ctrl._api_enabled = True
        ctrl._was_cancelled = False
        ctrl.service_id = "system_controller"
        published: list = []

        async def _publish(msg):
            published.append(msg)

        ctrl.publish = _publish

        await ctrl._announce_benchmark_complete()

        assert len(published) == 1
        msg = published[0]
        assert isinstance(msg, BenchmarkCompleteMessage)
        assert msg.was_cancelled is False

    @pytest.mark.asyncio
    async def test_cancellation_is_carried_on_the_message(self):
        ctrl = SystemController.__new__(SystemController)
        ctrl._api_enabled = True
        ctrl._was_cancelled = True
        ctrl.service_id = "system_controller"
        published: list = []

        async def _publish(msg):
            published.append(msg)

        ctrl.publish = _publish

        await ctrl._announce_benchmark_complete()
        assert published[0].was_cancelled is True

    @pytest.mark.asyncio
    async def test_nothing_published_without_an_api(self):
        ctrl = SystemController.__new__(SystemController)
        ctrl._api_enabled = False
        ctrl._was_cancelled = False
        ctrl.service_id = "system_controller"
        published: list = []

        async def _publish(msg):
            published.append(msg)

        ctrl.publish = _publish

        await ctrl._announce_benchmark_complete()
        assert published == []
