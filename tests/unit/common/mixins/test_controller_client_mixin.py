# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DEALER/ROUTER registration probe in BaseComponentService."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.environment import Environment

SERVICE_ID = "test-service-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bind_methods(mock: MagicMock) -> None:
    """Bind the real methods from BaseComponentService onto a mock."""
    for name in (
        "_register_until_ack",
        "_run_connection_probes",
        "_make_registration",
    ):
        setattr(mock, name, getattr(BaseComponentService, name).__get__(mock))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def svc():
    """Minimal mock of BaseComponentService with real registration methods bound."""
    mock = MagicMock(spec=BaseComponentService)
    mock.id = SERVICE_ID
    mock.service_id = SERVICE_ID
    mock.service_type = "test_service"
    mock.state = "running"
    mock.stop_requested = False
    mock._registration_ack_event = None
    mock.debug = MagicMock()
    mock.info = MagicMock()
    mock.warning = MagicMock()
    mock.run = SimpleNamespace(cfg=SimpleNamespace(runtime=SimpleNamespace()))
    mock.control_client = MagicMock()
    mock.control_client.send = AsyncMock()
    _bind_methods(mock)
    return mock


def _ack_after_n_sends(svc: MagicMock, n: int) -> None:
    """Set the registration ack event after the Nth send call."""
    call_count = 0
    original_send = svc.control_client.send

    async def _counting_send(*args, **kwargs):
        nonlocal call_count
        await original_send(*args, **kwargs)
        call_count += 1
        if call_count >= n and svc._registration_ack_event is not None:
            svc._registration_ack_event.set()

    svc.control_client.send = _counting_send


# ---------------------------------------------------------------------------
# Tests: _register_until_ack (Phase 1)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("time_traveler")
class TestRegistrationPhase:
    """Tests for the DEALER/ROUTER registration probe."""

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_registration_succeeds_immediately(self, svc) -> None:
        """Registration passes on first attempt."""
        _ack_after_n_sends(svc, 1)

        await svc._register_until_ack(
            send_interval=0.1,
            overall_timeout=90.0,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )

        svc.warning.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_registration_retries_until_ack(self, svc) -> None:
        """Registration retries sending until ack event is set."""
        _ack_after_n_sends(svc, 3)

        await svc._register_until_ack(
            send_interval=0.1,
            overall_timeout=90.0,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )

        svc.info.assert_called_once()
        info_msg = svc.info.call_args[0][0]
        assert "Registration" in info_msg
        assert "3 attempts" in info_msg

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_registration_times_out(self, svc) -> None:
        """Registration raises TimeoutError after overall timeout."""
        with pytest.raises(TimeoutError, match="Registration"):
            await svc._register_until_ack(
                send_interval=0.1,
                overall_timeout=1.0,
                initial_warning_threshold=5.0,
                warning_interval=10.0,
            )

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_registration_warns_when_slow(self, svc) -> None:
        """Registration logs warning after initial threshold."""
        _ack_after_n_sends(svc, 60)

        await svc._register_until_ack(
            send_interval=0.1,
            overall_timeout=90.0,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )

        assert svc.warning.call_count >= 1
        warning_msg = svc.warning.call_args[0][0]
        assert "Registration" in warning_msg

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_registration_exits_on_stop_requested(self, svc) -> None:
        """Registration exits cleanly when stop_requested is set."""
        svc.stop_requested = True

        await svc._register_until_ack(
            send_interval=0.1,
            overall_timeout=90.0,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )

        svc.control_client.send.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_registration_cleans_up_event(self, svc) -> None:
        """The ack event is set to None after registration completes."""
        _ack_after_n_sends(svc, 1)

        await svc._register_until_ack(
            send_interval=0.1,
            overall_timeout=90.0,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )

        assert svc._registration_ack_event is None


# ---------------------------------------------------------------------------
# Tests: _run_connection_probes (Phase 1 + Phase 2 orchestration)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("time_traveler")
class TestRunConnectionProbes:
    """Tests for the overridden _run_connection_probes that adds Phase 1."""

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_runs_registration_before_pubsub(self, svc, monkeypatch) -> None:
        """Phase 1 registration probe runs before Phase 2 PUB/SUB probe."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)

        call_order = []

        original_register = BaseComponentService._register_until_ack

        async def _track_registration(self, **kwargs):
            call_order.append("phase1")
            await original_register(self, **kwargs)

        svc._register_until_ack = _track_registration.__get__(svc)

        # Mock super()._run_connection_probes (Phase 2)
        from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin

        async def _track_phase2(self):
            call_order.append("phase2")

        monkeypatch.setattr(
            MessageBusClientMixin, "_run_connection_probes", _track_phase2
        )

        svc._run_connection_probes = (
            BaseComponentService._run_connection_probes.__get__(svc)
        )
        svc._startup_memory_reading = None

        _ack_after_n_sends(svc, 1)

        await svc._run_connection_probes()

        assert call_order == ["phase1", "phase2"]
