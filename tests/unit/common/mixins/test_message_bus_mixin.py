# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the connection probe loop in MessageBusClientMixin."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.environment import Environment
from aiperf.common.messages.command_messages import ConnectionProbeMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin

SERVICE_ID = "test-service-1"

# Warning thresholds hard-coded in _wait_for_successful_probe
INITIAL_WARNING_THRESHOLD = 5.0
WARNING_INTERVAL = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bind_probe_methods(mock: MagicMock) -> None:
    """Bind the real probe methods from MessageBusClientMixin onto a mock."""
    for name in (
        "_wait_for_successful_probe",
        "_probe_and_wait_for_response",
        "_process_connection_probe_message",
    ):
        setattr(mock, name, getattr(MessageBusClientMixin, name).__get__(mock))


def _make_responder(
    mock: MagicMock,
    pub_client: MagicMock,
    *,
    respond_after: int = 1,
    stop_after: int | None = None,
) -> None:
    """Replace mock.publish so probe responses arrive after *respond_after* calls.

    Args:
        respond_after: Set the probe event on the Nth publish call.
        stop_after: Set stop_requested on the Nth publish call (for early-exit tests).
    """

    async def _publish(message: ConnectionProbeMessage) -> None:
        pub_client.publish_calls.append(message)
        n = len(pub_client.publish_calls)
        if stop_after is not None and n >= stop_after:
            mock.stop_requested = True
        if respond_after is not None and n >= respond_after:
            mock._connection_probe_event.set()

    mock.publish = _publish


# ---------------------------------------------------------------------------
# Fixtures (use shared MockPubClient from conftest)
# ---------------------------------------------------------------------------


@pytest.fixture
def mixin(mock_pub_client):
    """Minimal mock of MessageBusClientMixin with real probe methods bound."""
    mock = MagicMock(spec=MessageBusClientMixin)
    mock.id = SERVICE_ID
    mock.stop_requested = False
    mock._connection_probe_event = asyncio.Event()
    mock.debug = MagicMock()
    mock.info = MagicMock()
    mock.warning = MagicMock()
    mock.publish = AsyncMock()
    _bind_probe_methods(mock)
    return mock


# ---------------------------------------------------------------------------
# Tests: _wait_for_successful_probe
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("time_traveler")
class TestProbeLoopSuccess:
    """Tests for successful probe completion paths."""

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_first_attempt_no_info_log(
        self, mixin, mock_pub_client, monkeypatch
    ) -> None:
        """First-attempt success should not emit info or warning logs."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=1)

        await mixin._wait_for_successful_probe()

        assert len(mock_pub_client.publish_calls) == 1
        mixin.info.assert_not_called()
        mixin.warning.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_publishes_correct_message(
        self, mixin, mock_pub_client, monkeypatch
    ) -> None:
        """Probe publishes ConnectionProbeMessage targeting itself."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=1)

        await mixin._wait_for_successful_probe()

        msg = mock_pub_client.publish_calls[0]
        assert isinstance(msg, ConnectionProbeMessage)
        assert msg.service_id == SERVICE_ID
        assert msg.target_service_id == SERVICE_ID

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_no_info_log_on_single_retry(
        self, mixin, mock_pub_client, monkeypatch
    ) -> None:
        """A single retry (2 total attempts) should not emit an info log."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=2)

        await mixin._wait_for_successful_probe()

        assert len(mock_pub_client.publish_calls) == 2
        mixin.info.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.looptime
    @pytest.mark.parametrize("respond_after", [3, 4, 5])
    async def test_retries_logs_info_on_multi_attempt_success(
        self, mixin, mock_pub_client, monkeypatch, respond_after
    ) -> None:
        """When probe succeeds after >2 failed attempts, an info log with attempt count is emitted."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=respond_after)

        await mixin._wait_for_successful_probe()

        assert len(mock_pub_client.publish_calls) == respond_after
        mixin.info.assert_called_once()
        info_msg = mixin.info.call_args[0][0]
        assert f"succeeded after {respond_after} attempts" in info_msg
        assert SERVICE_ID in info_msg


@pytest.mark.usefixtures("time_traveler")
class TestProbeLoopTimeout:
    """Tests for probe timeout behavior."""

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_raises_timeout_error(self, mixin, monkeypatch) -> None:
        """Overall timeout raises TimeoutError when probe never responds."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 5.0)
        # publish is AsyncMock — never sets the event

        with pytest.raises(TimeoutError):
            await mixin._wait_for_successful_probe()


@pytest.mark.usefixtures("time_traveler")
class TestProbeLoopWarnings:
    """Tests for warning log escalation."""

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_warning_after_initial_threshold(
        self, mixin, mock_pub_client, monkeypatch
    ) -> None:
        """A warning is logged once elapsed time >= 5s."""
        probe_interval = 1.0
        # 5 timeouts * 1.0s = 5.0s elapsed → triggers initial_warning_threshold
        respond_after = 6
        monkeypatch.setattr(
            Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", probe_interval
        )
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=respond_after)

        await mixin._wait_for_successful_probe()

        assert mixin.warning.call_count == 1
        warning_msg = mixin.warning.call_args[0][0]
        assert "still waiting" in warning_msg
        assert SERVICE_ID in warning_msg

    @pytest.mark.asyncio
    @pytest.mark.looptime
    @pytest.mark.parametrize(
        ("respond_after", "expected_warnings"),
        [
            (6, 1),  # 5s elapsed  → 1 warning  (at 5s)
            (16, 2),  # 15s elapsed → 2 warnings (at 5s, 15s)
            (26, 3),  # 25s elapsed → 3 warnings (at 5s, 15s, 25s)
        ],
    )
    async def test_warning_escalation_at_intervals(
        self, mixin, mock_pub_client, monkeypatch, respond_after, expected_warnings
    ) -> None:
        """Warnings are logged at 5s, then every 10s after that."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=respond_after)

        await mixin._wait_for_successful_probe()

        assert mixin.warning.call_count == expected_warnings

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_no_warning_when_fast_success(
        self, mixin, mock_pub_client, monkeypatch
    ) -> None:
        """No warnings emitted when probe succeeds within the initial threshold."""
        # 4 timeouts * 1.0s = 4.0s < 5.0s threshold
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=5)

        await mixin._wait_for_successful_probe()

        mixin.warning.assert_not_called()


@pytest.mark.usefixtures("time_traveler")
class TestProbeLoopStopRequested:
    """Tests for early exit via stop_requested."""

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_exits_cleanly_on_stop(
        self, mixin, mock_pub_client, monkeypatch
    ) -> None:
        """Probe loop exits without error when stop_requested is set."""
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_INTERVAL", 1.0)
        monkeypatch.setattr(Environment.SERVICE, "CONNECTION_PROBE_TIMEOUT", 90.0)
        _make_responder(mixin, mock_pub_client, respond_after=None, stop_after=2)

        await mixin._wait_for_successful_probe()

        # No success info because we didn't get a probe response
        mixin.info.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _process_connection_probe_message
# ---------------------------------------------------------------------------


class TestProcessConnectionProbeMessage:
    """Tests for _process_connection_probe_message."""

    @pytest.mark.asyncio
    async def test_sets_connection_probe_event(self, mixin) -> None:
        """Processing a probe message sets the connection probe event."""
        assert not mixin._connection_probe_event.is_set()

        probe_msg = ConnectionProbeMessage(
            service_id=SERVICE_ID, target_service_id=SERVICE_ID
        )
        await mixin._process_connection_probe_message(probe_msg)

        assert mixin._connection_probe_event.is_set()
