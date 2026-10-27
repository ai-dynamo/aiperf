# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.models import CreditPhaseStats
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.publisher import PhasePublisher


@pytest.mark.asyncio
class TestPhasePublisher:
    async def test_publish_phase_start(
        self,
        mock_pub_client: MagicMock,
        sample_phase_config: CreditPhaseConfig,
        sample_phase_stats: CreditPhaseStats,
    ) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
        )
        await pub.publish_phase_start(sample_phase_config, sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseStartMessage)
        assert msg.service_id == "tm-001"
        assert msg.stats is sample_phase_stats
        assert msg.config is sample_phase_config

    async def test_publish_sending_complete(
        self, mock_pub_client: MagicMock, sample_phase_stats: CreditPhaseStats
    ) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
        )
        await pub.publish_phase_sending_complete(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseSendingCompleteMessage)
        assert msg.service_id == "tm-001"
        assert msg.stats is sample_phase_stats

    async def test_publish_phase_complete(
        self, mock_pub_client: MagicMock, sample_phase_stats: CreditPhaseStats
    ) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
        )
        await pub.publish_phase_complete(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseCompleteMessage)
        assert msg.service_id == "tm-001"
        assert msg.stats is sample_phase_stats

    async def test_publish_progress(
        self, mock_pub_client: MagicMock, sample_phase_stats: CreditPhaseStats
    ) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
        )
        await pub.publish_progress(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseProgressMessage)
        assert msg.service_id == "tm-001"
        assert msg.stats is sample_phase_stats

    async def test_publish_credits_complete(self, mock_pub_client: MagicMock) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
        )
        await pub.publish_credits_complete()
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditsCompleteMessage)
        assert msg.service_id == "tm-001"

    async def test_request_profile_cancel_uses_injected_sender(
        self, mock_pub_client: MagicMock
    ) -> None:
        """PROFILE_CANCEL goes to the controller, never onto the pub bus.

        The controller owns the peer fan-out; a publish here would reach nobody
        now that no service subscribes to command messages.
        """
        sender = AsyncMock()
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=sender,
        )
        await pub.request_profile_cancel()
        sender.assert_awaited_once()
        mock_pub_client.publish.assert_not_called()

    async def test_update_server_metrics_runtime_status_active(
        self, mock_pub_client: MagicMock
    ) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
        )
        assert pub.server_metrics_runtime_active is False
        assert pub.server_metrics_collector_count == 0

        pub.update_server_metrics_runtime_status(
            enabled=True,
            endpoints_reachable=[
                "http://localhost:8000/metrics",
                "http://localhost:8001/metrics",
            ],
        )
        assert pub.server_metrics_runtime_active is True
        assert pub.server_metrics_collector_count == 2

    async def test_update_server_metrics_runtime_status_inactive_when_unreachable(
        self, mock_pub_client: MagicMock
    ) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
        )
        pub.update_server_metrics_runtime_status(
            enabled=False,
            endpoints_reachable=[],
        )
        assert pub.server_metrics_runtime_active is False
        assert pub.server_metrics_collector_count == 0

        # enabled=True with empty reachable list is still inactive
        pub.update_server_metrics_runtime_status(
            enabled=True,
            endpoints_reachable=[],
        )
        assert pub.server_metrics_runtime_active is False
        assert pub.server_metrics_collector_count == 0

    async def test_wait_for_warmup_boundary_ready_returns_on_ready(
        self, mock_pub_client: MagicMock
    ) -> None:
        ready = asyncio.Event()
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
            warmup_boundary_ready=ready,
        )
        asyncio.get_running_loop().call_later(0.01, ready.set)
        assert await pub.wait_for_warmup_boundary_ready(5.0) is True

    async def test_wait_for_warmup_boundary_ready_times_out(
        self, mock_pub_client: MagicMock
    ) -> None:
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
            warmup_boundary_ready=asyncio.Event(),
        )
        assert await pub.wait_for_warmup_boundary_ready(0.01) is False

    async def test_wait_for_warmup_boundary_ready_stops_on_cancel_event(
        self, mock_pub_client: MagicMock
    ) -> None:
        """A cancel during the wait returns promptly instead of running the full budget."""
        cancel_event = asyncio.Event()
        pub = PhasePublisher(
            pub_client=mock_pub_client,
            service_id="tm-001",
            profile_cancel_sender=AsyncMock(),
            warmup_boundary_ready=asyncio.Event(),
        )
        loop = asyncio.get_running_loop()
        loop.call_later(0.01, cancel_event.set)
        start = loop.time()
        assert (
            await pub.wait_for_warmup_boundary_ready(30.0, cancel_event=cancel_event)
            is False
        )
        assert loop.time() - start < 5.0
