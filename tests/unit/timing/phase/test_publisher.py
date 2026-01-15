# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PhasePublisher."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseStats
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
)
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.publisher import PhasePublisher


@pytest.fixture
def publisher(mock_pub_client: MagicMock) -> PhasePublisher:
    return PhasePublisher(pub_client=mock_pub_client, service_id="timing-manager-001")


@pytest.mark.asyncio
class TestPublishPhaseLifecycle:
    async def test_creates_phase_start_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_config: CreditPhaseConfig,
        sample_phase_stats: CreditPhaseStats,
    ) -> None:
        await publisher.publish_phase_start(sample_phase_config, sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseStartMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats
        assert msg.config is sample_phase_config

    async def test_creates_sending_complete_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_stats: CreditPhaseStats,
    ) -> None:
        await publisher.publish_phase_sending_complete(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseSendingCompleteMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats

    async def test_creates_phase_complete_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_stats: CreditPhaseStats,
    ) -> None:
        await publisher.publish_phase_complete(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseCompleteMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats


@pytest.mark.asyncio
class TestPublishProgress:
    async def test_creates_progress_message(
        self,
        publisher: PhasePublisher,
        mock_pub_client: MagicMock,
        sample_phase_stats: CreditPhaseStats,
    ) -> None:
        await publisher.publish_progress(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseProgressMessage)
        assert msg.service_id == "timing-manager-001"
        assert msg.stats is sample_phase_stats


@pytest.mark.asyncio
class TestPhasePublisherIntegration:
    async def test_all_lifecycle_events_use_same_service_id(
        self,
        mock_pub_client: MagicMock,
        sample_phase_config: CreditPhaseConfig,
        sample_phase_stats: CreditPhaseStats,
    ) -> None:
        service_id = "consistent-service-id"
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id=service_id)
        await publisher.publish_phase_start(sample_phase_config, sample_phase_stats)
        await publisher.publish_phase_sending_complete(sample_phase_stats)
        await publisher.publish_phase_complete(sample_phase_stats)
        await publisher.publish_progress(sample_phase_stats)
        await publisher.publish_credits_complete()
        assert mock_pub_client.publish.call_count == 5
        for call in mock_pub_client.publish.call_args_list:
            msg = call[0][0]
            assert msg.service_id == service_id

    async def test_different_stats_produce_different_messages(
        self, mock_pub_client: MagicMock
    ) -> None:
        publisher = PhasePublisher(pub_client=mock_pub_client, service_id="test-id")
        config1 = CreditPhaseConfig(
            phase=CreditPhase.WARMUP,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=10,
        )
        stats1 = CreditPhaseStats(
            phase=CreditPhase.WARMUP,
            requests_sent=10,
            requests_completed=5,
            requests_cancelled=0,
            final_requests_sent=10,
            start_ns=1000,
        )
        config2 = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            total_expected_requests=100,
        )
        stats2 = CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            requests_sent=100,
            requests_completed=90,
            requests_cancelled=2,
            final_requests_sent=100,
            start_ns=2000,
        )
        await publisher.publish_phase_start(config1, stats1)
        await publisher.publish_phase_start(config2, stats2)
        calls = mock_pub_client.publish.call_args_list
        msg1 = calls[0][0][0]
        msg2 = calls[1][0][0]
        assert msg1.stats.phase == CreditPhase.WARMUP
        assert msg2.stats.phase == CreditPhase.PROFILING
        assert msg1.stats.requests_sent == 10
        assert msg2.stats.requests_sent == 100
