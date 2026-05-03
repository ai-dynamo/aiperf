# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

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
        pub = PhasePublisher(pub_client=mock_pub_client, service_id="tm-001")
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
        pub = PhasePublisher(pub_client=mock_pub_client, service_id="tm-001")
        await pub.publish_phase_sending_complete(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseSendingCompleteMessage)
        assert msg.service_id == "tm-001"
        assert msg.stats is sample_phase_stats

    async def test_publish_phase_complete(
        self, mock_pub_client: MagicMock, sample_phase_stats: CreditPhaseStats
    ) -> None:
        pub = PhasePublisher(pub_client=mock_pub_client, service_id="tm-001")
        await pub.publish_phase_complete(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseCompleteMessage)
        assert msg.service_id == "tm-001"
        assert msg.stats is sample_phase_stats

    async def test_publish_progress(
        self, mock_pub_client: MagicMock, sample_phase_stats: CreditPhaseStats
    ) -> None:
        pub = PhasePublisher(pub_client=mock_pub_client, service_id="tm-001")
        await pub.publish_progress(sample_phase_stats)
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditPhaseProgressMessage)
        assert msg.service_id == "tm-001"
        assert msg.stats is sample_phase_stats

    async def test_publish_credits_complete(self, mock_pub_client: MagicMock) -> None:
        pub = PhasePublisher(pub_client=mock_pub_client, service_id="tm-001")
        await pub.publish_credits_complete()
        mock_pub_client.publish.assert_called_once()
        msg = mock_pub_client.publish.call_args[0][0]
        assert isinstance(msg, CreditsCompleteMessage)
        assert msg.service_id == "tm-001"
