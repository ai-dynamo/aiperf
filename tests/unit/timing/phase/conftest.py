# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig


@pytest.fixture
def mock_pub_client() -> MagicMock:
    m = MagicMock()
    m.publish = AsyncMock()
    return m


@pytest.fixture
def mock_phase_publisher() -> MagicMock:
    m = MagicMock()
    m.publish_phase_start = AsyncMock()
    m.publish_phase_sending_complete = AsyncMock()
    m.publish_phase_complete = AsyncMock()
    m.publish_progress = AsyncMock()
    m.publish_credits_complete = AsyncMock()
    return m


@pytest.fixture
def mock_credit_router() -> MagicMock:
    m = MagicMock()
    m.send_credit = AsyncMock()
    m.cancel_all_credits = AsyncMock()
    m.mark_credits_complete = MagicMock()
    m.reset = MagicMock()
    return m


@pytest.fixture
def mock_credit_manager() -> MagicMock:
    m = MagicMock()
    m.configure_for_phase = MagicMock()
    m.release_stuck_slots = MagicMock(return_value=(0, 0))
    return m


@pytest.fixture
def sample_phase_config() -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=100,
    )


@pytest.fixture
def sample_phase_stats() -> CreditPhaseStats:
    return CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        requests_sent=50,
        requests_completed=30,
        requests_cancelled=2,
        final_requests_sent=50,
        start_ns=1000000,
    )
