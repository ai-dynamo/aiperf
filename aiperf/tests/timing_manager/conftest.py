#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.services.timing_manager.config import TimingManagerConfig


class MockCreditManager:
    """Mock implementation of CreditManagerProtocol for testing."""

    def __init__(self):
        self.dropped_credits = []
        self.progress_calls = []
        self.credits_complete_calls = []

    async def drop_credit(self, conversation_id=None, credit_drop_ns=None):
        """Mock drop_credit method."""
        self.dropped_credits.append(
            {"conversation_id": conversation_id, "credit_drop_ns": credit_drop_ns}
        )

    async def publish_progress(self, start_time_ns, total, completed):
        """Mock publish_progress method."""
        self.progress_calls.append(
            {"start_time_ns": start_time_ns, "total": total, "completed": completed}
        )

    async def publish_credits_complete(self, cancelled):
        """Mock publish_credits_complete method."""
        self.credits_complete_calls.append({"cancelled": cancelled})


@pytest.fixture
def mock_credit_manager():
    """Fixture providing a mock credit manager."""
    return MockCreditManager()


@pytest.fixture
def config():
    """Fixture providing a TimingManagerConfig instance."""
    return TimingManagerConfig()
