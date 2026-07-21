# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.timing.phase.overshoot_poller import OvershootAbandonPoller


def _make_poller(*, cancelled_ids: set[int]) -> OvershootAbandonPoller:
    config = MagicMock()
    config.overshoot_poll_interval_sec = 0
    config.total_expected_requests = 10
    lifecycle = MagicMock()
    lifecycle.is_sending_complete = False
    lifecycle.is_complete = False
    counter = MagicMock()
    counter.requests_completed = 10
    progress = MagicMock()
    credit_router = MagicMock()
    credit_router.cancel_all_credits = AsyncMock(return_value=cancelled_ids)

    return OvershootAbandonPoller(
        config=config,
        lifecycle=lifecycle,
        counter=counter,
        progress=progress,
        credit_router=credit_router,
    )


class TestOvershootAbandonPollerAbandonedCreditIds:
    """abandoned_credit_ids must reflect the credit router's authoritative
    cancel-time snapshot -- the precise ID set later handed to RecordsManager
    for refuse-ingest/purge -- rather than the racy counter subtraction."""

    def test_empty_before_firing(self) -> None:
        poller = _make_poller(cancelled_ids=set())
        assert poller.abandoned_credit_ids == set()
        assert poller.fired is False

    @pytest.mark.asyncio
    async def test_captures_cancel_all_credits_return_value(self) -> None:
        poller = _make_poller(cancelled_ids={1, 2, 3})

        await poller._abandon_now(completed=10, target=10)

        assert poller.abandoned_credit_ids == {1, 2, 3}
        assert poller.fired is True

    @pytest.mark.asyncio
    async def test_no_in_flight_credits_yields_empty_set(self) -> None:
        poller = _make_poller(cancelled_ids=set())

        await poller._abandon_now(completed=10, target=10)

        assert poller.abandoned_credit_ids == set()
        assert poller.fired is True
