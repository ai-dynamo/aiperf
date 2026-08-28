# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the dataset-configuration gate.

DatasetConfiguredNotification is a one-shot PUB/SUB broadcast with no
replay. A RecordProcessor/RecordsManager that finishes subscribing after
DatasetManager already published it never receives it and, without
DatasetConfigCatchUp, blocks until CONFIGURATION_TIMEOUT (300s default).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.messages.dataset_messages import (
    DatasetConfigStatusRequest,
    DatasetConfigStatusResponse,
    DatasetConfiguredNotification,
)
from aiperf.records.dataset_gate import DatasetConfigCatchUp, await_dataset_configured


def _make_notification() -> DatasetConfiguredNotification:
    return DatasetConfiguredNotification.model_construct(
        service_id="dataset_manager",
        metadata=MagicMock(),
        client_metadata=MagicMock(),
    )


class TestDatasetConfigCatchUp:
    @pytest.mark.asyncio
    async def test_try_once_noop_if_event_already_set(self):
        """No request should be sent once the normal PUB/SUB path already
        configured the event -- the catch-up is purely a fallback."""
        request_client = AsyncMock()
        on_configured = AsyncMock()
        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=on_configured,
            service_id="rp-1",
        )
        event = asyncio.Event()
        event.set()

        await catch_up.try_once(event)

        request_client.request.assert_not_called()
        on_configured.assert_not_called()

    @pytest.mark.asyncio
    async def test_try_once_applies_notification_when_already_configured(self):
        """The core self-heal: DatasetManager reports it already published
        the notification, so the late joiner applies it immediately instead
        of waiting out the full timeout."""
        notification = _make_notification()
        request_client = AsyncMock()
        request_client.request.return_value = DatasetConfigStatusResponse(
            service_id="dataset_manager", notification=notification
        )
        on_configured = AsyncMock()
        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=on_configured,
            service_id="rp-1",
        )
        event = asyncio.Event()

        await catch_up.try_once(event)

        request_client.request.assert_awaited_once()
        sent_message = request_client.request.await_args.args[0]
        assert isinstance(sent_message, DatasetConfigStatusRequest)
        assert sent_message.service_id == "rp-1"
        on_configured.assert_awaited_once_with(notification)

    @pytest.mark.asyncio
    async def test_try_once_falls_back_when_not_yet_configured(self):
        """DatasetManager hasn't configured yet: no-op, caller falls back to
        waiting on the normal PUB/SUB notification."""
        request_client = AsyncMock()
        request_client.request.return_value = DatasetConfigStatusResponse(
            service_id="dataset_manager", notification=None
        )
        on_configured = AsyncMock()
        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=on_configured,
            service_id="rp-1",
        )
        event = asyncio.Event()

        await catch_up.try_once(event)

        on_configured.assert_not_called()
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_try_once_swallows_request_failure(self):
        """DatasetManager unreachable/timeout: not fatal, falls back to the
        normal PUB/SUB wait instead of propagating."""
        request_client = AsyncMock()
        request_client.request.side_effect = TimeoutError
        on_configured = AsyncMock()
        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=on_configured,
            service_id="rp-1",
        )
        event = asyncio.Event()

        await catch_up.try_once(event)

        on_configured.assert_not_called()

    @pytest.mark.asyncio
    async def test_try_once_only_queries_once_across_concurrent_callers(self):
        """Multiple records arriving concurrently before the gate opens must
        share a single in-flight request, not one each.

        Uses an explicit release gate (rather than a bare ``asyncio.sleep``,
        which this suite's autouse fixture makes resolve instantly) so the
        request is provably still in flight while the other four callers run
        ``try_once`` -- deterministic proof of single-flight, not an
        artifact of task-scheduling order.
        """
        release = asyncio.Event()
        request_client = AsyncMock()

        async def slow_request(*args, **kwargs):
            await release.wait()
            return DatasetConfigStatusResponse(service_id="dataset_manager")

        request_client.request.side_effect = slow_request
        on_configured = AsyncMock()
        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=on_configured,
            service_id="rp-1",
        )
        event = asyncio.Event()

        tasks = [asyncio.create_task(catch_up.try_once(event)) for _ in range(5)]
        await asyncio.sleep(0)  # let every task reach the in-flight request

        request_client.request.assert_awaited_once()

        release.set()
        await asyncio.gather(*tasks)

        request_client.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_try_once_does_not_requery_after_a_negative_attempt(self):
        """A negative catch-up result (not yet configured) must not trigger a
        second request on a later call -- it's a one-shot attempt per
        instance; the normal PUB/SUB path is the source of truth after that."""
        request_client = AsyncMock()
        request_client.request.return_value = DatasetConfigStatusResponse(
            service_id="dataset_manager", notification=None
        )
        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=AsyncMock(),
            service_id="rp-1",
        )
        event = asyncio.Event()

        await catch_up.try_once(event)
        await catch_up.try_once(event)

        request_client.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_try_once_does_not_reapply_if_notification_wins_the_race(self):
        """Regression: if the normal PUB/SUB notification is delivered (and
        applied, setting the event) while the catch-up request is still in
        flight, the cached response that arrives afterward must NOT be
        re-applied -- otherwise producers/observers get configured twice."""
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_request(*args, **kwargs):
            entered.set()
            await release.wait()
            return DatasetConfigStatusResponse(
                service_id="dataset_manager", notification=_make_notification()
            )

        request_client = AsyncMock()
        request_client.request.side_effect = slow_request
        on_configured = AsyncMock()
        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=on_configured,
            service_id="rp-1",
        )
        event = asyncio.Event()

        task = asyncio.create_task(catch_up.try_once(event))
        await entered.wait()

        # The normal PUB/SUB path wins the race: it applies configuration
        # and sets the event while the catch-up request is still pending.
        event.set()
        release.set()
        await task

        on_configured.assert_not_called()


class TestAwaitDatasetConfiguredWithCatchUp:
    @pytest.mark.asyncio
    async def test_catch_up_unblocks_immediately_without_waiting_out_timeout(
        self, monkeypatch
    ):
        """End-to-end: a late joiner passed a catch_up that successfully
        recovers the config must return True immediately, never touching the
        CONFIGURATION_TIMEOUT wait path."""
        notification = _make_notification()
        request_client = AsyncMock()
        request_client.request.return_value = DatasetConfigStatusResponse(
            service_id="dataset_manager", notification=notification
        )

        service = MagicMock()
        event = asyncio.Event()

        applied: list[DatasetConfiguredNotification] = []

        async def on_configured(msg: DatasetConfiguredNotification) -> None:
            applied.append(msg)
            event.set()

        catch_up = DatasetConfigCatchUp(
            request_client=request_client,
            on_configured=on_configured,
            service_id="rp-1",
        )

        async def fail_if_called(*args, **kwargs):
            raise AssertionError(
                "should not fall through to the timed wait after a successful catch-up"
            )

        monkeypatch.setattr(asyncio, "wait_for", fail_if_called)

        result = await await_dataset_configured(service, event, catch_up)

        assert result is True
        assert applied == [notification]

    @pytest.mark.asyncio
    async def test_no_catch_up_falls_back_to_normal_wait(self):
        """catch_up=None (unset, e.g. legacy/test double) must skip the
        catch-up path entirely and block on the event directly, exactly like
        the pre-fix gate -- proven by starting with the event unset (so the
        fast path can't short-circuit the assertion) and only completing
        once the event is set from outside."""
        service = MagicMock()
        event = asyncio.Event()

        task = asyncio.create_task(await_dataset_configured(service, event, None))
        await asyncio.sleep(0)
        assert not task.done()

        event.set()
        result = await asyncio.wait_for(task, timeout=1.0)

        assert result is True
