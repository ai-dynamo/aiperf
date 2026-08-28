# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared dataset-configuration gate for the record-processing services.

RecordProcessor and RecordsManager receive the DatasetConfiguredNotification on
the PUB/SUB bus but records on a separate PULL socket, with no ordering guarantee
between the two. Both must block record processing until the notification has
configured their processors (e.g. accuracy ground truths / task names).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.messages.dataset_messages import (
    DatasetConfigStatusRequest,
    DatasetConfigStatusResponse,
    DatasetConfiguredNotification,
)
from aiperf.common.models.error_models import ErrorDetails

if TYPE_CHECKING:
    from aiperf.common.base_component_service import BaseComponentService
    from aiperf.common.protocols import RequestClientProtocol


class DatasetConfigCatchUp:
    """One-shot late-join catch-up for ``DatasetConfiguredNotification``.

    ``DatasetConfiguredNotification`` is a single one-shot PUB/SUB broadcast:
    DatasetManager sends it once and never replays it. A subscriber (e.g. a
    RecordProcessor replica) that finishes subscribing *after* DatasetManager
    already published it never receives it, and without this class blocks
    until ``Environment.DATASET.CONFIGURATION_TIMEOUT``.

    This issues one bounded request/response query to DatasetManager asking
    whether configuration has already completed, and if so applies it
    immediately via ``on_configured`` instead of waiting out the full
    timeout. It is a self-heal for a missed broadcast, not the primary path
    -- the normal PUB/SUB notification is still expected to be the common
    case, and a failed or negative catch-up simply falls back to waiting on
    it. Concurrent callers (multiple records arriving before the gate opens)
    share a single in-flight attempt via the lock; only the first pays the
    request/response round trip.
    """

    def __init__(
        self,
        request_client: RequestClientProtocol,
        on_configured: Callable[[DatasetConfiguredNotification], Awaitable[None]],
        service_id: str,
    ) -> None:
        self._request_client = request_client
        self._on_configured = on_configured
        self._service_id = service_id
        self._lock = asyncio.Lock()
        self._attempted = False

    async def try_once(self, event: asyncio.Event) -> None:
        """Attempt the catch-up query exactly once for this instance's lifetime.

        No-ops immediately (without taking the lock) once ``event`` is set or
        a prior attempt has already run, so the hot path (event already set)
        never pays the lock acquisition after the first call.
        """
        if event.is_set() or self._attempted:
            return
        async with self._lock:
            if event.is_set() or self._attempted:
                return
            self._attempted = True
            try:
                response = await self._request_client.request(
                    DatasetConfigStatusRequest(service_id=self._service_id),
                    timeout=Environment.DATASET.CATCH_UP_REQUEST_TIMEOUT,
                )
            except Exception:
                # Not fatal: DatasetManager may not be reachable yet, or the
                # request may simply time out. The caller falls back to
                # waiting on the normal PUB/SUB notification.
                return
            if (
                isinstance(response, DatasetConfigStatusResponse)
                and response.notification is not None
            ):
                await self._on_configured(response.notification)


async def await_dataset_configured(
    service: BaseComponentService,
    event: asyncio.Event,
    catch_up: DatasetConfigCatchUp | None = None,
) -> bool:
    """Block until the dataset-configured ``event`` is set.

    If ``catch_up`` is provided and the event isn't already set, makes one
    bounded attempt to actively pull the configuration from DatasetManager
    before falling back to the timed wait -- recovers a subscriber that
    missed the one-shot ``DatasetConfiguredNotification`` broadcast instead
    of always blocking for the full ``CONFIGURATION_TIMEOUT``.

    Returns True once configured. On timeout, treats a missing dataset
    configuration as a fatal misconfiguration: reports it via a
    BaseServiceErrorMessage (so the run exits non-zero) and kills the service so
    the run aborts loudly instead of processing records without a configured
    dataset. Returns False in that case so the caller skips processing (``_kill``
    force-exits the process, so this return is a safety net if it ever does not).
    """
    # Fast path: once configured (the common case), avoid the per-record
    # wait_for timer allocation on the hot path.
    if event.is_set():
        return True
    if catch_up is not None:
        await catch_up.try_once(event)
        if event.is_set():
            return True
    try:
        await asyncio.wait_for(
            event.wait(), timeout=Environment.DATASET.CONFIGURATION_TIMEOUT
        )
        return True
    except TimeoutError:
        message = (
            "Dataset configuration not received after "
            f"{Environment.DATASET.CONFIGURATION_TIMEOUT}s; aborting run."
        )
        service.error(message)
        await service.publish(
            BaseServiceErrorMessage(
                service_id=service.service_id,
                error=ErrorDetails(message=message),
            )
        )
        await service._kill()
        return False
