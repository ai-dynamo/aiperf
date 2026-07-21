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
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.models.error_models import ErrorDetails

if TYPE_CHECKING:
    from aiperf.common.base_component_service import BaseComponentService


async def await_dataset_configured(
    service: BaseComponentService, event: asyncio.Event
) -> bool:
    """Block until the dataset-configured ``event`` is set.

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
