# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative worker-state query shared by progress and debug routers."""

from __future__ import annotations

import logging

from starlette.requests import HTTPConnection

from aiperf.common.environment import Environment
from aiperf.common.messages import CommandSuccessResponse, GetPodStatesCommand
from aiperf.controller.system_controller_models import PodStateSnapshot
from aiperf.plugin.enums import ServiceType

_logger = logging.getLogger(__name__)


async def query_controller_pod_states(
    conn: HTTPConnection,
) -> PodStateSnapshot | None:
    """Return controller state, or ``None`` when the authoritative path fails.

    A local controller handle is used when both components share a process.
    Kubernetes API sidecars use the typed command bus. Callers retain their
    bus-fed cache as the availability fallback for controller startup,
    shutdown, timeouts, and malformed responses.
    """
    controller = getattr(conn.app.state, "controller", None)
    if controller is None:
        service = getattr(conn.app.state, "service", None)
        controller = getattr(service, "controller", None)
    getter = getattr(controller, "get_pod_state_snapshot", None)
    if callable(getter):
        return getter()

    service = getattr(conn.app.state, "service", None)
    send_command = getattr(service, "send_command_and_wait_for_response", None)
    service_id = getattr(service, "service_id", None)
    if not callable(send_command) or not isinstance(service_id, str):
        return None

    try:
        response = await send_command(
            GetPodStatesCommand(
                service_id=service_id,
                target_service_type=ServiceType.SYSTEM_CONTROLLER,
            ),
            timeout=Environment.API_SERVER.GET_POD_STATES_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - all transport failures use the cache
        _logger.debug("Controller worker-state query failed; using bus cache: %r", exc)
        return None
    if not isinstance(response, CommandSuccessResponse):
        return None
    try:
        return PodStateSnapshot.model_validate(response.data)
    except (TypeError, ValueError) as exc:
        _logger.debug(
            "Controller worker-state response was invalid; using bus cache: %r",
            exc,
        )
        return None


__all__ = ["query_controller_pod_states"]
