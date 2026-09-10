# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative worker-state query shared by progress and debug routers."""

from __future__ import annotations

import logging

import orjson
from starlette.requests import HTTPConnection

from aiperf.common.control_structs import CommandOk
from aiperf.common.enums import CommandType
from aiperf.common.environment import Environment
from aiperf.controller.system_controller_models import PodStateSnapshot

_logger = logging.getLogger(__name__)


async def query_controller_pod_states(
    conn: HTTPConnection,
) -> PodStateSnapshot | None:
    """Return controller state, or ``None`` when the authoritative path fails.

    A local controller handle is used when both components share a process.
    Kubernetes API sidecars go over the DEALER/ROUTER control channel. Callers
    retain their bus-fed cache as the availability fallback for controller
    startup, shutdown, timeouts, and malformed responses.
    """
    controller = getattr(conn.app.state, "controller", None)
    if controller is None:
        service = getattr(conn.app.state, "service", None)
        controller = getattr(service, "controller", None)
    getter = getattr(controller, "get_pod_state_snapshot", None)
    if callable(getter):
        return getter()

    service = getattr(conn.app.state, "service", None)
    send_command = getattr(service, "send_command_to_controller", None)
    if not callable(send_command):
        return None

    try:
        response = await send_command(
            CommandType.GET_POD_STATES,
            timeout=Environment.API_SERVER.GET_POD_STATES_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - all transport failures use the cache
        _logger.debug("Controller worker-state query failed; using bus cache: %r", exc)
        return None
    if not isinstance(response, CommandOk):
        return None
    try:
        return PodStateSnapshot.model_validate(orjson.loads(response.payload))
    except (TypeError, ValueError, orjson.JSONDecodeError) as exc:
        _logger.debug(
            "Controller worker-state response was invalid; using bus cache: %r",
            exc,
        )
        return None


__all__ = ["query_controller_pod_states"]
