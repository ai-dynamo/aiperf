# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helper for the ``GET_POD_STATES`` RPC + bus-fed cache fallback.

The progress and debug routers both ask the SystemController for its
authoritative pod-state cache via the ``GET_POD_STATES`` command, and both
fall back to the local bus-fed mirror if the RPC is unavailable. Without a
shared helper the two implementations had already drifted (one rebuilt
``AggregateWorkerStatus`` inline, the other returned a raw payload dict).

This module owns the RPC + decode path; routers transform the result into
their own response shape.
"""

from __future__ import annotations

import logging
from typing import Any

import orjson
from starlette.requests import HTTPConnection

from aiperf.common.control_structs import CommandOk
from aiperf.common.enums import CommandType

_logger = logging.getLogger(__name__)


async def query_controller_pod_states(
    conn: HTTPConnection,
    timeout: float,
) -> dict[str, Any] | None:
    """Issue a ``GET_POD_STATES`` RPC to the SystemController.

    Returns the decoded snapshot dict on success, or ``None`` if the RPC is
    unavailable (controller starting / shutting down, command channel
    missing, timeout, decode error). Callers should fall back to their
    bus-fed mirror cache in that case.

    The snapshot shape is the controller's ``GET_POD_STATES`` response:

    .. code-block:: python

        {
            "pod_states": {pod_index: <WorkerPodStateMessage as dict>, ...},
            "worker_startup_states": {service_id: state_str, ...},
        }

    Args:
        conn: The starlette HTTPConnection (so this works for both HTTP and
              WebSocket dependencies). The FastAPIService is resolved from
              ``conn.app.state.service``.
        timeout: RPC timeout in seconds. Kept short so a slow / unresponsive
              controller falls back to the bus-fed cache rather than hanging
              the endpoint.

    Returns:
        Decoded snapshot dict, or ``None`` if the RPC is unavailable.
    """
    service = getattr(conn.app.state, "service", None)
    send_command = getattr(service, "send_command_to_controller", None)
    if service is None or not callable(send_command):
        return None
    try:
        response = await send_command(
            CommandType.GET_POD_STATES,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - any RPC failure -> cache fallback
        _logger.debug(
            "GET_POD_STATES RPC failed, falling back to bus-fed cache: %r",
            exc,
        )
        return None
    if not isinstance(response, CommandOk):
        return None
    try:
        snapshot = orjson.loads(response.payload)
    except (ValueError, TypeError) as exc:
        _logger.debug("GET_POD_STATES payload decode failed: %r", exc)
        return None
    if not isinstance(snapshot, dict):
        _logger.debug(
            "GET_POD_STATES payload decoded to non-dict top-level: %s",
            type(snapshot).__name__,
        )
        return None
    return snapshot


__all__ = ["query_controller_pod_states"]
