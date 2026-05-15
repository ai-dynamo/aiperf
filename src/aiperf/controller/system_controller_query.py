# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-facing query handlers on the SystemController.

These are ``@on_command`` hooks that respond to inbound ``Command`` structs
sent by sidecar services (notably the FastAPI service) over the existing
DEALER↔ROUTER control channel. They expose read-only views of controller
state so sidecars can serve K8s status without trusting their own bus-fed
mirror.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.hooks import on_command

if TYPE_CHECKING:
    from aiperf.common.messages import WorkerPodStateMessage


class SystemControllerQueryMixin:
    """``@on_command`` handlers that return controller-owned snapshots."""

    @on_command(CommandType.GET_POD_STATES)
    async def _on_get_pod_states(self, _message: Command) -> dict[str, object]:
        """Return the current ``_pod_states`` and ``_worker_startup_states``.

        The dispatcher orjson-encodes the returned dict into ``CommandOk.payload``
        so callers decode with ``orjson.loads(response.payload)``.

        Each :class:`WorkerPodStateMessage` is converted via ``model_dump()``
        because orjson does not natively encode msgspec Structs.
        """
        pod_states: dict[str, WorkerPodStateMessage] = getattr(self, "_pod_states", {})
        worker_startup_states: dict[str, str] = getattr(
            self, "_worker_startup_states", {}
        )
        return {
            "pod_states": {
                pod_index: msg.model_dump() for pod_index, msg in pod_states.items()
            },
            "worker_startup_states": dict(worker_startup_states),
        }
