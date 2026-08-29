# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Under Kubernetes the API outlives its benchmark until told to stop.

The controller pod deliberately stays up after a run so `aiperf kube results`
can read from it, and is retired explicitly via POST /api/shutdown -- what
`aiperf kube shutdown` and the operator's graceful-exit handshake drive.
Upstream's post-complete grace window is a competing mechanism: it is shorter
than the operator's monitor interval, so the listener vanishes between two
polls, the operator loses the endpoint, and the AIPerfJob never leaves its
pre-terminal phase.

These tests drive :meth:`BaseService._on_shutdown_command` **bound to a
FastAPIService**, i.e. the function the dispatcher actually invokes. The
previous revision called ``FastAPIService._on_shutdown_command`` unbound, which
is why it stayed green for months while the behaviour was dead in production:
that copy was shadowed by ``BaseService``'s hook and never ran. Verified on a
live GPU cluster 2026-08-29 -- the API container exited 0 five seconds after its
benchmark, with no deference log line, and ``aiperf kube results --from-pods``
then failed with "API service may not be listening on port 9090".
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.api.api_service import FastAPIService
from aiperf.common.base_service import BaseService
from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.plugin.enums import ServiceRunType


def _service(run_type: ServiceRunType) -> FastAPIService:
    svc = FastAPIService.__new__(FastAPIService)
    svc.run = MagicMock()
    svc.run.cfg.runtime.service_run_type = run_type
    svc.service_id = "api"
    svc.info = lambda *a, **k: None
    svc.debug = lambda *a, **k: None
    svc.stop = AsyncMock()
    svc._kill = AsyncMock()
    svc.control_client = AsyncMock()
    return svc


class TestBroadcastShutdownDeference:
    @pytest.mark.asyncio
    async def test_kubernetes_does_not_stop_the_api(self) -> None:
        """The whole point: under K8s the API must survive the broadcast."""
        svc = _service(ServiceRunType.KUBERNETES)

        await BaseService._on_shutdown_command(
            svc, Command(cid="c-1", cmd=CommandType.SHUTDOWN)
        )

        svc.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kubernetes_lets_the_dispatcher_send_the_single_ack(self) -> None:
        """Deferring returns normally so the dispatcher acks exactly once.

        Hand-acking here as well would put two responses on the wire for one
        command; the manual ack in the stopping path exists only because stop()
        closes the DEALER before the dispatcher could send it.
        """
        svc = _service(ServiceRunType.KUBERNETES)

        await BaseService._on_shutdown_command(
            svc, Command(cid="c-1", cmd=CommandType.SHUTDOWN)
        )

        svc.control_client.send.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_type",
        [
            param(ServiceRunType.MULTIPROCESSING, id="multiprocessing"),
        ],
    )  # fmt: skip
    async def test_other_run_types_still_stop(self, run_type: ServiceRunType) -> None:
        svc = _service(run_type)

        with pytest.raises(BaseException):  # noqa: B017,PT011 - CancelledError is a BaseException
            await BaseService._on_shutdown_command(
                svc, Command(cid="c-1", cmd=CommandType.SHUTDOWN)
            )

        svc.stop.assert_awaited_once()

    def test_deference_is_a_predicate_override_not_a_second_hook(self) -> None:
        """A second @on_command(SHUTDOWN) on a subclass is unreachable.

        Hook registration walks ``reversed(__mro__)`` and the dispatcher stops
        at the first match, so ``BaseService``'s copy always wins. The carve-out
        must therefore live in the predicate, which ordinary attribute lookup
        resolves to the most-derived class.
        """
        owners = [
            cls.__name__
            for cls in FastAPIService.__mro__
            if "_on_shutdown_command" in cls.__dict__
        ]
        assert owners == ["BaseService"], (
            f"FastAPIService must not redefine the SHUTDOWN hook; found {owners}"
        )
        assert (
            FastAPIService._defers_broadcast_shutdown
            is not BaseService._defers_broadcast_shutdown
        ), "FastAPIService must override the deference predicate"
