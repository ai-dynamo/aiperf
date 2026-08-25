# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Under Kubernetes the API outlives its benchmark until told to stop.

The controller pod deliberately stays up after a run so `aiperf kube results`
can read from it, and is retired explicitly via POST /api/shutdown -- what
`aiperf kube shutdown` and the operator's graceful-exit handshake drive.
Upstream's post-complete grace window is a competing mechanism: it is shorter
than the operator's monitor interval, so the listener vanishes between two
polls, the operator loses the endpoint, and the AIPerfJob never leaves its
pre-terminal phase. Observed on a real GPU cluster: every service container
exited 0 while the CR sat at Initializing/configuring indefinitely.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.api.api_service import FastAPIService
from aiperf.plugin.enums import ServiceRunType


def _service(run_type: ServiceRunType) -> FastAPIService:
    svc = FastAPIService.__new__(FastAPIService)
    svc.run = MagicMock()
    svc.run.cfg.runtime.service_run_type = run_type
    svc.info = lambda *a, **k: None
    svc.debug = lambda *a, **k: None
    return svc


class TestBroadcastShutdownDeference:
    @pytest.mark.asyncio
    async def test_kubernetes_ignores_the_broadcast(self, monkeypatch) -> None:
        svc = _service(ServiceRunType.KUBERNETES)
        stopped = AsyncMock()
        monkeypatch.setattr(
            "aiperf.common.base_component_service.BaseComponentService._on_shutdown_command",
            stopped,
        )
        await FastAPIService._on_shutdown_command(svc, MagicMock())
        stopped.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_type",
        [
            param(ServiceRunType.MULTIPROCESSING, id="multiprocessing"),
        ],
    )  # fmt: skip
    async def test_other_run_types_still_stop(
        self, run_type: ServiceRunType, monkeypatch
    ) -> None:
        svc = _service(run_type)
        stopped = AsyncMock()
        monkeypatch.setattr(
            "aiperf.common.base_component_service.BaseComponentService._on_shutdown_command",
            stopped,
        )
        await FastAPIService._on_shutdown_command(svc, MagicMock())
        stopped.assert_awaited_once()
