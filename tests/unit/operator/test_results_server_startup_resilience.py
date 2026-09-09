# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The results-server must start even when Kubernetes is unavailable.

Only the live-job endpoints need a cluster. Every results, sweeps, and artifact
route reads the PVC, so an unreachable or misconfigured apiserver has to degrade
this process, not prevent it from starting.

Both failure modes below were hit for real while debugging locally:

* A HANG. ``k8s_client()`` resolving a proxied/unreachable apiserver never
  returned, so the lifespan never yielded. The process bound no port and
  answered nothing, with no error explaining why.
* A ``TypeError``. ``kubernetes_asyncio``'s kubeconfig loader raised
  ``'NoneType' object does not support item assignment`` on a config it could
  not merge. It escaped the except clause and killed startup outright:
  "Application startup failed. Exiting."

Both left a results-server unable to serve the disk it exists to serve.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from kubernetes_asyncio import config as k8s_config

from aiperf.operator.results_server import _build_lifespan


class _HangingClient:
    """Stands in for a k8s_client that never resolves.

    Blocks on an Event that is never set rather than ``asyncio.sleep``. The
    repo installs an always-on auto-fixture that makes ``asyncio.sleep`` return
    instantly, so a sleep-based stand-in does not hang and this test passed
    even with the timeout removed -- caught by reverting the fix and watching
    it stay green.
    """

    async def __aenter__(self) -> None:
        await asyncio.Event().wait()

    async def __aexit__(self, *_: object) -> bool:
        return False


class _RaisingClient:
    """Stands in for a k8s_client whose config load explodes."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self) -> None:
        raise self._exc

    async def __aexit__(self, *_: object) -> bool:
        return False


async def _run_lifespan(
    tmp_path: Path, client_factory, timeout: float = 1.0
) -> tuple[list, float]:
    """Enter and exit the lifespan, returning the api holder and elapsed time."""
    api_holder: list = [None]
    db_holder: list = [None]
    lifespan = _build_lifespan(tmp_path, api_holder, db_holder)
    with (
        patch("aiperf.kubernetes.client.k8s_client", client_factory),
        patch(
            "aiperf.operator.environment.OperatorEnvironment.RESULTS.K8S_INIT_TIMEOUT_SEC",
            timeout,
        ),
    ):
        started = time.monotonic()
        async with lifespan(None):
            elapsed = time.monotonic() - started
    return api_holder, elapsed


@pytest.mark.asyncio
async def test_startup_is_bounded_when_the_k8s_client_hangs(tmp_path: Path) -> None:
    """A client that never resolves must not stall startup indefinitely."""
    api_holder, elapsed = await _run_lifespan(
        tmp_path, lambda *a, **k: _HangingClient(), timeout=1.0
    )

    # Generous bound: the point is that it returns at all, not the exact latency.
    assert elapsed < 30, f"startup was not bounded by the timeout ({elapsed:.1f}s)"
    assert api_holder[0] is None, "live endpoints should be disabled, not half-open"


@pytest.mark.parametrize(
    "exc",
    [
        # The kubeconfig-merge failure that killed startup outright.
        TypeError("'NoneType' object does not support item assignment"),
        k8s_config.ConfigException("Service host/port is not set."),
        ValueError("malformed kubeconfig"),
        KeyError("current-context"),
        OSError("connection refused"),
    ],
    ids=["typeerror-kubeconfig", "config-exception", "valueerror", "keyerror", "oserror"],
)  # fmt: skip
@pytest.mark.asyncio
async def test_startup_survives_client_construction_failures(
    tmp_path: Path, exc: Exception
) -> None:
    """Every plausible config/transport failure degrades instead of aborting."""
    api_holder, _ = await _run_lifespan(tmp_path, lambda *a, **k: _RaisingClient(exc))

    assert api_holder[0] is None


@pytest.mark.asyncio
async def test_an_unexpected_error_still_propagates(tmp_path: Path) -> None:
    """The guard is a list, not a bare except.

    Broadening this to catch everything would turn a genuine bug in the client
    into a silently degraded server, which is exactly the failure mode this
    module exists to make loud.
    """

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        await _run_lifespan(tmp_path, lambda *a, **k: _RaisingClient(_Boom("bug")))
