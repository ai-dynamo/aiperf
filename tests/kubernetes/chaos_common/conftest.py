# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plumbing for the unified-chaos test suite.

Provides:

* ``faults`` (function-scope): a fresh :py:class:`InjectorRegistry` per test,
  pre-registered with :py:class:`EchoInjector` so registry-only tests need
  zero cluster access.
* ``_chaos_namespace_sweeper`` (session-scope, autouse): on session teardown,
  force-deletes leftover ``aiperf-test-*`` / ``dynamo-test-*`` namespaces and
  the ``chaos-toxiproxy`` namespace so test crashes do not leave debris.
* ``pytest_addoption`` / ``pytest_configure`` re-exports for
  ``--chaos-sweep`` (cluster-scoped recovery, see :py:mod:`.recovery`).
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterator

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.subproc import run_command
from tests.kubernetes.chaos_common.injectors.echo import EchoInjector
from tests.kubernetes.chaos_common.recovery import (
    pytest_addoption as _recovery_addoption,
)
from tests.kubernetes.chaos_common.recovery import (
    pytest_configure as _recovery_configure,
)
from tests.kubernetes.chaos_common.registry import InjectorRegistry

logger = AIPerfLogger(__name__)

CHAOS_NAMESPACE_PREFIXES: tuple[str, ...] = ("aiperf-test-", "dynamo-test-")
"""Per-test-run namespace prefixes the sweeper will force-delete on teardown."""

CHAOS_STATIC_NAMESPACES: tuple[str, ...] = ("chaos-toxiproxy",)
"""Long-lived chaos infra namespaces the sweeper will force-delete on teardown."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Forward ``--chaos-sweep`` registration to :py:mod:`.recovery`."""
    _recovery_addoption(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Forward ``--chaos-sweep`` handling to :py:mod:`.recovery`.

    When the flag is passed, :py:mod:`.recovery` calls ``pytest.exit``, so
    no further configuration runs.
    """
    _recovery_configure(config)


@pytest.fixture
def faults() -> InjectorRegistry:
    """Per-test :py:class:`InjectorRegistry` pre-loaded with EchoInjector.

    Concrete suites that need cluster-backed injectors should request this
    fixture and ``register()`` additional injectors on top.
    """
    registry = InjectorRegistry()
    registry.register(EchoInjector())
    return registry


@pytest.fixture(scope="session", autouse=True)
def _chaos_namespace_sweeper() -> Iterator[None]:
    """Force-delete leftover chaos namespaces at session teardown.

    Best-effort: if ``kubectl`` is not on PATH (e.g. unit-test-only runs),
    the sweeper silently skips. Failures during deletion are logged, not
    raised, so a flaky cluster cannot mask a test exception.
    """
    yield

    if shutil.which("kubectl") is None:
        return
    try:
        asyncio.run(_sweep_chaos_namespaces())
    except Exception as exc:
        logger.warning(lambda exc=exc: f"chaos namespace sweeper failed: {exc!r}")


async def _sweep_chaos_namespaces() -> None:
    """List + force-delete every namespace matching the chaos contracts."""
    try:
        namespaces = await _list_namespaces()
    except Exception as exc:
        logger.warning(
            lambda exc=exc: (
                f"chaos sweeper could not list namespaces: {exc!r}; "
                "skipping (set up kubeconfig or use --chaos-sweep manually)"
            )
        )
        return

    to_delete: list[str] = []
    for ns in namespaces:
        if (
            any(ns.startswith(prefix) for prefix in CHAOS_NAMESPACE_PREFIXES)
            or ns in CHAOS_STATIC_NAMESPACES
        ):
            to_delete.append(ns)

    if not to_delete:
        return
    logger.info(
        lambda n=to_delete: (
            f"chaos sweeper: force-deleting {len(n)} leftover namespace(s): {n}"
        )
    )
    # Issue all deletes in parallel, with --wait=false so a stuck namespace
    # cannot block the rest of session teardown.
    await asyncio.gather(
        *(_force_delete_namespace(ns) for ns in to_delete),
        return_exceptions=True,
    )


async def _list_namespaces() -> list[str]:
    result = await run_command(
        ["kubectl", "get", "namespaces", "-o", "name"],
        timeout=30.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kubectl get namespaces failed (rc={result.returncode}): {result.stderr!r}"
        )
    return [
        line.removeprefix("namespace/").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


async def _force_delete_namespace(namespace: str) -> None:
    """Delete a namespace with ``--wait=false --grace-period=0 --force``.

    The wait=false flag means a stuck finalizer cannot hang the sweeper;
    a follow-up ``pytest --chaos-sweep`` will catch any residue.
    """
    result = await run_command(
        [
            "kubectl",
            "delete",
            "namespace",
            namespace,
            "--ignore-not-found",
            "--wait=false",
            "--grace-period=0",
            "--force",
        ],
        timeout=30.0,
    )
    if result.returncode != 0:
        logger.warning(
            lambda ns=namespace, err=result.stderr: (
                f"chaos sweeper: kubectl delete ns/{ns} returned non-zero: {err!r}"
            )
        )


# Re-export so static analysis sees the symbol live in this module
# (some test runners introspect ``conftest.pytest_addoption`` directly).
__all__: list[str] = [
    "CHAOS_NAMESPACE_PREFIXES",
    "CHAOS_STATIC_NAMESPACES",
    "faults",
    "pytest_addoption",
    "pytest_configure",
]
