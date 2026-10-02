# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plumbing for the unified-chaos test suite.

Provides:

* ``faults`` (function-scope): a fresh :py:class:`InjectorRegistry` per test,
  pre-registered with :py:class:`EchoInjector` so registry-only tests need
  zero cluster access.
* ``_chaos_namespace_sweeper`` (session-scope, opt-in): on session teardown,
  force-deletes leftover ``aiperf-test-*`` / ``dynamo-test-*`` namespaces and
  the ``chaos-toxiproxy`` namespace for cluster-backed suites that request it.
  **Refuses to run unless ``CHAOS_KUBE_CONTEXT`` or ``CHAOS_KUBECONFIG`` is
  set** — a bare ``kubectl`` without ``--context`` operates on the user's
  default current-context and could delete live production resources.
* ``pytest_addoption`` / ``pytest_configure`` re-exports for
  ``--chaos-sweep`` (cluster-scoped recovery, see :py:mod:`.recovery`).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio

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

CHAOS_KUBE_CONTEXT_ENV = "CHAOS_KUBE_CONTEXT"
"""Env var that points the sweeper at the chaos cluster context, e.g. ``kind-aiperf-chaos``."""

CHAOS_KUBECONFIG_ENV = "CHAOS_KUBECONFIG"
"""Env var pointing at the kubeconfig file for the chaos cluster."""


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


@pytest_asyncio.fixture(scope="package", loop_scope="package", autouse=True)
async def _purge_stale_aiperf_resources() -> AsyncIterator[None]:
    """Override the parent E2E cluster fixture for hermetic adapter tests."""
    yield


@pytest.fixture(scope="session")
def _chaos_namespace_sweeper() -> Iterator[None]:
    """Opt-in fixture that force-deletes chaos namespaces at session teardown.

    The adapter-contract suite is hermetic and does not request this fixture.
    Cluster-backed suites may request it explicitly. Best-effort failures are
    logged so cleanup cannot mask a test exception.

    **Hard-refuses** to run unless ``CHAOS_KUBE_CONTEXT`` or
    ``CHAOS_KUBECONFIG`` is set in the environment — a bare ``kubectl``
    without ``--context`` operates on the user's default current-context
    and could delete live production resources.  There is no safe default.
    """
    yield

    if shutil.which("kubectl") is None:
        return
    context = os.environ.get(CHAOS_KUBE_CONTEXT_ENV, "").strip()
    kubeconfig = os.environ.get(CHAOS_KUBECONFIG_ENV, "").strip()
    if not context and not kubeconfig:
        logger.warning(
            f"chaos sweeper refuses to run without {CHAOS_KUBE_CONTEXT_ENV} "
            f"or {CHAOS_KUBECONFIG_ENV} set — a bare kubectl would operate "
            "on the user's default current-context"
        )
        return
    kube_args = _chaos_kube_args(context, kubeconfig)
    try:
        asyncio.run(_sweep_chaos_namespaces(kube_args))
    except Exception as exc:
        logger.warning(lambda exc=exc: f"chaos namespace sweeper failed: {exc!r}")


def _chaos_kube_args(context: str, kubeconfig: str) -> list[str]:
    """Build --context/--kubeconfig flags for kubectl subprocesses.

    At least one of ``context`` or ``kubeconfig`` must be non-empty;
    callers enforce this before invoking any kubectl command.
    """
    args: list[str] = []
    if kubeconfig:
        args.extend(["--kubeconfig", kubeconfig])
    if context:
        args.extend(["--context", context])
    return args


async def _sweep_chaos_namespaces(kube_args: list[str]) -> None:
    """List + force-delete every namespace matching the chaos contracts."""
    try:
        namespaces = await _list_namespaces(kube_args)
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
        *(_force_delete_namespace(ns, kube_args) for ns in to_delete),
        return_exceptions=True,
    )


async def _list_namespaces(kube_args: list[str]) -> list[str]:
    result = await run_command(
        ["kubectl", *kube_args, "get", "namespaces", "-o", "name"],
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


async def _force_delete_namespace(namespace: str, kube_args: list[str]) -> None:
    """Delete a namespace with ``--wait=false --grace-period=0 --force``.

    The wait=false flag means a stuck finalizer cannot hang the sweeper;
    a follow-up ``pytest --chaos-sweep`` will catch any residue.
    """
    result = await run_command(
        [
            "kubectl",
            *kube_args,
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
