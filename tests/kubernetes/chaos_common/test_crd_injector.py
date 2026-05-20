# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`CRDInjector`.

Drives every dispatch branch with a mock :py:class:`KubectlClient` so we
can assert exactly which kubectl args are forwarded. Cluster access is
not required.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.kubernetes.chaos_common.base import FaultSpec
from tests.kubernetes.chaos_common.injectors.crd import CRDInjector


def _make_kubectl_mock(
    *,
    apply_result: str = "",
    run_returncode: int = 0,
    run_stderr: str = "",
) -> Any:
    """Build a mock KubectlClient that records every ``run`` / ``apply`` call.

    Both methods are ``AsyncMock`` so the injector ``await``s them naturally.
    ``run`` returns a real ``CompletedProcess`` so the injector's rc check
    works without further setup.
    """
    kubectl = AsyncMock()
    kubectl.run = AsyncMock(
        return_value=subprocess.CompletedProcess(
            args=["kubectl"],
            returncode=run_returncode,
            stdout="",
            stderr=run_stderr,
        )
    )
    kubectl.apply = AsyncMock(return_value=apply_result)
    return kubectl


def _make_injector(
    kubectl: Any,
    *,
    cr_kind: str = "aiperfjob",
    cr_api_group: str = "aiperf.nvidia.com",
    operator_namespace: str = "aiperf-system",
    operator_selector: str = "app.kubernetes.io/name=aiperf-operator",
) -> CRDInjector:
    return CRDInjector(
        kubectl=kubectl,
        cr_kind=cr_kind,
        cr_api_group=cr_api_group,
        operator_namespace=operator_namespace,
        operator_selector=operator_selector,
    )


def _run_args(kubectl: Any) -> list[tuple[Any, ...]]:
    """Return the positional args tuple for every ``kubectl.run`` call."""
    return [call.args for call in kubectl.run.call_args_list]


@pytest.mark.asyncio
async def test_crd_delete_uses_configured_kind() -> None:
    """The configured ``cr_kind`` flows through to the kubectl call."""
    kubectl = _make_kubectl_mock()
    injector = _make_injector(kubectl, cr_kind="dynamographdeployment")
    spec = FaultSpec(
        fault_id="crd.delete",
        target={"ns": "dyn-test", "name": "graph-1"},
    )

    async with await injector.inject(spec):
        pass

    assert len(kubectl.run.call_args_list) == 1
    args = _run_args(kubectl)[0]
    # Use the configured kind, NOT the legacy hardcoded "aiperfjob".
    assert args[0] == "delete"
    assert args[1] == "dynamographdeployment"
    assert args[2] == "graph-1"
    assert "-n" in args and "dyn-test" in args
    assert "--wait=false" in args
    # No "aiperfjob" leak anywhere.
    assert "aiperfjob" not in args


@pytest.mark.asyncio
async def test_crd_delete_twice_calls_delete_twice() -> None:
    """Two delete calls fire within ~1 second."""
    kubectl = _make_kubectl_mock()
    injector = _make_injector(kubectl)
    spec = FaultSpec(
        fault_id="crd.delete_twice",
        target={"ns": "aiperf-test-1", "name": "job-x"},
    )

    start = time.monotonic()
    async with await injector.inject(spec) as applied:
        elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert len(kubectl.run.call_args_list) == 2
    for args in _run_args(kubectl):
        assert args[0] == "delete"
        assert args[1] == "aiperfjob"
        assert args[2] == "job-x"
    assert applied.metadata["first_rc"] == 0
    assert applied.metadata["second_rc"] == 0


@pytest.mark.asyncio
async def test_crd_apply_invalid_restore_deletes_cr() -> None:
    """``crd.apply_invalid`` applies the manifest; restore deletes the CR."""
    kubectl = _make_kubectl_mock()
    injector = _make_injector(kubectl)
    manifest = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": "bad-job", "namespace": "aiperf-test-1"},
        "spec": {"benchmark": {"endpoint": {"urls": ["not a url"]}}},
    }
    spec = FaultSpec(
        fault_id="crd.apply_invalid",
        params={"manifest": manifest},
        target={"ns": "aiperf-test-1", "name": "bad-job"},
    )

    async with await injector.inject(spec):
        # Inside the block: apply already happened, no restore yet.
        assert kubectl.apply.await_count == 1
        assert kubectl.run.await_count == 0

    # After exit: restore triggered a delete with --ignore-not-found.
    assert kubectl.run.await_count == 1
    args = _run_args(kubectl)[0]
    assert args[0] == "delete"
    assert args[1] == "aiperfjob"
    assert args[2] == "bad-job"
    assert "--ignore-not-found" in args
    assert "--wait=false" in args


@pytest.mark.asyncio
async def test_crd_patch_with_restore_patch_reverses() -> None:
    """Forward patch + restore patch both run, with the configured type."""
    kubectl = _make_kubectl_mock()
    injector = _make_injector(kubectl)
    spec = FaultSpec(
        fault_id="crd.patch",
        params={
            "patch": '{"spec":{"cancel":true}}',
            "restore_patch": '{"spec":{"cancel":false}}',
            "patch_type": "merge",
        },
        target={"ns": "aiperf-test-1", "name": "job-y"},
    )

    async with await injector.inject(spec):
        pass

    assert len(kubectl.run.call_args_list) == 2
    forward = _run_args(kubectl)[0]
    restore = _run_args(kubectl)[1]
    assert forward[0] == "patch"
    assert forward[1] == "aiperfjob"
    assert forward[2] == "job-y"
    assert "merge" in forward
    assert '{"spec":{"cancel":true}}' in forward
    assert restore[0] == "patch"
    assert restore[1] == "aiperfjob"
    assert '{"spec":{"cancel":false}}' in restore


@pytest.mark.asyncio
async def test_crd_patch_without_restore_patch_warns_no_op() -> None:
    """Restore without ``restore_patch`` logs a warning and runs no kubectl."""
    kubectl = _make_kubectl_mock()
    injector = _make_injector(kubectl)
    spec = FaultSpec(
        fault_id="crd.patch",
        params={"patch": '{"spec":{"cancel":true}}'},
        target={"ns": "aiperf-test-1", "name": "job-z"},
    )

    # Attach a probe handler directly to the injector module's logger. The
    # AIPerfLogger wrapper uses ``self._logger._log`` internally, which goes
    # through Python's standard ``logging`` dispatch -- so a handler on the
    # named logger will see every record regardless of caplog's propagation
    # quirks under the project's custom log setup.
    captured: list[logging.LogRecord] = []

    class _Probe(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    probe = _Probe(level=logging.WARNING)
    target_logger = logging.getLogger("tests.kubernetes.chaos_common.injectors.crd")
    prior_level = target_logger.level
    target_logger.addHandler(probe)
    target_logger.setLevel(logging.WARNING)
    try:
        async with await injector.inject(spec):
            pass
    finally:
        target_logger.removeHandler(probe)
        target_logger.setLevel(prior_level)

    # Only the forward patch ran; no second kubectl call from the no-op restore.
    assert len(kubectl.run.call_args_list) == 1
    assert _run_args(kubectl)[0][0] == "patch"
    # The warning surfaced.
    warning_messages = [rec.getMessage() for rec in captured]
    assert any("restore" in m.lower() for m in warning_messages), (
        f"expected a restore-warning log; got messages={warning_messages!r}"
    )


@pytest.mark.asyncio
async def test_crd_annotate_then_restore_clears() -> None:
    """Annotate sets ``key=value``; restore uses the ``key-`` removal syntax."""
    kubectl = _make_kubectl_mock()
    injector = _make_injector(kubectl)
    key = "aiperf.nvidia.com/completion-claimed"
    spec = FaultSpec(
        fault_id="crd.annotate",
        params={"annotation_key": key, "value": "2026-05-19T00:00:00.000000Z"},
        target={"ns": "aiperf-test-1", "name": "job-a"},
    )

    async with await injector.inject(spec):
        pass

    assert len(kubectl.run.call_args_list) == 2
    set_args = _run_args(kubectl)[0]
    clear_args = _run_args(kubectl)[1]
    assert set_args[0] == "annotate"
    assert set_args[1] == "aiperfjob"
    assert set_args[2] == "job-a"
    assert f"{key}=2026-05-19T00:00:00.000000Z" in set_args
    assert "--overwrite" in set_args
    assert clear_args[0] == "annotate"
    assert f"{key}-" in clear_args
    assert "--overwrite" not in clear_args


@pytest.mark.asyncio
async def test_operator_kill_uses_configured_selector_and_namespace() -> None:
    """``operator.kill`` honors ctor selector/namespace, not AIPerf defaults."""
    kubectl = _make_kubectl_mock()
    injector = _make_injector(
        kubectl,
        operator_namespace="dynamo-system",
        operator_selector="app.kubernetes.io/name=dynamo-operator",
    )
    spec = FaultSpec(fault_id="operator.kill")

    async with await injector.inject(spec):
        pass

    assert len(kubectl.run.call_args_list) == 1
    args = _run_args(kubectl)[0]
    assert args[0] == "delete"
    assert args[1] == "pod"
    assert "-l" in args
    assert "app.kubernetes.io/name=dynamo-operator" in args
    assert "-n" in args
    assert "dynamo-system" in args
    assert "--force" in args
    assert "--grace-period=0" in args
    # No leak of the AIPerf defaults.
    assert "aiperf-system" not in args
    assert "app.kubernetes.io/name=aiperf-operator" not in args


def test_handles_prefix_match_crd_and_operator() -> None:
    """``HANDLES = ('crd', 'operator')`` covers both prefix trees, not unrelated ids."""
    assert CRDInjector.handles("crd") is True
    assert CRDInjector.handles("crd.delete") is True
    assert CRDInjector.handles("crd.delete_twice") is True
    assert CRDInjector.handles("crd.apply_invalid") is True
    assert CRDInjector.handles("crd.patch") is True
    assert CRDInjector.handles("crd.annotate") is True
    assert CRDInjector.handles("operator") is True
    assert CRDInjector.handles("operator.kill") is True
    assert CRDInjector.handles("operator.restart") is True
    # Negative cases.
    assert CRDInjector.handles("crds") is False
    assert CRDInjector.handles("operators") is False
    assert CRDInjector.handles("foo.crd") is False
    assert CRDInjector.handles("echo.crd") is False
    assert CRDInjector.handles("") is False
