# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Kubernetes cleanup assertion deadlines."""

from __future__ import annotations

import asyncio
from types import ModuleType, SimpleNamespace

import pytest
from pytest import param

from tests.kubernetes import test_helm, test_operator
from tests.kubernetes.helpers import deadline as deadline_helper
from tests.kubernetes.helpers import kubectl as kubectl_helper
from tests.kubernetes.helpers.deadline import await_before_deadline
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig


class _BlockingCreateDeployer:
    def __init__(self) -> None:
        self.default_job_namespace = "test-ns"
        self.cancelled = asyncio.Event()
        self.created_name: str | None = None
        self.created_namespace: str | None = None
        self.deleted_job: tuple[str, str] | None = None

    async def create_job(
        self,
        _config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.created_name = name
        self.created_namespace = namespace
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()

    async def delete_job(self, name: str, namespace: str) -> None:
        self.deleted_job = (name, namespace)


class _DelayedCreateAfterDeleteDeployer:
    def __init__(self) -> None:
        self.default_job_namespace = "test-ns"
        self.created_name: str | None = None
        self.created_namespace: str | None = None
        self.create_cancelled = asyncio.Event()
        self.first_delete_completed = asyncio.Event()
        self.delayed_create_committed = asyncio.Event()
        self.delete_calls = 0
        self.resource_exists = False
        self.server_create_task: asyncio.Task[None] | None = None

    async def create_job(
        self,
        _config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.created_name = name
        self.created_namespace = namespace

        async def commit_after_first_delete() -> None:
            await self.first_delete_completed.wait()
            await asyncio.sleep(0)
            self.resource_exists = True
            self.delayed_create_committed.set()

        self.server_create_task = asyncio.create_task(commit_after_first_delete())
        try:
            await asyncio.Event().wait()
        finally:
            self.create_cancelled.set()

    async def delete_job(self, _name: str, _namespace: str) -> None:
        self.delete_calls += 1
        self.resource_exists = False
        if self.delete_calls == 1:
            self.first_delete_completed.set()


class _DelayedCreateKubectl:
    def __init__(self, deployer: _DelayedCreateAfterDeleteDeployer) -> None:
        self.deployer = deployer

    async def run(
        self,
        *_args: str,
        namespace: str | None = None,
        check: bool = True,
    ) -> SimpleNamespace:
        assert namespace == self.deployer.default_job_namespace
        assert not check
        return SimpleNamespace(returncode=0 if self.deployer.resource_exists else 1)


class _DelayedCreateDuringTailWaitDeployer:
    def __init__(self) -> None:
        self.default_job_namespace = "test-ns"
        self.created_name: str | None = None
        self.created_namespace: str | None = None
        self.create_cancelled = asyncio.Event()
        self.absence_observed = asyncio.Event()
        self.delayed_create_committed = asyncio.Event()
        self.delete_calls = 0
        self.resource_exists = False
        self.server_create_task: asyncio.Task[None] | None = None

    async def create_job(
        self,
        _config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.created_name = name
        self.created_namespace = namespace

        async def commit_after_absence() -> None:
            await self.absence_observed.wait()
            self.resource_exists = True
            self.delayed_create_committed.set()

        self.server_create_task = asyncio.create_task(commit_after_absence())
        try:
            await asyncio.Event().wait()
        finally:
            self.create_cancelled.set()

    async def delete_job(self, name: str, namespace: str) -> None:
        assert name == self.created_name
        assert namespace == self.created_namespace
        self.delete_calls += 1
        self.resource_exists = False


class _TailWaitKubectl:
    def __init__(self, deployer: _DelayedCreateDuringTailWaitDeployer) -> None:
        self.deployer = deployer

    async def run(
        self,
        *args: str,
        namespace: str | None = None,
        check: bool = True,
    ) -> SimpleNamespace:
        assert args == ("get", "aiperfjob", self.deployer.created_name)
        assert namespace == self.deployer.created_namespace
        assert not check
        exists = self.deployer.resource_exists
        if not exists:
            self.deployer.absence_observed.set()
        return SimpleNamespace(returncode=0 if exists else 1)


class _AbsentKubectl:
    async def run(
        self,
        *_args: str,
        namespace: str | None = None,
        check: bool = True,
    ) -> SimpleNamespace:
        assert namespace == "test-ns"
        assert not check
        return SimpleNamespace(returncode=1)


class _BlockingProcess:
    def __init__(self) -> None:
        self.returncode = -9
        self.communication_cancelled = asyncio.Event()
        self.killed = False
        self.waited = False

    async def communicate(self, input: bytes) -> tuple[bytes, bytes]:
        assert input
        try:
            await asyncio.Event().wait()
        finally:
            self.communication_cancelled.set()
        return b"", b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.waited = True
        return self.returncode


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


class _ObservedDeployer:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.default_job_namespace = "test-ns"

    async def create_job(
        self,
        _config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(job_name=name, namespace=namespace)

    async def get_job_status(self, _name: str, _namespace: str) -> SimpleNamespace:
        return SimpleNamespace(phase="Running", jobset_name="cleanup-jobset")

    async def delete_job(self, _name: str, _namespace: str) -> None:
        self.clock.now += 20


class _ControllerPodKubectl:
    async def get_pods(
        self,
        _namespace: str,
        label_selector: str | None = None,
    ) -> list[SimpleNamespace]:
        assert label_selector == ("jobset.sigs.k8s.io/jobset-name=cleanup-jobset")
        return [SimpleNamespace(name="cleanup-jobset-controller-0")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_case", "cleanup_module"),
    [
        param(
            test_operator.TestOperatorCleanup(),
            test_operator,
            id="operator",
        ),
        param(
            test_helm.TestHelmCleanup(),
            test_helm,
            id="helm",
        ),
    ],
)  # fmt: skip
async def test_cleanup_create_in_flight_deadline_expires_actionably(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_case: test_operator.TestOperatorCleanup | test_helm.TestHelmCleanup,
    cleanup_module: ModuleType,
) -> None:
    """A stuck create is cancelled by the cleanup budget with useful context."""
    monkeypatch.setattr(cleanup_module, "CLEANUP_ASSERTION_TIMEOUT", 0.05)
    monkeypatch.setattr(cleanup_module, "CLEANUP_DELETION_POLL_RESERVE", 0.02)
    monkeypatch.setattr(cleanup_module, "CLEANUP_FAILURE_TEARDOWN_TIMEOUT", 0.05)
    monkeypatch.setattr(
        cleanup_module,
        "CLEANUP_FAILURE_TEARDOWN_POLL_INTERVAL",
        0.01,
    )
    deployer = _BlockingCreateDeployer()

    with pytest.raises(
        AssertionError,
        match="Cleanup deadline expired while creating AIPerfJob",
    ):
        async with asyncio.timeout(0.5):
            await cleanup_case.test_deleting_job_removes_resources(
                deployer,
                AIPerfJobConfig(),
                _AbsentKubectl(),
            )

    assert deployer.cancelled.is_set()
    assert deployer.created_name is not None
    assert deployer.created_namespace == deployer.default_job_namespace
    assert deployer.deleted_job == (
        deployer.created_name,
        deployer.default_job_namespace,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_case", "cleanup_module"),
    [
        param(
            test_operator.TestOperatorCleanup(),
            test_operator,
            id="operator",
        ),
        param(
            test_helm.TestHelmCleanup(),
            test_helm,
            id="helm",
        ),
    ],
)  # fmt: skip
async def test_cleanup_delayed_create_after_first_delete_does_not_leak_cr(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_case: test_operator.TestOperatorCleanup | test_helm.TestHelmCleanup,
    cleanup_module: ModuleType,
) -> None:
    """A server-side create that wins after the first delete is removed."""
    monkeypatch.setattr(cleanup_module, "CLEANUP_ASSERTION_TIMEOUT", 0.05)
    monkeypatch.setattr(cleanup_module, "CLEANUP_DELETION_POLL_RESERVE", 0.02)
    monkeypatch.setattr(cleanup_module, "CLEANUP_FAILURE_TEARDOWN_TIMEOUT", 0.05)
    monkeypatch.setattr(
        cleanup_module,
        "CLEANUP_FAILURE_TEARDOWN_POLL_INTERVAL",
        0.01,
    )
    deployer = _DelayedCreateAfterDeleteDeployer()

    with pytest.raises(
        AssertionError,
        match="Cleanup deadline expired while creating AIPerfJob",
    ):
        async with asyncio.timeout(0.5):
            await cleanup_case.test_deleting_job_removes_resources(
                deployer,
                AIPerfJobConfig(),
                _DelayedCreateKubectl(deployer),
            )

    await asyncio.wait_for(deployer.delayed_create_committed.wait(), timeout=0.1)

    assert deployer.create_cancelled.is_set()
    assert deployer.delayed_create_committed.is_set()
    assert deployer.server_create_task is not None
    assert deployer.server_create_task.done()
    assert deployer.delete_calls >= 2
    assert not deployer.resource_exists, (
        "the delayed API-server create survived the single fallback delete"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_case", "cleanup_module"),
    [
        param(
            test_operator.TestOperatorCleanup(),
            test_operator,
            id="operator",
        ),
        param(
            test_helm.TestHelmCleanup(),
            test_helm,
            id="helm",
        ),
    ],
)  # fmt: skip
async def test_cleanup_create_committing_during_final_sleep_does_not_leak_cr(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_case: test_operator.TestOperatorCleanup | test_helm.TestHelmCleanup,
    cleanup_module: ModuleType,
) -> None:
    """A create committed after the final absent GET is deleted before return."""
    monkeypatch.setattr(cleanup_module, "CLEANUP_ASSERTION_TIMEOUT", 0.05)
    monkeypatch.setattr(cleanup_module, "CLEANUP_DELETION_POLL_RESERVE", 0.02)
    monkeypatch.setattr(cleanup_module, "CLEANUP_FAILURE_TEARDOWN_TIMEOUT", 0.05)
    monkeypatch.setattr(
        cleanup_module,
        "CLEANUP_FAILURE_TEARDOWN_POLL_INTERVAL",
        1,
    )
    real_loop = asyncio.get_running_loop()
    clock = _Clock(real_loop.time())

    async def advance_to_sleep_boundary(delay: float) -> None:
        await asyncio.sleep(0)
        clock.now += delay

    monkeypatch.setattr(
        deadline_helper,
        "asyncio",
        SimpleNamespace(
            get_running_loop=lambda: clock,
            sleep=advance_to_sleep_boundary,
            timeout_at=asyncio.timeout_at,
        ),
    )
    deployer = _DelayedCreateDuringTailWaitDeployer()

    with pytest.raises(
        AssertionError,
        match="Cleanup deadline expired while creating AIPerfJob",
    ):
        async with asyncio.timeout(0.5):
            await cleanup_case.test_deleting_job_removes_resources(
                deployer,
                AIPerfJobConfig(),
                _TailWaitKubectl(deployer),
            )

    await asyncio.wait_for(deployer.delayed_create_committed.wait(), timeout=0.1)

    assert deployer.create_cancelled.is_set()
    assert deployer.server_create_task is not None
    assert deployer.server_create_task.done()
    assert deployer.delete_calls >= 2
    assert not deployer.resource_exists, (
        "the delayed API-server create committed during the final teardown sleep"
    )


@pytest.mark.asyncio
async def test_kubectl_apply_cancelled_at_deadline_terminates_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling a stuck real apply path kills and reaps its child process."""
    process = _BlockingProcess()

    async def create_subprocess(*_args: str, **_kwargs: object) -> _BlockingProcess:
        return process

    monkeypatch.setattr(
        kubectl_helper.asyncio,
        "create_subprocess_exec",
        create_subprocess,
    )
    loop = asyncio.get_running_loop()

    with pytest.raises(
        AssertionError,
        match="Cleanup deadline expired while creating AIPerfJob",
    ):
        await await_before_deadline(
            loop.time() + 0.05,
            "creating AIPerfJob",
            lambda: KubectlClient().apply("apiVersion: v1"),
        )

    assert process.communication_cancelled.is_set()
    assert process.killed
    assert process.waited


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_case", "cleanup_module"),
    [
        param(
            test_operator.TestOperatorCleanup(),
            test_operator,
            id="operator",
        ),
        param(
            test_helm.TestHelmCleanup(),
            test_helm,
            id="helm",
        ),
    ],
)  # fmt: skip
async def test_cleanup_zero_propagation_polls_reports_initialized_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_case: test_operator.TestOperatorCleanup | test_helm.TestHelmCleanup,
    cleanup_module: ModuleType,
) -> None:
    """A clock jump after deletion retains the normal propagation assertion."""
    real_loop = asyncio.get_running_loop()
    clock = _Clock(real_loop.time())
    monkeypatch.setattr(cleanup_module, "CLEANUP_ASSERTION_TIMEOUT", 10)
    monkeypatch.setattr(cleanup_module, "CLEANUP_DELETION_POLL_RESERVE", 5)
    monkeypatch.setattr(
        cleanup_module,
        "asyncio",
        SimpleNamespace(
            get_running_loop=lambda: clock,
            sleep=asyncio.sleep,
        ),
    )

    with pytest.raises(
        pytest.fail.Exception,
        match=(
            r"AIPerfJob deletion did not remove the CR, JobSet, and child pods: "
            r"cr_returncode=None, jobsets=\[\], pods=\[\]"
        ),
    ):
        await cleanup_case.test_deleting_job_removes_resources(
            _ObservedDeployer(clock),
            AIPerfJobConfig(),
            _ControllerPodKubectl(),
        )
