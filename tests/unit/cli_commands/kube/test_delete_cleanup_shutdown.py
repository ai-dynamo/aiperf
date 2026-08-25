# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The destructive kube verbs dropped at the kube4->kube5 squash.

`delete`, `cleanup` and `shutdown` existed on kube4 and vanished with no
commit recording the decision, leaving users to hand-write kubectl for every
teardown. Re-implemented against the operator-first CR model: deleting the CR
garbage-collects the JobSet and pods through ownerReferences.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.cli_commands.kube.cleanup import _is_terminal, cleanup
from aiperf.cli_commands.kube.delete import delete


@asynccontextmanager
async def _fake_client(**_: Any):
    yield MagicMock()


def _cr(name: str, phase: str) -> dict:
    return {
        "metadata": {
            "name": name,
            "uid": f"{name}-uid",
            "resourceVersion": f"{name}-rv",
        },
        "status": {"phase": phase},
    }


def _custom(*, get: Any = None, listing: dict | None = None) -> MagicMock:
    """Fake CustomObjectsApi. The listing belongs to aiperfjobs only.

    list_aiperf_crs queries both plurals, so a mock that answers identically
    for each would double every benchmark.
    """

    async def _list(*, plural: str, **_: Any) -> dict:
        if plural == "aiperfjobs":
            return listing or {"items": []}
        return {"items": []}

    return MagicMock(
        get_namespaced_custom_object=AsyncMock(side_effect=get),
        list_namespaced_custom_object=AsyncMock(side_effect=_list),
        delete_namespaced_custom_object=AsyncMock(),
        patch_namespaced_custom_object=AsyncMock(),
    )


@contextmanager
def _patched(
    custom: MagicMock,
    *,
    job_id: str = "job-1",
    namespace: str = "bench",
    core: MagicMock | None = None,
):
    """Patch the cluster surface all three verbs share."""
    core = core or MagicMock()
    with (
        patch("aiperf.kubernetes.client.k8s_client", _fake_client),
        patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
        patch("kubernetes_asyncio.client.CoreV1Api", return_value=core),
        patch(
            "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
            return_value=(job_id, namespace),
        ),
        patch("aiperf.kubernetes.console.clear_last_benchmark_if_matches"),
    ):
        yield


class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes_the_custom_resource(self) -> None:
        custom = _custom(get=[_cr("job-1", "Completed"), ApiException(status=404)])
        with _patched(custom):
            await delete("job-1", force=True)
        custom.delete_namespaced_custom_object.assert_awaited_once()
        kwargs = custom.delete_namespaced_custom_object.await_args.kwargs
        assert kwargs["plural"] == "aiperfjobs"
        assert kwargs["name"] == "job-1"

    @pytest.mark.asyncio
    async def test_custom_resource_delete_is_uid_preconditioned(self) -> None:
        cr = _cr("job-1", "Completed")
        cr["metadata"].update(
            uid="workload-uid",
            resourceVersion="workload-rv",
        )
        custom = _custom(get=[cr, ApiException(status=404)])

        with _patched(custom):
            await delete("job-1", force=True)

        body = custom.delete_namespaced_custom_object.await_args.kwargs["body"]
        assert body.preconditions.uid == "workload-uid"
        assert body.preconditions.resource_version == "workload-rv"

    @pytest.mark.asyncio
    async def test_unknown_name_deletes_nothing(self) -> None:
        custom = _custom(get=[ApiException(status=404), ApiException(status=404)])
        with _patched(custom):
            await delete("job-1", force=True)
        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refuses_without_force_when_not_a_tty(self) -> None:
        """Non-interactive callers cannot answer a prompt; deleting anyway is wrong."""
        custom = _custom(get=[_cr("job-1", "Completed"), ApiException(status=404)])
        with _patched(custom), patch("sys.stdin.isatty", return_value=False):
            await delete("job-1")
        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_named_job_and_sweep_require_explicit_kind(self) -> None:
        custom = _custom(
            get=[
                _cr("collision", "Running"),
                _cr("collision", "Running"),
            ]
        )

        with _patched(custom, job_id="collision"):
            await delete("collision", force=True)

        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_sweep_kind_deletes_only_the_sweep(self) -> None:
        custom = _custom(get=[_cr("collision", "Running")])

        with _patched(custom, job_id="collision"):
            await delete("collision", force=True, kind="sweep")

        kwargs = custom.delete_namespaced_custom_object.await_args.kwargs
        assert kwargs["plural"] == "aiperfsweeps"
        assert custom.get_namespaced_custom_object.await_count == 1

    @pytest.mark.asyncio
    async def test_wrong_explicit_kind_deletes_nothing(self) -> None:
        custom = _custom(get=[ApiException(status=404)])

        with _patched(custom):
            await delete("job-1", force=True, kind="sweep")

        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlabelled_matching_namespace_is_not_deleted(self) -> None:
        custom = _custom(get=[_cr("job-1", "Completed"), ApiException(status=404)])
        core = MagicMock()
        core.read_namespace = AsyncMock(
            return_value=MagicMock(metadata=MagicMock(uid="namespace-uid", labels={}))
        )
        core.delete_namespace = AsyncMock()

        with _patched(
            custom,
            namespace="aiperf-job-1",
            core=core,
        ):
            await delete("job-1", force=True, delete_namespace=True)

        custom.delete_namespaced_custom_object.assert_awaited_once()
        core.read_namespace.assert_awaited_once_with(name="aiperf-job-1")
        core.delete_namespace.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shared_helm_namespace_marker_is_not_job_ownership(self) -> None:
        custom = _custom(get=[_cr("benchmarks", "Completed"), ApiException(status=404)])
        core = MagicMock()
        core.read_namespace = AsyncMock(
            return_value=MagicMock(
                metadata=MagicMock(
                    uid="shared-namespace-uid",
                    resource_version="shared-namespace-rv",
                    labels={
                        "app": "aiperf",
                        "aiperf.nvidia.com/auto-generated": "true",
                    },
                )
            )
        )
        core.delete_namespace = AsyncMock()

        with _patched(
            custom,
            job_id="benchmarks",
            namespace="aiperf-benchmarks",
            core=core,
        ):
            await delete("benchmarks", force=True, delete_namespace=True)

        custom.delete_namespaced_custom_object.assert_awaited_once()
        core.read_namespace.assert_awaited_once_with(name="aiperf-benchmarks")
        core.delete_namespace.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_namespace_owned_by_another_job_is_not_deleted(self) -> None:
        custom = _custom(get=[_cr("job-1", "Completed"), ApiException(status=404)])
        core = MagicMock()
        core.read_namespace = AsyncMock(
            return_value=MagicMock(
                metadata=MagicMock(
                    uid="namespace-uid",
                    resource_version="namespace-rv",
                    labels={
                        "aiperf.nvidia.com/auto-generated": "true",
                        "aiperf.nvidia.com/job-id": "another-job",
                    },
                )
            )
        )
        core.delete_namespace = AsyncMock()

        with _patched(
            custom,
            namespace="aiperf-job-1",
            core=core,
        ):
            await delete("job-1", force=True, delete_namespace=True)

        custom.delete_namespaced_custom_object.assert_awaited_once()
        core.delete_namespace.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_labelled_matching_namespace_delete_is_uid_preconditioned(
        self,
    ) -> None:
        custom = _custom(get=[_cr("job-1", "Completed"), ApiException(status=404)])
        core = MagicMock()
        core.read_namespace = AsyncMock(
            return_value=MagicMock(
                metadata=MagicMock(
                    uid="namespace-uid",
                    resource_version="namespace-rv",
                    labels={
                        "aiperf.nvidia.com/auto-generated": "true",
                        "aiperf.nvidia.com/job-id": "job-1",
                    },
                )
            )
        )
        core.delete_namespace = AsyncMock()

        with _patched(
            custom,
            namespace="aiperf-job-1",
            core=core,
        ):
            await delete("job-1", force=True, delete_namespace=True)

        kwargs = core.delete_namespace.await_args.kwargs
        assert kwargs["name"] == "aiperf-job-1"
        assert kwargs["body"].preconditions.uid == "namespace-uid"
        assert kwargs["body"].preconditions.resource_version == "namespace-rv"


class TestCleanup:
    @pytest.mark.parametrize("phase", ["Succeeded", "PartiallyFailed"])
    def test_sweep_terminal_phases_are_eligible_for_cleanup(self, phase: str) -> None:
        assert _is_terminal("aiperfsweeps", _cr("sweep", phase))

    def _listing(self) -> dict:
        return {
            "items": [
                _cr("done-1", "Completed"),
                _cr("dead-1", "Failed"),
                _cr("live-1", "Running"),
            ]
        }

    @pytest.mark.asyncio
    async def test_leaves_running_benchmarks_alone(self) -> None:
        custom = _custom(listing=self._listing())
        with _patched(custom):
            await cleanup(force=True)
        removed = sorted(
            c.kwargs["name"]
            for c in custom.delete_namespaced_custom_object.await_args_list
        )
        # Terminal benchmarks go; a run in progress must survive a cleanup.
        assert removed == ["dead-1", "done-1"]

    @pytest.mark.asyncio
    async def test_all_cancels_before_deleting_a_running_benchmark(self) -> None:
        custom = _custom(listing={"items": [_cr("live-1", "Running")]})
        with _patched(custom):
            await cleanup(force=True, all_benchmarks=True)
        custom.patch_namespaced_custom_object.assert_awaited()
        assert custom.patch_namespaced_custom_object.await_args.kwargs["body"] == {
            "spec": {"cancel": True}
        }
        custom.delete_namespaced_custom_object.assert_awaited()

    @pytest.mark.asyncio
    async def test_dry_run_deletes_nothing(self) -> None:
        custom = _custom(listing=self._listing())
        with _patched(custom):
            await cleanup(dry_run=True, force=True)
        custom.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_namespace_is_a_noop(self) -> None:
        custom = _custom(listing={"items": []})
        with _patched(custom):
            await cleanup(force=True)
        custom.delete_namespaced_custom_object.assert_not_awaited()


class TestConfirmAction:
    @pytest.mark.parametrize(
        "answer,expected",
        [
            param("y", True, id="y"),
            param("yes", True, id="yes"),
            param("n", False, id="n"),
            param("", False, id="empty-defaults-to-no"),
        ],
    )  # fmt: skip
    def test_answers(self, answer: str, expected: bool) -> None:
        from aiperf.cli_commands.kube._kube_delete import confirm_action

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value=answer),
        ):
            assert confirm_action("Delete?") is expected

    def test_non_tty_declines(self) -> None:
        from aiperf.cli_commands.kube._kube_delete import confirm_action

        with patch("sys.stdin.isatty", return_value=False):
            assert confirm_action("Delete?") is False
