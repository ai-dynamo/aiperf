# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf kube list` cyclopts subcommand.

Focus is on:
- module exposes `app`; subcommand is registered in `aiperf kube`
- `list_jobs` callable signature accepts the documented flags
- the command opens k8s_client, dispatches to find_aiperf_job (with --job-id)
  or list_aiperf_jobs/list_jobsets (without), and prints a table
- mutually-exclusive status filters are rejected
- empty results print the "No AIPerf jobs found" message
- `_resolve_status_filter` enforces "at most one filter" semantics
- API failures surface through `cli_utils.exit_on_error` instead of bubbling
"""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.config.kube import KubeManageOptions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job_info(
    name: str = "job-a",
    namespace: str = "ns-1",
    phase: str = "Running",
    job_id: str = "abc123",
    created: str = "2026-05-01T00:00:00Z",
) -> Any:
    """Build a minimal AIPerfJobInfo for table printing."""
    from aiperf.kubernetes.models import AIPerfJobInfo

    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase=phase,
        job_id=job_id,
        jobset_name=f"{name}-jobset",
        created=created,
    )


def _make_jobset_info(
    name: str = "js-a",
    namespace: str = "ns-1",
    status: str = "Running",
    job_id_label: str | None = None,
    created: str = "2026-05-01T00:00:00Z",
    model: str | None = None,
    endpoint: str | None = None,
) -> Any:
    """Build a JobSetInfo from a synthesized raw dict."""
    from aiperf.kubernetes.constants import AIPerfLabels
    from aiperf.kubernetes.models import JobSetInfo

    labels: dict[str, str] = {}
    if job_id_label is not None:
        labels[AIPerfLabels.JOB_ID] = job_id_label
    raw = {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
            "annotations": {},
            "creationTimestamp": created,
        },
        "spec": {},
        "status": {},
    }
    info = JobSetInfo(
        name=name,
        namespace=namespace,
        jobset=raw,
        status=status,
        model=model,
        endpoint=endpoint,
    )
    return info


@asynccontextmanager
async def _fake_k8s_client(**_kw):
    """An async context manager yielding a sentinel api object."""
    yield MagicMock(name="ApiClient")


# ---------------------------------------------------------------------------
# Module wiring
# ---------------------------------------------------------------------------


def test_list_module_importable() -> None:
    """The list_ module must be importable and expose an `app` attribute."""
    from aiperf.cli_commands.kube import list_

    assert hasattr(list_, "app"), "list_.app (cyclopts App) must be defined"


def test_list_registered_in_kube_app() -> None:
    """The `list` subcommand must be wired into `aiperf kube`."""
    from aiperf.cli_commands.kube._app import app

    assert "list" in set(app)


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


class TestListCallableSignature:
    """`list_jobs` must accept the documented CLI flags as kwargs."""

    @pytest.mark.parametrize(
        "param_name",
        [
            "job_id",
            "all_namespaces",
            "running",
            "completed",
            "failed",
            "wide",
            "watch",
            "interval",
            "manage_options",
        ],
    )  # fmt: skip
    def test_signature_has_param(self, param_name: str) -> None:
        from aiperf.cli_commands.kube.list_ import list_jobs

        sig = inspect.signature(list_jobs)
        assert param_name in sig.parameters

    def test_signature_defaults(self) -> None:
        from aiperf.cli_commands.kube.list_ import list_jobs

        sig = inspect.signature(list_jobs)
        assert sig.parameters["job_id"].default is None
        assert sig.parameters["all_namespaces"].default is True
        assert sig.parameters["running"].default is False
        assert sig.parameters["completed"].default is False
        assert sig.parameters["failed"].default is False
        assert sig.parameters["wide"].default is False
        assert sig.parameters["watch"].default is False
        assert sig.parameters["interval"].default == 5
        assert sig.parameters["manage_options"].default is None


# ---------------------------------------------------------------------------
# _resolve_status_filter (pure helper)
# ---------------------------------------------------------------------------


class TestResolveStatusFilter:
    """The pure helper that maps three bool flags -> single phase string."""

    @pytest.mark.parametrize(
        "running,completed,failed,expected",
        [
            (False, False, False, None),
            param(True, False, False, "Running", id="running-only"),
            param(False, True, False, "Completed", id="completed-only"),
            param(False, False, True, "Failed", id="failed-only"),
        ],
    )  # fmt: skip
    def test_single_or_no_filter(
        self,
        running: bool,
        completed: bool,
        failed: bool,
        expected: str | None,
    ) -> None:
        from aiperf.cli_commands.kube.list_ import _resolve_status_filter

        assert (
            _resolve_status_filter(running=running, completed=completed, failed=failed)
            == expected
        )

    @pytest.mark.parametrize(
        "running,completed,failed",
        [
            param(True, True, False, id="running+completed"),
            param(True, False, True, id="running+failed"),
            param(False, True, True, id="completed+failed"),
            param(True, True, True, id="all-three"),
        ],
    )  # fmt: skip
    def test_multiple_filters_exit_nonzero(
        self, running: bool, completed: bool, failed: bool
    ) -> None:
        from aiperf.cli_commands.kube.list_ import _resolve_status_filter

        with (
            patch("aiperf.kubernetes.console.print_error") as mock_err,
            pytest.raises(SystemExit) as exc_info,
        ):
            _resolve_status_filter(running=running, completed=completed, failed=failed)

        assert exc_info.value.code == 1
        mock_err.assert_called_once()


# ---------------------------------------------------------------------------
# list_jobs end-to-end (mocked k8s client + fetchers)
# ---------------------------------------------------------------------------


class TestListJobsCommand:
    """End-to-end: list_jobs opens k8s_client and dispatches to fetchers."""

    @pytest.mark.asyncio
    async def test_no_jobs_prints_empty_message(self) -> None:
        """When list_aiperf_jobs and list_jobsets both return empty -> no-jobs message."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_aiperfjob_table") as mock_table,
        ):
            await list_jobs()

        mock_info.assert_called_once()
        assert "No AIPerf jobs found" in mock_info.call_args.args[0]
        mock_table.assert_not_called()

    @pytest.mark.asyncio
    async def test_lists_aiperf_jobs_and_renders_table(self) -> None:
        """Non-empty list_aiperf_jobs -> table is rendered, jobsets fallback skipped."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        infos = [_make_job_info(name="a"), _make_job_info(name="b")]
        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=infos),
            ) as mock_list,
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ) as mock_list_js,
            patch("aiperf.kubernetes.console.print_aiperfjob_table") as mock_table,
        ):
            await list_jobs()

        mock_list.assert_awaited_once()
        # JobSet fallback should NOT run when list_aiperf_jobs returns rows
        mock_list_js.assert_not_awaited()
        mock_table.assert_called_once()
        passed_jobs, kwargs = mock_table.call_args.args, mock_table.call_args.kwargs
        assert passed_jobs[0] == infos
        assert kwargs.get("wide") is False

    @pytest.mark.asyncio
    async def test_wide_flag_propagates_to_table(self) -> None:
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[_make_job_info()]),
            ),
            patch("aiperf.kubernetes.console.print_aiperfjob_table") as mock_table,
        ):
            await list_jobs(wide=True)

        assert mock_table.call_args.kwargs.get("wide") is True

    @pytest.mark.asyncio
    async def test_status_filter_forwarded_to_list_aiperf_jobs(self) -> None:
        """--running maps to status_filter='Running' on list_aiperf_jobs."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            await list_jobs(running=True)

        assert mock_list.await_args.kwargs.get("status_filter") == "Running"

    @pytest.mark.asyncio
    async def test_namespace_overrides_all_namespaces(self) -> None:
        """When --namespace is set, search_all becomes False even with -A."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            await list_jobs(
                manage_options=KubeManageOptions(namespace="my-ns"),
                all_namespaces=True,
            )

        kwargs = mock_list.await_args.kwargs
        assert kwargs.get("namespace") == "my-ns"
        assert kwargs.get("all_namespaces") is False

    @pytest.mark.asyncio
    async def test_default_search_all_is_true(self) -> None:
        """No namespace + all_namespaces=True -> all_namespaces forwarded as True."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            await list_jobs()

        assert mock_list.await_args.kwargs.get("all_namespaces") is True


class TestListJobsByJobId:
    """When job_id is given, find_aiperf_job is the only call path."""

    @pytest.mark.asyncio
    async def test_found_job_renders_table(self) -> None:
        from aiperf.cli_commands.kube.list_ import list_jobs

        info = _make_job_info(name="target-job")
        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=info),
            ) as mock_find,
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
            patch("aiperf.kubernetes.console.print_aiperfjob_table") as mock_table,
        ):
            await list_jobs(job_id="target-job")

        mock_find.assert_awaited_once()
        # find_aiperf_job(api, name, namespace) — name is positional, namespace too
        assert mock_find.await_args.args[1] == "target-job"
        mock_list.assert_not_awaited()
        mock_table.assert_called_once()
        assert mock_table.call_args.args[0] == [info]

    @pytest.mark.asyncio
    async def test_not_found_prints_empty_message(self) -> None:
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch("aiperf.kubernetes.console.print_info") as mock_info,
            patch("aiperf.kubernetes.console.print_aiperfjob_table") as mock_table,
        ):
            await list_jobs(job_id="nope")

        mock_info.assert_called_once()
        assert "No AIPerf jobs found" in mock_info.call_args.args[0]
        mock_table.assert_not_called()


class TestListJobsJobsetFallback:
    """When list_aiperf_jobs returns empty, list_jobsets must be tried."""

    @pytest.mark.asyncio
    async def test_jobsets_promoted_to_aiperf_job_info(self) -> None:
        """Empty AIPerfJob list + non-empty JobSet list -> table rendered from jobsets."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        js = _make_jobset_info(
            name="legacy-js",
            namespace="ns-1",
            status="Running",
            job_id_label="legacy-jid",
            model="m",
            endpoint="http://svc:8000",
        )

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[js]),
            ) as mock_list_js,
            patch("aiperf.kubernetes.console.print_aiperfjob_table") as mock_table,
        ):
            await list_jobs()

        mock_list_js.assert_awaited_once()
        mock_table.assert_called_once()
        promoted = mock_table.call_args.args[0]
        assert len(promoted) == 1
        assert promoted[0].name == "legacy-js"
        assert promoted[0].phase == "Running"
        assert promoted[0].model == "m"
        assert promoted[0].endpoint == "http://svc:8000"


class TestListErrorWrapping:
    """`exit_on_error` must catch underlying API exceptions and exit cleanly."""

    @pytest.mark.asyncio
    async def test_api_error_becomes_system_exit(self) -> None:
        """A RuntimeError from list_aiperf_jobs -> SystemExit, not a raw traceback."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(side_effect=RuntimeError("forbidden")),
            ),
            patch("aiperf.cli_utils.console"),
            pytest.raises(SystemExit) as exc_info,
        ):
            await list_jobs()

        assert exc_info.value.code == 1


class TestListWatchMode:
    """--watch should iterate and respect cancellation."""

    @pytest.mark.asyncio
    async def test_watch_loop_terminates_on_cancelled_error(self) -> None:
        """asyncio.sleep raising CancelledError exits cleanly without re-raising."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        sleep_mock = AsyncMock(side_effect=__import__("asyncio").CancelledError())

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[_make_job_info()]),
            ),
            patch("aiperf.kubernetes.console.print_aiperfjob_table") as mock_table,
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("asyncio.sleep", sleep_mock),
        ):
            # Should not raise even though sleep raises CancelledError
            await list_jobs(watch=True, interval=1)

        # Table rendered at least once before cancellation
        mock_table.assert_called()
        # Watch path clears the screen before render
        mock_console.clear.assert_called()


# ---------------------------------------------------------------------------
# _namespace_owners — 403 per-namespace fallback (6686417b68)
# ---------------------------------------------------------------------------


class TestNamespaceOwners:
    """_namespace_owners falls back to per-namespace reads on 403."""

    @pytest.mark.asyncio
    async def test_403_on_cluster_list_falls_back_to_per_namespace_reads(self) -> None:
        """A 403 on list_lease_for_all_namespaces must trigger per-namespace reads.

        Before 6686417b68, the 403 branch was absent: any ApiException from the
        cluster-wide list returned ``"?"`` for every namespace.  A namespace-scoped
        user therefore always saw ``"?"`` in the OWNER column, even when they had
        read access to the individual Lease.

        After the fix, a 403 causes _namespace_owners to call
        read_namespaced_lease for each namespace and return the real holder.
        """
        from kubernetes_asyncio.client.exceptions import ApiException

        from aiperf.cli_commands.kube.list_ import _namespace_owners

        job1 = _make_job_info(namespace="ns-a")
        job2 = _make_job_info(namespace="ns-b")

        # Cluster-wide list raises 403.
        # Per-namespace read for ns-a returns a live Lease; ns-b has no Lease (404).
        lease_mock = MagicMock()
        lease_mock.spec = MagicMock()
        lease_mock.spec.holder_identity = "scoped-op-1"
        lease_mock.spec.renew_time = None
        lease_mock.spec.acquire_time = None  # both None → lease treated as live
        lease_mock.spec.lease_duration_seconds = 30
        lease_mock.metadata = MagicMock()
        lease_mock.metadata.creation_timestamp = None

        async def _read_namespaced_lease(name, ns):
            if ns == "ns-a":
                return lease_mock
            raise ApiException(status=404)

        coord_api = MagicMock()
        coord_api.list_lease_for_all_namespaces = AsyncMock(
            side_effect=ApiException(status=403, reason="Forbidden")
        )
        coord_api.read_namespaced_lease = AsyncMock(side_effect=_read_namespaced_lease)

        with patch(
            "kubernetes_asyncio.client.CoordinationV1Api", return_value=coord_api
        ):
            result = await _namespace_owners(object(), [job1, job2])

        # ns-a has a live claim; ns-b has no Lease (404 → "-").
        assert result["ns-a"] != "?", "403 fallback must resolve ns-a to a real owner"
        assert result["ns-b"] == "-", "404 on read must return '-' (no claim)"

    @pytest.mark.asyncio
    async def test_non_403_api_exception_returns_question_marks(self) -> None:
        """A non-403 ApiException (e.g. 500) returns '?' for all namespaces."""
        from kubernetes_asyncio.client.exceptions import ApiException

        from aiperf.cli_commands.kube.list_ import _namespace_owners

        coord_api = MagicMock()
        coord_api.list_lease_for_all_namespaces = AsyncMock(
            side_effect=ApiException(status=500, reason="Internal Server Error")
        )

        with patch(
            "kubernetes_asyncio.client.CoordinationV1Api", return_value=coord_api
        ):
            result = await _namespace_owners(
                object(), [_make_job_info(namespace="ns-x")]
            )

        assert result == {"ns-x": "?"}


class TestScopedListNamespaceResolution:
    """A namespace-scoped list must resolve a namespace, never assume 'default'."""

    @pytest.mark.asyncio
    async def test_no_namespace_with_scoped_list_resolves_via_helper(self) -> None:
        """--no-all-namespaces without --namespace uses resolve_benchmark_namespace."""
        from aiperf.cli_commands.kube.list_ import list_jobs

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_k8s_client),
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_benchmark_namespace",
                return_value="ctx-ns",
            ) as mock_resolve,
            patch(
                "aiperf.kubernetes.client.list_aiperf_jobs",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.kubernetes.console.print_info"),
        ):
            await list_jobs(all_namespaces=False)

        mock_resolve.assert_called_once()
        assert mock_list.await_args.kwargs.get("namespace") == "ctx-ns"
        assert mock_list.await_args.kwargs.get("all_namespaces") is False
