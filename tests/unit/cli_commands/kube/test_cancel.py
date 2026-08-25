# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube cancel` drives the CRD's spec.cancel field.

The operator has always had a cancel handler wired to spec.cancel, but no CLI
verb set it -- users had to hand-write a kubectl patch. The verb existed on
kube4 and was dropped at the undocumented kube4->kube5 squash.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.cli_commands.kube.cancel import cancel
from aiperf.kubernetes.console import LastBenchmarkInfo


@asynccontextmanager
async def _fake_client(**_: Any):
    yield MagicMock()


def _custom(get_side_effect: Any) -> MagicMock:
    return MagicMock(
        get_namespaced_custom_object=AsyncMock(side_effect=get_side_effect),
        patch_namespaced_custom_object=AsyncMock(),
    )


async def _run(
    custom: MagicMock,
    *,
    job_id: str | None = "job-1",
    kind: str | None = None,
    last: LastBenchmarkInfo | None = None,
) -> None:
    with (
        patch("aiperf.kubernetes.client.k8s_client", _fake_client),
        patch(
            "kubernetes_asyncio.client.CustomObjectsApi",
            return_value=custom,
        ),
        patch(
            "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
            return_value=("job-1", "bench"),
        ),
        patch("aiperf.kubernetes.console.get_last_benchmark", return_value=last),
    ):
        await cancel(job_id, kind=kind)


class TestKubeCancel:
    @pytest.mark.asyncio
    async def test_patches_spec_cancel_on_a_running_job(self) -> None:
        custom = _custom([{"status": {"phase": "Running"}}, ApiException(status=404)])
        await _run(custom)
        custom.patch_namespaced_custom_object.assert_awaited_once()
        kwargs = custom.patch_namespaced_custom_object.await_args.kwargs
        assert kwargs["body"] == {"spec": {"cancel": True}}
        assert kwargs["plural"] == "aiperfjobs"
        assert kwargs["name"] == "job-1"
        assert kwargs["namespace"] == "bench"

    @pytest.mark.asyncio
    async def test_success_hint_uses_live_status_command(self) -> None:
        custom = _custom([{"status": {"phase": "Running"}}, ApiException(status=404)])
        with patch("aiperf.kubernetes.console.print_info") as print_info:
            await _run(custom)

        print_info.assert_called_once_with(
            "Watch it wind down with: aiperf kube list job-1 --watch"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phase", ["Completed", "Failed", "Cancelled"])
    async def test_terminal_job_is_not_patched(self, phase: str) -> None:
        custom = _custom([{"status": {"phase": phase}}, ApiException(status=404)])
        await _run(custom)
        custom.patch_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_the_sweep_kind(self) -> None:
        """A name that is not an AIPerfJob may still be an AIPerfSweep."""
        custom = _custom([ApiException(status=404), {"status": {"phase": "Running"}}])
        await _run(custom)
        kwargs = custom.patch_namespaced_custom_object.await_args.kwargs
        assert kwargs["plural"] == "aiperfsweeps"

    @pytest.mark.asyncio
    async def test_unknown_name_patches_nothing(self) -> None:
        custom = _custom([ApiException(status=404), ApiException(status=404)])
        await _run(custom)
        custom.patch_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_404_errors_propagate(self) -> None:
        custom = _custom([ApiException(status=403)])
        with pytest.raises(SystemExit):
            await _run(custom)
        custom.patch_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_named_job_and_sweep_require_explicit_kind(self) -> None:
        custom = _custom(
            [
                {"status": {"phase": "Running"}},
                {"status": {"phase": "Running"}},
            ]
        )

        await _run(custom)

        custom.patch_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_sweep_kind_cancels_only_the_sweep(self) -> None:
        custom = _custom([{"status": {"phase": "Running"}}])

        await _run(custom, kind="sweep")

        kwargs = custom.patch_namespaced_custom_object.await_args.kwargs
        assert kwargs["plural"] == "aiperfsweeps"
        assert custom.get_namespaced_custom_object.await_count == 1

    @pytest.mark.asyncio
    async def test_last_benchmark_kind_selects_same_named_sweep(self) -> None:
        custom = _custom([{"status": {"phase": "Running"}}])

        await _run(
            custom,
            job_id=None,
            last=LastBenchmarkInfo(
                job_id="job-1",
                namespace="bench",
                kind="AIPerfSweep",
            ),
        )

        kwargs = custom.patch_namespaced_custom_object.await_args.kwargs
        assert kwargs["plural"] == "aiperfsweeps"
        assert custom.get_namespaced_custom_object.await_count == 1
