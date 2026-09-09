# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolution of an omitted ``workersPerPod`` in the monitor handler.

``spec.benchmark.runtime.workersPerPod`` is optional and the CRD declares no
default; ``spec_converter`` normalizes it in memory only and never patches the
value back onto the CR. The monitor must therefore reconstruct the value the
deployment actually used rather than assuming 1.
"""

from typing import Any

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.operator.handlers.monitor import _resolve_workers_per_pod


def _spec(workers_per_pod: int | None) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    if workers_per_pod is not None:
        runtime["workersPerPod"] = workers_per_pod
    return {"benchmark": {"runtime": runtime}}


class TestResolveWorkersPerPod:
    """`_resolve_workers_per_pod` mirrors spec_converter's create-time rule."""

    @pytest.mark.parametrize(
        "total_workers",
        [
            param(0, id="total_unknown"),
            param(10, id="one_pod"),
            param(100, id="ten_pods"),
        ],
    )  # fmt: skip
    def test_resolve_workers_per_pod_omitted_field_returns_deployed_default(
        self, total_workers: int
    ) -> None:
        assert (
            _resolve_workers_per_pod(_spec(None), total_workers)
            == Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )

    def test_resolve_workers_per_pod_omitted_field_never_returns_one(self) -> None:
        """Regression: the old fallback of 1 under-reported ready workers 10x."""
        assert _resolve_workers_per_pod({}, 0) != 1

    @pytest.mark.parametrize(
        "configured,total_workers,expected",
        [
            param(2, 40, 2, id="explicit_divisible"),
            param(5, 5, 5, id="explicit_single_pod"),
            param(1, 16, 1, id="explicit_one"),
        ],
    )  # fmt: skip
    def test_resolve_workers_per_pod_explicit_field_is_honored(
        self, configured: int, total_workers: int, expected: int
    ) -> None:
        assert _resolve_workers_per_pod(_spec(configured), total_workers) == expected

    @pytest.mark.parametrize(
        "configured,total_workers",
        [
            param(None, 45, id="omitted_non_divisible"),
            param(3, 40, id="explicit_non_divisible"),
        ],
    )  # fmt: skip
    def test_resolve_workers_per_pod_non_divisible_total_collapses_to_one_pod(
        self, configured: int | None, total_workers: int
    ) -> None:
        """A JobSet cannot express a partial final pod; spec_converter collapses."""
        assert (
            _resolve_workers_per_pod(_spec(configured), total_workers) == total_workers
        )

    @pytest.mark.parametrize(
        "spec",
        [
            param({}, id="empty_spec"),
            param({"benchmark": {}}, id="no_runtime"),
            param({"benchmark": {"runtime": {"workersPerPod": None}}}, id="null_value"),
            param({"benchmark": {"runtime": {"workersPerPod": 0}}}, id="zero_value"),
        ],
    )  # fmt: skip
    def test_resolve_workers_per_pod_missing_or_falsy_returns_default(
        self, spec: dict[str, Any]
    ) -> None:
        assert (
            _resolve_workers_per_pod(spec, 0)
            == Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )
