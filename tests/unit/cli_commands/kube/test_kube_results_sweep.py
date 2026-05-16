# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube results <sweep-name>` (R2 wiring).

The job path is exercised in `test_kube.py`; this file targets the new
ResolvedSweep branch in `_run_results` and the new sweep fan-out helper
`retrieve_sweep_results_from_operator` in `aiperf.kubernetes.results`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.cli_commands.kube.results import (
    _default_sweep_output_dir,
    _run_results,
)
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.cli_helpers import ResolvedSweep


def _make_sweep_info(**overrides: Any) -> Any:
    """Build an AIPerfSweepInfo with sensible defaults (overridable)."""
    from aiperf.kubernetes.models import AIPerfSweepInfo

    defaults: dict[str, Any] = {
        "name": "my-sweep",
        "namespace": "bench-ns",
        "phase": "Running",
        "run_epoch": 1700000000,
        "total_variations": 4,
        "max_total_runs": 12,
        "completed_runs": 2,
        "failed_runs": 0,
        "created": "2026-01-15T10:30:00Z",
    }
    defaults.update(overrides)
    return AIPerfSweepInfo(**defaults)


def _make_resolved_sweep(name: str = "my-sweep", ns: str = "bench-ns") -> ResolvedSweep:
    api = MagicMock()
    api.close = AsyncMock()
    return ResolvedSweep(
        name=name, sweep_info=_make_sweep_info(name=name, namespace=ns), api=api
    )


# ============================================================
# CLI wiring: _run_results sweep branch
# ============================================================


class TestRunResultsSweepBranch:
    """Tests for the ResolvedSweep branch of `_run_results`."""

    async def test_results_invokes_sweep_path_when_target_is_resolvedsweep(
        self, tmp_path: Path
    ) -> None:
        """A ResolvedSweep return triggers the sweep fan-out, not the job path."""
        resolved = _make_resolved_sweep()
        opts = KubeManageOptions(namespace="bench-ns")
        out = tmp_path / "out"

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_target",
                new=AsyncMock(return_value=resolved),
            ),
            patch(
                "aiperf.cli_commands.kube.results._resolve_op_ns",
                new=AsyncMock(return_value="aiperf-system"),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_sweep_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_sweep_get,
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_operator",
                new=AsyncMock(),
            ) as mock_job_get,
            patch(
                "aiperf.kubernetes.results.retrieve_all_artifacts",
                new=AsyncMock(),
            ) as mock_all,
        ):
            await _run_results(
                job_id="my-sweep",
                manage_options=opts,
                output=out,
                from_pods=False,
                all_artifacts=True,
                shutdown=False,
                port=0,
                operator_namespace=None,
                run=None,
            )

        mock_sweep_get.assert_awaited_once()
        kwargs = mock_sweep_get.await_args.kwargs
        assert kwargs["operator_namespace"] == "aiperf-system"
        assert kwargs["local_port"] == 0
        # Positional args: (sweep_name, namespace, output_dir, api)
        args = mock_sweep_get.await_args.args
        assert args[0] == "my-sweep"
        assert args[1] == "bench-ns"
        assert args[2] == out
        mock_job_get.assert_not_awaited()
        mock_all.assert_not_awaited()
        resolved.api.close.assert_awaited_once()

    async def test_results_rejects_from_pods_for_sweep(self, tmp_path: Path) -> None:
        """--from-pods on a sweep prints an error and skips retrieval."""
        resolved = _make_resolved_sweep()
        opts = KubeManageOptions(namespace="bench-ns")

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_target",
                new=AsyncMock(return_value=resolved),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_sweep_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_sweep_get,
            patch("aiperf.kubernetes.console.print_error") as mock_print_error,
        ):
            await _run_results(
                job_id="my-sweep",
                manage_options=opts,
                output=tmp_path / "out",
                from_pods=True,
                all_artifacts=True,
                shutdown=False,
                port=0,
                operator_namespace=None,
                run=None,
            )

        mock_sweep_get.assert_not_awaited()
        resolved.api.close.assert_awaited_once()
        # At least one error printed mentioning --from-pods restriction
        msgs = [c.args[0] for c in mock_print_error.call_args_list]
        assert any("from-pods" in m for m in msgs)

    async def test_results_rejects_run_flag_for_sweep(self, tmp_path: Path) -> None:
        """--run is rejected for sweeps (not yet supported)."""
        resolved = _make_resolved_sweep()
        opts = KubeManageOptions(namespace="bench-ns")

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_target",
                new=AsyncMock(return_value=resolved),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_sweep_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_sweep_get,
            patch("aiperf.kubernetes.console.print_error") as mock_print_error,
        ):
            await _run_results(
                job_id="my-sweep",
                manage_options=opts,
                output=tmp_path / "out",
                from_pods=False,
                all_artifacts=True,
                shutdown=False,
                port=0,
                operator_namespace=None,
                run="1714069323",
            )

        mock_sweep_get.assert_not_awaited()
        resolved.api.close.assert_awaited_once()
        msgs = [c.args[0] for c in mock_print_error.call_args_list]
        assert any("--run" in m for m in msgs)


class TestDefaultSweepOutputDir:
    """Tests for the new `_default_sweep_output_dir` helper."""

    def test_returns_path_with_namespace_and_name(self) -> None:
        path = _default_sweep_output_dir("bench-ns", "my-sweep")
        s = str(path)
        assert "bench-ns" in s
        assert "my-sweep" in s

    def test_distinct_sweeps_distinct_paths(self) -> None:
        a = _default_sweep_output_dir("ns", "sweep-a")
        b = _default_sweep_output_dir("ns", "sweep-b")
        assert a != b


# ============================================================
# Fan-out: retrieve_sweep_results_from_operator
# ============================================================


class TestRetrieveSweepResultsFromOperator:
    """Tests for the new fan-out helper in `aiperf.kubernetes.results`."""

    async def test_fans_out_to_each_child(self, tmp_path: Path) -> None:
        """Each manifest entry triggers one per-child retrieve call."""
        from aiperf.kubernetes.results import retrieve_sweep_results_from_operator

        manifest = {
            "sweepRunEpoch": "1714069323",
            "children": [
                {
                    "namespace": "bench-ns",
                    "name": "sweep-c0",
                    "variationIndex": 0,
                    "variationLabel": "c8",
                    "trialIndex": None,
                    "childRunEpoch": "1714069300",
                },
                {
                    "namespace": "bench-ns",
                    "name": "sweep-c1",
                    "variationIndex": 1,
                    "variationLabel": "c16",
                    "trialIndex": 2,
                    "childRunEpoch": "1714069310",
                },
            ],
        }

        with (
            patch(
                "aiperf.kubernetes.results._fetch_children_manifest",
                new=AsyncMock(return_value=manifest),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_get,
        ):
            ok = await retrieve_sweep_results_from_operator(
                "my-sweep",
                "bench-ns",
                tmp_path,
                MagicMock(),
                local_port=0,
                operator_namespace="aiperf-system",
            )

        assert ok is True
        assert mock_get.await_count == 2
        # Per-child output dirs follow v<varidx>-t<trialidx> shape; trialIndex
        # null collapses to t0.
        call_args = [c.args for c in mock_get.await_args_list]
        names = [a[0] for a in call_args]
        out_dirs = [a[2] for a in call_args]
        assert names == ["sweep-c0", "sweep-c1"]
        assert out_dirs[0] == tmp_path / "v0-t0"
        assert out_dirs[1] == tmp_path / "v1-t2"
        # Manifest persisted alongside per-cell dirs.
        assert (tmp_path / "sweep_manifest.json").is_file()

    async def test_partial_failure_returns_false_after_all_attempted(
        self, tmp_path: Path
    ) -> None:
        """A failed child does not short-circuit; all children are attempted."""
        from aiperf.kubernetes.results import retrieve_sweep_results_from_operator

        manifest = {
            "sweepRunEpoch": "1714069323",
            "children": [
                {
                    "namespace": "ns",
                    "name": f"sweep-c{i}",
                    "variationIndex": i,
                    "variationLabel": f"v{i}",
                    "trialIndex": None,
                    "childRunEpoch": "1",
                }
                for i in range(3)
            ],
        }
        # First and third succeed, second fails.
        outcomes = [True, False, True]
        get = AsyncMock(side_effect=outcomes)

        with (
            patch(
                "aiperf.kubernetes.results._fetch_children_manifest",
                new=AsyncMock(return_value=manifest),
            ),
            patch("aiperf.kubernetes.results.retrieve_results_from_operator", new=get),
        ):
            ok = await retrieve_sweep_results_from_operator(
                "my-sweep",
                "ns",
                tmp_path,
                MagicMock(),
                local_port=0,
                operator_namespace="aiperf-system",
            )

        assert ok is False
        assert get.await_count == 3

    @pytest.mark.parametrize(
        ("manifest", "expected"),
        [
            ({"sweepRunEpoch": "1", "children": []}, False),
            (None, False),
        ],
    )
    async def test_empty_or_missing_manifest_returns_false(
        self, tmp_path: Path, manifest: dict | None, expected: bool
    ) -> None:
        """Empty children list or missing manifest both return False."""
        from aiperf.kubernetes.results import retrieve_sweep_results_from_operator

        with (
            patch(
                "aiperf.kubernetes.results._fetch_children_manifest",
                new=AsyncMock(return_value=manifest),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_get,
        ):
            ok = await retrieve_sweep_results_from_operator(
                "my-sweep",
                "ns",
                tmp_path,
                MagicMock(),
                local_port=0,
                operator_namespace="aiperf-system",
            )

        assert ok is expected
        mock_get.assert_not_awaited()
