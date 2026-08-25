# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube results <sweep-name>` (R2 wiring).

The job path is exercised in `test_kube.py`; this file targets the new
ResolvedSweep branch in `_run_results` and the new sweep fan-out helper
`retrieve_sweep_results_from_operator` in `aiperf.kubernetes.results`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
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

    async def test_missing_target_reports_failure(self, tmp_path: Path) -> None:
        """An unresolved job must propagate failure to the CLI exit status."""
        with patch(
            "aiperf.kubernetes.cli_helpers.resolve_target",
            new=AsyncMock(return_value=None),
        ):
            success = await _run_results(
                job_id="missing-job",
                manage_options=KubeManageOptions(namespace="bench-ns"),
                output=tmp_path / "out",
                from_pods=False,
                all_artifacts=True,
                shutdown=False,
                port=0,
                operator_namespace=None,
                run=None,
            )

        assert success is False

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
            success = await _run_results(
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
        assert success is True
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
            success = await _run_results(
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
        assert success is False
        resolved.api.close.assert_awaited_once()
        # At least one error printed mentioning --from-pods restriction
        msgs = [c.args[0] for c in mock_print_error.call_args_list]
        assert any("from-pods" in m for m in msgs)

    async def test_results_accepts_run_flag_for_sweep(self, tmp_path: Path) -> None:
        """--run pins whole-sweep downloads to the requested sweep epoch."""
        resolved = _make_resolved_sweep()
        opts = KubeManageOptions(namespace="bench-ns")

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
            patch("aiperf.kubernetes.console.print_error") as mock_print_error,
        ):
            success = await _run_results(
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

        mock_sweep_get.assert_awaited_once()
        assert success is True
        assert mock_sweep_get.await_args.kwargs["run"] == "1714069323"
        resolved.api.close.assert_awaited_once()
        mock_print_error.assert_not_called()


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

    def test_run_pinned_downloads_include_epoch(self) -> None:
        path = _default_sweep_output_dir("bench-ns", "my-sweep", run="1714069323")
        s = str(path)
        assert "bench-ns" in s
        assert "my-sweep" in s
        assert "1714069323" in s


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
            ) as mock_fetch,
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_get,
            patch(
                "aiperf.kubernetes.results.retrieve_sweep_aggregate_artifacts_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_aggregate_get,
        ):
            ok = await retrieve_sweep_results_from_operator(
                "my-sweep",
                "bench-ns",
                tmp_path,
                MagicMock(),
                local_port=0,
                operator_namespace="aiperf-system",
                run="1714069323",
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
        fetch_kwargs = mock_fetch.await_args.kwargs
        assert fetch_kwargs["run"] == "1714069323"
        run_kwargs = [c.kwargs["run"] for c in mock_get.await_args_list]
        assert run_kwargs == ["1714069300", "1714069310"]
        mock_aggregate_get.assert_awaited_once()
        assert mock_aggregate_get.await_args.kwargs["run"] == "1714069323"
        # Manifest persisted alongside per-cell dirs.
        assert (tmp_path / "sweep_manifest.json").is_file()

    async def test_aggregate_failure_returns_false_after_children_attempted(
        self, tmp_path: Path
    ) -> None:
        """A failed aggregate fetch makes epoch download fail after children run."""
        from aiperf.kubernetes.results import retrieve_sweep_results_from_operator

        manifest = {
            "sweepRunEpoch": "1714069323",
            "children": [
                {
                    "namespace": "ns",
                    "name": "sweep-c0",
                    "variationIndex": 0,
                    "variationLabel": "v0",
                    "trialIndex": None,
                    "childRunEpoch": "1",
                }
            ],
        }
        get = AsyncMock(return_value=True)

        with (
            patch(
                "aiperf.kubernetes.results._fetch_children_manifest",
                new=AsyncMock(return_value=manifest),
            ),
            patch("aiperf.kubernetes.results.retrieve_results_from_operator", new=get),
            patch(
                "aiperf.kubernetes.results.retrieve_sweep_aggregate_artifacts_from_operator",
                new=AsyncMock(return_value=False),
            ) as mock_aggregate_get,
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
        assert get.await_count == 1
        mock_aggregate_get.assert_awaited_once()

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
            patch(
                "aiperf.kubernetes.results.retrieve_sweep_aggregate_artifacts_from_operator",
                new=AsyncMock(return_value=True),
            ),
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


class _FakeContent:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
        yield self._body


class _FakeResponse:
    def __init__(
        self,
        *,
        body: bytes = b"",
        json_body: dict | None = None,
        status: int = 200,
    ) -> None:
        self.status = status
        self.headers: dict[str, str] = {}
        self.content = _FakeContent(body)
        self._json_body = json_body or {}

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            import aiohttp

            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
                message="error",
            )

    async def json(self, *, loads: object | None = None) -> dict:
        if loads is None:
            raise AssertionError("aiohttp response JSON must pass orjson.loads")
        return self._json_body


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        self._responses = responses
        self.requested: list[str] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def get(self, url: str, **_kwargs: object) -> _FakeResponse:
        self.requested.append(url)
        return self._responses[url]


class TestSweepAggregateArtifactClient:
    async def test_lists_and_downloads_from_sweep_epoch_artifact_api(
        self, tmp_path: Path
    ) -> None:
        from aiperf.kubernetes.results_operator import (
            _download_all_sweep_operator_files,
        )

        artifact_name = "sweep_aggregate/profile_export_aiperf.json"
        list_url = (
            "http://operator/api/v1/sweeps/bench/my-sweep/epochs/1714069323/artifacts"
        )
        file_url = f"{list_url}/{artifact_name}"
        session = _FakeSession(
            {
                list_url: _FakeResponse(
                    json_body={
                        "files": [
                            {
                                "name": artifact_name,
                                "stored_name": "profile_export_aiperf.json.zst",
                                "size_bytes": 2,
                                "compressed": False,
                            }
                        ]
                    }
                ),
                file_url: _FakeResponse(body=b"{}"),
            }
        )

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            downloaded = await _download_all_sweep_operator_files(
                api_base="http://operator",
                namespace="bench",
                sweep_name="my-sweep",
                output_dir=tmp_path,
                run="1714069323",
            )

        assert downloaded.downloaded == [(artifact_name, 2)]
        assert downloaded.complete is True
        assert session.requested == [list_url, file_url]
        assert (tmp_path / artifact_name).read_bytes() == b"{}"

    async def test_partial_aggregate_artifact_failure_is_reported(
        self, tmp_path: Path
    ) -> None:
        """A per-file failure must not discard the files that landed.

        Returning None on the first failure threw away a mostly-complete
        aggregate directory and named nothing; the job-level twin has been
        partial-tolerant since 5a51031db5. The caller still refuses to claim
        success -- see `complete`.
        """
        from aiperf.kubernetes.results_operator import (
            _download_all_sweep_operator_files,
        )

        ok_artifact = "sweep_aggregate/profile_export_aiperf.json"
        missing_artifact = "sweep_aggregate/profile_export_console.txt"
        list_url = (
            "http://operator/api/v1/sweeps/bench/my-sweep/epochs/1714069323/artifacts"
        )
        session = _FakeSession(
            {
                list_url: _FakeResponse(
                    json_body={
                        "files": [
                            {"name": ok_artifact, "size_bytes": 2},
                            {"name": missing_artifact, "size_bytes": 5},
                        ]
                    }
                ),
                f"{list_url}/{ok_artifact}": _FakeResponse(body=b"{}"),
                f"{list_url}/{missing_artifact}": _FakeResponse(status=500),
            }
        )

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            downloaded = await _download_all_sweep_operator_files(
                api_base="http://operator",
                namespace="bench",
                sweep_name="my-sweep",
                output_dir=tmp_path,
                run="1714069323",
            )

        assert downloaded is not None
        assert downloaded.complete is False
        assert downloaded.downloaded == [(ok_artifact, 2)]
        assert downloaded.failed == [missing_artifact]
        assert session.requested == [
            list_url,
            f"{list_url}/{ok_artifact}",
            f"{list_url}/{missing_artifact}",
        ]
        assert (tmp_path / ok_artifact).read_bytes() == b"{}"
