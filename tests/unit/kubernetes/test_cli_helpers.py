# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.cli_helpers."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.kubernetes.cli_helpers import (
    ResolvedJob,
    ResolvedSweep,
    format_age,
    resolve_job,
    resolve_target,
)
from aiperf.kubernetes.constants import AIPerfLabels
from aiperf.kubernetes.models import AIPerfJobInfo, AIPerfSweepInfo, JobSetInfo


def _raw_jobset(status_obj: dict | None = None) -> dict[str, Any]:
    """Build a minimal raw JobSet dict for testing."""
    raw: dict[str, Any] = {
        "metadata": {"name": "test", "namespace": "default"},
    }
    if status_obj is not None:
        raw["status"] = status_obj
    return raw


class TestJobSetInfoStatus:
    """Tests for status parsing via JobSetInfo.from_raw."""

    @pytest.mark.parametrize(
        "status_obj,expected",
        [
            (
                {"conditions": [{"type": "Completed", "status": "True"}]},
                "Completed",
            ),
            (
                {
                    "conditions": [{"type": "Failed", "status": "True"}],
                    "replicatedJobsStatus": [
                        {"name": "controller", "failed": 1, "ready": 0, "succeeded": 0}
                    ],
                },
                "Failed",
            ),
            ({"conditions": [], "ready": 1}, "Running"),
            ({"conditions": [], "ready": 0}, "Running"),
            ({}, "Running"),
            (
                {"conditions": [{"type": "Completed", "status": "False"}]},
                "Running",
            ),
        ],  # fmt: skip
    )
    def test_status_from_raw(self, status_obj: dict, expected: str) -> None:
        """Test extracting status from various JobSet objects."""
        info = JobSetInfo.from_raw(_raw_jobset(status_obj))
        assert info.status == expected

    def test_status_no_status_key(self) -> None:
        """Test JobSet without status key defaults to Running."""
        info = JobSetInfo.from_raw(_raw_jobset())
        assert info.status == "Running"

    def test_status_completed_takes_priority(self) -> None:
        """Test JobSet with multiple conditions - Completed takes priority."""
        info = JobSetInfo.from_raw(
            _raw_jobset(
                {
                    "conditions": [
                        {"type": "Running", "status": "True"},
                        {"type": "Completed", "status": "True"},
                    ]
                }
            )
        )
        assert info.status == "Completed"


class TestFormatAge:
    """Tests for format_age function."""

    def test_format_age_seconds(self) -> None:
        """Test formatting age in seconds."""
        now = datetime.now(UTC)
        timestamp = (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        # Allow some tolerance for test execution time
        assert result.endswith("s")
        assert int(result[:-1]) >= 29

    def test_format_age_minutes(self) -> None:
        """Test formatting age in minutes."""
        now = datetime.now(UTC)
        timestamp = (now - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result == "15m"

    def test_format_age_hours(self) -> None:
        """Test formatting age in hours."""
        now = datetime.now(UTC)
        timestamp = (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result == "3h"

    def test_format_age_empty_string(self) -> None:
        """Test formatting with empty string."""
        assert format_age("") == "Unknown"

    def test_format_age_boundary_59_seconds(self) -> None:
        """Test boundary at 59 seconds."""
        now = datetime.now(UTC)
        timestamp = (now - timedelta(seconds=59)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result.endswith("s")

    def test_format_age_boundary_60_seconds(self) -> None:
        """Test boundary at exactly 60 seconds becomes 1m."""
        now = datetime.now(UTC)
        timestamp = (now - timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
        result = format_age(timestamp)
        assert result == "1m"


class TestLabelConstants:
    """Tests for label constants."""

    def test_aiperf_label_format(self) -> None:
        """Test AIPerfLabels.SELECTOR constant format."""
        assert AIPerfLabels.SELECTOR == "app=aiperf"

    def test_aiperf_job_id_label_format(self) -> None:
        """Test AIPerfLabels.JOB_ID constant format."""
        assert AIPerfLabels.JOB_ID == "aiperf.nvidia.com/job-id"


class TestJobSetInfo:
    """Tests for JobSetInfo dataclass."""

    def test_jobset_info_creation(self, sample_jobset) -> None:
        """Test creating JobSetInfo instance."""
        info = JobSetInfo(
            name="test",
            namespace="default",
            jobset=sample_jobset,
            status="Running",
        )
        assert info.name == "test"
        assert info.namespace == "default"
        assert info.status == "Running"
        assert info.jobset == sample_jobset


# ============================================================
# ResolvedJob
# ============================================================


def _make_job_info(**overrides: Any) -> AIPerfJobInfo:
    """Build an AIPerfJobInfo with sensible defaults, overridden by kwargs."""
    defaults: dict[str, Any] = {
        "name": "my-job",
        "namespace": "bench-ns",
        "phase": "Running",
        "job_id": "abc123",
        "jobset_name": "aiperf-abc123",
        "created": "2026-01-15T10:30:00Z",
    }
    defaults.update(overrides)
    return AIPerfJobInfo(**defaults)


class TestResolvedJob:
    """Tests for the ResolvedJob wrapper class."""

    def test_jobset_name_delegates_to_job_info(self) -> None:
        """Test that jobset_name property reads from job_info."""
        info = _make_job_info(jobset_name="js-name")
        resolved = ResolvedJob(name="n", job_info=info, api=MagicMock())
        assert resolved.jobset_name == "js-name"

    def test_jobset_name_none_when_absent(self) -> None:
        """Test that jobset_name is None when job_info has no jobset_name."""
        info = _make_job_info(jobset_name=None)
        resolved = ResolvedJob(name="n", job_info=info, api=MagicMock())
        assert resolved.jobset_name is None

    def test_namespace_delegates_to_job_info(self) -> None:
        """Test that namespace property reads from job_info."""
        info = _make_job_info(namespace="prod")
        resolved = ResolvedJob(name="n", job_info=info, api=MagicMock())
        assert resolved.namespace == "prod"

    def test_job_id_delegates_to_job_info(self) -> None:
        """Test that job_id property reads from job_info."""
        info = _make_job_info(job_id="xyz789")
        resolved = ResolvedJob(name="n", job_info=info, api=MagicMock())
        assert resolved.job_id == "xyz789"

    def test_name_stored_directly(self) -> None:
        """Test that the name attribute is stored directly on ResolvedJob."""
        info = _make_job_info()
        resolved = ResolvedJob(name="lookup-name", job_info=info, api=MagicMock())
        assert resolved.name == "lookup-name"

    def test_api_stored_directly(self) -> None:
        """Test that the api attribute is stored directly on ResolvedJob."""
        api = MagicMock()
        resolved = ResolvedJob(name="n", job_info=_make_job_info(), api=api)
        assert resolved.api is api


# ============================================================
# resolve_job
# ============================================================


class TestResolveJob:
    """Tests for the resolve_job async helper."""

    async def test_resolve_job_found_via_aiperfjob_cr(self) -> None:
        """Test that resolve_job returns ResolvedJob when AIPerfJob CR is found."""
        api = MagicMock()
        job_info = _make_job_info(name="bench-1", namespace="ns-1", job_id="bench-1")

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=job_info),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(),
            ) as mock_find_jobset,
        ):
            result = await resolve_job("bench-1", namespace="ns-1")

        assert result is not None
        assert isinstance(result, ResolvedJob)
        assert result.name == "bench-1"
        assert result.job_info is job_info
        assert result.api is api
        mock_find_jobset.assert_not_awaited()

    async def test_resolve_job_fallback_to_jobset(self) -> None:
        """Test that resolve_job falls back to JobSet when no AIPerfJob CR exists."""
        api = MagicMock()
        jobset_info = JobSetInfo(
            name="aiperf-fallback",
            namespace="default",
            jobset={
                "metadata": {
                    "name": "aiperf-fallback",
                    "namespace": "default",
                    "creationTimestamp": "2026-01-10T08:00:00Z",
                    "labels": {"aiperf.nvidia.com/job-id": "fallback-id"},
                    "annotations": {},
                },
                "status": {},
            },
            status="Running",
            model="llama-3",
            endpoint="http://llm:8000",
        )

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=jobset_info),
            ),
        ):
            result = await resolve_job("fallback-id", namespace="default")

        assert result is not None
        assert isinstance(result, ResolvedJob)
        assert result.job_info.namespace == "default"
        assert result.job_info.model == "llama-3"
        assert result.job_info.jobset_name == "aiperf-fallback"
        assert result.api is api

    async def test_resolve_job_not_found_returns_none(self) -> None:
        """Test that resolve_job returns None when neither CR nor JobSet exists."""
        api = MagicMock()
        api.close = AsyncMock()

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await resolve_job("nonexistent", namespace="default")

        assert result is None
        api.close.assert_awaited_once()

    async def test_resolve_job_no_job_id_no_last_benchmark_returns_none(self) -> None:
        """Test that resolve_job returns None when job_id is None and no last benchmark."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark", return_value=None
        ):
            result = await resolve_job(None)

        assert result is None

    async def test_resolve_job_passes_kubeconfig_and_context(self) -> None:
        """Test that kubeconfig and kube_context are forwarded to api loading."""
        api = MagicMock()
        open_client = AsyncMock(return_value=api)

        with (
            patch("aiperf.kubernetes.cli_helpers._open_api_client", new=open_client),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=_make_job_info(name="j", job_id="j")),
            ),
        ):
            await resolve_job(
                "j",
                namespace="ns",
                kubeconfig="/my/kube",
                kube_context="ctx",
            )

        open_client.assert_awaited_once_with(kubeconfig="/my/kube", kube_context="ctx")

    async def test_resolve_job_not_found_prints_namespace_hint(self, capsys) -> None:
        """Test that resolve_job prints searched namespace when not found."""
        api = MagicMock()
        api.close = AsyncMock()

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ),
        ):
            await resolve_job("missing", namespace="prod")

        captured = capsys.readouterr()
        assert "prod" in captured.out

    async def test_resolve_job_not_found_defaults_to_benchmark_namespace(
        self, capsys
    ) -> None:
        """Test that resolve_job defaults to aiperf-benchmarks when namespace is None."""
        api = MagicMock()
        api.close = AsyncMock()

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ),
        ):
            await resolve_job("missing", namespace=None)

        captured = capsys.readouterr()
        assert "aiperf-benchmarks" in captured.out


# ============================================================
# ResolvedSweep
# ============================================================


def _make_sweep_info(**overrides: Any) -> AIPerfSweepInfo:
    """Build an AIPerfSweepInfo with sensible defaults, overridden by kwargs."""
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


class TestResolvedSweep:
    """Tests for the ResolvedSweep wrapper class."""

    def test_resolved_sweep_namespace_and_phase_properties(self) -> None:
        """namespace and phase delegate to sweep_info."""
        info = _make_sweep_info(namespace="prod", phase="Succeeded")
        resolved = ResolvedSweep(name="n", sweep_info=info, api=MagicMock())
        assert resolved.namespace == "prod"
        assert resolved.phase == "Succeeded"

    def test_resolved_sweep_name_and_api_stored_directly(self) -> None:
        """name and api attributes are stored directly on ResolvedSweep."""
        api = MagicMock()
        resolved = ResolvedSweep(
            name="lookup-name", sweep_info=_make_sweep_info(), api=api
        )
        assert resolved.name == "lookup-name"
        assert resolved.api is api

    async def test_resolved_sweep_aclose_idempotent(self) -> None:
        """Calling aclose twice is safe (close is itself idempotent)."""
        api = MagicMock()
        api.close = AsyncMock()
        resolved = ResolvedSweep(name="n", sweep_info=_make_sweep_info(), api=api)
        await resolved.aclose()
        await resolved.aclose()
        assert api.close.await_count == 2


# ============================================================
# resolve_target
# ============================================================


class TestResolveTarget:
    """Tests for the resolve_target async helper."""

    async def test_resolve_target_returns_resolved_job_when_aiperf_job_found(
        self,
    ) -> None:
        """If find_aiperf_job hits, resolve_target returns a ResolvedJob."""
        api = MagicMock()
        job_info = _make_job_info(name="bench-1", namespace="ns-1", job_id="bench-1")

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=job_info),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(),
            ) as mock_find_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(),
            ) as mock_find_jobset,
        ):
            result = await resolve_target("bench-1", namespace="ns-1")

        assert isinstance(result, ResolvedJob)
        assert result.name == "bench-1"
        assert result.job_info is job_info
        assert result.api is api
        mock_find_sweep.assert_not_awaited()
        mock_find_jobset.assert_not_awaited()

    async def test_resolve_target_returns_resolved_sweep_when_aiperfsweep_found(
        self,
    ) -> None:
        """If only the sweep CR exists, resolve_target returns a ResolvedSweep."""
        api = MagicMock()
        sweep_info = _make_sweep_info(name="bench-sweep", namespace="ns-1")

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=sweep_info),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(),
            ) as mock_find_jobset,
        ):
            result = await resolve_target("bench-sweep", namespace="ns-1")

        assert isinstance(result, ResolvedSweep)
        assert result.name == "bench-sweep"
        assert result.sweep_info is sweep_info
        assert result.api is api
        mock_find_jobset.assert_not_awaited()

    async def test_resolve_target_falls_back_to_jobset_when_neither_cr_found(
        self,
    ) -> None:
        """When job + sweep both miss, resolve_target falls back to JobSet."""
        api = MagicMock()
        jobset_info = JobSetInfo(
            name="aiperf-fallback",
            namespace="default",
            jobset={
                "metadata": {
                    "name": "aiperf-fallback",
                    "namespace": "default",
                    "creationTimestamp": "2026-01-10T08:00:00Z",
                    "labels": {"aiperf.nvidia.com/job-id": "fallback-id"},
                    "annotations": {},
                },
                "status": {},
            },
            status="Running",
            model="llama-3",
            endpoint="http://llm:8000",
        )

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=jobset_info),
            ),
        ):
            result = await resolve_target("fallback-id", namespace="default")

        assert isinstance(result, ResolvedJob)
        assert result.job_info.namespace == "default"
        assert result.job_info.model == "llama-3"
        assert result.job_info.jobset_name == "aiperf-fallback"
        assert result.api is api

    async def test_resolve_target_returns_none_when_nothing_found(self, capsys) -> None:
        """When all three lookups miss, resolve_target returns None.

        The error message names BOTH AIPerfJob and AIPerfSweep so the user
        can pick the right follow-up command.
        """
        api = MagicMock()
        api.close = AsyncMock()

        with (
            patch(
                "aiperf.kubernetes.cli_helpers._open_api_client",
                new=AsyncMock(return_value=api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await resolve_target("missing", namespace="prod")

        assert result is None
        api.close.assert_awaited_once()
        captured = capsys.readouterr()
        assert "AIPerfJob" in captured.out
        assert "AIPerfSweep" in captured.out
        assert "missing" in captured.out
