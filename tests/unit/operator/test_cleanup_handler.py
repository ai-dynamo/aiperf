# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.operator.handlers.cleanup`` not covered by ``test_main.py``.

``test_main.py`` exercises the phase-gating, TTL expiry, and shutil failure
paths. This file focuses on the *security* guard (path-traversal rejection)
and the default-TTL fallback.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest

from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers.cleanup import cleanup_old_results
from aiperf.operator.status import Phase


class TestCleanupPathTraversalGuard:
    """Tests that cleanup refuses to act on paths outside ``RESULTS.DIR``."""

    @pytest.mark.asyncio
    async def test_refuses_path_outside_results_dir(self, tmp_path: Path) -> None:
        """Verify a results_path escaping RESULTS.DIR is skipped and NOT deleted."""
        results_root = tmp_path / "aiperf-results"
        results_root.mkdir()

        # A sibling dir outside results_root — this must NEVER be deleted.
        outside = tmp_path / "not-aiperf" / "stuff"
        outside.mkdir(parents=True)
        victim = outside / "important.txt"
        victim.write_text("do not delete me")

        # Age it past TTL so only the guard can save it.
        old_ts = datetime.now(timezone.utc).timestamp() - (99 * 86400)
        os.utime(outside, (old_ts, old_ts))

        with mock_patch.object(OperatorEnvironment.RESULTS, "DIR", results_root):
            await cleanup_old_results(
                body={},
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "job-evil",
                    "resultsPath": str(outside),
                    "resultsTtlDays": 1,
                },
                name="test-job",
            )

        assert outside.exists()
        assert victim.read_text() == "do not delete me"

    @pytest.mark.asyncio
    async def test_cleans_up_failed_phase(self, tmp_path: Path) -> None:
        """Verify FAILED phase jobs are cleaned up (they can leak partial artifacts)."""
        results_dir = tmp_path / "job-failed"
        results_dir.mkdir()

        old_ts = datetime.now(timezone.utc).timestamp() - (40 * 86400)
        os.utime(results_dir, (old_ts, old_ts))

        with (
            mock_patch("aiperf.operator.events.results_cleaned"),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", tmp_path),
        ):
            await cleanup_old_results(
                body={},
                status={
                    "phase": Phase.FAILED,
                    "jobId": "job-failed",
                    "resultsPath": str(results_dir),
                    "resultsTtlDays": 30,
                },
                name="test-job",
            )

        assert not results_dir.exists()

    @pytest.mark.asyncio
    async def test_uses_default_ttl_when_missing(self, tmp_path: Path) -> None:
        """Verify the environment default TTL is used when ``resultsTtlDays`` absent."""
        results_dir = tmp_path / "job-default"
        results_dir.mkdir()

        # Age it just past the env default.
        default_ttl = OperatorEnvironment.RESULTS.TTL_DAYS
        age_days = default_ttl + 1
        old_ts = datetime.now(timezone.utc).timestamp() - (age_days * 86400)
        os.utime(results_dir, (old_ts, old_ts))

        with (
            mock_patch("aiperf.operator.events.results_cleaned"),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", tmp_path),
        ):
            await cleanup_old_results(
                body={},
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "job-default",
                    "resultsPath": str(results_dir),
                    # Note: no resultsTtlDays.
                },
                name="test-job",
            )

        assert not results_dir.exists()
