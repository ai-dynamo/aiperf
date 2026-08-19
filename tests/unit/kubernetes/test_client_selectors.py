# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.client_selectors.

Pure string construction — these selectors are passed as ``label_selector=``
to the Kubernetes API, so the exact comma/equals syntax is load-bearing.
A single dropped comma would match every pod in the namespace instead of
one job's pods.
"""

from __future__ import annotations

from aiperf.kubernetes.client_selectors import controller_selector, job_selector
from aiperf.kubernetes.constants import AIPerfLabels, JobSetLabels


class TestJobSelector:
    """job_selector combines repo-wide SELECTOR + per-job JOB_ID."""

    def test_contains_both_required_clauses(self) -> None:
        result = job_selector("abc123")

        assert AIPerfLabels.SELECTOR in result
        assert f"{AIPerfLabels.JOB_ID}=abc123" in result

    def test_clauses_are_comma_separated(self) -> None:
        result = job_selector("abc123")
        parts = result.split(",")

        assert len(parts) == 2
        assert AIPerfLabels.SELECTOR in parts
        assert f"{AIPerfLabels.JOB_ID}=abc123" in parts

    def test_empty_job_id_produces_equals_with_empty_value(self) -> None:
        """Empty job_id shouldn't collapse the selector — the k8s API will
        still accept it syntactically; the caller is responsible for input."""
        result = job_selector("")

        assert result == f"{AIPerfLabels.SELECTOR},{AIPerfLabels.JOB_ID}="

    def test_job_id_with_hyphens_and_digits_preserved(self) -> None:
        result = job_selector("run-42-abc")
        assert f"{AIPerfLabels.JOB_ID}=run-42-abc" in result


class TestControllerSelector:
    """controller_selector extends job_selector with the replicated-job filter."""

    def test_contains_all_three_clauses(self) -> None:
        result = controller_selector("abc123")

        assert AIPerfLabels.SELECTOR in result
        assert f"{AIPerfLabels.JOB_ID}=abc123" in result
        assert f"{JobSetLabels.REPLICATED_JOB_NAME}=controller" in result

    def test_exactly_three_comma_separated_clauses(self) -> None:
        parts = controller_selector("abc123").split(",")
        assert len(parts) == 3

    def test_replicated_job_name_is_literal_controller(self) -> None:
        """Worker pods share the AIPerfLabels but not this replicated-job-name.
        Changing the 'controller' literal would match worker pods too."""
        result = controller_selector("abc123")
        assert "replicatedjob-name=controller" in result
        assert "replicatedjob-name=worker" not in result

    def test_controller_selector_is_stricter_than_job_selector(self) -> None:
        """Any pod matching controller_selector must also match job_selector."""
        job_id = "abc123"
        controller_clauses = set(controller_selector(job_id).split(","))
        job_clauses = set(job_selector(job_id).split(","))

        assert job_clauses.issubset(controller_clauses)
