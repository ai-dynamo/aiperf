# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures and options for the K8s-vs-local audit suite.

The audit suite reuses the cluster, image, and mock-server fixtures from
``tests/kubernetes/conftest.py`` and adds:

- ``--audit-repeats N`` (default 1): per-side repeat count. When N > 1, each
  side runs N times and per-metric medians are diffed; useful locally when
  investigating a divergence.
- ``--audit-strict-tolerance`` (default False): make tolerance-bucket
  findings fail the test. Off by default because the tolerance bucket is an
  environment-quality signal — on single-node kind clusters the operator
  side runs ~50-80% slower than bare-pod due to mock-server CPU contention
  with the operator's 13+ containers, which is irreducible test-env noise,
  not a correctness defect. Exact and Structural buckets always gate the
  test; turn this flag on when running against a properly-sized cluster.
- ``audit_artifacts_dir`` fixture: per-test directory under
  ``tests/_artifacts/audit/<case-id>/`` where both modes' artifacts and the
  rendered report are written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--audit-repeats",
        action="store",
        type=int,
        default=1,
        help="Per-side repeat count for the k8s_audit suite (default 1).",
    )
    parser.addoption(
        "--audit-strict-tolerance",
        action="store_true",
        default=False,
        help="Fail audit cases on tolerance-bucket findings (numeric metric drift). "
        "Off by default — tolerance is environment-dependent on single-node kind. "
        "Always-on: Exact (record/error counts, dataset hash) and Structural "
        "(file presence, schemas) buckets gate the test.",
    )


@pytest.fixture(scope="session")
def audit_repeats(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--audit-repeats"))


@pytest.fixture(scope="session")
def audit_strict_tolerance(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--audit-strict-tolerance"))


@pytest.fixture
def audit_artifacts_dir(request: pytest.FixtureRequest) -> Path:
    """Per-test artifacts directory.

    Uses the parametrize id (the AuditCase.case_id) so cases are isolated.
    """
    case_id = (
        request.node.callspec.id
        if hasattr(request.node, "callspec")
        else request.node.name
    )
    base = _REPO_ROOT / "tests" / "_artifacts" / "audit" / case_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "operator").mkdir(exist_ok=True)
    (base / "bare").mkdir(exist_ok=True)
    return base
