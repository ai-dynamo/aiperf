# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the Kubernetes audit CI route."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_recipe(source: str, target: str) -> str:
    """Return a Make target's recipe through the next target definition."""
    match = re.search(
        rf"^{re.escape(target)}:.*?(?=^[A-Za-z][A-Za-z0-9_-]*(?: [A-Za-z0-9_-]+)*:|\\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing Make target: {target}"
    return match.group()


def test_kubernetes_audit_ci_route_selects_serial_audit_suite() -> None:
    """The bounded audit workflow must invoke its serial Make target."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "run-kubernetes-audit-tests.yml"
    ).read_text(encoding="utf-8")

    recipe = _make_recipe(
        makefile, "kubernetes-audit-tests-ci test-kubernetes-audit-ci"
    )

    assert "pytest tests/kubernetes/audit/" in recipe
    assert "-m 'k8s and k8s_audit'" in recipe
    assert "-n 0" in recipe
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "make kubernetes-audit-tests-ci args='--k8s-skip-build" in workflow
