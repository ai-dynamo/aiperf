# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-vs-bare-pod correctness audit.

For each ``AuditCase`` in ``cases.AUDIT_CASES``, this test:

1. Runs the case via the operator path; downloads results via
   ``aiperf kube results``.
2. Runs the same case via a bare ``batch/v1.Job`` (no operator); copies
   results via ``kubectl cp``.
3. Diffs the two artifact trees through three buckets (exact / tolerance /
   structural) and asserts no findings.

The bare-pod side is the oracle. Tolerance bands handle wall-clock-noisy
numeric stats; exact and structural buckets must match.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from tests.kubernetes.audit.bare_pod import BarePodConfig, BarePodDeployer
from tests.kubernetes.audit.cases import AUDIT_CASES, AuditCase
from tests.kubernetes.audit.diff import (
    AuditFindings,
    diff_exact,
    diff_structural,
    diff_tolerance,
)
from tests.kubernetes.audit.operator_runner import (
    OperatorAuditConfig,
    OperatorAuditRunner,
)
from tests.kubernetes.audit.report import render_json, render_markdown
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import OperatorDeployer


@pytest.mark.k8s_audit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", AUDIT_CASES, ids=lambda c: c.case_id)
async def test_operator_vs_bare_pod(
    case: AuditCase,
    kubectl: KubectlClient,
    operator_deployer: OperatorDeployer,
    mock_server: None,
    audit_artifacts_dir: Path,
) -> None:
    """One audit case: operator path vs bare-pod path, three-bucket diff."""
    namespace = f"audit-{case.case_id}-{uuid.uuid4().hex[:6]}"
    op_dir = audit_artifacts_dir / "operator"
    bare_dir = audit_artifacts_dir / "bare"

    operator_runner = OperatorAuditRunner(
        deployer=operator_deployer,
        config=OperatorAuditConfig(),
    )
    await operator_runner.run(
        case=case,
        namespace=namespace,
        dest_dir=op_dir,
        timeout=900,
    )

    bare = BarePodDeployer(kubectl=kubectl, config=BarePodConfig())
    await bare.run(
        case=case,
        namespace=namespace,
        dest_dir=bare_dir,
        timeout=900,
    )

    findings_list = (
        diff_exact(operator_dir=op_dir, bare_dir=bare_dir, case=case)
        + diff_tolerance(operator_dir=op_dir, bare_dir=bare_dir, case=case)
        + diff_structural(operator_dir=op_dir, bare_dir=bare_dir, case=case)
    )
    findings = AuditFindings(case_id=case.case_id, findings=findings_list)

    (audit_artifacts_dir / "audit-report.json").write_text(render_json(findings))
    md = render_markdown(findings)
    (audit_artifacts_dir / "report.md").write_text(md)

    if not findings.empty:
        print(md)
    assert findings.empty, (
        f"audit failures for {case.case_id}: see {audit_artifacts_dir}/report.md"
    )
