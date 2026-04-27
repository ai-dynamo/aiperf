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
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pytest

from tests.kubernetes.audit.bare_pod import BarePodConfig, BarePodDeployer
from tests.kubernetes.audit.cases import AUDIT_CASES, SWEEP_AUDIT_CASES, AuditCase
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
from tests.kubernetes.audit.sweep_runner import SweepAuditRunner, SweepCell
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import OperatorDeployer


@pytest.mark.k8s_audit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", AUDIT_CASES, ids=lambda c: c.case_id)
async def test_operator_vs_bare_pod(
    case: AuditCase,
    kubectl: KubectlClient,
    operator_ready: OperatorDeployer,
    audit_artifacts_dir: Path,
) -> None:
    """One audit case: operator path vs bare-pod path, three-bucket diff."""
    namespace = f"audit-{case.case_id}-{uuid.uuid4().hex[:6]}"
    op_dir = audit_artifacts_dir / "operator"
    bare_dir = audit_artifacts_dir / "bare"

    operator_runner = OperatorAuditRunner(
        deployer=operator_ready,
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


def _swept_concurrency(case: AuditCase, cell: SweepCell) -> int:
    """Resolve which concurrency value applies to a given variation index.

    The case's ``sweep`` dict maps a single dimension to its values list; the
    cell's ``variation_index`` indexes into that list. The current AuditCase
    convention is one swept dimension at a time; assert it.
    """
    if case.sweep is None or len(case.sweep) != 1:
        raise AssertionError(
            f"sweep audit case {case.case_id} must have exactly one swept dim"
        )
    ((dim_name, values),) = case.sweep.items()
    if dim_name != "concurrency":
        # Today only "concurrency" is wired through BarePodDeployer.swept_value;
        # extend if the suite ever sweeps a different dim.
        raise AssertionError(
            f"sweep audit case {case.case_id}: only 'concurrency' is currently supported, "
            f"got '{dim_name}'"
        )
    if cell.variation_index >= len(values):
        raise AssertionError(
            f"variation_index {cell.variation_index} out of range for sweep dim "
            f"'{dim_name}' with {len(values)} values"
        )
    return int(values[cell.variation_index])


def _prefix_findings(findings: list, prefix: str) -> list:
    """Return a new list of Findings with ``field`` prefixed by ``prefix:``."""
    return [dataclass_replace(f, field=f"{prefix}:{f.field}") for f in findings]


@pytest.mark.k8s_audit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", SWEEP_AUDIT_CASES, ids=lambda c: c.case_id)
async def test_operator_vs_bare_pod_sweep(
    case: AuditCase,
    kubectl: KubectlClient,
    operator_ready: OperatorDeployer,
    audit_artifacts_dir: Path,
) -> None:
    """Sweep-with-trials audit: AIPerfSweep vs N sequential bare-pod runs."""
    namespace = f"audit-{case.case_id}-{uuid.uuid4().hex[:6]}"
    op_root = audit_artifacts_dir / "operator"
    bare_root = audit_artifacts_dir / "bare"

    sweep_runner = SweepAuditRunner(kubectl=kubectl, config=OperatorAuditConfig())
    cells = await sweep_runner.run(
        case=case,
        namespace=namespace,
        dest_dir=op_root,
        timeout=1800,
    )

    bare = BarePodDeployer(kubectl=kubectl, config=BarePodConfig())

    all_findings: list = []
    for cell in cells:
        cell_id = f"v{cell.variation_index}-t{cell.trial_index}"
        bare_cell_dir = bare_root / cell_id
        await bare.run(
            case=case,
            namespace=namespace,
            dest_dir=bare_cell_dir,
            swept_value=_swept_concurrency(case, cell),
            timeout=900,
        )

        cell_findings = (
            diff_exact(operator_dir=cell.local_dir, bare_dir=bare_cell_dir, case=case)
            + diff_tolerance(
                operator_dir=cell.local_dir, bare_dir=bare_cell_dir, case=case
            )
            + diff_structural(
                operator_dir=cell.local_dir, bare_dir=bare_cell_dir, case=case
            )
        )
        all_findings.extend(_prefix_findings(cell_findings, cell_id))

    findings = AuditFindings(case_id=case.case_id, findings=all_findings)

    (audit_artifacts_dir / "audit-report.json").write_text(render_json(findings))
    md = render_markdown(findings)
    (audit_artifacts_dir / "report.md").write_text(md)

    if not findings.empty:
        print(md)
    assert findings.empty, (
        f"sweep audit failures for {case.case_id}: see {audit_artifacts_dir}/report.md"
    )
