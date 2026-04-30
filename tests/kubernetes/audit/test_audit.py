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

import json
import uuid
from dataclasses import replace as dataclass_replace
from pathlib import Path

import aiohttp
import pytest

from tests.kubernetes.audit.bare_pod import BarePodConfig, BarePodDeployer
from tests.kubernetes.audit.cases import AUDIT_CASES, SWEEP_AUDIT_CASES, AuditCase
from tests.kubernetes.audit.diff import (
    AuditFindings,
    diff_exact,
    diff_index_consistency,
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
    audit_strict_tolerance: bool,
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

    exact = diff_exact(operator_dir=op_dir, bare_dir=bare_dir, case=case)
    tolerance = diff_tolerance(operator_dir=op_dir, bare_dir=bare_dir, case=case)
    structural = diff_structural(operator_dir=op_dir, bare_dir=bare_dir, case=case)
    findings_list = exact + tolerance + structural
    findings = AuditFindings(case_id=case.case_id, findings=findings_list)

    (audit_artifacts_dir / "audit-report.json").write_text(render_json(findings))
    md = render_markdown(findings)
    (audit_artifacts_dir / "report.md").write_text(md)

    if not findings.empty:
        print(md)

    # Tolerance is environment-dependent (kind single-node CPU contention
    # routinely produces 50-80% throughput drift); gate only on Exact and
    # Structural unless --audit-strict-tolerance was passed.
    gating = exact + structural + (tolerance if audit_strict_tolerance else [])
    assert not gating, (
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
    audit_strict_tolerance: bool,
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

    all_gating: list = []
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

        cell_exact = diff_exact(
            operator_dir=cell.local_dir, bare_dir=bare_cell_dir, case=case
        )
        cell_tolerance = diff_tolerance(
            operator_dir=cell.local_dir, bare_dir=bare_cell_dir, case=case
        )
        cell_structural = diff_structural(
            operator_dir=cell.local_dir, bare_dir=bare_cell_dir, case=case
        )
        cell_findings = cell_exact + cell_tolerance + cell_structural
        cell_gating = (
            cell_exact
            + cell_structural
            + (cell_tolerance if audit_strict_tolerance else [])
        )
        all_findings.extend(_prefix_findings(cell_findings, cell_id))
        all_gating.extend(_prefix_findings(cell_gating, cell_id))

    findings = AuditFindings(case_id=case.case_id, findings=all_findings)

    (audit_artifacts_dir / "audit-report.json").write_text(render_json(findings))
    md = render_markdown(findings)
    (audit_artifacts_dir / "report.md").write_text(md)

    if not findings.empty:
        print(md)
    assert not all_gating, (
        f"sweep audit failures for {case.case_id}: see {audit_artifacts_dir}/report.md"
    )


# Subset of AUDIT_CASES used for the operator-only index_consistency suite.
# Running the full case matrix here would double cluster time for what is
# effectively a per-row schema check; one representative case is sufficient
# to catch drift between the disk JSON and the runs_index narrow columns.
_INDEX_CONSISTENCY_CASES = AUDIT_CASES[:1]


async def _fetch_operator_index_row(
    *,
    kubectl: KubectlClient,
    operator_namespace: str,
    namespace: str,
    job_id: str,
    timeout_seconds: int = 60,
) -> dict | None:
    """Port-forward to the operator pod and fetch the runs-index row.

    Polls ``/admin/index/run/{ns}/{job_id}`` until the row materializes (the
    operator writes it during the completion handler, which races with the
    audit's job-success signal). Returns ``None`` if the row never appears.
    """
    pods = await kubectl.run(
        "get",
        "pods",
        "-n",
        operator_namespace,
        "-l",
        "app.kubernetes.io/name=aiperf-operator",
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=True,
    )
    pod_name = pods.stdout.strip()
    if not pod_name:
        raise RuntimeError(f"no operator pod found in namespace {operator_namespace}")

    async with kubectl.port_forward(
        pod_name, remote_port=8081, namespace=operator_namespace
    ) as local_port:
        url = f"http://127.0.0.1:{local_port}/admin/index/run/{namespace}/{job_id}"
        deadline = timeout_seconds
        async with aiohttp.ClientSession() as session:
            for attempt in range(deadline // 2):
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        if resp.status != 404:
                            text = await resp.text()
                            raise RuntimeError(
                                f"unexpected {resp.status} from {url}: {text}"
                            )
                except aiohttp.ClientConnectorError:
                    pass
                # Row not yet written; retry.
                import asyncio

                await asyncio.sleep(2)
                _ = attempt
        return None


@pytest.mark.k8s_audit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _INDEX_CONSISTENCY_CASES, ids=lambda c: c.case_id)
async def test_operator_index_matches_disk(
    case: AuditCase,
    kubectl: KubectlClient,
    operator_ready: OperatorDeployer,
    audit_artifacts_dir: Path,
) -> None:
    """Operator-only: confirm the runs_index row matches disk for narrow metrics.

    This is the ``index_consistency`` bucket from the fast-job-sweep-index
    plan: after each operator-side run, compare the six
    ``DEFAULT_COMPARE_METRICS`` (avg/p50/p99) flat-column values stored by
    the operator's runs index with the matching stats inside the on-disk
    ``profile_export_aiperf.json``. Any drift is silent corruption of the
    leaderboard / history / compare APIs that read those columns directly.

    Bare-pod runs do not write to the index, so this test runs operator-side
    only; the standard three-bucket diff in ``test_operator_vs_bare_pod``
    continues to be the cross-mode oracle.
    """
    namespace = f"audit-idx-{case.case_id}-{uuid.uuid4().hex[:6]}"
    op_dir = audit_artifacts_dir / "operator"

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

    # The OperatorAuditRunner picks job_name from a uuid suffix and does
    # not return it; reconstruct from the AIPerfJob in the audit namespace.
    jobs_json = await kubectl.run(
        "get",
        "aiperfjobs",
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    jobs = json.loads(jobs_json.stdout)
    items = jobs.get("items", [])
    if not items:
        raise RuntimeError(f"no AIPerfJob found in {namespace} after operator run")
    job_id = items[0]["metadata"]["name"]

    index_row = await _fetch_operator_index_row(
        kubectl=kubectl,
        operator_namespace=operator_ready.OPERATOR_NAMESPACE,
        namespace=namespace,
        job_id=job_id,
    )

    findings = diff_index_consistency(operator_dir=op_dir, index_row=index_row)
    audit_findings = AuditFindings(case_id=f"{case.case_id}-index", findings=findings)
    (audit_artifacts_dir / "audit-index-report.json").write_text(
        render_json(audit_findings)
    )
    md = render_markdown(audit_findings)
    (audit_artifacts_dir / "index-report.md").write_text(md)
    if not audit_findings.empty:
        print(md)

    assert not findings, (
        f"index_consistency failures for {case.case_id}: "
        f"see {audit_artifacts_dir}/index-report.md"
    )
