# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path

import pytest

from aiperf_k8s_operator.contract import validate_bootstrap_metadata, validate_envelope
from aiperf_k8s_operator.main import reconcile_job
from aiperf_k8s_operator.reconciliation import build_jobset, validate_references

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "contracts" / "native-k8s" / "v1" / "fixtures"
PACKAGE = ROOT / "aiperf-k8s-operator" / "src" / "aiperf_k8s_operator"


def fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def test_operator_sources_never_import_legacy_aiperf_package() -> None:
    for source in PACKAGE.glob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
        imports.extend(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
        assert not [name for name in imports if name == "aiperf" or name.startswith("aiperf.")], source


def test_envelope_projects_exact_two_jobsets() -> None:
    envelope = validate_envelope(fixture("valid-multi-cell-envelope.json"))
    jobset = build_jobset(envelope)
    jobs = jobset["spec"]["replicatedJobs"]
    assert [job["name"] for job in jobs] == ["controller", "cell"]
    assert jobs[0]["template"]["spec"]["containers"][1]["name"] == "results-sidecar"
    assert jobs[1]["replicas"] == 4
    assert all(container["image"] == envelope.image_digest for job in jobs for container in job["template"]["spec"]["containers"])


def test_metadata_validation_never_reads_secret_data() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    metadata = {
        role.bootstrap.secret_name: {
            "immutable": True,
            "metadata": {
                "name": role.bootstrap.secret_name,
                "labels": {"aiperf.nvidia.com/role": role.name},
                "annotations": {"aiperf.nvidia.com/sha256": role.bootstrap.sha256},
            },
            "data": {"must-not-be-read": "not-a-real-secret"},
        }
        for role in envelope.roles
    }
    validate_references(envelope, metadata)
    with pytest.raises(ValueError, match="role label"):
        validate_bootstrap_metadata(envelope.roles[0].bootstrap, {"immutable": True, "metadata": {"name": envelope.roles[0].bootstrap.secret_name}})


@pytest.mark.asyncio
async def test_reconcile_creates_projected_jobset() -> None:
    class FakeJobSets:
        kwargs: dict[str, object]

        async def create_namespaced_custom_object(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    jobsets = FakeJobSets()
    status = await reconcile_job(envelope, jobsets)

    assert status == {"phase": "Pending", "runId": envelope.run_id, "jobSet": envelope.job_id}
    assert jobsets.kwargs["group"] == "jobset.x-k8s.io"
    assert jobsets.kwargs["namespace"] == envelope.namespace
    assert jobsets.kwargs["body"] == build_jobset(envelope)
