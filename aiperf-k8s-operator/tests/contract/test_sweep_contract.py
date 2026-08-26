# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AIPerfSweep contract: schema, CRD copies, and validate_sweep_envelope."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from aiperf_k8s_operator.contract import validate_sweep_envelope

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "contracts" / "native-k8s" / "v1" / "fixtures"
CONTRACTS = ROOT / "contracts" / "native-k8s" / "v1"
VENDORED = (
    ROOT / "aiperf-k8s-operator" / "src" / "aiperf_k8s_operator" / "contracts" / "v1"
)
CHART = ROOT / "deploy" / "aiperf-k8s-operator" / "helm" / "aiperf-k8s-operator"
DEPLOY_CRDS = ROOT / "deploy" / "aiperf-k8s-operator" / "crds"
HELM_CRDS = CHART / "crds"

_SCHEMA_NAMES = [
    "controller-envelope.schema.json",
    "image-capabilities.schema.json",
    "progress-status.schema.json",
    "results-manifest.schema.json",
    "sweep-envelope.schema.json",
]


def sweep_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def render_chart(*values: str) -> list[dict]:
    values = (
        "image.repository=example.invalid/aiperf-k8s-operator",
        "image.tag=test",
        *values,
    )
    result = subprocess.run(
        [
            "helm",
            "template",
            "operator",
            str(CHART),
            "--namespace",
            "control-plane",
            "--include-crds",
            *(argument for value in values for argument in ("--set", value)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [resource for resource in yaml.safe_load_all(result.stdout) if resource]


def test_sweep_envelope_accepts_valid_fixture() -> None:
    envelope = validate_sweep_envelope(sweep_fixture("valid-sweep-envelope.json"))
    assert envelope.sweep_id == "sweep-1"
    assert envelope.trials == 1
    assert len(envelope.axes) == 1
    assert envelope.axes[0].parameter == "runtime.concurrency"
    assert envelope.sweep_controller.name == "sweep-controller"


def test_sweep_envelope_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        validate_sweep_envelope(sweep_fixture("unknown-field-sweep-envelope.json"))


def test_sweep_envelope_rejects_unsupported_version() -> None:
    with pytest.raises(ValueError, match="native-k8s/v2"):
        validate_sweep_envelope(sweep_fixture("invalid-version-sweep-envelope.json"))


def test_sweep_envelope_rejects_wrong_role_name() -> None:
    payload = sweep_fixture("valid-sweep-envelope.json")
    payload["sweepController"]["name"] = "controller"
    with pytest.raises(ValueError):
        validate_sweep_envelope(payload)


def test_sweep_schema_copies_are_identical() -> None:
    """All five contract schema files must be byte-identical to their vendored copies."""
    for name in _SCHEMA_NAMES:
        canonical = (CONTRACTS / name).read_bytes()
        vendored = (VENDORED / name).read_bytes()
        assert canonical == vendored, (
            f"{name}: contracts/ and vendored operator copy differ — "
            "run `cp contracts/native-k8s/v1/{name} "
            "aiperf-k8s-operator/src/aiperf_k8s_operator/contracts/v1/{name}` to fix"
        )


def test_crd_copies_are_identical() -> None:
    """CRD YAML files must be byte-identical between deploy/crds/ and helm/crds/."""
    for crd_file in DEPLOY_CRDS.glob("*.yaml"):
        helm_copy = HELM_CRDS / crd_file.name
        assert helm_copy.exists(), (
            f"helm/crds/{crd_file.name} is missing — copy from deploy/crds/"
        )
        assert crd_file.read_bytes() == helm_copy.read_bytes(), (
            f"{crd_file.name}: deploy/crds/ and helm/crds/ copies differ"
        )


def test_sweep_crd_status_cel_rules_are_fully_guarded() -> None:
    """Pin the exact CEL rule strings so a future edit cannot drop the has() guards."""
    crd = yaml.safe_load(
        (DEPLOY_CRDS / "aiperfsweeps.aiperf.nvidia.com.yaml").read_text()
    )
    status_schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
        "properties"
    ]["status"]
    assert status_schema["x-kubernetes-validations"] == [
        {
            "rule": (
                "!has(oldSelf.sweepId) || "
                "(has(self.sweepId) && self.sweepId == oldSelf.sweepId)"
            ),
            "message": "sweepId is immutable once set",
        },
        {
            "rule": (
                "!has(oldSelf.completedRuns) || "
                "(has(self.completedRuns) && self.completedRuns >= oldSelf.completedRuns)"
            ),
            "message": "completedRuns cannot decrease",
        },
    ]


def test_helm_renders_the_sweep_crd() -> None:
    resources = render_chart()
    crd_names = {
        resource["metadata"]["name"]
        for resource in resources
        if resource.get("kind") == "CustomResourceDefinition"
    }
    assert "aiperfsweeps.aiperf.nvidia.com" in crd_names

    sweep_crd = next(
        resource
        for resource in resources
        if resource.get("kind") == "CustomResourceDefinition"
        and resource["metadata"]["name"] == "aiperfsweeps.aiperf.nvidia.com"
    )
    spec = sweep_crd["spec"]
    assert spec["group"] == "aiperf.nvidia.com"
    assert spec["names"]["kind"] == "AIPerfSweep"
    assert spec["names"]["shortNames"] == ["aps"]
    version = spec["versions"][0]
    assert version["name"] == "v1alpha1"
    assert version["served"] is True
    assert version["storage"] is True
    schema = version["schema"]["openAPIV3Schema"]
    spec_schema = schema["properties"]["spec"]
    assert spec_schema["x-kubernetes-validations"] == [
        {"rule": "self == oldSelf", "message": "AIPerfSweep spec is immutable"}
    ]
    status_schema = schema["properties"]["status"]
    assert status_schema["properties"]["phase"]["enum"] == [
        "Pending",
        "Running",
        "Completed",
        "Failed",
    ]
