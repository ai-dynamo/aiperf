# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the AIPerfSweep CRD generator.

Covers:
- Top-level shape of the generated CRD dict (kind, names, scope, schema paths).
- CEL ``x-kubernetes-validations`` immutability rules on critical spec fields.
- ``additionalPrinterColumns`` for ``kubectl get aiperfsweeps``.
- Helm chart emits a template containing the AIPerfSweep CRD.
"""

from __future__ import annotations

from pathlib import Path


def test_aiperfsweep_crd_has_required_paths():
    """Generate the AIPerfSweep CRD dict and assert schema shape."""
    from tools.generate_crd import build_aiperfsweep_crd

    crd = build_aiperfsweep_crd()
    assert crd["kind"] == "CustomResourceDefinition"
    assert crd["spec"]["names"]["kind"] == "AIPerfSweep"
    assert crd["spec"]["names"]["plural"] == "aiperfsweeps"
    assert crd["spec"]["scope"] == "Namespaced"

    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    spec_props = schema["properties"]["spec"]["properties"]
    assert "sweep" in spec_props
    assert "multiRun" in spec_props
    assert "convergence" in spec_props
    assert "failurePolicy" in spec_props
    assert "template" in spec_props
    assert "cancel" in spec_props


def test_aiperfsweep_immutability_rules_on_critical_fields():
    """spec.sweep, multiRun, convergence each carry a CEL immutability rule."""
    from tools.generate_crd import build_aiperfsweep_crd

    crd = build_aiperfsweep_crd()
    spec_props = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
        "spec"
    ]["properties"]
    for field in ("sweep", "multiRun", "convergence"):
        validations = spec_props[field].get("x-kubernetes-validations") or []
        assert any("oldSelf == self" in v.get("rule", "") for v in validations), (
            f"{field} missing immutability rule"
        )


def test_aiperfsweep_printer_columns_present():
    from tools.generate_crd import build_aiperfsweep_crd

    crd = build_aiperfsweep_crd()
    printer = crd["spec"]["versions"][0].get("additionalPrinterColumns") or []
    names = [c["name"] for c in printer]
    assert "Phase" in names
    assert "Age" in names


def test_helm_chart_emits_aiperfsweep_crd_template():
    """The generator writes a chart template for the AIPerfSweep CRD."""
    chart_dir = Path("deploy/helm/aiperf-operator/templates")
    # Either a separate file OR an additional YAML document in the existing
    # crd.yaml is acceptable.
    candidates = list(chart_dir.glob("crd*.yaml"))
    assert candidates, "no crd*.yaml templates found"
    text = "\n---\n".join(c.read_text() for c in candidates)
    assert "AIPerfSweep" in text, "AIPerfSweep CRD not emitted in chart templates"
    assert "aiperfsweeps" in text
