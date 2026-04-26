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


def test_aiperfsweep_template_benchmark_has_strict_walked_schema():
    """spec.template.spec.benchmark walks AIPerfConfig (Task 6 of plan).

    The previous blanket ``x-kubernetes-preserve-unknown-fields: true`` is
    replaced with a real walk; only narrow shorthand boundaries (models,
    endpoint.urls, top-level shortcuts) keep the marker.
    """
    from tools.generate_crd import build_aiperfsweep_crd

    crd = build_aiperfsweep_crd()
    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    template_spec = schema["properties"]["spec"]["properties"]["template"][
        "properties"
    ]["spec"]
    benchmark = template_spec["properties"]["benchmark"]

    # Top-level benchmark must NOT carry a blanket preserve-unknown marker —
    # individual fields are walked and validated by the apiserver.
    assert "properties" in benchmark, "benchmark should be a strictly walked object"
    assert benchmark.get("x-kubernetes-preserve-unknown-fields") is not True, (
        "benchmark must not be a blanket preserve-unknown — Task 6 walks it"
    )

    # Narrow markers at known shorthand boundaries.
    bp = benchmark["properties"]
    assert bp["models"].get("x-kubernetes-preserve-unknown-fields") is True, (
        "models accepts shorthand and must keep the marker"
    )
    assert (
        bp["endpoint"]["properties"]["urls"].get("x-kubernetes-preserve-unknown-fields")
        is True
    ), "endpoint.urls accepts shorthand and must keep the marker"

    # Strict fields: runtime should be fully typed (no top-level marker).
    runtime = bp["runtime"]
    assert runtime.get("x-kubernetes-preserve-unknown-fields") is not True, (
        "runtime should be strictly validated, no preserve-unknown blanket"
    )
    assert "properties" in runtime, "runtime should expose its real properties"

    # Top-level shortcut siblings (Task 5) are present and marked.
    for shortcut in ("model", "dataset", "warmup", "profiling"):
        assert shortcut in bp, f"{shortcut} shortcut sibling missing"
        assert bp[shortcut].get("x-kubernetes-preserve-unknown-fields") is True, (
            f"{shortcut} shortcut must carry preserve-unknown marker"
        )


def test_aiperfjob_benchmark_has_strict_walked_schema():
    """AIPerfJob spec.benchmark walks AIPerfConfig (Task 6 of plan).

    Mirrors :func:`test_aiperfsweep_template_benchmark_has_strict_walked_schema`
    but on the AIPerfJob CRD where the benchmark blanket previously lived.
    """
    from tools.generate_crd import _build_crd

    crd = _build_crd({})
    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    benchmark = schema["properties"]["spec"]["properties"]["benchmark"]

    assert "properties" in benchmark, "benchmark should be a strictly walked object"
    assert benchmark.get("x-kubernetes-preserve-unknown-fields") is not True, (
        "benchmark must not be a blanket preserve-unknown — Task 6 walks it"
    )

    bp = benchmark["properties"]
    assert bp["models"].get("x-kubernetes-preserve-unknown-fields") is True
    assert (
        bp["endpoint"]["properties"]["urls"].get("x-kubernetes-preserve-unknown-fields")
        is True
    )
    assert bp["runtime"].get("x-kubernetes-preserve-unknown-fields") is not True
    assert "properties" in bp["runtime"]

    for shortcut in ("model", "dataset", "warmup", "profiling"):
        assert shortcut in bp, f"{shortcut} shortcut sibling missing"
        assert bp[shortcut].get("x-kubernetes-preserve-unknown-fields") is True
