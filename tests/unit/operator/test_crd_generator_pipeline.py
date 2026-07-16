# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CRD generator pipeline units."""

from __future__ import annotations

from tools.generate_crd import (
    HELM_CHART_FILE,
    HELM_CRD_FILE,
    HELM_SWEEP_CRD_FILE,
    CRDGenerator,
    CRDSchemaSource,
    KubernetesSchemaConverter,
)


def test_crd_schema_source_loads_job_and_sweep_schema_roots() -> None:
    source = CRDSchemaSource()

    job_schema = source.job_schema()
    sweep_schema = source.sweep_schema()

    assert job_schema["title"] == "AIPerfJobSpec"
    assert sweep_schema["title"] == "AIPerfSweepSpec"
    assert "benchmark" in job_schema["properties"]
    assert "benchmark" in sweep_schema["properties"]
    assert "image" in job_schema["properties"]
    assert "childMetadata" in sweep_schema["properties"]


def test_kubernetes_schema_converter_preserves_existing_top_level_fields() -> None:
    source = CRDSchemaSource()
    converter = KubernetesSchemaConverter()

    properties = converter.aiperf_config_fields(source.config_schema())

    assert "benchmark" in properties
    assert "plot" in properties
    assert "deployment" not in properties


def test_aiperfjob_spec_requires_benchmark() -> None:
    from tools.generate_crd import CRDDocumentBuilder

    source = CRDSchemaSource()
    job = CRDDocumentBuilder().aiperfjob_crd(source.job_schema())
    job_spec = job["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
        "spec"
    ]

    assert "benchmark" in job_spec["required"]


def test_crd_schema_enhancer_keeps_sweep_kind_rules() -> None:
    from tools.generate_crd import CRDDocumentBuilder, CRDSchemaEnhancer

    source = CRDSchemaSource()
    converter = KubernetesSchemaConverter()
    enhancer = CRDSchemaEnhancer()
    builder = CRDDocumentBuilder(converter=converter, enhancer=enhancer)

    job = builder.aiperfjob_crd(source.job_schema())
    sweep = builder.aiperfsweep_crd(source.sweep_schema())

    job_spec = job["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
        "spec"
    ]
    sweep_spec = sweep["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
        "properties"
    ]["spec"]

    assert any(
        "AIPerfJob" in rule["message"] for rule in job_spec["x-kubernetes-validations"]
    )
    assert any(
        "AIPerfSweep" in rule["message"]
        for rule in sweep_spec["x-kubernetes-validations"]
    )


def test_crd_yaml_renderer_adds_spdx_and_escapes_helm_templates() -> None:
    from tools.generate_crd import CRDYAMLRenderer

    renderer = CRDYAMLRenderer()
    content = renderer.aiperfjob_yaml(
        {
            "apiVersion": "apiextensions.k8s.io/v1",
            "kind": "CustomResourceDefinition",
            "metadata": {"name": "example"},
            "spec": {"template": "{{ .Release.Name }}"},
        }
    )

    assert content.startswith("# SPDX-FileCopyrightText:")
    assert '{{ "{{" }} .Release.Name {{ "}}" }}' in content


def test_crd_generator_emits_expected_files() -> None:
    result = CRDGenerator().generate()

    paths = {generated.path for generated in result.files}

    assert paths == {HELM_CRD_FILE, HELM_SWEEP_CRD_FILE, HELM_CHART_FILE}
    assert "AIPerfSweep CRD" in result.summary
    assert all(generated.content.endswith("\n") for generated in result.files)
