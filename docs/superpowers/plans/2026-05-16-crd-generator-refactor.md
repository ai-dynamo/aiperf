# CRD Generator Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `tools/generate_crd.py` into a config-schema-style pipeline while preserving the generated CRD output and Kubernetes spec roots.

**Architecture:** Keep `AIPerfJobSpec` and `AIPerfSweepSpec` as the schema roots. Introduce small orchestration classes in `tools/generate_crd.py` around the existing pure functions, then migrate the generator entry point to those classes without changing CLI usage.

**Tech Stack:** Python 3.10+, Pydantic JSON Schema, Kubernetes CRD OpenAPI v3, PyYAML, pytest, `tools._core.Generator`.

---

## File Structure

- Modify: `tools/generate_crd.py`
  - Add `CRDSchemaSource`, `KubernetesSchemaConverter`, `CRDSchemaEnhancer`, `CRDDocumentBuilder`, and `CRDYAMLRenderer`.
  - Keep compatibility wrappers for existing tests: `_convert_schema`, `convert_aiperf_config_fields`, `_build_crd`, `build_aiperfsweep_crd`, `render_helm_crd_yaml`, `render_helm_sweep_crd_yaml`.
  - Keep `CRDGenerator` as the public `tools._core.Generator` entry point.
- Modify: `tests/unit/operator/test_aiperfsweep_crd_generation.py`
  - Add class-level smoke tests for the new generator units while preserving existing behavior tests.
- Create: `tests/unit/operator/test_crd_generator_pipeline.py`
  - Focused integration tests for the new pipeline and generated-file wiring.

---

### Task 1: Add pipeline smoke tests

**Files:**
- Create: `tests/unit/operator/test_crd_generator_pipeline.py`
- Test: `tests/unit/operator/test_crd_generator_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/unit/operator/test_crd_generator_pipeline.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CRD generator pipeline units."""

from __future__ import annotations

from pathlib import Path

from tools.generate_crd import (
    CRDDocumentBuilder,
    CRDGenerator,
    CRDSchemaEnhancer,
    CRDSchemaSource,
    CRDYAMLRenderer,
    HELM_CHART_FILE,
    HELM_CRD_FILE,
    HELM_SWEEP_CRD_FILE,
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
    assert "deployment" in properties
    assert "plot" in properties["benchmark"].get("properties", {})


def test_crd_schema_enhancer_keeps_sweep_kind_rules() -> None:
    source = CRDSchemaSource()
    converter = KubernetesSchemaConverter()
    enhancer = CRDSchemaEnhancer()
    builder = CRDDocumentBuilder(converter=converter, enhancer=enhancer)

    job = builder.aiperfjob_crd(source.job_schema())
    sweep = builder.aiperfsweep_crd(source.sweep_schema())

    job_spec = job["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"]
    sweep_spec = sweep["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"]

    assert any("AIPerfJob" in rule["message"] for rule in job_spec["x-kubernetes-validations"])
    assert any("AIPerfSweep" in rule["message"] for rule in sweep_spec["x-kubernetes-validations"])


def test_crd_yaml_renderer_adds_spdx_and_escapes_helm_templates() -> None:
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
    assert "{{ `{{ .Release.Name }}` }}" in content


def test_crd_generator_emits_expected_files() -> None:
    result = CRDGenerator().generate()

    paths = {generated.path for generated in result.files}

    assert paths == {HELM_CRD_FILE, HELM_SWEEP_CRD_FILE, HELM_CHART_FILE}
    assert "AIPerfSweep CRD" in result.summary
    assert all(generated.content.endswith("\n") for generated in result.files)
```

- [ ] **Step 2: Run tests to verify they fail on missing classes**

Run:

```bash
uv run pytest -n auto tests/unit/operator/test_crd_generator_pipeline.py
```

Expected: FAIL with import errors for `CRDSchemaSource`, `KubernetesSchemaConverter`, `CRDSchemaEnhancer`, `CRDDocumentBuilder`, and `CRDYAMLRenderer`.

- [ ] **Step 3: Commit failing tests only if requested by reviewer**

Do not commit a red test by default. Leave the test file staged only after implementation passes.

---

### Task 2: Introduce schema source and converter classes

**Files:**
- Modify: `tools/generate_crd.py:74-407`
- Test: `tests/unit/operator/test_crd_generator_pipeline.py`

- [ ] **Step 1: Add `CRDSchemaSource` and `KubernetesSchemaConverter`**

In `tools/generate_crd.py`, after the configuration constants and before `_resolve_ref`, add:

```python
class CRDSchemaSource:
    """Load raw Pydantic schemas used as CRD roots."""

    def config_schema(self) -> dict[str, Any]:
        from aiperf.config.config import AIPerfConfig

        return AIPerfConfig.model_json_schema()

    def job_schema(self) -> dict[str, Any]:
        from aiperf.operator.models import AIPerfJobSpec

        return AIPerfJobSpec.model_json_schema(mode="validation", by_alias=True)

    def sweep_schema(self) -> dict[str, Any]:
        from aiperf.operator.models import AIPerfSweepSpec

        return AIPerfSweepSpec.model_json_schema(mode="validation", by_alias=True)
```

After `convert_aiperf_config_fields`, add:

```python
class KubernetesSchemaConverter:
    """Convert Pydantic JSON Schema nodes into Kubernetes OpenAPI nodes."""

    def schema_node(
        self,
        schema: dict[str, Any],
        defs: dict[str, Any],
        depth: int = 0,
    ) -> dict[str, Any]:
        return _convert_schema(schema, defs, depth)

    def aiperf_config_fields(
        self, schema: dict[str, Any], *, verbose: bool = False
    ) -> dict[str, Any]:
        return convert_aiperf_config_fields(schema, verbose=verbose)
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run pytest -n auto tests/unit/operator/test_crd_generator_pipeline.py::test_crd_schema_source_loads_job_and_sweep_schema_roots tests/unit/operator/test_crd_generator_pipeline.py::test_kubernetes_schema_converter_preserves_existing_top_level_fields
```

Expected: first two tests PASS; later tests may still fail because other classes are not defined.

---

### Task 3: Introduce enhancer and document builder classes

**Files:**
- Modify: `tools/generate_crd.py:410-1328`
- Test: `tests/unit/operator/test_crd_generator_pipeline.py`
- Test: `tests/unit/operator/test_aiperfsweep_crd_generation.py`

- [ ] **Step 1: Add `CRDSchemaEnhancer`**

After `_walk_dict_apply`, add:

```python
class CRDSchemaEnhancer:
    """Apply AIPerf-specific CRD schema decorations."""

    def decorate_all(self, node: dict[str, Any]) -> None:
        _decorate_all(node)

    def ensure_type_on_preserve_unknown(self, node: dict[str, Any]) -> None:
        _ensure_type_on_preserve_unknown(node)

    def strip_internal_sentinels(self, node: dict[str, Any]) -> None:
        _strip_mixed_union_sentinels(node)
```

- [ ] **Step 2: Add `CRDDocumentBuilder`**

After `build_aiperfsweep_crd`, add:

```python
class CRDDocumentBuilder:
    """Build complete Kubernetes CRD documents."""

    def __init__(
        self,
        *,
        converter: KubernetesSchemaConverter | None = None,
        enhancer: CRDSchemaEnhancer | None = None,
    ) -> None:
        self.converter = converter or KubernetesSchemaConverter()
        self.enhancer = enhancer or CRDSchemaEnhancer()

    def aiperfjob_crd(self, _job_schema: dict[str, Any]) -> dict[str, Any]:
        config_schema = CRDSchemaSource().config_schema()
        config_properties = self.converter.aiperf_config_fields(config_schema)
        return _build_crd(config_properties)

    def aiperfsweep_crd(self, _sweep_schema: dict[str, Any]) -> dict[str, Any]:
        return build_aiperfsweep_crd()
```

This first version is intentionally a compatibility façade. It exposes the pipeline seams while preserving existing pure-function behavior.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest -n auto tests/unit/operator/test_crd_generator_pipeline.py::test_crd_schema_enhancer_keeps_sweep_kind_rules tests/unit/operator/test_aiperfsweep_crd_generation.py
```

Expected: PASS.

---

### Task 4: Introduce renderer class and wire `CRDGenerator`

**Files:**
- Modify: `tools/generate_crd.py:1333-1532`
- Test: `tests/unit/operator/test_crd_generator_pipeline.py`

- [ ] **Step 1: Add `CRDYAMLRenderer`**

After `render_helm_sweep_crd_yaml`, add:

```python
class CRDYAMLRenderer:
    """Render CRD documents into Helm-safe YAML."""

    def aiperfjob_yaml(self, crd: dict[str, Any]) -> str:
        return render_helm_crd_yaml(crd)

    def aiperfsweep_yaml(self, crd: dict[str, Any]) -> str:
        return render_helm_sweep_crd_yaml(crd)
```

- [ ] **Step 2: Rewrite `CRDGenerator.generate()` as a thin pipeline**

Replace the body of `CRDGenerator.generate()` with:

```python
    def generate(self) -> GeneratorResult:
        sys.path.insert(0, "src")

        source = CRDSchemaSource()
        converter = KubernetesSchemaConverter()
        enhancer = CRDSchemaEnhancer()
        builder = CRDDocumentBuilder(converter=converter, enhancer=enhancer)
        renderer = CRDYAMLRenderer()

        config_schema = source.config_schema()
        if self.verbose:
            defs = config_schema.get("$defs", {})
            props = config_schema.get("properties", {})
            print_step(
                f"JSON Schema: {len(defs)} definitions, {len(props)} top-level properties"
            )

        job_crd = builder.aiperfjob_crd(source.job_schema())
        sweep_crd = builder.aiperfsweep_crd(source.sweep_schema())

        version = _get_project_version()
        chart_yaml = _sync_chart_app_version(version)

        config_properties = converter.aiperf_config_fields(config_schema)
        field_count = len(config_properties)
        return GeneratorResult(
            files=[
                GeneratedFile(HELM_CRD_FILE, renderer.aiperfjob_yaml(job_crd)),
                GeneratedFile(HELM_SWEEP_CRD_FILE, renderer.aiperfsweep_yaml(sweep_crd)),
                GeneratedFile(HELM_CHART_FILE, chart_yaml),
            ],
            summary=f"CRD with {field_count} AIPerfConfig fields + AIPerfSweep CRD",
        )
```

- [ ] **Step 3: Run pipeline tests**

Run:

```bash
uv run pytest -n auto tests/unit/operator/test_crd_generator_pipeline.py
```

Expected: PASS.

---

### Task 5: Add compatibility assertions to existing CRD tests

**Files:**
- Modify: `tests/unit/operator/test_aiperfsweep_crd_generation.py`
- Test: `tests/unit/operator/test_aiperfsweep_crd_generation.py`

- [ ] **Step 1: Add tests for wrapper parity**

Append these tests to `tests/unit/operator/test_aiperfsweep_crd_generation.py`:

```python

def test_crd_document_builder_matches_legacy_job_builder() -> None:
    from aiperf.config.config import AIPerfConfig
    from tools.generate_crd import (
        CRDDocumentBuilder,
        CRDSchemaSource,
        convert_aiperf_config_fields,
        _build_crd,
    )

    source = CRDSchemaSource()
    legacy = _build_crd(convert_aiperf_config_fields(AIPerfConfig.model_json_schema()))
    refactored = CRDDocumentBuilder().aiperfjob_crd(source.job_schema())

    assert refactored == legacy


def test_crd_document_builder_matches_legacy_sweep_builder() -> None:
    from tools.generate_crd import CRDDocumentBuilder, CRDSchemaSource, build_aiperfsweep_crd

    source = CRDSchemaSource()

    assert CRDDocumentBuilder().aiperfsweep_crd(source.sweep_schema()) == build_aiperfsweep_crd()
```

- [ ] **Step 2: Run compatibility tests**

Run:

```bash
uv run pytest -n auto tests/unit/operator/test_aiperfsweep_crd_generation.py
```

Expected: PASS.

---

### Task 6: Verify generated output and commit implementation

**Files:**
- Modify: `tools/generate_crd.py`
- Create: `tests/unit/operator/test_crd_generator_pipeline.py`
- Modify: `tests/unit/operator/test_aiperfsweep_crd_generation.py`

- [ ] **Step 1: Run generator check**

Run:

```bash
make check-crd
```

Expected: PASS. If it fails, inspect the diff in `deploy/helm/aiperf-operator/templates/crd-aiperfjob.yaml`, `deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml`, and `deploy/helm/aiperf-operator/Chart.yaml`. The expected refactor should not change generated output.

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run pytest -n auto tests/unit/operator/test_crd_generator_pipeline.py tests/unit/operator/test_aiperfsweep_crd_generation.py
```

Expected: PASS.

- [ ] **Step 3: Run ergonomics and ruff floor checks for touched Python files**

Run:

```bash
make check-ergonomics && make check-ruff-baselined && ruff format tools/generate_crd.py tests/unit/operator/test_crd_generator_pipeline.py tests/unit/operator/test_aiperfsweep_crd_generation.py && ruff check --fix tools/generate_crd.py tests/unit/operator/test_crd_generator_pipeline.py tests/unit/operator/test_aiperfsweep_crd_generation.py
```

Expected: PASS or clean auto-fixes. If ruff changes files, rerun Step 2.

- [ ] **Step 4: Review diff**

Run:

```bash
git diff -- tools/generate_crd.py tests/unit/operator/test_crd_generator_pipeline.py tests/unit/operator/test_aiperfsweep_crd_generation.py
```

Expected: Diff only introduces pipeline classes, rewires `CRDGenerator`, and adds tests. Generated CRD YAML should be unchanged.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add tools/generate_crd.py tests/unit/operator/test_crd_generator_pipeline.py tests/unit/operator/test_aiperfsweep_crd_generation.py
git commit -s -m "$(cat <<'EOF'
refactor(crd): structure generator as schema pipeline

Introduce explicit CRD schema source, converter, enhancer, document builder, and renderer seams while preserving generated Kubernetes manifests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: Commit succeeds without bypassing hooks.
