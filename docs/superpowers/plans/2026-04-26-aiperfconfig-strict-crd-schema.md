# AIPerfConfig Strict CRD Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace `x-kubernetes-preserve-unknown-fields: true` blanket on `benchmark` in both AIPerfJob and AIPerfSweep CRDs with a fully walked schema. Use `preserve-unknown-fields` only at narrow shorthand-boundary fields (mixed-type unions Kubernetes structural schemas can't express). Strict apiserver validation everywhere else.

**Architecture:** Add Pydantic JSON-schema overrides at each shorthand-accepting field via `model_config["json_schema_extra"]` (or per-field `Field(json_schema_extra=...)`). These overrides emit `x-kubernetes-preserve-unknown-fields: true` for that subtree. Update `tools/generate_crd.py` to walk `AIPerfConfig` instead of stopping at the boundary. Both CRDs (`crd.yaml`, `crd-aiperfsweep.yaml`) regenerate from the same source.

**Tech Stack:** Pydantic v2, Kubernetes structural schema (apiextensions.k8s.io/v1).

**Branch policy:** Continue on `ajc/k8s`. No worktrees.

**Test policy:**
- ONE `uv run pytest -n auto tests/unit/` per task.
- `git commit --no-verify`. Run ruff manually.
- Pass `model="opus"` to subagents.

---

## File Map

**Modified:**
- `src/aiperf/config/models.py` — `ModelsConfig` json_schema_extra (or whatever class hosts the polymorphic field; verify by reading)
- `src/aiperf/config/distributions.py` — `Distribution` (lines 80, 207) json_schema_extra
- `src/aiperf/config/endpoint.py` — `EndpointConfig` `url`/`urls` json_schema_extra (line 283)
- `src/aiperf/config/artifacts.py` — `TelemetryConfig` (line 400), `GPUTelemetryConfig` (line 469) json_schema_extra
- `src/aiperf/config/types.py` — `isl_stddev`/`osl_stddev` shorthand (line 100); these may not need schema changes (just sibling fields)
- `src/aiperf/config/config.py` — Add `model`, `dataset`, `warmup`, `profiling` as schema-visible siblings on `AIPerfConfig` for top-level shorthand handling
- `src/aiperf/config/dataset.py` — Add `isl`, `osl` as schema-visible siblings on `SyntheticDataset` (verify class name when reading)
- `tools/generate_crd.py` — Replace the hand-coded blanket `benchmark: {x-kubernetes-preserve-unknown-fields: true}` (line 528-538) with a real walk of `AIPerfConfig`
- `deploy/helm/aiperf-operator/templates/crd.yaml` — regenerated
- `deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml` — regenerated
- `tests/unit/operator/test_aiperfsweep_crd_generation.py` — update assertions
- `tests/unit/operator/` — any other CRD generation tests

**New:**
- `tests/unit/operator/test_crd_round_trip.py` — round-trip every documented YAML through Pydantic JSON-schema-style validation against the generated CRD schema

---

## Task 4: Field-level shorthand JSON-schema overrides

**Goal:** Add `x-kubernetes-preserve-unknown-fields: true` JSON-schema markers at each field that accepts mixed-type shorthand.

**Files:**
- `src/aiperf/config/models.py` (locate the polymorphic `models` field — likely `AIPerfConfig.models: ModelsAdvanced` plus the str/list[str]/object before-validator on its parent)
- `src/aiperf/config/distributions.py:80,207`
- `src/aiperf/config/endpoint.py:283`
- `src/aiperf/config/artifacts.py:400,469`
- Tests: add `tests/unit/config/test_json_schema_shorthand.py`

### Step 4.1: Read

Read in full:
- `src/aiperf/config/_benchmark_normalizers.py` (lines 72-117 — what `models` shorthand accepts; lines 100-117 normalize_benchmark_input top-level shortcuts)
- `src/aiperf/config/distributions.py:65-220` (`FixedDistribution.coerce_scalar` and `WeightedDistribution.inline_weight`)
- `src/aiperf/config/endpoint.py:275-310` (urls/url normalize)
- `src/aiperf/config/artifacts.py:395-490` (telemetry url shorthand)
- `src/aiperf/config/models.py` (full file; locate `ModelsAdvanced` and where it's referenced in AIPerfConfig)
- `src/aiperf/config/config.py:240-340` (AIPerfConfig fields list to confirm `models`/`phases`/`datasets` types)

### Step 4.2: Write the failing test

Create `tests/unit/config/test_json_schema_shorthand.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify shorthand-accepting fields emit x-kubernetes-preserve-unknown-fields in JSON schema."""

from aiperf.config import AIPerfConfig
from aiperf.config.distributions import FixedDistribution
from aiperf.config.endpoint import EndpointConfig


def _walk(schema: dict, path: str) -> dict:
    """Resolve a dotted property path through a JSON schema (handles $ref/defs)."""
    # Implementation hint: walk schema["properties"][k]["properties"][k2]...
    # Resolve "$ref": "#/$defs/Foo" against schema["$defs"]["Foo"].
    # Skip arrays/oneOf for this helper — fail loudly if encountered.
    ...


def test_models_field_marks_preserve_unknown_fields():
    """The models field must emit x-kubernetes-preserve-unknown-fields: true (mixed type union)."""
    schema = AIPerfConfig.model_json_schema()
    models_schema = _walk(schema, "models")
    assert models_schema.get("x-kubernetes-preserve-unknown-fields") is True, (
        f"AIPerfConfig.models is shorthand-polymorphic and must mark "
        f"x-kubernetes-preserve-unknown-fields=true; got {models_schema!r}"
    )


def test_endpoint_urls_field_marks_preserve_unknown_fields():
    """endpoint.urls accepts str | list[str] shorthand — mark preserve-unknown."""
    schema = EndpointConfig.model_json_schema()
    urls_schema = schema["properties"].get("urls", {})
    assert urls_schema.get("x-kubernetes-preserve-unknown-fields") is True


def test_distribution_field_marks_preserve_unknown_fields():
    """Distribution-valued fields accept int | float | object shorthand."""
    # Assert on a representative consumer field, e.g. PromptsConfig.isl
    from aiperf.config.types import PromptsConfig  # adjust if class name differs
    schema = PromptsConfig.model_json_schema()
    isl_schema = schema["properties"].get("isl", {})
    assert isl_schema.get("x-kubernetes-preserve-unknown-fields") is True


def test_telemetry_url_marks_preserve_unknown_fields():
    """TelemetryConfig and GPUTelemetryConfig url fields accept str | object shorthand."""
    from aiperf.config.artifacts import GPUTelemetryConfig, TelemetryConfig
    for cls in (TelemetryConfig, GPUTelemetryConfig):
        schema = cls.model_json_schema()
        url_schema = schema["properties"].get("url", {})
        assert url_schema.get("x-kubernetes-preserve-unknown-fields") is True, (
            f"{cls.__name__}.url must mark preserve-unknown-fields"
        )
```

The implementer must adapt class/field names and the `_walk` helper to the actual model layout discovered in Step 4.1.

### Step 4.3: Run test — confirm failure

`uv run pytest -n auto tests/unit/config/test_json_schema_shorthand.py`

Expected: FAIL.

### Step 4.4: Implement

For each shorthand-accepting field, attach `json_schema_extra={"x-kubernetes-preserve-unknown-fields": True}` to the `Field(...)` call. Example for `models`:

```python
# In whichever class declares the polymorphic models field
models: ModelsAdvanced = Field(
    ...,
    description="...",
    json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
)
```

For class-level overrides (e.g. on `Distribution` — every instance of `Distribution` should emit the marker), use `model_config["json_schema_extra"]`:

```python
class Distribution(BaseConfig):
    model_config = ConfigDict(
        ...,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
    )
```

Adapt to each call site.

**Skip** `Distribution.weight inline_weight` (line 207) — that's an internal canonicalization, not a user-facing shorthand. Document in a code comment why.

### Step 4.5: Format, lint, run

```
ruff format src/aiperf/config/ tests/unit/config/test_json_schema_shorthand.py
ruff check --fix src/aiperf/config/ tests/unit/config/test_json_schema_shorthand.py
uv run pytest -n auto tests/unit/
```

Expected: all green.

### Step 4.6: Commit

```
git add -u
git commit --no-verify -s -m "$(cat <<'EOF'
feat(config): mark shorthand-accepting fields preserve-unknown in JSON schema

Add json_schema_extra={"x-kubernetes-preserve-unknown-fields": True} to
fields that accept mixed-type shorthand (models, distributions, endpoint
urls, telemetry urls). Kubernetes structural schemas can't express
mixed-type unions; this marks the narrow boundary subtrees so apiserver
defers their validation to AIPerfConfig.model_validate.
EOF
)"
```

---

## Task 5: Top-level sibling shortcuts on AIPerfConfig and SyntheticDataset

**Goal:** Make `model`, `dataset`, `warmup`, `profiling` (on `AIPerfConfig`) and `isl`, `osl`, `isl_stddev`, `osl_stddev` (on `SyntheticDataset` / its prompts subtree) schema-visible so apiserver lets them through, while keeping the existing before-validator hoist behavior.

**Strategy:** Add each shortcut as a field on the parent model with `Field(default=None, exclude=True, json_schema_extra={"x-kubernetes-preserve-unknown-fields": True})`. The before-validator already pops them. Setting `exclude=True` keeps `model_dump` clean. Setting `default=None` means they're optional — explicitly absent by default.

**Alternative:** if `exclude=True` doesn't suppress them in dumps the way we want, instead add them via `__get_pydantic_json_schema__` so they appear in the schema but not on the model. Implementer's call based on what works.

**Files:**
- `src/aiperf/config/config.py` (AIPerfConfig)
- `src/aiperf/config/dataset.py` (SyntheticDataset — verify name)

### Step 5.1: Read

- `src/aiperf/config/_benchmark_normalizers.py:51-117` (confirm exactly which top-level keys get hoisted at AIPerfConfig level)
- `src/aiperf/config/_benchmark_normalizers.py:120-194` (confirm which keys get hoisted at SyntheticDataset level — `isl`, `osl`)
- `src/aiperf/config/dataset.py` (locate SyntheticDataset, find `prompts` field)
- `src/aiperf/config/types.py:90-130` (`isl_stddev`/`osl_stddev` — these are siblings of `isl`/`osl` already? confirm)
- `src/aiperf/config/config.py:240-330` (existing fields list to insert siblings in stable order)

### Step 5.2: Write failing tests

Append to `tests/unit/config/test_json_schema_shorthand.py`:

```python
def test_aiperf_config_schema_exposes_top_level_shortcuts():
    """model/dataset/warmup/profiling must appear as optional schema siblings."""
    schema = AIPerfConfig.model_json_schema()
    props = schema["properties"]
    for key in ("model", "dataset", "warmup", "profiling"):
        assert key in props, f"AIPerfConfig schema missing shortcut sibling {key!r}"
        assert props[key].get("x-kubernetes-preserve-unknown-fields") is True, (
            f"shortcut {key!r} must mark preserve-unknown-fields"
        )


def test_aiperf_config_runtime_still_validates_with_shortcut():
    """The before-validator hoist is unchanged — passing model: 'foo' still works."""
    cfg = AIPerfConfig.model_validate(
        {
            "model": "test-model",
            "endpoint": {"type": "chat", "url": "http://x:8000"},
            "phases": [{"name": "profiling", "type": "concurrency", "concurrency": 1}],
            "datasets": [{"name": "default", "type": "synthetic"}],
        }
    )
    assert cfg.models.items[0].name == "test-model"
    # model dump should not include the shortcut key
    dumped = cfg.model_dump(exclude_none=True)
    assert "model" not in dumped


def test_synthetic_dataset_schema_exposes_isl_osl_shortcuts():
    """isl/osl must appear as optional schema siblings on SyntheticDataset."""
    from aiperf.config.dataset import SyntheticDataset  # adjust class name if different
    schema = SyntheticDataset.model_json_schema()
    props = schema["properties"]
    for key in ("isl", "osl"):
        assert key in props, f"SyntheticDataset schema missing shortcut sibling {key!r}"
        assert props[key].get("x-kubernetes-preserve-unknown-fields") is True
```

### Step 5.3: Run test — confirm failure

`uv run pytest -n auto tests/unit/config/test_json_schema_shorthand.py -k "shortcut"`

Expected: FAIL.

### Step 5.4: Implement

On `AIPerfConfig`, add four optional shortcut siblings near the top of the field list (insertion order matters for schema):

```python
model: Annotated[
    Any | None,
    Field(
        default=None,
        exclude=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
        description=(
            "Shorthand for `models`. Accepts a string, list of strings, or "
            "ModelsAdvanced object. Hoisted into `models` by the before-validator."
        ),
    ),
]
dataset: Annotated[
    Any | None,
    Field(
        default=None,
        exclude=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
        description="Shorthand for a single `datasets` entry. Hoisted into `datasets`.",
    ),
]
warmup: Annotated[
    Any | None,
    Field(
        default=None,
        exclude=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
        description="Shorthand for a warmup phase entry. Rolled into `phases`.",
    ),
]
profiling: Annotated[
    Any | None,
    Field(
        default=None,
        exclude=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
        description="Shorthand for a profiling phase entry. Rolled into `phases`.",
    ),
]
```

The before-validator (`normalize_benchmark_input`) already pops these keys, so the model-level field just exists for schema visibility. Confirm the order does not affect the existing CRD on AIPerfJob (those keys aren't currently emitted).

On `SyntheticDataset` (verify class name), do the same for `isl`, `osl`, `isl_stddev`, `osl_stddev` — but only those that are NOT already canonical fields. Check existing field list before adding duplicates.

### Step 5.5: Format, lint, run

```
ruff format src/aiperf/config/config.py src/aiperf/config/dataset.py tests/unit/config/test_json_schema_shorthand.py
ruff check --fix src/aiperf/config/config.py src/aiperf/config/dataset.py tests/unit/config/test_json_schema_shorthand.py
uv run pytest -n auto tests/unit/
```

Expected: all green.

### Step 5.6: Commit

```
git add -u
git commit --no-verify -s -m "$(cat <<'EOF'
feat(config): expose top-level shortcut siblings in JSON schema

Add model/dataset/warmup/profiling as optional schema-visible siblings on
AIPerfConfig and isl/osl on SyntheticDataset. Each marks
x-kubernetes-preserve-unknown-fields=true so apiserver lets them through
to AIPerfConfig.model_validate, where the existing before-validator
hoists them into the canonical fields. exclude=True keeps model_dump
output canonical.
EOF
)"
```

---

## Task 6: Walk AIPerfConfig in CRD generator + round-trip every documented YAML

**Goal:** Replace `tools/generate_crd.py:528-538` (the hand-coded `benchmark: {x-kubernetes-preserve-unknown-fields: true}`) with a real walk of `AIPerfConfig`. Both CRDs (`crd.yaml` and `crd-aiperfsweep.yaml`) regenerate to a strict schema with narrow shorthand boundaries.

**Files:**
- `tools/generate_crd.py:503-554`
- `deploy/helm/aiperf-operator/templates/crd.yaml` (regenerated)
- `deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml` (regenerated)
- `tests/unit/operator/test_aiperfsweep_crd_generation.py` (update assertions)
- `tests/unit/operator/` — search for other CRD generation tests
- New: `tests/unit/operator/test_crd_round_trip_yamls.py`

### Step 6.1: Read

- `tools/generate_crd.py` in full (988 lines — focus on `_resolve_ref`, `_walk_schema`, `_build_crd`, `_deployment_config_properties`)
- `tests/unit/operator/test_aiperfsweep_crd_generation.py` (full file)
- `find tests/unit -name "*crd_gen*" -o -name "*crd_schema*"` to locate any other CRD generation tests

### Step 6.2: Write failing test — round-trip

Create `tests/unit/operator/test_crd_round_trip_yamls.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every documented YAML must round-trip through AIPerfConfig.model_validate.

This guards against the strict CRD schema rejecting tutorial/example YAMLs.
We test against AIPerfConfig directly (not the apiserver) — the CRD schema
is derived from AIPerfConfig, so AIPerfConfig acceptance is necessary
(but not sufficient) for kubectl apply to work.

Files scanned:
    docs/tutorials/*.yaml, docs/tutorials/**/*.yaml
    docs/kubernetes/*.yaml
    examples/**/*.yaml
"""

from pathlib import Path

import pytest
import yaml

from aiperf.config import AIPerfConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def _yaml_files() -> list[Path]:
    roots = [
        REPO_ROOT / "docs" / "tutorials",
        REPO_ROOT / "docs" / "kubernetes",
        REPO_ROOT / "examples",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.yaml"))
            files.extend(root.rglob("*.yml"))
    return sorted(files)


def _extract_aiperf_configs(yaml_path: Path) -> list[tuple[str, dict]]:
    """Return (yaml_id, config_dict) for every AIPerfConfig-shaped block."""
    text = yaml_path.read_text()
    docs = list(yaml.safe_load_all(text))
    out: list[tuple[str, dict]] = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        # AIPerfJob CR: spec.benchmark is the AIPerfConfig
        if doc.get("kind") == "AIPerfJob" and "spec" in doc and "benchmark" in doc["spec"]:
            out.append((f"{yaml_path.name}#{i}.spec.benchmark", doc["spec"]["benchmark"]))
        # AIPerfSweep CR: spec.template.spec.benchmark
        elif doc.get("kind") == "AIPerfSweep":
            template = doc.get("spec", {}).get("template", {}).get("spec", {})
            if "benchmark" in template:
                out.append((f"{yaml_path.name}#{i}.spec.template.spec.benchmark", template["benchmark"]))
        # Bare AIPerfConfig (CLI YAML)
        elif {"models", "model", "endpoint"} & set(doc.keys()):
            out.append((f"{yaml_path.name}#{i}", doc))
    return out


def _all_configs() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for yaml_path in _yaml_files():
        out.extend(_extract_aiperf_configs(yaml_path))
    return out


@pytest.mark.parametrize(
    "yaml_id, config",
    _all_configs(),
    ids=lambda p: p if isinstance(p, str) else "...",
)
def test_documented_yaml_validates_through_aiperf_config(yaml_id: str, config: dict):
    AIPerfConfig.model_validate(config)
```

### Step 6.3: Run test — confirm baseline

`uv run pytest -n auto tests/unit/operator/test_crd_round_trip_yamls.py`

Expected: ALL PASS (the baseline before changing the generator). If any documented YAML fails today, that's a pre-existing bug unrelated to this task — open an issue or skip.

### Step 6.4: Update generator

In `tools/generate_crd.py:528-538`, replace the hand-coded benchmark block:

```python
# OLD
spec_properties["benchmark"] = {
    "type": "object",
    "x-kubernetes-preserve-unknown-fields": True,
    "description": "...",
}
```

With a real walk of `AIPerfConfig` (re-using the existing schema-walking helper, e.g. `_walk_schema(AIPerfConfig.model_json_schema(), defs={})` — adapt to the actual function signature in the file):

```python
# NEW
from aiperf.config import AIPerfConfig

aiperf_config_schema = _walk_schema(
    AIPerfConfig.model_json_schema(mode="validation"),
    defs={},
    depth=0,
)
spec_properties["benchmark"] = {
    "type": "object",
    "description": "Benchmark configuration (AIPerfConfig).",
    **{k: v for k, v in aiperf_config_schema.items() if k != "type"},
}
```

The exact integration depends on the existing helper signatures; the implementer must read the file and adapt. Key invariants:
- The `x-kubernetes-preserve-unknown-fields: true` markers from Tasks 4+5 must propagate into the generated CRD.
- Top-level shortcut siblings must appear as schema properties.
- Strict typing (e.g. `phases.items` are typed as `PhaseConfig` discriminated union) must come through correctly.

### Step 6.5: Regenerate CRDs

```
uv run python tools/generate_crd.py
```

Expected: `crd.yaml` and `crd-aiperfsweep.yaml` both regenerate with full benchmark schema.

### Step 6.6: Verify Kubernetes structural schema rules

For each `oneOf`/`anyOf` block in the regenerated CRDs, check:
1. The branches don't conflict on `type`/`properties`/`additionalProperties` at the parent level.
2. No mixed-type `oneOf` (string vs object vs array at one field) without `x-kubernetes-preserve-unknown-fields: true` covering it.

If `kubectl` is available locally, run `kubectl apply --dry-run=server -f deploy/helm/aiperf-operator/templates/crd.yaml` against any cluster (a `kind` cluster is fine). If not, skip and rely on round-trip tests + Task 7 verification.

### Step 6.7: Update CRD-generation tests

The existing tests in `tests/unit/operator/test_aiperfsweep_crd_generation.py` and any sibling files likely assert on the old `x-kubernetes-preserve-unknown-fields: true` blanket on benchmark. Update those assertions to:
- Confirm `benchmark` is a real walked schema (has `properties` containing `models`, `endpoint`, etc.)
- Confirm the narrow `preserve-unknown-fields` markers are present on the shorthand boundaries

### Step 6.8: Format, lint, run

```
ruff format tools/generate_crd.py tests/unit/operator/
ruff check --fix tools/generate_crd.py tests/unit/operator/
uv run pytest -n auto tests/unit/
```

Expected: all green. Pay attention to:
- `tests/unit/operator/test_crd_round_trip_yamls.py` (baseline must hold)
- `tests/unit/operator/test_aiperfsweep_crd_generation.py` (updated assertions)

### Step 6.9: Verify CRD diff is sane

```
git diff --stat -- deploy/helm/aiperf-operator/templates/
```

Expected: `crd.yaml` grows substantially (the hand-coded benchmark blanket is replaced with real schema). `crd-aiperfsweep.yaml` may shrink or grow depending on whether the in-AIPerfSweep walk previously produced more or less than what it does now (since Tasks 4+5 added `preserve-unknown-fields` markers that effectively prune subtrees).

Spot check by running `git diff -- deploy/helm/aiperf-operator/templates/crd.yaml | wc -l` — if it's above 5000 lines, that's expected (AIPerfConfig is large). Open `crd.yaml` and confirm:
- `benchmark.properties.models` has `x-kubernetes-preserve-unknown-fields: true`
- `benchmark.properties.endpoint.properties.urls` has the marker
- `benchmark.properties.phases.items.<discriminator>` is typed
- `benchmark.properties.runtime` is fully strictly typed (no shorthand)
- Top-level `benchmark.properties.{model, dataset, warmup, profiling}` exist and have the marker

### Step 6.10: Commit

```
git add -u
git commit --no-verify -s -m "$(cat <<'EOF'
feat(crd): walk AIPerfConfig in benchmark schema for both CRDs

Replace the blanket x-kubernetes-preserve-unknown-fields: true on
spec.benchmark with a fully walked AIPerfConfig schema. Apiserver now
strictly validates every benchmark field except the narrow shorthand
boundaries (models, distributions, telemetry urls, top-level
shortcuts) — those keep preserve-unknown-fields: true and defer to
AIPerfConfig.model_validate.

Round-trip test guards every documented YAML in docs/tutorials,
docs/kubernetes, and examples against AIPerfConfig.
EOF
)"
```

---

## Out of Scope

- `kubectl apply --dry-run=server` against a real cluster — left for follow-up integration testing.
- Refactoring the existing AIPerfConfig before-validators into proper field-level normalizers — they work; not load-bearing for this task.
- Tightening the `pod_template` JSON schema beyond what Pydantic emits — separate concern.
