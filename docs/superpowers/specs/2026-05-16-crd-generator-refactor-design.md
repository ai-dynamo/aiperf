# CRD Generator Refactor Design

## Goal

Refactor `tools/generate_crd.py` into a generator pipeline shaped like `tools/generate_config_schema.py` while preserving the current CRD output. The refactor keeps `AIPerfJobSpec` and `AIPerfSweepSpec` as the schema roots because those models add Kubernetes-only fields around the shared benchmark config.

## Non-goals

- Do not generate CRDs from `src/aiperf/config/schema/aiperf-config.schema.json`.
- Do not remove Kubernetes-only spec fields such as image, pod template, scheduling, failure policy, cancel, TTL, or child metadata.
- Do not intentionally change the rendered CRDs unless an existing generator bug is found and called out separately.
- Do not reintroduce unsupported multi-dataset or named phase behavior.

## Architecture

`CRDGenerator` remains the `tools._core.Generator` entry point. Its `generate()` method becomes a thin orchestration layer:

1. Load raw Pydantic JSON schemas from `AIPerfJobSpec` and `AIPerfSweepSpec`.
2. Apply CRD-specific schema enhancement passes.
3. Convert JSON Schema nodes into Kubernetes structural OpenAPI nodes.
4. Attach CRD document metadata, status schemas, printer columns, subresources, and CEL rules.
5. Render deterministic Helm YAML and sync `Chart.yaml` `appVersion` from `pyproject.toml`.

## Components

### `CRDSchemaSource`

Imports the operator models and returns raw Pydantic JSON schemas for `AIPerfJobSpec` and `AIPerfSweepSpec`. This isolates import-time errors and gives the generator one source for schema roots.

### `KubernetesSchemaConverter`

Owns generic JSON Schema to Kubernetes OpenAPI conversion:

- `$ref` resolution
- nullable handling
- `anyOf` / `oneOf` conversion
- preserve-unknown behavior
- unsupported-key stripping
- recursion-depth fallback
- deterministic property conversion

### `CRDSchemaEnhancer`

Owns AIPerf-specific CRD adjustments:

- CEL validation attachment
- kind-specific `spec.sweep` rules
- shorthand and preserve-unknown boundaries
- required-field adjustments
- config-port compatibility details such as `plot` and `noSweepTable`

### `CRDDocumentBuilder`

Builds the complete CRD documents around converted spec schemas:

- API group/version/kind/plural
- names and categories
- status schema
- terminal and readiness conditions
- additional printer columns
- subresources

### `CRDYAMLRenderer`

Owns Helm-safe YAML rendering:

- SPDX header
- deterministic key ordering where required
- multiline string handling
- Kubernetes-friendly boolean/null formatting

## Compatibility Requirements

The first implementation should be output-preserving. Generated files should remain byte-for-byte identical unless the refactor exposes a real bug that is intentionally fixed.

The generator must continue to:

- derive CRDs from `AIPerfJobSpec` and `AIPerfSweepSpec`
- include Kubernetes-only deployment and orchestration fields
- keep `benchmark` aligned with the current config models
- preserve opaque boundaries where Kubernetes cannot inspect union internals
- enforce kind-specific `spec.sweep` absence/presence rules
- sync Helm `appVersion` from `pyproject.toml`

## Testing

Add or update tests to cover:

- `CRDGenerator().generate()` emits the CRD and chart files expected by the workflow.
- Makefile and pre-commit wiring still run `tools.generate_crd --check`.
- Refactored converter/enhancer units preserve known tricky behavior: nullable fields, preserve-unknown fields, mixed unions, and CEL attachment.
- Generated CRDs retain Kubernetes-only fields from `AIPerfJobSpec` / `AIPerfSweepSpec`.
- Generated CRDs retain config-port fields such as `plot` and `noSweepTable`.
- Existing generated CRDs remain byte-stable or semantically equivalent after the refactor.

## Rollout

1. Refactor internals without changing public command usage: `python -m tools.generate_crd [--check|--verbose]`.
2. Run the focused generator tests.
3. Run `make check-crd` to verify generated files are current.
4. If generated output changes, inspect and classify each change before accepting it.
