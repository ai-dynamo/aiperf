<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Configuration Schema

This directory contains the JSON Schema for AIPerf YAML configuration files.

## Files

- `aiperf-config.schema.json` - JSON Schema (draft 2020-12) for `AIPerfConfig` YAML files. Schema version `2.0.0`, paired with `AIPerfConfig.schema_version = "2.0"` in `src/aiperf/config/config.py`.

## IDE Integration

### VSCode

Add this to your YAML file header for automatic schema validation and autocompletion. The path is relative to the YAML file's location:

```yaml
# yaml-language-server: $schema=https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/config/schema/aiperf-config.schema.json
```

(That is the form used by the templates in `src/aiperf/config/templates/*.yaml`.)

Or configure workspace settings in `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./src/aiperf/config/schema/aiperf-config.schema.json": [
      "src/aiperf/config/templates/*.yaml",
      "**/aiperf-config.yaml",
      "**/benchmark.yaml"
    ]
  }
}
```

### IntelliJ / PyCharm

1. Open Settings → Languages & Frameworks → Schemas and DTDs → JSON Schema Mappings
2. Add a new mapping:
   - Schema file: `src/aiperf/config/schema/aiperf-config.schema.json`
   - File path pattern: `aiperf-config.yaml` or `**/templates/*.yaml` — avoid a bare `*.yaml` pattern, which will incorrectly validate unrelated YAML files (CI configs, helm values, etc.) against the AIPerf schema.

## Maintenance

The schema in `aiperf-config.schema.json` is generated from the `AIPerfConfig`/`BenchmarkConfig` Pydantic models by `tools/generate_config_schema.py`. When config models change, regenerate it with:

```bash
make generate-config-schema
```

Before committing config-model or generator changes, verify the checked-in schema is current with:

```bash
make check-config-schema
```

Treat the Pydantic models as the source of truth — if the generated JSON file and the models disagree, update the generator or regenerate the schema so they match.

## Schema Features

The schema includes:

- **Discriminated unions** with `oneOf` and a `type` discriminator for:
  - Dataset types: `synthetic`, `file`, `public` (under `benchmark.datasets[]`; `composed` is roadmap-only and not accepted by the runtime today)
  - Phase types: `concurrency`, `poisson`, `gamma`, `constant`, `user_centric`, `fixed_schedule` (under `benchmark.phases[]`)
  - Communication types: `ipc`, `tcp`, `dual` (under `benchmark.runtime.communication`)
  - Sweep types: `grid`, `scenarios`, `adaptive_search` (under top-level `sweep`)
- **Full descriptions** propagated from Pydantic `Field(description=...)`.
- **Constraints** — minimum/maximum, regex patterns, and enum values are enforced.
- **Required fields** — clearly marked at every level.

## Example Configuration

A minimal valid AIPerf YAML (the same shape as `src/aiperf/config/templates/minimal.yaml`):

```yaml
# yaml-language-server: $schema=https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/config/schema/aiperf-config.schema.json

benchmark:
  models:
    - meta-llama/Llama-3.1-8B-Instruct

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    streaming: true

  datasets:
    - name: main
      type: synthetic
      entries: 1000
      prompts:
        isl: 512
        osl: 128

  phases:
    - name: profiling
      type: concurrency
      requests: 1000
      concurrency: 32
```

Notes:

- `benchmark:` is the envelope around the workload; `models`/`endpoint`/`datasets`/`phases` live under it, not at the YAML root. Top-level keys are `schemaVersion`, `benchmark`, `sweep`, `multiRun`, `variables`, `randomSeed`.
- `entries` (not `count`) is the field name on synthetic datasets.
- Each `phases[]` entry needs an explicit `type:` discriminator.
- Shorthand forms exist: `model:` → `models:`, singular `dataset:` → one-entry `datasets:`, flat `phases: { type: ..., ... }` → one-entry `phases:` list. See `templates/minimal.yaml` for the shorthand-only form.
