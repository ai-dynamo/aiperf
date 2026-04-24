<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Configuration Schema

This directory contains the JSON Schema for AIPerf YAML configuration files.

## Files

- `aiperf-config.schema.json` - JSON Schema for config YAML files

## IDE Integration

### VSCode

Add this to your YAML file header for automatic schema validation and autocompletion:

```yaml
# yaml-language-server: $schema=./path/to/aiperf-config.schema.json
```

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
   - File path pattern: `*.yaml` (or more specific patterns)

## Regenerating the Schema

The schema is auto-generated from Pydantic models. To regenerate:

```bash
# Generate schema
python -m tools.generate_config_schema

# Check if schema is up-to-date (CI validation)
python -m tools.generate_config_schema --check

# Verbose output
python -m tools.generate_config_schema --verbose
```

## Schema Features

The generated schema includes:

- **Discriminated Unions**: Properly typed `oneOf` with discriminator mappings for:
  - Dataset types (synthetic, file, public, composed)
  - Phase types (load, user_centric, trace)
  - Communication types (ipc, tcp)
  - Sweep types (grid, scenarios, sequential)
  - Export types (records, summary, raw)

- **Full Descriptions**: All field descriptions from Pydantic `Field(description=...)` are included

- **Constraints**: Minimum/maximum values, patterns, and enums are enforced

- **Required Fields**: Clear indication of which fields are mandatory

## Example Configuration

```yaml
# yaml-language-server: $schema=./src/aiperf/config/schema/aiperf-config.schema.json

models:
  - meta-llama/Llama-3.1-8B-Instruct

endpoint:
  urls:
    - http://localhost:8000/v1/chat/completions
  streaming: true

datasets:
  main:
    type: synthetic
    count: 1000
    prompts:
      isl: 512
      osl: 128

phases:
  - name: profiling
    requests: 1000
    concurrency: 32
```
