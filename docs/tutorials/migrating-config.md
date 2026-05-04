<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Migrating to the envelope config shape

AIPerf YAML configs use an envelope shape that separates sweep machinery from the swept benchmark body. If you are migrating from a pre-envelope flat config, follow this guide.

## What changed

**Before (flat):**

```yaml
models: [llama]
endpoint:
  urls: ["http://localhost:8000/v1/chat/completions"]
datasets:
  - {name: main, type: synthetic, entries: 200}
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
random_seed: 42
sweep:
  type: grid
  variables:
    "phases.profiling.concurrency": [1, 2, 4]
```

**After (envelope):**

```yaml
random_seed: 42
sweep:
  type: grid
  variables:
    "benchmark.phases.profiling.concurrency": [1, 2, 4]
benchmark:
  models: [llama]
  endpoint:
    urls: ["http://localhost:8000/v1/chat/completions"]
  datasets:
    - {name: main, type: synthetic, entries: 200}
  phases:
    - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
```

## Run the migration script

```bash
uv run python tools/migrate_config_yaml.py path/to/config.yaml --in-place
```

The script:
- Re-indents body fields under a top-level `benchmark:` key.
- Keeps envelope keys (`sweep`, `multi_run`, `variables`, `random_seed`) at top level.
- Prefixes grid `sweep.variables` keys with `benchmark.` (e.g. `phases.profiling.concurrency` -> `benchmark.phases.profiling.concurrency`).
- Wraps body fields inside `sweep.runs[i]` under a per-run `benchmark:` key.
- Preserves comments via ruamel.yaml.
- Idempotent — running it twice is a no-op.

## Body fields (move under `benchmark:`)

`models`, `endpoint`, `datasets`, `phases`, `artifacts`, `slos`, `tokenizer`, `gpu_telemetry`, `server_metrics`, `runtime`, `logging`, `metrics`, `accuracy`.

## Envelope fields (stay at top level)

`sweep`, `multi_run`, `variables`, `random_seed`.

## Templates and Jinja

Templates that referenced body keys without a prefix continue to work:

```yaml
variables:
  rate: 100
benchmark:
  phases:
    - name: profiling
      type: rate
      rate: "{{ rate }}"  # still works
```

The loader aliases body keys at the top level of the Jinja context. You can also use the explicit `{{ benchmark.phases.profiling.rate }}` form.

## Why the change

The envelope shape mirrors how AIPerfSweep CRDs are structured on the K8s side: cross-variation machinery (sweep, multi_run, variables, random_seed) at envelope level; the swept benchmark workload as the body. This eliminates a long-standing asymmetry between the YAML and CRD surfaces, and makes scenario merge logic trivial — only the `benchmark:` subtree merges per variation; envelope fields are constant across variations.

## Common gotchas

- **Scenario `runs[i]` body keys must be wrapped under `benchmark:`.** A run carrying `phases:` directly raises `unknown field 'phases' in sweep run [0]; allowed: name, variables, benchmark`. Use `runs: [{benchmark: {phases: [...]}}]`.
- **Grid `sweep.variables` paths must be envelope-rooted.** `phases.profiling.rate: [...]` raises `non-sweepable subtree`. Use `benchmark.phases.profiling.rate`.
- **`AIPerfJob` CRDs no longer carry sweep blocks inside `spec.benchmark`.** For sweeps on Kubernetes, use `AIPerfSweep`.
