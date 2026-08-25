---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Config Validation
---

# Config Validation

`aiperf kube validate` performs **client-side** validation of one or more
`AIPerfJob` **or** `AIPerfSweep` YAML files against the full CRD schema, the
Pydantic spec models, and Kubernetes resource-naming rules. It dispatches per
document on the `kind:` field (`validate.py` `SUPPORTED_KINDS = {AIPerfJob,
AIPerfSweep}`), routing to `AIPerfJobSpec` or `AIPerfSweepSpec`. It does not
contact the cluster — making it safe to run in CI, pre-commit hooks, and local
editors.

> **Two layers, same rules.** `aiperf kube validate` runs the same structural
> checks the apiserver enforces at `kubectl apply`. For the catalog of CEL
> `x-kubernetes-validations` rules (shorthand-vs-canonical, mutual exclusion,
> `apiHost ⇒ apiPort`, etc.) see [CRD Validation Rules](crd-validation.md).
> Item-internal Pydantic validators (phase-name uniqueness, phase→dataset
> reference integrity, "seamless not on first phase") run only at apply
> time on the operator side because CEL can't see into opaque
> preserve-unknown array items — the client-side `validate` command runs
> them via Pydantic, so it catches both layers in one pass.

## When to use

| Situation | Tool |
|---|---|
| Validate YAML before `kubectl apply` or `aiperf kube profile` | `aiperf kube validate` (this doc) |
| Check that a *live cluster* can actually schedule the job (quotas, GPU availability, operator health) | `aiperf kube preflight` |
| Confirm the operator is installed and responsive | `aiperf kube list` |

Think of `validate` as the offline static check and `preflight` as the online
dynamic check. Both are cheap; run `validate` on every commit and `preflight`
before the first apply of a new job.

Typical integration points:

- **CI gate** — run `find recipes -name perf.yaml -print0 | xargs -0 aiperf
  kube validate --strict` in a GitHub Action or GitLab job before merging
  changes to benchmark specs.
- **Pre-commit** — catch typos in `spec.benchmark` fields before they reach the
  cluster (see [Integration recipes](#integration-recipes)).
- **IDE / Makefile target** — add `make validate-jobs` so contributors get fast
  feedback without spinning up a cluster.

## CLI reference

```text
aiperf kube validate <files...> [--strict] [--output text|json]
```

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `files` | — | `Path...` (positional, one or more) | — | Paths to `AIPerfJob` or `AIPerfSweep` YAML files. Globs are expanded by the shell. |
| `--strict` | `-s` | bool | `false` | Treat warnings (unknown spec fields) as errors. |
| `--output` | `-o` | `text` \| `json` | `text` | Output format. `text` prints a coloured per-file summary; `json` prints a machine-parseable array. |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | All files passed. Warnings may still be present in non-strict mode. |
| `1` | At least one file failed validation, or an internal error occurred. |

### Examples

```bash
# Validate a single job file
aiperf kube validate aiperfjob.yaml

# Validate selected checked-in recipes
aiperf kube validate recipes/llama-3-70b/vllm/agg/perf.yaml \
  recipes/qwen3-32b/vllm/agg-round-robin/perf.yaml

# Treat unknown spec fields as hard errors (recommended in CI)
aiperf kube validate --strict aiperfjob.yaml

# Machine-parseable output for scripting
aiperf kube validate -o json aiperfjob.yaml | jq '.[] | select(.passed==false)'
```

## What gets validated

`validate` runs the following checks on each file, in order. Structural errors
that make later checks impossible short-circuit the file (remaining checks are
skipped for that file only).

1. **File reachability** — the path exists and is a regular file.
2. **YAML parse** — the document is valid YAML and decodes to a mapping.
3. **Required top-level fields**:
   - `apiVersion` must equal the current operator API version
     (`aiperf.nvidia.com/v1alpha1`).
   - `kind` must be one of `AIPerfJob` or `AIPerfSweep`. The kind selects
     which spec model the document is validated against; an `AIPerfJob` must
     omit `spec.sweep` while an `AIPerfSweep` requires it.
   - `metadata` must be a mapping with a `name` field.
   - `spec` must be a mapping.
   - `spec.benchmark` must be a mapping containing at least one of `models` or
     `endpoint`.
4. **Kubernetes naming** — `metadata.name` must:
   - be at most **253 characters** (`K8S_NAME_MAX_LENGTH`), and
   - match the RFC 1123 subdomain pattern `[a-z0-9]([a-z0-9-]*[a-z0-9])?`
     (lowercase alphanumerics and hyphens; must start and end with an
     alphanumeric).
5. **Unknown field detection** (warning by default, error with `--strict`):
   - **Top-level `spec`** is compared against `KNOWN_SPEC_FIELDS`
     (`validate.py`) — the deployment fields `image`, `imagePullPolicy`,
     `keepFailedPods`, `resourceMode`, `connectionsPerWorker`,
     `timeoutSeconds`, `ttlSecondsAfterFinished`, `resultsTtlDays`, `cancel`,
     `podTemplate`, `scheduling`, `skipEndpointCheck`, `failurePolicy`; the
     envelope fields `schemaVersion`, `sweep`, `multiRun`, `plot`, `variables`,
     `randomSeed`, `noSweepTable`, `childMetadata`; plus the nested `benchmark` block. Stray top-level keys
     often mean a benchmark-config field was placed at `spec.<x>` instead of
     `spec.benchmark.<x>` — the warning message says so explicitly.
   - **`spec.benchmark`** is compared against `CONFIG_FIELDS`
     (`kubernetes/spec_converter.py`): every `BenchmarkConfig` model field, its
     serialization aliases, plus the shorthand keys `model`, `dataset`,
     `warmup`, `profiling`.
6. **`AIPerfConfig` construction** — `spec.benchmark` is fed through
   `AIPerfJobSpecConverter.to_aiperf_config()`, which performs the same env-var
   and Jinja2 expansion as a local CLI file load, then validates the result
   against the Pydantic model. Type, range, and cross-field errors surface
   here.
7. **Endpoint sanity** — at least one model name must be present, and every
   entry in `endpoint.urls` must start with `http://` or `https://`.
8. **Deployment-config extraction** — top-level spec fields are materialised
   into a `DeploymentConfig` via `to_deployment_config()`. Catches malformed
   `podTemplate`, invalid `resourceMode`, bad `scheduling` blocks, etc.
9. **Endpoint credential transport** — credential-bearing endpoint fields
   must have the matching Secret-backed pod environment (`AIPERF_INJECTED_API_KEY`,
   `AIPERF_INJECTED_HEADERS`, or `AIPERF_INJECTED_ENDPOINT_URLS`). Literal
   secrets and plain-value environment variables fail before deployment.
10. **Worker-count calculation** — `calculate_workers()` must return a value
   `>= 1` given the current `concurrency`, `connectionsPerWorker`, and mode.
11. **Kind-specific spec validation** — the Config-v2 envelope is rendered
    before the complete `AIPerfJobSpec` or `AIPerfSweepSpec` check. Raw Jinja
    values therefore validate as their resolved numeric or structured types,
    while unknown variables and invalid rendered values still fail closed.

> Note: `validate` is intentionally conservative about what it considers
> "unknown". Any key in `CONFIG_FIELDS` (every `BenchmarkConfig` field, its
> aliases, and the shorthand keys) is accepted under `spec.benchmark`, so
> newly added config fields do not require a docs update to this page.

## JSON output schema

With `-o json`, a single JSON array is printed to stdout. Each element
corresponds to one input file, in the order given on the command line.

```jsonc
[
  {
    "path": "string",         // filesystem path as provided
    "passed": true,           // bool; true iff errors is empty
    "errors":   ["string..."], // fatal issues (empty when passed=true)
    "warnings": ["string..."]  // non-fatal issues; upgraded to errors under --strict
  }
]
```

### Example — all files pass

```json
[
  {
    "path": "recipes/llama-3-70b/vllm/agg/perf.yaml",
    "passed": true,
    "errors": [],
    "warnings": []
  }
]
```

### Example — multiple errors

```json
[
  {
    "path": "recipes/broken.yaml",
    "passed": false,
    "errors": [
      "kind: expected one of ['AIPerfJob', 'AIPerfSweep'], got 'AIPerfConfig'",
      "metadata.name: 'My_Benchmark' is not a valid Kubernetes resource name (must match [a-z0-9][a-z0-9-]*[a-z0-9])",
      "Unknown spec fields (did you mean to put these under spec.benchmark?): models, endpoint",
      "endpoint.urls: 'localhost:8000' must start with http:// or https://"
    ],
    "warnings": []
  }
]
```

Scripting tip — fail a CI job on any warning, not just errors:

```bash
find recipes -name perf.yaml -print0 \
  | xargs -0 aiperf kube validate -o json \
  | jq -e 'all(.passed and (.warnings | length == 0))'
```

## Integration recipes

### Pre-commit hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: aiperf-kube-validate
      name: aiperf kube validate
      entry: aiperf kube validate --strict
      language: system
      files: ^recipes/.*\.ya?ml$
      pass_filenames: true
```

### GitHub Actions step

```yaml
- name: Validate AIPerfJob specs
  run: |
    uv tool install aiperf
    find recipes -name perf.yaml -print0 | xargs -0 aiperf kube validate --strict
```

In a matrix/monorepo setup, use JSON output to surface a compact report:

```yaml
- name: Validate AIPerfJob specs
  run: |
    find recipes -name perf.yaml -print0 \
      | xargs -0 aiperf kube validate -o json > validation.json
    jq -r '.[] | select(.passed==false) | "::error file=\(.path)::\(.errors[0])"' validation.json
```

### Makefile target

```makefile
.PHONY: validate-jobs
validate-jobs:
	find recipes -name perf.yaml -print0 | xargs -0 aiperf kube validate --strict
```

## See also

- [`production.md`](production.md) — production deployment guide, including
  the recommended CI pipeline.
- [`configuration.md`](configuration.md) — reference for `spec` and
  `spec.benchmark` fields.
- `aiperf kube preflight` — live-cluster counterpart to `validate`.
