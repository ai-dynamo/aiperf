---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: CRD Validation Rules
---

# CRD Validation Rules

When you `kubectl apply` an `AIPerfJob` or `AIPerfSweep` CR, the Kubernetes
apiserver runs two layers of validation **before** the operator ever sees the
resource:

1. **Structural schema** — types, enums, `minimum`/`maximum`, `required`. If
   this layer rejects, the resource is never persisted.
2. **CEL `x-kubernetes-validations`** — cross-field invariants compiled into
   the CRD. These mirror the Pydantic `@model_validator` rules on
   `AIPerfConfig` and `AIPerfSweepSpec`, but fire at admission time so a bad
   CR is rejected with a clear message before any pod is scheduled.

The CRD that defines both layers is auto-generated from the AIPerfConfig
Pydantic models — see [the dev flow doc](../dev/kubernetes-flow.md#crd-generator)
for how to add new rules.

## Shorthand acceptance

The structural `required` list on `spec.benchmark` is just `[endpoint]`. The
canonical fields `models`, `datasets`, and `phases` each pair with a CEL
OR-rule that also accepts a shorthand sibling. This means **kubectl apply
accepts the CLI-YAML idiom** without rewriting:

```yaml
# Shorthand form — accepted by the apiserver, normalized by the operator's
# before-validator on reconcile.
spec:
  benchmark:
    endpoint:
      urls: ["http://server:8000/v1/chat/completions"]
      type: chat
    model: meta-llama/Llama-3.1-8B-Instruct  # singular, scalar
    dataset:                                 # singular, dict
      type: synthetic
    profiling:                               # phase shorthand
      type: concurrency
```

```yaml
# Canonical form — also accepted; identical post-normalization shape.
spec:
  benchmark:
    endpoint:
      urls: ["http://server:8000/v1/chat/completions"]
      type: chat
    models: [meta-llama/Llama-3.1-8B-Instruct]
    datasets:
    - name: main
      type: synthetic
    phases:
    - name: profiling
      type: concurrency
```

You **cannot** mix the two forms for the same slot — the operator's
`normalize_before_validation` raises a Pydantic ``ValueError`` on reconcile
(`status.phase=Failed` with `set 'datasets' (canonical) OR 'dataset'
(shorthand), not both`). The check can't move to CEL because the shorthand
fields are typeless preserve-unknown siblings — see the "Rules NOT enforced
at apiserver level" table below.

## Rule catalog

Each entry below names the CEL rule, the message users see on rejection, and
the original Python validator the rule mirrors. Rules without a Python
counterpart are new invariants the apiserver enforces at admission.

### Benchmark-level rules (apply to both `AIPerfJob.spec.benchmark` and `AIPerfSweep.spec.benchmark`)

| Tier | Rule | Mirrors |
|---|---|---|
| 2G | `parameterSweepSameSeed=true ⇒ randomSeed` | `validate_sweep_same_seed_requires_seed` |
| 2I | `sweep ⇒ ui ≠ 'dashboard'` | `validate_sweep_no_dashboard_ui` |

### Endpoint-level rules

| Tier | Rule | Mirrors |
|---|---|---|
| 1B | `type='template' ⇒ template` | `_validate_template_required` |
| 1B | `template ⇒ type='template'` | (new at apiserver layer) |
| 2J | `requestContentType='multipart/form-data' ⇒ type='video_generation'` | `_validate_request_content_type` (subset) |
| 4O | `urls.all(u, isURL(u))` | (new at apiserver layer) |

### Runtime-level rules

| Tier | Rule | Mirrors |
|---|---|---|
| 1F | `apiHost ⇒ apiPort` | `_validate_api_host_requires_port` |
| 1F | `workersMin ≤ workers` | (new at apiserver layer) |

### Multi-run rules

| Tier | Rule | Mirrors |
|---|---|---|
| 2H | `convergenceMetric ⇒ mode ≠ 'repeated'` | documented in CLAUDE.md, now enforced |

### Artifacts rules

| Tier | Rule | Mirrors |
|---|---|---|
| 3K | `benchmarkId` immutable after first set | (new at apiserver layer) |

### AIPerfJob spec-level rules

| Tier | Rule | Mirrors |
|---|---|---|
| 3N | `scheduling.queueName` immutable after first set | Kueue contract |

### AIPerfSweep top-level rules

| Tier | Rule | Mirrors |
|---|---|---|
| 1C | `has(sweep) || has(multiRun) || has(convergence)` | `_validate_axis_combination` |
| 1C | `convergence ⇒ multiRun` | `_validate_axis_combination` |
| 1C | `convergence ⇒ multiRun.trials unset` | `_validate_axis_combination` |
| — | `sweep` immutable after creation | (existing) |
| — | `multiRun` immutable after creation | (existing) |
| — | `convergence` immutable after creation | (existing) |

### AIPerfSweep template-level rules

| Tier | Rule | Mirrors |
|---|---|---|
| 1D | `!has(self.sweep)` (on `spec.benchmark`) | `_validate_axis_combination` (forbidden_attrs) |
| 1D | `!has(self.multiRun)` (on `spec.benchmark`) | `_validate_axis_combination` (forbidden_attrs) |

### AIPerfSweep convergence rules

| Tier | Rule | Mirrors |
|---|---|---|
| 1E | `minRuns ≤ maxRuns` | `_validate_run_bounds` |

## Rules NOT enforced at apiserver level

Some Pydantic validators **cannot** be moved to CEL because the array items
they reference are emitted as opaque `x-kubernetes-preserve-unknown-fields`
blobs (the `phases[]` and `datasets[]` items are heterogeneous Pydantic
discriminated unions). These stay in the operator's `@model_validator`
decorators and surface only on reconcile:

| Python validator | Why CEL can't see it |
|---|---|
| `normalize_before_validation` (shorthand-or-canonical OR-requirement) | needs `has(self.model)` etc; shorthand siblings are typeless preserve-unknown fields |
| `normalize_before_validation` (shorthand-and-canonical mutual exclusion) | same — typeless preserve-unknown fields |
| `validate_phase_names_unique` | needs `phases[].name`; items are opaque |
| `validate_datasets_unique_names` | needs `datasets[].name`; items are opaque |
| `validate_dataset_references` | needs `phases[].dataset` and `datasets[].name` |
| `validate_seamless_not_on_first_phase` | needs `phases[0].seamless` |
| `validate_phase_dataset_compatibility` | walks plugin-provided dataset/phase metadata |
| `validate_prefill_requires_streaming` | needs `phases[].prefill_concurrency` |

If you submit a CR that the apiserver accepts but the operator later rejects,
the failure shows up as `status.phase=Failed` with the validation error in
`status.error` (or in operator pod logs).

## Example error messages

Each CEL rejection names the rule that fired, so the failure points directly
at what to fix.

```text
$ kubectl apply -f bad.yaml
The AIPerfJob "x" is invalid: spec.benchmark: Invalid value: "object":
  set 'datasets' (canonical) OR 'dataset' (shorthand), not both

$ kubectl apply -f bad-template.yaml
The AIPerfJob "x" is invalid: spec.benchmark.endpoint: Invalid value: "object":
  endpoint.template is required when endpoint.type='template'

$ kubectl apply -f no-axis-sweep.yaml
The AIPerfSweep "x" is invalid: spec: Invalid value: "object":
  AIPerfSweep requires at least one of sweep/multiRun/convergence;
  for a single benchmark use AIPerfJob via `aiperf kube profile`

$ kubectl apply -f sweep-in-template.yaml
The AIPerfSweep "x" is invalid: spec.benchmark: Invalid value: "object":
  benchmark.sweep is forbidden — set spec.sweep at the
  AIPerfSweep top level instead
```

## Extending the rule set

Adding a new CEL rule is a small change in `tools/generate_crd.py`:

1. Decide which **shape** the rule applies to (benchmark, endpoint, runtime,
   multiRun, artifacts) and pick the matching `_decorate_*_node` helper, or
   add a new shape detector if your target node has a unique property
   fingerprint.
2. Append a `{"rule": ..., "message": ...}` entry to that helper's
   `_add_validation_rules(...)` call.
3. Add a structural assertion to
   `tests/unit/operator/test_aiperfsweep_crd_generation.py`.
4. Regenerate (`uv run python tools/generate_crd.py`) and verify the regen
   is idempotent (`tools/generate_crd.py --check`).
5. Round-trip against a real apiserver (kind cluster + `kubectl apply
   --dry-run=server`) — the CEL compiler runs at CRD-install time and will
   reject rules that reference undeclared fields or opaque items.

CEL constraints that aren't obvious from the Pydantic side:

- `has(self.X)` only works on properties that are **declared** in the
  schema. Properties under `x-kubernetes-preserve-unknown-fields` are
  invisible to CEL.
- Array items emitted as opaque preserve-unknown blobs cannot be
  dereferenced (no `phases[].name`, no `datasets[0].seamless`).
- `oldSelf` is only available in transition rules and triggers on update.
  Use `!has(oldSelf.X) || oldSelf.X == self.X` for "first-set freezes"
  semantics.
- The K8s apiserver compiles CEL at CRD install time; rule errors fail
  the install with a clear `compilation failed: undefined field 'X'`
  message.

## See also

- [`docs/dev/kubernetes-flow.md`](../dev/kubernetes-flow.md) — operator/CR
  lifecycle, including how the CRD generator decorator pattern is wired.
- [`docs/kubernetes/validate.md`](validate.md) — `aiperf kube validate` runs
  the same schema check **client-side** so CI catches violations before
  `kubectl apply`.
- [`docs/kubernetes/configuration.md`](configuration.md) — full CR-field
  reference.
