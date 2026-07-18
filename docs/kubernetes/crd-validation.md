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
Pydantic models — see the dev flow doc
for how to add new rules.

## Shorthand acceptance

The structural `required` list on `spec.benchmark` is just `[endpoint]`, and
`spec.benchmark` carries an **empty** `x-kubernetes-validations: []` block — the
canonical fields (`models`, `datasets`, `phases`) and their shorthand siblings
(`model`, `dataset`, `warmup`, `profiling`) are all typeless
preserve-unknown fields, so no CEL rule requires or excludes any of them. The
apiserver accepts either idiom purely because none is `required` and unknown
keys are preserved; the operator's before-validator does the actual
shorthand↔canonical normalization on reconcile. This means **kubectl apply
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
(`status.phase=Failed` with `'dataset' cannot be used with 'datasets'. Use
'dataset' for a single dataset or 'datasets' for multiple named datasets.`).
The check can't move to CEL because the shorthand fields are typeless
preserve-unknown siblings — see the "Rules NOT enforced at apiserver level"
table below.

## Rule catalog

The tables below list **only the CEL rules `tools/generate_crd.py` actually
emits** (verified against the generated `crd-aiperfjob.yaml` /
`crd-aiperfsweep.yaml`). Each entry gives the verbatim CEL expression and the
message users see on rejection. Cross-field invariants that CEL cannot express
are enforced by Pydantic on the operator side instead — see
[Operator-side (Pydantic) invariants](#operator-side-pydantic-invariants) below.

### Endpoint rules — `spec.benchmark.endpoint` (both kinds)

| CEL rule | Message |
|---|---|
| `!has(self.type) \|\| self.type != 'template' \|\| has(self.template)` | `endpoint.template is required when endpoint.type='template'` |
| `!has(self.template) \|\| (has(self.type) && self.type == 'template')` | `endpoint.template is only used when endpoint.type='template'` |
| `!has(self.requestContentType) \|\| self.requestContentType != 'multipart/form-data' \|\| (has(self.type) && self.type == 'video_generation')` | `requestContentType='multipart/form-data' is only supported on endpoint.type='video_generation' today` |
| `!has(self.path) \|\| self.path.startsWith('/')` | `endpoint.path must start with '/' (e.g. '/v1/chat/completions', not 'v1/chat/completions')` |

### Runtime rules — `spec.benchmark.runtime` (both kinds)

| CEL rule | Message |
|---|---|
| `!has(self.apiHost) \|\| has(self.apiPort)` | `runtime.apiHost requires runtime.apiPort to be set` |
| `!has(self.workersMin) \|\| !has(self.workers) \|\| self.workersMin <= self.workers` | `runtime.workersMin must be <= runtime.workers` |

### Artifacts rules — `spec.benchmark.artifacts` (both kinds)

| CEL rule | Message |
|---|---|
| `!has(oldSelf.benchmarkId) \|\| oldSelf.benchmarkId == self.benchmarkId` | `artifacts.benchmarkId is immutable once set; mutating it would orphan artifacts already keyed by the old id` |

### AIPerfJob spec-level rules

| CEL rule | Path | Message |
|---|---|---|
| `!has(self.sweep)` | `spec` | `AIPerfJob.spec.sweep must be null/omitted. Use kind: AIPerfSweep for parameter sweeps.` |
| `!has(oldSelf.queueName) \|\| oldSelf.queueName == self.queueName` | `spec.scheduling` | `scheduling.queueName is immutable once set (Kueue treats queueName as immutable after admission)` |

### AIPerfSweep spec-level rules

| CEL rule | Path | Message |
|---|---|---|
| `has(self.sweep)` | `spec` | `AIPerfSweep.spec.sweep is required. Use kind: AIPerfJob for single benchmarks.` |
| `oldSelf == self` | `spec.sweep` | `spec.sweep is immutable after creation` |
| `oldSelf == self` | `spec.multiRun` | `spec.multiRun is immutable after creation` |

> The AIPerfSweep `spec.scheduling` immutability rule and the `spec.sweep`
> absence rule are **AIPerfJob-only**; the AIPerfSweep CRD instead asserts
> `spec.sweep` is present. `spec.benchmark` carries an empty
> `x-kubernetes-validations: []` on both kinds.

## Operator-side (Pydantic) invariants

Several cross-field invariants **cannot** be expressed in CEL — either the
fields they reference are typeless preserve-unknown siblings (`model`,
`dataset`, `warmup`, `profiling`), or the `phases[]` / `datasets[]` array items
are emitted as opaque `x-kubernetes-preserve-unknown-fields` blobs (they are
heterogeneous Pydantic discriminated unions). CEL `has(self.X)` won't compile
against a typeless field, and opaque array items cannot be dereferenced. These
checks stay in the operator's `@model_validator` decorators and surface only on
reconcile (they are also run client-side by `aiperf kube validate`):

| Python validator (`src/aiperf/config/config.py` unless noted) | What it enforces |
|---|---|
| `normalize_before_validation` (via `normalizers._check_mutual_exclusivity`) | shorthand↔canonical mutual exclusion (`model`/`models`, `dataset`/`datasets`, `warmup`+`profiling`/`phases`) |
| `parse_datasets` / `parse_datasets_input` (`loader/normalizers.py`) | each `datasets[]` entry is a mapping with a required `name` |
| `validate_phase_names_unique` | duplicate phase names within `phases` |
| `validate_profiling_phase_required` | at least one non-warmup profiling phase |
| `validate_seamless_not_on_first_phase` | first phase may not set `seamless=true` |
| `validate_phase_dataset_compatibility` | each phase is compatible with its resolved dataset |
| `validate_prefill_requires_streaming` | a phase with `prefill_concurrency` requires `endpoint.streaming=true` |
| `validate_sweep_no_dashboard_ui` | `sweep` set ⇒ `runtime.ui` is not `dashboard` |
| `_apply_consistent_seed_default` | auto-fills a consistent random seed for cross-trial sweeps when none is given |
| `_require_sweep_on_aiperfsweep` (`operator/models.py`) | AIPerfSweep requires a `sweep` block (mirrors the `has(self.sweep)` CEL rule) |
| `_reject_non_finite_sweep_knobs` (`operator/models.py`) | rejects NaN/inf on sweep tuning knobs |

> There is no `validate_datasets_unique_names` or `validate_dataset_references`
> validator — dataset-name presence is checked by `parse_datasets_input` and
> phase↔dataset coupling by `validate_phase_dataset_compatibility`.

If you submit a CR that the apiserver accepts but the operator later rejects,
the failure shows up as `status.phase=Failed` with the validation error in
`status.error` (or in operator pod logs).

## Example error messages

Each CEL rejection names the rule that fired, so the failure points directly
at what to fix.

```text
$ kubectl apply -f bad-template.yaml
The AIPerfJob "x" is invalid: spec.benchmark.endpoint: Invalid value: "object":
  endpoint.template is required when endpoint.type='template'

$ kubectl apply -f sweep-as-job.yaml
The AIPerfJob "x" is invalid: spec: Invalid value: "object":
  AIPerfJob.spec.sweep must be null/omitted. Use kind: AIPerfSweep for
  parameter sweeps.

$ kubectl apply -f job-as-sweep.yaml
The AIPerfSweep "x" is invalid: spec: Invalid value: "object":
  AIPerfSweep.spec.sweep is required. Use kind: AIPerfJob for single
  benchmarks.
```

The shorthand↔canonical mutual-exclusion error is **not** a CEL rejection at
`kubectl apply`; it surfaces on reconcile as `status.phase=Failed` with the
Pydantic message `'dataset' cannot be used with 'datasets'. Use 'dataset' for
a single dataset or 'datasets' for multiple named datasets.`

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

- `docs/dev/kubernetes-flow.md` — operator/CR
  lifecycle, including how the CRD generator decorator pattern is wired.
- [`docs/kubernetes/validate.md`](validate.md) — `aiperf kube validate` runs
  the same schema check **client-side** so CI catches violations before
  `kubectl apply`.
- [`docs/kubernetes/configuration.md`](configuration.md) — full CR-field
  reference.
