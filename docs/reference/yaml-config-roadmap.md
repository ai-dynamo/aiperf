---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: YAML Config Roadmap
---

# YAML Configuration Roadmap

> [!IMPORTANT]
> **This document is forward-looking.** The shapes, field names, and behaviors described below are *not all wired end-to-end yet*. Some sections describe seams that exist in the code but are not reachable from a config file; others describe features that are still at the design stage. Do not treat any YAML in this document as a working example unless it appears in [YAML Configuration Files](../tutorials/yaml-config.md). Field names may change before they ship.

## Scope

This document describes planned extensions to the YAML configuration format. It exists so that contributors and power users can see where the format is headed, why the seams in the current loader were placed where they were, and which workloads will become expressible once the missing pieces land.

For the format as it works today, see [YAML Configuration Files](../tutorials/yaml-config.md). For the schema, see `src/aiperf/config/schema/aiperf-config.schema.json`.

## Where the format is today

The v2 envelope is partway between single-config and the multi-phase / multi-dataset shape this document targets. The seams are intentional, but several stop short of being usable end-to-end.

What works today:

- **Multi-model selection is wired.** `benchmark.models` is a `ModelsAdvanced` block with `items: list[ModelItem]` and a `strategy` field — `round_robin`, `random`, or `weighted`. `modality_aware` is roadmap-only and is not accepted by the current validator. The singular `model:` shorthand is normalized into the items list by `_normalize_models`. Multi-model in one run is a real feature, not a roadmap item.
- **`benchmark.phases: [...]`** is a list, validated as a discriminated union over phase types. The singular `phases: { type: ..., ... }` shorthand is normalized to a one-entry list named `profiling` by `_normalize_dataset_and_phases`. Top-level `warmup:` / `profiling:` shorthand is normalized to a `[warmup, profiling]` list by `_normalize_warmup_profiling_to_phases`.
- **Singular `dataset:`** is auto-promoted to a one-entry list with `name: "default"` by `_normalize_single_dataset_listed`.
- **Sweep parameter paths** address phases and datasets by their user-given name. Path keying logic lives in `expand_sweep` and its helpers; the `phases.profiling.<X>` handling there (`_profiling_alias_candidates`) is now only a legacy fallback for configs with no phase literally named `profiling`.
- **User-named phases with an explicit `kind`.** `BasePhaseConfig.name` is a free-form identifier (`str` with `pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$"`), and a separate `BasePhaseConfig.kind: PhaseKind | None` field carries the warmup-vs-profiling runtime role. See "User-named phases" below.

What does **not** yet hold end-to-end:

- **`benchmark.datasets` is hard-capped at one entry.** `BenchmarkConfig.datasets` is a `list[DatasetConfig]` with `min_length=1, max_length=1`. The list shape exists only so the same schema can be shared between YAML and the `AIPerfSweep` CRD; the field's own description states "the runtime currently loads exactly one dataset." Multiple-dataset input is rejected at validation time, not at runtime.
- **Per-phase dataset selection is half-scaffolded.** `TimingResolver._validate_fixed_schedule_timing` reads a per-phase dataset via `getattr(phase, "dataset", None) or run.cfg.get_default_dataset_name()`, but no `dataset:` field exists on `BasePhaseConfig` yet, so the lookup always falls through to the default. The seam is anticipating a feature that hasn't landed.
- **A phase-vs-dataset compatibility checker exists, but only along three axes.** `check_phase_dataset_compatibility` currently rejects three combinations: a phase that requires a stop condition with none of requests/duration/sessions set against a non-graph dataset (graph workloads infer a single-corpus-pass stop), a phase that `requires_sequential_sampling` (today, just `fixed_schedule`) against a file dataset that doesn't use sequential sampling, and a phase that `requires_multi_turn` (today, just `user_centric`) against a non-multi-turn file dataset. Other compatibility axes — synthetic-vs-trace for `fixed_schedule`, dataset format mismatches — are not yet enforced here.

The roadmap items below describe how each of those gaps closes.

## User-named phases (current behavior)

> [!NOTE]
> Unlike the rest of this document, this section describes **shipped behavior**. It is retained here because the multi-dataset and per-phase-model roadmap items below build directly on it.

### Motivation

Two phases (one warmup, one profiling) covers most synthetic load tests. It runs out of expressivity quickly:

- Cold-cache warmup followed by warm-cache warmup followed by profiling — three phases, two of them with warmup semantics.
- KV-cache priming under a low rate, then a stepped rate sweep across three rate levels in the same run, with each step's results reported separately.
- A trace-replay profiling phase split into an "early window" and "late window" so you can compare steady-state vs. ramp behavior in one job.

Each of these is expressible today by giving every phase its own name and an explicit `kind`.

### Shape

```yaml
benchmark:
  phases:
    - name: cold_cache_warmup
      kind: warmup                 # explicit kind, independent of the name
      type: concurrency
      concurrency: 4
      requests: 50

    - name: warm_cache_warmup
      kind: warmup
      type: concurrency
      concurrency: 16
      requests: 100

    - name: steady_state_profile
      kind: profiling
      type: poisson
      rate: 30.0
      duration: 120

    - name: tail_profile
      kind: profiling
      type: poisson
      rate: 50.0
      duration: 120
```

How it behaves:

- `BasePhaseConfig.name` is free-form, validated against the identifier regex `^[A-Za-z_][A-Za-z0-9_-]*$`. It is a workflow label, not a phase kind.
- `BasePhaseConfig.kind` (`PhaseKind | None`) carries the warmup-vs-profiling runtime role that the credit and results pipeline distinguishes.
- `exclude_from_results` is driven by `kind`, not by string equality on `name`: `kind: warmup` is always excluded, `kind: profiling` always included, and an explicit value inconsistent with the kind is rejected by `BasePhaseConfig._validate_phase_constraints`.
- Legacy two-phase configs continue to load: normalization infers `kind` from the canonical names `warmup` / `profiling` via `_infer_phase_kind`, applied by each shorthand normalizer. The reserved names are also pinned to their matching kind in `BasePhaseConfig._validate_phase_constraints`.
- Sweep parameter paths address phases by user-given name (`phases.steady_state_profile.rate`). `phases.profiling.<X>` remains only as a legacy alias fallback.

### Remaining gaps

Naming is plumbed through config, validation, and sweep expansion. Still open:

- **Reports and artifacts** — per-phase JSON/Parquet/CSV filename components and report headers derived consistently from the phase name for arbitrary names, not just the two canonical ones.
- **Metrics rollups** — bucketing per-phase results under the user-given name, and keeping distinct-named phases from being aggregated together per the project's no-cross-aggregation rule.

## Multiple datasets, real-world

`datasets:` is a one-element list today: the field declares `min_length=1, max_length=1` so the schema can be shared with the `AIPerfSweep` CRD without forking. Lifting the cap is the prerequisite for every workload below.

### Motivating workloads

- **Synthetic warmup, trace replay for profiling.** Warmup runs cheap synthetic prompts to prime the KV cache; profiling replays a captured production trace whose timing and content matter.
- **A/B prompt distributions in one run.** Compare a short-prompt distribution against a long-prompt distribution under the same model, endpoint, and concurrency — without launching two jobs and collating results manually.
- **Specialized accuracy-and-perf in one job.** A perf-oriented synthetic dataset followed by a small accuracy-graded dataset that exercises the same deployment, with results aggregated into one report.

### Target shape

```yaml
benchmark:
  datasets:
    - name: warmup_synth
      type: synthetic
      entries: 50
      prompts: {isl: 256, osl: 64}

    - name: prod_trace
      type: file
      path: ./traces/prod-2026-04.jsonl

    - name: long_tail
      type: synthetic
      entries: 200
      prompts:
        isl: {mean: 4096, stddev: 512}
        osl: {mean: 256, stddev: 64}

  phases:
    - name: warmup
      kind: warmup
      dataset: warmup_synth
      type: concurrency
      concurrency: 4
      requests: 50

    - name: replay
      kind: profiling
      dataset: prod_trace
      type: fixed_schedule

    - name: long_tail_probe
      kind: profiling
      dataset: long_tail
      type: poisson
      rate: 10.0
      duration: 60
```

### Required wiring

1. **Lift the `max_length=1` cap on `BenchmarkConfig.datasets`**, replacing the schema-share comment with a real multi-dataset contract.
2. **Add `dataset: <name>` to `BasePhaseConfig`** so the partial scaffolding in `TimingResolver._validate_fixed_schedule_timing` becomes a real read instead of always falling through to `get_default_dataset_name()`.
3. **Validate that every `phase.dataset` resolves** to an entry in `benchmark.datasets`. Use the existing "did you mean?" hinting infrastructure for typos.
4. **Extend `check_phase_dataset_compatibility`.** Today it checks three rules: the required-stop rule (a phase with `_stop_condition_required` and no requests/duration/sessions is only valid against a graph dataset), `requires_sequential_sampling` (file-dataset sampling strategy), and `requires_multi_turn` (file-dataset format). Add: synthetic-vs-trace mismatches for `fixed_schedule`, dataset-format compatibility per phase type, and any rules that fall out of multi-dataset semantics. The fixed-schedule timing-data check in `TimingResolver._validate_fixed_schedule_timing` can move here once it has a real `phase.dataset` to read.
5. **Dataset preloading.** Today, the dataset manager prepares one dataset. With multiple datasets in play, prepare each up-front, key shared resources (tokenizer, prompt cache) by dataset name, and stream the right one to the credit issuer per phase.
6. **Reporting.** Per-phase JSON exports already partition by phase; once phases reference distinct datasets, include the dataset name in each phase's metadata block so downstream tools can group by it without re-deriving from the config.

### Compatibility matrix (planned)

| Phase `type`        | Synthetic | File (trace) | Public | Composed |
|---------------------|:---------:|:------------:|:------:|:--------:|
| `concurrency`       | yes       | yes          | yes    | yes      |
| `poisson`/`gamma`/`constant` | yes | yes        | yes    | yes      |
| `user_centric`      | yes       | yes (multi-turn format only) | conditional | conditional |
| `fixed_schedule`    | no        | yes (sequential sampling, with timing fields) | conditional | conditional |

The `user_centric` and `fixed_schedule` constraints are partially enforced today: `requires_multi_turn(USER_CENTRIC)` and `requires_sequential_sampling(FIXED_SCHEDULE)` are checked against file datasets in `check_phase_dataset_compatibility`. The synthetic-vs-`fixed_schedule` rejection and the timing-data check (currently in `TimingResolver._validate_fixed_schedule_timing`) move into the same checker as part of this work.

## Cross-cutting extensions

### Per-phase model selection

Multi-model in one run is already supported via `ModelsAdvanced.strategy` (`round_robin`, `random`, `weighted`) — a single phase can route across the full `items` list. `modality_aware` remains roadmap-only. What is *not* supported is binding a **specific model to a specific phase**, which lets you compare two models within one job under matched arrival patterns:

```yaml
benchmark:
  models:
    items:
      - {name: llama-3-8b}
      - {name: llama-3-70b}
  phases:
    - name: small_model_profile
      model: llama-3-8b      # narrows the active model for this phase
      type: poisson
      rate: 30.0
      duration: 120

    - name: large_model_profile
      model: llama-3-70b
      type: poisson
      rate: 30.0
      duration: 120
```

`phases[].model` would be a name reference into `models.items`, narrowing the selection strategy to a single fixed pick for the duration of the phase. This stays compatible with the project's no-aggregate-across-runs rule: each phase's results are reported independently, and the report makes the model name part of the phase header.

### Per-phase endpoint

Most users will not need this, but it falls out cleanly once datasets and models are per-phase: a phase that targets a different deployment (different URL, different `endpoint.type`) can be expressed without a separate job. Useful for side-by-side gateway-vs-direct comparisons or for benchmarking a fallback path. Likely gated behind explicit opt-in to discourage accidental misconfiguration.

### Phase ordering, dependencies, and conditional execution

The current model assumes a strict linear ordering of `phases[]`. Several enhancements compose:

- **Skip-on-condition.** A phase can declare a precondition (e.g. only run if the previous phase met a goodput threshold). Useful for adaptive ramp tests that should bail out early instead of burning compute past saturation.
- **Phase dependencies.** Allow phases to be declared as a DAG rather than a list, so the loader can run independent phases in sequence but stop the whole job if a parent phase fails its convergence criteria.
- **Cross-phase carry-over.** Make explicit which warmup state (KV cache, prompt cache, scheduler state) is intended to persist into a profiling phase, so the dataset manager and credit issuer can plan for it instead of relying on side-effects.

These are deliberately listed as separate items: each is independently useful, and we should not bundle them into a single "phases v3" change.

### Reusable phase / dataset fragments

Once configs grow to four or five phases, repetition becomes the readability problem. Two complementary mechanisms:

- **YAML anchors and merge keys** — works today, but is awkward and editor support is uneven.
- **Native `templates:` block under the envelope** — define a named partial config; reference it from a phase or dataset entry with `extends: <name>`. Resolution happens before sweep expansion so sweep parameter paths still address concrete phases.

```yaml
templates:
  base_profile:
    type: poisson
    duration: 120
    grace_period: 30

benchmark:
  phases:
    - name: low_rate
      extends: base_profile
      rate: 10.0
    - name: high_rate
      extends: base_profile
      rate: 50.0
```

## Out of scope

Items deliberately not on this roadmap:

- **Cross-run aggregation.** Reporting that sums or averages metrics across distinct AIPerfJob runs is forbidden by the project's measurement contract; named phases inside one run do not change that.
- **Live editing during a run.** YAML configs are static for the duration of a job. Live re-tuning belongs to a different layer (the orchestrator API, not the config format).
- **Free-form Python expressions.** Jinja `{{ }}` is intentionally restricted; arbitrary Python is not coming back.
