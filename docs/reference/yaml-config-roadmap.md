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

- **Multi-model selection is wired.** `benchmark.models` is a `ModelsAdvanced` block (`src/aiperf/config/_models_core.py:93`; `src/aiperf/config/models.py` is a re-export shim) with `items: list[ModelItem]` and a `strategy` field — `round_robin`, `random`, or `weighted`. `modality_aware` is roadmap-only: the `ModelSelectionStrategy` enum (`src/aiperf/common/enums/enums.py:397-407`) defines only the three above, so it is not accepted by the current validator. The singular `model:` shorthand is normalized into the items list (`src/aiperf/config/loader/normalizers.py:79-89`). Multi-model in one run is a real feature, not a roadmap item.
- **Free-form, unique phase names.** `BasePhaseConfig.name` is a plain `str` (`min_length=1`), not a `Literal` (`src/aiperf/config/phases.py:72-80`). Any identifier is allowed and `validate_phase_names_unique` (`src/aiperf/config/config.py:438-449`) rejects duplicates. Names are used for result-file naming, sweep targeting, and status/logs.
- **`benchmark.phases: [...]`** is a list, validated as a discriminated union over phase types. The singular `phases: { type: ..., ... }` shorthand is normalized to a one-entry list named `profiling` (`src/aiperf/config/loader/normalizers.py:99-103`). Top-level `warmup:` / `profiling:` shorthand is normalized to a `[warmup, profiling]` list.
- **`phase.dataset` exists and is read.** `BasePhaseConfig.dataset: str | None` (`src/aiperf/config/phases.py:99-106`) names the dataset for that phase; the fixed-schedule timing resolver already reads it (`getattr(phase, "dataset", None) or run.cfg.get_default_dataset_name()`, `src/aiperf/config/resolution/resolvers.py:358-360`).
- **Multiple datasets are accepted.** `BenchmarkConfig.datasets` is `list[DatasetConfig]` with only `min_length=1` — no `max_length` cap (`src/aiperf/config/config.py:199-209`). Singular `dataset:` is auto-promoted to a one-entry list with `name: "default"` (`src/aiperf/config/loader/normalizers.py:92-97`).
- **Sweep parameter paths** address phases and datasets by name. Path keying logic lives in `src/aiperf/config/sweep/expand.py`; see the `phases.profiling.<X>` special case at `expand.py:495-499` and `_find_phase_or_recipe_alias` at `expand.py:554-567`.

What does **not** yet hold end-to-end:

- **No explicit `kind` field; warmup vs. profiling is still name-derived.** There is no `kind:` discriminator. The warmup/profiling distinction comes from string equality on the phase `name`: `_validate_phase_constraints` (`src/aiperf/config/phases.py:244-259`) forces `exclude_from_results=True` for a phase named `warmup` and `False` for one named `profiling`, and rejects an inconsistent explicit value. A phase with any other name defaults to `exclude_from_results=False` (included). So while names are free-form, the *semantics* of "this is a warmup" still hinge on the literal string `warmup` rather than an explicit kind.
- **Per-phase dataset routing is partial.** `phase.dataset` exists and the fixed-schedule timing resolver reads it, but `validate_phase_dataset_compatibility` (`src/aiperf/config/config.py:509-525`) checks each phase against `get_default_dataset()` — the *default* dataset only, not the phase's own `phase.dataset`. Full per-phase routing (compatibility checked against the referenced dataset, plus multi-dataset preloading in the dataset manager) has not landed.
- **A phase-vs-dataset compatibility checker exists, but only along two axes.** `check_phase_dataset_compatibility` (`src/aiperf/config/resolution/predicates.py:208-250`) currently rejects only two combinations: a phase that `requires_sequential_sampling` (today, just `fixed_schedule`) against a file dataset that doesn't use sequential sampling, and a phase that `requires_multi_turn` (today, just `user_centric`) against a non-multi-turn file dataset. Other compatibility axes — synthetic-vs-trace for `fixed_schedule`, dataset format mismatches — are not yet enforced here.

The roadmap items below describe how each of those gaps closes.

## N user-named phases

### Motivation

Two phases (one warmup, one profiling) covers most synthetic load tests. It runs out of expressivity quickly:

- Cold-cache warmup followed by warm-cache warmup followed by profiling — three phases, two of them with warmup semantics.
- KV-cache priming under a low rate, then a stepped rate sweep across three rate levels in the same run, with each step's results reported separately.
- A trace-replay profiling phase split into an "early window" and "late window" so you can compare steady-state vs. ramp behavior in one job.

All of these are expressible in YAML today only by collapsing distinct logical phases under the same name (`profiling`, `profiling`, `profiling`) and disambiguating later by index, which loses the clarity the named-phase shape was meant to give.

### Target shape

```yaml
benchmark:
  phases:
    - name: cold_cache_warmup
      kind: warmup                 # explicit kind; replaces the implicit name->kind mapping
      type: concurrency
      concurrency: 4
      requests: 50
      exclude_from_results: true

    - name: warm_cache_warmup
      kind: warmup
      type: concurrency
      concurrency: 16
      requests: 100
      exclude_from_results: true

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

Key changes:

- `name` becomes free-form (validated against a permissive identifier regex), rather than a `Literal`.
- A new `kind` field carries the warmup-vs-profiling distinction the credit pipeline currently derives from the name. `exclude_from_results` is then driven by `kind`, not by string equality on `name`.
- Reports, artifact subdirectories, and sweep parameter paths address phases by user-given name (`phases.steady_state_profile.rate`).
- Existing two-phase configs continue to load: `name: warmup` defaults `kind: warmup`, `name: profiling` defaults `kind: profiling`.

### Required wiring

End-to-end naming touches roughly five layers:

1. `src/aiperf/config/phases.py` — `BasePhaseConfig.name` is already a free-form `str`; what remains is a new `kind: Literal["warmup", "profiling"]` field with name-based defaults so warmup semantics stop hinging on the literal string `warmup`.
2. Credit issuer (`PhaseRunner` and `CreditIssuer`) — index phases by name rather than by `is_warmup` boolean.
3. Records manager / metrics rollups — bucket per-phase results under the user-given name; prevent cross-phase aggregation across distinct names per the existing project rule.
4. Reports and artifacts — per-phase JSON/Parquet/CSV files use the phase name as a filename component.
5. Sweep expansion (`src/aiperf/config/sweep/expand.py`) — already addresses phases by name; minor changes needed only if the keying logic assumes the two-element set.

## Multiple datasets, real-world

`datasets:` already accepts multiple entries — the field is `list[DatasetConfig]` with only `min_length=1` (no `max_length` cap), and `phase.dataset` already names the per-phase dataset. What remains is wiring those references through validation and runtime loading so the workloads below become expressible.

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

> **Recently landed.** Two foundational pieces already shipped: the `datasets` field carries no `max_length` cap (`src/aiperf/config/config.py:199-209`), and `BasePhaseConfig.dataset: str | None` exists and is read by the fixed-schedule timing resolver (`src/aiperf/config/resolution/resolvers.py:358-360`). The remaining items below turn those seams into full per-phase routing.

1. **Validate that every `phase.dataset` resolves** to an entry in `benchmark.datasets`. Today `validate_phase_dataset_compatibility` (`src/aiperf/config/config.py:509-525`) still checks each phase against `get_default_dataset()` only; extend it to resolve and check `phase.dataset`, using the existing "did you mean?" hinting infrastructure for typos.
2. **Extend `check_phase_dataset_compatibility`** (`src/aiperf/config/resolution/predicates.py:208-250`). Today it only checks `requires_sequential_sampling` (file-dataset sampling strategy) and `requires_multi_turn` (file-dataset format). Add: synthetic-vs-trace mismatches for `fixed_schedule`, dataset-format compatibility per phase type, and any rules that fall out of multi-dataset semantics. The fixed-schedule timing-data check in `TimingResolver._validate_fixed_schedule_timing` (`src/aiperf/config/resolution/resolvers.py:353-372`) can move here once compatibility resolves against the phase's own dataset.
3. **Dataset preloading.** Today, the dataset manager prepares one dataset. With multiple datasets in play, prepare each up-front, key shared resources (tokenizer, prompt cache) by dataset name, and stream the right one to the credit issuer per phase.
4. **Reporting.** Per-phase JSON exports already partition by phase; once phases reference distinct datasets, include the dataset name in each phase's metadata block so downstream tools can group by it without re-deriving from the config.

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
