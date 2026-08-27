<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Accuracy and multi-run orchestration parity audit

**Python baseline:** `/mnt/4tb/aiperf-parity-py-main/src/aiperf/` at git rev
`bc359bf8fd` (`origin/main`). All Python `path:line` citations below are against
that tree. Rust citations are against
`/home/anthony/nvidia/projects/aiperf/ajc/rust/rust/`, which has no counterpart
on `origin/main` and is unaffected.

An earlier revision of this audit compared against a local feature branch 4345
commits ahead of `origin/main`. One finding (originally P1 #3, adaptive
`error_rate` SLA units) turned out to rest on branch-local code and has been
withdrawn; see [Withdrawn after baseline correction](#withdrawn-after-baseline-correction).
Verdict: 8 findings still valid, 1 withdrawn, 0 changed.

| # | Finding | Severity | Baseline verdict |
|---|---------|----------|------------------|
| 1 | `--accuracy-benchmark` silently ignored | P0 | STILL VALID (strengthened) |
| 2 | `pareto-sweep` concurrency default | P1 | STILL VALID |
| 3 | Adaptive `error_rate` scale/denominator | — | **WITHDRAWN** (branch artifact) |
| 4 | `--convergence-metric` never stops early | P1 | STILL VALID |
| 5 | Per-trial warmup stripping key | P2 | STILL VALID |
| 6 | Accuracy result keys / JSONL / console sink | P2 | STILL VALID |
| 7 | YAML `variables.*` sweep axis | P2 | STILL VALID |
| 8 | YAML magic lists | P2 | STILL VALID |
| 9 | Per-trial seed derivation and collision | P2 | STILL VALID (line re-cite) |

Findings 2, 4, 5, 6, 7, 8 cite only files byte-identical to baseline at identical
line numbers. Finding 1 gained supporting evidence. Finding 9's
`orchestrator.py` citation shifted from 145-148 to 123-126; content unchanged.
Findings 5 and the artifact-layout consistency entry cite `strategies.py:321` and
`orchestrator.py:51-55` respectively (baseline numbering). No finding cited
`accuracy/worker.py`, `graders/_math_strip.py`, `graders/_codegen_worker*.py`, or
`benchmarks/mmlu.py`, so those divergences invalidate nothing — but they do
remove the reassurance that grading logic cannot drift.

## Summary

The single largest risk in this domain is that `--accuracy-benchmark` is accepted
by the native CLI, lowered into `cfg.accuracy`, and then never consulted by the
protocol-v2 projection: `workload_kind()` can only return `scheduled` or `graph`,
and the only producer of `NativeDatasetPlan::StaticAccuracy` is a workload
factory keyed on a `static_accuracy` workload id that nothing ever emits. A user
who runs an accuracy benchmark gets a normal synthetic perf run, exit code zero,
and no accuracy artifacts. Reinforcing this: the Python module the native
evaluator subprocess targets, `aiperf.accuracy.worker`, does not exist in the
baseline package at all, so even a wired-up workload id would not grade anything.
Everything downstream in accuracy parity (result key names, the per-record JSONL,
the console table) is therefore latent rather than live, but the artifact
contract has already diverged and will ship wrong once the path is connected.

Multi-run orchestration is much closer, and the mechanical parts — grid product
order, zip length validation, axis sort order, artifact directory layout, the
confidence-aggregate schema, the goodput formula, and SLO direction/inclusivity —
match. The live divergences are about *how many runs happen*: `pareto-sweep` defaults to
five concurrency values in Python and one in Rust (a 5x run-count change from the
same command line), and `--convergence-metric` is accepted by Rust but never
stops trials early. The adaptive SLA evaluator — including the `error_rate`
scale and denominator — matches baseline exactly.

Four lower-severity items round it out: per-trial warmup stripping keys off a
phase's `exclude_from_results` flag in Python but off the literal phase name
`"warmup"` in Rust; the per-trial seed derivation differs and, worse, Rust's
`base + variation + trial` arithmetic makes distinct cells collide on one seed;
and two YAML sweep spellings Python accepts (`variables.*` axes, bare magic
lists) fail loudly in Rust rather than silently, which is the acceptable
direction but is undocumented.

## Findings

### 1. `--accuracy-benchmark` is accepted and silently ignored — no grading, no accuracy artifacts

**Severity:** P0
**Status:** KNOWN(still-true) — backlog `P0.1`, `docs/dev/python-rust-parity-gaps.md:81`

**Python evidence**

`--accuracy-benchmark` populates `AccuracyConfig`, which gates the whole
accuracy pipeline. Python is explicit that accuracy-only flags without
`--accuracy-benchmark` would be silently ignored, and refuses rather than
allowing it:

```118:121:src/aiperf/config/accuracy.py
            f"Accuracy options {flag_names} were set but --accuracy-benchmark "
            f"is not. Accuracy mode requires --accuracy-benchmark to select a "
            f"benchmark; otherwise these flags are silently ignored. "
            f"Available benchmarks: {', '.join(available) or '(none)'}."
```

With the benchmark set, the dataset loader converts graded problems into
conversations carrying ground truth:

```12:14:src/aiperf/dataset/loader/accuracy_dataset_loader.py
BenchmarkProblem i. Each Conversation carries accuracy_ground_truth and
accuracy_task so that DatasetManager can propagate them through
```

and the graded summary is injected into the exported records:

```1108:1111:src/aiperf/controller/system_controller.py
        self._profile_results.results.records.extend(
            self._accuracy_results.to_metric_results()
        )
        self._accuracy_results_injected = True
```

**Rust evidence**

The flag is parsed and lowered into config:

```1047:1048:rust/cli/src/load.rs
fn build_accuracy(flags: &ProfileFlags) -> Option<crate::model::config::Accuracy> {
    let benchmark = flags.accuracy_benchmark.clone()?;
```

But the projection can only select two workloads:

```102:114:rust/runtime/src/config/model/workload_kind.rs
pub fn workload_kind(cfg: &BenchmarkConfig) -> WorkloadKind {
    let is_graph = cfg
        .datasets
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|dataset| is_graph_format(dataset_format_token(dataset)));
    if is_graph {
        WorkloadKind::Graph
    } else {
        WorkloadKind::Scheduled
    }
}
```

The in-source comment claims the accuracy path is reached through the dataset
plan instead of a workload id:

```41:44:rust/runtime/src/config/model/workload_kind.rs
/// `StaticAccuracy` is intentionally not represented: today's projection selects
/// a static-accuracy run through the dataset *plan*
/// ([`NativeDatasetPlan::StaticAccuracy`](crate::engine)), not a distinct
/// workload id — the emitted workload id is still `scheduled`.
```

That claim is not backed by code. The only producer of
`NativeDatasetPlan::StaticAccuracy` is `lower_static_accuracy`, reachable only
from the workload factory gated on the `static_accuracy` id:

```565:565:rust/runtime/src/engine/online_execution.rs
            workload_config::<StaticAccuracyWorkloadConfigV2>(workload, "static_accuracy")?;
```

and `rg '"static_accuracy"' rust/ -g '*.rs' -g '!*tests*'` returns only
`registry.rs:1554` (the descriptor id), `online_execution.rs:565`, and
`online_execution.rs:588` — no projection site. `rust/runtime/src/engine/protocol_v2.rs`
contains no occurrence of `accuracy` at all; its only workload selection is:

```373:373:rust/runtime/src/engine/protocol_v2.rs
        let workload_id = workload_kind.workload_id();
```

Consequently `prepare_static_accuracy` short-circuits for every stock profile run:

```1565:1567:rust/runtime/src/engine/execute/entrypoints.rs
    let NativeDatasetPlan::StaticAccuracy(spec) = &request.dataset else {
        return Ok(None);
    };
```

Independently, the Python module the native evaluator subprocess launches has no
counterpart in the baseline package:

```105:112:rust/runtime/src/accuracy_core/worker.rs
    /// Build the standard `python -u -m aiperf.accuracy.worker` command.
            .arg("aiperf.accuracy.worker")
```

```1184:1185:rust/runtime/src/engine/online_execution.rs
fn default_accuracy_worker_module() -> String {
    "aiperf.accuracy.worker".into()
```

`ls /mnt/4tb/aiperf-parity-py-main/src/aiperf/accuracy/` contains
`accumulator.py`, `accuracy_console_exporter.py`, `accuracy_data_exporter.py`,
`accuracy_record_processor.py`, `benchmark_loader.py`, `benchmarks/`, `graders/`,
`__init__.py`, `jsonl_writer.py`, `models.py`, `protocols.py` — and no
`worker.py`. So even if the workload id were projected, the evaluator subprocess
would fail to import its module.

**Observable user impact**
`aiperf profile --accuracy-benchmark mmlu ...` exits zero, runs the configured
synthetic/file dataset instead of the benchmark problems, grades nothing, and
writes no `accuracy_results.csv`, no `accuracy_export.jsonl`, and no `accuracy.*`
keys in `profile_export_aiperf.json`. There is no warning. Under Python the same
command loads MMLU problems, grades every response, and emits all three
artifacts.

**Confidence:** High — verified by exhaustive search for every producer of the
`static_accuracy` workload id and of `NativeDatasetPlan::StaticAccuracy`.

### 2. `pareto-sweep` defaults to five concurrency values in Python and one in Rust

**Severity:** P1
**Status:** NEW (concrete instance under the generic backlog item `P1.7`,
`docs/dev/python-rust-parity-gaps.md:435`)

**Python evidence**

```63:63:src/aiperf/search_recipes/_pareto_sweep.py
    _DEFAULT_CONCURRENCY: ClassVar[tuple[int, ...]] = (1, 4, 16, 64, 256)
```

```118:124:src/aiperf/search_recipes/_pareto_sweep.py
    def _resolve_concurrencies(self, overrides: dict[str, Any]) -> list[int]:
        raw = overrides.get("concurrency")
        if raw is None:
            return list(self._DEFAULT_CONCURRENCY)
        if isinstance(raw, list):
            return [int(v) for v in raw]
        return [int(raw)]
```

Python also refuses a degenerate single-point Pareto sweep:

```89:94:src/aiperf/search_recipes/_pareto_sweep.py
        if len(pairs) * len(concurrencies) < 2:
            raise ValueError(
                f"recipe {self.name!r}: a Pareto sweep with a single point "
                "is meaningless. Pass at least 2 pairs OR at least 2 "
                "concurrency values."
            )
```

**Rust evidence**

```403:413:rust/cli/src/search.rs
    let conc: Vec<i64> = match flags.concurrency.as_deref() {
        Some(c) => c
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("bad concurrency {s:?}"))
            })
            .collect::<anyhow::Result<_>>()?,
        None => vec![1],
    };
```

There is no minimum-point check; `expand_pareto` returns whatever the cross
product yields.

**Observable user impact**
`aiperf profile --search-recipe pareto-sweep --isl-osl-pairs 128/128,1024/128`
produces 10 runs (2 shapes x 5 concurrencies) and 10 artifact directories under
Python, and 2 runs (2 shapes x concurrency 1) under Rust. The Pareto frontier is
computed over an entirely different, and far sparser, observation set, so the
selected frontier points differ. A single-shape invocation is an error in Python
and a silent one-run "sweep" in Rust. No warning either way.

**Confidence:** High — both defaults and both cross-product loops read directly.

### 3. Adaptive `error_rate` SLA scale and denominator — WITHDRAWN

**Status:** WITHDRAWN (branch artifact). See
[Withdrawn after baseline correction](#withdrawn-after-baseline-correction).
Baseline Python computes `error_rate` exactly as Rust does. The heading is
retained so finding numbers stay stable.

### 4. `--convergence-metric` is accepted but never stops trials early

**Severity:** P1
**Status:** KNOWN(still-true) — backlog `P1.8`, `docs/dev/python-rust-parity-gaps.md:448`

**Python evidence**

Setting `multi_run.convergence` (which `--convergence-metric` populates) switches
the strategy to adaptive stopping bounded by `num_runs`:

```23:29:src/aiperf/config/sweep/multi_run.py
class ConvergenceConfig(BaseConfig):
    """Adaptive trial-stopping criterion.

    Presence of this object on `MultiRunConfig.convergence` enables
    adaptive stopping: trials run until the criterion fires (or
    `MultiRunConfig.num_runs` is reached, whichever comes first).
    """
```

with defaults `stat=avg`, `mode=ci_width`, `threshold=0.10`, `min_runs=2`
(multi_run.py:39-71), consumed by:

```369:372:src/aiperf/orchestrator/strategies.py
    def should_continue(self, results: list[RunResult]) -> bool:
        """Continue unless max reached or criterion converged (after min)."""
        n = len(results)
        if n >= self.max_runs:
```

**Rust evidence**

Three of the four convergence flags are declared unimplemented and warned about:

```390:393:rust/cli/src/profile.rs
    ("--convergence-mode", |f| f.convergence_mode.is_some()),
    ("--convergence-stat", |f| f.convergence_stat.is_some()),
    ("--convergence-threshold", |f| {
        f.convergence_threshold.is_some()
```

`--convergence-metric` is *not* on that list. It is validated as if it enabled
adaptive convergence:

```472:476:rust/cli/src/profile.rs
    if flags.convergence_metric.is_some() && flags.num_profile_runs.unwrap_or(1) <= 1 {
        anyhow::bail!(
            "--convergence-metric requires --num-profile-runs > 1. \
             Set --num-profile-runs to at least 2 to enable adaptive convergence."
        );
```

but its only actual effect is to switch on an extra artifact:

```136:136:rust/cli/src/sweep/aggregate.rs
        if flags.convergence_metric.is_some() {
```

The trial planner materializes every `(trial, variation)` cell up front with no
convergence hook and no early-exit path:

```81:83:rust/cli/src/sweep/run.rs
    let mut cells = Vec::new();
    for trial in 0..trials {
        for variation in &expansion.variations {
```

**Observable user impact**
`--num-profile-runs 10 --convergence-metric time_to_first_token` runs as few as
2 trials under Python (stopping once the 95% CI half-width falls under 10% of the
mean) and always exactly 10 under Rust. Wall-clock time, the number of
`profile_runs/run_NNNN` directories, `num_profile_runs`/`num_successful_runs` in
the aggregate, and every aggregated statistic (computed over a different number
of samples) all differ. The error message at profile.rs:474 actively tells the
user that convergence is enabled.

**Confidence:** High.

### 5. Per-trial warmup stripping keys off the phase name in Rust and the `exclude_from_results` flag in Python

**Severity:** P2
**Status:** NEW (adjacent to backlog `P1.11`, `docs/dev/python-rust-parity-gaps.md:492`, which covers phase projection rather than trial warmup handling)

**Python evidence**

```320:324:src/aiperf/orchestrator/strategies.py
        config = config.model_copy(deep=True)
        config.phases = [p for p in config.phases if not p.exclude_from_results]
        for phase in config.get_profiling_phases():
            phase.agentic_cache_warmup_duration = None
        return config
```

**Rust evidence**

```141:145:rust/cli/src/sweep/run.rs
pub(crate) fn drop_warmup(run: &mut BenchmarkRun) {
    if let Some(phases) = run.cfg.phases.as_mut() {
        phases.retain(|p| p.common.name != "warmup");
    }
}
```

Rust does model the flag (`rust/cli/src/yaml.rs:1539-1540`,
`rust/runtime/src/config/resolve.rs:1499`); `drop_warmup` just does not consult it.

**Observable user impact**
With `--num-profile-runs 3` and an authored phase named anything other than
`warmup` but flagged `excludeFromResults: true` (for example `name: prewarm`),
Python drops it on trials 2 and 3 while Rust keeps running it on every trial. The
trials therefore measure different system states, and the confidence aggregate's
mean/CI shift accordingly. Separately, Python zeroes
`agentic_cache_warmup_duration` on the surviving profiling phases so the
synthesized agentic cache-pressure substage is suppressed on trials 2+; Rust has
no equivalent, so that substage re-runs every trial. No warning either way.

**Confidence:** High for the name-vs-flag divergence.

### 6. Python's exported accuracy result keys and per-record accuracy artifact have no Rust counterpart

**Severity:** P2 (latent behind finding 1, but a shipped-contract divergence)
**Status:** NEW

**Python evidence** — four families of `accuracy.*` result keys, emitted in a
load-bearing order and injected into the perf JSON/CSV:

```20:23:src/aiperf/accuracy/models.py
ACCURACY_OVERALL_TAG = "accuracy.overall"
ACCURACY_TASK_TAG_PREFIX = "accuracy.task."
ACCURACY_UNPARSED_TAG = "accuracy.unparsed"
ACCURACY_UNPARSED_TASK_TAG_PREFIX = "accuracy.unparsed.task."
```

```210:211:src/aiperf/accuracy/models.py
        Emitted in this exact order (load-bearing for byte-exact JSON/CSV):
        overall, tasks sorted, unparsed overall, unparsed tasks sorted.
```

Plus a per-record JSONL sink and a console table:

```20:20:src/aiperf/accuracy/jsonl_writer.py
class AccuracyJSONLWriter(
```

```23:23:src/aiperf/accuracy/accuracy_console_exporter.py
class AccuracyConsoleExporter(AIPerfLoggerMixin):
```

**Rust evidence** — one accuracy sink exists, `accuracy_results.csv`, and it
matches Python's CSV shape:

```22:22:rust/runtime/src/export/accuracy_csv.rs
const ACCURACY_CSV_FILE: &str = "accuracy_results.csv";
```

`rg accuracy rust/runtime/src/export/*.rs -l` returns only `accuracy_csv.rs` and
`mod.rs`: there is no JSONL sink and no console sink. The only `accuracy.*`
metric tags in the Rust catalog are differently named and flagged internal, so
they never reach an exported artifact:

```262:263:rust/runtime/src/metrics_core/catalog.rs
            Self::AccuracyCorrect => "accuracy.correct",
            Self::AccuracyUnparsed => "accuracy.unparsed",
```

```1826:1843:rust/runtime/src/metrics_core/catalog.rs
    spec!(
        AccuracyCorrect,
        "Accuracy Correct",
        Ratio,
        Aggregate,
        Some(AggregationKind::Sum),
        MetricFlags::INTERNAL,
        []
    ),
    spec!(
        AccuracyUnparsed,
        "Accuracy Unparsed",
        Ratio,
        Aggregate,
        Some(AggregationKind::Sum),
        MetricFlags::INTERNAL,
        []
    ),
```

Note the name collision: `accuracy.unparsed` exists on both sides with different
semantics (Python: overall unparsed *ratio* with `count=total_evaluated`; Rust: an
internal sum).

**Observable user impact**
Once accuracy execution is connected, a user parsing
`profile_export_aiperf.json` for `accuracy.overall` or `accuracy.task.<id>`, or
reading `accuracy_export.jsonl` per-record grades, or expecting the accuracy
console table (including Python's 100%-unparsed warning), finds none of them.
`accuracy_results.csv` is the only surviving artifact.

**Confidence:** High for the absence; the impact is contingent on finding 1
being fixed.

### 7. YAML sweep axis `variables.<name>`: envelope Jinja block in Python, hard failure in Rust

**Severity:** P2 (loud refusal, but Python accepted it and the refusal is undocumented)
**Status:** NEW

**Python evidence** — `variables.*` is the one envelope-rooted escape, written at
the config root so Jinja re-renders per variation:

```214:217:src/aiperf/config/sweep/expand.py
        if resolved.split(".", 1)[0] == "variables":
            envelope_paths[resolved] = values
        else:
            body_paths[resolved] = values
```

```261:265:src/aiperf/config/sweep/expand.py
            if field_path in envelope_paths:
                # variables.<name> -> envelope-level Jinja block (re-rendered per variation)
                _set_nested_value(variant, field_path, value)
            else:
                _set_nested_value(body, field_path, value)
```

**Rust evidence** — no `variables` special case in the validator
(`rust/cli/src/sweep/yaml_sweep.rs:281-319`, where
`NON_SWEEPABLE_FIRST = ["sweep", "multi_run", "random_seed"]` at line 44), and
every resolved path is written under `benchmark`:

```208:222:rust/cli/src/sweep/yaml_sweep.rs
            let benchmark = config
                .as_object_mut()
                .and_then(|o| {
                    o.entry("benchmark")
                        .or_insert_with(|| Value::Object(Map::new()));
                    o.get_mut("benchmark")
                })
                .ok_or_else(|| anyhow::anyhow!("config root must be a mapping"))?;
...
                set_nested_value(benchmark, path, value.clone())?;
```

`variables` is declared on the root `ConfigFile` (`rust/cli/src/yaml.rs:649`) but
not on `Benchmark`, which is `deny_unknown_fields`
(`rust/cli/src/yaml.rs:687-689`), so the resulting `benchmark.variables.<name>`
fails deserialization.

**Observable user impact**
A config sweeping `variables.model_size: [7, 70]` produces two distinct Jinja-
rendered runs under Python and aborts the whole invocation under Rust with an
unknown-field error naming `variables`. Loud, so low severity — but nothing tells
the user that the envelope-variable sweep escape was dropped.

**Confidence:** High for the routing difference; High for loudness (the field is
absent from a `deny_unknown_fields` struct). The exact error text was not
executed.

### 8. YAML magic lists (a list where a scalar phase field is expected) auto-sweep in Python and fail in Rust

**Severity:** P2 (loud refusal; undocumented)
**Status:** NEW

**Python evidence** — a list at a phase-rooted magic-list field is detected with
no `sweep:` block present and expanded into variations:

```71:74:src/aiperf/config/sweep/expand.py
    if not variations:
        magic_sweeps = detect_sweep_fields(data.get("benchmark") or {})
        if magic_sweeps:
            variations = _expand_magic_lists(data, magic_sweeps)
```

```145:152:src/aiperf/config/sweep/expand.py
    def _collect(phase: dict[str, Any], prefix: str) -> None:
        for key, value in phase.items():
            if (
                isinstance(value, list)
                and key in MAGIC_LIST_FIELDS
                and all(isinstance(v, int | float) for v in value)
            ):
                sweep_fields[f"{prefix}.{key}"] = value
```

**Rust evidence** — sweeps require an explicit block, and the phase field is a
scalar:

```149:152:rust/cli/src/sweep/yaml_sweep.rs
pub fn parse(config: &Value) -> anyhow::Result<Option<YamlSweep>> {
    let Some(sweep) = config.get("sweep").and_then(Value::as_object) else {
        return Ok(None);
    };
```

```1507:1507:rust/cli/src/yaml.rs
    concurrency: Option<u32>,
```

**Observable user impact**
`benchmark.phases[profiling].concurrency: [10, 20, 30]` with no `sweep:` block
yields three runs under Python and a deserialization failure under Rust. The
equivalent CLI spelling (`--concurrency 10,20,30`) works on both sides, since
Python hoists it into a real grid `sweep` block
(`src/aiperf/config/flags/converter.py:307-319`) and Rust expands it through
`rust/cli/src/sweep/mod.rs:219`.

**Confidence:** High.

### 9. Per-trial seed derivation differs, and Rust's arithmetic makes distinct cells collide

**Severity:** P2
**Status:** KNOWN(still-true) — backlog `P1.8`, `docs/dev/python-rust-parity-gaps.md:448`

**Python evidence** — SHA-256 over `(seed, "<label>:trial:<n>")`:

```123:126:src/aiperf/orchestrator/orchestrator.py
    if plan.multi_run.vary_seed_per_trial and plan.random_seed is not None:
        return derive_variation_seed(
            plan.random_seed, f"{variation.label}:trial:{trial}"
        )
```

```123:135:src/aiperf/config/sweep/multi_run.py
    vary_seed_per_trial: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "When True, derive a distinct seed for each trial of a variation "
                "via SHA-256 over `(envelope_seed, variation.label, trial)`. "
```

**Rust evidence** — additive offsets:

```34:44:rust/cli/src/sweep/run.rs
    pub fn seed(&self, index: usize, trial: u32) -> Option<u64> {
        self.base.map(|b| {
            let variation = if self.same_seed { 0 } else { index as u64 };
            let trial_offset = if self.vary_per_trial {
                u64::from(trial)
            } else {
                0
            };
            b + variation + trial_offset
        })
    }
```

Also, `multiRun.varySeedPerTrial` is not a field of Rust's `deny_unknown_fields`
`MultiRunSection` (`rust/cli/src/yaml.rs:486-503`), so the YAML spelling Python
accepts is rejected; only the `--vary-seed-per-trial` flag
(`rust/cli/src/flags.rs:325-333`) works.

**Observable user impact**
Under `--vary-seed-per-trial`, `b + variation + trial` collides: variation 1 /
trial 0 and variation 0 / trial 1 both receive `b + 1`, so two different sweep
cells draw byte-identical synthetic data where Python gives each a distinct
stream. Independently, the same `--random-seed` produces different per-trial
prompts between implementations, so per-record artifacts are not reproducible
across them. The YAML key rejection is loud.

**Confidence:** High.

## Checked and consistent

- **Grid cross-product order.** Python `itertools.product` over
  `sorted(field_names)` (`src/aiperf/config/sweep/expand.py:248`) and Rust
  `cartesian` over a `BTreeMap` of resolved paths
  (`rust/cli/src/sweep/yaml_sweep.rs:241-255`, sorting at
  `rust/cli/src/sweep/mod.rs:219` and `rust/cli/src/sweep/plan.rs:180`) both sort
  axes alphabetically by dotted path and vary the last axis fastest.
  `serde_json`'s `preserve_order` feature is enabled in this workspace
  (confirmed via `cargo tree -e features`), so map-shaped inputs also keep
  authored order where the code relies on it.
- **Zip mismatched lengths.** Both refuse. Python:
  `src/aiperf/config/sweep/expand.py:295-297` length check. Rust:
  `rust/cli/src/sweep/yaml_sweep.rs:180-181` `"zip sweep parameters must all
  have equal length"`.
- **Artifact layout for sweeps x trials.** The five-row layout tables are
  identical, including the `run_NNNN` (no-sweep multi-run) vs `trial_NNNN`
  (sweep) asymmetry and the `REPEATED` default: Python
  `src/aiperf/orchestrator/orchestrator.py:51-55` and
  `src/aiperf/orchestrator/orchestrator.py:55-100`; Rust
  `rust/cli/src/sweep/artifact_dir.rs:5-11` and
  `rust/cli/src/sweep/artifact_dir.rs:39-47`.
- **Trial combination method.** Both aggregate the *mean of per-run summary
  stats* (not pooled records), keyed `<tag>_<stat>`: Python
  `src/aiperf/orchestrator/aggregation/confidence.py:185-198`; Rust
  `rust/cli/src/sweep/confidence.rs:202-205`. Percentiles are therefore
  means-of-percentiles on both sides.
- **Confidence aggregate schema.** Metadata keys match
  (`aggregation_type`, `num_profile_runs`, `num_successful_runs`, `failed_runs`
  as `{label,error}`, `confidence_level`, `run_labels`, and the `single_run`
  degraded flag): Python
  `src/aiperf/orchestrator/aggregation/confidence.py:157-183`; Rust
  `rust/cli/src/sweep/confidence.rs:248-270`. Rust adds `cooldown_seconds`
  (additive). Filenames match:
  `profile_export_aiperf_aggregate.{json,csv}` and
  `profile_export_aiperf_sweep.{json,csv}`
  (`rust/cli/src/sweep/aggregate.rs:375-376`).
- **Per-variation seed.** `base + variation_index`, or `base` under same-seed, on
  both sides: Python `src/aiperf/config/loader/plan.py` (`base_seed +
  variation_idx`); Rust `rust/cli/src/sweep/run.rs:36-42`. Seeding is applied
  only on sweep/trial paths, not single runs, on both sides.
- **Goodput.** Formula and denominators agree:
  `good_request_count` requires *all* configured SLOs to pass, `goodput =
  good_request_count / benchmark_duration`, `good_request_fraction =
  good_request_count / (request_count + error_request_count)`. Both sides also
  omit the goodput family entirely when no SLOs are configured
  (`src/aiperf/post_processors/base_metrics_processor.py`;
  `rust/runtime/src/metrics_core/accumulator.rs` `compute_good_request` early
  return). An earlier suspicion that Python emitted zeros here was wrong.
- **SLO comparison direction and inclusivity.** `>=` for larger-is-better, `<=`
  otherwise, inclusive on both sides
  (`rust/runtime/src/metrics_core/definition.rs` `passes_threshold`), with the
  same display-to-native unit conversion (`SloThreshold::from_display` vs
  Python's conversion) and the same rejection of unknown SLO metric tags.
- **Compact adaptive SLA normalization.** Identical nested
  `{metric: {stat: {op: threshold}}}` flattening with the same three error
  messages: `src/aiperf/config/adaptive_scale_phase.py:30-52` vs
  `rust/cli/src/yaml.rs:1803-1829`.
- **Adaptive rate metrics.** All three match baseline exactly.
  `error_rate` is `100 * errors / (successes + errors)` on both sides —
  percentage points, cancellations excluded
  (`src/aiperf/timing/strategies/adaptive_scale_sla.py:242-248` vs
  `rust/runtime/src/adaptive_core/sla.rs:328-336`), and Python documents the
  intent at `adaptive_scale_sla.py:228-233`. `cancellation_rate` is
  `cancelled / total` (`adaptive_scale_sla.py:251-257` vs `sla.rs:337-344`) and
  `success_rate` is `len(samples) / total` (`adaptive_scale_sla.py:150-156` vs
  `sla.rs:320-327`), both fractions. ITL and the linear-interpolation percentile
  rule also agree (p95 of [10,20,30,40,50]ms is 48.0 on both).
  One asymmetry, loud and in Rust's favour: Rust rejects an `error_rate`
  threshold outside `[0, 100]` and warns when `0 < t < 1`
  (`rust/runtime/src/adaptive_core/sla.rs:362-375`), where baseline Python has no
  threshold range validation on `SLAFilter`
  (`src/aiperf/config/sweep/adaptive.py:57-58`). Not a silent change.
- **Log-spaced recipe axes.** `logspace_int_steps` reproduces Python's
  banker's-rounding `round()` plus sort-and-dedupe
  (`src/aiperf/search_recipes/builtins.py` `_logspace_int_steps` vs
  `rust/cli/src/search.rs`), and the `concurrency-ramp`, `prefill-ttft-curve`,
  and `decode-itl-curve` default min/max/step values match, as does the
  hardcoded 8-step `max-concurrency-under-sla --search-style grid`.
- **Pareto scenario labels and directories.** `shape_{isl}_{osl}_c{conc}` labels
  and `isl_{isl}__osl_{osl}__concurrency_{conc}` directory names agree
  (`src/aiperf/search_recipes/_pareto_sweep.py:126-133` vs
  `rust/cli/src/search.rs:419-432`) — only the concurrency *set* differs
  (finding 2).
- **Accuracy CSV.** `accuracy_results.csv` header `task,correct,total,unparsed,accuracy`
  with a leading `OVERALL` row then alphabetically sorted tasks matches Python's
  ordering contract (`src/aiperf/accuracy/models.py:210-211` vs
  `rust/runtime/src/export/accuracy_csv.rs:6-8,66`).
- **Accuracy grading algorithm — no comparison is possible.** Rust implements no
  graders natively: `rust/runtime/src/accuracy_core/` is only `mod.rs`,
  `protocol.rs`, and `worker.rs`, with no exact-match normalization,
  answer-key handling, or multi-reference scoring. It delegates to a Python
  worker subprocess whose module does not exist at baseline (see finding 1). So
  baseline Python's thirteen graders
  (`src/aiperf/accuracy/graders/`: `exact_match.py`, `math.py`, `_math_strip.py`,
  `multiple_choice.py`, `lighteval_grader.py`, `gsm8k_grader.py`, `mmlu_pro.py`,
  `code_execution.py`, `_codegen_worker*.py`, `_choice_extract.py`) and MMLU's
  answer-key handling (`src/aiperf/accuracy/benchmarks/mmlu.py:147-148,202-204`)
  have no Rust counterpart to diverge from — there is no scoring-rule parity
  finding to make, in either direction. Rust's Wilson confidence interval on
  accuracy (`rust/runtime/src/metrics_core/accuracy.rs:489-494`) has no Python
  counterpart and is additive, out of scope.

## Withdrawn after baseline correction

### (was P1 #3) Adaptive `error_rate` SLA: fraction vs percentage, and the cancellation denominator

**Withdrawn — branch artifact.** Both halves of the claim fail against baseline.

The earlier revision cited a branch-local `error_rate_value` computing
`stats.errors / stats.total` — a fraction over successes + errors + cancellations.
Baseline computes the same value as Rust, in the same units, with the same
denominator:

```242:248:src/aiperf/timing/strategies/adaptive_scale_sla.py
        match stat:
            case "avg" | "min" | "max":
                completed = len(stats.samples) + stats.errors
                if completed == 0:
                    return 0.0
                return 100.0 * stats.errors / completed
        raise ValueError(f"Unsupported error_rate SLA stat: {stat}")
```

and says so explicitly:

```228:233:src/aiperf/timing/strategies/adaptive_scale_sla.py
        """Window error rate in percentage points, matching the exported metric.

        The exported ``request_error_rate`` metric is ``100 * errors /
        (successes + errors)``, so the adaptive-scale evaluator uses the same
        unit, and the same successes+errors denominator shape: a threshold of
        ``1`` means 1%, and cancellations are excluded.
```

Compare Rust (`rust/runtime/src/adaptive_core/sla.rs:328-336`):
`let completed = stats.completed() + stats.errors; 100.0 * stats.errors as f64 / completed as f64`.
Identical formula, identical unit, identical denominator. Neither sub-claim — the
100x scale difference nor the silent cancellation-denominator difference —
survives. Rust's warning at `sla.rs:369-374` telling users that thresholds are
percentage points is correct guidance that matches baseline semantics, not a
disclosure of a divergence.

The branch had regressed `error_rate_value` to a fraction (that file is +2/-34
relative to baseline); the audit read the regression as the reference.

### (was a "checked and consistent" entry) "Grading cannot drift because Rust delegates to the Python worker"

**Withdrawn.** The reassurance depended on `src/aiperf/accuracy/worker.py`, which
is branch-only (+890/-0, no baseline counterpart). At baseline there is no
`aiperf.accuracy.worker` module for Rust to delegate to. Replaced by the
"Accuracy grading algorithm — no comparison is possible" entry above, and folded
into finding 1 as supporting evidence.

## Unverified / needs runtime check

- **`tokens_per_second` SLO targeting.** Whether the Python metric registry maps
  this tag to a per-request metric (`output_token_throughput_per_request`) or an
  aggregate was not resolved from the registry alone. If the two sides bind it to
  different metrics, a `tokens_per_second` SLO would pass/fail differently.
  Needs: a run with a `tokens_per_second` SLO on both sides, comparing
  `good_request_count`.
- **Exact Rust error text for findings 7 and 8.** Both were confirmed to be
  structurally impossible to deserialize (absent field on a
  `deny_unknown_fields` struct; `Option<u32>` receiving a sequence), but the
  emitted message was not observed. Needs: `aiperf config validate` on a config
  with a `variables.*` sweep axis and on one with a bare magic list.
- **Whether the Rust search-history file and end-of-sweep comparison table match
  Python's schema field-for-field.** The aggregate filenames and the confidence
  metadata block were verified; the sweep table's column set and
  `rust/cli/src/search_history.rs` versus Python's search history were only
  spot-checked. Needs: a two-point sweep on both sides, diffing
  `profile_export_aiperf_sweep.csv` headers.
- **Zero-successful-run behavior.** Python raises
  (`src/aiperf/orchestrator/aggregation/confidence.py:128-134`) while Rust's
  `write_confidence_aggregate` has no such guard and would emit an aggregate with
  empty metrics. This overlaps the known exit-code classification gap
  (`docs/dev/python-rust-parity-gaps.md:264`) and was not pursued further here.
