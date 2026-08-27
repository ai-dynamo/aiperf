<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Config schema parity audit

Domain: the declarative configuration schema (Config v2 YAML) — key names, types,
defaults, validation, and resolution/precedence.

**Python baseline: `/mnt/4tb/aiperf-parity-py-main/src/aiperf/`, git rev
`bc359bf8fd` (`origin/main`).** Every `src/aiperf/...` path and line number in
this report refers to that tree. An earlier revision of this report cited a
feature branch 4345 commits ahead of `origin/main`; see
[Withdrawn after baseline correction](#withdrawn-after-baseline-correction).
Rust citations are unchanged (`origin/main` has no `rust/` tree).

Python side read: `config/` (`config.py`, `base.py`, `models.py`, `endpoint.py`,
`runtime.py`, `phases.py`, `ramp.py`, `artifacts.py`, `metrics.py`,
`tokenizer.py`, `dataset/` including `dataset/system_prompt.py`, `loader/`,
`resolution/`, `templates/`), plus the consumers that give a field its observable
meaning (`exporters/`, `workers/worker_manager.py`, `common/environment.py`,
`timing/strategies/fixed_schedule.py`, `dataset/composer/`, `plugin/plugins.yaml`).

Rust side read: `rust/cli/src/yaml.rs`, `rust/cli/src/load.rs`,
`rust/cli/src/expand.rs`, `rust/cli/src/config/templates_data.rs`,
`rust/runtime/src/config/` (`resolve.rs`, `validate.rs`, `phase_validate.rs`,
`system_prompt.rs`, `model/`), `rust/runtime/src/engine/protocol_v2.rs`,
`rust/runtime/src/fixed_schedule.rs`, `rust/runtime/src/engine/dataset_input.rs`,
`rust/runtime/src/dataset/compose.rs`.

Several findings below were additionally confirmed by running the built binary
(`rust/target/debug/aiperf config validate`) against baseline config files.
Note that `config validate` runs YAML deserialization and section projection but
not the full `rust/runtime/src/config/validate.rs` pass, so a "valid" result from
it proves only that the *loader* accepts the input.

## Summary

The highest-impact divergence is that **6 of the 27 templates shipped by upstream
Python do not load at all on the native binary**: they author `dataset.isl` /
`dataset.osl`, an upstream shorthand that Rust's `deny_unknown_fields`
`DatasetSection` has no counterpart for (finding 1, empirically confirmed). The
largest *silent* divergence is `artifacts.summary`: the one value both sides
accept (`json`) means "JSON summary plus the unconditionally-written CSV summary"
in Python but "JSON summary only" in Rust, so the 25 templates that author
`summary: [json]` stop emitting `profile_export_aiperf.csv` with no error
(finding 2). Third, the phase `type` discriminator is load-bearing in Python (a
discriminated union under `extra="forbid"`) but advisory in Rust: a rate phase
authored without `rate` silently degrades to a concurrency-1 closed loop, and a
`type: concurrency` phase carrying a `rate` silently becomes Poisson. Fourth,
`runtime.workers` omitted resolves to materially different counts (Python
`max(1, min(int(cpu*0.75)-1, 32))` then capped by the concurrency target; Rust
full `available_parallelism()`). On the good-news side, backlog item P1.1 ("Rust
Config v2 authoring is permissive where Python is strict") is substantially
fixed — every YAML section struct carries `deny_unknown_fields` except the
`models` residue — and the `dataset.system_prompt` /
`dataset.system_prompt_file` surface (a file this audit could not previously see)
is at parity apart from `~` expansion.

## Findings

### 1. Six of upstream's 27 shipped templates fail to load: `dataset.isl` / `dataset.osl` has no Rust counterpart

**Severity:** P1
**Status:** NEW (exposed only by the corrected baseline — the branch had already
rewritten exactly these six templates)

**Python evidence** — `src/aiperf/config/dataset/config.py:149-176` declares the
shorthand on `SyntheticDataset`:

```python
    isl: Annotated[
        ...
                "Shorthand sibling for `prompts.isl`. Accepts a fixed integer or "
                "distribution dict. Hoisted into `prompts.isl` by the before-"
    osl: Annotated[
```

hoisted at `src/aiperf/config/dataset/config.py:259-264`:

```python
    def _hoist_isl_osl_shortcuts(cls, data: Any) -> Any:
        """Hoist top-level isl/osl into prompts.{isl,osl} for direct validation.
```

Six shipped templates use it at the `dataset:` level:
`templates/gpu_telemetry.yaml:47-48`, `http_trace_metrics.yaml:58-59`,
`latency_test.yaml:42-43`, `ramping.yaml:47-48`, `time_based_soak.yaml`,
`warmup_profiling.yaml`.

**Rust evidence** — `isl` exists only inside `PromptsSection`
(`rust/cli/src/yaml.rs:1427`); `DatasetSection` carries
`#[serde(deny_unknown_fields)]` (`rust/cli/src/yaml.rs:1150-1152`) and declares no
`isl`/`osl`. `extract_prompts` (`rust/cli/src/yaml.rs:2832-2837`) reads only
`prompts.isl`.

**Observable user impact:** running all 27 baseline templates through the built
binary, 21 pass and 6 fail identically:

```
FAIL gpu_telemetry.yaml  :: unknown field `isl`, expected one of `name`, `type`, `dataset`, …
FAIL http_trace_metrics.yaml
FAIL latency_test.yaml
FAIL ramping.yaml
FAIL time_based_soak.yaml
FAIL warmup_profiling.yaml
```

The refusal is loud, which would normally cap this at P2. It is rated P1 because
the scope-defined check "do the shipped templates produce the same run?" fails
outright for 22% of them, the shorthand is documented in Python as a first-class
authoring form, and no migration notice exists. A reader who re-rates it P2 on
the loudness rule should still treat it as the top remediation item.

**Confidence:** High (empirically reproduced).

### 2. `artifacts.summary` accepts different values on each side, and the shared `json` value produces a different artifact set

**Severity:** P1
**Status:** CHANGED from the previous revision (the `genai_perf` vocabulary claim
was a branch artifact; the core divergence and the template blast radius survive,
with a corrected count)

**Python evidence** — `src/aiperf/config/artifacts.py:37`:

```python
SummaryExportFormat = Literal["json"]
```

`src/aiperf/config/artifacts.py:111-121`:

```python
    summary: Annotated[
        list[SummaryExportFormat] | Literal[False],
        Field(
            default_factory=lambda: ["json"],
            description="Summary export formats. "
            "Only 'json' is wired up to this field; the CSV summary is "
            "emitted regardless. Set to false to disable the summary JSON "
            "file only.",
        ),
    ]
```

The docstring is corroborated by the exporters.
`src/aiperf/exporters/metrics_json_exporter.py:28-31` gates only the JSON file:

```python
        summary = exporter_config.cfg.artifacts.summary
        if summary is False or "json" not in summary:
            ...
                "MetricsJsonExporter disabled: 'json' not in artifacts.summary"
```

`src/aiperf/exporters/metrics_csv_exporter.py:25-40` reads no `summary` at all —
it unconditionally targets `artifacts.profile_export_csv_file`; the per-phase
writer at `src/aiperf/exporters/exporter_manager.py:164-171` likewise writes the
CSV with no format gate.

**Rust evidence** — `rust/cli/src/yaml.rs:765-767`:

```rust
    /// Summary export formats (`[json,csv]`). Unauthored ships both.
    #[serde(default)]
    summary: Option<Vec<String>>,
```

`rust/cli/src/yaml.rs:2386-2401`:

```rust
        let summary_formats = match self.artifacts.as_ref().and_then(|a| a.summary.as_ref()) {
            Some(v) => {
                for f in v {
                    anyhow::ensure!(
                        f == "json" || f == "csv",
                        "artifacts.summary: unknown format {f:?} (expected `json` or `csv`)"
                    );
                }
```

`rust/runtime/src/config/model/export.rs:461-468`:

```rust
        let unauthored = summary_formats.is_empty();
        let json = unauthored || summary_formats.iter().any(|f| f == "json");
        let csv = unauthored || summary_formats.iter().any(|f| f == "csv");
```

and those two booleans are exactly the file gates
(`rust/runtime/src/config/model/export.rs:118-121`).

**Observable user impact:**

| authored | Python (baseline) | Rust |
| --- | --- | --- |
| omitted | `…aiperf.json` + `…aiperf.csv` | both |
| `[json]` | both | **`.json` only — CSV silently missing** |
| `[csv]` | hard error (not in `Literal["json"]`) | `.csv` only |
| `false` | JSON suppressed, CSV still written | hard error (type mismatch) |

The `[json]` row is the silent one, and it is what almost every shipped template
authors: **25 of the 27** `.yaml` files in `src/aiperf/config/templates/` contain
`summary: [json]` (e.g. `templates/ramping.yaml`; the two exceptions,
`minimal.yaml` and `inline_dataset.yaml`, omit `artifacts` entirely and so get
both files on both sides). Any downstream tooling that reads
`profile_export_aiperf.csv` breaks with no error and no warning.

**Confidence:** High for the `[json]` divergence (both gates read directly).
Medium-high that Python's summary CSV has no other suppression path — I read the
exporter constructor and both call sites but did not execute a Python run.

### 3. A rate-controlled phase authored without `rate` silently becomes a concurrency-1 phase in Rust

**Severity:** P1
**Status:** STILL VALID (NEW)

**Python evidence** — `src/aiperf/config/phases.py:563-570`:

```python
    @model_validator(mode="after")
    def validate_rate_source(self) -> Self:
        """Require exactly one of a scalar rate or a rate series."""
        if self.rate is None and self.rate_series is None:
            raise ValueError("rate-controlled phases require rate or rate_series")
```

**Rust evidence** — `rust/cli/src/yaml.rs:2952-2974`: the authored `type` only
selects an arrival distribution *if* a rate survived; otherwise the phase becomes
a concurrency phase regardless of what `type` said:

```rust
        let default_concurrency = section.concurrency.unwrap_or(1);
        if let Some(rate) = rate {
            match rate_mode.as_deref() {
                Some("gamma") => PhaseKind::Gamma { … },
                Some("constant") => PhaseKind::Constant { … },
                _ => PhaseKind::Poisson { rate, concurrency: section.concurrency },
            }
        } else {
            PhaseKind::Concurrency {
                concurrency: default_concurrency,
            }
        }
```

The later guard cannot catch it, because it keys off the *resolved* kind, which is
no longer a rate kind — `rust/runtime/src/config/phase_validate.rs:151-153`:

```rust
    } else if is_rate_phase && scalar_rate.is_none_or(|rate| !(rate.is_finite() && rate > 0.0)) {
        anyhow::bail!("rate-controlled phases require rate or rate_series");
    }
```

Confirmed empirically: `phases: {type: poisson, requests: 100}` (no `rate`) is
reported valid by the native loader.

**Observable user impact:** a dropped or misspelled `rate:` errors out under
Python but under Rust runs a concurrency-1 closed-loop benchmark to completion and
reports it as a normal run. Throughput, latency, and concurrency numbers are all
for a different load model than the one authored. No warning is emitted.

**Confidence:** High.

### 4. `fixed_schedule` with only `endOffset` set flips `auto_offset` off in Rust

**Severity:** P1
**Status:** STILL VALID (NEW)

**Python evidence** — `src/aiperf/config/phases.py:666-673`:

```python
    auto_offset: Annotated[
        bool,
        Field(
            default=True,
            description="Normalize trace timestamps to start at 0. "
            "Subtracts minimum timestamp from all entries.",
        ),
    ]
```

`src/aiperf/config/phases.py:693-702` rejects only the `auto_offset` +
`start_offset` combination; `end_offset` alone leaves the `True` default in place.
Schedule zero then comes from the first entry —
`src/aiperf/timing/strategies/fixed_schedule.py:111-117`:

```python
        if self._config.auto_offset_timestamps:
            self._schedule_zero_ms = self._absolute_schedule[0].timestamp_ms
        elif self._config.fixed_schedule_start_offset is not None:
            self._schedule_zero_ms = float(self._config.fixed_schedule_start_offset)
        else:
            self._schedule_zero_ms = 0.0
```

**Rust evidence** — `rust/cli/src/yaml.rs:2944-2951`:

```rust
    } else if phase_type == Some("fixed_schedule") {
        PhaseKind::FixedSchedule {
            auto_offset: section
                .auto_offset
                .unwrap_or(section.start_offset.is_none() && section.end_offset.is_none()),
```

The consuming logic is byte-identical to Python's
(`rust/runtime/src/fixed_schedule.rs:115-119`), so the whole divergence is the
resolved boolean.

**Observable user impact:** for `phases: {type: fixed_schedule, endOffset: 60000}`
Python normalizes and fires the first request at t=0; Rust leaves schedule zero at
0.0 and delays the first request by the trace's own first timestamp. For a trace
carrying absolute epoch milliseconds (a common mooncake/bailian shape) that is an
effectively unbounded initial idle — the run appears to hang rather than error.
For a trace already based at 0 the two agree, which is why this hides.

**Confidence:** High on the resolved-boolean divergence; the severity of the
downstream effect depends on the trace's timestamp base.

### 5. `runtime.workers` omitted resolves to different worker counts

**Severity:** P1
**Status:** STILL VALID (NEW) — confirmed upstream, not branch code. `config/runtime.py`'s
only branch-side delta is an added `cells` field; `workers/worker_manager.py` and
`common/environment.py` are byte-identical between baseline and branch.

**Python evidence** — `src/aiperf/config/runtime.py:76-86` declares
`workers: int | None = None` ("null = auto-detect based on CPU cores"), and
`src/aiperf/workers/worker_manager.py:65-91` is the resolver:

```python
        self.max_workers = runtime.workers
        if self.max_workers is None:
            # Default to 75% of the CPU cores - 1, with a cap of Environment.WORKER.MAX_WORKERS_CAP, and a minimum of 1
            self.max_workers = max(
                1,
                min(
                    int(self.cpu_count * Environment.WORKER.CPU_UTILIZATION_FACTOR) - 1,
                    Environment.WORKER.MAX_WORKERS_CAP,
                ),
            )
        # Cap the worker count to the max concurrency, but only if the user is in concurrency mode.
        if self.max_concurrency and self.max_concurrency < self.max_workers:
            self.max_workers = self.max_concurrency
```

with `CPU_UTILIZATION_FACTOR = 0.75`
(`src/aiperf/common/environment.py:1623-1629`) and `MAX_WORKERS_CAP = 32`
(`src/aiperf/common/environment.py:1654-1659`).

**Rust evidence** — `rust/runtime/src/engine/protocol_v2.rs:240-247`:

```rust
fn default_worker_count() -> u64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(1)
}
```

applied at `rust/runtime/src/engine/protocol_v2.rs:384-394` with no CPU factor,
no absolute cap, and no concurrency cap.

**Observable user impact:** the same `runtime: {}` (or no `runtime:` block) on a
64-core host yields 32 workers under Python and 64 under Rust; on a 144-core host,
32 vs 144. With `--concurrency 4` Python collapses to 4 workers while Rust still
starts `W` workers. Under the default `global` dispatch the aggregate admission
gate keeps total in-flight requests correct, so the leak is mostly in per-worker
partitioning (dataset shard boundaries, per-worker rate pacing, record
interleaving, connection-pool fan-out); under `sharded` — the default when
`runtime.cells > 1`, per `rust/runtime/src/engine/protocol_v2.rs:265-278` — the
request budget and rate are statically divided by a different `W`, so pacing
granularity and per-worker residues differ outright.

**Confidence:** High on the resolved counts. Medium on the magnitude of the
observable metric difference under the default `global` dispatch (see Unverified).

### 6. `models.items[]` silently ignores unknown keys — the one remaining typo trap in the Rust loader

**Severity:** P1
**Status:** STILL VALID. KNOWN(now-fixed) *for the general claim*, NEW *for this
residue*. Backlog P1.1 says "Rust YAML parsing intentionally ignores unknown keys
in several authoring structs". That is no longer true in general: every top-level
and section struct in `rust/cli/src/yaml.rs` carries
`#[serde(deny_unknown_fields)]` (`ConfigFile` at `:631-633`, `EndpointSection` at
`:1082-1084`, `DatasetSection` at `:1150-1152`, `PhaseSection` at `:1493-1495`,
`ArtifactsSection` at `:754-756`, `RuntimeSection`, `SynthesisSection`,
`UserFileSection`, `CancellationSection`, `SlaFilterSection`,
`AdaptiveScaleBlock`, …). The `models` section is the exception.

**Python evidence** — `src/aiperf/config/models.py:55` and `:115`
(both files byte-identical to the branch, so this evidence is unaffected):

```python
class ModelItem(BaseConfig):
    model_config = ConfigDict(extra="forbid")
```

```python
class ModelsAdvanced(BaseConfig):
    model_config = ConfigDict(extra="forbid")
```

with real fields `name`, `weight`, `lora`, `modalities`, `tokenizer`
(`src/aiperf/config/models.py:57-104`).

**Rust evidence** — `rust/cli/src/yaml.rs:1040-1046` (the mapping branch of
`ModelItem`'s hand-written `Deserialize`) and `rust/cli/src/yaml.rs:996-1001` (the
mapping branch of `ModelsSection`), neither of which sets `deny_unknown_fields`:

```rust
                #[derive(Deserialize)]
                struct Full {
                    name: String,
                }
                let full = Full::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(ModelItem { name: full.name })
```

Per-item `weight` is therefore never read;
`rust/runtime/src/config/resolve.rs:1020-1030` hardcodes it:

```rust
    let models = Models {
        strategy: inputs.model_strategy.unwrap_or(ModelStrategy::RoundRobin),
        items: inputs
            .model_names
            .iter()
            .map(|name| ModelItem {
                name: name.clone(),
                weight: None,
            })
```

**Observable user impact:** `models: {items: [{name: a, weigt&#104;: 0.7}, …]}` — a
typo, an unsupported key, or a deliberate `lora`/`tokenizer`/`modalities`
override — is rejected by Python and silently dropped by Rust. The
`strategy: weighted` case does *not* silently mis-route: the dropped weights make
`rust/runtime/src/engine/protocol_v2.rs:848-852` fail closed
("`models.items[N].weight` is required for weighted selection"), and Python's own
`weighted` path raises at config validation
(`src/aiperf/config/models.py:134-155`: "All models must have weights specified
when using weighted strategy"), so the two agree there. The trap is the
silently-accepted typo. Python additionally requires the weights to sum to
`1.0 ± 0.01` (`config/models.py:152-155`), a rule Rust can never reach from YAML
because it never reads the field.

**Confidence:** High.

### 7. `type: concurrency` carrying a `rate` silently becomes a Poisson rate phase in Rust

**Severity:** P2
**Status:** STILL VALID (NEW)

**Python evidence** — `src/aiperf/config/phases.py:506-517`: `ConcurrencyPhase`
declares no `rate` field, and `BasePhaseConfig.model_config` is
`ConfigDict(extra="forbid")` (`src/aiperf/config/phases.py:75`), so `rate` on a
concurrency phase is an "extra inputs are not permitted" error.

**Rust evidence** — `rust/cli/src/yaml.rs:2912-2915` computes `rate_mode` only for
the three rate types, and `rust/cli/src/yaml.rs:2965-2968` falls through to
Poisson for anything else with a rate present:

```rust
                _ => PhaseKind::Poisson {
                    rate,
                    concurrency: section.concurrency,
                },
```

**Observable user impact:** a config Python refuses runs on Rust as an open-loop
Poisson arrival benchmark with the declared `concurrency` demoted to a cap.
Direction is Python-strict/Rust-permissive, so it only bites configs authored
against Rust and later validated against Python, or a user who believed
`type: concurrency` was authoritative.

**Confidence:** High.

### 8. Baseline phase cross-field constraints with no Rust counterpart

**Severity:** P2
**Status:** NEW (exposed by the corrected baseline)

**Python evidence** — `src/aiperf/config/phases.py:449-460`:

```python
        if (
            self.prefill_concurrency is not None
            and self.concurrency is not None
            and self.prefill_concurrency > self.concurrency
        ):
            raise ValueError(
                f"Phase '{self.name}': prefill_concurrency must be <= concurrency"
            )
        if self.grace_period is not None and self.duration is None:
            raise ValueError(
                f"Phase '{self.name}': grace_period requires duration to be set"
            )
```

**Rust evidence** — `rust/runtime/src/config/validate.rs:24-36` enumerates the
whole validation pass; it contains no `prefill_concurrency <= concurrency` check
and no `grace_period`-requires-`duration` check. The only
`prefill_concurrency` rule is the streaming requirement
(`rust/runtime/src/config/validate.rs:132-140`). Both were confirmed accepted by
the native loader:

```
{type: concurrency, concurrency: 4, prefillConcurrency: 16, requests: 10}  => valid
{type: concurrency, concurrency: 4, requests: 10, gracePeriod: 5}          => valid
```

**Observable user impact:** `gracePeriod` on a request-bounded phase is accepted
and inert in Rust (grace only gates the duration deadline) where Python refuses
the config outright — a silently-ignored field, the P0-class shape, held at P2
only because the mis-authored knob has no effect on the measurement rather than a
wrong one. `prefillConcurrency > concurrency` is likewise accepted and clamped by
the smaller real concurrency instead of erroring.

**Confidence:** High on the strictness divergence (static plus empirical). Medium
on the exact runtime effect of each accepted value; I did not run a benchmark.

**Note on `_stop_condition_required`.** Baseline
`src/aiperf/config/phases.py:439-448` also requires one of
`requests`/`duration`/`sessions` per phase (waived for `FixedSchedulePhase` at
`:659`). Rust *does* implement this, at
`rust/runtime/src/config/validate.rs:93-113`, with the same
`fixed_schedule` waiver — so it is at parity. `aiperf config validate` accepts a
stop-condition-less phase only because that command does not run the
`validate.rs` pass; `aiperf profile` does. This is a `config validate` coverage
gap, not a schema divergence.

### 9. `endpoint.url` scheme and host validation has no Rust counterpart

**Severity:** P2
**Status:** NEW (was an Unverified item in the previous revision; the baseline
correction changed the Python side and the claim is now verified)

**Python evidence** — `src/aiperf/config/endpoint.py:504-517`:

```python
            # Reject anything that lacks a scheme, a netloc, or a hostname.
            # ``http://:18765`` parses as scheme=http, netloc=':18765', hostname=None
            if not parsed.scheme or not parsed.netloc or not parsed.hostname:
                    f"URL {url!r} is missing scheme or host. "
            if parsed.scheme.lower() not in ("http", "https"):
                    f"URL {url!r} has unsupported scheme {parsed.scheme!r}. "
```

The scheme allowlist is exactly `("http", "https")`. Scheme-less values get
`http://` prepended first (`src/aiperf/config/endpoint.py:115-126` →
`src/aiperf/config/loader/parsing.py:45-70`).

**Rust evidence** — `rust/runtime/src/config/resolve.rs:2138-2145` is the whole of
Rust's URL normalization:

```rust
pub(crate) fn normalize_url(url: &str) -> String {
    if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    }
}
```

There is no config-time scheme allowlist and no hostname requirement. Confirmed
against the built binary:

```
http://localhost:8000   => valid      localhost:8000        => valid
ftp://localhost:8000    => valid      grpc://localhost:8001 => valid
mailto:x                => valid      http://:18765         => valid
```

**Observable user impact:** `mailto:x` has no `://`, so Rust rewrites it to
`http://mailto:x` and carries that into connect; Python rejects it at load.
`http://:18765` (no hostname) is a load error in Python and accepted by Rust.
`grpc://` is legitimately Rust-only capability and out of scope; `ftp://` and the
hostname-less form are the real gap. All eventually surface as connect failures
rather than silent wrong results, hence P2.

**Confidence:** High (empirically reproduced).

### 10. Enum spellings: Python normalizes case and `-`/`_`; Rust requires the exact canonical string

**Severity:** P2
**Status:** STILL VALID (NEW)

**Python evidence** — `src/aiperf/common/enums/base_enums.py:10` and `:86-102`
(file byte-identical to the branch):

```python
def _normalize_name(value: str) -> str:
    return value.lower().replace("-", "_")
```

```python
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized_value = _normalize_name(value)
            for member in cls:
                if _normalize_name(member.value) == normalized_value:
                    return member
```

Every enum-typed config field built on `CaseInsensitiveStrEnum` (or on
`plugin/enums.py`'s generated enums) therefore accepts `ROUND_ROBIN`,
`round-robin`, `Round_Robin`, `FIXED-SCHEDULE`, `Pooled`, …

**Rust evidence** — exact-match parsers, e.g. `rust/cli/src/load.rs:1294-1301`:

```rust
pub(crate) fn parse_model_strategy(s: &str) -> anyhow::Result<ModelStrategy> {
    Ok(match s {
        "round_robin" => ModelStrategy::RoundRobin,
        "random" => ModelStrategy::Random,
        "weighted" => ModelStrategy::Weighted,
        other => anyhow::bail!("unknown --model-selection-strategy {other:?}"),
```

`rust/cli/src/load.rs:1304-1310` (`parse_connection_reuse`) and
`rust/cli/src/yaml.rs:2904-2911` (phase `type`) are the same shape.

**Observable user impact:** loud refusal, not silent — but it is an undocumented
strictness change on a large surface. Also note the Python `PhaseType` enum is
generated from the arrival-pattern *and* timing-strategy plugin registries
(`src/aiperf/plugin/enums.py:173-186`), so names like `agentic_replay` are legal
phase types in Python and rejected outright by the six-way match in
`rust/cli/src/yaml.rs:2904-2911`. `endpoint.urlStrategy` is the well-handled
counter-example: Rust rejects anything but `round_robin` rather than silently
downgrading (`rust/cli/src/yaml.rs:1883-1890`).

**Confidence:** High.

### 11. Ramp map form `{duration, strategy}` is rejected by Rust; only the scalar form exists

**Severity:** P2
**Status:** STILL VALID. KNOWN(still-true) — backlog P1.3 lists "ramp strategy"
among the confirmed projection losses.

**Python evidence** — `src/aiperf/config/ramp.py:17-45` (a `RampConfig` with
`duration` plus `strategy: RampType = LINEAR` at `:35-40`) and `:47-60`, where the
scalar shorthand expands to `{"duration": …}` and therefore also gets `linear`
(file byte-identical to the branch).

**Rust evidence** — `rust/cli/src/yaml.rs:612-629`: `de_duration_opt` deserializes
an untagged `Num | Str` only, so a mapping is a deserialization error.
`rust/cli/src/yaml.rs:2995-3006` then hardcodes the strategy:

```rust
            concurrency_ramp: section.concurrency_ramp.map(|duration| Ramp {
                duration,
                strategy: "linear".into(),
            }),
```

**Observable user impact:** the scalar form (`concurrencyRamp: 10s`) is at parity.
`concurrencyRamp: {duration: 10, strategy: exponential}` is accepted by Python and
is a load error in Rust, so `exponential`/`poisson` ramp curves are unreachable
from a native config file. Loud, hence P2.

**Confidence:** High.

### 12. Python-only config keys: two are accepted-and-ignored, the rest hard-fail

**Severity:** P2
**Status:** CHANGED from the previous revision — the accepted-key set was
re-derived from baseline and three previously-listed `endpoint.*` keys turned out
to be branch-only. KNOWN(still-true, narrowed) against backlog P1.3/P1.4.

**Rust evidence** — the only two keys accepted and deliberately no-op'd are
enumerated at `rust/cli/src/yaml.rs:101-114` and warned once at `:117-130`:

```rust
const UNIMPLEMENTED_KEYS: &[(&str, fn(&ConfigFile) -> bool)] = &[
    ("plot", |c| c.plot.is_some()),
    ("runtime.ui", |c| { … }),
];
```

Everything else Python-only is a `deny_unknown_fields` error. The set below was
derived from the baseline field declarations and confirmed one key at a time
against the built binary:

- Top level: `no_sweep_table` (`src/aiperf/config/config.py:770-869`).
- `benchmark.logging`, `benchmark.metrics`, `benchmark.accuracy`
  (`src/aiperf/config/config.py:117-770`). `benchmark.metrics` is notable:
  Python's `MetricsConfig` is an intentionally empty `extra="forbid"` section
  (`src/aiperf/config/metrics.py:13-22`), so `metrics: {}` is legal there and an
  unknown key in Rust.
- `runtime.record_processors`, `service_run_type`, `communication`, `api_port`,
  `api_host`, `dataset_api_base_url`, `workers_per_pod`,
  `record_processors_per_pod`, `stats_interval`
  (`src/aiperf/config/runtime.py:88-196`). `runtime.workers`, `workers_min`, and
  `ui` are the three that survive into Rust's `RuntimeSection`
  (`rust/cli/src/yaml.rs:940-959`).
- `endpoint.transport`, `endpoint.template`, `endpoint.uuid_and_strip`
  (`src/aiperf/config/endpoint.py:212-221`, `:267-276`, `:344-360`) vs
  `EndpointSection` (`rust/cli/src/yaml.rs:1084-1148`). Both survivors moved
  rather than vanished: transport selection is a sibling `benchmark.transport`
  block in Rust, and `uuid_and_strip` moved to `dataset.uuidAndStrip`
  (`rust/cli/src/yaml.rs:1229-1230`). A Python config authoring either at its
  Python path gets an unknown-field error with no hint of the new location.
- `artifacts.auto_plot`, `artifacts.plot_required`
  (`src/aiperf/config/artifacts.py:178-201`) vs `ArtifactsSection`
  (`rust/cli/src/yaml.rs:756-786`).
- Per-phase `seamless`, `timing_mode`, `failed_request_threshold`,
  `trajectory_start_min_ratio`, `trajectory_start_max_ratio`,
  `burst_phase_starts`, `system_idle_gap_cap_seconds`,
  `agentic_warmup_grace_period` (`src/aiperf/config/phases.py:134-394`) vs
  `PhaseSection` (`rust/cli/src/yaml.rs:1493-1576`). Rust's phase model *has*
  these fields but the YAML path hardcodes them (`rust/cli/src/yaml.rs:2983`,
  `:2989`, `:3011-3013`), so they are reachable only from CLI flags or a
  top-level key, never from a phase block.

**Observable user impact:** loud, so mostly out of scope — recorded because a
Python config file is not portable to the native binary without edits, and
because `plot` / `runtime.ui` are the only two silently-inert keys (they do emit a
`tracing::warn!`, which is easy to miss under `--ui none` but is not strictly
silent).

**Confidence:** High (empirically reproduced key by key).

### 13. Phase `name` is required in Python's list form and defaulted in Rust

**Severity:** P2
**Status:** STILL VALID (NEW)

**Python evidence** — `src/aiperf/config/phases.py:77-90`:
`name: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")]` with no
default. The `name` is injected only for the single-mapping shorthand
(`src/aiperf/config/loader/normalizers.py:116-120`) and for the
`warmup:`/`profiling:` envelope form (`:75-91`); an explicit `phases:` **list**
entry must name itself.

**Rust evidence** — `rust/cli/src/yaml.rs:1497` (`name: Option<String>`) and
`rust/cli/src/yaml.rs:2898-2901`:

```rust
    let name = section
        .name
        .clone()
        .unwrap_or_else(|| "profiling".to_string());
```

Confirmed empirically: `phases: [{type: concurrency, concurrency: 4, requests: 10}]`
loads.

**Observable user impact:** a single unnamed list entry is a validation error in
Python and a valid single profiling phase in Rust. A *two*-entry unnamed list is
caught by Rust's duplicate-name check
(`rust/runtime/src/config/phase_validate.rs:49-52`), so the divergence is confined
to the single-entry case. Direction is Rust-permissive.

**Confidence:** High.

### 14. `dataset.system_prompt_file` does not expand `~` in Rust

**Severity:** P2
**Status:** NEW (from `config/dataset/system_prompt.py`, a baseline file the
previous revision could not see)

**Python evidence** — `src/aiperf/config/dataset/system_prompt.py:84-91` reads the
file through `safe_read_template_path`, whose first step is
`src/aiperf/common/path_safety.py:37`:

```python
        path = Path(ts).expanduser()
```

**Rust evidence** — `rust/runtime/src/config/system_prompt.rs:70-77` is the whole
of Rust's path resolution:

```rust
fn absolute_prompt_path(path: &Path) -> Result<SystemPromptError> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|current| current.join(path))
```

`~` is treated as an ordinary path component by the `openat` walk at
`:144-169`. Confirmed empirically:

```
systemPromptFile: "~/sysprompt.txt"                 => could not be read: … (os error 2)
systemPromptFile: "/home/anthony/sysprompt.txt"     => valid
```

**Observable user impact:** a config that reads a home-relative system prompt
loads in Python and is a startup error in Rust. Loud, hence P2.

**Confidence:** High (empirically reproduced).

## Checked and consistent

Verified equal on both sides by reading both implementations against baseline
`bc359bf8fd`:

- **`dataset.system_prompt` / `dataset.system_prompt_file` (audited fresh).**
  Baseline `src/aiperf/config/dataset/system_prompt.py` (the file the previous
  revision could not see) is at parity with
  `rust/runtime/src/config/system_prompt.rs` and
  `rust/runtime/src/config/validate.rs:210-263` on every axis except finding 14:
  - Field names and defaults: both `None`
    (`system_prompt.py:31-55` vs `rust/runtime/src/config/model/dataset.rs:365,
    479, 527` — Rust carries them on all three dataset variants, matching the
    Python mixin's use by `SyntheticDataset`/`FileDataset`/`PublicDataset` at
    `dataset/config.py:89, 285, 786`).
  - Precedence between the two sources: neither wins — both sides reject having
    both set (`system_prompt.py:69-73` vs `system_prompt.rs:29-35`).
  - Whitespace-only text and whitespace-only file contents rejected on both
    (`system_prompt.py:76-97` vs `system_prompt.rs:36-44, 59-66`).
  - File must be a readable regular UTF-8 file with no symlinked path component
    on both (`path_safety.py:40-55` vs `system_prompt.rs:79-195`, including the
    per-component `O_NOFOLLOW` walk).
  - Read once at config-validation time on both, so the benchmark cannot observe
    a mid-run edit (`system_prompt.py:61-68` vs the
    `file_prompt_is_owned_after_one_resolution` test at `system_prompt.rs:222-233`).
  - Join semantics: verbatim text first, joined to any authored system message by
    exactly `"\n\n"` — `SYSTEM_PROMPT_JOIN_SEP = "\n\n"`
    (`src/aiperf/common/constants.py:40`) applied at
    `src/aiperf/dataset/composer/base.py:614-620`, versus
    `rust/runtime/src/dataset/compose.rs:335-351` and its
    `"verbatim\n\nauthored system"` assertion at `:663`.
  - Additive to synthetic user ISL on both (Python's rationale at
    `dataset/composer/base.py:225-233`; Rust's
    `verbatim_system_prompt_is_additive_to_synthetic_user_isl` at
    `rust/runtime/src/dataset/loader/synthetic.rs:1115-1119`).
  - Endpoint allowlist identical: Python gates on the
    `consumes_system_message` plugin flag, set by exactly `chat`, `responses`,
    `messages`, `chat_embeddings` (`src/aiperf/plugin/plugins.yaml:215, 256, 274,
    291`, enforced at `config/config.py:605-619`); Rust hardcodes the same four
    (`rust/runtime/src/config/validate.rs:252-261`).
  - Mutual exclusivity with `prefix_prompts.shared_system_length` / `pool_size` /
    `length` on both (`config/config.py:588-603` vs
    `rust/runtime/src/config/validate.rs:226-245`), and both scope it to the
    synthetic dataset because `prefix_prompts` exists only there
    (`config/dataset/config.py:177`).
  - Both accept a verbatim prompt as satisfying
    `cache_bust=warmup_isolation_system` (`config/config.py:697-716` vs
    `rust/runtime/src/config/validate.rs:267-288`).
- **Unknown-field strictness (Python half re-confirmed from baseline).** Backlog
  P1.1 is now substantially fixed. Every YAML section struct in
  `rust/cli/src/yaml.rs` carries `deny_unknown_fields` (see finding 6 for the list
  and the one exception), matching baseline `extra="forbid"` on `AIPerfConfig`
  (`config/config.py:786`), `BenchmarkConfig` (`:158`), `EndpointConfig`
  (`config/endpoint.py:113`), `RuntimeConfig` (`config/runtime.py:39`),
  `ArtifactsConfig` (`config/artifacts.py:81`), `BasePhaseConfig`
  (`config/phases.py:75`), `ModelItem`/`ModelsAdvanced`
  (`config/models.py:55, 115`), `MetricsConfig` (`config/metrics.py:22`), and
  `TemplateConfig` (`config/endpoint.py:84`). Rust also declares `sweep`,
  `variables`, `multiRun`, `schemaVersion`, and `datasets[].name` explicitly so
  pre-processed keys stay legal rather than becoming unknown-key errors
  (`rust/cli/src/yaml.rs:631-666`, `:1158-1160`).
- **camelCase/snake_case dual acceptance.** Python generates camelCase aliases
  with `populate_by_name=True` (`config/base.py:21-24`); Rust adds an explicit
  `#[serde(alias = "…")]` per field. Spot-checked across `endpoint`, `dataset`,
  `runtime`, and `artifacts` via the binary's own "expected one of" lists, which
  enumerate both spellings.
- **`endpoint.timeout`**: default `21600.0` on both — `TIMEOUT = 6 * 60 * 60`
  (`config/endpoint.py:58`) applied at `config/endpoint.py:189-198`, versus
  `rust/runtime/src/config/resolve.rs:75, 1051`. `0` means "no timeout" on both
  (Python's field doc; Rust's
  `rust/runtime/src/transport/http/transport/http_transport.rs:702-704`
  `positive_timeout` filters `> 0`).
- **`endpoint.type`** default `chat` (`config/endpoint.py:54, 147-158` vs
  `rust/cli/src/yaml.rs:1862-1867`).
- **`endpoint.url` / `urls`**: required on both, singular-to-plural shorthand on
  both, and `http://` prepended when the value has no scheme
  (`config/endpoint.py:115-126`, `config/loader/parsing.py:45-70` vs
  `rust/cli/src/yaml.rs:1089-1090, 1892-1897` and
  `rust/runtime/src/config/resolve.rs:2138-2145`). Scheme/host *validation*
  diverges — see finding 9.
- **`endpoint.url_strategy`** default `round_robin` (`config/endpoint.py:57,
  138-145` vs `rust/cli/src/yaml.rs:1883-1890`).
- **`dataset.entries`** default `100` (`config/dataset/config.py:106-116` vs
  `rust/cli/src/load.rs:38` + `rust/cli/src/yaml.rs:2236`).
- **`artifacts.records`** default `["jsonl"]` and `records: false` disabling
  per-record export — both supported, including the explicit refusal of
  `records: true` (`config/artifacts.py:122-130` vs
  `rust/cli/src/yaml.rs:810-828, 2379-2383`). The *vocabulary* is
  Rust-wider: baseline is `RecordsExportFormat = Literal["jsonl"]`
  (`config/artifacts.py:38`) while Rust also accepts `csv` and `parquet`. That is
  a Rust-only capability, hence out of scope, but it means `records: [parquet]` is
  a Python load error.
- **`artifacts.dir`** default `./artifacts` (`config/artifacts.py:83-90` and
  `OutputDefaults.ARTIFACT_DIRECTORY` at `:43`; Rust takes the same default).
- **`tokenizer`**: `revision` defaults to `main`, `trust_remote_code` and
  `apply_chat_template` default `false` (`config/tokenizer.py:52-77` vs
  `rust/runtime/src/config/model/tokenizer.rs:10-30`). File byte-identical to the
  branch.
- **Phase workflow validation.** Both enforce: at least one phase; non-empty and
  pattern-matching names; Windows-reserved names rejected; case-insensitive unique
  names; `warmup`/`profiling` name↔kind coherence; explicit `kind` required for
  non-canonical names; exactly-consistent `exclude_from_results` versus kind
  (both *error* on an inconsistent authored value rather than coercing); at least
  one profiling phase; `seamless` forbidden on the first phase; a per-phase stop
  condition waived for `fixed_schedule` (`config/phases.py:405-461`, `:394`,
  `:659` vs `rust/runtime/src/config/validate.rs:24-131` and
  `rust/runtime/src/config/phase_validate.rs:34-100`). Two cross-field rules in
  the same Python validator have no Rust counterpart — see finding 8.
- **`--benchmark-duration` grace default** `30.0 s` on both
  (`config/flags/cli_config.py:2113-2126` vs
  `rust/runtime/src/config/phase_validate.rs:13, 238-243`). A YAML-authored
  `duration:` with no `gracePeriod:` leaves grace unset on both.
- **Sweep/multi-run consistent seed** auto-fills `42` on both
  (`config/config.py:1044-1057` vs `rust/cli/src/sweep/run.rs:15` +
  `rust/cli/src/profile.rs:1473-1488`), gated by the same `set_consistent_seed`
  toggle (`rust/cli/src/yaml.rs:497-498`).
- **Top-level `randomSeed` reaches dataset RNG.** Python falls back from
  `dataset.random_seed` to the run seed
  (`src/aiperf/dataset/composer/synthetic.py:41-43`); Rust does the same via
  `spec.random_seed.map_or(context.run_rng_root, …)`
  (`rust/runtime/src/engine/dataset_input.rs:925-927, 974-976, 1023-1025` with
  the root built from `run.identity.random_seed` at
  `rust/runtime/src/engine/online_execution.rs:1265`). Despite the comment at
  `rust/runtime/src/config/resolve.rs:431-434`, dataset determinism is not lost.
- **`${VAR}` / `${VAR:default}` substitution**: same regex shape, same
  missing-variable error, same unterminated-`${` error, and the same whole-string
  scalar coercion (bool → int → float → string)
  (`config/loader/env_vars.py:20-139` — file byte-identical to the branch — vs
  `rust/cli/src/expand.rs:28-113`).
- **`dataset`/`model`/`phases` shorthand normalization.** Both expand
  `model: str` → `models.items[0].name`, `dataset:` → `datasets: [{name: default,
  …}]`, and a flat `phases: {type: …}` → a one-entry list
  (`config/loader/normalizers.py:105-121` — byte-identical to the branch — vs
  `rust/cli/src/yaml.rs` `yaml_phase_to_model` and the `datasets`/`models`
  fallbacks).
- **`artifacts.userFiles`** is a list of file entries on both
  (`config/artifacts.py:169-176` vs `rust/cli/src/yaml.rs:784-785`).
- **Shipped templates are byte-identical where they load.**
  `rust/cli/src/config/templates_data.rs` `include_str!`s files out of
  `src/aiperf/config/templates/`, so template *content* cannot drift — which is
  exactly why findings 1 and 2 land on many of them at once.

## Withdrawn after baseline correction

Claims from the previous revision of this report that came from the feature
branch rather than `origin/main`, and are hereby withdrawn:

1. **`artifacts.summary` accepts `genai_perf`.** The branch widened
   `SummaryExportFormat` to `Literal["json", "genai_perf"]`. Baseline is
   `Literal["json"]` (`config/artifacts.py:37`). The "disjoint vocabularies"
   framing is therefore wrong: baseline Python accepts `["json"]` or `false`,
   Rust accepts `json`/`csv`, and the overlap is `json`. The CSV-emission half of
   the finding survives unchanged — see finding 2.
2. **"26 of the 28 shipped templates."** The branch added
   `templates/dynosim_offline_replay.yaml`, so the denominator was inflated.
   Baseline has 27 `.yaml` templates and 25 author `summary: [json]`.
3. **`endpoint.http2`, `endpoint.connection_limit`, `endpoint.keepalive_timeout`
   are Python-only keys Rust rejects.** None of the three exists in baseline
   `config/endpoint.py`; all were branch additions. The Python-only endpoint key
   set is exactly `transport`, `template`, `uuid_and_strip` — see finding 12.
4. **Python's URL scheme allowlist is `{http, https, grpc, grpcs, dynosim}`.**
   That was branch code. Baseline is `("http", "https")`
   (`config/endpoint.py:513`). The finding is now stronger and verified — see
   finding 9.
5. **The `agentic_cache_warmup_duration` per-phase key is Python-only.** It is a
   branch-only field on `BasePhaseConfig`; baseline `config/phases.py` does not
   declare it. It is removed from finding 12's key list. (Rust *does* carry it,
   at `rust/runtime/src/config/validate.rs:105`, so this direction was inverted.)

No whole finding was withdrawn; two findings (2 and 12) were reclassified CHANGED
because their evidence or blast radius shifted. `config/dynosim.py`, the
branch-only file, was never cited.

## Unverified / needs runtime check

- **Magnitude of finding 5 under the default `global` dispatch.** The resolved
  worker counts provably differ; whether the reported aggregate metrics differ
  materially at `workers = 32` vs `workers = 144` needs an A/B run of the same
  config on both engines with a deterministic `aiperf-mock-server`, comparing
  per-record admission ordering and rate-slot residues. I did not run one.
- **Whether Python's summary CSV can be suppressed by any other path.** I read
  `MetricsCsvExporter.__init__` (`exporters/metrics_csv_exporter.py:25-40`) and
  the phase-export call site (`exporters/exporter_manager.py:164-171`) and found
  no `artifacts.summary` gate, but I did not enumerate every exporter-registry
  entry point, so "CSV emitted regardless" rests on those two sites plus the
  field's own docstring.
- **`artifacts.prefix` suffix-stripping parity.** Python strips a specific suffix
  list (`config/artifacts.py:247-272`, `_PREFIX_SUFFIXES_TO_STRIP`); Rust accepts
  `artifacts.prefix` (`rust/cli/src/yaml.rs:780-782`) but I did not trace its
  normalization, so whether `prefix: foo_raw.jsonl` yields the same `foo` stem on
  both sides is unchecked.
- **`dataset.format` default for `type: file`.** Python defaults to `single_turn`
  (`config/dataset/config.py:328-331`); Rust carries `format: Option<String>`
  (`rust/cli/src/yaml.rs:1193`) and I did not follow the unauthored branch through
  `resolve.rs` to confirm the same default.
- **Runtime effect of the two accepted-but-invalid phase values in finding 8.**
  `gracePeriod` without `duration` and `prefillConcurrency > concurrency` are
  provably accepted by the Rust loader; that each is inert rather than
  behavior-changing is inferred from where the value is consumed, not measured.
