<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Artifacts and export format parity audit

Domain: everything AIPerf writes to disk or pushes to an external system —
artifact filenames, directory layout, and the exact schema of every output file.

**Python baseline: `/mnt/4tb/aiperf-parity-py-main/src/aiperf/`, git rev
`bc359bf8fd` (`origin/main`).** Every `src/aiperf/...` path and line number below
is relative to that checkout. Rust citations are relative to this repository.
An earlier revision of this report cited a feature branch; all Python evidence
has been re-derived and re-cited against the baseline, and the per-finding
outcome is recorded in "Baseline correction: finding classification" below.

Backlog cross-reference: `docs/dev/python-rust-parity-gaps.md` (dated
2026-07-17). Relevant pre-existing entries: P0.1 (accuracy unreachable), P1.31
(raw/per-record DTO fidelity), P1.33 (summary schema at missing/non-finite
values), P1.34 (artifact policy not fully projected), P1.35 (OTLP narrower),
P1.36 (MLflow narrower), P1.37 (W&B offline-only), P2.8 (Parquet contract).

## Summary

The highest-impact difference is not a schema field at all: Rust dropped
Python's auto-generated per-run artifact subdirectory, so every default
invocation writes into a flat `artifacts/` and back-to-back runs silently
overwrite each other's results (finding 1). Second, `--profile-export-prefix`
now produces *different filenames* than Python for the primary summary and
timeslice artifacts (`foo_aiperf.json` vs `foo.json`) and is ignored outright
for the console, server-metrics, and network-latency files (finding 2). Third,
`artifacts.summary: ["json"]` — the *only* value baseline Python accepts, and its
own default — means "summary JSON plus an unconditional summary CSV" upstream but
"JSON only" in Rust, so the CSV silently disappears (finding 3). Inside
`profile_export_aiperf.json`, `run_info` is an empty object and top-level
`start_time`/`end_time` are gone while `schema_version` still reads `"1.4"`, so
there is no version signal for the change (finding 4); distribution `sum`
disappears from every record metric in both JSON and the CSV `sum` column
(finding 5); and scalar/derived metrics gain `min`/`max` keys Python never
emitted (finding 6). The summary CSV lost Python's entire GPU-telemetry section
(finding 7). Per-record JSONL flipped several omit-when-absent fields to
explicit `null` and serializes integer token counts as floats (finding 8). No
credential leak was found: Rust redacts `input_config` and raw-record request
headers in the same places Python does; re-confirmed against baseline
`common/redact.py` (see "Checked and consistent").

Evidence includes two real artifact pairs: a Python-emitted run at
`artifacts/meta-llama_Llama-3.1-8B-Instruct-openai-chat-concurrency64/` and a
Rust-emitted run at `rust/artifacts/`.

**Provenance of the Python sample.** That directory name is itself evidence of
which Python produced it. Baseline `_compute_artifact_name` joins
`<model>-<service_kind>-<endpoint_type>-<stimulus>`
(`src/aiperf/config/resolution/resolvers.py:161-175`), yielding the `openai-chat`
segment seen on disk. The feature branch dropped `service_kind` and would have
produced `...-chat-concurrency64`. The sample therefore matches the *baseline*
naming rule, so it is not branch-tainted and no confidence downgrade is applied
to findings resting on it. Its metric *population* is still run-shape dependent
(see "Unverified"), which is why per-key set comparisons are excluded.

## Baseline correction: finding classification

Seven baseline files differ from the previously-used branch. Only four of my
findings touched any of them, and none was withdrawn.

| # | Severity | Classification | What changed against baseline |
| --- | --- | --- | --- |
| 1 | P0 | **STILL VALID** (strengthened) | Baseline creates the same per-run subdirectory; the name template is *richer* than reported (includes `service_kind`) |
| 2 | P1 | STILL VALID | Re-cited only |
| 3 | P1 | **CHANGED** | Baseline vocabulary is `Literal["json"]`, not `["json","genai_perf"]`; restated as the changed *meaning* of `json` |
| 4 | P1 | STILL VALID | Re-cited only |
| 5 | P1 | STILL VALID | Re-cited only; does not depend on `metrics_base_exporter.py` |
| 6 | P1 | STILL VALID | Re-cited only; does not depend on `metrics_base_exporter.py` |
| 7 | P1 | STILL VALID | Re-cited only |
| 8 | P1 | STILL VALID | Re-cited only; does not depend on `inference_result_parser.py` |
| 9 | P1 | STILL VALID | Re-cited only |
| 10 | P1 | STILL VALID | Re-cited only |
| 11 | P1 | STILL VALID | Re-cited only |
| 12 | P1 | **CHANGED** (strengthened) | Evidence re-derived: my original citation of `aiperf.*` *attributes* was wrong, but baseline emits 14 named `aiperf.timing.*` series plus an `aiperf.<tag>` histogram per non-spec metric |
| 13 | P2 | **CHANGED** (narrowed) | `tokens_in_flight` is Rust-only upstream, so that half is out of scope; the finding now covers `decode_duration` only |
| 14 | P2 | STILL VALID | Re-cited only |

Survived: 14. Withdrawn: 0. Changed: 3 (findings 3, 12, 13).

Files whose branch-vs-baseline delta turned out **not** to underpin any finding:
`exporters/metrics_base_exporter.py` (its `_prepare_metrics` INTERNAL/EXPERIMENTAL
drop is unchanged in substance at baseline lines 30-64, and it selects *which*
metrics export, not their schema — findings 5, 6 and 7 rest on
`record_models.py:107-131`, `export_models.py` and `metrics_csv_exporter.py`,
all byte-identical); `exporters/console_metrics_exporter.py` (baseline still
renders `record.header`, line 217-218, which is the mechanism finding 13 relies
on); `records/inference_result_parser.py` (finding 8 rests on
`buffered_jsonl_writer_mixin.py` and `record_models.py`, not the parser);
`config/resolution/plan.py` (no finding cites it); `config/otel.py` (baseline
contains only config knobs and no metric names, so finding 12's evidence was
always the post-processor and strategy modules, which are byte-identical).

## File inventory diff

Default configuration (no `artifacts.prefix`), single non-sweep run.

| Filename / pattern | Python | Rust | Schema identical? |
| --- | --- | --- | --- |
| `<artifact_dir>/<model>-<service_kind>-<endpoint_type>-<stimulus>/` subdir | yes | **no** | n/a — layout (finding 1) |
| `profile_export_aiperf.json` | yes | yes | **no** (findings 4, 5, 6) |
| `profile_export_aiperf.csv` | yes (unconditional) | yes (gated) | **no** (findings 3, 5, 7, 13) |
| `profile_export_aiperf_timeslices.{json,csv}` | yes | yes | yes |
| `profile_export.jsonl` | yes | yes | **no** (finding 8) |
| `profile_export_raw.jsonl` | yes | yes | close; Rust adds `response_headers` |
| `profile_export_records.csv` | **not accepted by config** | yes | Rust-only |
| `profile_export.parquet` | **not accepted by config** | yes | Rust-only |
| `profile_export_console.txt` | yes (prefixable) | yes (**not** prefixable) | n/a — filename (finding 2) |
| `inputs.json` | yes | yes | yes |
| `outputs.json` | yes | yes | `schema_version` differs (finding 14) |
| `gpu_telemetry_export.jsonl` | yes | yes | not compared (telemetry domain) |
| `server_metrics_export.{json,csv,jsonl,parquet}` | yes (prefixable) | yes (**not** prefixable) | n/a — filename (finding 2) |
| `profile_export_network_latency.jsonl` | yes (prefixable) | yes (**not** prefixable) | n/a — filename (finding 2) |
| `accuracy_export.jsonl` (per record) | yes | **no counterpart** | n/a (finding 11) |
| `accuracy_results.csv` (aggregate) | **no counterpart** | yes | n/a (finding 11) |
| `mlflow_export.json` | yes | **no** | n/a (finding 9) |
| `output_fragments/output_fragments_*.jsonl` | yes (temp, deleted) | no | internal staging only |
| `checkpoints/` | yes | not compared | out of scope |

Sweep/trial nesting is equivalent on both sides — see "Checked and consistent".

## Findings

### 1. Python's auto-generated per-run artifact subdirectory is gone; consecutive runs overwrite each other

**Severity:** P0 · **Status:** NEW · **Baseline: STILL VALID (strengthened)**

**Python evidence** — baseline `src/aiperf/config/resolution/resolvers.py:97-107`.
This is the load-bearing citation for the whole finding, so it is quoted in full
from the baseline file:

```python
# Auto-generate descriptive subdirectory if the user didn't set a custom dir.
if "dir" not in cfg.artifacts.model_fields_set:
    subdir_name = self._compute_artifact_name(cfg)
    if subdir_name:
        artifact_dir = artifact_dir / subdir_name

run.artifact_dir = artifact_dir
run.cfg.artifacts.dir = artifact_dir
artifact_dir.mkdir(parents=True, exist_ok=True)
```

So the subdirectory is created only when the user did *not* author
`artifacts.dir` — i.e. exactly the default-invocation case.

`_compute_artifact_name` (`src/aiperf/config/resolution/resolvers.py:139-175`)
joins three `-`-separated parts: the model name (with `/` → `_`, and a `_multi`
suffix for multi-model runs, lines 151-159), `f"{service_kind}-{endpoint_type}"`
resolved from the endpoint plugin registry (lines 161-168), and the stimulus of
the first non-warmup phase (lines 170-173, via `_get_stimulus` /`_describe_phase`
at lines 178-201: `concurrency<N>`, `user_centric-users<N>-qps<R>`,
`fixed_schedule`, or a rate-phase rendering). The docstring's own example is
`llama-3-8b-openai-chat-concurrency10`.

Note the correction to my earlier report: the template includes the
**`service_kind`** segment, which the feature branch had removed. The template is
`<model>-<service_kind>-<endpoint_type>-<stimulus>`, matching the on-disk sample
`meta-llama_Llama-3.1-8B-Instruct-openai-chat-concurrency64` exactly.

**Rust evidence** — the artifact directory is the bare flag value or the literal
`artifacts`, with no derived segment, on both the flag path
(`rust/cli/src/load.rs:672`) and the YAML path (`rust/cli/src/yaml.rs:2738`):

```rust
artifact_dir: flags
    .artifact_dir
    .clone()
    .unwrap_or_else(|| PathBuf::from("artifacts")),
```

`rust/cli/src/sweep/artifact_dir.rs:40` returns the base unchanged for the
non-sweep single-trial case: `(false, false) => base.to_path_buf(),`.

**Observable user impact.** Two runs that differ only in load level:

```
aiperf profile -m llama --endpoint-type chat -u ... --concurrency 10
aiperf profile -m llama --endpoint-type chat -u ... --concurrency 20
```

Python: `artifacts/llama-openai-chat-concurrency10/profile_export_aiperf.json`
and `artifacts/llama-openai-chat-concurrency20/profile_export_aiperf.json`.
Rust: both write `artifacts/profile_export_aiperf.json`; the second run
clobbers the first's summary, CSV, records JSONL, and console capture with no
warning. Confirmed by the on-disk samples: the Python run sits under
`artifacts/meta-llama_Llama-3.1-8B-Instruct-openai-chat-concurrency64/` while
the Rust run's files sit directly in `rust/artifacts/`.

**Confidence:** High (baseline code on both sides plus sample layout; the sample
directory name is itself produced by the baseline template, not the branch's).

### 2. `--profile-export-prefix` produces different filenames and is ignored for several artifacts

**Severity:** P1 · **Status:** KNOWN(still-true) — P1.34, sharpened to exact
names · **Baseline: STILL VALID**

**Python evidence** — `src/aiperf/config/artifacts.py:274-286`; the `_aiperf`
infix is dropped whenever a prefix is supplied:

```python
@property
def profile_export_json_file(self) -> Path:
    base = self._base()
    name = f"{base}.json" if base else "profile_export_aiperf.json"
    return self.dir / name
```

Same pattern for `.csv` (lines 274-279) and `f"{base}_timeslices.json"` (lines
302-311). Prefixing also covers the console capture (lines 332-337,
`f"{base}_console.txt"`), server metrics (lines 346-351, 371-383, 385-390),
network latency (lines 360-369), and accuracy (lines 353-358).

**Rust evidence** — the summary stem is derived from the records path
(`rust/runtime/src/engine/protocol_v2.rs:488-496`) and the `_aiperf` infix is
always appended (`rust/runtime/src/export/genai_perf.rs:165-171`):

```rust
let stem = name.strip_suffix(".jsonl").unwrap_or(name);
if !stem.is_empty() {
    export_cfg.genai_perf.stem = stem.to_string();
    export_cfg.timeslice.stem = Some(format!("{stem}_aiperf"));
}
```

```rust
if cfg.genai_perf.json {
    let json = render_json(report, &cfg.genai_perf);
    std::fs::write(artifact_dir.join(format!("{stem}_aiperf.json")), json)?;
}
```

The console, accuracy, and server-metrics filenames are compile-time constants
with no prefix input: `rust/runtime/src/export/console_txt.rs:33`
(`const CONSOLE_TXT_FILENAME: &str = "profile_export_console.txt";`),
`rust/runtime/src/export/server_metrics/mod.rs:28-29`,
`rust/runtime/src/export/accuracy_csv.rs:22`, and
`rust/runtime/src/config/model/telemetry.rs:209` (network latency).
`rust/cli/src/load.rs:2745` asserts the intended behavior only for the records
path: `assert_eq!(arts.records_path.as_deref(), Some("myrun.jsonl"));`.

**Observable user impact.** With `--profile-export-prefix myrun`:

| Artifact | Python | Rust |
| --- | --- | --- |
| summary JSON | `myrun.json` | `myrun_aiperf.json` |
| summary CSV | `myrun.csv` | `myrun_aiperf.csv` |
| timeslices | `myrun_timeslices.json` | `myrun_aiperf_timeslices.json` |
| console | `myrun_console.txt` | `profile_export_console.txt` |
| server metrics | `myrun_server_metrics.json` | `server_metrics_export.json` |
| network latency | `myrun_network_latency.jsonl` | `profile_export_network_latency.jsonl` |

A secondary case: because the stem is derived from `records_path`, any config
that leaves no records JSONL — `records: false` (accepted by both sides),
`--sketch-metrics`, or Rust-only `records: [csv]` — leaves `records_path` `None`
(`rust/runtime/src/config/resolve.rs:1630`), so the prefix is silently dropped
from the summary artifacts and they revert to `profile_export_aiperf.*` while
`myrun_records.csv` still carries it.

Finally, the suffix-strip lists differ. Python strips `_console.txt` and
`_network_latency.jsonl` (`src/aiperf/config/artifacts.py:247-262`, entries at
lines 252 and 256); Rust does
not (`rust/runtime/src/config/resolve.rs:155-171`), so
`--profile-export-prefix foo_console.txt` yields Python `foo.json` and Rust
`foo_console.txt_aiperf.json`.

**Confidence:** High.

### 3. `artifacts.summary: ["json"]` means "JSON + CSV" in Python but "JSON only" in Rust, silently suppressing `profile_export_aiperf.csv`

**Severity:** P1 · **Status:** NEW · **Baseline: CHANGED (restated)**

**Convergence note.** The config-schema auditor independently reported that the
two sides accept *disjoint* vocabularies, `json|genai_perf` in Python vs
`json|csv` in Rust. Re-derived from baseline, that is not the shape of the
problem: `genai_perf` does not exist upstream. Baseline
`src/aiperf/config/artifacts.py:37` is

```python
SummaryExportFormat = Literal["json"]
```

so the only list value baseline Python accepts is `"json"`, and Rust's
`json|csv` is a strict **superset**. There is no disjointness and no
Python-accepted-but-Rust-rejected list value. **The trap is entirely the changed
meaning of the shared `json` value**, plus a separate loud break on
`summary: false`. Both are re-derived below.

**Python evidence** — only the JSON exporter consults `artifacts.summary`
(`src/aiperf/exporters/metrics_json_exporter.py:28-31`):

```python
summary = exporter_config.cfg.artifacts.summary
if summary is False or "json" not in summary:
    raise DataExporterDisabled(
        "MetricsJsonExporter disabled: 'json' not in artifacts.summary"
    )
```

`MetricsCsvExporter` has no such gate anywhere in the file — it takes its path
unconditionally at `src/aiperf/exporters/metrics_csv_exporter.py:25-27` and
`_generate_content` (lines 42-70) never consults `artifacts.summary`. So the
upstream CSV-emission rule is: **the summary CSV is written whenever there are
results, regardless of `artifacts.summary`.** The field's own description states
this (`src/aiperf/config/artifacts.py:111-120`):

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

**Rust evidence** — `rust/runtime/src/config/model/export.rs:461-468`:

```rust
let unauthored = summary_formats.is_empty();
let json = unauthored || summary_formats.iter().any(|f| f == "json");
let csv = unauthored || summary_formats.iter().any(|f| f == "csv");
```

and `rust/runtime/src/export/genai_perf.rs:169`
(`if cfg.genai_perf.csv { ... }`).

Rust's `unauthored` branch is what hides this. Omit `summary` entirely and both
flags go true, so the default artifact set matches Python. Author the one value
Python allows — `summary: ["json"]`, which is also Python's own default and
therefore feels like a no-op — and `csv` becomes false.

**Observable user impact.** Three cases, only the middle one silent:

| Config | Python | Rust |
| --- | --- | --- |
| `summary` omitted | JSON + CSV | JSON + CSV |
| `summary: ["json"]` | JSON + CSV | **JSON only — CSV silently gone** |
| `summary: false` | CSV only (JSON suppressed) | **hard serde error** |

The middle row is the finding: writing out the default value changes the output
file set. Any downstream job that parses `profile_export_aiperf.csv` breaks with
a missing-file error rather than a config error.

The third row is loud rather than silent (noted per instructions, not counted as
a silent change): Python's `summary: false` is a valid `Literal[False]`
(`src/aiperf/config/artifacts.py:112`) that suppresses the JSON while still
writing the CSV; it fails Rust's `Option<Vec<String>>` deserialization
(`rust/cli/src/yaml.rs:767`) with a serde type error. Rust's extra `csv` value is
a Rust-only expansion and out of scope.

The same asymmetry applies to `records`: baseline
`RecordsExportFormat = Literal["jsonl"]` (`src/aiperf/config/artifacts.py:38`),
so `records: [csv]` and `records: [parquet]` are Rust-only additions rather than
shared vocabulary. This resolves an item previously listed as unverified.

**Confidence:** High.

### 4. Summary JSON drops top-level `start_time`/`end_time` and emits an empty `run_info`, under an unchanged `schema_version`

**Severity:** P1 · **Status:** KNOWN(still-true) — P1.33 ("top-level times … not
consistently projected", target "complete run metadata") · **Baseline: STILL
VALID**

**Python evidence** — `src/aiperf/common/models/export_models.py:336-341`
declares the fields on `JsonExportData` (class at line 281), `RunInfo` is
declared at line 179, and the exporter populates all of them
(`src/aiperf/exporters/metrics_json_exporter.py:60-83`):

```python
start_time = (
    datetime.fromtimestamp(self._results.start_ns / NANOS_PER_SECOND)
    if self._results.start_ns
    else None
)
...
export_data = JsonExportData(
    ...
    run_info=RunInfo.from_run(self._run),
    ...
    start_time=start_time,
    end_time=end_time,
)
```

`RunInfo.from_run` populates every coordinate:

```python
run_info: RunInfo | None = None
was_cancelled: bool | None = None
error_summary: list[ErrorDetailsCount] | None = None
start_time: datetime | None = None
end_time: datetime | None = None
```

```python
return cls(
    benchmark_id=run.benchmark_id,
    sweep_id=run.sweep_id,
    random_seed=run.random_seed,
    trial=run.trial,
    run_label=run.label or None,
    ...
    cli_command=run.cli_command,
)
```

`cli_command` is the redacted `sys.argv` reconstruction
(`src/aiperf/common/redact.py:295` `build_cli_command`, with the arg redaction at
`:231` and `:272`).

**Rust evidence** — `run_info` is spliced verbatim from the envelope
(`rust/runtime/src/export/genai_perf.rs:568-570`) and the envelope is built with
an empty object (`rust/runtime/src/config/resolve.rs:1698-1705`):

```rust
if let Some(run_info) = &cfg.envelope.run_info {
    root.insert("run_info".to_owned(), run_info.clone());
}
```

```rust
let mut export = crate::config::model::export::Export::build(
    &endpoint_type,
    &inputs.summary_formats,
    &benchmark_id,
    input_config.clone(),
    serde_json::json!({}),
    &inputs.model_names,
);
```

`render_json` (`rust/runtime/src/export/genai_perf.rs:528-646`) never inserts
`start_time` or `end_time`; those strings appear only inside
`telemetry_data.summary` (lines 476-484).

**Observable user impact.** Python summary
(`artifacts/meta-llama_.../profile_export_aiperf.json`):

```json
"run_info": {"benchmark_id": "592ca7fa63cd", "trial": 0,
             "cli_command": "aiperf profile --model 'meta-llama/Llama-3.1-8B-Instruct' ... --concurrency 64"},
"start_time": "2026-07-27T21:24:26.347694",
"end_time": "2026-07-27T21:24:28.758479"
```

Rust summary (`rust/artifacts/profile_export_aiperf.json`):

```json
"run_info": {}
```

with no `start_time` / `end_time` keys at all. Both files report
`"schema_version": "1.4"`, so a consumer has no signal to branch on. Run
reproducibility provenance (the invocation and the resolved seed) is lost.

**Confidence:** High (code plus both samples).

### 5. Distribution `sum` is never emitted for record metrics — JSON key absent, CSV `sum` cell empty

**Severity:** P1 · **Status:** KNOWN(still-true) — P1.33 ("Distribution sums …
not consistently projected") · **Baseline: STILL VALID**

**Python evidence** — `JsonMetricResult.sum` is declared, with a docstring saying
it is present precisely for record-type metrics
(`src/aiperf/common/models/export_models.py:60-66`), and
`MetricResult.to_json_result` (`src/aiperf/common/models/record_models.py:107`)
copies every `STAT_KEYS` entry, `sum` included (lines 129-131):

```python
for stat in STAT_KEYS:
    setattr(result, stat, getattr(self, stat, None))
```

`STAT_KEYS` includes `sum` (`src/aiperf/common/constants.py:20`), and the CSV
writer emits one column per `STAT_KEYS` entry
(`src/aiperf/exporters/metrics_csv_exporter.py:91-104`, header at line 96:
`header = ["Metric"] + list(STAT_KEYS)`). This path does not go through
`metrics_base_exporter.py`, whose only role is dropping INTERNAL/EXPERIMENTAL
metrics wholesale.

**Rust evidence** — the `Distribution` arm of `project_stats` leaves `sum` at
its `None` initializer (`rust/runtime/src/export/genai_perf.rs:252-261`):

```rust
ReportStats::Distribution(stats) => {
    projected.avg = finite(stats.avg);
    projected.min = finite(stats.min);
    projected.max = finite(stats.max);
    projected.std = finite(stats.std);
    projected.count = stats.count.map(|count| count as u64);
    for (index, label) in PERCENTILE_LABELS.iter().enumerate() {
        projected.percentiles[index] = finite(stats.percentiles.get(*label).copied());
    }
}
```

`sum` is set only for `Counter` (line 273) and `Histogram` (line 277).
`format_number(None)` renders an empty CSV cell
(`rust/runtime/src/export/genai_perf.rs:709-714`).

**Observable user impact.** JSON, same metric:

```
Python: "request_latency": {"unit":"ms","avg":0.987,...,"count":40000,"sum":39496.465232999995}
Rust:   "request_latency": {"unit":"ms","avg":218.0,...,"count":1}
```

CSV, `Metric,avg,min,max,sum,p1,…` header on both sides:

```
Python: Request Latency (ms),0.99,0.22,14.27,39496.47,0.30,…
Rust:   Request Latency (ms),218.00,218.00,218.00,,218.00,…
```

Every record metric row now has an empty `sum` cell. A downstream reader that
coerces the `sum` column to a number gets an empty-string parse failure rather
than a null.

**Confidence:** High (code plus both samples).

### 6. Scalar and derived metrics gain `min`/`max` (and counters `sum`) that Python never emitted

**Severity:** P1 · **Status:** NEW · **Baseline: STILL VALID**

**Python evidence** — a derived/aggregate metric carries only its computed
value; `to_json_result` (`src/aiperf/common/models/record_models.py:107-131`)
copies whatever stats the `MetricResult` has, and for scalars that is `avg`
alone. Unset stats stay `None` and the exporter drops them
(`src/aiperf/exporters/metrics_json_exporter.py:200-201`,
`model_dump(mode="json", exclude_unset=True, exclude_none=True)`). The emitted
result is visible in the Python sample:

```json
"request_count": {"unit": "requests", "avg": 40000.0}
"benchmark_duration": {"unit": "sec", "avg": 2.409118755}
"request_throughput": {"unit": "requests/sec", "avg": 16603.58166943082}
```

**Rust evidence** — `project_stats` fans the single value out to three (or four)
stats (`rust/runtime/src/export/genai_perf.rs:262-274`):

```rust
ReportStats::Scalar(stats) => {
    let value = finite(Some(stats.value))?;
    projected.avg = Some(value);
    projected.min = Some(value);
    projected.max = Some(value);
}
ReportStats::Counter(stats) => {
    let total = finite(Some(stats.total))?;
    projected.avg = Some(total);
    projected.min = Some(total);
    projected.max = Some(total);
    projected.sum = Some(total);
}
```

**Observable user impact.** Same three keys in `rust/artifacts/profile_export_aiperf.json`:

```json
"request_count": {"unit": "requests", "avg": 1.0, "min": 1.0, "max": 1.0, "sum": 1.0}
"benchmark_duration": {"unit": "sec", "avg": 0.218, "min": 0.218, "max": 0.218}
"request_throughput": {"unit": "requests/sec", "avg": 4.587155963302752, "min": 4.587155963302752, "max": 4.587155963302752}
```

A consumer that uses the presence of `min`/`max` to decide "this metric is a
distribution, plot a box" now misclassifies every scalar; one that reads
`request_count.max` gets a value that is not a maximum of anything. Note the
`count` omission rule *is* replicated (`scalar_tags`,
`rust/runtime/src/export/genai_perf.rs:228-230`) — Python's version is the
`is_scalar` branch in `to_json_result`
(`src/aiperf/common/models/record_models.py:119-125`,
`count=None if is_scalar else self.count`) — so the divergence is specifically
`min`/`max`/`sum`.

**Confidence:** High (code plus both samples).

### 7. Summary CSV lost Python's GPU-telemetry section

**Severity:** P1 · **Status:** NEW · **Baseline: STILL VALID**

**Python evidence** — `src/aiperf/exporters/metrics_csv_exporter.py:67-68` and
`_write_telemetry_section`, whose optional-column helper is at lines 140-171:

```python
# Add telemetry data section if available
if self._telemetry_results:
    self._write_telemetry_section(writer)
```

```python
header_row = ["Endpoint", "GPU_Index", "GPU_Name", "GPU_UUID", "Platform"]
optional_headers, optional_fields = self._get_optional_headers_and_fields(
    "Hostname", "Namespace", "Pod Name"
)
header_row.extend(["Metric", *STAT_KEYS])
header_row.extend(optional_headers)
writer.writerow(header_row)
```

**Rust evidence** — `render_csv` writes exactly two sections and returns
(`rust/runtime/src/export/genai_perf.rs:735-747`):

```rust
let mut request: Vec<&(String, Projected)> = collected
    .iter()
    .filter(|(_, projected)| projected.has_percentiles())
    .collect();
let mut system: Vec<&(String, Projected)> = collected
    .iter()
    .filter(|(_, projected)| !projected.has_percentiles())
    .collect();
```

`rg -n "GPU_Index|GPU_UUID|GPU_Name" rust/runtime/src/export/*.rs` returns
nothing. Rust does project GPU telemetry into the summary *JSON*
(`render_telemetry_data`, `rust/runtime/src/export/genai_perf.rs:354`), so this
is a CSV-only loss.

**Observable user impact.** With `--gpu-telemetry <dcgm-url>`, Python's
`profile_export_aiperf.csv` ends with a third table (visible in the Python
sample at line 70):

```
Endpoint,GPU_Index,GPU_Name,GPU_UUID,Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
```

Rust's CSV ends after the `Metric,Value` scalar section. A spreadsheet or pandas
pipeline that reads per-GPU power/utilization out of the CSV gets nothing and
must be rewritten against the JSON.

**Confidence:** High.

### 8. Per-record JSONL: omitted-when-absent fields became explicit `null`, and integer metric values became floats

**Severity:** P1 · **Status:** NEW (adjacent to KNOWN P1.31 for the missing
DAG/source fields) · **Baseline: STILL VALID**

**Python evidence** — the JSONL writer
(`src/aiperf/post_processors/record_export_jsonl_writer.py:39-44`, gated on
`ExportLevel.RECORDS`/`RAW`) dumps through
`src/aiperf/common/mixins/buffered_jsonl_writer_mixin.py:124-130`:

```python
# Use exclude_none=True to omit None fields (smaller output)
# scrub_non_finite enforces "null on disk = absent" across the
scrub_non_finite(
    ...
    exclude_none=True,
```

Absent optionals therefore never appear as keys, and NaN/Inf is scrubbed to
`None` first (`src/aiperf/common/finite.py`) so it is omitted rather than written.
`MetricValue.value` is `MetricValueTypeT` (`int | float`), so integer metrics
stay integers. This path does not involve
`records/inference_result_parser.py`.

**Rust evidence** — `rust/runtime/src/engine/records.rs:102-150`:

```rust
struct RecordRow {
    metadata: RecordMetadata,
    metrics: BTreeMap<String, RecordMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spec_decode_acceptance: Option<ObservedSpecDecodeAcceptance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_data: Option<Value>,
    error: Option<RecordError>,
}

struct RecordMetadata {
    session_num: u64,
    x_request_id: String,
    x_correlation_id: String,
    conversation_id: Option<String>,
    turn_index: u32,
    credit_issued_ns: Option<i64>,
    ...
}

struct RecordMetric {
    value: f64,
    unit: String,
}
```

`error`, `conversation_id`, and `credit_issued_ns` carry no
`skip_serializing_if`, so they serialize as `null`. `value: f64` forces every
metric through float formatting.

**Observable user impact.** A successful non-conversational record:

```
Python: {"metadata":{"session_num":0,"x_request_id":"…","turn_index":0,…},
         "metrics":{"output_sequence_length":{"value":105,"unit":"tokens"},…}}
Rust:   {"metadata":{"session_num":0,"x_request_id":"…","x_correlation_id":"…",
                     "conversation_id":null,"turn_index":0,"credit_issued_ns":null,…},
         "metrics":{"output_sequence_length":{"value":105.0,"unit":"tokens"},…},
         "error":null}
```

A strict consumer that treats key presence as "this field applies" now sees
`conversation_id` and `error` on every row; one that expects
`isinstance(value, int)` for token counts, or does an exact string comparison on
the serialized value, sees `105.0`. The same `f64` widening applies to
`outputs.json` metrics (`rust/runtime/src/engine/records.rs:357`,
`metrics: BTreeMap<&'static str, f64>`).

Separately, and already tracked under P1.31: Rust's `RecordMetadata` has no
counterpart for Python's `source_trace_id`, `source_outer_idx`,
`source_inner_idx`, `source_kind`, `agent_depth`, `parent_correlation_id`, or
`root_correlation_id`.

**Confidence:** High for the `null` and float claims (code on both sides).
Medium on the exact Python key set for a given run, since Python's set is
run-shape dependent.

### 9. `mlflow_export.json` is never written, and `aiperf plot --mlflow-upload` reads it

**Severity:** P1 · **Status:** NEW (P1.36 mentions a "metadata sidecar" but not
this consumer) · **Baseline: STILL VALID**

**Python evidence** — `src/aiperf/exporters/mlflow_data_exporter.py:467-505`
(`_write_export_metadata`) writes it atomically into the artifact directory
before upload; the filename constant is
`src/aiperf/config/mlflow.py:46` (`EXPORT_METADATA_FILE = Path("mlflow_export.json")`):

```python
metadata: MLflowExportMetadata = {
    "tracking_uri": redact_url(self._tracking_uri),
    "experiment": self._experiment_name,
    "run_id": run_id,
    "run_name": run_name,
    "benchmark_id": self._benchmark_id,
    ...
}
payload = orjson.dumps(scrub_non_finite(metadata), option=orjson.OPT_INDENT_2)
tmp_file = self._metadata_file.with_suffix(".json.tmp")
tmp_file.write_bytes(payload)
tmp_file.replace(self._metadata_file)
```

The schema lives in `src/aiperf/exporters/mlflow_metadata.py:60`
(`class MLflowExportMetadata(TypedDict, total=False)`), and the file is deliberately
uploaded as a run artifact alongside the others
(`src/aiperf/exporters/mlflow_data_exporter.py:248-278`).

**Rust evidence** — the only mention of the file in the Rust tree is an upload
*skip* (`rust/runtime/src/export/mlflow.rs:307-310`); nothing creates it:

```rust
// Do not upload stale MLflow metadata from an earlier attempt.
if rel_posix == "mlflow_export.json" {
    continue;
}
```

**Observable user impact.** `aiperf plot` is one of the two commands still
delegated to Python, and its documented flow resolves the MLflow run from this
file (`src/aiperf/cli_commands/plot.py:47`):

```
# Generate plots and upload them to the MLflow run from mlflow_export.json
aiperf plot --paths artifacts/my-run --mlflow-upload
```

After a Rust-driven `aiperf profile --mlflow-tracking-uri …`, that command has
no run to attach to and the user must supply `--mlflow-run-id` by hand. The file
also disappears from the artifact directory listing and from the MLflow run's
own artifact bundle.

**Confidence:** High.

### 10. MLflow receives no params at all

**Severity:** P1 · **Status:** KNOWN(still-true) — P1.36 ("projected
parameters") · **Baseline: STILL VALID**

**Python evidence** — `src/aiperf/exporters/mlflow_data_exporter.py:357-384`
(`_build_param_payload`) builds up to ten params:

```python
params: dict[str, str] = {
    "endpoint.type": str(self._cfg.endpoint.type),
    "endpoint.models": ",".join(self._cfg.get_model_names()),
    "endpoint.urls": ",".join(redact_url(url) for url in self._cfg.endpoint.urls),
    "output.artifact_directory": str(self._cfg.artifacts.artifact_directory),
}
...
params["timing.mode"] = str(phase.type)
params["loadgen.concurrency"] = str(phase.concurrency)
params["loadgen.request_rate"] = str(rate)
params["loadgen.request_count"] = str(phase.requests)
params["loadgen.benchmark_duration"] = str(phase.duration)
params["aiperf.cli_command"] = redact_cli_command(cli_command)
```

**Rust evidence** — `rust/runtime/src/config/model/export.rs:417`:

```rust
params: std::collections::BTreeMap::new(),
```

The field is plumbed through to the uploader
(`rust/runtime/src/export/mlflow.rs:74`, `126`) but nothing ever fills it.

**Observable user impact.** The MLflow run's Parameters tab is empty. Metrics
and tags are equivalent (`aiperf.version`, `benchmark_id`,
`aiperf.was_cancelled`, `tag`/`tag.stat` keys,
`aiperf.completed_requests`, `aiperf.total_expected_requests` — verified at
`rust/runtime/src/export/mlflow.rs:173-206`), so the run looks populated while
every reproducibility parameter, including the redacted CLI command, is absent.
Run-name derivation matches (`aiperf-<benchmark_id[:8]>`;
`src/aiperf/exporters/mlflow_data_exporter.py:324-327` vs
`rust/runtime/src/export/mlflow.rs:167-169`).

**Confidence:** High.

### 11. Accuracy mode writes a differently-named file at a different granularity

**Severity:** P1 · **Status:** KNOWN(still-true) — P0.1 covers reachability; the
artifact rename/regrouping is the observable half · **Baseline: STILL VALID**

**Python evidence** — `src/aiperf/accuracy/jsonl_writer.py:46` writes one JSON
object per graded record to `accuracy_export.jsonl` (class docstring at lines
25-29: "one JSON line per graded response … the grade (pass/unparsed/confidence),
the expected/actual answers, and the grader's reasoning"; profiling-phase-only
filter at lines 63-64), whose name is defined at
`src/aiperf/config/artifacts.py:353-358`:

```python
@property
def accuracy_export_jsonl_file(self) -> Path:
    """Path for the per-record accuracy JSONL export file."""
    base = self._base()
    name = f"{base}_accuracy.jsonl" if base else "accuracy_export.jsonl"
    return self.dir / name
```

**Rust evidence** — the only accuracy sink is an aggregate CSV
(`rust/runtime/src/export/accuracy_csv.rs:22,66-69`), and the exporter registry
(`rust/runtime/src/export/mod.rs:283-300`) contains no accuracy-JSONL entry:

```rust
const ACCURACY_CSV_FILE: &str = "accuracy_results.csv";
```

```rust
writer.write_record(["task", "correct", "total", "unparsed", "accuracy"])?;
write_row(&mut writer, "OVERALL", &summary.overall)?;
```

**Observable user impact.** An accuracy run produces
`accuracy_export.jsonl` (per-record grading detail) under Python and
`accuracy_results.csv` (one `OVERALL` row plus per-task rollups) under Rust.
Different filename, different extension, different format, and no per-record
grading rows at all — so any per-question error analysis is impossible. Rust
also writes no file when the overall population is empty
(`rust/runtime/src/export/accuracy_csv.rs:55-58`).

**Confidence:** High for the artifact difference. The per-record *field* schema
cannot be compared because Rust has no per-record accuracy artifact.

### 12. OTLP push carries only the four GenAI histograms; every `aiperf.*` series is gone

**Severity:** P1 · **Status:** KNOWN(still-true) — P1.35 · **Baseline: CHANGED
(evidence corrected, finding strengthened)**

**Correction.** My earlier evidence for this finding listed `aiperf.benchmark_phase`,
`aiperf.endpoint.type` and similar — those are *attribute* keys, not metric
names, and citing them was wrong. `config/otel.py` (the file that gained 41 lines
on the branch) is also not the evidence: baseline `config/otel.py` is 63 lines of
config knobs (`metrics_url`, `stream_metrics_enabled`, `stream_timing_enabled`,
`custom_resource_attributes`, `gen_ai_provider`) and contains no metric names at
all. The real evidence is the post-processor and its strategies, all
byte-identical to baseline. Re-derived, upstream's metric set is concretely
larger than I reported.

**Python evidence** — three sources of instrument names upstream:

1. The four GenAI spec histograms, same as Rust
   (`src/aiperf/post_processors/strategies/genai_semconv.py:184,189,194,199,204`):
   `gen_ai.client.operation.duration`,
   `gen_ai.client.operation.time_to_first_chunk`,
   `gen_ai.client.operation.time_per_output_chunk`, and
   `gen_ai.client.token.usage` (input and output token metrics merge into the
   last via the `gen_ai.token.type` attribute, lines 211-220).

2. **An `aiperf.<tag>` histogram for every metric with no spec translation.**
   `src/aiperf/post_processors/strategies/metric_results.py:60-61` falls through
   to `get_or_create_histogram(metric_name)` with no `unit`, and
   `src/aiperf/post_processors/otel_metrics_results_processor.py:298-300` prefixes
   exactly that case:

   ```python
   # When unit is provided, the caller already passed a fully-qualified
   # metric name (e.g. from GenAI semconv); don't prepend "aiperf.".
   instrument_name = metric_name if unit else f"aiperf.{metric_name}"
   ```

3. **Fourteen named timing series**, from
   `src/aiperf/post_processors/strategies/timing_results.py:18-35` — eight
   counters and six up-down counters:

   ```python
   _COUNTER_FIELDS = {
       "aiperf.timing.requests.sent", "aiperf.timing.requests.completed",
       "aiperf.timing.requests.cancelled", "aiperf.timing.requests.errors",
       "aiperf.timing.sessions.sent", "aiperf.timing.sessions.completed",
       "aiperf.timing.sessions.cancelled", "aiperf.timing.sessions.turns_total",
   }
   _GAUGE_FIELDS = {
       "aiperf.timing.requests.in_flight", "aiperf.timing.sessions.in_flight",
       "aiperf.timing.phase.timeout_triggered",
       "aiperf.timing.phase.grace_timeout_triggered",
       "aiperf.timing.phase.was_cancelled", "aiperf.timing.phase.elapsed_sec",
   }
   ```

   (keys elided to names here; the dict maps each name to its source field.)
   These are emitted as deltas at lines 51-66 and 69-91.

**Rust evidence** — `rust/runtime/src/export/otel.rs` declares exactly three
duration specs plus one token-usage metric:

```rust
DurationSpec { report_key: "request_latency",      spec_name: "gen_ai.client.operation.duration", … },
DurationSpec { report_key: "time_to_first_token",  spec_name: "gen_ai.client.operation.time_to_first_chunk", … },
DurationSpec { report_key: "inter_token_latency",  spec_name: "gen_ai.client.operation.time_per_output_chunk", … },
```

plus `gen_ai.client.token.usage`. No `aiperf.`-prefixed metric name and no
counter or up-down-counter instrument appears anywhere in the file.

**Observable user impact.** A Grafana/Prometheus dashboard built on the Python
exporter shows no data for any of the 14 named `aiperf.timing.*` series, nor for
the `aiperf.<tag>` histogram of any metric without a GenAI spec mapping — which
is most of the metric catalog, including every throughput, sequence-length, and
error-count metric. Only the four spec histograms arrive. Instrument *kind* is
also lost: the counters and gauges have no Rust counterpart at all, so
`rate()`-style queries have nothing to read. Python additionally exports
periodically through the OTel SDK while Rust performs one post-run push, so the
time-series shape of even the surviving four differs.

**Confidence:** High. Metric names read directly from baseline source; the
`aiperf.` prefix rule is the single expression quoted above.

### 13. `decode_duration` leaks its raw snake_case tag into the CSV `Metric` column and console

**Severity:** P2 · **Status:** NEW · **Baseline: CHANGED (narrowed to one metric)**

**Correction.** I originally paired `decode_duration` with `tokens_in_flight`.
Baseline has no registered metric with the tag `tokens_in_flight` — upstream it
exists only as an analysis-time sweepline curve
(`src/aiperf/metrics/accumulator_sweeps.py:64-90`), never as an exported metric
with a header. I checked all 18 raw snake_case labels in the Rust sample CSV
(`active_*`/`effective_*` throughput and concurrency family, `tokens_in_flight`,
`decode_duration`) against baseline tag definitions: **only `decode_duration`
corresponds to an upstream exported metric.** The rest are Rust-only metrics and
therefore out of scope. Severity stays P2.

**Python evidence** — `src/aiperf/metrics/types/decode_duration_metric.py:15-17`:

```python
tag = "decode_duration"
header = "Decode Duration"
short_header = "Decode Duration"
```

Its flags are `STREAMING_TOKENS_ONLY | PERCENTILE_INCLUDES_FAILED_REQUESTS`
(lines 21-24) — notably *not* `INTERNAL` or `EXPERIMENTAL`, so
`metrics_base_exporter._prepare_metrics` does not drop it and it reaches the CSV
for any streaming run. The CSV writes `metric.header`
(`src/aiperf/exporters/metrics_csv_exporter.py:117-122`) and the console writes
`record.header` (`src/aiperf/exporters/console_metrics_exporter.py:217-218`), so
both surfaces are affected.

**Rust evidence** — the header lookup falls back to the raw tag
(`rust/runtime/src/export/genai_perf.rs:220-224`):

```rust
let header = cfg
    .header_map
    .get(name)
    .cloned()
    .unwrap_or_else(|| name.to_owned());
```

and `rust/runtime/resources/metric_metadata.json` has 114 `header_map` entries
with no key for `decode_duration`. Rust does carry the *short* header for the
same tag (`rust/runtime/src/metrics_core/catalog.rs:529`,
`DecodeDuration => Some("Decode Duration")`), which is why the gap is specific to
the full-`header` lookup used by the CSV.

**Observable user impact.** In `rust/artifacts/profile_export_aiperf.csv`:

```
decode_duration (ms),208.00,208.00,208.00,,208.00,…
```

Python emits `Decode Duration (ms)` for the same metric. Because rows are keyed
by the `Metric` column, a lookup by display name misses this row.

**Confidence:** High. (I checked and rejected three similar suspicions —
`OSL Mismatch Diff`, `Usage Completion Diff`, `Usage Prompt Diff` looked renamed
against the older on-disk Python sample but match the current Python headers at
`src/aiperf/metrics/types/osl_mismatch_metrics.py:90` and
`src/aiperf/metrics/types/usage_diff_metrics.py:51,113`.)

### 14. `outputs.json` declares a different `schema_version`

**Severity:** P2 · **Status:** NEW · **Baseline: STILL VALID**

**Python evidence** — `src/aiperf/exporters/outputs_json_exporter.py:79-82`:

```python
output = {
    "schema_version": "1.0",
    "data": records,
}
```

**Rust evidence** — `rust/runtime/src/engine/records.rs:876,882`:

```rust
pub(crate) const OUTPUTS_SCHEMA_VERSION: &str = "1.1";
pub(crate) const OUTPUTS_PREFIX: &str = "{\n  \"schema_version\": \"1.1\",\n  \"data\": [";
```

**Observable user impact.** A consumer pinned to `schema_version == "1.0"`
rejects Rust's `outputs.json`. The change is at least *announced* (the version
string differs), and the only added field is Rust-only `reasoning_text`
(`rust/runtime/src/engine/records.rs:359`), so severity is low. Everything else
matches: key order, the six-metric allowlist
(`rust/runtime/src/engine/records.rs:362-369` vs
`src/aiperf/post_processors/outputs_json_record_processor.py:60-67`), the
profiling-phase-only filter, and the `(session_num, turn_index)` sort.

**Confidence:** High.

## Checked and consistent

- **Default filenames without a prefix.** `profile_export_aiperf.{json,csv}`,
  `profile_export_aiperf_timeslices.{json,csv}`, `profile_export.jsonl`,
  `profile_export_raw.jsonl`, `profile_export_records.csv`,
  `profile_export.parquet`, `profile_export_console.txt`,
  `gpu_telemetry_export.jsonl`, `server_metrics_export.{json,csv,jsonl,parquet}`,
  `profile_export_network_latency.jsonl`, `inputs.json`, `outputs.json` all
  agree byte-for-byte between baseline `OutputDefaults`
  (`src/aiperf/config/artifacts.py:41-70`) and the
  Rust constants / `artifact_export_stem` default
  (`rust/runtime/src/config/resolve.rs:172-174`).
- **Sweep and trial directory nesting.** The five-row layout table in
  `rust/cli/src/sweep/artifact_dir.rs:5-14` matches
  `src/aiperf/orchestrator/orchestrator.py:51-55` exactly, including the
  `run_NNNN` vs `trial_NNNN` asymmetry (called out in that file's own note at
  lines 64-65) and the REPEATED/INDEPENDENT ordering.
  Variation directory names use the same `{last_path_segment}_{value}` form
  joined by `__` (`src/aiperf/config/sweep/config.py:682-687,730-751` vs
  `rust/cli/src/sweep/mod.rs:243,250`).
- **Summary CSV header and structure.** Header row
  `Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std` with CRLF line
  endings, minimal quoting, a blank separator record, then a `Metric,Value`
  scalar section — identical in both samples. Section membership uses the same
  "has any percentile" rule
  (`src/aiperf/exporters/metrics_csv_exporter.py:87-89` vs
  `rust/runtime/src/export/genai_perf.rs:194-196`), rows sort by tag on both
  sides, and the metric-name suffix rule (`" (unit)"` unless the unit is empty,
  `count`, or `requests`) matches
  (`src/aiperf/exporters/metrics_csv_exporter.py:117-122` vs
  `rust/runtime/src/export/genai_perf.rs:697-706`). CRLF survives on the Python
  side because the base exporter opens with `newline=""`
  (`src/aiperf/exporters/metrics_base_exporter.py:98-100`), letting the `csv`
  module emit its own terminators.
- **Summary JSON key order and non-finite policy.** Metric-object field order
  (`unit, avg, p1…p99, min, max, std, count, sum`) matches
  `JsonMetricResult`'s declaration order
  (`src/aiperf/common/models/export_models.py:25-66`). Both sides omit
  non-finite values rather than writing `NaN`/`Infinity`: Python via
  `scrub_non_finite` + `exclude_none`
  (`src/aiperf/common/finite.py`, applied at
  `src/aiperf/exporters/metrics_json_exporter.py:195-203` — the comment there
  states the round-trip exists precisely so non-finite values are not coerced to
  `null`), Rust via `finite()` + `insert_number`
  (`rust/runtime/src/export/genai_perf.rs:324-330`). Top-level order
  (`schema_version, aiperf_version, benchmark_id, declared metrics,
  telemetry_data, input_config, run_info, was_cancelled, error_summary,
  warmup_metrics, …`) matches, and `error_summary` is present-as-empty-array on
  both.
- **`error_summary` item shape.** `{"error_details": {code?, type, message},
  "count": N}`, with `code` omitted when absent
  (`rust/runtime/src/export/genai_perf.rs:679-694`).
- **Timeslice exports.** Both default to no timeslicing
  (`slice_duration = None`: `src/aiperf/config/artifacts.py:74` vs
  `rust/runtime/src/config/model/metrics.rs` slice field optional). JSON shape
  is `{"timeslices": [{start_ns, end_ns, [is_complete], metrics…}],
  "input_config": {…}}` with `is_complete` emitted *only* for partial trailing
  slices on both sides
  (`src/aiperf/common/models/export_models.py:136,155` and
  `src/aiperf/exporters/timeslice_metrics_json_exporter.py:76` vs
  `rust/runtime/src/export/timeslice.rs:292-296`), and slice order is conveyed
  by array position with no index field. Partial trailing windows are emitted by
  both.
- **`inputs.json`.** `{"data": [{"session_id": …, "payloads": [ …verbatim
  bodies… ]}]}` on both sides
  (`src/aiperf/dataset/dataset_manager.py:474-482` vs
  `rust/runtime/src/engine/records.rs:819-827`), pretty-printed.
- **Secret redaction — re-confirmed against baseline.** No leak found. This
  negative was re-derived from the baseline checkout after the baseline
  correction; `common/redact.py`, `config/endpoint.py` and
  `raw_record_writer_processor.py` are all byte-identical to the branch, so the
  conclusion is unchanged. Specifically:
  - Baseline `_SENSITIVE_HEADER_NAMES` (`src/aiperf/common/redact.py:21-33`) is
    exactly nine entries — `authorization`, `proxy-authorization`, `x-api-key`,
    `api-key`, `ocp-apim-subscription-key`, `x-goog-api-key`, `x-functions-key`,
    `aeg-sas-key`, `x-amz-security-token` — matching Rust's
    `SENSITIVE_HEADER_NAMES` (`rust/runtime/src/config/redact.rs:15-25`)
    name-for-name, and both substitute the identical `"<redacted>"` sentinel
    (`src/aiperf/common/redact.py:8`).
  - `input_config` in the summary JSON is redacted at *serialization* time by
    Pydantic field serializers, so runtime credentials stay intact:
    `src/aiperf/config/endpoint.py:128-129` (`urls`), `:180-181` (`api_key`,
    `when_used="json"`), `:287-288` (`headers`, `when_used="json"`), with
    defense-in-depth at `src/aiperf/common/models/model_endpoint_info.py:86-89`.
  - Raw-record request headers are redacted on both sides
    (`src/aiperf/post_processors/raw_record_writer_processor.py:108`,
    `request_headers=redact_headers(record.request.request_headers)`, vs
    `rust/runtime/src/engine/records.rs:1022,1069`).
  - The external-push paths redact too: MLflow params and metadata via
    `redact_url`/`redact_cli_command`
    (`src/aiperf/exporters/mlflow_data_exporter.py:362,382,447,451,486`), W&B
    config via `redact_cli_command`
    (`src/aiperf/exporters/wandb_data_exporter.py:165`), and OTLP resource
    attributes via `redact_url`
    (`src/aiperf/post_processors/otel_streaming_fanout.py:116`).
  - Error strings are scrubbed with the general-purpose regex path
    (`src/aiperf/common/models/error_models.py:42`,
    `redact_string(repr(value))`); Rust's equivalent lives in
    `rust/runtime/src/engine/redaction.rs`. I did not construct an error whose
    message embeds a bearer token on both sides, so this specific sub-path is
    asserted from code shape rather than observed output.
- **MLflow metric key layout and run naming.** `tag` for the representative
  value, `tag.<stat>` otherwise, plus `aiperf.completed_requests` and
  `aiperf.total_expected_requests`
  (`src/aiperf/exporters/mlflow_data_exporter.py:343-354`); default run name
  `aiperf-<benchmark_id[:8]>`
  (`src/aiperf/exporters/mlflow_data_exporter.py:324-327` vs
  `rust/runtime/src/export/mlflow.rs:167-169`).
- **`outputs.json` content rules.** Same six-metric allowlist
  (`src/aiperf/post_processors/outputs_json_record_processor.py:60-67`), same
  profiling-phase-only filter, same `(session_num, turn_index)` ordering, same
  key order.

## Withdrawn after baseline correction

Nothing. All 14 findings survive against baseline rev `bc359bf8fd`; three were
restated (findings 3, 12, 13) and one was strengthened (finding 1). The
per-finding outcomes are tabulated in "Baseline correction: finding
classification" above.

One *unverified item* was resolved rather than withdrawn: baseline
`RecordsExportFormat = Literal["jsonl"]` (`src/aiperf/config/artifacts.py:38`)
means Python rejects `records: [csv]` and `records: [parquet]` at config-validation
time and has no writer for either, so `profile_export_records.csv` and
`profile_export.parquet` are unambiguously Rust-only rather than a shared
capability that silently no-ops on one side. Folded into finding 3.

## Unverified / needs runtime check

- **CSV integer formatting.** Python's `_format_number` returns
  `f"{int(value)}"` for `numbers.Integral`
  (`src/aiperf/exporters/metrics_csv_exporter.py:131-133`) while Rust always
  formats `{:.2}` (`rust/runtime/src/export/genai_perf.rs:709-714`). Every value
  in the Python sample rendered with `.00`, so the integer branch may be
  unreachable in practice (`MetricResult` stats appear to be floats). Would need
  a Python run where a `min`/`max` stat is an `int` to know whether this ever
  produces `6` vs `6.00`.
- **Metric population differences from `_prepare_metrics`.** Baseline
  `metrics_base_exporter._prepare_metrics` (lines 30-64) drops any metric flagged
  `INTERNAL` or `EXPERIMENTAL` unless `Environment.DEV.SHOW_INTERNAL_METRICS` /
  `SHOW_EXPERIMENTAL_METRICS` is set. I confirmed this does not affect findings
  5-8 (schema shape) or 13 (`decode_duration` carries neither flag), but I did
  not enumerate Rust's equivalent flag-gating, so whether the two sides export
  the *same set* of metrics under default env is a metrics-coverage question
  outside this domain.
- **W&B payload shape.** P1.37 records that Rust is offline-only. I did not
  compare the individual `wandb` summary/config key names between
  `src/aiperf/exporters/wandb_data_exporter.py:161-165` and
  `rust/runtime/src/export/wandb/`; that needs a `.wandb` datastore decode
  against a live Python run to compare key-for-key.
- **`http_req_*` and `usage_prompt_cache_read_*` summary keys.** The Python
  sample carries 19 keys the Rust sample lacks (`http_req_duration`,
  `http_req_waiting`, `usage_prompt_cache_read_tokens`,
  `overall_usage_prompt_cache_read_pct`, …). The two samples ran different
  configurations (the Python one had HTTP tracing on), so this is not evidence
  of a gap. Determining whether Rust emits these keys under `--export-trace`
  belongs to the metrics-coverage domain and needs a matched pair of runs.
- **Warmup-phase and zero-record edge shapes.** Rust omits `warmup_metrics`
  when the collected set is empty
  (`rust/runtime/src/export/genai_perf.rs:580-589`) and omits both summary
  artifacts' metric bodies when no metric projects. I did not run a zero-record
  or all-failed benchmark on either side, so the exact file shape on those paths
  (whether Python writes an empty-metrics summary where Rust writes none, or
  vice versa) is unconfirmed.
