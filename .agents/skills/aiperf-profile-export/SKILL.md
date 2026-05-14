---
name: aiperf-profile-export
description: Use when inspecting an aiperf output directory — "what's in profile_export?", "read the jsonl", "compare these two aiperf runs", "why is metric X showing as NaN", "post-process aiperf output", "plot results", "inspect inputs.json", "what did the model output", "what does the JSONL stream contain". Maps the output artifacts (inputs.json, profile_export.jsonl, profile_export_aiperf.{json,csv}, optional profile_export_aiperf_timeslices.{json,csv}, optional outputs.json, optional server_metrics_export.parquet, optional gpu_telemetry_export.jsonl) to their Pydantic schemas and to which question each one answers.
---

# AIPerf Profile Export Analysis

An `aiperf profile` run produces several output artifacts in its `-o <out>/` directory. They overlap but each answers a different question.

## The artifacts

| File | Contains | When to use | Pydantic model |
|---|---|---|---|
| `inputs.json` | The exact requests sent (prompts, params, request order). | "What did aiperf send?" | `InputsFile` (in `aiperf.common.models`) |
| `profile_export.jsonl` | Per-request records, one JSON object per line. Includes timestamps, raw response bodies, parse status. **This is the authoritative per-request source.** | "What did the server return for request N?", "What were the per-request latency / token-count distributions?", debugging response parsers. | `MetricRecordInfo` and adjacent models in `aiperf.common.models.record_models` |
| `profile_export_aiperf.json` | Aggregate summary — every registered metric's `unit/avg/p1/p5/p10/p25/p50/p75/p90/p95/p99/min/max/std/count/sum`. Schema 1.1 (`count` and `sum` added; `count` may be omitted for derived/scalar metrics, `sum` absent for derived/rate metrics). | "What's the headline summary?" | `JsonExportData` + `JsonMetricResult` in `aiperf.common.models.export_models` (`SCHEMA_VERSION = "1.1"`) |
| `profile_export_aiperf.csv` | Same aggregate summary, CSV-shaped for spreadsheets. | Quick eyeballing in Excel/Numbers/Sheets. | (matches the JSON) |
| `profile_export_aiperf_timeslices.{json,csv}` (when enabled) | Windowed per-slice aggregates over the run's duration — useful for spotting non-stationary behavior. | "How did metrics evolve over the run?" | Windowed variant of the aggregate models above |
| `outputs.json` (when `--export-outputs-json` is set + `--export-level records` or `raw`) | Per-request model responses merged with their matched metrics, keyed by `session_num:turn_index`. | "What did the model actually SAY?" / downstream post-processing of responses. | Assembled by `OutputsJsonExporter` from `output_fragments_*.jsonl` + `profile_export.jsonl` |
| `server_metrics_export.parquet` (optional) | Server-side Prometheus/DCGM telemetry, columnar. **Server metrics ONLY — not per-request data.** | "What was the server's GPU utilization / queue depth over the run?" | Server-metrics schemas in `aiperf.common.models.server_metrics_models` |
| `gpu_telemetry_export.jsonl` (optional) | DCGM GPU telemetry as a streaming JSONL. | "What were per-GPU power / temperature samples?" | GPU-telemetry models |

**There is NO per-request parquet output.** The per-request authoritative source is `profile_export.jsonl`. The summary `profile_export_aiperf.{json,csv}` rounds and truncates. The only parquet emitted is the optional server-metrics one (enabled via `--server-metrics-formats parquet`), and it covers server-side telemetry, not per-request latency. **Model responses live in `outputs.json` (opt-in)** — not in any parquet.

## Read the per-request data

```python
import json
import pandas as pd          # pandas is the project's in-tree dataframe lib

records = []
with open("artifacts/my-run/profile_export.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
df = pd.DataFrame(records)
print(df.columns)
print(df.describe(include="all"))

# Filter errored requests (field name depends on the version's record schema;
# inspect `df.columns` first):
errored = df[df.get("error").notna()] if "error" in df.columns else df.iloc[0:0]

# Latency percentiles (verify the column name against your record schema first):
if "request_latency_ns" in df.columns:
    df["request_latency_ns"].quantile([0.5, 0.9, 0.95, 0.99])
```

For typed access, import the record-schema model from `aiperf.common.models.record_models` (look for `MetricRecordInfo` and adjacent classes) and `model_validate_json` each line. Field names in the JSONL use `_ns` suffixes for time values; inspect the schema before assuming.

`pandas` and `pyarrow` are project deps (`pyproject.toml`). `polars` is NOT a project dep — don't reach for it.

## Read the aggregate summary

```python
import json

summary = json.loads(open("artifacts/my-run/profile_export_aiperf.json").read())
# summary's top-level keys: per-metric stats (avg, p1, p5, p10, p25, p50, p75, p90, p95, p99, min, max, std, count, sum, unit).
# Note: aiperf uses `avg`, not `mean`. `count` and `sum` are schema 1.1 additions and may be absent
# for derived/rate metrics. Schema is JsonExportData (SCHEMA_VERSION="1.1") in aiperf.common.models.export_models.
```

## Read the inputs

```python
from aiperf.common.models import InputsFile
import orjson

inputs = InputsFile.model_validate(orjson.loads(open("artifacts/my-run/inputs.json", "rb").read()))
# inputs.sessions, inputs.config, etc.
```

## Read server-metrics parquet (if enabled)

```python
import pandas as pd

df = pd.read_parquet("artifacts/my-run/server_metrics_export.parquet")
print(df.columns)
```

Only available when the run was launched with `--server-metrics-formats parquet`. Contains Prometheus scrape data, not per-request data.

## Composing tools

| Task | Tool |
|---|---|
| Plot metrics from one or more runs | `aiperf plot` (auto-creates `~/.aiperf/plot_config.yaml` on first use) |
| Re-analyze a saved trace as if it were live | `aiperf analyze-trace` |
| Assemble per-category matrices (SPEED-Bench format) | `aiperf speed-bench-report ./artifacts/run_a/ ./artifacts/run_b/ ...` |
| Quick diff between two runs | Read both jsonl files, join on request index in pandas, diff columns |

## Reproducibility caveat

`--random-seed` controls **dataset synthesis** reproducibility (prompts will be identical across runs). It does NOT control **metric values** — timing measurements vary run-to-run even with identical inputs and identical mock-server flags. Don't expect bit-identical numbers; expect bit-identical inputs.

For deterministic mock latency, pass `--fast` to `aiperf-mock-server` (zero TTFT/ITL).

## Cross-run comparison rules

Convention this skill follows: each `aiperf profile` invocation is one independent sample. When comparing across runs:

- Compare runs that share model, endpoint, request count, concurrency, and seed — varying only the dimension under study.
- Prefer side-by-side reporting over computing aggregates across runs (sum/avg/max across independent runs collapses meaningful differences into one number).
- Per-run percentiles (p50/p90/p99) carry more signal than per-run-means for latency analysis.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "The CSV summary says p99 = 120ms, that's the answer" | The summary rounds. If precision matters, read `profile_export.jsonl` and compute the quantile yourself. |
| "I'll look up `summary['metric']['mean']`" | aiperf uses `avg`, not `mean`. Access `summary[metric]['avg']`. |
| "I'll just `cat profile_export.jsonl \| grep error`" | Bulky and fragile against escaped strings. Parse the JSON lines and filter in pandas. |
| "I'll write my own JSON schema for these files" | Pydantic models exist: `JsonExportData` / `JsonMetricResult` (`aiperf.common.models.export_models`, schema 1.1) for the aggregate; `MetricRecordInfo` (`aiperf.common.models.record_models`) for per-record. Import them. |
| "Two runs have similar configs, I'll average their p99" | Each run is independent. Average isn't meaningful. Report each run; show both. |
| "I'll use polars because that's what I know" | polars is NOT a project dep. Use pandas — it's already imported throughout `src/aiperf/`. |
| "There should be a parquet of per-request data, I'll go find it" | There isn't. `server_metrics_export.parquet` (if present) is server-side telemetry only. Per-request data is `profile_export.jsonl`; model responses are `outputs.json` (opt-in). |

## Common mistakes

- **Reading the JSON summary instead of the per-request JSONL** — loses precision and per-request detail.
- **Globbing `**/*.parquet`** assuming per-request data is there — the only parquet is server-metrics (opt-in).
- **Forgetting `--random-seed`** when comparing across runs — different prompts = different latency distributions; impossible to interpret.
- **Treating absent columns as zeros.** If your metric never registered, the column is missing, not zero. Check the column list with `df.columns` before indexing.

## Composition

- `aiperf-correctness-testing` for the runtime side that produces the artifacts.
- `aiperf-add-metric` if a metric you expected isn't there — your metric didn't register correctly.
