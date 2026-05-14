---
name: aiperf-add-metric
description: Use BEFORE adding a new metric, derived metric, aggregate, or modifying metric dependencies in aiperf — "add a new metric", "compute X from Y", "I want to track Z in profile_export", "add a derived metric", "register an aggregate". The metric system has three base classes and a registry validator that raises MetricTypeError at startup (not at definition) on bad deps, so locally-passing tests + broken-in-prod is the common failure mode.
---

# AIPerf Add Metric

The aiperf metrics system has three base classes, auto-registration via `__init_subclass__`, and a dependency validator that runs at startup. Common failure mode: locally the test passes, but `aiperf profile` fails at boot with `MetricTypeError` from `metric_registry.py` (`_validate_dependencies`, lines 202-244).

## Base class decision

| Base | When | Example |
|---|---|---|
| `BaseRecordMetric` | Per-request raw values (latency, TTFT, ITL). Recorded inline as the request completes. | TimeToFirstToken, RequestLatency |
| `BaseAggregateMetric` | Aggregations over the full run (counts, sums, percentiles). Computed at the end. | TotalRequestCount |
| `BaseDerivedMetric` | Computed from OTHER metrics (e.g. RPS = total_requests / duration). Declares `required_metrics`. | RequestsPerSecond |
| `BaseAggregateCounterMetric` | Specialized counter aggregate. | (subclass when you need a tight counter loop) |
| `DerivedSumMetric` | Convenience for "sum of metric X". | InputTokenThroughput |

When in doubt: read sibling metrics under `src/aiperf/metrics/`.

## Steps

### 1. Pick the base class and place the file

```
src/aiperf/metrics/
  base_record_metric.py
  base_aggregate_metric.py
  base_derived_metric.py
  ...
  your_new_metric.py     # new file here
```

Auto-registration: `__init_subclass__` on the base wires it to `MetricRegistry` at import time. You don't add an entry to a YAML; the import side-effect is the registration.

### 2. Implement

```python
# src/aiperf/metrics/your_new_metric.py
from typing import ClassVar
from aiperf.common.enums import MetricType
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.types import MetricTagT

class YourNewMetric(BaseDerivedMetric):
    tag: ClassVar[MetricTagT] = "your_new"
    display_unit: ClassVar[str] = "ms"        # see display_units.py for canonical strings
    required_metrics: ClassVar[set[MetricTagT] | None] = {DepMetricA.tag, DepMetricB.tag}

    def compute(self, ctx) -> float:
        a = ctx.get(DepMetricA.tag)
        b = ctx.get(DepMetricB.tag)
        return a / b if b else 0.0
```

The class-level attributes are `ClassVar`s, not Pydantic fields. `tag` is a string identifier; `required_metrics` is a **set of tag strings** (not a list of `MetricType`). The base `BaseDerivedMetric` declares `type: ClassVar[MetricType]` — you don't set that on subclasses. Inspect `src/aiperf/metrics/base_metric.py` for the exact attribute contract before implementing.

If the metric is record-style (per-request), inherit from `BaseRecordMetric` and implement its required method (inspect the base class for the exact signature — naming may differ from `compute`).

### 3. Declare `required_metrics` correctly

The registry validator (in `src/aiperf/metrics/metric_registry.py` — `_validate_dependencies`, roughly lines 202-244) walks each declared dependency at startup and raises `MetricTypeError` on:

- A `required_metrics` tag that isn't registered.
- A required metric whose `type` is not in `_allowed_dependencies_by_type[self.type]` (e.g., a `RECORD` metric requiring an `AGGREGATE` dependency).

Cycle detection lives in a separate method (`create_dependency_order` in the same file) and raises when the topological sort can't terminate — different error path, also at startup.

Unit tests mocking metric instances pass without exercising this validator. Run the full registry validation:

```bash
aiperf profile --request-count 1 --concurrency 1 --model gpt-4o-mini --url <mock-url> --tokenizer builtin
# If startup logs "MetricTypeError", fix the deps before continuing.
```

### 4. Display units

Add a canonical display unit string to `src/aiperf/metrics/display_units.py` if your metric uses a unit not already there. The exporters (CSV/JSON/parquet) consult this for column labels.

### 5. Wire to exporters (if column-naming is non-default)

By default, the parquet/JSON/CSV exporters auto-include registered metrics. If your metric needs a custom column name, a custom format, or to be excluded from one export format, edit the exporter under `src/aiperf/exporters/`.

### 6. Validate end-to-end

```bash
# 1. mock server (delegate to aiperf-mock-server)
# 2. profile run
aiperf profile --model gpt-4o-mini --url $MOCK_URL --request-count 50 --concurrency 4 --random-seed 42 --tokenizer builtin -o /tmp/metric-check/
# 3. inspect parquet
python -c "import json, pandas as pd; df = pd.DataFrame([json.loads(l) for l in open('/tmp/metric-check/profile_export.jsonl')]); print(df.columns); print(df['your_new'].describe())"
```

Confirm:
- The column exists.
- Values are in the expected range.
- No NaN cluster from an unhandled-edge-case in `compute()`.

### 7. Docs (mandatory)

Per CLAUDE.md's Documentation table, metric definitions OR formulas update `docs/metrics-reference.md`. Add an entry with the formula, units, dependencies, and any caveats (e.g. "undefined when N < 2").

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll skip `required_metrics`, the import order will handle it" | The registry validator runs at startup and fails. Declare deps explicitly. |
| "Unit tests pass with MagicMock metrics, ship it" | The registry validator never fires under MagicMock. Run a real `aiperf profile` once before merging. |
| "Display unit doesn't matter, the exporter will figure it out" | The exporter labels the column with whatever string you set. A wrong unit ships to users; a missing unit produces blank labels. |
| "I'll skip docs/metrics-reference.md, the code is self-documenting" | Per project convention, metric changes are documentation-required. PR blocks otherwise. |
| "My metric divides by N; I'll let it raise on N=0 in production" | Edge-case at startup is fine; edge-case mid-run causes lost samples. Return 0.0 or NaN explicitly. |

## Common mistakes

- **Picking the wrong base class.** `BaseDerivedMetric` for things that should be `BaseAggregateMetric` produces re-computation per record (slow + wrong).
- **Missing `metric_type` declaration.** `__init_subclass__` needs it to register.
- **Importing the new module never happens.** If your module isn't imported during `aiperf` startup, the auto-registration doesn't fire. Check that `__init__.py` or another always-imported module pulls it in.
- **Using a metric value that hasn't been computed yet.** Derived metrics run in dependency order; if you depend on an aggregate that's computed at end-of-run, your derived metric also runs at end-of-run, not mid-stream.

## Composition

- `aiperf-correctness-testing` to confirm the metric appears in `profile_export.jsonl` for each relevant endpoint.
- `aiperf-add-env-var` if the metric introduces a tunable (threshold, percentile selection, etc.).
