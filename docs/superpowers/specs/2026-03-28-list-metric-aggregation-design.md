# List metric aggregation mode for summary percentiles

## Goal

Add a first-class benchmark config option that controls how list-valued record metrics are aggregated into summary percentile statistics.

The immediate use case is `inter_chunk_latency`, but the setting should apply to all current and future list-valued record metrics.

## Problem

Today, the summary aggregation path treats list-valued record metrics generically by flattening each list into the metric array via `MetricArray.extend(...)`. This produces exact summary percentiles, but it also means summary aggregation must retain every sample from every list-valued metric.

This is acceptable for small runs, but `inter_chunk_latency` can produce very large per-chunk arrays for long streamed responses. We want a first-class way to switch list-valued metric percentile aggregation from exact storage to t-digest-based approximation while keeping the exported summary shape unchanged.

## Non-goals

- Changing how per-record list-valued metrics are collected
- Changing how per-record JSONL or CSV exports include or exclude list-valued metrics
- Changing the output schema for JSON or YAML summary exports
- Changing scalar metric aggregation
- Changing plot-time per-request derived statistics

## User-facing design

Add a new user-facing config field:

- `metrics.list_metric_aggregation: exact | tdigest`

Default:

- `exact`

Semantics:

- Applies only to record metrics whose runtime value is a list
- Applies only to summary aggregation
- Does not affect whether list-valued metrics are present in per-record exports
- Does not affect the names or fields of exported summary metrics

Example:

```yaml
metrics:
  listMetricAggregation: exact
```

```yaml
metrics:
  listMetricAggregation: tdigest
```

## Why this naming

`list_metric_aggregation` is more accurate than `per_chunk_aggregation`.

The existing export toggle `artifacts.per_chunk_data` is about whether raw list-valued metrics are emitted in per-record exports. This new setting is different: it controls how list-valued metrics are aggregated into summary statistics. Naming it `list_metric_aggregation` keeps the scope generic and avoids confusion with export behavior.

## Existing behavior

### Record export behavior

Per-record exports already treat list-valued metrics generically:

- JSONL and CSV record exports omit list-valued metrics by default
- Setting `artifacts.per_chunk_data=true` includes them

This design intentionally preserves that behavior.

### Summary behavior

The summary aggregation path currently:

- creates a `MetricArray` for each record metric
- appends scalar values with `append(...)`
- flattens list values with `extend(...)`
- computes exact percentiles from the resulting full in-memory sample set

This generic handling is the precedent for making the new config apply to all list-valued record metrics rather than only `inter_chunk_latency`.

## Architecture

### Config surface

Add an enum-backed field to the user-facing metrics configuration model.

Requirements:

- Enum values: `exact`, `tdigest`
- Default value: `exact`
- Standard config validation behavior for unknown values
- User-facing YAML/JSON uses camelCase aliases as usual

### Aggregation behavior

The aggregation choice should be applied only at the summary-building boundary where record metrics are consolidated.

Behavior by metric type:

- Scalar record metrics continue using the current `MetricArray` path unchanged
- Aggregate metrics continue using their existing aggregation logic unchanged
- List-valued record metrics use the configured list-metric aggregation mode

Behavior by mode:

- `exact`: preserve current behavior by flattening list values into the exact sample store
- `tdigest`: feed list samples into a t-digest-backed accumulator instead of retaining all samples in the exact array

The key design constraint is that the implementation must not assume the generic exact `MetricArray.extend(...)` path is always the right abstraction boundary for list metrics. The mode switch belongs in the list-metric summary aggregation path, not in metric collection or record export.

## Summary output

Summary output must remain identical across modes.

For both `exact` and `tdigest`, exported summary metrics for list-valued record metrics must continue to emit the same fields already used by `JsonMetricResult`, including:

- `avg`
- `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99`
- `min`
- `max`
- `std`

This preserves compatibility with:

- JSON summary export
- YAML summary export
- downstream tooling that consumes current summary files

No export schema changes are part of this design.

## Dependency model

`t-digest` support is a required dependency of the project once this feature lands.

Requirements:

- no optional-install path
- no runtime fallback from `tdigest` to `exact`
- no environment flag to disable the dependency

If the config selects `tdigest`, the system should use the required installed dependency directly.

## Implementation constraints

- Keep the scope focused on summary aggregation for list-valued record metrics
- Do not change the semantics of `artifacts.per_chunk_data`
- Do not introduce special-case user-facing config only for `inter_chunk_latency`
- Do not broaden the setting to scalar metrics
- Do not change plot-time derived per-request ICL summary code in this work

## Testing

Add tests that cover:

1. Config parsing
   - accepts `exact`
   - accepts `tdigest`
   - rejects invalid enum values

2. Summary shape stability
   - list-valued metric summaries expose the same output fields in both modes

3. Numerical behavior
   - `exact` continues to match current exact results
   - `tdigest` produces percentile values within an agreed tolerance against exact results on fixed list-valued samples

4. Regression coverage
   - scalar metrics are unaffected by the new setting
   - per-record export behavior for list-valued metrics remains controlled only by `artifacts.per_chunk_data`

## Rollout and compatibility

This is a backward-compatible config addition.

- Existing configs continue to work because the default is `exact`
- Existing summary consumers continue to work because output shape is unchanged
- Users opt into approximate list-metric percentile aggregation explicitly by selecting `tdigest`

## Open decisions resolved during brainstorming

- Scope the setting to all list-valued record metrics, not only `inter_chunk_latency`
- Use a first-class user-facing config field, not an environment-only toggle
- Use an enum, not a boolean
- Name the field `list_metric_aggregation`
- Default to `exact`
- Keep output schema identical across modes
- Treat `tdigest` as a required dependency
