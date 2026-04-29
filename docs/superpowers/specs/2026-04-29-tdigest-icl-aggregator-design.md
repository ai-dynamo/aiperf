<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# T-Digest Aggregator for `InterChunkLatencyMetric`

## Problem

`InterChunkLatencyMetric` (ICL) is the only `BaseRecordMetric[list[int]]` in
the codebase: each request contributes a list of inter-chunk gap durations,
one element per gap between consecutive streamed chunks. The shape is
intentional — collapsing per-request gaps to a single mean would defeat the
purpose of the metric (otherwise it would be indistinguishable from
inter-token latency).

At the run level, `MetricResultsProcessor` extends a `MetricArray` (a NumPy-
backed `GrowableArray` of `float64`) with the full per-request list. This
gives exact statistics — at a memory cost that grows linearly with
`records × chunks_per_record × 8 B`.

A 1 M-request benchmark with ~5 000 chunks per response (long-context
streaming) puts the records-manager at ~40 GB of resident memory just for
ICL's run-level array. That far exceeds the records-manager pod's memory
budget and was the root cause of an OOM on the `ajc/k8s` ramp at 1 M
concurrency.

## Solution

Replace the run-level `MetricArray` for list-valued record metrics with a
`TDigestListMetricAggregator` backed by a t-digest sketch. T-digest gives:

- **Constant memory** — a few KB per metric per timeslice, regardless of
  sample count.
- **Exact `count`, `sum`, `min`, `max`, `avg`, `std`** — tracked by the
  aggregator itself (running scalars: count, sum, sum-of-squares, min,
  max). The t-digest sketch is consulted only for percentiles.
- **Approximate percentiles** — relative error ~0.5 % on p50/p90/p99
  via t-digest. Side-channel scalars cost 5 × `float64` regardless of
  sample count.

The error band on percentiles (~0.5 %) is well below the run-to-run jitter
of any real LLM-inference benchmark, so the swap is observationally
indistinguishable from exact for the run-level summary path. All other
stats stay byte-equal to `MetricArray`.

## Why no flag, no mode, no config

A previous design (on the `ajc/k8s` branch) gated this behind a
`MetricFlags.AGGREGATE_TDIGEST` flag and a global `ListMetricAggregationMode`
config option, with ICL set to `AGGREGATE_TDIGEST`. That design was
considered and rejected for this port because:

1. **ICL is the only consumer.** Grep confirms `InterChunkLatencyMetric`
   is the only `BaseRecordMetric[list[...]]` in the codebase. There is
   no second list-valued record metric to differentiate from.
2. **No second mode is useful.** Any future list-valued record metric
   that grows with `records × samples_per_record` will hit the same
   memory cliff and need t-digest. Any list-valued record metric that
   stays small enough for exact storage hasn't been proposed.
3. **Auto-select by sample count is a YAGNI trap.** Switching mid-stream
   would mean either re-aggregating or discarding samples already seen.
   T-digest at low N already approximates exact (centroid count ≤ sample
   count).
4. **The 0.5 % percentile error is below benchmark noise.** No code path
   (export, plot, test) currently makes a tighter assumption.

If a future metric genuinely needs an exact path back, it can be re-added
with one isinstance branch in `_create_metric_result`. We do not pay the
abstraction cost up-front.

## Scope

### In scope

1. New module `src/aiperf/metrics/list_metric_aggregation.py` (~50 LOC)
   with a single class `TDigestListMetricAggregator` that mirrors the
   `MetricArray.to_result(tag, header, unit) -> MetricResult` contract.
2. `MetricResultsProcessor.process_result()`: when a `RECORD`-typed
   metric value arrives as a `list`, store a `TDigestListMetricAggregator`
   in `_results[tag]` (instead of extending a `MetricArray`).
3. `MetricResultsProcessor._create_metric_result()`: add an `isinstance`
   branch for the t-digest aggregator, delegating to its `to_result`.
4. New dependency: `tdigest~=0.5.2.2` in `pyproject.toml`.
5. Test retrofit:
   `tests/unit/post_processors/test_metric_results_processor.py::
   test_process_result_record_metric_list_values` — assert the new
   aggregator type and its stat outputs.
6. New test file: `tests/unit/metrics/test_list_metric_aggregation.py` —
   unit tests for the aggregator class (`extend`, `append`, `to_result`,
   stat correctness within tolerance).

### Out of scope

- Any change to per-record metric storage or per-record JSONL export
  (the `RawRecordWriterProcessor` path stays exact, untouched).
- Any change to the `aiperf plot` data loader (it reads per-record
  values, not the run-level aggregate).
- `MetricFlags.AGGREGATE_TDIGEST` flag, `ListMetricAggregationMode`
  enum, `build_list_metric_aggregator(mode)` builder, or any
  config / CLI knob.
- Anything from the `2d76a5e54` "msgspec metric wires" commit on
  `ajc/k8s`. The msgspec wire-encoding refactor is a separate concern
  and would land as its own PR.

## Architecture

### Aggregator contract

`TDigestListMetricAggregator` is a thin façade over `tdigest.TDigest`:

```python
class TDigestListMetricAggregator:
    def __init__(self) -> None: ...
    def append(self, value: int | float) -> None: ...
    def extend(self, values: Iterable[int | float]) -> None: ...
    def to_result(
        self, tag: MetricTagT, header: str, unit: str
    ) -> MetricResult: ...
```

The constructor is parameter-free. The class deliberately does NOT
inherit from `MetricArray` — it's a sibling type that shares the
`to_result` contract.

The two write methods (`append` for scalar, `extend` for list batches)
mirror the surface that `MetricResultsProcessor` already calls on
`MetricArray`. `extend` consumes any iterable; the implementation calls
`self._td.update(value)` once per element. (T-digest's `batch_update`
is internally a Python-level loop, so there is no batched fast-path to
exploit.)

`to_result` returns the same `MetricResult` schema the rest of the
processor expects — `min`, `max`, `avg`, `sum`, `std`, `count`, and the
nine percentile fields (`p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`,
`p95`, `p99`). `count`, `sum`, `min`, `max`, `avg`, `std` come exact
from the aggregator's running side-channel scalars (formula in
`Implementation notes`). Only `p1`...`p99` are read from the t-digest
sketch and carry the ~0.5 % relative error.

### Selection rule in the processor

`MetricResultsProcessor.process_result()` already type-dispatches on
`metric_type == MetricType.RECORD`. Inside that branch, the existing
code does:

```python
if tag not in results_dict:
    results_dict[tag] = MetricArray()
if isinstance(value, list):
    results_dict[tag].extend(value)
else:
    results_dict[tag].append(value)
```

The new code:

```python
if tag not in results_dict:
    # If the very first value is a list, this metric will be list-valued
    # forever (the metric class shape doesn't change mid-run). Pick the
    # storage type at first-touch.
    results_dict[tag] = (
        TDigestListMetricAggregator()
        if isinstance(value, list)
        else MetricArray()
    )
if isinstance(value, list):
    results_dict[tag].extend(value)
else:
    results_dict[tag].append(value)
```

This is a one-line structural change at the lazy-init site. The
`extend` / `append` calls below stay byte-identical because the t-digest
aggregator exposes the same two methods.

`_create_metric_result` gains one isinstance branch:

```python
if isinstance(values, TDigestListMetricAggregator):
    return values.to_result(tag, metric_class.header, str(metric_class.unit))
if isinstance(values, MetricArray):
    return values.to_result(tag, metric_class.header, str(metric_class.unit))
```

The two branches collapse to "values must expose `to_result`" once we
have a Protocol — but we do not introduce that abstraction here. Adding
a Protocol now would only serve a hypothetical future. One isinstance
branch is the YAGNI-minimal expression of today's reality.

### Data flow

```mermaid
flowchart LR
    A[ParsedResponseRecord] --> B[InterChunkLatencyMetric._parse_record]
    B -->|list[int]| C[MetricRecordsData]
    C --> D[MetricResultsProcessor.process_result]
    D -->|tag is list-valued, first touch| E[TDigestListMetricAggregator]
    E -->|append/extend| E
    D -->|finalize| F[_create_metric_result]
    F -->|isinstance TDigest| G[to_result]
    G --> H[MetricResult]
    H --> I[JsonMetricResult export]
```

The per-record path (not shown) flows through
`RawRecordWriterProcessor` and is unchanged: each request's full ICL
list is preserved on disk in the JSONL records, exactly as today.

## Implementation notes

- **Side-channel scalars.** The aggregator maintains five running
  scalars alongside the digest: `count`, `sum`, `sum_sq`, `min`, `max`.
  Each `append` updates all five (and the digest); each `extend(v)`
  updates them in a tight loop. Cost per element: 4 float ops + 1 int
  increment + one `td.update(v)`.
- **Exact mean and std.** `avg = sum/count`. `std =
  sqrt(max(0, sum_sq/count - avg*avg))` — population std, matching
  `np.std(arr)`'s default (which is what `MetricArray.to_result` uses).
  The clamp via `max(0, ...)` guards against tiny floating-point
  underflow when all samples are equal.
- **Min/max from side-channel.** `tdigest~=0.5.2.2` does not expose
  `min`/`max` as O(1) accessors. Tracking them ourselves is simpler
  and bit-exact.
- **Empty digest.** `to_result` on an empty aggregator returns a
  `MetricResult` with `count=0` and the percentile / min / max / avg /
  std fields set to `None`, matching `MetricArray`'s behavior. The
  exporter (`JsonMetricResult`) already declares these `| None`.
- **Determinism under tests.** T-digest's centroid layout depends on
  insertion order. Our auto-fixture pins `RNG=42`, so deterministic
  test data produces deterministic centroid output. Tests assert
  percentile values via `pytest.approx(..., rel=0.005)` (0.5 %) to give
  the digest enough room across small fixture sizes.

## Error handling

- Malformed input (e.g. NaN, `-inf`) is passed through to `tdigest`
  without filtering — it's the caller's responsibility, identical to
  `MetricArray`'s contract.
- A list value that is empty (`[]`) is a no-op (`extend([])`); the
  aggregator object is created but no centroids are added. `to_result`
  on an empty aggregator returns the `count=0` shape above.

## Testing

### `tests/unit/metrics/test_list_metric_aggregation.py` (new)

Coverage:

- `test_empty_aggregator_returns_count_zero_result` — `to_result` on a
  fresh aggregator has `count=0` and `None` for all stat fields.
- `test_append_single_value_count_one` — one call, `count == 1`,
  `min == max == avg == value`, `std == 0`.
- `test_extend_with_list_count_matches_len`.
- `test_min_max_exact_across_random_inputs` — generate 10 000 floats
  with seeded RNG, assert `min`/`max` exact to bit equality.
- `test_count_sum_exact_across_random_inputs` — same, for count and sum.
- `test_avg_std_exact_against_numpy` — generate 10 000 floats with
  seeded RNG, assert `avg` matches `np.mean(arr)` and `std` matches
  `np.std(arr)` to within float64 round-off (`pytest.approx(rel=1e-12)`).
- `test_percentiles_within_tolerance` — generate 100 000 floats from a
  known distribution (uniform), assert
  `p50 == approx(0.5, rel=0.005)`, `p99 == approx(0.99, rel=0.005)`.
- `test_to_result_schema_matches_metric_array` — feed identical input
  through `MetricArray` and `TDigestListMetricAggregator`; assert the
  `MetricResult` field set is identical, the bit-exact fields
  (`count`, `sum`, `min`, `max`, `avg`, `std`) match exactly, and the
  percentile fields agree to `pytest.approx(rel=0.005)`.
- `test_repeated_extend_accumulates` — extend twice, count adds.
- `test_mixed_int_and_float_values` — both pathways accepted.

### `tests/unit/post_processors/test_metric_results_processor.py` (modify)

Existing test `test_process_result_record_metric_list_values` currently
asserts `isinstance(processor._results["test_record"], MetricArray)` and
`list(processor._results["test_record"].data) == [10.0, 20.0, 30.0]`.

Modified assertion: `isinstance(processor._results["test_record"],
TDigestListMetricAggregator)`, plus a stat-shape check (count, min, max
exact). The full list comparison is dropped because t-digest doesn't
preserve individual samples.

The scalar-path test `test_process_result_record_metric` is unchanged
(scalar record metrics still use `MetricArray`).

### Coverage we deliberately do NOT add

- We do not add an end-to-end ICL benchmark test that exercises the
  whole aiperf pipeline at scale. The existing ICL unit tests
  (`tests/unit/metrics/test_inter_chunk_latency_metric.py`) cover
  the per-record path, which is unchanged. The processor-level test
  above covers the run-level path.

## Dependency

- `tdigest~=0.5.2.2` (compatible-release pin, same as `ajc/k8s`).
  Pure-Python; no transitive C/Rust deps.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Percentile assertion in some test exceeds 0.5 % tolerance | Low | Test failures will be visible immediately; bump tolerance per-test if a small-N case needs it. |
| `tdigest` library has a bug at scale we don't catch | Low | Min/max/count/sum/std stay exact via our side-channel — only percentiles flow through the digest. Wide error → visible in any percentile assertion. |
| Future contributor adds a new list-valued record metric assuming exact storage | Low | The processor's first-touch type-dispatch makes the storage choice automatic; the contributor doesn't need to think about it. Worst case: their percentile is 0.5 % off, exact stays exact. |
| Tightly-coupled downstream code expects a `MetricArray` | Low | Searched: only `_create_metric_result` does an isinstance check. Updated. No other consumer touches the run-level storage type directly. |

## Acceptance criteria

1. `make check-ergonomics` passes (no new file/function-size or
   nesting violations).
2. `make check-ruff-baselined` passes (no new ruff baseline entries).
3. `pre-commit run --all-files` passes.
4. `uv run pytest tests/unit/ -n auto` passes — same pass count as
   baseline, plus the new test file's cases.
5. The new module exposes only `TDigestListMetricAggregator` (and its
   imports) — no enum, no flag, no mode, no config.
6. `pyproject.toml` adds exactly one dependency line (`tdigest~=0.5.2.2`).
7. The diff against `origin/main` is under 200 LOC of production code
   (target: ~80 LOC of `list_metric_aggregation.py` + ~10 LOC of
   processor wiring + 1 toml line).

## Out-of-band cleanups bundled

None. The `b334b5c91` commit on `ajc/k8s` bundled helm-chart edits and a
publish-script refactor that have no relationship to the t-digest work.
Those are explicitly excluded from this port and remain on `ajc/k8s`.
