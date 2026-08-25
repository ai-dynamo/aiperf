# Task 2 scoped independent re-review

## Verdict

APPROVED.

Review target: fix commit `d7b460325599057012ee67a1c7f1fdffb5d533b1`
over initial Task 2 commit `a222b4ae7d`, with Task 2 parent `20f7339aa3`.
Task 3 was not reviewed or started.

## Rejected-finding verification

### Important — exact histogram overflow at ingest, append, and context pooling

Resolved. `SpecDecodeHistogramPool` now retains `u128` counts
(`rust/runtime/src/metrics_core/store.rs:668`). Direct ingestion widens each
canonical `u64` before addition (`store.rs:1781-1788`); worker append and
multi-instance context pooling add the already-widened values
(`store.rs:1052-1056` and `1315-1326`). The three focused regressions independently
exercise all three paths past `u64::MAX` and obtain the exact value
`18446744073709551616` (`rust/runtime/src/metrics_core/accumulator.rs:1958-2008`).
No path saturates or wraps at the rejected boundary.

### Important — accepted-per-verified overflow suppression

Resolved. The numerator and denominator operands are widened to `f64` before
addition (`rust/runtime/src/metrics_core/store.rs:1755-1763`), so valid `u64`
inputs cannot overflow an intermediate integer. Only a mathematically zero
denominator suppresses the metric. The regression at
`rust/runtime/src/metrics_core/accumulator.rs:2011-2022` covers the former
overflow boundary and obtains `0.5`.

### Important — legacy positional MessagePack compatibility

Resolved. Every legacy `ColumnStore` field retains its parent order through
`ingested_total`; all Task 2 additions now follow it at the positional tail
(`rust/runtime/src/metrics_core/store.rs:795-848`). The legacy-layout regression
mirrors the exact `20f7339aa3` field sequence, serializes that sequence, and
successfully decodes it into the current store with new fields defaulted
(`store.rs:1879-1955`). The current exact/sketch cellular round-trip remains
covered at `rust/runtime/src/cellular/shard.rs:495-543`.

### Important — canonical exact-row and sketch/cellular lifecycle

Resolved. An index-aligned optional canonical column is tail-appended at
`rust/runtime/src/metrics_core/store.rs:846-847`, populated only inside the
successful/non-cancelled speculative projection (`store.rs:1741-1780`), copied
through exact worker append (`store.rs:1068-1069` and `1121-1124`), exposed by a
typed accessor (`store.rs:1302-1305`), and cleared with other row-owned sketch
state (`store.rs:940-955`). Exact retention, sketch clearing, exact/sketch append,
and current cellular MessagePack survival are covered by the focused tests at
`rust/runtime/src/metrics_core/accumulator.rs:1885-1955`, `2070-2095`, and
`rust/runtime/src/cellular/shard.rs:495-543`.

### Important — index-only public context disagreement

Resolved. `ColumnStore::mask_for` now selects the phase/index branch whenever
either component is present and applies an index without requiring a phase
(`rust/runtime/src/metrics_core/store.rs:1403-1427`). This agrees with sketch
resolution and histogram selection. The public struct-literal regression at
`rust/runtime/src/metrics_core/accumulator.rs:2024-2068` proves exact, sketch,
and pooled equality across profiling and warmup records for
`phase = None, phase_index = Some(0)`.

## Widened report and cellular wire boundary

The widened value remains typed through `AccumulatorSummary` and `NativeReport`
as `BTreeMap<u64, u128>`
(`rust/runtime/src/metrics_core/accumulator.rs:353-362` and
`rust/runtime/src/metrics_core/report.rs:1080-1095`). A fresh external probe
constructed a two-record pool equal to `u64::MAX + 1`; direct `NativeReport`
JSON serialization preserved the exact decimal literal
`18446744073709551616`. This matches the production native-report boundary,
which uses `serde_json::to_string_pretty`
(`rust/runtime/src/report.rs:42-45`).

`serde_json::to_value` cannot represent a numeric `u128` above `u64::MAX` and
returns `number out of range`; that is not used by the production native-report
writer and is not a Task 2 blocker. Any future `Value`-building exporter must
handle that serializer limitation explicitly; no Task 3 exporter was reviewed
here.

The cellular MessagePack representation intentionally remains the compatible
`u64` count form. Its custom serializer performs `u64::try_from` and returns a
structured encoding error rather than truncating or saturating
(`rust/runtime/src/metrics_core/store.rs:670-712`). The focused cellular
regression at `rust/runtime/src/cellular/shard.rs:545-596` proves refusal above
`u64::MAX`.

## Remaining fix scan

No new blockers found. The fix adds no lock, channel, task, clock access,
dependency, logging, or production `unwrap()`/`expect()`. The absent-stat path
does not allocate. Exact mode clones and boxes the canonical value only for a
record that actually carries speculative-decoding stats; sketch mode drops that
row-owned value immediately after harvesting while retaining only the required
pool and sketches. MessagePack narrowing allocates temporary maps only at the
cellular serialization boundary, not per request or token.

## Fresh verification

Run from `rust/` at exact HEAD `d7b460325599057012ee67a1c7f1fdffb5d533b1`:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime spec_decode --lib -- --nocapture
```

Result: 20 passed, 0 failed, 1742 filtered out. All 84 warnings were unrelated
to the reviewed diff. `cargo fmt --all -- --check` and
`git diff --check a222b4ae7d d7b460325599057012ee67a1c7f1fdffb5d533b1`
also completed successfully. The only untracked worktree entry remained the
pre-existing `.venv` symlink.
