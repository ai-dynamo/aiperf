# Task 2 independent review

## Verdict

NOT APPROVED.

Review target: Task 2 commit `a222b4ae7d` against parent `20f7339aa3` and
`docs/specs/2026-08-25-native-spec-decode-acceptance-metrics.md`. Task 3
exporter behavior was not reviewed or started.

## Findings

### Important — exact histogram accumulation can panic or silently lose exactness

`rust/runtime/src/metrics_core/store.rs:1712-1718` adds every valid canonical
histogram count to the phase-instance pool with unchecked `u64 +=`. Two valid
records with counts `u64::MAX` and `1` in the same bucket therefore panic in a
debug build and wrap in a release build. The same unchecked operation is used
when worker stores append (`store.rs:995-999`) and when multiple retained pools
are combined for an export context (`store.rs:1249-1260`).

This is not a place to saturate. Task 2 promises an exact histogram whose count
sum reconciles with `total_spec_decode_steps`; saturation would silently emit a
false pool. Use checked/fallible addition, or retain a wider exact internal sum
and fail a checked conversion at the `u64` report/wire boundary. Add regressions
for direct insertion, `append_store`, and multi-pool context resolution.

Fresh reproduction used only public AIPerf APIs from a temporary crate outside
the repository. Ingesting `{0: u64::MAX}` followed by `{0: 1}` in the same
phase/index terminated with:

```text
thread 'main' panicked at .../metrics_core/store.rs:1718:17:
attempt to add with overflow
```

### Important — a defined accepted-per-verified value is suppressed on `u64` addition overflow

`rust/runtime/src/metrics_core/store.rs:1682-1695` performs both numerator and
denominator additions in `u64` and treats overflow like absence. For a valid
canonical record with `num_draft_tokens = u64::MAX`, `num_spec_steps = 1`, and
zero accepted drafts, the mathematical denominator is nonzero and the required
formula is defined, but `spec_decode_accepted_per_verified` is omitted.

A fresh external probe printed:

```text
accepted_per_verified_present=false
```

Compute the sums in `u128`, or cast each operand before addition. Per the spec,
only a mathematically zero denominator suppresses this metric.

### Important — the new fields break decoding of the existing positional MessagePack wire form

`ColumnStorePartition::{to_bytes,from_bytes}` uses `rmp_serde::to_vec` and
`from_slice` (`rust/runtime/src/cellular/shard.rs:371-378`), so its nested
`ColumnStore` struct is encoded positionally. The parent layout ended with
`phase_codes, correlation_codes, ... sketch, ingested_total`. Task 2 inserts
`phase_indices` between `phase_codes` and `correlation_codes`
(`rust/runtime/src/metrics_core/store.rs:757-760`) and inserts
`spec_decode_histograms` before legacy `ingested_total` (`store.rs:781-792`).
`#[serde(default)]` does not protect fields inserted in the middle of a sequence:
legacy elements shift into the new field types and old partition bytes fail or
misdecode.

Preserve every legacy field in its original order and append both
`phase_indices` and `spec_decode_histograms` after legacy `ingested_total`.
Alternatively introduce an explicitly versioned/named wire DTO. The added test
at `rust/runtime/src/cellular/shard.rs:496-538` is only a current-version
round-trip; add a checked-in parent-era byte fixture that the new decoder must
accept with both new fields defaulted.

### Important — exact `ColumnStore` does not retain the canonical request value required by the design

The spec requires `ColumnStore` to retain the optional canonical
`ObservedSpecDecodeAcceptance` beside each exact row, then clear that row-owned
value after sketch harvest. The complete field list at
`rust/runtime/src/metrics_core/store.rs:748-792` has no such column, and
`populate_spec_decode_metrics` (`store.rs:1668-1720`) only projects scalar
columns and updates the pool. Consequently an exact store or cellular exact
partition cannot recover the canonical per-request value, and sketch clearing
has no row-owned canonical value to clear.

Add an index-aligned optional canonical column, populate it only for successful,
non-cancelled rows, preserve it through exact worker append and cellular
MessagePack, and clear it with the other row-owned state in sketch mode. Tests
should assert exact-row retention and sketch-row removal separately from pooled
histogram survival.

### Important — a public phase-index context produces different exact and sketch selections

`ExportContext` exposes all fields publicly
(`rust/runtime/src/metrics_core/window.rs:48-56`), so a caller can construct
`phase = None, phase_index = Some(i)`. The exact mask ignores `phase_index`
whenever `phase` is absent (`rust/runtime/src/metrics_core/store.rs:1338-1355`).
The sketch resolver applies the index across phases (`store.rs:697-723`), and
the histogram selector also applies it (`store.rs:1250-1255`). The same public
context can therefore summarize all exact rows, only indexed sketch values, and
only indexed pooled counts, violating the required exact/sketch/pool agreement
and the requirement to apply an index when present.

Either make the invalid state unconstructible/explicitly rejected, or make the
exact mask honor the index consistently. Cover this public struct-literal case
in exact and sketch modes.

## Verified behavior

The ordinary-value implementation is otherwise aligned with Task 2:

- all eleven tags are appended with the specified names, kinds, units, display
  order, visibility, console group, and direction metadata;
- the six record projections and five aggregate/derived formulas match the
  design for normal finite values, including zero-denominator suppression;
- speculative metrics and pools are populated only behind the existing
  successful, non-cancelled gate;
- phase-only and phase-plus-index contexts agree in the covered exact/sketch
  happy paths, and phase-only selection merges instances;
- the exact pool survives sketch row clearing, ordinary worker append, and
  current-version cellular MessagePack round trips;
- absent stats produce no speculative metrics, and time-bounded summaries
  return no pooled histogram;
- `AccumulatorSummary` carries the sorted optional pool and `NativeReport`
  clones it through a typed optional field that serializes only when present;
  and
- no Task 3 exporter was implemented. The sole console change is the exhaustive
  title match for the new enum variant; the group is not ordered or parsed for
  rendering in this commit.

No new lock, channel, task, clock read, dependency, production `unwrap()`, or
production `expect()` was introduced. The histogram `BTreeMap` work occurs only
when typed speculative stats are present; the absent fast path remains a single
optional-value branch.

## Fresh verification

Run from `rust/` with the shared environment and isolated target directory:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime spec_decode --lib -- --nocapture
```

Result: 12 passed, 0 failed, 1742 filtered out. The warnings were unrelated to
the reviewed diff. This focused suite proves the normal-value paths but does not
exercise the five boundary/contract failures above.
