# Task 2 report — metric aggregation and typed report

## Scope

Implemented all eleven canonical speculative-decoding metric identities, six
successful-record projections, equal-request and token-weighted scalar folds,
phase-instance selection, and the exact pooled acceptance histogram. Exact and
sketch stores preserve matching phase aggregates and indexed selections; the
histogram survives sketch row clearing, worker append, MessagePack cellular
partitions, and report construction.

## TDD receipt

Initial RED:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime spec_decode --lib
```

Compilation failed on the intended missing feature surface: the eleven metric
tags, `MetricConsoleGroup::SpecDecode`, `ExportContext::phase_index`, summary
pool accessor, and `NativeReport` pool field did not exist.

Final focused GREEN:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime spec_decode --lib -- --nocapture
```

Result before independent review: 12 passed, 0 failed, 1742 filtered out. The Task 2 subset covers the
canonical worked example, all eleven values, phase-only and concrete-index
selection, warmup exclusion, absence suppression, exact/sketch direct-versus-
append parity, cellular MessagePack fold parity, and native-report histogram.
The count/fingerprint, derived-coverage, and definition-ID snapshot sentinels
also pass in focused runs.

## Adjacent regression evidence

`cargo test -p aiperf-runtime metrics_core:: --lib` initially passed 121 tests
and found the two intended catalog sentinels that were outside the spec-decode
filter. Both were updated and rerun GREEN. Its remaining failure is unrelated
shared-head baseline drift: the existing native-v2 report golden expects
`aiperf_version: 0.0.0`, while the current package supplies `0.12.0`. Task 2
does not change package version handling or that golden.

## Implementation notes

- Record averages stay distributions; total steps/accepted/draft values are
  exact derived sums. Token-weighted acceptance length is
  `1 + accepted / steps`; overall draft acceptance is
  `100 * accepted / drafted`.
- Accepted-per-verified is projected per record as
  `(accepted + steps) / (drafted + steps)` after widening operands, so a
  mathematically defined value is not suppressed by `u64` addition overflow.
- Sketch storage keeps a phase aggregate and an optional phase-instance sketch,
  so ordinary phase summaries preserve the existing public lookup contract
  while indexed summaries do not require exact rows.
- Histogram pools use exact `u128` aggregate counts even in sketch mode and
  return `None` for time-bounded summaries because exact row-to-bucket ownership
  is not retained there. Cellular MessagePack remains a `u64` wire contract and
  refuses, rather than truncates or saturates, an unrepresentable pooled count.
- Console rendering, v1 artifact projection, and processed-record output remain
  Task 3 work. The sole console exporter production change is the exhaustive
  title match required for the new enum variant; the group is not yet ordered
  or parsed for display.
- The untracked `.venv` symlink remains outside every commit.

## Self-review

The implementation adds no lock, channel, task, clock read, or dependency.
Histogram merge order is deterministic, count accumulation uses checked wire
normalization from Task 1, numeric serialization remains finite-or-absent, and
the exact/sketch/cellular folds share one typed store boundary. No production
`unwrap()` or `expect()` was introduced. Independent review is the gate before
Task 3.

## Independent-review fix round

The first independent review rejected Task 2 on five important boundary and
compatibility contracts: unchecked histogram overflow at direct ingest, worker
append, and context pooling; overflow-suppressed accepted-per-verified values;
mid-struct positional MessagePack additions; missing exact-row canonical value
retention; and inconsistent public index-only context selection.

The focused regressions first reproduced five behavioral failures (12 passed,
5 failed). The minimal fix widens pooled counts, performs checked cellular-wire
narrowing, tail-appends every new positional field after legacy
`ingested_total`, retains and clears the optional canonical row value with the
exact/sketch lifecycle, and applies an authored phase index consistently across
exact rows, sketches, and pools. The parent-layout compatibility regression
serializes the exact legacy field sequence and decodes it with the current
store.

Final fix-round GREEN:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime spec_decode --lib -- --nocapture
```

Result: 20 passed, 0 failed, 1742 filtered out. `cargo fmt --all -- --check`
also passed. The adjacent `metrics_core::` sweep passed 130 tests and retained
only the already-recorded unrelated version-golden failure (`0.12.0` actual,
`0.0.0` expected). Scoped re-review of the fix commit remains the Task 3 gate.
