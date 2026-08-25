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

Result: 12 passed, 0 failed, 1742 filtered out. The Task 2 subset covers the
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
  `(accepted + steps) / (drafted + steps)` with checked integer arithmetic.
- Sketch storage keeps a phase aggregate and an optional phase-instance sketch,
  so ordinary phase summaries preserve the existing public lookup contract
  while indexed summaries do not require exact rows.
- Histogram pools are exact `BTreeMap`s even in sketch mode and return `None`
  for time-bounded summaries because exact row-to-bucket ownership is not
  retained there.
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
