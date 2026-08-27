# metrics_core regression at eadd5c665f — root cause and fix

## Summary

One production-code defect, one three-line fix, zero test changes.

Root-cause commit: **e8456a5720** `refactor(endpoints,metrics): introduce interned boundary ids`.

That commit replaced the `MetricTag` enum in `rust/runtime/src/metrics_core/catalog.rs`
with a macro-generated interned id in `rust/runtime/src/metrics_core/tag_id.rs`
(`define_builtin_metric_tags!`). The transcription of the 138 variants was **not
order-preserving**. Two blocks were silently regrouped "logically":

| tag | order in old enum / in `CATALOG` | order in the new macro |
|---|---|---|
| `TimeToLastRoundTrip` | 126 (after `ActiveTotalThroughput`) | 15 (after `TimeToFirstOutputToken`) |
| `AverageRoundTripTime` | 127 (after `ActiveTotalThroughput`) | 16 |
| `InterChunkLatency` | 16 (before `DecodeDuration`) | 19 (after `DecodeDuration`) |
| `DecodeDuration` | 17 | 18 |

The variant *set* was identical, so the change compiled and every name-keyed path
(serde, `resolve`, `as_str`, registry) kept working. But dense identity is a
positional contract:

```rust
// runtime/src/metrics_core/catalog.rs:2030-2040
/// `CATALOG` is ordered by declaration discriminant (guarded by the
/// `catalog_is_discriminant_ordered` test).
&CATALOG[tag.index()].def
Some(&CATALOG[tag.index()])
```

`CATALOG` (a 138-row `static` array in `catalog.rs`) was **not** reordered, and
its row order still matched the historical enum exactly (verified by extracting
both orderings and diffing them). So after e8456a5720, `metric_definition(tag)`
and `spec_for(tag)` returned the **wrong catalog row** for every tag whose dense
index shifted — the whole span from index 14 through 126, i.e. ~113 of 138
metrics. Each affected tag inherited a neighbouring metric's unit, value kind,
console group, flags (including `LARGER_IS_BETTER` and the rate-derivation
flags), and dependency list.

Subsequent commits `f5f79956d4` and `743fb11664` adapted call sites to the new
interned type but did not touch the ordering, so the defect survived to
`eadd5c665f`.

## Failure classes, all one cause

1. **`metrics_core::catalog::tests::catalog_is_discriminant_ordered`** — the
   direct guard. `CATALOG` row 14 held `InterTokenLatency`, whose new
   `index()` was 16. This test is exactly the tripwire that was supposed to
   catch the regression; it fired and was merged red.
2. **`catalog::tests::websocket_lag_tags_append_after_existing_dense_identities`** —
   asserts the two round-trip tags are appended *after* `ActiveTotalThroughput`
   (the documented append-only rule for new dense identities).
   `left: 14, right: 127`.
3. **`catalog::tests::metric_definition_matches_catalog`** — asserts
   `metric_definition(s.tag)` is pointer-equal to `&s.def`; broken by the same
   index shift.
4. **`metrics_core::accumulator::tests::*` (6 tests)** — `None` vs `Some(10.0)`,
   `Some(9.0)` vs `Some(1.0)`: the accumulator read the wrong `MetricSpec`, so
   value kinds, rate flags and aggregation kinds were mismatched and metrics
   were computed into the wrong slots or skipped entirely.
5. **`metrics_core::report::tests::*` (2 tests)** — the serialized v2 report
   dropped `"rate"` stats and emitted `output_token_throughput` where
   `request_throughput` was expected: derived rate metrics keyed off the
   mis-resolved specs.
6. **`metrics::tests::*` (4 tests), `accuracy::tests::…`,
   `gpu_telemetry::accumulator::tests::…`, `realtime::tests::…`** — downstream
   consumers of the same corrupted spec lookups.

## Test vs production verdict

**All 18 failing tests were correct; production code was wrong.** No test was
weakened, deleted, ignored, or re-baselined, and no `#[allow]` was added. The
project contract (`CLAUDE.md`: metric identity, catalog ordering and report
serialization are behavior-preserving) plus the in-code doc comment at
`catalog.rs:2030` and the append-only rule encoded in
`websocket_lag_tags_append_after_existing_dense_identities` all authorize the
old ordering as the ground truth. There is no documented intentional contract
change in any of the eleven suspect commits — the reordering was incidental to
a mechanical transcription.

## Fix

`rust/runtime/src/metrics_core/tag_id.rs`: restore the `define_builtin_metric_tags!`
declaration order to the `CATALOG` row order — move `TimeToLastRoundTrip` and
`AverageRoundTripTime` back after `ActiveTotalThroughput`, and restore
`InterChunkLatency` before `DecodeDuration`. Verified by diffing the extracted
macro order against the extracted `CATALOG` row order: identical.

## Commits and bundles

| sha | subject | bundle |
|---|---|---|
| `f53eb2fee3` | fix(metrics): restore dense metric tag declaration order | `bundles/f53eb2fee3-restore-dense-tag-order.bundle` |

Bundle tip verified with `git bundle list-heads`:
`f53eb2fee3689f53bed4ffd74cbd35ea5e2853bb refs/heads/ajc/native-plugin-metrics-regression`.

## Test counts

`RUSTFLAGS="--cfg tokio_unstable" cargo test -p aiperf-runtime --lib`

- before (at `eadd5c665f`): **1896 passed, 18 failed**, 7 ignored
- after (at `f53eb2fee3`): **1914 passed, 0 failed**, 7 ignored

The 18 observed failures were exactly the set named in the task brief. The "15
pre-existing failures at f247b0102d" did not reproduce on this workstation.

**CORRECTION (Graham review I1).** The claim originally made here — "the suite is
fully green after the fix" — was false. It rested on `--lib` alone; the mandated
`--features engine` gate was never run. Corrected claim: **green on `--lib`; the
`--features engine` gate exits 101 with 6 failures, all attributed to base debt,
missing untracked worktree fixtures, or load flakiness.** The engine gate was
subsequently run against a base control at `eadd5c665f`, which fails 25 tests; the
6 failures at HEAD are a strict subset, so the fix removes 19 failures and
introduces zero. Full run records and per-test attribution:
`metrics-regression-verification.md`.

## Gates

| gate | exit | note |
|---|---|---|
| `cargo test -p aiperf-runtime --lib` | 0 | 1914 passed, 0 failed |
| `cargo fmt --check` | 1 | pre-existing toolchain drift only; **0** diffs in `tag_id.rs` |
| `cargo clippy -p aiperf-runtime --lib -- -D warnings` | 101 | 219 pre-existing errors; **0** in `tag_id.rs` |

`git diff --name-only eadd5c665f..HEAD` reports a single file
(`rust/runtime/src/metrics_core/tag_id.rs`), and neither the fmt nor the clippy
output mentions it. Both gate failures are local rustc 1.88.0 vs authoritative
1.98.0 drift on untouched files, and were left alone per the task rules.
