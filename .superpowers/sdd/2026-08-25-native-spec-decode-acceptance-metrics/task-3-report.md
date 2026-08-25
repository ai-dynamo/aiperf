# Task 3 report — console and JSON artifact projection

## Scope

Projected Task 2's typed speculative-decoding report group and exact pooled
histogram into the native console and GenAI-Perf v1 JSON, and copied Task 1's
canonical request value into retained processed JSONL rows. CSV, Parquet, raw
records, outputs.json, and the existing SGLang server-metrics table are
unchanged.

## TDD receipt

Initial authoritative RED:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime spec_decode --lib --features engine -- --nocapture
```

Result: 24 passed, 5 failed, 2300 filtered out. The five failures were exactly
the missing worked-example console title/histogram, capped console histogram,
ordinary v1 JSON pool, widened-count v1 JSON pool, and processed JSONL canonical
object. The three absence regressions already passed.

Final authoritative GREEN used the identical command: 30 passed, 0 failed,
2300 filtered out, with four unrelated baseline warnings. A self-review micro
cycle first reproduced that an authored zero-count high key did not take the
cap branch, then verified the one-line cap-by-key fix.

## Adjacent regression evidence

- `cargo test -p aiperf-runtime export:: --lib --features engine`: 120 passed,
  0 failed. This includes every existing v1 JSON/CSV golden, console golden,
  and SGLang speculative server-metrics test.
- `cargo test -p aiperf-runtime engine::records::tests --lib --features engine`:
  10 passed, 0 failed, covering JSONL, CSV, Parquet, raw records, and outputs.
- `cargo fmt --all -- --check` and `git diff --check` passed.

## Implementation notes

- `MetricConsoleGroup::SpecDecode` renders between Reasoning and Default with
  the exact `NVIDIA AIPerf: Spec Decode` title. The histogram fills interior
  gaps, uses ties-to-even integer percentages, and folds every authored key at
  least eight into `>=8` without changing the SGLang table.
- The v1 JSON root uses a small typed serializer wrapper because
  `serde_json::Value` cannot hold `u128` past `u64::MAX`; direct serialization
  preserves exact widened counts. The field is omitted when absent or empty,
  and CSV remains scalar-only.
- `RecordRow` clones the optional canonical DTO only at retained processed-row
  construction. The existing catalog projection supplies all six record
  metrics; absent stats add neither the object nor speculative metric keys.

Independent Task 3 review APPROVED exact commit
`89dbbb91acc76fa786f342e662af779b40a2add5`; see
`task-3-independent-review.md`. The fresh authoritative, exporter, and records
suites passed 30/30, 121/121, and 10/10 respectively, and no findings remain.
Task 4 may begin.
