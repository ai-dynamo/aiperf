# Task 4 report — deterministic mock and real-profile E2E

## Scope

Added one false-by-default mock-server switch that emits the canonical vLLM
speculative-decoding acceptance object on chat responses. Non-streaming places
it on the sole choice. Streaming emits a dedicated finish-only typed chunk
before the usage-only chunk, exercising the Task 1 fast path without content or
usage on the stats-bearing frame.

Added native Rust E2E coverage that launches the in-process Rust mock and the
real `aiperf profile` binary. The present case asserts all eleven summary
metrics, the exact pooled histogram, the dedicated console section and line,
and both processed JSONL records. The default case asserts that all of those
summary, console, pooled, canonical-record, and per-record metric surfaces are
absent. The present case also checks representative canonical CSV headers, and
the absent case suppresses every speculative-decoding CSV row.

## TDD and debugging receipt

The handler and E2E tests were authored first. Their initial RED failed on the
absent mock config field and response fixture. After the minimal mock fixture
was implemented, focused handler coverage passed 2/2.

The first real-profile execution carried the exact summary values and both
processed-record objects but did not render the dedicated console histogram.
Tracing the product path showed that Task 3's unit test injected catalog
metadata while `Export::build` loads the embedded
`runtime/resources/metric_metadata.json`. A focused regression against that
default-profile construction failed 0/1 with the intended missing-metadata
assertion:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime \
  default_profile_metadata_groups_canonical_spec_decode_metrics -- --nocapture
```

Adding only the eleven canonical entries to the embedded resource made the
identical command pass 1/1. The six visible metrics use the `spec_decode` group
and exact display orders; the five scalar-only metrics remain in `none`.

The first independent review then reproduced a public v1 CSV compatibility
defect: the embedded resource's `header_map` and `scalar_tags` did not yet carry
the same canonical identities. The strict fix-round regression extended the
same `Export::build` test across console, v1 headers, and scalar classification.
It failed 0/1 exactly on the missing `spec_decode_acceptance_length` header.
Adding all eleven canonical header entries and the five derived/aggregate
scalar identities made the identical library-only command pass 1/1. The real
E2E now asserts canonical CSV names and absence as well.

## Final focused GREEN

- `cargo test -p aiperf-mock-server spec_decode_acceptance -- --nocapture`:
  3 passed, 0 failed, including zero-output tool-call finish deferral.
- `cargo build -p aiperf-cli`: exit 0, rebuilding the native executable with
  the corrected embedded resource.
- `AIPERF_E2E_BIN=/mnt/4tb/aiperf-target-port-013/debug/aiperf cargo test -p
  aiperf-e2e-tests --test test_spec_decode_acceptance -- --nocapture`: 14
  passed, 0 failed, including the canonical-present and default-absent product
  cases with representative canonical CSV names. Cargo used
  `RUSTC_WRAPPER=sccache` and the Task 4 target directory.
- `cargo test -p aiperf-mock-server write_chat_response_matches_serde --
  --nocapture`: 1 passed, 0 failed, proving the hand-written default-disabled
  non-streaming response remains byte-equal to serde output.
- `cargo fmt --all -- --check`, `jq empty
  rust/runtime/resources/metric_metadata.json`, and `git diff --check`: exit 0.

The commands emitted only baseline workspace warnings. The initial independent
Task 4 review was NOT APPROVED at `f657dd91cb` for the incomplete embedded v1
metadata; see `task-4-independent-review.md`. Scoped re-review of the fix is
APPROVED at `0c57560d39` with no remaining findings; see
`task-4-rereview.md`. Final branch review may proceed.
