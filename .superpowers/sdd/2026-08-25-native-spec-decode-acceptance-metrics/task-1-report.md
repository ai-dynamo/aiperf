# Task 1 report — canonical wire capture and observer propagation

## Scope

Implemented the transport-neutral `ObservedSpecDecodeAcceptance` record, strict
vLLM normalization, single-choice HTTP/SSE capture, observer fan-out, and owned
`RecordIngest` propagation. The streamed-chat fast path retains a finish-only
stats chunk without forcing the common token-chunk path through generic JSON.

## TDD receipt

Initial RED:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime spec_decode --lib
```

Compilation failed on the intended missing feature surface: the canonical DTO
and parser module, `RequestObserver::on_spec_decode_acceptance`, the
`RecordIngest` field, and the typed `ChatChoice` field did not exist.

The dedicated tee test was also observed RED with both delegate recordings
empty before the callback was forwarded. After the forwarding implementation,
the same test passed once per delegate.

Authoritative recovered GREEN (run by the controller in the same worktree and
isolated target):

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime --features engine --lib spec_decode -- --nocapture
```

Result: 6 passed, 0 failed; only four unchanged baseline warnings.

An additional non-engine focused run passed 5 feature tests (canonical worked
example, malformed aggregate, multi-choice suppression, observer ownership,
and tee fan-out). The finish-only typed-chunk regression is compiled as part of
the runtime test target and is included in final full-suite verification.

## Implementation notes

- Stats are captured only for `chat` and `completions`, only from a sole choice,
  and the last non-empty object wins.
- Normalization runs once after usage reconciliation, so the canonical record
  receives the reconciled completion-token count.
- Malformed objects emit one structured warning and otherwise remain absent.
- Common absent requests add only one optional boxed pending field; no lock,
  task, channel, dependency, or unconditional generic-JSON parse was added.
- The local `.venv` symlink is an untracked worktree aid and is deliberately
  excluded from the commit.

## Self-review

The diff is limited to the new wire/observer/ingest seam and one required direct
`RecordIngest` literal. It preserves endpoint parse behavior, terminal status,
usage precedence, and the existing SGLang server-metrics path. Independent
review remains the gate before Task 2.

## Independent-review fix round

The first independent review rejected Task 1 on one important invariant and one
minor coverage gap. A focused RED run of the authoritative command executed
eight tests: the new real endpoint-dispatch test passed, while the new range
test failed because `draft_acceptance_rate` accepted `-0.01` and `1.01`.

The adapter now rejects every finite value outside `0.0..=1.0`. The dispatch
test launches an SSE endpoint with content, a finish-only stats chunk, a later
usage-only chunk, and `[DONE]`; it proves the fast path produces a completed
dispatch and a terminal `RecordIngest` containing the canonical stats with the
reconciled completion-token count.

GREEN:

```text
RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 \
  cargo test -p aiperf-runtime --features engine --lib spec_decode -- --nocapture
```

Result: 8 passed, 0 failed, 2300 filtered out; the same four unchanged baseline
warnings remained. Scoped re-review is pending.
