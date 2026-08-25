# Commit 035 — `03c9c6ddc5`

## Upstream intent

Upstream removes the single-request client lock, routes concurrent grade
responses by request id, batches already-queued codegen work, and kills the
worker process group even when its leader has exited. It adds unit coverage for
batch construction, response demultiplexing, concurrent fault/cancellation
handling, stale replies, and close races, plus one real LiveCodeBench concurrent
component test.

## Native applicability

The native product already sends an ordered `EvaluatorGradeBatch` through the
Rust-launched `aiperf.accuracy.worker`; commit #1 owns the Python LiveCodeBench
delegation and batch construction. The remaining native gap was in the Rust
stdio supervisor: its caller wrote one request and immediately read one reply,
so it could neither correlate out-of-order replies nor unblock multiple pending
requests after a reader fault. Cleanup also waited only for the worker leader.

The native port installs one response reader, an id-keyed pending table, and
serialized frame writes. Session teardown signals and verifies disappearance of
the complete Unix process group after leader wait. No upstream Python-only file
is part of the native implementation commit.

## Upstream-to-native test map

| Upstream behavior | Native evidence |
| --- | --- |
| Concurrent client calls complete and replies are correlated by id rather than position (`TestConcurrency`) | `accuracy_core::worker::tests::grade_batch_demuxes_out_of_order_responses` issues two simultaneous requests through the supervisor transport, receives reversed replies from a real subprocess, and binds each result to its request id. |
| Reader EOF or protocol failure faults pending and later work (`TestConcurrency::test_fault_cancels_all_pending_futures` and malformed-reader cases) | `accuracy_core::worker::tests::worker_exit_is_infrastructure_error` exercises reader EOF, while `reader_fault_before_request_rejects_the_next_request` proves an idle reader failure is retained rather than leaving the next request parked. |
| Cleanup kills the group even if the worker leader is already exiting (`_kill` coverage) | `shutdown_reaps_worker_process_group_descendants`, `drop_signals_worker_process_group_descendants`, and `runtime/tests/accuracy_worker_native_path.rs::native_worker_grades_batch_and_reaps_descendants` spawn a real descendant and prove both graceful and drop cleanup signal the owned process group. |
| Worker-side queued-request batching and per-problem metric reconstruction (`TestHandleBatch`, `TestRunWorkerLoopBatch`, and the real LCB component test) | Python-worker-specific and already inside the canonical evaluator delegated by commit #1. `native_worker_grades_batch_and_reaps_descendants` sends an actual two-item batch through the public Rust evaluator seam; Rust does not duplicate Lighteval metric reconstruction. |
| Cancellation, timeout, stale-id, and spawn/close races in `CodegenGradingWorker` | Python-client-specific restart/deadline policy, not a second Rust implementation target. Rust's applicable lifecycle boundary is covered by typed reader failure, `rejected_shutdown_faults_the_session`, drop-time group signalling, and real subprocess integration; Python tests are not accepted as native integration parity. |

## Verification and review

Native implementation commits are `b0fe2a85d5` and `e1dd5d49f1`; the exact
two-parent ancestry merge is `7cd1a5bf29`, whose second parent is the full
upstream id and whose tree equals its first parent's tree. Graham's first pass
found three Important lifecycle defects: idle reader faults were not retained,
drop killed only the leader, and a rejected shutdown acknowledgement left a
reusable session. Commit `429050fbf0` fixes all three. A full corrected-range
re-review found zero Critical or Important findings: **GRAHAM APPROVED**.

Verification after the fixes:

- `cargo test -p aiperf-runtime --features engine --lib accuracy_core::worker`:
  12 passed, 0 failed.
- `cargo test -p aiperf-runtime --features engine --test accuracy_worker_native_path`:
  1 passed, 0 failed.
- `cargo fmt --all -- --check`, `tools/check_agent_files_sync.py`,
  `tools/check_docs_current.py`, and exact-range whitespace checks: passed.
