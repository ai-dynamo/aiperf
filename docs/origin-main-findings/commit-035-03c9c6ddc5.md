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
| Concurrent client calls complete and replies are correlated by id rather than position (`TestConcurrency`) | `accuracy_core::worker::tests::grade_batch_demuxes_out_of_order_responses` and `runtime/tests/accuracy_worker_native_path.rs::native_worker_demuxes_out_of_order_grades_and_reaps_descendants` issue two simultaneous requests and receive reversed replies. |
| Reader EOF or protocol failure faults pending work (`TestConcurrency::test_fault_cancels_all_pending_futures` and malformed-reader cases) | `accuracy_core::worker::tests::worker_exit_is_infrastructure_error` exercises the real reader EOF path; the reader drains every registered oneshot through `fault_pending_requests`. |
| Cleanup kills the group even if the worker leader is already exiting (`_kill` coverage) | `accuracy_core::worker::tests::shutdown_reaps_worker_process_group_descendants` and the native-path integration spawn a descendant, accept graceful leader shutdown, and assert that the descendant no longer exists. |
| Worker-side queued-request batching and per-problem metric reconstruction (`TestHandleBatch`, `TestRunWorkerLoopBatch`, and the real LCB component test) | Python-worker-specific and already inside the canonical evaluator delegated by commit #1. The Rust port preserves its ordered batch protocol and does not duplicate Lighteval metric reconstruction in Rust. |
| Cancellation, timeout, stale-id, and spawn/close races in `CodegenGradingWorker` | Python-client-specific lifecycle policy, not a second Rust implementation target. Rust's applicable boundary is typed reader failure and session reap; its native tests exercise those effects through an actual subprocess rather than accepting Python tests as integration parity. |

## Verification and review

The authoritative commands and final Graham verdict are recorded in the
campaign ledger when the exact ancestry merge and review are complete.
