# Native LCB Codegen Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore concurrent, request-correlated native evaluator supervision and reap every evaluator-owned Unix descendant.

**Architecture:** Keep the existing `AccuracyEvaluator` protocol and canonical Python grading ownership. Split Rust stdio handling into serialized frame writes plus one response reader backed by an id-keyed pending table; retain process ownership in one session object and reap its whole Unix process group at teardown.

**Tech Stack:** Rust 2024, Tokio process/I/O/synchronization primitives, serde JSONL, libc Unix process groups, Cargo integration tests.

**Spec:** `docs/specs/accuracy.md`

## Global Constraints

- Do not commit the upstream Python-only files or the worktree-local `.venv`.
- The ancestry commit must name `03c9c6ddc5e6227782e53ded177f1227d332af48` as its exact second parent without importing commit #34's tree behavior.
- Preserve the public JSONL request/response wire and canonical Python scoring ownership.
- Use `/mnt/4tb/aiperf-origin-port-035-target` with `RUSTC_WRAPPER=sccache` for Rust verification.

---

### Task 1: Correlate concurrent evaluator responses

**Files:**
- Modify: `rust/runtime/src/accuracy_core/worker.rs`
- Test: `rust/runtime/src/accuracy_core/worker.rs`

**Interfaces:**
- Consumes: `WorkerRequest::id()` and strict `WorkerResponse` JSONL frames.
- Produces: `dispatch_request<T>(stdin, pending, request) -> Result<T, EvaluatorWorkerError>` and one reader resolving `BTreeMap<u64, oneshot::Sender<_>>` entries.

- [x] Add `grade_batch_demuxes_out_of_order_responses`, whose fixture emits two replies in reverse id order and whose literal assertions bind each problem and confidence to the original request.
- [x] Run the focused test against the sequential reader and observe that two in-flight requests cannot both complete.
- [x] Add a dedicated response reader, id-keyed pending table, serialized JSONL write section, and fail-all reader-fault handling.
- [x] Run `cargo test -p aiperf-runtime --lib accuracy_core::worker` and observe 9 passing tests.
- [x] Commit the native supervisor implementation as `b0fe2a85d5`.

### Task 2: Reap the evaluator process group

**Files:**
- Modify: `rust/runtime/src/accuracy_core/worker.rs`
- Test: `rust/runtime/src/accuracy_core/worker.rs`

**Interfaces:**
- Consumes: the evaluator leader pid after `setsid()`.
- Produces: bounded `reap_session(session, require_success)` cleanup that waits for the leader, signals `-pgid`, and verifies group absence.

- [x] Add `shutdown_reaps_worker_process_group_descendants`, which records a real sleeping descendant pid and fails if that process survives shutdown.
- [x] Start each Unix worker session through `pre_exec(libc::setsid)` and retain its process-group id.
- [x] Make graceful, fault, and forced teardown signal the group regardless of leader status and wait within `PROCESS_GROUP_REAP_TIMEOUT`.
- [x] Run the focused worker suite and observe the descendant-reap regression pass.

### Task 3: Prove the engine-feature native path

**Files:**
- Create: `rust/runtime/tests/accuracy_worker_native_path.rs`
- Modify: `rust/runtime/src/endpoints/mod.rs`
- Modify: `rust/runtime/src/engine/application.rs`
- Modify: `rust/runtime/src/engine/graph_execution.rs`

**Interfaces:**
- Consumes: public `PythonEvaluator::spawn`, the evaluator protocol, and the completed #32/#33 cache-bust/endpoint policy fields.
- Produces: a real Rust-launched subprocess test covering public two-item batch grading and descendant reap under `--features engine`; reversed-reply demultiplexing stays in the real-subprocess unit test where the private transport can be driven concurrently without adding a test-only public API.

- [x] Add the integration fixture module and assertions for two public batch items plus descendant disappearance.
- [x] Run the engine-focused build, repair the #33 endpoint policy path through a `pub(crate)` re-export, and initialize #32's `cache_bust_target` in the engine test fixture.
- [x] Run `cargo test -p aiperf-runtime --features engine --lib accuracy_core::worker` and observe 9 passing tests.
- [x] Run `cargo test -p aiperf-runtime --features engine --test accuracy_worker_native_path` and observe 1 passing test.
- [x] Commit the Rust-only integration slice as `e1dd5d49f1`.

### Task 4: Record exact ancestry and close review

**Files:**
- Modify: `docs/porting-origin-main-campaign.md`
- Modify: `docs/origin-main-findings/commit-035-03c9c6ddc5.md`
- Modify: `docs/specs/accuracy.md`
- Modify: `llms.txt`

**Interfaces:**
- Consumes: exact upstream commit `03c9c6ddc5e6227782e53ded177f1227d332af48` and the code range `df4237e7ce..HEAD`.
- Produces: a two-parent merge, upstream-to-native test mapping, verification receipt, and zero unresolved Graham findings.

- [x] Commit the spec, plan, and upstream semantic test map.
- [x] Create an `ours`-tree merge whose exact second parent is `03c9c6ddc5e6227782e53ded177f1227d332af48`; verify the merge changes no tree paths.
- [x] Perform two Graham passes over every changed Rust hunk, fix every validated Critical/Important finding, and re-review the corrected range.
- [x] Re-run the focused engine unit and native integration commands, formatting, docs checks, and exact-range diff checks.
- [x] Record the concrete commit ids, counts, and final review verdict in the campaign ledger.
