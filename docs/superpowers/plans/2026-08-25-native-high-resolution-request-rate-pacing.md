# Native High-Resolution Request-Rate Pacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve sub-millisecond local request-rate schedules through ordinary wakeup jitter while bounding catch-up bursts, using the native high-resolution clock and producing exact-count real-clock evidence.

**Architecture:** Keep `RealClock` as the only timing authority. Add a pure bounded-reanchor policy and a construction-time parser for the upstream catch-up environment value; apply them only to the local/sharded renewal loop, because global dispatch already retains dense shared slots. Verify policy under `SimClock`, then characterize real high-rate delivery with an in-process dispatcher.

**Tech Stack:** Rust 2024, Tokio current-thread runtime, Linux timerfd/AsyncFd, native `Clock`/`SimClock`, `/usr/bin/sccache`.

**Spec:** `docs/specs/2026-08-25-native-high-resolution-request-rate-pacing.md`

## Global Constraints

- Base implementation on merge `86a93aaec1`, whose parents must remain exactly the prior campaign head and `21f8ad7b3e621285a1682b336df16607e7d3bb9f`.
- Never cherry-pick the upstream commit and never import cumulative TraceLab #44 changes.
- Route all scheduling time through `Clock`; do not call `Instant::now`, `SystemTime::now`, or Tokio timers in the request-rate path.
- Do not add a pacer thread, `Arc<Mutex<_>>`, channel, per-tick environment lookup, log, heap allocation, or redundant clock handle.
- Preserve continuation priority, interval draw timing, sampler advancement, rate-ramp behavior, stop/cancel behavior, global dense-slot/corpus ordering, and `SimClock` determinism.
- Treat `AIPERF_TIMING_MAX_CATCHUP_SECONDS` as construction policy: default `0.01`, finite inclusive range `0..=10`, rounded integer nanoseconds, informative failure.
- Use `RUSTC_WRAPPER=/usr/bin/sccache` and `CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-045-target` for every Cargo command.
- Keep verification output pristine. Do not fix unrelated Python hook, CLI-doc generation, ergonomics-baseline, or concurrent campaign churn.

---

### Task 1: Prove bounded late-slot policy and environment validation

**Files:**
- Modify: `rust/runtime/src/timing/arrival.rs`
- Modify: `rust/runtime/src/request_rate.rs`

**Interfaces:**
- Produces failing tests for pure `bounded_reanchor_target(target_ns, now_ns, max_catchup_ns)` behavior.
- Produces failing parser tests for `parse_max_catchup_seconds(Option<&str>) -> Result<i64>`.

- [ ] Add literal RED cases for below-window retention, exact boundary, beyond-window re-anchor, zero-window behavior, and saturated subtraction.
- [ ] Add literal RED cases for omission/default, `0`, `10`, fractional seconds, non-numeric, `NaN`/infinity, negative, and above-maximum input; consolidate equivalent invalid cases in a table.
- [ ] Run only the named library tests and retain the expected unresolved-symbol/behavior failure as RED evidence.
- [ ] Commit tests separately only if the repository permits a compiling RED commit; otherwise retain RED output in the task report and commit with Task 2.

### Task 2: Apply the minimal native pacing policy

**Files:**
- Modify: `rust/runtime/src/timing/arrival.rs`
- Modify: `rust/runtime/src/request_rate.rs`
- Modify: `rust/runtime/tests/request_rate_sim.rs`

**Interfaces:**
- Consumes `AIPERF_TIMING_MAX_CATCHUP_SECONDS` once at workload construction.
- Produces one private integer-nanosecond policy field and bounded local target selection without changing `GlobalRateGate`.

- [ ] Implement the pure saturated re-anchor helper and parser with default/range constants.
- [ ] Capture the environment once in `RequestRateWorkload::with_components`; propagate a descriptive constructor error.
- [ ] Replace unconditional local re-anchor with the bounded helper while retaining the single `now_ns` sample and existing yield path.
- [ ] Add an injected-policy constructor seam only if deterministic workload integration requires it; keep it crate-private and allocation-free.
- [ ] Add `SimClock` workload assertions showing small lateness catches up on the original grid and large lateness re-anchors without a burst storm.
- [ ] Run the focused RED tests to GREEN, then `cargo test -p aiperf-runtime --features engine --lib request_rate` and the `request_rate_sim` integration target.
- [ ] Commit production code and its behavioral tests together.

### Task 3: Prove real high-rate delivery and synchronize current-truth docs

**Files:**
- Create: `rust/runtime/tests/request_rate_real.rs`
- Modify: `docs/specs/2026-08-25-native-high-resolution-request-rate-pacing.md`
- Modify: `docs/specs/scheduling.md`
- Modify: `docs/specs/README.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `.github/copilot-instructions.md`
- Modify: `.cursor/rules/python.mdc`
- Modify: `llms.txt`

**Interfaces:**
- Consumes the real `RequestRateWorkload` and `RealClock` with an immediate in-process dispatcher.
- Produces exact count, elapsed time, achieved-rate evidence and synchronized clock/scheduling documentation.

- [ ] Write the integration test first with a request count and constant sub-millisecond interval that fails under per-tick re-anchor/timer quantization but completes quickly in normal CI.
- [ ] Assert the exact requested/completed count and broad elapsed/rate bounds; never assert individual microsecond wakeup latency.
- [ ] Run the integration in debug and release profiles; retain exact count, elapsed, and achieved-rate output as the benchmark receipt.
- [ ] Update current-truth scheduling and architecture prose, including the unchanged global dense-slot boundary.
- [ ] Synchronize all four agent instruction bodies, update `llms.txt`, and run both docs guards.
- [ ] Commit integration evidence and synchronized documentation.

### Task 4: Review, repair, and close tracker #45

**Files:**
- Modify: `docs/origin-main-findings/commit-045-21f8ad7b3e.md`
- Modify: `docs/porting-origin-main-campaign.md`
- Create: `.superpowers/sdd/2026-08-25-native-high-resolution-request-rate-pacing/graham-review.md`
- Create as needed: `.superpowers/sdd/2026-08-25-native-high-resolution-request-rate-pacing/graham-rereview.md`

**Interfaces:**
- Consumes the full first-parent range from `f423b618da` through branch tip plus exact upstream ancestry.
- Produces zero unresolved Critical/Important whole-branch or Graham findings and auditable closure evidence.

- [ ] Run task-scoped spec/quality review after each implementation task and repair every blocking finding through the original implementer.
- [ ] Run an independent whole-branch review, then the full Graham hot-path review for errors, allocation, descriptors, cancellation, clock discipline, concurrency, logging, names, comments, tests, and scope.
- [ ] Repair every Critical/Important finding and obtain explicit re-review approval.
- [ ] Run fresh format, focused integration, runtime library with/without `engine`, Clippy, docs guards, range whitespace, ancestry, status, and scope-diff commands with the mandated sccache/target directory.
- [ ] Update the finding and campaign row with exact commits, test counts, benchmark receipt, ancestry proof, and final `GRAHAM APPROVED` verdict.
- [ ] Commit closure artifacts separately and leave the isolated worktree clean except for the requested untracked `.venv` link.
