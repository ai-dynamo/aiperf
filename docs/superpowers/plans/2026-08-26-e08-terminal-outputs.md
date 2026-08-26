# E08 NativeGraph Terminal Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a validated NativeGraph `terminal_outputs` declaration execute and return only resolvable opaque output handles in its completion supplement.

**Architecture:** The graph executor already produces a completed channel snapshot and the live driver already maps declared names from `TraceStageResult.output_handles` to `TraceTerminalSupplement`. Add a cold-path execution-owned freezer that extends the immutable worker catalog with canonical raw segments for the selected completed channels, then give those handles to the existing driver contract. Remove the contradictory driver-open refusal while retaining its exact missing-output boundary.

**Tech Stack:** Rust 2024, Tokio current-thread tests, `serde_json`, BLAKE3-backed `SegmentPool`/`InMemorySegmentStore`, NativeGraph engine feature.

**Spec:** `docs/specs/2026-08-26-e08-terminal-outputs.md`

## Global Constraints

- Preserve the validated lowering contract; do not add author-controlled terminal channel selection after lowering.
- Freeze only declared terminal values as `Payload::Raw` canonical JSON on the cold terminal path.
- Keep raw values out of `TraceTerminalSupplement` and preserve the current `BTreeMap<String, Handle>` wire shape.
- Keep `terminal_outputs: []` byte-for-byte compatible: omitted from serialized supplements.
- Do not add locks, shared mutation, or per-token work to graph request paths.
- Use a unique `CARGO_TARGET_DIR` under `/mnt/4tb` and `CARGO_BUILD_JOBS=1` for every Cargo witness.
- Do not update the tracker until independent Graham approval and the implementation commit exist.

---

## File structure

- Modify: `rust/runtime/src/graph/driver.rs` — expose the selected declared terminal-channel names through the staged-driver interface without leaking driver internals.
- Modify: `rust/runtime/src/engine/graph_execution.rs` — freeze selected completed channels, retain their frozen catalog locally through terminal-result handling, and populate `TraceStageResult.output_handles`.
- Modify: `rust/runtime/src/eval/native_graph/live_driver.rs` — admit non-empty validated declarations and retain exact missing-output selection behavior.
- Modify: `rust/runtime/tests/native_graph_driver.rs` — direct live-driver declared-output, missing-output, and exact-selection regressions.
- Modify: `rust/runtime/tests/native_graph_live_paths.rs` — dynamic terminal-output deferral and final exact-selection regression.
- Modify: `rust/runtime/src/engine/graph_execution.rs` test module — end-to-end staged-executor freezing and resolution regression, replacing the current refusal-only test.

### Task 1: Pin the driver contract with a RED test

**Files:**
- Modify: `rust/runtime/tests/native_graph_driver.rs:registered_live_driver_progresses_through_a_bounded_graph_stage`
- Modify: `rust/runtime/tests/native_graph_driver.rs:live_program`

**Interfaces:**
- Consumes: `TraceStageResult { output_handles: BTreeMap<String, Handle>, .. }`.
- Produces: a regression proving `TraceTerminalSupplement.terminal_outputs` contains exactly the declared name and its provided opaque handle.

- [ ] **Step 1: Add a declared-output fixture and failing success regression**

  Change the fixture's JSON to `"terminal_outputs": ["output"]`. In the
  completed observation, supply `BTreeMap::from([("output".into(),
  Handle::new(7)), ("undeclared".into(), Handle::new(8))])`. Assert the
  completion supplement equals `BTreeMap::from([("output".into(),
  Handle::new(7))])`.

- [ ] **Step 2: Run the direct RED witness**

  Run:

  ```bash
  source .venv/bin/activate
  cd rust
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E08-red \
    cargo test -p aiperf-runtime --features engine --test native_graph_driver \
    registered_live_driver_progresses_through_a_bounded_graph_stage -- --exact
  ```

  Expected: FAIL at driver open with `requires frozen terminal handles before
  stage execution`.

- [ ] **Step 3: Add the missing-output regression**

  Reuse the declared fixture, observe a completed stage with
  `output_handles: BTreeMap::new()`, and assert the error includes
  `did not receive declared terminal output "output"`. Assert the driver does
  not yield a completion supplement afterward.

- [ ] **Step 4: Commit the test-only RED state**

  ```bash
  git add rust/runtime/tests/native_graph_driver.rs
  git commit -m "test: pin native graph terminal output contract"
  ```

### Task 2: Add the cold-path frozen output seam

**Files:**
- Modify: `rust/runtime/src/graph/driver.rs:TraceProgramDriver`
- Modify: `rust/runtime/src/engine/graph_execution.rs:GraphWorkerBackend::execute_staged_driver`
- Test: `rust/runtime/src/engine/graph_execution.rs:tests`

**Interfaces:**
- Consumes: the driver's validated ordered terminal-channel selection,
  `TraceResult.channels: BTreeMap<String, ChanVal>`, and the worker's
  `Arc<dyn SegmentStore>`.
- Produces: `TraceStageResult.output_handles: BTreeMap<String, Handle>` and a
  private owned staged-terminal result whose frozen catalog resolves every returned handle.

- [ ] **Step 1: Add the engine RED test for freezing and resolution**

  Replace `lowered_terminal_contract_refuses_before_graph_executor_dispatch`
  with an end-to-end staged-driver test. Use a one-channel completed graph with
  `terminal_outputs: ["output"]`, capture the `TraceStageResult` received by a
  test driver, and assert it contains an `output` handle. Resolve that handle
  through the private staged-terminal result's frozen catalog, match `Payload::Raw { wire }`,
  and assert `wire == serde_json::to_vec(&completed_channel_value)`.

- [ ] **Step 2: Run the engine RED witness**

  Run:

  ```bash
  source .venv/bin/activate
  cd rust
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E08-red \
    cargo test -p aiperf-runtime --features engine --lib \
    graph_execution::tests::lowered_terminal_contract_executes_with_resolvable_output -- --exact
  ```

  Expected: FAIL because `open` rejects the declaration; after that guard is
  removed it must still fail until the executor provides a non-empty output map.

- [ ] **Step 3: Define the narrow driver selection API**

  Add a default `TraceProgramDriver::terminal_output_channels(&self) -> &[String]`
  returning `&[]`. Override it in `NativeGraphLiveTraceProgramDriver` to return
  its already validated `terminal_outputs`. Forward the method through driver
  decorators such as test-only `ObservedStockTraceDriver`. Do not give the
  executor access to driver-private control-flow DTOs.

- [ ] **Step 4: Implement the freezer**

  Add a private engine helper that:

  ```rust
  fn freeze_terminal_outputs(
      channels: &BTreeMap<String, ChanVal>,
      selected: &[String],
      base: Arc<dyn SegmentStore>,
  ) -> Result<FrozenTerminalOutputs, TraceError>
  ```

  `FrozenTerminalOutputs` contains the selected `handles` and an
  `Arc<dyn SegmentStore>` resolving them. The helper looks up every selected channel, skips absent and
  `ChanVal::Unset` entries, serializes each concrete value with
  `serde_json::to_vec`, thaws `base` only when at least one concrete value must
  be interned, calls `pool.intern_raw(None, bytes)`, and freezes the extended
  pool. With no concrete selected value it returns the unchanged `base` Arc, so
  empty declarations add no segment or arena allocation. Missing selections are deliberately omitted: the live driver must
  issue the existing typed `TraceDriverError` at static completion or final
  dynamic completion. Empty selection returns an empty map and no terminal
  store allocation.

- [ ] **Step 5: Retain the frozen catalog locally for the terminal result**

  Add a private `StagedTraceTerminal` containing the completed supplement and
  `Arc<dyn SegmentStore>`. Keep the current terminal store as a
  local variable in `execute_staged_driver`; initialize its base from
  `self.segments`, then thaw the prior terminal store on later dynamic stages so
  every previously issued handle remains resolvable. Return the private owned
  result and keep it alive in `execute_trace` through supplement emission. Do
  not add a `GraphWorkerBackend` map, lock, or shared mutation, and do not mutate
  `GraphBackendFactoryConfig.segments` or the request materializer's immutable
  catalog.

- [ ] **Step 6: Populate the stage result**

  Immediately after `execute_static_trace_result` succeeds, call the freezer
  with `driver.terminal_output_channels()`, retain/extend the local catalog, and
  assign its map to `TraceStageResult.output_handles` instead of
  `BTreeMap::new()`. In the live driver's dynamic cursor, retain the latest
  handle received for each declared name across stages. When dynamic readiness
  becomes empty, collect exactly all declarations from that retained map and
  return the existing missing-output `TraceDriverError` if any name is absent.

- [ ] **Step 7: Pin dynamic deferral and final selection**

  In `rust/runtime/tests/native_graph_live_paths.rs`, add
  `dynamic_terminal_output_waits_for_terminal_stage_and_retains_latest_handle`
  using the existing bounded dynamic branch fixture. Declare one output produced
  by the later selected stage; observe the first completed stage without that
  handle and assert progression continues, then observe the terminal stage with
  the declared handle plus an undeclared handle. Assert the final supplement
  contains exactly the declared/latest handle. Add the companion terminal-stage
  omission case and require the existing diagnostic naming the declaration.

- [ ] **Step 8: Commit the engine seam and regressions**

  ```bash
  git add rust/runtime/src/graph/driver.rs rust/runtime/src/engine/graph_execution.rs rust/runtime/tests/native_graph_live_paths.rs
  git commit -m "feat: freeze native graph terminal outputs"
  ```

### Task 3: Admit declarations and verify GREEN

**Files:**
- Modify: `rust/runtime/src/eval/native_graph/live_driver.rs:TraceProgramDriver::open`
- Test: `rust/runtime/tests/native_graph_driver.rs`
- Test: `rust/runtime/src/engine/graph_execution.rs:tests`

**Interfaces:**
- Consumes: `TraceStageResult.output_handles` produced by Task 2.
- Produces: an exact `TraceTerminalSupplement.terminal_outputs` map or the
  existing typed missing-output observation error.

- [ ] **Step 1: Remove only the contradictory open-time refusal**

  Delete the `if !self.terminal_outputs.is_empty()` block in
  `NativeGraphLiveTraceProgramDriver::open`. Keep provenance, stage-bound,
  static/dynamic graph validation, and `observe_stage`'s selected-map
  collection unchanged.

- [ ] **Step 2: Run direct driver GREEN tests**

  Run:

  ```bash
  source .venv/bin/activate
  cd rust
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E08 \
    cargo test -p aiperf-runtime --features engine --test native_graph_driver
  ```

  Expected: PASS, including declared-output exact-selection, missing-output,
  and legacy-empty-wire cases.

- [ ] **Step 3: Run staged-engine GREEN tests**

  Run:

  ```bash
  source .venv/bin/activate
  cd rust
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E08 \
    cargo test -p aiperf-runtime --features engine --lib graph_execution::tests
  ```

  Expected: PASS, including the resolvable frozen-output regression and staged
  cancellation/cleanup coverage.

- [ ] **Step 4: Run the relevant NativeGraph suite and mechanical gates**

  Run:

  ```bash
  source .venv/bin/activate
  cd rust
  CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/mnt/4tb/aiperf-smell-target-E08 \
    cargo test -p aiperf-runtime --features engine --test native_graph_driver --test native_graph_lowering
  cargo fmt --check
  git diff --check
  ```

  Expected: all focused tests pass; report any unrelated workspace formatting
  drift separately rather than reformatting outside E08.

- [ ] **Step 5: Commit the driver admission change**

  ```bash
  git add rust/runtime/src/eval/native_graph/live_driver.rs rust/runtime/tests/native_graph_driver.rs
  git commit -m "fix: execute declared native graph terminal outputs"
  ```

### Task 4: Independent completion gate

**Files:**
- Create: `.superpowers/sdd/2026-08-25-rust-matrix-remediation/task-e08-independent-graham-review.md`
- Modify: `docs/rust-code-smell-remediation-tracker.md`

**Interfaces:**
- Consumes: implementation commits, RED/Green command output, and the E08 spec.
- Produces: an independent Graham PASS receipt and tracker completion entry.

- [ ] **Step 1: Request an independent Graham review**

  Give a reviewer the implementation commit range and require review of the
  frozen-store lifetime, segment-handle resolution, cancellation cleanup,
  missing-output diagnostics, hot-path impact, and test realism. The author
  must not review their own work.

- [ ] **Step 2: Resolve every blocking finding with a new focused RED-to-GREEN cycle**

  For each blocker, add or strengthen a regression, demonstrate it fails on
  the immediately preceding implementation, make the smallest repair, rerun
  the affected suites, and commit the repair before requesting a fresh review.

- [ ] **Step 3: Record only an approved completion**

  After a PASS receipt and clean relevant tests, update E08 to `Complete` with
  implementation commit IDs, test evidence, and reviewer receipt; commit the
  tracker update separately.

## Plan self-review

- Spec coverage: Tasks 1–3 cover declaration admission, exact opaque selection,
  content-addressed freezing, resolution, missing outputs, and empty-wire
  compatibility. Task 4 enforces the campaign's independent-review and tracker
  gates.
- Placeholder scan: no `TBD`, `TODO`, or unspecified test actions remain.
- Type consistency: Task 2 defines the only new driver method and freezer
  signature before Task 3 consumes its `TraceStageResult` output map.
