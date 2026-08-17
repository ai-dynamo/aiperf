# Runtime Clippy Baseline Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Make `cargo clippy -p aiperf-runtime --all-targets --features engine -- -D warnings` pass without suppressions while preserving Harbor behavior.

**Architecture:** Group the recorded diagnostics by independent subsystem. Each batch has focused tests, the strict clippy command, a commit, and an independent Graham inspection before the next batch starts.

## Global Constraints

- No lint allowances or lint-policy changes.
- Preserve public contracts and Clock-driven behavior.
- Do not proceed to the next batch without Graham approval.

### Task 1: Record the complete baseline

**Files:** Create `.superpowers/sdd/2026-08-16-runtime-clippy-remediation-inventory.md`.

- [ ] Run `env -u RUSTC_WRAPPER cargo clippy -p aiperf-runtime --all-targets --features engine -- -D warnings 2>&1 | tee /tmp/aiperf-runtime-clippy.txt`.
- [ ] Record every `error` and source location grouped as graph/recording, execution, datasets, agent replay/AgentX, or engine tests.
- [ ] Commit the inventory and obtain Graham inspection.

### Task 2: Mechanical idioms and dead code

**Files:** Modify only inventory files with numeric grouping, `is_none_or`, collapsed `if`, slice signatures, redundant conversions, and dead-code removals.

- [ ] Add a behavior test before removing or narrowing a helper.
- [ ] Apply only behavior-preserving transformations, such as `gate.as_ref().is_none_or(|g| g.on_lane_terminal(&id))`.
- [ ] Run focused tests and strict clippy; commit; obtain Graham approval.

### Task 3: Async and locking findings

**Files:** Modify `rust/runtime/src/agentic_replay.rs` and exact inventory peers with await-held guards.

- [ ] Add an async resume regression.
- [ ] Extract the owned handoff from the guard before awaiting `execute_profiling_resume`.
- [ ] Run the agent replay tests and strict clippy; commit; obtain Graham approval.

### Task 4: Dataset and engine test construction

**Files:** Modify exact dataset and engine test files listed in the inventory.

- [ ] Replace default-then-field assignment with direct initializers, single-element loops with blocks, and `len() >= 1` with `!is_empty()`.
- [ ] Run affected dataset/engine tests and strict clippy; commit; obtain Graham approval.

### Task 5: Final Harbor acceptance

- [ ] Run runtime and engine tests, CLI eval tests, and all three P0 targets.
- [ ] Run strict clippy, `cargo fmt --check`, diff check, and both documentation guards.
- [ ] Run serial ignored Docker Harbor benchmark and lifecycle suites using a freshly built binary.
- [ ] Obtain a final full Graham review, update both ledgers, and mark the goal complete only after every command is green.
