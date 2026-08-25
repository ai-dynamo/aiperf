# Empty Raw Messages Delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve explicit empty `raw_messages` deltas as no-op message contributions in native Rust endpoints.

**Architecture:** Keep `EndpointTurn.raw_messages` as `Option<Vec<Value>>`, but retain `Some(empty)` when the dataset contained an explicit raw-message segment. Endpoint rendering and lowerers branch on option presence: `Some` splices zero or more raw items, while `None` synthesizes one structured message. Reset context remains applied before the delta is appended.

**Tech Stack:** Rust 2024, `serde_json::Value`, native `aiperf-runtime` unit tests.

**Spec:** `docs/origin-main-findings/commit-020-446d2cd4b3.md`

## Global Constraints

- Preserve `None` versus `Some(empty)` at serialization boundaries.
- Do not emit `content: []` for synthetic messages.
- Keep endpoint lowerers transport-neutral and avoid unrelated refactors.

---

### Task 1: Add red native endpoint regressions

**Files:**
- Modify: `rust/runtime/tests/endpoints_endpoints.rs`
- Modify: `rust/runtime/tests/endpoints_anthropic_messages.rs`

- [x] **Step 1: Write failing tests** for Chat and Responses empty raw-message no-op. Existing endpoint coverage retains `None` structured synthesis; reset-context behavior is unchanged by this port.
- [x] **Step 2: Run the focused runtime endpoint tests** and confirm the empty raw-message tests fail because the current implementation synthesized a message.

### Task 2: Preserve explicit raw-message presence during resolution

**Files:**
- Modify: `rust/runtime/src/dataset/request.rs`

- [x] **Step 1: Track whether the resolved turn had an explicit raw-message handle.**
- [x] **Step 2: Retain `Some(empty)` instead of collapsing it to `None`.**
- [x] **Step 3: Run the focused endpoint tests and confirm the new resolution contract reaches formatters.**

### Task 3: Make native message lowerers honor empty deltas

**Files:**
- Modify: `rust/runtime/src/endpoints/implementation.rs`

- [x] **Step 1: Change Chat, Messages, and Responses rendering branches to use option presence rather than non-empty length.**
- [x] **Step 2: Keep synthetic rendering for `None` only and preserve reset-context behavior.**
- [x] **Step 3: Run focused runtime tests, then the applicable native endpoint regression suite.**

### Task 4: Verify and review

**Files:**
- Review: all changed Rust files and this plan/spec.

- [x] **Step 1: Run formatting, focused tests, and the relevant native endpoint regression set.**
- [x] **Step 2: Perform Graham-style review for ownership, allocations, error handling, and minimal diff.** No findings: the change is option-presence branching plus one resolution presence bit, with no new synchronization, hot-path allocation, or unchecked production error.
- [x] **Step 3: Commit implementation and closure with exact upstream merge ancestry.**
