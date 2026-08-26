# Native rejected peak-context diagnostics implementation plan

> **Author:** Sol
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development and
> superpowers:test-driven-development for every behavior slice.

**Goal:** Report the smallest actionable context cap after a native WEKA
(including TraceLab) or Dynamo recorded selection rejects every candidate.

**Architecture:** Keep selection ordering and decoding boundaries unchanged.
The AgentX filter stores its minimum in `SelectionStats`; Graph-IR WEKA and
Dynamo compute their corresponding minimum locally and share only error text.
TraceLab delegates to WEKA with its own source label.

**Tech Stack:** Rust 2024, serde JSON recorded graph compilers, Config v2,
native binary integration tests, sccache.

**Spec:** `docs/specs/2026-08-25-native-rejected-peak-diagnostics.md`

## Global Constraints

- Work in `/mnt/4tb/aiperf-origin-port-054` based on `cf09af50346db254beb5a7e8595b2e5fceeeeb39`.
- Begin every production behavior with a focused observed RED test.
- Use `RUSTC_WRAPPER=/usr/bin/sccache`, `SCCACHE_DIR=/mnt/4tb/sccache-port054`, and `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port054`.
- Do not import upstream Python files. The final target-only merge records exact upstream `bfe33151de75426710e51ca054823aa91342cebc` as second parent.

---

### Task 1: Shared AgentX selection minimum and error

**Files:**
- Modify: `rust/runtime/src/agentx/selection.rs`
- Modify: `rust/runtime/src/agentx/loader.rs`

**Interfaces:**
- Produces: `SelectionStats.smallest_observed: Option<i64>` and an empty-
  selection error containing source/caps/minimum.
- Consumes: existing `filter_then_cap` scan order and `trace_peak_context_length`.

- [x] Write a failing selection test that scans peaks `0, 12, 8` and expects
  `Some(0)`, then a failing loader test with rejected peaks `950, 104` that
  expects a `104` admission hint.
- [x] Run `cargo test -p aiperf-runtime agentx:: --lib` and record RED.
- [x] Add minimum accounting and cause-specific empty-selection formatting;
  do not change successful selection results.
- [x] Rerun the focused test command and record GREEN (106 passed, 1 ignored).
- [x] Commit the slice in `4022b433c9`.

### Task 2: Graph-IR WEKA and Dynamo selection diagnostics

**Files:**
- Modify: `rust/runtime/src/graph/recorded/weka/mod.rs`
- Modify: `rust/runtime/src/graph/recorded/dynamo/mod.rs`
- Modify: `rust/runtime/src/graph/recorded/tracelab.rs`
- Modify: `rust/runtime/src/graph/recorded/mod.rs` if a shared formatter is
  the smallest coherent boundary.

**Interfaces:**
- Produces: causal context-cap errors with smallest trace/tree peak.
- Consumes: the exact existing `peak_context` and `request_peak_context`
  semantics.

- [x] Write failing compiler/selector tests with two rejected candidates of
  unequal peak and assert the smaller suggested cap for each format.
- [x] Run the focused runtime test filters and record RED.
- [x] Add local minimum tracking only along the active context-filter branch;
  preserve WEKA no-decode-after-root-cap, TraceLab source identity, and Dynamo
  whole-tree grouping.
- [x] Rerun both focused suites and record GREEN (78 passed, 1 ignored).
- [x] Commit the slice in `4022b433c9`.

### Task 3: Public product regression, review, and ancestry receipt

**Files:**
- Modify/Create: focused native CLI or E2E test under `rust/cli/tests/` or
  `rust/e2e-tests/tests/`
- Modify: `docs/origin-main-findings/commit-054-bfe33151de.md`
- Modify: `docs/porting-origin-main-campaign.md`

**Interfaces:**
- Produces: a process-boundary assertion and reviewable closure receipt.

- [x] Write a failing public Config-v2/native-binary test that rejects every
  temporary WEKA trace under `max_context_length` and expects the exact
  smallest-cap tail.
- [x] Run the focused test with the built native binary and record RED.
- [x] Add only missing projection or error propagation discovered by the test.
- [x] Rerun the public test plus all Task 1/2 suites, `cargo fmt --check`, and
  changed-scope Clippy with sccache; record GREEN.
- [x] Commit implementation, perform a full Graham review, record the
  inherited broad-suite version-snapshot failure, update the campaign tracker
  after independent approval, and create/verify the two-parent target-only
  upstream merge.
