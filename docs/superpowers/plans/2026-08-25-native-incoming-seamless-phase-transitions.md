# Native incoming seamless phase transitions implementation plan

> **Author:** Sol
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:test-driven-development for every behavior slice and
> superpowers:verification-before-completion before each success claim.

**Goal:** Prove and preserve the public incoming `seamless` phase contract
across every native execution path, with exact upstream merge ancestry and full
Graham approval.

**Architecture:** Keep the existing two-level phase model: public
`PhaseSpec.seamless` is incoming, while internal `PhaseConfig.seamless` is the
current phase's outbound handoff. Preserve their existing adapter and add a
worker-local server-profiler ownership coordinator shared by every phase
sidecar in one run. Cellular readiness remains controller-owned but tracks each
overlapping phase separately and acquires the same first-start/last-stop
ownership contract.

**Tech stack:** Rust 2024, serde protocol v2, Tokio current-thread `LocalSet`,
native `Clock`/`SimClock`, Hyper/Axum integration tests, Cargo, sccache.

**Spec:**
`docs/specs/2026-08-25-native-incoming-seamless-phase-transitions.md`

## Global constraints

- Work only in `/mnt/4tb/aiperf-origin-port-050`, based exactly on
  `482f85924152caaee32e7cab6fae37ce15e64c3a`.
- Begin every missing behavior with an observed failing Rust assertion.
- Use `RUSTC_WRAPPER=sccache`, `SCCACHE_DIR=/mnt/4tb/sccache-port050`, and
  `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port050` for Cargo commands.
- Preserve the public protocol-v2 wire shape and the existing current-thread,
  worker-local timing architecture.
- Do not add clocks, timers, locks, channels, or background threads.
- The final target-specific merge must retain the reviewed native tree and name
  exact upstream `ade1f69eb13dfa0e87e49b2c027f6fe29c03d402` as second parent.
  Never cherry-pick or import the upstream Python tree.

---

### Task 1: Source audit, design, and exact target ancestry

**Files:**
- Create: `docs/origin-main-findings/commit-050-ade1f69eb1.md`
- Create: `docs/specs/2026-08-25-native-incoming-seamless-phase-transitions.md`
- Create: this Sol implementation plan
- Modify: `docs/specs/README.md`

**Interfaces:**
- Consumes the exact upstream diff and every native phase-construction path.
- Produces an auditable semantic map and exact two-parent target merge.

- [x] Inspect the upstream runner, orchestrator, and all five changed tests.
- [x] Inspect native protocol, lowering, scheduled, sharded, offline, graph,
  orchestrator, runner, sidecar, and existing test seams.
- [x] Record the correct timing behavior and the missing profiler ownership.
- [x] Commit the finding, specification, plan, and spec index.
- [x] Create and verify a target-specific two-parent tree-preserving merge.

### Task 2: TDD the authored transition direction

**Files:**
- Modify: `rust/runtime/src/engine/execute/dataset_build.rs`

**Interfaces:**
- Consumes authored `Vec<PhaseSpec>` values.
- Produces the exact outbound handoff vector passed to native `PhaseConfig`.

- [x] Add an inverse test proving a phase's own incoming flag cannot make it
  hand off to a non-seamless successor.
- [x] Add a workflow test proving only immediate predecessors of incoming
  seamless phases hand off.
- [x] Add a final-phase assertion proving authored seamless never creates an
  outbound transition past the end of the phase list.
- [x] Retain the existing helper because the mapping assertions are already green;
  existing helper and record that the RED phase was an assertion-tightening
  failure against the old incomplete test contract.
- [x] Run focused engine tests to GREEN and commit the slice.

### Task 3: Port applicable lifecycle tests and integration evidence

**Files:**
- Modify as needed: `rust/runtime/tests/timing_phase_orchestrator.rs`
- Modify as needed: `rust/runtime/tests/phase_runtime_online.rs`
- Modify as needed: `rust/runtime/tests/phase_runtime_sim.rs`

**Interfaces:**
- Consumes the lowered outbound `PhaseConfig` contract.
- Produces proof of overlap, non-overlap, final barrier, background failure,
  and sidecar/drain behavior.

- [x] Map every changed upstream test assertion to native evidence.
- [x] Run the real-HTTP overlap test and inspect phase start/end ordering.
- [x] Run simulated seamless and non-seamless transition tests.
- [x] Run detached predecessor-failure cancellation coverage.
- [x] Add a failing ownership test proving overlapping profiling phases emit one
  start and one last-owner stop.
- [x] Add native phase-sidecar return-drain and cellular overlap regressions.
- [ ] Commit any required integration coverage separately.

### Task 4: Full verification and Graham review

**Files:**
- Create: `.superpowers/sdd/2026-08-25-native-incoming-seamless-phase-transitions/graham-review.md`
- Create: `.superpowers/sdd/2026-08-25-native-incoming-seamless-phase-transitions/graham-rereview.md`
- Modify: `docs/origin-main-findings/commit-050-ade1f69eb1.md`
- Modify: `docs/porting-origin-main-campaign.md`

**Interfaces:**
- Produces zero unresolved Critical/Important findings, fresh test receipts,
  exact ancestry/tree proof, and tracker closure.

- [ ] Run formatting, focused tests, runtime library tests with and without
  `engine`, changed-scope Clippy, and range whitespace checks using sccache and
  the dedicated `/mnt/4tb` target.
- [ ] Perform a fresh full Graham review over the exact base-to-tip range.
- [ ] Repair every Critical/Important finding with a regression test first and
  repeat review until approved.
- [ ] Request a separate independent Graham review from the parent agent and
  obtain explicit approval before closure.
- [ ] Commit review receipts and closure evidence only after both approvals.
- [ ] Verify exact merge parents, first-parent tree equality, clean status, and
  absence of imported upstream Python changes.
