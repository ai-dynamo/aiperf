# Native default session-affinity header implementation plan

> **Author:** Sol
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:test-driven-development for every behavior slice and
> superpowers:verification-before-completion before each success claim.

**Goal:** Make the native HTTP client send default additive
`X-Session-Affinity` headers derived from stable correlation IDs, with full
unit and native-binary evidence.

**Architecture:** Extend the existing pure HTTP header-composition seam with a
default policy bit held by `HttpTransport`.  Preserve the other derived-header
policies and observe production behavior through existing mock raw records.

**Tech stack:** Rust 2024, Hyper, Axum mock server, native E2E harness, cargo,
sccache.

**Spec:** `docs/specs/2026-08-25-native-session-affinity-header.md`

## Global constraints

- Work only in `/mnt/4tb/aiperf-origin-port-055`, based exactly on
  `cf09af50346db254beb5a7e8595b2e5fceeeeb39`.
- Begin each production behavior with an observed failing Rust test.
- Use `RUSTC_WRAPPER=/usr/bin/sccache`,
  `SCCACHE_DIR=/mnt/4tb/sccache-port055`, and
  `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port055`.
- Preserve the existing opt-in meaning of `X-Session-ID`, SGLang routing, and
  Dynamo session headers.
- The final provenance merge must retain the reviewed native tree and record
  exact upstream `dd3f09b0c34710470444bad17c9e7050c1cd694a` as second parent.
  Never cherry-pick or import its Python tree.

---

### Task 1: Default header composition

**Files:**

- Modify: `rust/runtime/src/transport/http/transport/headers.rs`
- Modify: `rust/runtime/src/transport/http/transport/http_transport.rs`

**Interfaces:**

- Produces: canonical default `X-Session-Affinity` in all HTTP transport send
  paths with a correlation ID.
- Consumes: `RequestConfig.correlation_id`, caller headers, and the existing
  immutable transport policy fields.

- [ ] Write failing pure-composer tests for default presence/equality,
  no-correlation omission, stale case-insensitive replacement, and
  independence from opt-in `X-Session-ID`.
- [ ] Run the focused runtime test and record RED.
- [ ] Add the default affinity parameter/policy and its final authoritative
  strip-then-set composition.
- [ ] Update every call site, run the focused suite GREEN, and commit the
  behavior slice.

### Task 2: Native binary E2E and documentation

**Files:**

- Modify: `rust/e2e-tests/tests/test_port_raw_parity.rs`
- Modify: `docs/benchmark-modes/trace-replay.md`
- Modify: `docs/environment-variables.md`

**Interfaces:**

- Produces: raw-record parity evidence for Python and native runs and current
  public guidance.
- Consumes: real native binary, Rust mock capture, raw artifact parser, and
  the pure transport policy from Task 1.

- [ ] Write a failing native/Python raw projection for default affinity,
  custom correlation-header naming, and separately opted-in `X-Session-ID`.
- [ ] Run it RED against the old native binary behavior.
- [ ] Update the projection/assertions and public docs after Task 1 is green.
- [ ] Run the real binary test GREEN, then commit the E2E/documentation slice.

### Task 3: Verification, review, and closure

**Files:**

- Modify: `docs/origin-main-findings/commit-055-dd3f09b0c3.md`
- Modify: `docs/porting-origin-main-campaign.md`
- Modify: `docs/specs/README.md`

- [ ] Run formatting, focused runtime tests, native binary E2E, runtime/CLI
  checks appropriate to the changed transport surface, and changed-scope
  Clippy.
- [ ] Perform a full Graham review over the exact base-to-tip diff; repair
  every finding with TDD and rerun affected tests.
- [ ] Commit closure evidence, create an actual two-parent `ours` provenance
  merge, and verify parent order, first-parent tree equality, exact second
  parent, clean status, and no imported upstream Python files.
- [ ] Return the immutable closure tip and evidence to root for independent
  Graham approval.  Do not integrate or remove this worktree.
