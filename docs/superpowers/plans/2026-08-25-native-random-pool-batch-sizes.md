# Native random-pool batch sizes implementation plan

> **Author:** Sol
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:test-driven-development for every behavior slice and
> superpowers:verification-before-completion before each success claim.

**Goal:** Deliver all four file-backed random-pool batch sizes through native
CLI/YAML/config resolution into safe, deterministic loader execution, with
real-binary evidence and exact upstream merge ancestry.

**Architecture:** Extend the existing minimal `Inputs` projection beside the
already-landed image field; keep format validation in the dataset resolver and
content-dependent safety validation in `RandomPoolComposer`. Reuse the existing
native JSON/JSONL loader, sampler, body construction, and dry-run binary harness.

**Tech stack:** Rust 2024, clap, serde/serde_json, Config v2, native dataset
composer, cargo tests, sccache.

**Spec:** `docs/specs/2026-08-25-native-random-pool-batch-sizes.md`

## Global constraints

- Work only in `/mnt/4tb/ajc/port-060-c2889280a6`, based exactly on
  `1d1978c22e00786ccf8739a599fd5b70d0d1b191`.
- Begin each production behavior with an observed failing Rust test.
- Use `RUSTC_WRAPPER=sccache`, `SCCACHE_DIR=/mnt/4tb/sccache-port060`, and
  `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port060`.
- Preserve the existing image-YAML projection and unrelated base churn.
- The final merge must retain the reviewed native tree and record exact upstream
  `c2889280a66fc85b44e9456fd7020874c73a44fc` as its second parent. Never
  cherry-pick or import the upstream Python tree.

---

### Task 1: Config and CLI projection

**Files:**
- Modify: `rust/runtime/src/config/resolve.rs`
- Modify: `rust/cli/src/load.rs`
- Modify: `rust/cli/src/yaml.rs`

**Interfaces:**
- Produces: four optional random-pool authored values in `Inputs` and exact
  `*_batch_size` file options.
- Consumes: existing clap flags and YAML modality-section batch fields.

- [x] Write failing CLI-only projection tests for four distinct values and an
  explicit zero.
- [x] Write failing YAML tests proving all-four preservation, camel-case decode,
  per-field CLI precedence, and unset-CLI preservation.
- [x] Run focused CLI tests and record the missing text/audio/video projection
  failures.
- [x] Implement only the four-field projection and precedence behavior.
- [x] Rerun focused tests, record GREEN, and commit the slice.

### Task 2: Dataset-kind, format, and directory validation

**Files:**
- Modify: `rust/runtime/src/config/resolve.rs`
- Modify: focused resolver/CLI tests in the same modules.

**Interfaces:**
- Produces: pre-execution validation errors and accepted unit directory values.
- Consumes: the optional authored values from Task 1.

- [x] Write failing tests for public datasets, a non-random-pool file format,
  each directory non-unit modality, explicit directory units, and synthetic
  YAML non-interference.
- [x] Run focused tests and record that invalid configurations currently resolve.
- [x] Add the minimal branch-local validation without changing synthetic batch
  semantics.
- [x] Rerun focused tests, record GREEN, and commit the slice.

### Task 3: Content-aware safe composition

**Files:**
- Modify: `rust/runtime/src/dataset/loader/random_pool.rs`

**Interfaces:**
- Produces: safe associated/flattened mode selection and explicit validation
  errors.
- Consumes: parsed `PoolEntry` modality contents and four loader option values.

- [x] Write failing real-document composer tests for four modality counts,
  default inclusion, zero suppression, absent-modality zero preserving named
  association, and deterministic replacement sampling.
- [x] Write failing tests for multi-pool, named-group, image UUID, and all-present-
  zero refusal.
- [x] Run the focused runtime tests and record each semantic failure.
- [x] Implement present-modality mode selection, metadata guards, and non-empty
  validation without changing the unit associated path.
- [x] Rerun focused runtime tests, record GREEN, and commit the slice.

### Task 4: Native-binary loader integration

**Files:**
- Create or modify: a focused module under `rust/dry-run-tests/tests/`

**Interfaces:**
- Consumes: the built native `aiperf` binary, real JSON/JSONL random-pool source,
  Config-v2 YAML, and raw-record artifacts.
- Produces: end-to-end evidence that authored batches reach loader composition
  and outbound request construction.

- [x] Inventory the existing dry-run harness and choose the smallest fixture.
- [x] Add a test whose expected request content requires a non-unit text
  batch and at least one second modality/default interaction.
- [x] Reuse the projection and loader RED evidence from Tasks 1 and 3 for the
  behavior exercised end to end.
- [x] Complete only the missing integration wiring or assertions.
- [x] Run the native-binary test, inspect the record artifact, record GREEN, and
  commit the slice.

### Task 5: Review, verification, ancestry, and closure

**Files:**
- Create: `.superpowers/sdd/2026-08-25-native-random-pool-batch-sizes/graham-review.md`
- Create: `.superpowers/sdd/2026-08-25-native-random-pool-batch-sizes/graham-rereview.md`
- Modify: `docs/origin-main-findings/commit-060-c2889280a6.md`
- Modify: `docs/porting-origin-main-campaign.md`

**Interfaces:**
- Produces: review-approved native implementation, exact two-parent target-only
  merge, and evidence-backed tracker closure.

- [ ] Run formatting, focused runtime/CLI/config/integration tests, appropriate
  feature-bearing tests, and changed-scope Clippy with fresh output.
- [ ] Request an independent full Graham review over the exact base-to-tip diff;
  record every Critical/Important finding and its evidence.
- [ ] Apply review repairs with a failing regression test first, rerun focused
  verification, and request independent re-review until approved.
- [ ] Commit the review receipts and closure evidence; update tracker only after
  approval.
- [ ] Create an actual two-parent `ours`-tree merge with exact upstream as second
  parent, then verify parent order, first-parent tree equality, clean status, and
  absence of imported upstream Python changes.
