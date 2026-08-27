<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Task 1D-R implementation-readiness correction

## Decision

Foundation Task 1D-R implements the reliability-continuation contract only
after Task 1D lands. This correction fixes the compile, authority, budget, and
test seams needed to implement that task; it does not add adapter, P2/P4,
checkpoint-coordinator, result-compaction, or product behavior.

The public reliability vocabulary remains the vocabulary in
`reliability-continuation-course-correction.md`, subject to the exact additions
below. Private fields remain private, live authority remains non-deserializable,
and only the host classifier may select `FailRun`.

## Exact owned files

Task 1D-R owns the files already listed in the foundation plan and additionally
modifies:

- `rust/runtime/src/streaming/budget.rs` for bounded synchronous acquisition and
  one combined-then-split pair acquisition;
- `rust/runtime/src/streaming/blocking.rs` for the new handled-cut field in unit
  fixtures;
- `rust/runtime/tests/streaming_budget.rs` for synchronous and pair-acquisition
  accounting/cancellation proofs;
- `rust/runtime/tests/streaming_blocking.rs` and
  `rust/runtime/tests/streaming_checkpoint_participants.rs` to replace direct
  `CommittedParticipantState::new` construction with verified current-v4 reader
  fixtures.

No Task 1D-R Rust change is required in `identity.rs`, `results.rs`, an adapter,
or a later P2/P4 module.

## Budget seams

The landed asynchronous budget alone cannot implement synchronous failed-action
enqueue, and two sequential acquisitions from one budget can hold the first
lease forever while waiting for the second. Task 1D-R therefore extends the
Task 1B API exactly as follows:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetCharge {
    pub items: usize,
    pub bytes: usize,
}

impl StreamingResourceBudget {
    pub fn try_acquire(
        &self,
        items: usize,
        bytes: usize,
    ) -> Result<BudgetLease, BudgetError>;

    pub async fn acquire_pair(
        &self,
        first: BudgetCharge,
        second: BudgetCharge,
    ) -> Result<(BudgetLease, BudgetLease), BudgetError>;
}

impl BudgetLease {
    pub fn split_off(
        &mut self,
        items: usize,
        bytes: usize,
    ) -> Result<BudgetLease, BudgetError>;
}
```

`try_acquire` uses Tokio owned `try_acquire_many_owned` in the same
item-then-byte order as `acquire`; byte failure drops the item permit before
return. A no-permits result is the new stable `BudgetError::CapacityUnavailable`,
distinct from an oversized request. It never waits or mutates counters on
failure.

`acquire_pair` checked-adds both charges, acquires the combined item/byte charge
once, and then uses `split_off` to produce two exact leases. `split_off` moves
owned permits and divides recorded charges without releasing or reacquiring
capacity; both returned charges sum exactly to the original charge. This is the
only Task 1D-R multi-resource pattern. Tombstone payload/view, export
encoded/parsed, and receipt-partition payload/view ownership use it, so they
cannot hold one sub-resource while waiting for another. Cancellation before the
combined acquisition returns leaves no charge; cancellation after it cannot
occur because splitting is synchronous.

`enqueue_failed_action` computes the bounded pending-entry and encoded-receipt
charge, calls the reporter-owned budget's `try_acquire`, and retains the
move-only lease with the issue before returning `QueuedActionFailure`. Capacity
refusal returns `StreamingReliabilityError::StateBudget` without advancing a
frontier or counter. Re-enqueueing the same sealed evidence and semantic issue
after capacity is available returns the same issue identity and counts it once.

## Public construction and borrow surface

Private-field public types must still be callable by their intended owners.
Task 1D-R supplies these checked constructors and accessors; none returns a
`BudgetLease`, private proof, private decision, or mutable receipt field.

- `StreamingIssueComponentId::{new, as_str}` and
  `StreamingInputDomainIdentity::{new, stream_identity, source_identity}`.
- `StreamingIssueScope::kind` and the scope identity borrow accessors needed to
  validate order.
- Scope-specific `OrdinaryStreamingIssue::{run_diagnostic, partition, record,
  session, action, export, checkpoint_attempt}` constructors plus
  `run`, `scope`, `class`, `stage`, `code`, `semantic_context_digest`, and
  `order`. Constructors reject `Invariant`; record/session constructors derive
  an order key containing the same input domain; action derives its global
  sequence order; export and checkpoint-attempt bind their full generation or
  epoch/attempt identity.
- `StreamingIssueThresholdRule::new` plus `rule_id`, `scope`, `class`, `code`,
  `retry_limit`, `exhausted_disposition`, and `admission_fence_count`.
  `PreparedStreamingIssuePolicy::{new, digest, rule_for}`; `rule_for` returns a
  borrowed checked rule.
- `StreamingIssueOutcome::{issue_id, disposition, needs_admission_fence}`;
  `StreamingIssueCounterKey::{domain, rule_id}`; and
  `StreamingIssueCounterView::{get, iter}`.
- `HandledIssueCut::{empty, receipt_root, input_frontier_root,
  quarantine_tombstone_root}`. `empty` returns the canonical three empty roots;
  only the reporter and strict decoder construct a non-empty cut.
- `PreparedIssueReceiptPartitionView::{run, barrier, receipt_root, handled_cut,
  payload_bytes, payload_charge_bytes, view_charge_bytes}`.
- `PreparedActionRetry::retry_ordinal` and
  `PreparedActionBackpressure::needs_admission_fence`. The approved sealed
  `ActionFailureDisposition` remains exhaustive: `Pending`, `Retry`, and
  `Backpressure` cannot yield failure identity; only
  `TerminalActionReceipt(PreparedActionFailureIdentity)` can be consumed into a
  failed action terminal receipt.
- `PreparedActionFailureIdentity` retains its approved run/action/sequence/
  issue/evidence accessors. No conversion exists from either retry type.
- `PreparedSessionQuarantineInstall::{barrier, tombstone_root, view_revision,
  receipt_binding_root, payload_bytes, payload_charge_bytes,
  view_charge_bytes}`.
- `BudgetOwnedExportIssueReceipt::{encoded_charge_bytes,
  parsed_charge_bytes}`; `DerivedExportReceiptReference` borrow accessors for all
  four digest/length fields; and
  `PreparedExportAttemptFailure::{receipt, issue_id, is_exhausted,
  attempt_ordinal, counter_before, receipt_reference}`.

The export attempt ordinal and counter exposed by the prepared failure are
comparison inputs, not derived-status authority. Task 6C1 independently derives
`last_attempt_ordinal` and `counter_before` from the predecessor status, stores
both in `Exhausted`, and rejects the transition unless the prepared failure
agrees. Restore seeds validation from those status fields and only then compares
the receipt. First-attempt exhaustion is ordinal zero and counter-before zero;
`Complete` and `Exhausted` remain terminal status states.

The checked action view traits live in `action.rs`, and the quarantine view
trait lives in `session.rs`; each public trait has a private same-parent-module
sealed supertrait. `reliability.rs` consumes those traits but does not own their
seals. This lets only later child host modules implement them without exposing a
reliability-private constructor.

## Clone-safe handled cut and commit authority

`HandledIssueCut` is fixed-size authority and derives `Clone`, `Debug`, `Eq`,
`PartialEq`, and `Serialize`. It manually implements strict
unknown-field-denying `Deserialize` through the same private checked constructor
used by current-v4 verification. Its fields remain private. This preserves
`CheckpointCut: Clone` and therefore the landed barrier, descriptor, candidate,
generation, and receipt clone contracts.

`CheckpointCut` gains public `handled_issues: HandledIssueCut`. Existing fresh
fixtures use `HandledIssueCut::empty`; a non-empty value comes from a prepared
reporter view. All `CheckpointCut` literals in checkpoint, blocking, memory, and
shared test support are migrated in this task.

`CommittedParticipantReceipt` additionally stores the committed generation's
exact `result_index_root: ContentDigest` and exposes
`result_index_root(&self) -> &ContentDigest`. The constructor derives it from
the authoritative `CommittedCheckpointGeneration`; callers cannot supply it.
The reporter retires detailed receipts only when the callback's run, full
generation, participant descriptor/cut, and result-index root match the root
recorded for its staged receipt view. A pre-CAS drop or mismatched root retains
the view and detailed receipts unchanged.

## Current-v4 and legacy-v3 leased authority

The exact post-1D-R backend surface is the
`LeasedCheckpointGeneration`/`CurrentV4CheckpointGeneration` overlay in the
checkpoint-results plan. Ownership is:

- `checkpoint.rs`: `LegacyParticipantState`,
  `CurrentV4ParticipantStateContext`, current-v4/legacy wire DTOs, and
  crate-private `CommittedParticipantState::from_current_v4_reader`;
- `checkpoint_backend.rs`: `CurrentV4CheckpointGeneration`,
  `CheckpointGenerationStorageVersion`, opaque `LeasedCheckpointGeneration`,
  its borrowed `LeasedCheckpointGenerationView`, the sealed common reader
  trait, current and legacy reader traits, and the crate-private predecessor
  projection.

`LeasedCheckpointGeneration` implements the sealed
`VersionedLeasedGenerationReader`; external code cannot implement that trait or
construct the wrapper. Its common surface is generation identity plus result
index/segment reads. `view()` returns either a current reader with
`read_participant -> CommittedParticipantState` or a legacy reader with
`read_legacy_participant -> LegacyParticipantState`. `LegacyParticipantState`
has only `descriptor()` and `payload_bytes()` and no conversion into initializer
authority.

`StreamingCheckpointBackend::open_latest` returns
`Result<Option<LeasedCheckpointGeneration>, CheckpointError>`.
`begin_generation` accepts
`Option<CurrentV4CheckpointGeneration>`. The current wrapper has a private
field and only `generation()` publicly; only the current reader's crate-private
projection can mint it. `CheckpointCommitMetadata.previous` remains an untrusted
raw claim compared during prevalidation with the sealed expected predecessor.

The generation decoder first rejects bytes exceeding the backend's configured
generation-object limit. Current-v4 encoding contains the explicit strict field
`storage_version: "v4"` and the handled cut. Landed v3 encoding has neither.
The bounded discriminator applies these rules before full decode:

1. a present `storage_version` is decoded only by that named version; unknown,
   malformed, or failed-v4 verification returns `ObjectVerification` and never
   falls back;
2. absent `storage_version` is eligible for v3 only when the exact v3 top-level
   and cut field inventories are present and `handled_issues` is absent;
3. a v4-shaped handled cut without `storage_version: "v4"`, or v3 bytes with a
   handled cut, is malformed rather than legacy.

V3 verifies the v3 hash domain and constructs private legacy semantic state;
v4 verifies the v4 hash domain and alone constructs committed/current authority.

The memory backend adds the doc-hidden bounded integration seam
below. The two fixture DTOs live in `checkpoint_backend.rs`; the import method
lives in `checkpoints/memory.rs`. `LegacyV3FixtureObject::new` compact-copies
the encoded bytes and rejects an unrepresentable retained-byte charge.
`LegacyV3ReadOnlyFixture::new` rejects duplicate object digests and checked
item/byte-total overflow before retaining its boxed inventory. Role-specific
digest verification remains in the importer, where the legacy hash domains and
reachable descriptors are available.

```rust
#[doc(hidden)]
pub struct LegacyV3FixtureObject { /* private digest and exact boxed bytes */ }

impl LegacyV3FixtureObject {
    pub fn new(
        digest: ContentDigest,
        encoded: &[u8],
    ) -> Result<Self, CheckpointError>;
}

#[doc(hidden)]
pub struct LegacyV3ReadOnlyFixture { /* private run, head, and objects */ }

impl LegacyV3ReadOnlyFixture {
    pub fn new(
        run: StreamRunIdentity,
        head: CheckpointGeneration,
        generation_object: LegacyV3FixtureObject,
        participant_objects: Box<[LegacyV3FixtureObject]>,
        result_index_object: LegacyV3FixtureObject,
        result_objects: Box<[LegacyV3FixtureObject]>,
    ) -> Result<Self, CheckpointError>;
}

impl MemoryCheckpointBackend {
    #[doc(hidden)]
    pub async fn import_legacy_v3_read_only_fixture(
        &self,
        fixture: LegacyV3ReadOnlyFixture,
    ) -> Result<(), CheckpointError>;
}
```

The importer bounds the generation object before decoding; enforces the same
strict v3 decoder, object digest/length/reachability checks, exact run/head
identity, and backend storage budgets; requires an empty run head and object
namespace; acquires the complete missing-object storage charge before mutation;
and atomically installs only a `LegacyV3ReadOnly` head. It cannot create current
authority or overwrite any head. Unit tests may use a smaller private builder;
public integration support uses only this checked seam.

## Existing test migration

The public `CommittedParticipantState::new` is removed. Shared test support adds
`committed_current_v4_participant_state`, which creates a memory generation,
opens it as `CurrentV4`, and reads the reachable participant through the current
reader. Blocking and checkpoint-participant tests use that helper. Privacy
tests still prove that copying legacy descriptor/payload bytes into a new
`BudgetedCheckpointBytes` cannot call the crate-private promotion function.

The `LegacyParticipantState` compile-fail examples import it from
`aiperf_runtime::streaming::checkpoint`, its owning module, not
`checkpoint_backend`.

## Task 1E boundary

Task 1E injects a separately owned reporter into both conformance harnesses:

```rust
pub async fn assert_source_conformance(
    factory: &dyn StreamingDatasetSourceFactory,
    reporter: Box<dyn StreamingIssueReporter>,
    cases: SourceConformanceCases,
);

pub async fn assert_format_conformance(
    factory: &dyn StreamingDatasetFormatFactory,
    reporter: Box<dyn StreamingIssueReporter>,
    cases: FormatConformanceCases,
);
```

The harness takes ownership of the separately constructed reporter; neither a
source nor a format owns it. The harness borrows that owned reporter only after
a stage future returns and releases the borrow before the next
source/format/control await. Ordinary scripted faults are classified and
reported, then the harness proves the next valid unit remains available.
`StreamSourceError::Stopped` is owned by host stop control: it is
valid only after the harness calls the separate source control's `stop`, wakes
the pending source future, creates no issue receipt, advances no source seal or
frontier, and is never adapter-authored continuation policy.

## Task 5B publication atomicity

Task 1D-R preserves the landed Task 5B transaction and publication order
verbatim:

1. validate run, sealed expected lineage, exact successor epoch, complete
   participant inventory, result epoch, totals, and roots;
2. build and self-verify the complete current-v4 candidate;
3. encode the generation and result index and reject conflicting immutable
   objects;
4. acquire the complete missing immutable-storage charge;
5. honor `AfterPrevalidationBeforePublication` before any state mutation;
6. take the one final mutable backend-state borrow, compare the exact versioned
   run head, insert immutable objects, and replace the head.

No `RefCell` borrow crosses an await. Open snapshots identity/length, acquires
its read lease, then rechecks the exact versioned head before minting authority.
Begin with `None` inspects the actual head before returning a transaction and
returns `LegacyReadOnlyHead` for v3; it never treats omission as overwrite.
Legacy import follows the same acquire-before-single-mutation rule. No version
check, fixture seam, reporter callback, or migration path publishes a partial
generation.

## Executable RED additions

Before production changes, add these named REDs to the existing Task 1D-R
suites:

- `synchronous_action_enqueue_refuses_immediately_without_advancing_state`;
- `combined_pair_acquisition_cannot_hold_one_sublease_while_waiting_for_other`;
- `cancelled_combined_pair_acquisition_leaves_zero_charge`;
- `handled_issue_cut_is_clone_compatible_with_checkpoint_cut`;
- `committed_receipt_binds_exact_result_index_root`;
- `mismatched_result_index_root_retains_detailed_receipts`;
- `current_participant_restore_uses_verified_reader_not_public_constructor`;
- `checked_legacy_fixture_is_bounded_read_only_and_cannot_overwrite_head`;
- `unknown_or_malformed_explicit_v4_never_falls_back_to_v3`;
- `v4_shape_without_explicit_v4_discriminator_is_refused`;
- `action_disposition_variants_expose_only_their_approved_type_state`;
- `first_and_later_exhaustion_compare_status_owned_ordinal_and_counter`;
- `conformance_reporter_is_released_before_each_await`;
- `host_stop_wakes_pending_source_without_issue_or_seal`.

The existing v2 issue golden, v4 generation golden, v3 read-only, reverse/skew,
privacy, stale tombstone, export restore, pre-CAS retry, and high-fault budget
tests remain required.
