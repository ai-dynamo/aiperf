<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Streaming reliability-continuation course correction

## Audit finding

The reliability review found that the approved plans typed failures by stage
but still allowed ordinary adapter, endpoint, and derived-sink errors to bubble
through `Result::Err` into one global run-abort path. That would make a long
benchmark less reliable than the finite path and would discard useful truthful
work after a single corrupt datum or unavailable optional sink. The user-set
priority is continuation with explicit evidence, bounded state, and exact
authority boundaries.

## Decision

Native streaming benchmark execution is reliability-first. An ordinary bad
partition, record, session, endpoint response, exporter attempt, or compaction
attempt is evidence about the run; it is not automatically authority to abort
the run. The host classifies each typed issue, chooses a scoped disposition,
records a deterministic receipt, applies a frozen threshold, and continues
whenever truthful ordering and accounting remain possible.

This record amends the approved streaming design and every linked subsystem
plan. It preserves existing security, identity, budget, checkpoint, CAS, and
truthful-cut requirements. It narrows unconditional `FailRun` authority; it
does not weaken validation or turn corrupt state into a partial success.

## Terminal boundary

`FailRun` is valid only for one of these conditions:

1. logical-run, publication-proof, writer-lease, or CAS authority mismatch;
2. an ordering, watermark, or checkpoint cut that cannot be represented
   truthfully without inventing or omitting committed membership;
3. conflicting content for one stable identity, or drift in a frozen semantic
   input such as source snapshot identity,
   projection schema, participant inventory, execution/result plan digest,
   tokenizer/synthesis profile, or cellular placement digest;
4. item/byte lease, membership, metric, receipt, or result-index accounting
   corruption.

Configuration/schema errors and unavailable required security capabilities
still refuse before execution effects. Authentication,
TLS, no-follow, digest, length, replay-frame, cellular admission, and secret
redaction checks remain fail-closed. A runtime integrity failure that proves
frozen semantic drift is terminal under item 3. A transient read, malformed
benchmark datum, endpoint 4xx/5xx/timeout, or derived-output sink failure is not.
Runtime authentication may refresh and retry while preserving the exact frozen
immutable identity. It may fail the run only when no authorized immutable source
can be acquired without substituting or falsifying that identity.

Ordinary threshold exhaustion may stop new admission and safely drain or seal
the available prefix, but it cannot construct `FailRun`. This keeps overload
and data-quality policy distinct from authority/invariant failure.

## Host-owned vocabulary

Foundation Task 1D-R owns the following public vocabulary in
`rust/runtime/src/streaming/reliability.rs`. Adapters and endpoints report
facts; only the host policy and ledger choose dispositions.

```rust
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct StreamingIssueComponentId(String);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingInputDomainIdentity {
    stream_identity: ContentDigest,
    source_identity: ImmutableObjectIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum StreamingIssueScope {
    Run,
    Partition {
        input_domain: StreamingInputDomainIdentity,
        object: ImmutableObjectIdentity,
    },
    Record {
        input_domain: StreamingInputDomainIdentity,
        record_id: StableRecordId,
    },
    Session {
        input_domain: StreamingInputDomainIdentity,
        session_key: StableSessionKey,
    },
    Action { action_id: StableActionId },
    Export {
        exporter_id: StreamingIssueComponentId,
        generation: CheckpointGeneration,
    },
    CheckpointAttempt {
        generation: CheckpointEpoch,
        attempt_ordinal: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingIssueClass {
    Retryable,
    Permanent,
    Invariant,
    Capacity,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingTerminalInvariant {
    RunAuthorityMismatch,
    SourceIdentityAuthorityMismatch,
    PublicationProofMismatch,
    WriterLeaseMismatch,
    CasExpectationMismatch,
    SecurityAuthorityMismatch,
    ConflictingStableContent,
    ImpossibleTruthfulOrdering,
    ImpossibleTruthfulWatermark,
    ImpossibleTruthfulCut,
    FrozenSemanticDrift,
    AccountingCorruption,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingIssueDisposition {
    Retry,
    Backpressure,
    Quarantine,
    Hole,
    Continue,
    TerminalActionReceipt,
    ExportIncomplete,
    FailRun,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingIssueScopeKind {
    Run,
    Partition,
    Record,
    Session,
    Action,
    Export,
    CheckpointAttempt,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingIssueOrderKey {
    pub input_domain: Option<StreamingInputDomainIdentity>,
    pub source_position: Option<SourcePosition>,
    pub global_sequence: Option<GlobalSequence>,
    pub retry_ordinal: u32,
    pub scope_tiebreaker: ContentDigest,
}

#[derive(Debug, Eq, PartialEq)]
pub struct OrdinaryStreamingIssue {
    run: StreamRunIdentity,
    scope: StreamingIssueScope,
    class: StreamingIssueClass,
    stage: StreamingFailureStage,
    code: StreamingIssueComponentId,
    semantic_context_digest: ContentDigest,
    order: StreamingIssueOrderKey,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct StreamingIssueThresholdRule {
    rule_id: StreamingIssueComponentId,
    scope: StreamingIssueScopeKind,
    class: StreamingIssueClass,
    code: Option<StreamingIssueComponentId>,
    retry_limit: u32,
    exhausted_disposition: StreamingIssueDisposition,
    admission_fence_count: Option<NonZeroU64>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct StreamingIssueDecision {
    disposition: StreamingIssueDisposition,
    rule: StreamingIssueThresholdRule,
    needs_admission_fence: bool,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct PersistedStreamingIssueReceipt {
    issue_id: ContentDigest,
    run: StreamRunIdentity,
    scope: StreamingIssueScope,
    class: StreamingIssueClass,
    stage: StreamingFailureStage,
    code: StreamingIssueComponentId,
    semantic_context_digest: ContentDigest,
    order: StreamingIssueOrderKey,
    terminal_invariant: Option<StreamingTerminalInvariant>,
    disposition: StreamingIssueDisposition,
    threshold: StreamingIssueThresholdReceipt,
}

pub struct BudgetOwnedStreamingIssueReceipt {
    encoded: BudgetedCheckpointBytes,
}

pub struct PreparedIssueReceiptPartitionView {
    run: StreamRunIdentity,
    barrier: CheckpointBarrier,
    receipt_root: ContentDigest,
    handled_cut: HandledIssueCut,
    payload: BudgetedCheckpointBytes,
    view_lease: BudgetLease,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct HandledIssueCut {
    receipt_root: ContentDigest,
    input_frontier_root: ContentDigest,
    quarantine_tombstone_root: ContentDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionQuarantineClosureKind {
    AuthoredClose,
    HardWatermark,
    FiniteSeal,
    CompleteSortedRun,
    PolicyExhaustion,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct SessionQuarantineClosureProof {
    kind: SessionQuarantineClosureKind,
    causal_frontier: SessionCausalFrontier,
    evidence_digest: ContentDigest,
}

pub struct SessionQuarantineTombstone {
    run: StreamRunIdentity,
    input_domain: StreamingInputDomainIdentity,
    session_key: StableSessionKey,
    issue_id: ContentDigest,
    causal_frontier: SessionCausalFrontier,
    closure_proof: SessionQuarantineClosureProof,
    encoded: BudgetedCheckpointBytes,
    parsed_lease: BudgetLease,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingIssueOutcome {
    issue_id: ContentDigest,
    disposition: StreamingIssueDisposition,
    needs_admission_fence: bool,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct StreamingIssueThresholdReceipt {
    policy_digest: ContentDigest,
    rule_id: StreamingIssueComponentId,
    prior_matching_count: u64,
    resulting_matching_count: u64,
    retry_ordinal: u32,
    is_exhausted: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StreamingIssueCounterDomain {
    Run,
    Input(StreamingInputDomainIdentity),
    Action,
    Export {
        exporter_id: StreamingIssueComponentId,
        generation: CheckpointGeneration,
    },
    CheckpointAttempt,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StreamingIssueCounterKey {
    domain: StreamingIssueCounterDomain,
    rule_id: StreamingIssueComponentId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingIssueCounterView<'a> {
    counters: &'a BTreeMap<StreamingIssueCounterKey, u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingIssueSummary {
    pub total: u64,
    pub by_scope: BTreeMap<StreamingIssueScopeKind, u64>,
    pub by_class: BTreeMap<StreamingIssueClass, u64>,
    pub by_disposition: BTreeMap<StreamingIssueDisposition, u64>,
    pub is_admission_fenced: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamingReliabilityError {
    InvalidComponentId,
    InvalidScopeOrder,
    PolicyDigestMismatch,
    CounterOverflow,
    IllegalDisposition,
    IllegalFailRun,
    IllegalTerminalInvariant,
    ForeignRun,
    StateBudget(StateBudgetFailureCode),
    CorruptCheckpointState,
    AmbiguousPolicyRule,
    NonContiguousIssueFrontier,
    ReceiptBudget(StateBudgetFailureCode),
    InvalidActionTerminalMembership,
    IncompleteActionInventory,
    QuarantineReceiptUnavailable,
    StaleQuarantineTombstoneView,
    QuarantineInstallBudget(StateBudgetFailureCode),
    ExportReceiptRunMismatch,
    ExportReceiptGenerationMismatch,
    ExportReceiptSinkMismatch,
    ExportReceiptAttemptMismatch,
    ExportReceiptPolicyMismatch,
    ExportReceiptDigestLengthMismatch,
    NonContiguousExportCounter,
    DerivedExportReceiptUnreachable,
    ExportReceiptBudget(StateBudgetFailureCode),
}

#[derive(Debug)]
pub struct PreparedStreamingIssuePolicy {
    digest: ContentDigest,
    rules: Box<[StreamingIssueThresholdRule]>,
}

enum CheckedActionSequenceOutcome {
    Succeeded,
    Failed { issue_id: ContentDigest },
}

struct SealedActionGapClosureProof {
    membership_root: ContentDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionTerminalMembershipOutcomeView {
    Succeeded,
    Failed { issue_id: ContentDigest },
}

/// Defined in `action.rs`; only its child action-host modules can implement the
/// private sealed supertrait.
mod action_view_seal {
    pub trait CheckedActionFailureTerminalEvidenceView {}
    pub trait CheckedActionTerminalMembershipView {}
    pub trait FrozenActionInventoryView {}
}

pub trait CheckedActionFailureTerminalEvidenceView:
    action_view_seal::CheckedActionFailureTerminalEvidenceView
{
    fn run(&self) -> &StreamRunIdentity;
    fn action_id(&self) -> StableActionId;
    fn sequence(&self) -> GlobalSequence;
    fn terminal_evidence_digest(&self) -> ContentDigest;
}

pub trait CheckedActionTerminalMembershipView:
    action_view_seal::CheckedActionTerminalMembershipView
{
    fn run(&self) -> &StreamRunIdentity;
    fn action_id(&self) -> StableActionId;
    fn sequence(&self) -> GlobalSequence;
    fn outcome(&self) -> ActionTerminalMembershipOutcomeView;
    fn membership_digest(&self) -> ContentDigest;
}

/// Defined in `action.rs`; only the child workload host can implement the
/// private sealed supertrait for its frozen inventory.
pub trait FrozenActionInventoryView: action_view_seal::FrozenActionInventoryView {
    fn run(&self) -> &StreamRunIdentity;
    fn through(&self) -> GlobalSequence;
    fn membership_root(&self) -> ContentDigest;
    fn contains_terminal(
        &self,
        sequence: GlobalSequence,
        membership_digest: ContentDigest,
    ) -> bool;
}

/// Defined in `session.rs`; only its P1B child module can implement the private
/// sealed supertrait for a retained-map borrow.
mod session_view_seal {
    pub trait SessionQuarantineTombstoneView {}
}

pub trait SessionQuarantineTombstoneView:
    session_view_seal::SessionQuarantineTombstoneView
{
    fn run(&self) -> &StreamRunIdentity;
    fn tombstone_root(&self) -> ContentDigest;
    fn revision(&self) -> u64;
    fn canonical_encoded_entries(&self) -> &[u8];
}

pub struct CheckedActionTerminalFact {
    action_id: StableActionId,
    sequence: GlobalSequence,
    outcome: CheckedActionSequenceOutcome,
}

pub struct PreparedActionFailureIdentity {
    run: StreamRunIdentity,
    action_id: StableActionId,
    sequence: GlobalSequence,
    issue_id: ContentDigest,
    terminal_evidence_digest: ContentDigest,
}

#[derive(Debug)]
pub struct QueuedActionFailure {
    reporter_token: u64,
}

#[derive(Debug)]
pub struct PreparedActionRetry {
    retry_ordinal: u32,
}

#[derive(Debug)]
pub struct PreparedActionBackpressure {
    needs_admission_fence: bool,
}

#[derive(Debug)]
pub enum ActionFailureDisposition {
    Pending(QueuedActionFailure),
    Retry(PreparedActionRetry),
    Backpressure(PreparedActionBackpressure),
    TerminalActionReceipt(PreparedActionFailureIdentity),
}

// All payload fields and constructors are private to the reliability module;
// matching is public, but constructing or converting a disposition is not.

impl PreparedActionFailureIdentity {
    pub fn run(&self) -> &StreamRunIdentity { &self.run }
    pub fn action_id(&self) -> StableActionId { self.action_id }
    pub fn sequence(&self) -> GlobalSequence { self.sequence }
    pub fn issue_id(&self) -> ContentDigest { self.issue_id }
    pub fn terminal_evidence_digest(&self) -> ContentDigest { self.terminal_evidence_digest }
}

pub struct CheckedNoMoreActionsBefore {
    through: GlobalSequence,
    proof: SealedActionGapClosureProof,
}

pub struct PreparedSessionQuarantineInstall {
    barrier: CheckpointBarrier,
    tombstone_root: ContentDigest,
    view_revision: u64,
    receipt_binding_root: ContentDigest,
    payload: BudgetedCheckpointBytes,
    view_lease: BudgetLease,
    payload_charge_bytes: usize,
    view_charge_bytes: usize,
}

pub enum ResultSinkAttemptOutcome {
    Failed(OrdinaryStreamingIssue),
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct PersistedExportIssueReceipt {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    attempt_ordinal: u32,
    issue_id: ContentDigest,
    policy_digest: ContentDigest,
    counter_before: u64,
    counter_after: u64,
    embedded_receipt_digest: ContentDigest,
    embedded_receipt_length: u64,
    embedded_receipt: PersistedStreamingIssueReceipt,
}

pub struct BudgetOwnedExportIssueReceipt {
    receipt: PersistedExportIssueReceipt,
    encoded: BudgetedCheckpointBytes,
    parsed_lease: BudgetLease,
    encoded_charge_bytes: usize,
    parsed_charge_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DerivedExportReceiptReference {
    receipt_digest: ContentDigest,
    receipt_length: u64,
    embedded_receipt_digest: ContentDigest,
    embedded_receipt_length: u64,
}

pub(crate) struct CheckedExportAttemptDecision {
    issue_id: ContentDigest,
    is_exhausted: bool,
}

pub struct PreparedExportAttemptFailure {
    decision: CheckedExportAttemptDecision,
    receipt: BudgetOwnedExportIssueReceipt,
    reference: DerivedExportReceiptReference,
}

pub struct DurableExportReceiptValidationContext {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    policy_digest: ContentDigest,
    expected_attempt_ordinal: u32,
    expected_counter_before: u64,
}

impl DurableExportReceiptValidationContext {
    pub(crate) async fn from_final_generation(
        reader: &LeasedCheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        expected_attempt_ordinal: u32,
        expected_counter_before: u64,
    ) -> Result<Self, StreamingReliabilityError>;
}

impl BudgetOwnedExportIssueReceipt {
    pub fn encoded_charge_bytes(&self) -> usize { self.encoded_charge_bytes }
    pub fn parsed_charge_bytes(&self) -> usize { self.parsed_charge_bytes }
}

impl PreparedExportAttemptFailure {
    pub fn receipt(&self) -> &BudgetOwnedExportIssueReceipt { &self.receipt }
    pub fn issue_id(&self) -> ContentDigest { self.decision.issue_id }
    pub fn is_exhausted(&self) -> bool { self.decision.is_exhausted }
    pub fn receipt_reference(&self) -> &DerivedExportReceiptReference { &self.reference }
}

pub async fn restore_durable_export_issue_receipt(
    encoded: BudgetedCheckpointBytes,
    expected_reference: &DerivedExportReceiptReference,
    context: &DurableExportReceiptValidationContext,
    parsed_budget: &StreamingResourceBudget,
) -> Result<BudgetOwnedExportIssueReceipt, StreamingReliabilityError>;

impl PreparedSessionQuarantineInstall {
    pub fn payload_charge_bytes(&self) -> usize { self.payload_charge_bytes }
    pub fn view_charge_bytes(&self) -> usize { self.view_charge_bytes }
}

pub enum IssueSequenceUpdate {
    Issue(OrdinaryStreamingIssue),
    NoMoreBefore {
        input_domain: StreamingInputDomainIdentity,
        through: SourcePosition,
    },
    CheckedActionTerminal(CheckedActionTerminalFact),
    CheckedNoMoreActionsBefore(CheckedNoMoreActionsBefore),
    PreparedSessionQuarantineInstall(PreparedSessionQuarantineInstall),
}

#[async_trait::async_trait(?Send)]
pub trait StreamingIssueReporter: StreamingCheckpointParticipant {
    fn enqueue_failed_action(
        &mut self,
        evidence: &dyn CheckedActionFailureTerminalEvidenceView,
        issue: OrdinaryStreamingIssue,
    ) -> Result<QueuedActionFailure, StreamingReliabilityError>;
    fn poll_failed_action(
        &mut self,
        queued: QueuedActionFailure,
    ) -> Result<ActionFailureDisposition, StreamingReliabilityError>;
    fn prepare_action_terminal(
        &mut self,
        membership: &dyn CheckedActionTerminalMembershipView,
    ) -> Result<CheckedActionTerminalFact, StreamingReliabilityError>;
    fn prepare_no_more_actions_before(
        &mut self,
        inventory: &dyn FrozenActionInventoryView,
        through: GlobalSequence,
    ) -> Result<CheckedNoMoreActionsBefore, StreamingReliabilityError>;
    async fn prepare_session_quarantine_install(
        &mut self,
        view: &dyn SessionQuarantineTombstoneView,
        issue_id: ContentDigest,
        barrier: &CheckpointBarrier,
        budget: &StreamingResourceBudget,
    ) -> Result<PreparedSessionQuarantineInstall, StreamingReliabilityError>;
    fn verify_session_quarantine_install(
        &self,
        prepared: &PreparedSessionQuarantineInstall,
        current_view: &dyn SessionQuarantineTombstoneView,
        barrier: &CheckpointBarrier,
    ) -> Result<(), StreamingReliabilityError>;
    async fn prepare_export_attempt_failure(
        &mut self,
        run: &StreamRunIdentity,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        attempt_ordinal: u32,
        outcome: ResultSinkAttemptOutcome,
        budget: &StreamingResourceBudget,
    ) -> Result<PreparedExportAttemptFailure, StreamingReliabilityError>;
    async fn report(
        &mut self,
        update: IssueSequenceUpdate,
    ) -> Result<Option<StreamingIssueOutcome>, StreamingReliabilityError>;
    async fn receipt_partition_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedIssueReceiptPartitionView, StreamingReliabilityError>;
    fn counters(&self) -> StreamingIssueCounterView<'_>;
    fn summary(&self) -> Result<StreamingIssueSummary, StreamingReliabilityError>;
}
```

`StreamingReliabilityError` implements `Display` and `Error` without retaining
raw source data, credentials, endpoint payloads, or free-form error strings.

`StreamingIssueComponentId::new` accepts 1-128 byte ASCII identifiers matching
`[a-z][a-z0-9_]*`; `TryFrom<String>` delegates to that constructor.
`StreamingInputDomainIdentity::new` binds the frozen stream-identity digest and
exact immutable source identity. Record and session scopes must contain it, and
their order key must repeat the same domain. Scope constructors reject a
missing or different domain before allocating receipt bytes.

`OrdinaryStreamingIssue` is a live, non-serializable fact with private fields
and scope-specific constructors. It cannot carry `StreamingTerminalInvariant`,
cannot select a disposition, and is not `Clone`. The reliability module owns a
private exhaustive `HostFailure` enum, private `VerifiedHostIssue`, and private
`classify_host_failure` match. Stage-specific crate entry points accept concrete
typed source/checkpoint/security/accounting errors and immediately invoke that
match; there is no marker trait, no `pub(crate)` proof type, and no constructor
accepting a terminal tag. Adding a terminal-capable error variant therefore
makes the exhaustive match fail compilation until its classification is
reviewed.

The persisted form is deliberately separate. `PersistedStreamingIssueReceipt`
is serialize-only, non-`Clone`, and has private fields. A private strict wire DTO
may deserialize bytes, but only module-private
`verify_persisted_receipt(wire, committed_generation, policy)` can produce a
live verified receipt. It requires reachability from that exact committed
generation, recomputes the issue ID and policy decision, and revalidates any
terminal invariant through the exhaustive classifier table. Serde can never
construct `OrdinaryStreamingIssue`, `VerifiedHostIssue`,
`StreamingIssueDecision`, or `BudgetOwnedStreamingIssueReceipt`.

`StreamingIssueDecision` is created only by the concrete prepared policy inside
the module. Its exhaustive scope-by-disposition validator is:

| Scope/class | Allowed nonterminal dispositions | Terminal disposition |
|---|---|---|
| run + retryable/permanent/capacity diagnostic | `Continue` only with private `VerifiedNoMembershipLoss` | none |
| partition + retryable/permanent/capacity | `Retry`, `Backpressure`, `Hole` | none |
| record + retryable/permanent/capacity | `Retry`, `Backpressure`, `Quarantine` | none |
| session + retryable/permanent/capacity | `Retry`, `Backpressure`, `Quarantine` | none |
| action + retryable/permanent/capacity | `Retry`, `Backpressure`, `TerminalActionReceipt` | none |
| export + retryable/permanent/capacity | `Retry`, `Backpressure`, `ExportIncomplete` | none |
| checkpoint-attempt + retryable/permanent/capacity | `Retry`, `Backpressure` | none |
| every scope + module-classifier-verified invariant | none | `FailRun` |

No authored threshold rule may name `Continue` or `FailRun`.
Public ordinary-fact constructors reject `StreamingIssueClass::Invariant`; only
the module-private host classifier can create that class. The validator tests
the full `StreamingIssueScopeKind × StreamingIssueClass × disposition` product,
so adding any enum variant makes the exhaustive table fail to compile or test.
`VerifiedNoMembershipLoss` is module-private, has no serde form, and is emitted
only for a run-scoped diagnostic that drops no input, action, result, session,
or checkpoint membership. Capacity remains backpressure/retry/fence unless
lease accounting itself is corrupt. The summary counts retained continued
failures, and product `is_degraded` includes a nonzero continued-failure count.

## Deterministic receipts and thresholds

The issue ID uses Task 1A `domain_hash`: every field, including the domain, is
prefixed by its `u64` little-endian byte length before BLAKE3. The domain is
`aiperf.streaming.issue-receipt.v2`; the remaining canonical fields are:

```text
logical replay run bytes
scope and stable scope identity, including stream and source identity for records/sessions
issue class
failure stage and stable code
semantic context digest
order-key input-domain presence, stream identity, and source identity
source position or global sequence
host-assigned logical retry ordinal
scope tiebreaker
optional checked terminal invariant
```

Enum tags and component IDs are lowercase snake-case ASCII bytes. Integer
payloads are fixed-width little-endian. Each optional field is a separate
one-byte `0`/`1` presence field followed by its payload field only when present.
Scope-specific identities follow the scope tag in declaration order. The fixed
golden fixture is: run `[0x11; 32]`; input-domain stream digest `[0x21; 32]`;
immutable source identity from `[0x20; 32]`; record scope with record ID `[0x22; 32]`;
permanent/decode/syntax; semantic digest `[0x33; 32]`; source position `7`;
no global sequence; retry ordinal `0`; tiebreaker `[0x44; 32]`; no terminal
invariant. The exact v2 issue ID is
`92e68da0eae7dc5acf38db5f66eeb0f2214cbe358fdbfc43c4c0dcdd59892db6`.

It never includes process incarnation, cell/worker topology, wall time, retry
sleep, discovery race, or an error string. The host assigns the retry ordinal
from checkpointed scope state before an attempt; a crash reuses that ordinal,
while a receipt-authorized next attempt increments it. Re-reporting an issue ID returns the
same receipt without incrementing a counter. Every owner submits issues only
after its source frontier or global sequence makes the `StreamingIssueOrderKey`
final. Therefore restart and worker skew cannot change threshold crossing.

The reliability policy is validated before source polling, included in the
frozen execution-plan digest, and stored by the run-scoped issue-ledger
checkpoint participant. Matching is deterministic: an exact
`(scope, class, code)` rule always wins over the sole wildcard
`(scope, class, None)` rule. Construction rejects duplicate rule IDs, duplicate
exact match keys, multiple wildcards for one scope/class, missing wildcard
coverage, and every scope/disposition combination outside the table above.

The policy digest uses domain `aiperf.streaming.issue-policy.v1`. It sorts
checked rules by `(scope, class, exact-before-wildcard, code, rule_id)` and
length-frames rule ID, scope-kind tag, class tag, optional code, retry limit,
exhausted disposition, and optional admission-fence count. Input order cannot
change either the digest or selected rule.

One central `OrderedIssueSequencer` inside the ledger owns threshold order. It
keeps a budgeted pending map and a checkpointed contiguous/no-more-before
frontier for each `StreamingInputDomainIdentity`; record/session/partition facts
cannot be decided until the producer advances that domain past their source
position. Threshold counters for those scopes are domain-local, keyed by the
exact input domain plus rule ID. A late-discovered second domain therefore
cannot reorder or change a first domain's threshold crossing, and no frozen
global domain inventory is required.

Actions use a separate bounded pending map keyed by `GlobalSequence` and a
checkpointed terminal/no-more-actions-before frontier. Raw terminal and gap
updates are not public vocabulary. On failure, P2 first exposes a borrowed
`CheckedActionFailureTerminalEvidenceView` after exact action/sequence terminal
evidence exists but before an issue ID or terminal receipt exists. It passes
that sealed view plus one checked `OrdinaryStreamingIssue` to reporter-owned
`enqueue_failed_action`. That synchronous method verifies matching action
scope/order, retains the detailed issue in the reporter's bounded sequencer,
and returns a move-only `QueuedActionFailure` without advancing the action
frontier. `poll_failed_action` is also synchronous and consumes that token. It
returns `Pending(token)` while dense global ordering lacks predecessor or
no-more-before evidence, so P2 releases the reporter borrow and polls again at
the next explicit event boundary; no `&mut StreamingIssueReporter` is held
across an await. Once order permits classification it returns exactly one
sealed type-state variant for every allowed action-policy branch: `Retry` or
`Backpressure` carry only their private checked retry/fence decisions and no
value accepted by a terminal-receipt constructor, while
`TerminalActionReceipt` alone carries `PreparedActionFailureIdentity`. P2 may
reschedule on `Retry`, fence/pause on `Backpressure`, and consumes only the
terminal variant into its checked
`BudgetOwnedActionTerminalReceipt`; only then can its private terminal
membership expose `CheckedActionTerminalMembershipView` with the reporter-
retained issue ID. Dropping either a queued token or terminal preparation does
not remove the retained receipt: re-enqueueing the same terminal evidence and
semantic issue returns the same decision and issue ID without another counter
increment; replaying a polled decision is idempotent, while different content
for that evidence is a conflict. Success needs no first phase. P4 exposes only a borrowed `FrozenActionInventoryView`
implemented by its immutable action inventory. All three action views have action-module
private sealed supertraits, so sibling modules can pass a legitimate view but
cannot implement one or forge its fields. The reliability-owned reporter
methods `prepare_action_terminal` and `prepare_no_more_actions_before` verify
those views and mint the opaque, private-field `CheckedActionTerminalFact` and
`CheckedNoMoreActionsBefore`. The former binds exact run, stable action,
global sequence, membership digest, and success or the reporter's retained
checked failure receipt; the latter derives a sealed gap-closure proof only
after the frozen inventory accounts for every action through the frontier.
The reporter rechecks action/sequence membership,
failure-issue retention, duplicates, and contiguous gap coverage before accepting
either update. `IssueSequenceUpdate::Issue` rejects action scope, so it cannot
bypass the sealed terminal path. The single
action counter domain is therefore advanced in dense global-sequence order, not
worker arrival order. Run and checkpoint counters are run-local; export counters
are keyed by exact `(sink, full CheckpointGeneration)`. Canonical input order is
`(input_domain, source_position, scope, tiebreaker)` and canonical action order is
`(global_sequence, tiebreaker)`. Reverse arrival, a domain discovered late,
successes interleaved with failures, restart, adversarial skew, and one-versus-
eight-worker scripts must produce the same receipt IDs, decisions, counters,
and receipt root.

The issue-ledger participant checkpoints:

- the policy digest;
- bounded counters by `StreamingIssueCounterKey` and scope class;
- the contiguous/no-more-before frontier and bounded pending facts per input domain;
- the action terminal/no-more-actions-before frontier and bounded pending facts;
- a content-addressed receipt-index root plus an optional bounded hot lookup window;
- the exact per-scope retry ordinal needed to reproduce the current/next decision.

Every detailed receipt is encoded once into exact compact immutable bytes and
stored only as non-`Clone` `BudgetOwnedStreamingIssueReceipt`. The immediate
report call returns only fixed-size `StreamingIssueOutcome`; no caller receives
a freely cloneable heap receipt. The wrapper retains only the compact bytes;
module-private checked borrowed decoding uses a bounded scratch lease and cannot
detach a parsed heap DTO. Receipt-index keys, map capacity, tombstone fields,
and view metadata are charged separately and exactly. At a barrier,
`receipt_partition_view(&barrier)` returns one non-destructive, move-only,
budget-owned `PreparedIssueReceiptPartitionView` covering the sequencer frontier.
Task 6B consumes that view into an immutable result partition and Task 5E stages
it in the same transaction as the ledger participant state. Cancellation or
pre-CAS failure drops only the view and retains the ledger receipts for an
identical retry. The ledger retires detailed live receipts only after receiving
the same run/generation commit receipt whose reachable result root contains the
view. Restore verifies run, policy digest, receipt-index root, counters,
frontiers, and generation membership before accepting new facts; replay counts
each issue once.

For derived-sink failures, the caller constructs only
`ResultSinkAttemptOutcome::Failed(ordinary_export_issue)`. The reporter-owned
`prepare_export_attempt_failure(run, full_generation, sink, ordinal, outcome,
budget)` verifies all four bindings, resolves the exact detailed receipt it
derives from the ordinary issue, applies the frozen export threshold using the
status-owned dense attempt ordinal as the per-sink counter, and returns one move-only
`PreparedExportAttemptFailure`. Its private checked decision cannot be detached
from the verified `BudgetOwnedExportIssueReceipt`. The latter owns compact
exact-encoded bytes under one lease and separately charges its parsed fixed
DTO, `PreparedExportAttemptFailure` plus checked-decision inline storage,
compact sink/code storage, and embedded full detailed receipt;
integer-only accessors pin both charges. Foreign bindings, a missing retained
policy/generation authority, or either allocation refusal returns a typed
`StreamingReliabilityError` before sink transition construction.
The persisted receipt derives `Serialize` but not `Deserialize` and embeds the
full persisted detailed receipt plus its digest/length, frozen policy digest,
and counter-before/after. Export failures never mutate the immutable final
generation or its restored issue ledger. The derived status transaction writes
the exact receipt object first, then atomically CASes status bytes containing
`DerivedExportReceiptReference` (outer and embedded digest/length); an orphan
before CAS is unreachable, while success makes the complete receipt reachable.
Reopen needs only the leased final generation and derived status store. It
builds `DurableExportReceiptValidationContext` from the generation's reachable
reliability participant policy digest, the exact sink, and prior status ordinal/
counter, then calls reliability-owned `restore_durable_export_issue_receipt`.
The strict decoder verifies outer and embedded digest/length, run, full
generation, sink, attempt ordinal, issue ID, policy digest, scope/order, and
checked counter transition. It reconstructs the deterministic per-sink counter
from status plus receipt; no mutable checkpoint ledger lookup occurs. Tampered,
missing, unreachable, foreign, or noncontiguous content never creates live
receipt authority.

Frozen rules define retry counts and disposition transitions, for example
`Retry -> Hole`, `Retry -> Quarantine`, `Retry -> TerminalActionReceipt`, or
`Retry -> ExportIncomplete`. Thresholds may also select `Backpressure` or
return `needs_admission_fence` so the pipeline stops accepting new work,
commits the truthful prefix, and reports degraded completion. They never turn
an ordinary issue into `FailRun`.

## Handled cuts and quarantine tombstones

Foundation Task 1D-R extends `CheckpointCut` with one checked
`handled_issues: HandledIssueCut`. The cut contains the receipt-index root and a
bounded content-addressed root of each input domain's no-more-before frontier.
`HandledIssueCut` has private borrow-only fields; its strict manual
deserialization re-runs the checked constructor, and no public literal can
substitute either root.
The reliability participant state contains the same roots, sequencer pending
root, action pending/frontier root, quarantine-tombstone root, and counters.
Candidate construction requires equality between the cut, participant
descriptor, staged receipt partition, and P1B tombstone-root acknowledgement. A
`Hole` may advance only with its exact receipt; `Quarantine` may advance only
after the move-only `PreparedSessionQuarantineInstall` has checked P1B's
non-destructive tombstone view against the issue and the same tombstone root is
reachable from that barrier's generation. Because this
changes canonical cut encoding, Task
1D-R bumps the generation hash domain to
`aiperf.streaming.committed-checkpoint-generation.v4` and pins a new golden;
v3 generations remain bounded read-only restore inputs and cannot be mixed into
or followed by a v4 run. A private strict versioned wire decoder first enforces
the authored maximum bytes, rejects unknown fields/versions, and verifies the
version-selected hash domain (`v3` without handled issues, `v4` with all handled
roots). Valid v3 constructs private legacy decoder state; the backend open seam
exposes an explicit versioned leased authority distinguishing `CurrentV4` from
`LegacyV3ReadOnly`. `begin_generation` accepts only a sealed
`CurrentV4CheckpointGeneration` predecessor, so legacy authority is
unrepresentable as succession input in memory, local, layered, and object
backends. The versioned common reader exposes no participant-state method;
matching its legacy branch yields only a private-field
`LegacyParticipantState` with borrow-only descriptor/payload access and no
conversion to `CommittedParticipantState`. Task 1D-R replaces the landed public
storage constructor with crate-private `from_current_v4_reader`, which requires
a private current-v4 context binding run, full generation, and reachable
descriptor digest. Even legacy bytes copied into a newly budgeted payload cannot mint that
context. Legacy authority therefore supports
leased reads and export only, never participant initialization or successor
construction. A begin call with `expected = None` still checks the actual
per-run head and returns typed `LegacyReadOnlyHead` without mutation when v3 is
present; omission is not an overwrite path. Malformed current bytes are an
`ObjectVerification` refusal, never silently interpreted as v3.

P1B is the sole owner of the typed private, non-`Clone`, budget-owned
`SessionQuarantineTombstone` keyed by
`(StreamingInputDomainIdentity, StableSessionKey)`. It binds the run, issue ID,
landed `SessionCausalFrontier`, disposition, and a checked
`SessionQuarantineClosureProof`. The proof is constructed only from authored
close, hard watermark, finite seal, verified complete sorted run, or exhausted
frozen quarantine policy; partition EOF is not evidence. Every later fragment
for that key extends or matches the causal frontier and is excluded; it can never
recreate live session state. P1B retains the map and tombstones across chunks
and resume; neither is moved into the reporter. Foundation Task 1D-R's reporter
method `prepare_session_quarantine_install(view, issue_id, barrier, budget)`
consumes no P1B state: it resolves `issue_id` against its retained detailed
receipt, checks the P1B sealed borrowed view, and produces a separately
budgeted, move-only `PreparedSessionQuarantineInstall` bound to the exact
barrier, root, and receipt binding. P1B alone implements the
session-module-private sealed view trait through a borrow of its retained map;
callers cannot construct a detached entry slice. The acknowledgement stores
compact exact-owned payload bytes under one payload lease and separately
charges its run/barrier/root/receipt/view metadata under one view lease;
checked construction pins both counts and borrow-only accessors expose counts
without exposing either lease. Tasks 5E/6B verify and stage that acknowledgement beside the receipt
partition. Dropping it before CAS leaves P1B state intact for byte-identical
retry. A checked later-fragment extension updates the causal frontier,
invalidates the prior root/acknowledgement, and requires a newly prepared root
at the next barrier. Acceptance rechecks both root and monotonic view revision,
so a stale acknowledgement cannot become valid again through a digest replay.
Immediately before staging, 5E/6B call the reporter-owned
`verify_session_quarantine_install(prepared, current_view, barrier)` with a
fresh P1B borrow; it compares run, barrier, root, revision, receipt binding, and
payload digest without consuming either object.
The reporter cannot mark the quarantine handled until the
fresh acknowledgement and receipt root commit together. The tombstone survives checkpoint/resume and
is retained until a verified source no-more-before frontier proves no future
fragment for the session, a generation commits that final frontier and receipt,
and ordinary generation-reader retention no longer reaches the tombstone.

## Action-terminal ownership and restore

`ActiveExecution` has private fields and owns an optional
`BudgetOwnedActionTerminalReceipt`. The live `ActionTerminalReceipt` has private
fields, is non-`Clone`, implements `Serialize` but not `Deserialize`, and is
inseparable from compact encoded bytes plus the exact parsed-heap lease. Its
outcome is `Copy + Debug + Eq + PartialEq + Serialize`. A private strict wire
DTO may be restored only by a bounded function supplied the expected run,
action, terminal-membership context, and budget; checked construction rejects
foreign run/action replay and success/error membership collision before live
authority exists. `ActiveExecution::take_terminal_receipt` transfers the whole
wrapper intact to the results partition. No API exposes raw receipt bytes or a
lease independently.

## Derived-sink recovery authority

Derived status keys use the complete `CheckpointGeneration` `(epoch, digest)`,
not an epoch alone. Every configured sink starts as
`PendingAttempt { next_ordinal: 0 }` with no issue. A failed attempt advances by
exact status CAS to `PendingRetry` with one issue ID and next ordinal; exhaustion
becomes `Exhausted { last_attempt_ordinal, counter_before, .. }` with both
values derived by the checked transition from predecessor status, and
successful durable output becomes `Complete`. First-attempt and multi-retry
restart tests reopen exhausted status from only the leased final generation and
fresh derived store, seed validation from those independent status fields, and
then compare (never trust) the embedded receipt. Status
and receipt live objects are non-`Clone`, budget-owned, and never exposed apart
from their exact encoded-byte and parsed-allocation leases.

Absence is not interpreted as retry. Before pending-row paging, a private
recovery verifier walks retained generations under leases, reconciles each full
generation against the frozen sink inventory, and atomically creates missing
`PendingAttempt(0)` rows or verifies durable `Complete`. One clock-injected
bounded retry supervisor receives an attempt authority containing both the
leased full generation and its `LeasedGenerationReader`; neither may be
reconstructed from an epoch or dropped during an attempt. Task 6C1 owns a
private checked derived-status transition candidate keyed by `(run, full
generation, sink, ordinal)`. Its typed persisted export receipt binds the
verified issue ID and full reachable receipt authority while separately
charging exact encoded bytes and parsed allocation. The candidate alone permits
absent → `PendingAttempt(0)`, pending(n) → `PendingRetry(n+1, issue)` or
`Exhausted(issue)` with matching receipt attempt/ID, and pending → `Complete`
with a sealed durable-output proof. It rejects ordinal overflow; `Complete` and
`Exhausted` have no successor. Store CAS accepts only this candidate, and load
revalidates receipt reachability and content. The store and transition authority
are crate-private, so an extension cannot fabricate a candidate. This is derived authority only and never changes execution
generation CAS. Status charge covers both exact encoded bytes and all parsed
heap/inline retained allocation. Crash tests cover generation-before-initial-row,
receipt-before-status-CAS, status-CAS-before-retry, and durable-output-before-
`Complete` CAS.

## Refreshable credential authority

HF receives one injected redacted `HfCredentialProvider`; Task A0 owns the sole
shared redacted `AwsCredentialProviderAuthority` consumed by S3 and object
checkpointing. Both refresh through bounded host-`Clock` backoff and never put
credential bytes in debug output, errors, checkpoints, provenance, or receipts.
An HF authentication retry rebuilds the request for the exact frozen commit,
shard, and immutable object and never re-resolves a symbolic revision.
Exhaustion may become `Hole` only while immutable identity is unchanged. If no
authorized immutable source can be acquired without substituting identity, the
host validates source-authority/frozen-semantic drift and fails the run.

## Fault and disposition matrix

| Fault | Scope / class | Default progression | Owning task | Required observation |
|---|---|---|---|---|
| HF shard fetch timeout or S3 ranged-read timeout | partition / retryable | retry, then hole | A3 / A6 | later immutable partitions continue; hole receipt is checkpointed |
| HF/S3 object identity, revision, ETag, length, or digest changes after freeze | run / invariant | fail-run | A3 / A6 | no bytes decoded under the changed semantic snapshot |
| JSONL syntax/schema/oversize datum | record / permanent | quarantine | A2 | next record decodes; quarantined record never enters logical metrics |
| Baseten invalid row isolated to one trace | record or session / permanent | quarantine | A4 | remaining row groups/sessions continue with stable membership |
| Baseten schema/projection drift after preparation | run / invariant | fail-run | A4 | no mixed-schema generation is published |
| missing predecessor local to one recoverable session | session / permanent | quarantine after authored wait/retry policy | P1B | all owned state and leases for that session retire deterministically |
| conflicting content for one stable record/session/action identity | run / invariant | fail-run | 1D-R / P1 / P2 | no ambiguous membership or state is committed |
| endpoint timeout, HTTP/gRPC terminal error, or invalid terminal payload | action / retryable or permanent | bounded retry, then terminal-action-receipt disposition | P2 / P4 | exactly one logical terminal membership and error metric; later actions issue |
| endpoint terminal-failure threshold reached | action / permanent | terminal-action-receipt plus optional admission fence | P2 / P3 | no fail-run; truthful partial/final degraded status is available |
| checkpoint participant/object write or sync transient | checkpoint-attempt / retryable | retry | 5C / 5E / 5F2 | prior generation remains authoritative; execution state is not rolled back past it |
| checkpoint budget unavailable | checkpoint-attempt / capacity | backpressure/retry | 5E | no partial generation and no accounting loss |
| stale writer, foreign run/proof, or CAS expectation mismatch | checkpoint-attempt / invariant | fail-run | 5B / 5E / 5F2 | no notification or state mutation after refusal |
| impossible watermark regression, truthful cut, or membership accounting | run / invariant | fail-run | 7A / 5E / 6B | no fabricated generation or result root |
| compaction read/write/sync transient | export / retryable | retry | 6C1 | committed generation remains reconstructable and execution is never rolled back |
| exporter/report persistence transient | export / retryable | retry | 6D | durable pending sink status retains the generation lease |
| exporter becomes permanently unavailable | export / permanent | export-incomplete with exhausted sink status | 6D | authoritative native generation remains readable; product reports incomplete export |
| cellular authentication, placement digest, ownership epoch, or release proof mismatch | run / invariant | fail-run | C1-C5 | existing fail-closed no-early-issue behavior remains |

Authorization or capability refusal found before the run remains a startup
error rather than an issue receipt. The table governs faults after a logical run
and frozen plan exist.

## Post-commit sink status

Compaction, report persistence, and optional exporters are derived sinks, not
checkpoint authority. Tasks 6C1 and 6D own a durable status keyed by
`(run, full CheckpointGeneration, sink_id)`. Its initial state is
`PendingAttempt { next_ordinal: 0 }` with no issue; later states are
`PendingRetry { next_ordinal, last_issue_id }`, `Complete`, and
`Exhausted { last_issue_id, last_attempt_ordinal, counter_before }`.
The exhausted ordinal and predecessor counter are checked status-authored
authority derived from the prior state, never inferred from receipt bytes;
they independently seed strict receipt validation after restart.
Pending/exhausted live status retains the
inseparable typed receipt carrying both that ID and its full verified authority;
it is non-`Clone` and owns separate exact encoded/parsed leases. A private checked recovery verifier,
never absence alone, creates the initial state. A bounded restartable supervisor
pages pending statuses only after retained-generation/sink reconciliation and
retains both the generation lease and leased reader until durable success or
checked exhaustion. `PendingRetry`/`Exhausted` publication is one derived-status
transaction whose private checked candidate inserts the inseparable verified
export receipt object and CASes the status pointer atomically. No raw
`next + Option<receipt>` API exists. No transition can change the committed
generation or resume cut.

## Task and dependency amendment

The serialized contract foundation is now:

```text
0 -> 1A -> 1B -> 5A -> 1C -> 5A-R -> 5B -> 1D -> 1D-R -> 1E
```

Task 1D-R lands before every durable backend, adapter, action host, checkpoint
coordinator, result epoch/compactor, and executable workload. In addition to
neutral policy/ledger vocabulary and its conformance suite, it owns the
`checkpoint_backend.rs`/memory retrofit that makes current-v4 versus legacy-v3
leased authority explicit and prevents v3 succession by type. Downstream tasks own their
stage-specific construction of `OrdinaryStreamingIssue` or typed host-classifier
input, cleanup after a
disposition, and fault-injection tests. Product tasks own Config-v2 policy,
public diagnostics/status, real-binary continuation matrices, and soak gates.

## Required RED/GREEN proof

| Plan/task | RED proof | GREEN proof |
|---|---|---|
| Foundation 1D-R | privacy/serde attempts forge terminal or action/gap authority; mismatched action/sequence and unproved gap; ambiguous policy rules reorder matches; reverse/skew/one-versus-eight arrival changes thresholds; moved/stale tombstone acknowledgement; backend v3 erased into predecessor; detailed receipts escape budget | sealed exhaustive classifier and action facts, exact-before-wildcard policy, v2 receipt golden, checkpointed per-domain sequencer, non-destructive budgeted receipt/tombstone views, explicit versioned leased open, restore-once counters, v4 handled cut |
| Adapters A2/A3/A4/A6 | first malformed record/shard aborts source | next valid unit continues; hole/quarantine receipt survives checkpoint/resume |
| Pipeline P1B/P2/P3/P4 | quarantined session resurrects from a later chunk; terminal action receipt crosses run/action or collides with success; endpoint threshold depends on success/reset ordering | durable causal tombstone, private checked receipt, cumulative deterministic threshold, later issue, optional safe admission fence |
| Checkpoint 5E/6B/6C1/6D | handled frontier lacks same-generation receipt/tombstone acknowledgement; pre-CAS cancellation loses receipt or P1B tombstone; absent sink status fabricates retry; arbitrary/terminal/overflow transition or tampered export receipt; crashes strand pending sink work | receipt/cut/root equality, retained retry views, sealed checked sink transitions, verified receipt reachability, bounded restart supervisor, generation stays reconstructable |
| Product V1/V3/V4/V5/V6 | absent/forged policy, ordinary-fault abort, security mismatch continuation | strict config, complete fault matrix, real-binary continuation, bounded soak, terminal-boundary audit |

Every task observes its intended RED before production changes and runs one
focused GREEN suite. Product V3 owns the cross-backend fault matrix; V4 owns
real-binary source/format/endpoint/export continuation; V6 records exact evidence
for every terminal-boundary row.

## Non-goals

- No best-effort weakening of digest, length, schema, provenance, or CAS checks.
- No adapter-owned retry loop or free-form error string as policy authority.
- No wall-clock or worker-race threshold.
- No unbounded in-memory issue log.
- No rollback of an authoritative generation because a derived sink failed.
- No promise that quarantined or holed input contributes successful request
  metrics; the issue receipt and degraded completeness counters are the truth.
