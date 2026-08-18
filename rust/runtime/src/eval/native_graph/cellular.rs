// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned NativeGraph cellular placement and result folding.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Display, Formatter},
    rc::Rc,
};

use uuid::Uuid;

use super::matrix::ResourceCapacityLedger;
use crate::{
    cellular::{CellPartitionError, ModuloCellPartition},
    eval::{
        ArtifactDigest, AttemptId, EpisodeAssignmentId, EpisodeComparability, EpisodeExecution,
        EpisodeIntegrity, EpisodeResult, MatrixError, NativeGraphCompletedAttempt,
        ResolvedNativeGraphSuite, ResourceLeaseRequest, ResourceLimits, append_identity_field,
    },
};

/// Immutable controller-minted ownership for one resolved suite assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphCellPlacement {
    output_index: usize,
    cell_id: u32,
    assignment_id: EpisodeAssignmentId,
    attempt_id: AttemptId,
}

impl NativeGraphCellPlacement {
    /// Returns the canonical suite output position for this placement.
    pub const fn output_index(&self) -> usize {
        self.output_index
    }

    /// Returns the controller-minted owning cell.
    pub const fn cell_id(&self) -> u32 {
        self.cell_id
    }

    /// Borrows the immutable assignment identity accepted by the fold.
    pub fn assignment_id(&self) -> &EpisodeAssignmentId {
        &self.assignment_id
    }

    /// Borrows the deterministic attempt identity for this placement.
    pub fn attempt_id(&self) -> &AttemptId {
        &self.attempt_id
    }
}

/// Immutable controller plan assigning a resolved suite to modulo-owned cells.
#[derive(Clone, Debug)]
pub struct NativeGraphCellularPlan {
    cell_count: u32,
    suite_digest: ArtifactDigest,
    placements: Vec<NativeGraphCellPlacement>,
    placement_by_assignment: BTreeMap<ArtifactDigest, ExpectedPlacement>,
}

impl NativeGraphCellularPlan {
    /// Mints stable modulo-cell placements for every resolved suite assignment.
    pub fn from_suite(
        suite: &ResolvedNativeGraphSuite,
        cell_count: u32,
    ) -> Result<Self, CellularFoldError> {
        ModuloCellPartition::new(0, cell_count).map_err(CellularFoldError::InvalidPartition)?;

        let mut placements = Vec::with_capacity(suite.trials().len());
        let mut placement_by_assignment = BTreeMap::new();
        for (output_index, trial) in suite.trials().iter().enumerate() {
            let output_index_u64 = u64::try_from(output_index)
                .map_err(|_| CellularFoldError::OutputIndexNotRepresentable { output_index })?;
            let cell_id = u32::try_from(output_index_u64 % u64::from(cell_count))
                .map_err(|_| CellularFoldError::OutputIndexNotRepresentable { output_index })?;
            let assignment = trial.assignment_id().digest().clone();
            let expected = ExpectedPlacement {
                output_index,
                cell_id,
                task_digest: trial.imported().task.digest.clone(),
                trial_digest: trial.trial_digest().clone(),
                resources: trial.resource_handle(),
            };
            if placement_by_assignment
                .insert(assignment.clone(), expected)
                .is_some()
            {
                return Err(CellularFoldError::DuplicateControllerAssignment { assignment });
            }
            placements.push(NativeGraphCellPlacement {
                output_index,
                cell_id,
                assignment_id: trial.assignment_id().clone(),
                attempt_id: trial.attempt_id().clone(),
            });
        }

        Ok(Self {
            cell_count,
            suite_digest: suite.suite_digest().clone(),
            placements,
            placement_by_assignment,
        })
    }

    /// Returns the fixed number of cells covered by this plan.
    pub const fn cell_count(&self) -> u32 {
        self.cell_count
    }

    /// Borrows placements in canonical suite output order.
    pub fn placements(&self) -> &[NativeGraphCellPlacement] {
        &self.placements
    }

    /// Returns the immutable digest covering this exact suite-to-cell projection.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-cellular-plan-v1",
        );
        append_identity_field(
            &mut material,
            "suite",
            self.suite_digest.as_str().as_bytes(),
        );
        append_identity_field(&mut material, "cell-count", &self.cell_count.to_le_bytes());
        for placement in &self.placements {
            append_identity_field(
                &mut material,
                "assignment",
                placement.assignment_id.as_str().as_bytes(),
            );
            append_identity_field(
                &mut material,
                "attempt",
                placement.attempt_id.as_str().as_bytes(),
            );
            append_identity_field(
                &mut material,
                "output-index",
                &(placement.output_index as u64).to_le_bytes(),
            );
            append_identity_field(&mut material, "cell-id", &placement.cell_id.to_le_bytes());
        }
        ArtifactDigest::from_bytes(&material)
    }

    /// Starts an empty keyed fold that accepts exactly this plan's receipts.
    pub fn begin_fold<T>(&self) -> NativeGraphCellularFold<T> {
        NativeGraphCellularFold {
            expected_by_assignment: self.placement_by_assignment.clone(),
            assignment_by_output: self
                .placements
                .iter()
                .map(|placement| placement.assignment_id.digest().clone())
                .collect(),
            values: std::iter::repeat_with(|| None)
                .take(self.placements.len())
                .collect(),
        }
    }
}

/// Controller-owned fold for one typed receipt per planned assignment.
#[derive(Debug)]
pub struct NativeGraphCellularFold<T> {
    expected_by_assignment: BTreeMap<ArtifactDigest, ExpectedPlacement>,
    assignment_by_output: Vec<ArtifactDigest>,
    values: Vec<Option<T>>,
}

impl<T> NativeGraphCellularFold<T> {
    /// Accepts one receipt only when its assignment and submitting cell match the plan.
    pub fn accept(
        &mut self,
        cell_id: u32,
        assignment_id: &EpisodeAssignmentId,
        value: T,
    ) -> Result<(), CellularFoldError> {
        let assignment = assignment_id.digest().clone();
        let Some(expected) = self.expected_by_assignment.get(&assignment) else {
            return Err(CellularFoldError::UnknownAssignment { assignment });
        };
        if expected.cell_id != cell_id {
            return Err(CellularFoldError::WrongCell {
                assignment,
                expected: expected.cell_id,
                actual: cell_id,
            });
        }
        let Some(slot) = self.values.get_mut(expected.output_index) else {
            return Err(CellularFoldError::MissingControllerPlacement { assignment });
        };
        if slot.is_some() {
            return Err(CellularFoldError::DuplicateAssignment { assignment });
        }
        *slot = Some(value);
        Ok(())
    }

    /// Finishes only a complete fold, returning receipts in canonical output order.
    pub fn finish(self) -> Result<Vec<T>, CellularFoldError> {
        let mut ordered = Vec::with_capacity(self.values.len());
        for (output_index, value) in self.values.into_iter().enumerate() {
            let Some(value) = value else {
                return Err(CellularFoldError::MissingAssignment {
                    assignment: self.assignment_by_output[output_index].clone(),
                });
            };
            ordered.push(value);
        }
        Ok(ordered)
    }
}

#[derive(Clone, Debug)]
struct ExpectedPlacement {
    output_index: usize,
    cell_id: u32,
    task_digest: ArtifactDigest,
    trial_digest: ArtifactDigest,
    resources: Rc<ResourceLeaseRequest>,
}

/// Opaque controller-minted identity for one capacity-backed cellular lease.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NativeGraphCellLeaseId(Uuid);

/// One controller-issued, nonreusable capacity lease for a planned assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphCellLease {
    id: NativeGraphCellLeaseId,
    assignment_id: EpisodeAssignmentId,
    attempt_id: AttemptId,
    cell_id: u32,
}

impl NativeGraphCellLease {
    /// Borrows the immutable assignment bound by this lease.
    pub fn assignment_id(&self) -> &EpisodeAssignmentId {
        &self.assignment_id
    }

    /// Borrows the deterministic attempt identity bound by this lease.
    pub fn attempt_id(&self) -> &AttemptId {
        &self.attempt_id
    }

    /// Returns the sole controller-minted cell that may complete this lease.
    pub const fn cell_id(&self) -> u32 {
        self.cell_id
    }
}

/// Immutable limits selected by the controller for every cellular result receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeGraphCellularReceiptLimits {
    max_receipt_bytes: usize,
    max_attempt_id_bytes: usize,
    max_evidence_digests: usize,
}

impl NativeGraphCellularReceiptLimits {
    /// Creates positive, finite bounds before any cell assignment is issued.
    pub fn new(
        max_receipt_bytes: usize,
        max_attempt_id_bytes: usize,
        max_evidence_digests: usize,
    ) -> Result<Self, NativeGraphCellularReceiptError> {
        if max_receipt_bytes == 0 {
            return Err(NativeGraphCellularReceiptError::InvalidLimit {
                field: "max_receipt_bytes",
            });
        }
        if max_attempt_id_bytes == 0 {
            return Err(NativeGraphCellularReceiptError::InvalidLimit {
                field: "max_attempt_id_bytes",
            });
        }
        if max_evidence_digests == 0 {
            return Err(NativeGraphCellularReceiptError::InvalidLimit {
                field: "max_evidence_digests",
            });
        }
        Ok(Self {
            max_receipt_bytes,
            max_attempt_id_bytes,
            max_evidence_digests,
        })
    }

    /// Returns the maximum retained receipt payload size.
    pub const fn max_receipt_bytes(&self) -> usize {
        self.max_receipt_bytes
    }

    /// Returns the maximum retained attempt identifier size.
    pub const fn max_attempt_id_bytes(&self) -> usize {
        self.max_attempt_id_bytes
    }

    /// Returns the maximum immutable evidence identities retained by one receipt.
    pub const fn max_evidence_digests(&self) -> usize {
        self.max_evidence_digests
    }
}

/// Controller-minted bounded assignment facts a cell must preserve until completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphCellAssignment {
    plan_digest: ArtifactDigest,
    task_digest: ArtifactDigest,
    output_index: usize,
    cell_id: u32,
    grant: ArtifactDigest,
    assignment_id: EpisodeAssignmentId,
    trial_digest: ArtifactDigest,
    attempt_id: AttemptId,
}

impl NativeGraphCellAssignment {
    /// Borrows the controller plan identity that authorizes this placement.
    pub fn plan_digest(&self) -> &ArtifactDigest {
        &self.plan_digest
    }

    /// Borrows the imported task identity this assignment may execute.
    pub fn task_digest(&self) -> &ArtifactDigest {
        &self.task_digest
    }

    /// Returns the controller-owned output position for this result.
    pub const fn output_index(&self) -> usize {
        self.output_index
    }

    /// Returns the only cell authorized to complete this assignment.
    pub const fn cell_id(&self) -> u32 {
        self.cell_id
    }

    /// Borrows the nonreusable controller-minted completion grant.
    pub fn grant(&self) -> &ArtifactDigest {
        &self.grant
    }

    /// Borrows the immutable suite-assignment identity.
    pub fn assignment_id(&self) -> &EpisodeAssignmentId {
        &self.assignment_id
    }

    /// Borrows the immutable resolved trial identity.
    pub fn trial_digest(&self) -> &ArtifactDigest {
        &self.trial_digest
    }

    /// Borrows the deterministic execution-attempt identity.
    pub fn attempt_id(&self) -> &AttemptId {
        &self.attempt_id
    }
}

/// Bounded, digest-only projection of one sealed completed NativeGraph attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphCellResultReceipt {
    assignment: NativeGraphCellAssignment,
    result: EpisodeResult,
    completed_attempt_digest: ArtifactDigest,
}

impl NativeGraphCellResultReceipt {
    /// Projects a sealed completion and scored result into one controller-validated receipt.
    pub fn from_completed(
        assignment: &NativeGraphCellAssignment,
        completed: &NativeGraphCompletedAttempt,
        result: EpisodeResult,
        limits: &NativeGraphCellularReceiptLimits,
    ) -> Result<Self, NativeGraphCellularReceiptError> {
        if completed.frozen_attempt().trial_digest() != assignment.trial_digest() {
            return Err(NativeGraphCellularReceiptError::CompletedTrialMismatch {
                expected: assignment.trial_digest().clone(),
                actual: completed.frozen_attempt().trial_digest().clone(),
            });
        }
        if completed.frozen_attempt().attempt() != assignment.attempt_id() {
            return Err(NativeGraphCellularReceiptError::CompletedAttemptMismatch {
                expected: assignment.attempt_id().clone(),
                actual: completed.frozen_attempt().attempt().clone(),
            });
        }
        if result.trial_digest() != assignment.trial_digest() {
            return Err(NativeGraphCellularReceiptError::ResultTrialMismatch {
                expected: assignment.trial_digest().clone(),
                actual: result.trial_digest().clone(),
            });
        }
        if result.attempt_id() != assignment.attempt_id() {
            return Err(NativeGraphCellularReceiptError::ResultAttemptMismatch {
                expected: assignment.attempt_id().clone(),
                actual: result.attempt_id().clone(),
            });
        }
        let receipt = Self {
            assignment: assignment.clone(),
            completed_attempt_digest: completed.frozen_attempt().identity_digest(),
            result,
        };
        receipt.validate(limits)?;
        Ok(receipt)
    }

    /// Borrows the controller-minted assignment this receipt completes.
    pub fn assignment(&self) -> &NativeGraphCellAssignment {
        &self.assignment
    }

    /// Borrows the independently scored, immutable episode result facts.
    pub fn result(&self) -> &EpisodeResult {
        &self.result
    }

    /// Borrows the sealed Task 9 completed-attempt identity required in result evidence.
    pub fn completed_attempt_digest(&self) -> &ArtifactDigest {
        &self.completed_attempt_digest
    }

    /// Computes the canonical identity over every retained receipt fact.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut material = Vec::new();
        append_identity_field(
            &mut material,
            "domain",
            b"aiperf-native-graph-cell-result-receipt-v1",
        );
        append_identity_field(
            &mut material,
            "plan",
            self.assignment.plan_digest.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "task",
            self.assignment.task_digest.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "output-index",
            &(self.assignment.output_index as u64).to_le_bytes(),
        );
        append_identity_field(
            &mut material,
            "cell-id",
            &self.assignment.cell_id.to_le_bytes(),
        );
        append_identity_field(
            &mut material,
            "grant",
            self.assignment.grant.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "assignment",
            self.assignment.assignment_id.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "trial",
            self.assignment.trial_digest.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "attempt",
            self.assignment.attempt_id.as_str().as_bytes(),
        );
        append_identity_field(
            &mut material,
            "integrity",
            integrity_name(self.result.integrity()).as_bytes(),
        );
        append_identity_field(
            &mut material,
            "execution",
            execution_name(self.result.execution()).as_bytes(),
        );
        append_identity_field(
            &mut material,
            "comparability",
            comparability_name(self.result.comparability()).as_bytes(),
        );
        match self.result.verified_reward() {
            Some(reward) => {
                append_identity_field(&mut material, "reward", &reward.to_bits().to_le_bytes())
            }
            None => append_identity_field(&mut material, "reward-unavailable", b"1"),
        }
        append_identity_field(
            &mut material,
            "completed-attempt",
            self.completed_attempt_digest.as_str().as_bytes(),
        );
        for evidence in self.result.evidence() {
            append_identity_field(
                &mut material,
                "result-evidence",
                evidence.as_str().as_bytes(),
            );
        }
        ArtifactDigest::from_bytes(&material)
    }

    fn validate(
        &self,
        limits: &NativeGraphCellularReceiptLimits,
    ) -> Result<(), NativeGraphCellularReceiptError> {
        let attempt_bytes = self.assignment.attempt_id.as_str().len();
        if attempt_bytes > limits.max_attempt_id_bytes {
            return Err(NativeGraphCellularReceiptError::AttemptIdLimitExceeded {
                actual: attempt_bytes,
                limit: limits.max_attempt_id_bytes,
            });
        }
        if self.result.evidence().len() > limits.max_evidence_digests {
            return Err(NativeGraphCellularReceiptError::EvidenceLimitExceeded {
                actual: self.result.evidence().len(),
                limit: limits.max_evidence_digests,
            });
        }
        if !self
            .result
            .evidence()
            .contains(&self.completed_attempt_digest)
        {
            return Err(NativeGraphCellularReceiptError::MissingCompletedAttemptEvidence);
        }
        if self
            .result
            .verified_reward()
            .is_some_and(|reward| !reward.is_finite())
        {
            return Err(NativeGraphCellularReceiptError::NonFiniteReward);
        }
        let size = self.encoded_size_bytes()?;
        if size > limits.max_receipt_bytes {
            return Err(NativeGraphCellularReceiptError::ReceiptByteLimitExceeded {
                actual: size,
                limit: limits.max_receipt_bytes,
            });
        }
        Ok(())
    }

    fn encoded_size_bytes(&self) -> Result<usize, NativeGraphCellularReceiptError> {
        let fixed = [
            self.assignment.plan_digest.as_str().len(),
            self.assignment.task_digest.as_str().len(),
            self.assignment.grant.as_str().len(),
            self.assignment.assignment_id.as_str().len(),
            self.assignment.trial_digest.as_str().len(),
            self.completed_attempt_digest.as_str().len(),
            self.assignment.attempt_id.as_str().len(),
            std::mem::size_of::<u64>(),
            std::mem::size_of::<u32>(),
            std::mem::size_of::<u64>(),
        ];
        fixed
            .into_iter()
            .chain(
                self.result
                    .evidence()
                    .iter()
                    .map(|digest| digest.as_str().len()),
            )
            .try_fold(0usize, |total, bytes| {
                total
                    .checked_add(bytes)
                    .ok_or(NativeGraphCellularReceiptError::ReceiptSizeOverflow)
            })
    }
}

/// Controller-owned cellular completion boundary over sealed attempts and scored results.
pub struct NativeGraphCellResultAuthority {
    plan: NativeGraphCellularPlan,
    plan_digest: ArtifactDigest,
    limits: NativeGraphCellularReceiptLimits,
    leases: NativeGraphCellLeaseAuthority<EpisodeResult>,
    issued: BTreeMap<ArtifactDigest, NativeGraphCellLease>,
    settled: BTreeSet<ArtifactDigest>,
}

impl NativeGraphCellResultAuthority {
    /// Preflights capacities and fixes the receipt limits before a cell receives work.
    pub fn new(
        plan: NativeGraphCellularPlan,
        resources: ResourceLimits,
        limits: NativeGraphCellularReceiptLimits,
    ) -> Result<Self, NativeGraphCellularReceiptError> {
        let plan_digest = plan.identity_digest();
        let leases = NativeGraphCellLeaseAuthority::new(plan.clone(), resources)
            .map_err(NativeGraphCellularReceiptError::Lease)?;
        Ok(Self {
            plan,
            plan_digest,
            limits,
            leases,
            issued: BTreeMap::new(),
            settled: BTreeSet::new(),
        })
    }

    /// Issues one exact assignment with a nonreusable completion grant for `cell_id`.
    pub fn issue_for_cell(
        &mut self,
        cell_id: u32,
    ) -> Result<Option<NativeGraphCellAssignment>, NativeGraphCellularReceiptError> {
        let Some(lease) = self
            .leases
            .issue_for_cell(cell_id)
            .map_err(NativeGraphCellularReceiptError::Lease)?
        else {
            return Ok(None);
        };
        let grant = self.next_grant();
        let assignment = self.assignment_for_lease(&lease, grant.clone())?;
        self.issued.insert(grant, lease);
        Ok(Some(assignment))
    }

    /// Validates and folds one result only from its controller-minted cell and grant.
    pub fn complete_from_cell(
        &mut self,
        cell_id: u32,
        receipt: NativeGraphCellResultReceipt,
    ) -> Result<(), NativeGraphCellularReceiptError> {
        receipt.validate(&self.limits)?;
        let grant = receipt.assignment.grant.clone();
        let lease = match self.issued.get(&grant) {
            Some(lease) => lease.clone(),
            None if self.settled.contains(&grant) => {
                return Err(NativeGraphCellularReceiptError::ReplayedGrant { grant });
            }
            None => return Err(NativeGraphCellularReceiptError::UnknownGrant { grant }),
        };
        let expected = self.assignment_for_lease(&lease, grant.clone())?;
        self.validate_assignment(cell_id, &expected, receipt.assignment())?;
        self.leases
            .complete_from_cell(cell_id, lease, receipt.result)
            .map_err(NativeGraphCellularReceiptError::Lease)?;
        self.issued.remove(&grant);
        self.settled.insert(grant);
        Ok(())
    }

    /// Aborts every issued receipt grant and returns every underlying resource lease.
    pub fn abort_all(&mut self) {
        self.leases.abort_all();
        self.settled.extend(self.issued.keys().cloned());
        self.issued.clear();
    }

    /// Returns results in canonical suite order only after every issued grant has settled.
    pub fn finish(self) -> Result<Vec<EpisodeResult>, NativeGraphCellularReceiptError> {
        if !self.issued.is_empty() {
            return Err(NativeGraphCellularReceiptError::OutstandingGrants {
                count: self.issued.len(),
            });
        }
        self.leases
            .finish()
            .map_err(NativeGraphCellularReceiptError::Lease)
    }

    fn assignment_for_lease(
        &self,
        lease: &NativeGraphCellLease,
        grant: ArtifactDigest,
    ) -> Result<NativeGraphCellAssignment, NativeGraphCellularReceiptError> {
        let assignment = lease.assignment_id.digest();
        let expected = self
            .plan
            .placement_by_assignment
            .get(assignment)
            .ok_or_else(
                || NativeGraphCellularReceiptError::MissingControllerPlacement {
                    assignment: assignment.clone(),
                },
            )?;
        Ok(NativeGraphCellAssignment {
            plan_digest: self.plan_digest.clone(),
            task_digest: expected.task_digest.clone(),
            output_index: expected.output_index,
            cell_id: expected.cell_id,
            grant,
            assignment_id: lease.assignment_id.clone(),
            trial_digest: expected.trial_digest.clone(),
            attempt_id: lease.attempt_id.clone(),
        })
    }

    fn validate_assignment(
        &self,
        actual_cell: u32,
        expected: &NativeGraphCellAssignment,
        actual: &NativeGraphCellAssignment,
    ) -> Result<(), NativeGraphCellularReceiptError> {
        if actual.plan_digest != self.plan_digest {
            return Err(NativeGraphCellularReceiptError::PlanMismatch);
        }
        if actual_cell != expected.cell_id || actual.cell_id != expected.cell_id {
            return Err(NativeGraphCellularReceiptError::WrongCell {
                expected: expected.cell_id,
                actual: actual_cell,
            });
        }
        if actual.task_digest != expected.task_digest {
            return Err(NativeGraphCellularReceiptError::TaskIdentityMismatch);
        }
        if actual.output_index != expected.output_index {
            return Err(NativeGraphCellularReceiptError::OutputIndexMismatch);
        }
        if actual.assignment_id != expected.assignment_id {
            return Err(NativeGraphCellularReceiptError::AssignmentIdentityMismatch);
        }
        if actual.trial_digest != expected.trial_digest {
            return Err(NativeGraphCellularReceiptError::TrialIdentityMismatch);
        }
        if actual.attempt_id != expected.attempt_id {
            return Err(NativeGraphCellularReceiptError::AttemptIdentityMismatch);
        }
        Ok(())
    }

    fn next_grant(&self) -> ArtifactDigest {
        loop {
            let grant = ArtifactDigest::from_bytes(Uuid::new_v4().as_bytes());
            if !self.issued.contains_key(&grant) && !self.settled.contains(&grant) {
                return grant;
            }
        }
    }
}

/// Controller receipt issuance or terminal validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeGraphCellularReceiptError {
    /// A controller-selected immutable limit was zero.
    InvalidLimit {
        /// The invalid limit field.
        field: &'static str,
    },
    /// A sealed completion belonged to a different resolved trial.
    CompletedTrialMismatch {
        /// Controller-authorized resolved trial identity.
        expected: ArtifactDigest,
        /// Sealed completion's resolved trial identity.
        actual: ArtifactDigest,
    },
    /// A sealed completion belonged to a different execution attempt.
    CompletedAttemptMismatch {
        /// Controller-authorized execution attempt identity.
        expected: AttemptId,
        /// Sealed completion's execution attempt identity.
        actual: AttemptId,
    },
    /// A scored result belonged to a different resolved trial.
    ResultTrialMismatch {
        /// Controller-authorized resolved trial identity.
        expected: ArtifactDigest,
        /// Result's resolved trial identity.
        actual: ArtifactDigest,
    },
    /// A scored result belonged to a different execution attempt.
    ResultAttemptMismatch {
        /// Controller-authorized execution attempt identity.
        expected: AttemptId,
        /// Result's execution attempt identity.
        actual: AttemptId,
    },
    /// An attempted identity exceeded the controller-selected retained-size bound.
    AttemptIdLimitExceeded {
        /// Actual UTF-8 length.
        actual: usize,
        /// Immutable controller-selected maximum.
        limit: usize,
    },
    /// A result contained more evidence identities than the controller authorized.
    EvidenceLimitExceeded {
        /// Actual number of evidence identities.
        actual: usize,
        /// Immutable controller-selected maximum.
        limit: usize,
    },
    /// A result omitted the sealed completed-attempt identity from its evidence.
    MissingCompletedAttemptEvidence,
    /// A result attempted to retain a non-finite verifier reward.
    NonFiniteReward,
    /// A receipt's bounded retained identity projection was too large.
    ReceiptByteLimitExceeded {
        /// Actual retained identity byte count.
        actual: usize,
        /// Immutable controller-selected maximum.
        limit: usize,
    },
    /// Receipt byte accounting overflowed before a receipt was retained.
    ReceiptSizeOverflow,
    /// A receipt named no controller-issued completion grant.
    UnknownGrant {
        /// Unrecognized grant identity.
        grant: ArtifactDigest,
    },
    /// A terminal operation attempted to reuse a consumed completion grant.
    ReplayedGrant {
        /// Already consumed grant identity.
        grant: ArtifactDigest,
    },
    /// A receipt was minted for a different controller plan.
    PlanMismatch,
    /// A receipt arrived from a cell other than the controller-minted owner.
    WrongCell {
        /// Controller-minted owner cell.
        expected: u32,
        /// Actual submitting cell.
        actual: u32,
    },
    /// A receipt named a different imported task.
    TaskIdentityMismatch,
    /// A receipt named a different controller output position.
    OutputIndexMismatch,
    /// A receipt named a different suite assignment.
    AssignmentIdentityMismatch,
    /// A receipt named a different resolved trial.
    TrialIdentityMismatch,
    /// A receipt named a different deterministic attempt.
    AttemptIdentityMismatch,
    /// An internally issued lease no longer had a trusted controller placement.
    MissingControllerPlacement {
        /// Assignment missing from controller plan state.
        assignment: ArtifactDigest,
    },
    /// Finalization was requested while controller-minted grants remained active.
    OutstandingGrants {
        /// Number of outstanding grants.
        count: usize,
    },
    /// The underlying controller lease authority rejected an operation.
    Lease(NativeGraphCellLeaseError),
}

impl Display for NativeGraphCellularReceiptError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit { field } => {
                write!(
                    f,
                    "native graph cellular receipt limit {field} must be positive"
                )
            }
            Self::CompletedTrialMismatch { .. } => {
                f.write_str("sealed completion does not match the controller trial")
            }
            Self::CompletedAttemptMismatch { .. } => {
                f.write_str("sealed completion does not match the controller attempt")
            }
            Self::ResultTrialMismatch { .. } => {
                f.write_str("scored result does not match the controller trial")
            }
            Self::ResultAttemptMismatch { .. } => {
                f.write_str("scored result does not match the controller attempt")
            }
            Self::AttemptIdLimitExceeded { actual, limit } => write!(
                f,
                "native graph cellular attempt identity is {actual} bytes, above limit {limit}"
            ),
            Self::EvidenceLimitExceeded { actual, limit } => write!(
                f,
                "native graph cellular receipt has {actual} evidence identities, above limit {limit}"
            ),
            Self::MissingCompletedAttemptEvidence => f.write_str(
                "native graph cellular receipt omitted its sealed completed-attempt identity",
            ),
            Self::NonFiniteReward => {
                f.write_str("native graph cellular receipt has a non-finite verifier reward")
            }
            Self::ReceiptByteLimitExceeded { actual, limit } => write!(
                f,
                "native graph cellular receipt retains {actual} bytes, above limit {limit}"
            ),
            Self::ReceiptSizeOverflow => {
                f.write_str("native graph cellular receipt byte accounting overflowed")
            }
            Self::UnknownGrant { .. } => {
                f.write_str("native graph cellular receipt has an unknown completion grant")
            }
            Self::ReplayedGrant { .. } => {
                f.write_str("native graph cellular receipt completion grant was already consumed")
            }
            Self::PlanMismatch => {
                f.write_str("native graph cellular receipt does not match the controller plan")
            }
            Self::WrongCell { expected, actual } => write!(
                f,
                "native graph cellular receipt belongs to cell {expected}, not cell {actual}"
            ),
            Self::TaskIdentityMismatch => {
                f.write_str("native graph cellular receipt does not match the imported task")
            }
            Self::OutputIndexMismatch => f.write_str(
                "native graph cellular receipt does not match the controller output position",
            ),
            Self::AssignmentIdentityMismatch => {
                f.write_str("native graph cellular receipt does not match the suite assignment")
            }
            Self::TrialIdentityMismatch => {
                f.write_str("native graph cellular receipt does not match the resolved trial")
            }
            Self::AttemptIdentityMismatch => {
                f.write_str("native graph cellular receipt does not match the execution attempt")
            }
            Self::MissingControllerPlacement { .. } => {
                f.write_str("native graph cellular controller plan has no issued placement")
            }
            Self::OutstandingGrants { count } => write!(
                f,
                "native graph cellular receipt authority cannot finish with {count} active grants"
            ),
            Self::Lease(error) => {
                write!(f, "native graph cellular receipt authority failed: {error}")
            }
        }
    }
}

impl std::error::Error for NativeGraphCellularReceiptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lease(error) => Some(error),
            _ => None,
        }
    }
}

fn integrity_name(integrity: EpisodeIntegrity) -> &'static str {
    match integrity {
        EpisodeIntegrity::Valid => "valid",
        EpisodeIntegrity::InvalidProvider => "invalid-provider",
        EpisodeIntegrity::InvalidRuntime => "invalid-runtime",
        EpisodeIntegrity::InvalidEvidence => "invalid-evidence",
    }
}

fn execution_name(execution: EpisodeExecution) -> &'static str {
    match execution {
        EpisodeExecution::Completed => "completed",
        EpisodeExecution::Failed => "failed",
        EpisodeExecution::Truncated => "truncated",
        EpisodeExecution::Cancelled => "cancelled",
    }
}

fn comparability_name(comparability: EpisodeComparability) -> &'static str {
    match comparability {
        EpisodeComparability::Scored => "scored",
        EpisodeComparability::Unscored => "unscored",
    }
}

/// Controller-owned issuance, completion, and cancellation authority for one typed fold.
pub struct NativeGraphCellLeaseAuthority<T> {
    cell_count: u32,
    placements: Vec<NativeGraphCellPlacement>,
    expected_by_assignment: BTreeMap<ArtifactDigest, ExpectedPlacement>,
    pending_assignments: BTreeSet<ArtifactDigest>,
    issued: BTreeMap<NativeGraphCellLeaseId, IssuedCellLease>,
    settled: BTreeSet<NativeGraphCellLeaseId>,
    ledger: ResourceCapacityLedger,
    fold: NativeGraphCellularFold<T>,
}

impl<T> NativeGraphCellLeaseAuthority<T> {
    /// Preflights every trusted plan request before issuing any cell lease.
    pub fn new(
        plan: NativeGraphCellularPlan,
        limits: ResourceLimits,
    ) -> Result<Self, NativeGraphCellLeaseError> {
        for placement in &plan.placements {
            let assignment = placement.assignment_id.digest();
            let expected = plan
                .placement_by_assignment
                .get(assignment)
                .ok_or_else(|| NativeGraphCellLeaseError::MissingControllerPlacement {
                    assignment: assignment.clone(),
                })?;
            ResourceCapacityLedger::validate_request(
                &limits,
                expected.output_index,
                expected.resources.as_ref(),
            )
            .map_err(|source| NativeGraphCellLeaseError::InvalidResourceRequest {
                output_index: expected.output_index,
                source,
            })?;
        }
        let pending_assignments = plan
            .placements
            .iter()
            .map(|placement| placement.assignment_id.digest().clone())
            .collect();
        let fold = plan.begin_fold();

        Ok(Self {
            cell_count: plan.cell_count,
            placements: plan.placements,
            expected_by_assignment: plan.placement_by_assignment,
            pending_assignments,
            issued: BTreeMap::new(),
            settled: BTreeSet::new(),
            ledger: ResourceCapacityLedger::from(&limits),
            fold,
        })
    }

    /// Issues the next canonical placement for `cell_id` when global capacity permits it.
    pub fn issue_for_cell(
        &mut self,
        cell_id: u32,
    ) -> Result<Option<NativeGraphCellLease>, NativeGraphCellLeaseError> {
        if cell_id >= self.cell_count {
            return Err(NativeGraphCellLeaseError::UnknownCell { cell_id });
        }
        let Some(placement) = self.placements.iter().find_map(|placement| {
            let assignment = placement.assignment_id.digest();
            (placement.cell_id == cell_id && self.pending_assignments.contains(assignment))
                .then(|| placement.clone())
        }) else {
            return Ok(None);
        };
        let assignment = placement.assignment_id.digest().clone();
        let expected = self
            .expected_by_assignment
            .get(&assignment)
            .ok_or_else(|| NativeGraphCellLeaseError::MissingControllerPlacement {
                assignment: assignment.clone(),
            })?;
        if !self.ledger.try_reserve(expected.resources.as_ref()) {
            return Ok(None);
        }

        let id = self.next_lease_id();
        let lease = NativeGraphCellLease {
            id: id.clone(),
            assignment_id: placement.assignment_id,
            attempt_id: placement.attempt_id,
            cell_id,
        };
        let removed = self.pending_assignments.remove(&assignment);
        debug_assert!(
            removed,
            "selected placement must remain pending until issuance"
        );
        self.issued.insert(
            id,
            IssuedCellLease {
                assignment,
                attempt_id: lease.attempt_id.clone(),
                cell_id,
                resources: expected.resources.clone(),
            },
        );
        Ok(Some(lease))
    }

    /// Folds a completion only from the issuing cell, then returns its capacity exactly once.
    pub fn complete_from_cell(
        &mut self,
        cell_id: u32,
        lease: NativeGraphCellLease,
        value: T,
    ) -> Result<(), NativeGraphCellLeaseError> {
        let issued = self.issued_lease(&lease)?;
        if issued.cell_id != cell_id {
            return Err(NativeGraphCellLeaseError::WrongCell {
                lease: lease.id,
                expected: issued.cell_id,
                actual: cell_id,
            });
        }
        self.fold
            .accept(cell_id, &lease.assignment_id, value)
            .map_err(NativeGraphCellLeaseError::Fold)?;
        let issued = self.issued.remove(&lease.id).ok_or_else(|| {
            NativeGraphCellLeaseError::UnknownLease {
                lease: lease.id.clone(),
            }
        })?;
        self.ledger.release(issued.resources.as_ref());
        self.settled.insert(lease.id);
        Ok(())
    }

    /// Cancels one issued lease and returns its capacity without folding a receipt.
    pub fn abort(&mut self, lease: NativeGraphCellLease) -> Result<(), NativeGraphCellLeaseError> {
        self.issued_lease(&lease)?;
        let issued = self
            .issued
            .remove(&lease.id)
            .ok_or_else(|| self.lease_lookup_error(&lease.id))?;
        self.ledger.release(issued.resources.as_ref());
        self.settled.insert(lease.id);
        Ok(())
    }

    /// Cancels every currently issued lease, returning all controller-owned capacity.
    pub fn abort_all(&mut self) {
        for (id, issued) in std::mem::take(&mut self.issued) {
            self.ledger.release(issued.resources.as_ref());
            self.settled.insert(id);
        }
    }

    /// Returns canonical results only after no issued lease remains and the fold is complete.
    pub fn finish(self) -> Result<Vec<T>, NativeGraphCellLeaseError> {
        if !self.issued.is_empty() {
            return Err(NativeGraphCellLeaseError::OutstandingLeases {
                count: self.issued.len(),
            });
        }
        self.fold.finish().map_err(NativeGraphCellLeaseError::Fold)
    }

    fn next_lease_id(&self) -> NativeGraphCellLeaseId {
        loop {
            let id = NativeGraphCellLeaseId(Uuid::new_v4());
            if !self.issued.contains_key(&id) && !self.settled.contains(&id) {
                return id;
            }
        }
    }

    fn issued_lease(
        &self,
        lease: &NativeGraphCellLease,
    ) -> Result<&IssuedCellLease, NativeGraphCellLeaseError> {
        let issued = self
            .issued
            .get(&lease.id)
            .ok_or_else(|| self.lease_lookup_error(&lease.id))?;
        if &issued.assignment != lease.assignment_id.digest()
            || issued.attempt_id != lease.attempt_id
            || issued.cell_id != lease.cell_id
        {
            return Err(NativeGraphCellLeaseError::LeaseIdentityMismatch {
                lease: lease.id.clone(),
            });
        }
        Ok(issued)
    }

    fn lease_lookup_error(&self, lease: &NativeGraphCellLeaseId) -> NativeGraphCellLeaseError {
        if self.settled.contains(lease) {
            NativeGraphCellLeaseError::ReplayedLease {
                lease: lease.clone(),
            }
        } else {
            NativeGraphCellLeaseError::UnknownLease {
                lease: lease.clone(),
            }
        }
    }
}

#[derive(Clone, Debug)]
struct IssuedCellLease {
    assignment: ArtifactDigest,
    attempt_id: AttemptId,
    cell_id: u32,
    resources: Rc<ResourceLeaseRequest>,
}

/// Controller lease issuance or terminal-state validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeGraphCellLeaseError {
    /// A trusted controller plan had no placement state for an assignment.
    MissingControllerPlacement {
        /// The absent assignment identity.
        assignment: ArtifactDigest,
    },
    /// A trusted plan request exceeds the controller's global resource limits.
    InvalidResourceRequest {
        /// Canonical output position rejected before any lease is issued.
        output_index: usize,
        /// Underlying capacity validation failure.
        source: MatrixError,
    },
    /// A caller named no controller-minted cell.
    UnknownCell {
        /// The unknown cell identifier.
        cell_id: u32,
    },
    /// A completion or abort named no controller-issued lease.
    UnknownLease {
        /// The unrecognized opaque lease token.
        lease: NativeGraphCellLeaseId,
    },
    /// A terminal operation attempted to reuse a settled lease.
    ReplayedLease {
        /// The already settled opaque lease token.
        lease: NativeGraphCellLeaseId,
    },
    /// A lease was presented with facts other than those the controller issued.
    LeaseIdentityMismatch {
        /// The opaque token associated with the mismatched facts.
        lease: NativeGraphCellLeaseId,
    },
    /// A cell other than the issuing cell attempted to complete a lease.
    WrongCell {
        /// The opaque lease token.
        lease: NativeGraphCellLeaseId,
        /// The controller-minted owner.
        expected: u32,
        /// The reporting cell.
        actual: u32,
    },
    /// The plan fold rejected a terminal receipt.
    Fold(CellularFoldError),
    /// A caller requested final output while leases remain active.
    OutstandingLeases {
        /// Number of leases still holding controller capacity.
        count: usize,
    },
}

impl Display for NativeGraphCellLeaseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingControllerPlacement { assignment } => write!(
                f,
                "native graph cellular lease authority has no placement for assignment {}",
                assignment.as_str()
            ),
            Self::InvalidResourceRequest {
                output_index,
                source,
            } => write!(
                f,
                "native graph cellular placement at output index {output_index} is not admissible: {source}"
            ),
            Self::UnknownCell { cell_id } => {
                write!(
                    f,
                    "native graph cellular lease authority does not own cell {cell_id}"
                )
            }
            Self::UnknownLease { .. } => f.write_str("native graph cellular lease is unknown"),
            Self::ReplayedLease { .. } => {
                f.write_str("native graph cellular lease was already settled")
            }
            Self::LeaseIdentityMismatch { .. } => {
                f.write_str("native graph cellular lease facts do not match controller issuance")
            }
            Self::WrongCell {
                expected, actual, ..
            } => write!(
                f,
                "native graph cellular lease belongs to cell {expected}, not cell {actual}"
            ),
            Self::Fold(error) => write!(f, "native graph cellular lease receipt rejected: {error}"),
            Self::OutstandingLeases { count } => write!(
                f,
                "native graph cellular lease authority cannot finish with {count} active leases"
            ),
        }
    }
}

impl std::error::Error for NativeGraphCellLeaseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidResourceRequest { source, .. } => Some(source),
            Self::Fold(error) => Some(error),
            _ => None,
        }
    }
}

/// Controller plan or fold validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CellularFoldError {
    /// The requested cell partition cannot assign every output index exactly once.
    InvalidPartition(CellPartitionError),
    /// A platform output index cannot be represented by the modulo partition seam.
    OutputIndexNotRepresentable {
        /// The source suite output index.
        output_index: usize,
    },
    /// The resolved suite repeated an assignment identity before controller planning.
    DuplicateControllerAssignment {
        /// The duplicate assignment digest.
        assignment: ArtifactDigest,
    },
    /// A receipt named an assignment outside this controller plan.
    UnknownAssignment {
        /// The unrecognized assignment digest.
        assignment: ArtifactDigest,
    },
    /// A planned assignment was submitted by a different cell.
    WrongCell {
        /// The planned assignment digest.
        assignment: ArtifactDigest,
        /// The controller-minted owning cell.
        expected: u32,
        /// The cell that submitted the receipt.
        actual: u32,
    },
    /// A receipt attempted to overwrite a prior accepted assignment receipt.
    DuplicateAssignment {
        /// The already accepted assignment digest.
        assignment: ArtifactDigest,
    },
    /// A plan invariant lacked a materialized output slot.
    MissingControllerPlacement {
        /// The affected assignment digest.
        assignment: ArtifactDigest,
    },
    /// The fold was incomplete and cannot proceed to aggregation.
    MissingAssignment {
        /// The missing assignment digest.
        assignment: ArtifactDigest,
    },
}

impl Display for CellularFoldError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPartition(error) => write!(f, "invalid cellular partition: {error}"),
            Self::OutputIndexNotRepresentable { output_index } => write!(
                f,
                "native graph cellular output index {output_index} cannot be represented by the partition"
            ),
            Self::DuplicateControllerAssignment { assignment } => write!(
                f,
                "native graph cellular plan repeated assignment {}",
                assignment.as_str()
            ),
            Self::UnknownAssignment { assignment } => write!(
                f,
                "native graph cellular fold received unknown assignment {}",
                assignment.as_str()
            ),
            Self::WrongCell {
                assignment,
                expected,
                actual,
            } => write!(
                f,
                "native graph cellular assignment {} belongs to cell {expected}, not cell {actual}",
                assignment.as_str()
            ),
            Self::DuplicateAssignment { assignment } => write!(
                f,
                "native graph cellular fold received assignment {} more than once",
                assignment.as_str()
            ),
            Self::MissingControllerPlacement { assignment } => write!(
                f,
                "native graph cellular controller plan has no output slot for assignment {}",
                assignment.as_str()
            ),
            Self::MissingAssignment { assignment } => write!(
                f,
                "native graph cellular fold is missing assignment {}",
                assignment.as_str()
            ),
        }
    }
}

impl std::error::Error for CellularFoldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPartition(error) => Some(error),
            _ => None,
        }
    }
}
