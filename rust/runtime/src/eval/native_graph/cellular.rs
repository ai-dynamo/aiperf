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
        ArtifactDigest, AttemptId, EpisodeAssignmentId, MatrixError, ResolvedNativeGraphSuite,
        ResourceLeaseRequest, ResourceLimits,
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
