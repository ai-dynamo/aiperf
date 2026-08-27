// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral checkpoint budget admission shared by storage backends.
//!
//! Every checkpoint backend owns the same five budget kinds and must refuse
//! with the same stable vocabulary. The item-before-byte precedence encoded
//! here is the contract: a request that exceeds both dimensions reports the
//! item refusal, so two backends never disagree about which limit was hit.

use crate::streaming::{
    budget::{BudgetError, BudgetLease, BudgetLimits, BudgetSnapshot, StreamingResourceBudget},
    checkpoint::{
        CheckpointBackendBudgetFailureCode, CheckpointBackendBudgetKind, CheckpointError,
    },
};

/// One named backend budget with its configured limits.
#[derive(Clone, Debug)]
pub(crate) struct BackendBudget {
    kind: CheckpointBackendBudgetKind,
    limits: BudgetLimits,
    resource: StreamingResourceBudget,
}

impl BackendBudget {
    /// Validate one configured budget before any backend state is retained.
    pub(crate) fn new(
        kind: CheckpointBackendBudgetKind,
        limits: BudgetLimits,
    ) -> Result<Self, CheckpointError> {
        if limits.max_items == 0 {
            return Err(backend_error(
                kind,
                CheckpointBackendBudgetFailureCode::ItemCapacity,
            ));
        }
        if limits.max_bytes == 0 {
            return Err(backend_error(
                kind,
                CheckpointBackendBudgetFailureCode::ByteCapacity,
            ));
        }
        let resource = StreamingResourceBudget::new(limits)
            .map_err(|error| map_budget_error(kind, limits, 0, 0, error))?;
        Ok(Self {
            kind,
            limits,
            resource,
        })
    }

    /// Acquire one aggregate reservation, refusing capacity before waiting.
    pub(crate) async fn acquire(
        &self,
        items: usize,
        bytes: usize,
    ) -> Result<BudgetLease, CheckpointError> {
        if items > self.limits.max_items {
            return Err(backend_error(
                self.kind,
                CheckpointBackendBudgetFailureCode::ItemCapacity,
            ));
        }
        if bytes > self.limits.max_bytes {
            return Err(backend_error(
                self.kind,
                CheckpointBackendBudgetFailureCode::ByteCapacity,
            ));
        }
        self.resource
            .acquire(items, bytes)
            .await
            .map_err(|error| map_budget_error(self.kind, self.limits, items, bytes, error))
    }

    /// Borrow the configured limits.
    pub(crate) const fn limits(&self) -> BudgetLimits {
        self.limits
    }

    /// Snapshot current charges and high-water telemetry.
    pub(crate) fn snapshot(&self) -> BudgetSnapshot {
        self.resource.snapshot()
    }
}

/// Build one stable backend budget refusal.
pub(crate) const fn backend_error(
    budget: CheckpointBackendBudgetKind,
    code: CheckpointBackendBudgetFailureCode,
) -> CheckpointError {
    CheckpointError::BackendBudget { budget, code }
}

/// Map one resource-budget failure onto the stable checkpoint vocabulary.
pub(crate) fn map_budget_error(
    kind: CheckpointBackendBudgetKind,
    limits: BudgetLimits,
    items: usize,
    bytes: usize,
    error: BudgetError,
) -> CheckpointError {
    let code = match error {
        BudgetError::ZeroCapacity if limits.max_items == 0 => {
            CheckpointBackendBudgetFailureCode::ItemCapacity
        }
        BudgetError::ZeroCapacity => CheckpointBackendBudgetFailureCode::ByteCapacity,
        BudgetError::RequestExceedsCapacity if items > limits.max_items => {
            CheckpointBackendBudgetFailureCode::ItemCapacity
        }
        BudgetError::RequestExceedsCapacity if bytes > limits.max_bytes => {
            CheckpointBackendBudgetFailureCode::ByteCapacity
        }
        BudgetError::Closed => CheckpointBackendBudgetFailureCode::Closed,
        // Backend budgets use only async acquisition, which cannot return the
        // nonblocking-only capacity refusal.
        BudgetError::CapacityUnavailable => CheckpointBackendBudgetFailureCode::Unrepresentable,
        BudgetError::PermitCountTooLarge
        | BudgetError::AccountingOverflow
        | BudgetError::CannotGrowLease
        | BudgetError::InvalidFragmentItemCharge { .. }
        | BudgetError::ActionPayloadUndercharged { .. }
        | BudgetError::PartialLeasedBuffer { .. }
        | BudgetError::RequestExceedsCapacity => {
            CheckpointBackendBudgetFailureCode::Unrepresentable
        }
    };
    backend_error(kind, code)
}
