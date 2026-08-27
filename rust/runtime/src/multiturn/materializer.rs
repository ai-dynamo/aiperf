// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local rebuild of identity-only routed credits.
//!
//! `--dispatch global-push` routes a credit carrying only identity and has the
//! receiving worker build the body. The concrete implementation owns a dataset
//! and a tokenizer; neither may reach the plugin boundary, so the boundary sees
//! only this trait.

use crate::multiturn::CreditIdentity;
use crate::transport::core::PreparedTurn;
use anyhow::Result;

/// Rebuilds one routed credit into a dispatchable turn.
///
/// Object-safe on purpose: the host owns the dataset and tokenizer behind the
/// implementation, and a transport plugin only ever calls through this trait.
pub trait CreditMaterializer {
    /// Rebuild the turn identified by `identity` from resident run state.
    fn materialize(&self, identity: CreditIdentity) -> Result<PreparedTurn>;
}

impl CreditMaterializer for super::WorkerMaterializer {
    fn materialize(&self, identity: CreditIdentity) -> Result<PreparedTurn> {
        let turn = super::WorkerMaterializer::materialize(self, &identity)?;
        Ok(PreparedTurn::from_turn(
            turn,
            &self.recipe.primary_model_name,
        ))
    }
}
