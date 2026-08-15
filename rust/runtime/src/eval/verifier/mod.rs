// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Isolated declared-artifact and reward contracts for Harbor verifiers.

mod artifacts;
mod execution;
mod regrade;
mod reward;

pub use artifacts::{ArtifactTransferError, DeclaredArtifactTransfer};
pub use execution::{
    VerifierExecutionError, VerifierMode, VerifierSandboxFactory, prepare_verifier,
};
pub use regrade::{RegradeError, RegradeRequest, VerifierResult, regrade};
pub use reward::{
    RewardDocument, RewardError, RewardParseOutcome, invalid_reward_evidence,
    parse_reward_with_evidence,
};
