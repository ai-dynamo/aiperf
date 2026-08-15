// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Isolated declared-artifact and reward contracts for Harbor verifiers.

mod artifacts;
mod regrade;
mod reward;

pub use artifacts::{ArtifactTransferError, DeclaredArtifactTransfer};
pub use regrade::{RegradeError, RegradeRequest, VerifierResult, regrade};
pub use reward::{RewardDocument, RewardError};
