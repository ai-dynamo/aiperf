// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Fallible source-preserving semantic graph lowering and paired baselines.

mod comparison;
mod lowering;

pub use comparison::{PairedComparisonError, PairedComparisonSpec};
pub use lowering::{
    FidelityError, FidelityOutcome, SemanticGraph, SemanticNode, lower_semantic_graph,
};
