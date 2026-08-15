// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Narrow, typed semantic lowering that refuses unsupported operations.

/// Source-preserving native semantic graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticGraph {
    nodes: Vec<SemanticNode>,
}

impl SemanticGraph {
    /// Creates a graph with explicit source-semantic nodes.
    pub fn new(nodes: Vec<SemanticNode>) -> Result<Self, FidelityError> {
        if nodes.is_empty() {
            return Err(FidelityError::EmptyGraph);
        }
        Ok(Self { nodes })
    }
}

/// Semantic operation before lowering to executable Graph-IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticNode {
    /// A provider-backed language-model node.
    Llm,
    /// An explicit tool operation.
    Tool,
    /// An unsupported controller synchronization point.
    Barrier,
}

/// Declared outcome of attempted semantic lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FidelityOutcome {
    /// Every source node has a native executable equivalent.
    Exact,
    /// The selected semantic operation cannot be executed without substitution.
    Unsupported,
}

/// Typed semantic lowering refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FidelityError {
    /// Graphs require at least one semantic node.
    EmptyGraph,
    /// A semantic operation has no safe native lowering.
    Unsupported(FidelityOutcome),
}

impl FidelityError {
    /// Returns the explicit fidelity outcome for this refusal.
    pub const fn outcome(&self) -> FidelityOutcome {
        match self {
            Self::EmptyGraph | Self::Unsupported(_) => FidelityOutcome::Unsupported,
        }
    }
}

impl std::fmt::Display for FidelityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGraph => formatter.write_str("semantic graph must not be empty"),
            Self::Unsupported(_) => formatter.write_str("semantic graph contains unsupported node"),
        }
    }
}

impl std::error::Error for FidelityError {}

/// Verifies that every semantic node has an exact native lowering.
pub fn lower_semantic_graph(graph: &SemanticGraph) -> Result<FidelityOutcome, FidelityError> {
    if graph.nodes.iter().any(|node| matches!(node, SemanticNode::Barrier)) {
        return Err(FidelityError::Unsupported(FidelityOutcome::Unsupported));
    }
    Ok(FidelityOutcome::Exact)
}
