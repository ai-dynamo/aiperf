// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Narrow, typed semantic lowering that refuses unsupported operations.

/// An executable node emitted from an exactly lowerable semantic operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutableSemanticNode {
    /// A provider-backed language-model operation.
    Llm,
    /// A sandboxed tool operation.
    Tool,
}

/// An executable program that preserves the lowered source-node order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweredSemanticGraph {
    nodes: Vec<ExecutableSemanticNode>,
}

impl LoweredSemanticGraph {
    /// Returns the exact fidelity outcome for this executable program.
    pub const fn outcome(&self) -> FidelityOutcome {
        FidelityOutcome::Exact
    }

    /// Borrows the ordered executable nodes without exposing source-only nodes.
    pub fn nodes(&self) -> &[ExecutableSemanticNode] {
        &self.nodes
    }
}

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
    /// A source node prevented lowering at its preserved source position.
    UnsupportedNode {
        /// Zero-based index of the refusing source node.
        index: usize,
        /// Source operation that has no exact executable form.
        node: SemanticNode,
    },
}

impl FidelityError {
    /// Returns the explicit fidelity outcome for this refusal.
    pub const fn outcome(&self) -> FidelityOutcome {
        match self {
            Self::EmptyGraph | Self::Unsupported(_) | Self::UnsupportedNode { .. } => {
                FidelityOutcome::Unsupported
            }
        }
    }
}

impl std::fmt::Display for FidelityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGraph => formatter.write_str("semantic graph must not be empty"),
            Self::Unsupported(_) => formatter.write_str("semantic graph contains unsupported node"),
            Self::UnsupportedNode { index, node } => {
                write!(
                    formatter,
                    "semantic graph node {index} ({node:?}) has no exact lowering"
                )
            }
        }
    }
}

impl std::error::Error for FidelityError {}

/// Lowers every source node into an exact ordered executable program.
pub fn lower_semantic_graph(graph: &SemanticGraph) -> Result<LoweredSemanticGraph, FidelityError> {
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    for (index, node) in graph.nodes.iter().copied().enumerate() {
        let executable = match node {
            SemanticNode::Llm => ExecutableSemanticNode::Llm,
            SemanticNode::Tool => ExecutableSemanticNode::Tool,
            SemanticNode::Barrier => {
                return Err(FidelityError::UnsupportedNode { index, node });
            }
        };
        nodes.push(executable);
    }
    Ok(LoweredSemanticGraph { nodes })
}
