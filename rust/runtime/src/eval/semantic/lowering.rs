// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Narrow, typed semantic lowering that refuses unsupported operations.

use std::error::Error;
use std::fmt::{self, Display};

use crate::graph::model::GraphTraceProgram;

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

/// Immutable source bytes and declared execution selection for one graph lowerer.
///
/// The selected factory owns any source-family-specific bindings. This request
/// deliberately carries bytes rather than a caller-owned path so lowering never
/// reopens a mutable package origin.
#[derive(Clone, Copy, Debug)]
pub struct GraphLoweringRequest<'a> {
    /// Versioned source grammar selected by package import.
    pub source_schema: &'a str,
    /// Execution profile selected by package import.
    pub execution_profile: &'a str,
    /// Exact immutable source bytes retained by package import.
    pub source: &'a [u8],
}

/// Capabilities declared by one source-to-Graph-IR lowering factory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphLowererCapabilities {
    source_schemas: Vec<String>,
    execution_profiles: Vec<String>,
}

impl GraphLowererCapabilities {
    /// Construct one deterministic capability inventory.
    pub fn new(
        source_schemas: impl IntoIterator<Item = String>,
        execution_profiles: impl IntoIterator<Item = String>,
    ) -> Self {
        let mut source_schemas = source_schemas.into_iter().collect::<Vec<_>>();
        source_schemas.sort();
        source_schemas.dedup();
        let mut execution_profiles = execution_profiles.into_iter().collect::<Vec<_>>();
        execution_profiles.sort();
        execution_profiles.dedup();
        Self {
            source_schemas,
            execution_profiles,
        }
    }

    /// Reports whether the factory accepts a source schema identifier.
    pub fn supports_source_schema(&self, source_schema: &str) -> bool {
        self.source_schemas
            .iter()
            .any(|supported| supported == source_schema)
    }

    /// Reports whether the factory accepts an execution-profile identifier.
    pub fn supports_execution_profile(&self, execution_profile: &str) -> bool {
        self.execution_profiles
            .iter()
            .any(|supported| supported == execution_profile)
    }

    /// Borrows supported source schemas in deterministic order.
    pub fn source_schemas(&self) -> &[String] {
        &self.source_schemas
    }

    /// Borrows supported execution profiles in deterministic order.
    pub fn execution_profiles(&self) -> &[String] {
        &self.execution_profiles
    }
}

/// Uniform object-safe refusal returned by registered graph lowerers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphLoweringError(String);

impl GraphLoweringError {
    /// Build a source-lowering refusal with its source-family context retained.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for GraphLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for GraphLoweringError {}

/// Object-safe source-to-existing-Graph-IR lowering seam.
///
/// Factories are selected and frozen by the runtime extension layer. They may
/// retain source-family bindings, but every successful call yields the shared
/// [`GraphTraceProgram`] rather than a parallel execution representation.
pub trait GraphLowererFactory: Send + Sync {
    /// Return the source grammars and execution profiles this factory accepts.
    fn capabilities(&self) -> GraphLowererCapabilities;

    /// Lower exact immutable source bytes into the existing trace program type.
    fn lower(
        &self,
        request: GraphLoweringRequest<'_>,
    ) -> Result<GraphTraceProgram, GraphLoweringError>;
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
