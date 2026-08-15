// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Error types for the dataflow runtime.

use crate::graph::channel_store::StoreError;

/// A trace-terminating error: it aborts the trace's remaining node firings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    /// A channel orphaned (a producer failed and its readers can no longer be
    /// satisfied), or another channel-store error.
    Store(StoreError),
    /// A configured client cancellation terminated one node and therefore the trace.
    Cancelled(String),
    /// A graph contains a node kind this executor does not implement yet.
    UnsupportedNode {
        /// Stable graph node identifier.
        node_id: String,
        /// Serialized node-kind label.
        kind: &'static str,
    },
    /// A trace program selected a driver unavailable at this execution boundary.
    UnsupportedDriver(String),
    /// Any other structural error (e.g. an unsupported graph topology or a cycle).
    Other(String),
}

impl TraceError {
    /// A short, stable classification of the error kind.
    pub fn kind(&self) -> &'static str {
        match self {
            TraceError::Store(StoreError::Orphaned { .. }) => "orphan",
            TraceError::Store(_) => "store",
            TraceError::Cancelled(_) => "cancelled",
            TraceError::UnsupportedNode { .. } => "unsupported_node",
            TraceError::UnsupportedDriver(_) => "unsupported_driver",
            TraceError::Other(_) => "other",
        }
    }
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::Store(e) => write!(f, "{e}"),
            TraceError::Cancelled(message) => f.write_str(message),
            TraceError::UnsupportedNode { node_id, kind } => {
                write!(f, "graph node {node_id:?} has unsupported kind {kind:?}")
            }
            TraceError::UnsupportedDriver(kind) => {
                write!(f, "graph trace program has unsupported driver {kind:?}")
            }
            TraceError::Other(m) => write!(f, "{m}"),
        }
    }
}
impl std::error::Error for TraceError {}

impl From<StoreError> for TraceError {
    fn from(e: StoreError) -> Self {
        TraceError::Store(e)
    }
}
