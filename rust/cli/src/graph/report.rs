// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Stable fatal-error vocabulary for native graph commands.

use serde::{Deserialize, Serialize};

/// Native graph operation selected by the caller.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphOperation {
    /// Validate one graph input.
    Validate,
    /// Explain one graph input.
    Explain,
    /// Visualize one graph input.
    Visualize,
}

impl GraphOperation {
    /// Return the stable operation name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Validate => "validate",
            Self::Explain => "explain",
            Self::Visualize => "visualize",
        }
    }
}

/// Stable class of a graph-command fatal error.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphCommandErrorCode {
    /// Clap rejected the public argument shape.
    InvalidArguments,
    /// The local source does not exist.
    SourceNotFound,
    /// The source is not a local path.
    SourceNotLocal,
    /// The requested adapter format is unavailable.
    FormatUnsupported,
    /// The tokenizer is neither built in nor local.
    TokenizerUnsupported,
    /// The local tokenizer could not be loaded.
    TokenizerLoadFailed,
    /// The adapter could not decode the source.
    InputDecodeFailed,
    /// The adapter could not lower the source.
    InputLoweringFailed,
    /// A selected trace does not exist.
    TraceNotFound,
    /// The output target is invalid.
    OutputInvalid,
    /// The output target could not be written.
    OutputWriteFailed,
}

impl GraphCommandErrorCode {
    /// Return the stable kebab-case code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArguments => "invalid-arguments",
            Self::SourceNotFound => "source-not-found",
            Self::SourceNotLocal => "source-not-local",
            Self::FormatUnsupported => "format-unsupported",
            Self::TokenizerUnsupported => "tokenizer-unsupported",
            Self::TokenizerLoadFailed => "tokenizer-load-failed",
            Self::InputDecodeFailed => "input-decode-failed",
            Self::InputLoweringFailed => "input-lowering-failed",
            Self::TraceNotFound => "trace-not-found",
            Self::OutputInvalid => "output-invalid",
            Self::OutputWriteFailed => "output-write-failed",
        }
    }
}

/// Versioned JSON envelope for expected graph-command failures.
#[derive(Debug, Deserialize, Serialize)]
pub struct GraphErrorReport {
    /// Schema identifier.
    pub schema_version: String,
    /// Selected operation.
    pub operation: GraphOperation,
    /// Stable error class.
    pub code: GraphCommandErrorCode,
    /// Bounded, content-safe message.
    pub message: String,
    /// Canonical local source if it was reached.
    pub source: Option<String>,
}

/// An expected graph-command failure retained until dispatcher-owned rendering.
#[derive(Debug)]
pub struct GraphCommandError {
    /// Stable error class.
    pub code: GraphCommandErrorCode,
    /// Bounded, content-safe message.
    pub message: String,
    /// Canonical local source if it was reached.
    pub source: Option<String>,
}

impl GraphCommandError {
    /// Build an expected command failure.
    pub fn new(
        code: GraphCommandErrorCode,
        message: impl Into<String>,
        source: Option<String>,
    ) -> Self {
        Self {
            code,
            message: bound_message(message.into()),
            source,
        }
    }

    /// Convert to the public JSON envelope.
    pub fn report(&self, operation: GraphOperation) -> GraphErrorReport {
        GraphErrorReport {
            schema_version: "aiperf.graph.error.v1".to_owned(),
            operation,
            code: self.code,
            message: self.message.clone(),
            source: self.source.clone(),
        }
    }
}

/// Restrict public messages to 1024 Unicode scalar values.
pub fn bound_message(message: String) -> String {
    const MAX_SCALARS: usize = 1024;
    if message.chars().count() <= MAX_SCALARS {
        return message;
    }
    let prefix: String = message.chars().take(MAX_SCALARS - 1).collect();
    format!("{prefix}…")
}
