// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-server library errors without an application-layer error dependency.

use std::fmt;

/// Error returned by content-server validation, preparation, or lifecycle work.
#[derive(Debug)]
#[non_exhaustive]
pub enum ContentServerError {
    /// Authored or derived configuration violates the server contract.
    InvalidConfiguration(String),
    /// A filesystem or socket operation failed.
    Io {
        /// Human-readable operation being attempted.
        operation: String,
        /// Underlying operating-system error.
        source: std::io::Error,
    },
    /// The supervised serving task could not be joined.
    Task(String),
    /// An advertised base URL could not be parsed.
    Url {
        /// Authored URL value.
        input: String,
        /// URL parser failure.
        source: url::ParseError,
    },
}

impl ContentServerError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration(message.into())
    }

    pub(crate) fn io(operation: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            operation: operation.into(),
            source,
        }
    }

    pub(crate) fn url(input: impl Into<String>, source: url::ParseError) -> Self {
        Self::Url {
            input: input.into(),
            source,
        }
    }
}

impl fmt::Display for ContentServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::Io { operation, source } => write!(formatter, "{operation}: {source}"),
            Self::Task(message) => write!(formatter, "content-server task failed: {message}"),
            Self::Url { input, source } => {
                write!(formatter, "invalid content-server URL {input:?}: {source}")
            }
        }
    }
}

impl std::error::Error for ContentServerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Url { source, .. } => Some(source),
            Self::InvalidConfiguration(_) | Self::Task(_) => None,
        }
    }
}

/// Content-server library result.
pub type Result<T> = std::result::Result<T, ContentServerError>;
