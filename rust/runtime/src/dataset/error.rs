// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for dataset parsing, composition, storage, and materialization.

use std::fmt::{self, Display};

use crate::dataset::segment::Handle;

/// Result type used throughout the dataset crate.
pub type Result<T> = std::result::Result<T, DatasetError>;

/// A load-time or materialization error with enough context to fix the input.
#[derive(Debug)]
pub enum DatasetError {
    /// A dense segment handle does not exist in the selected store.
    UnknownHandle(Handle),
    /// A segment was interned under a parent handle that does not exist.
    UnknownParent(Handle),
    /// More than `u32::MAX` unique segments were interned.
    SegmentCapacityExceeded,
    /// A payload was used in a materialization position that requires another kind.
    PayloadKind {
        /// Dense handle of the incompatible payload.
        handle: Handle,
        /// Human-readable expected payload kind.
        expected: &'static str,
        /// Human-readable actual payload kind.
        actual: &'static str,
    },
    /// A supposedly pre-serialized JSON slice is malformed for the requested operation.
    InvalidWire(String),
    /// A dataset contains a duplicate session identifier.
    DuplicateSession(String),
    /// A lookup referenced a session that is not present.
    UnknownSession(String),
    /// A sampler was constructed from no sampleable conversations.
    EmptySampler,
    /// A dynamic assembly program referenced an unresolved splice key.
    MissingSplice(String),
    /// A [`BodyPlan`](crate::body_plan::BodyPlan) reserved field slot was left
    /// unfilled at materialization, or a fill named a slot the plan never
    /// reserved.
    ReservedField(String),
    /// A tokenizer could not encode, decode, or initialize.
    Tokenizer(String),
    /// A parsed or composed dataset row violates its format contract.
    Validation(String),
    /// No registered loader recognized the source probe.
    LoaderNotFound(String),
    /// More than one registered loader recognized the source probe.
    AmbiguousLoader(Vec<String>),
    /// A filesystem operation failed while loading content.
    Io(std::io::Error),
    /// JSON decoding or encoding failed.
    Json(serde_json::Error),
    /// Endpoint formatting rejected a reconstructed request.
    Endpoint(crate::endpoints::EndpointError),
}

impl Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownHandle(handle) => write!(f, "unknown segment handle {handle}"),
            Self::UnknownParent(handle) => write!(f, "unknown parent segment handle {handle}"),
            Self::SegmentCapacityExceeded => {
                write!(f, "segment arena exceeded the u32 handle capacity")
            }
            Self::PayloadKind {
                handle,
                expected,
                actual,
            } => write!(
                f,
                "segment handle {handle} contains {actual}, expected {expected}"
            ),
            Self::InvalidWire(message) => write!(f, "invalid pre-serialized wire JSON: {message}"),
            Self::DuplicateSession(id) => write!(f, "duplicate dataset session id {id:?}"),
            Self::UnknownSession(id) => write!(f, "unknown dataset session id {id:?}"),
            Self::EmptySampler => write!(f, "dataset sampler requires at least one conversation"),
            Self::MissingSplice(key) => write!(f, "unresolved dynamic message splice {key:?}"),
            Self::ReservedField(message) => {
                write!(f, "invalid body-plan field reservation: {message}")
            }
            Self::Tokenizer(message) => write!(f, "tokenizer error: {message}"),
            Self::Validation(message) => write!(f, "invalid dataset: {message}"),
            Self::LoaderNotFound(source) => {
                write!(f, "no dataset loader recognizes {source}")
            }
            Self::AmbiguousLoader(names) => write!(
                f,
                "multiple dataset loaders recognize the source: {}",
                names.join(", ")
            ),
            Self::Io(error) => write!(f, "dataset I/O error: {error}"),
            Self::Json(error) => write!(f, "dataset JSON error: {error}"),
            Self::Endpoint(error) => write!(f, "dataset request formatting error: {error}"),
        }
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Endpoint(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DatasetError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for DatasetError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<crate::endpoints::EndpointError> for DatasetError {
    fn from(value: crate::endpoints::EndpointError) -> Self {
        Self::Endpoint(value)
    }
}
