// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The three category boundary traits a plugin implements.
//!
//! Each factory does the same two things: it names itself, and it turns authored
//! configuration into an opaque prepared value plus a receipt. Validation happens
//! once, at startup, before any request is issued — so a misconfigured plugin
//! fails a plan rather than a phase.
//!
//! Authored configuration crosses as canonical JSON bytes rather than a decoded
//! value. The host does not know the factory's configuration schema and must not
//! need a shared serde version to hand it over; the factory decodes with
//! whatever it links. An absent `config` key is [`EMPTY_AUTHORED_CONFIG`], not a
//! missing value, so `{id}` and `{id, config: {}}` are the same plan.

use core::fmt::{self, Display, Formatter};

use crate::capture::ExporterCaptureRequirementsV1;
use crate::id::RegistryId;
use crate::prepared::{PreparedEndpoint, PreparedExporter, PreparedTransport};
use crate::transport::TransportExecutionShapeV1;
use crate::validation::{PluginCategory, ValidationError};

/// The authored configuration a plan supplies when it names no `config` key.
pub const EMPTY_AUTHORED_CONFIG: &[u8] = b"{}";

/// One factory's authored configuration, as canonical JSON bytes.
#[derive(Debug, Clone, Copy)]
pub struct AuthoredConfigV1<'a> {
    id: &'a RegistryId,
    json: &'a [u8],
}

impl<'a> AuthoredConfigV1<'a> {
    /// Bind authored bytes to the factory identifier they were authored under.
    pub const fn new(id: &'a RegistryId, json: &'a [u8]) -> Self {
        Self { id, json }
    }

    /// Bind the empty configuration for a plan entry that named no `config`.
    pub const fn empty(id: &'a RegistryId) -> Self {
        Self {
            id,
            json: EMPTY_AUTHORED_CONFIG,
        }
    }

    /// The factory identifier the configuration was authored under.
    pub const fn id(&self) -> &'a RegistryId {
        self.id
    }

    /// The canonical JSON bytes.
    pub const fn json(&self) -> &'a [u8] {
        self.json
    }

    /// Whether the plan authored no fields.
    pub fn is_empty_object(&self) -> bool {
        self.json == EMPTY_AUTHORED_CONFIG
    }
}

/// Why a category factory refused.
///
/// Every variant names the category so a host aggregating failures across
/// categories reports which position refused without a second lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoryError {
    /// The authored configuration is not valid for this factory.
    InvalidConfiguration {
        /// The refusing category.
        category: PluginCategory,
        /// What was wrong with the configuration.
        reason: String,
    },
    /// The plan asked for a capability this factory does not have.
    UnsupportedCapability {
        /// The refusing category.
        category: PluginCategory,
        /// The capability that was asked for.
        capability: String,
    },
    /// A boundary value was refused before the factory ran.
    Validation(ValidationError),
    /// The factory refused at run time for an implementation-specific reason.
    Runtime {
        /// The refusing category.
        category: PluginCategory,
        /// Why it refused.
        reason: String,
    },
}

impl CategoryError {
    /// The category that refused, or `None` for a pre-factory validation error.
    pub const fn category(&self) -> Option<PluginCategory> {
        match self {
            Self::InvalidConfiguration { category, .. }
            | Self::UnsupportedCapability { category, .. }
            | Self::Runtime { category, .. } => Some(*category),
            Self::Validation(_) => None,
        }
    }
}

impl Display for CategoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { category, reason } => {
                write!(formatter, "{category} configuration rejected: {reason}")
            }
            Self::UnsupportedCapability {
                category,
                capability,
            } => write!(
                formatter,
                "{category} factory does not support {capability:?}"
            ),
            Self::Validation(error) => Display::fmt(error, formatter),
            Self::Runtime { category, reason } => {
                write!(formatter, "{category} factory refused: {reason}")
            }
        }
    }
}

impl core::error::Error for CategoryError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ValidationError> for CategoryError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

/// The result of asking one category factory to validate a plan entry.
///
/// A refusal is a value rather than an `Err` at the aggregate level: the host
/// collects every category's outcome and reports all of them, instead of
/// stopping at the first factory that objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoryOutcome<T> {
    /// The factory accepted and produced a prepared value.
    Accepted(T),
    /// The factory refused.
    Refused(CategoryError),
}

impl<T> CategoryOutcome<T> {
    /// Whether the factory accepted.
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted(_))
    }

    /// Convert into a `Result`, discarding the outcome shape.
    pub fn into_result(self) -> Result<T, CategoryError> {
        match self {
            Self::Accepted(value) => Ok(value),
            Self::Refused(error) => Err(error),
        }
    }
}

impl<T> From<Result<T, CategoryError>> for CategoryOutcome<T> {
    fn from(result: Result<T, CategoryError>) -> Self {
        match result {
            Ok(value) => Self::Accepted(value),
            Err(error) => Self::Refused(error),
        }
    }
}

/// A factory that validates and prepares one endpoint dialect.
pub trait EndpointFactory {
    /// The registered identifier this factory answers to.
    fn id(&self) -> &RegistryId;

    /// Validate authored configuration and prepare the endpoint.
    fn validate(&self, config: AuthoredConfigV1<'_>) -> Result<PreparedEndpoint, CategoryError>;
}

/// A factory that validates and prepares one transport.
pub trait TransportFactory {
    /// The registered identifier this factory answers to.
    fn id(&self) -> &RegistryId;

    /// The single execution shape this transport occupies.
    ///
    /// A transport that could answer with both shapes would be placed at two
    /// incompatible points in the run's control flow; the return type is a
    /// single value precisely so that cannot be expressed.
    fn execution_shape(&self) -> TransportExecutionShapeV1;

    /// Validate authored configuration and prepare the transport.
    fn validate(&self, config: AuthoredConfigV1<'_>) -> Result<PreparedTransport, CategoryError>;
}

/// A factory that validates and prepares one exporter.
pub trait ExporterFactory {
    /// The registered identifier this factory answers to.
    fn id(&self) -> &RegistryId;

    /// The captures this exporter needs, declared before the run starts.
    ///
    /// The host reads this to decide whether to pay for exact-record retention
    /// at all, so it must be answerable without validated configuration.
    fn capture_requirements(&self) -> ExporterCaptureRequirementsV1;

    /// Validate authored configuration and prepare the exporter.
    fn validate(&self, config: AuthoredConfigV1<'_>) -> Result<PreparedExporter, CategoryError>;
}
