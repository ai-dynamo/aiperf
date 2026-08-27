// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed errors returned across the plugin source API boundary.
//!
//! The boundary is `panic = abort`, so no unwind may cross it. Every failure a
//! plugin or the host can recover from is therefore an explicit value in this
//! module rather than a panic. All error types are `Send + Sync + 'static` so a
//! host can move them off the registering thread and box them behind
//! [`std::error::Error`].

use std::fmt;

use crate::id::{REGISTRY_ID_MAX_LEN, RegistryId};

/// Why an authored identifier could not be normalized into a [`RegistryId`].
///
/// Variants map one-to-one onto the normalization-version-1 rules so a host can
/// report the exact rule an authored manifest violated instead of a generic
/// "invalid name".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryIdError {
    /// The requested normalization version is not implemented by this API
    /// generation. Only version `1` is supported.
    UnsupportedVersion {
        /// The version the caller asked for.
        requested: u8,
    },
    /// The authored input contained a non-ASCII byte. Normalization is defined
    /// over ASCII only, so non-ASCII input is rejected rather than transformed.
    NonAscii,
    /// The authored input was empty, or contained only ASCII space and tab and
    /// so normalized to the empty string.
    Empty,
    /// The normalized identifier exceeded [`REGISTRY_ID_MAX_LEN`] bytes.
    TooLong {
        /// The normalized length in bytes.
        len: usize,
    },
    /// The normalized identifier did not start with `[a-z0-9]`.
    InvalidStart {
        /// The offending leading character.
        character: char,
    },
    /// The normalized identifier contained a byte outside `[a-z0-9_]`.
    InvalidCharacter {
        /// The offending character.
        character: char,
        /// Byte offset of the offending character in the normalized string.
        offset: usize,
    },
    /// The authored input contained two adjacent separator bytes (`-` or `_`).
    /// Rejecting these before folding `-` to `_` keeps distinct authored
    /// spellings from collapsing onto one normalized identifier.
    ConsecutiveSeparators {
        /// Byte offset of the second separator in the trimmed authored input.
        offset: usize,
    },
}

impl fmt::Display for RegistryIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { requested } => write!(
                f,
                "unsupported registry-id normalization version {requested}; only version 1 exists"
            ),
            Self::NonAscii => {
                write!(f, "registry ids must be ASCII; the authored value was not")
            }
            Self::Empty => write!(f, "registry ids must not normalize to an empty string"),
            Self::TooLong { len } => write!(
                f,
                "normalized registry id is {len} bytes, exceeding the {REGISTRY_ID_MAX_LEN}-byte limit"
            ),
            Self::InvalidStart { character } => write!(
                f,
                "normalized registry id must start with [a-z0-9], found {character:?}"
            ),
            Self::InvalidCharacter { character, offset } => write!(
                f,
                "normalized registry id contains {character:?} at byte {offset}; only [a-z0-9_] is allowed"
            ),
            Self::ConsecutiveSeparators { offset } => write!(
                f,
                "authored registry id contains consecutive separators at byte {offset}"
            ),
        }
    }
}

impl std::error::Error for RegistryIdError {}

/// A recoverable failure raised by [`AIPerfExtension::register`].
///
/// Registration is the only plugin-authored call the host makes during the
/// entry transaction, so this is the one error a plugin returns rather than
/// aborting. The host rejects the whole package when it sees one.
///
/// [`AIPerfExtension::register`]: crate::extension::AIPerfExtension::register
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionError {
    registry_id: Option<RegistryId>,
    reason: Box<str>,
}

impl ExtensionError {
    /// Build a registration failure that is not attributable to one identifier.
    pub fn registration_failed(reason: impl Into<Box<str>>) -> Self {
        Self {
            registry_id: None,
            reason: reason.into(),
        }
    }

    /// Build a registration failure attributed to one normalized identifier.
    pub fn for_id(registry_id: RegistryId, reason: impl Into<Box<str>>) -> Self {
        Self {
            registry_id: Some(registry_id),
            reason: reason.into(),
        }
    }

    /// The identifier this failure is attributed to, when there is one.
    pub fn registry_id(&self) -> Option<&RegistryId> {
        self.registry_id.as_ref()
    }

    /// The human-readable reason the registration failed.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for ExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.registry_id {
            Some(id) => write!(f, "plugin registration failed for `{id}`: {}", self.reason),
            None => write!(f, "plugin registration failed: {}", self.reason),
        }
    }
}

impl std::error::Error for ExtensionError {}

impl From<RegistryIdError> for ExtensionError {
    fn from(error: RegistryIdError) -> Self {
        Self::registration_failed(error.to_string())
    }
}

/// Why a source API version string could not be parsed.
///
/// The boundary accepts only canonical `major.minor.patch` decimal triples;
/// equivalent non-canonical spellings such as `1.0` or `01.0.0` are rejected so
/// a version can be compared by value and reproduced byte-identically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceApiVersionError {
    /// The value did not have exactly three `.`-separated components.
    ComponentCount {
        /// The number of components found.
        found: usize,
    },
    /// A component was empty, non-decimal, redundantly zero-padded, or did not
    /// fit in `u16`.
    Component {
        /// Zero-based index of the offending component.
        index: usize,
    },
}

impl fmt::Display for SourceApiVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComponentCount { found } => write!(
                f,
                "source API version must have exactly 3 dot-separated components, found {found}"
            ),
            Self::Component { index } => write!(
                f,
                "source API version component {index} is not a canonical decimal u16"
            ),
        }
    }
}

impl std::error::Error for SourceApiVersionError {}
