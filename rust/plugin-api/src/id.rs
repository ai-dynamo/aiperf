// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalized registry identifiers.
//!
//! Every package name, category name, and capability name that crosses the
//! plugin boundary is a [`RegistryId`]: an owned, already-normalized string.
//! Normalization happens exactly once, at construction, so downstream lookup,
//! priority resolution, and alias handling compare bytes rather than re-running
//! a fold at every call site.
//!
//! # Normalization version 1
//!
//! Version 1 is the only version this API generation implements. Given an
//! authored value it:
//!
//! 1. rejects any non-ASCII input;
//! 2. trims ASCII space and tab from both ends;
//! 3. rejects two adjacent authored separator bytes (`-` or `_`);
//! 4. ASCII-lowercases the remainder;
//! 5. replaces each `-` byte with `_`;
//! 6. requires the result to match `^[a-z0-9][a-z0-9_]{0,127}$`.
//!
//! Step 3 runs on the authored spelling, before `-` folds to `_`, so
//! `a--b`, `a__b`, and `a-_b` are all rejected instead of collapsing onto the
//! single normalized identifier `a__b`. Authored spelling is display-only; only
//! the normalized form has identity.

use std::fmt;

use crate::error::RegistryIdError;

/// The only registry-id normalization version this API generation implements.
pub const REGISTRY_ID_NORMALIZATION_VERSION: u8 = 1;

/// Maximum length in bytes of a normalized registry identifier.
///
/// The grammar is `^[a-z0-9][a-z0-9_]{0,127}$`: one leading character plus at
/// most 127 more.
pub const REGISTRY_ID_MAX_LEN: usize = 128;

/// An owned, normalized identifier used for every name that crosses the plugin
/// boundary.
///
/// A `RegistryId` can only be produced through [`RegistryId::new`], so holding
/// one is proof the value already satisfies the version-1 grammar. Equality,
/// ordering, and hashing are over the normalized bytes.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegistryId(Box<str>);

impl RegistryId {
    /// Normalize `input` under normalization version `version`.
    ///
    /// Returns the exact rule that was violated on failure; see
    /// [`RegistryIdError`]. Only [`REGISTRY_ID_NORMALIZATION_VERSION`] is
    /// accepted, so a future version is a typed rejection rather than a silent
    /// fold under the wrong rules.
    pub fn new(input: &str, version: u8) -> Result<Self, RegistryIdError> {
        if version != REGISTRY_ID_NORMALIZATION_VERSION {
            return Err(RegistryIdError::UnsupportedVersion {
                requested: version,
            });
        }
        if !input.is_ascii() {
            return Err(RegistryIdError::NonAscii);
        }

        let trimmed = input.trim_matches(|c| c == ' ' || c == '\t');
        if trimmed.is_empty() {
            return Err(RegistryIdError::Empty);
        }

        // Consecutive-separator detection reads the authored spelling: after the
        // `-` to `_` fold the distinct authored forms would be indistinguishable.
        let bytes = trimmed.as_bytes();
        for offset in 1..bytes.len() {
            if is_separator(bytes[offset]) && is_separator(bytes[offset - 1]) {
                return Err(RegistryIdError::ConsecutiveSeparators { offset });
            }
        }

        let normalized: String = trimmed
            .chars()
            .map(|c| if c == '-' { '_' } else { c.to_ascii_lowercase() })
            .collect();

        if normalized.len() > REGISTRY_ID_MAX_LEN {
            return Err(RegistryIdError::TooLong {
                len: normalized.len(),
            });
        }

        let mut characters = normalized.char_indices();
        // The trimmed value is non-empty and normalization is length-preserving
        // over ASCII, so there is always a first character here.
        let Some((_, first)) = characters.next() else {
            return Err(RegistryIdError::Empty);
        };
        if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
            return Err(RegistryIdError::InvalidStart { character: first });
        }
        for (offset, character) in characters {
            let is_allowed =
                character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_';
            if !is_allowed {
                return Err(RegistryIdError::InvalidCharacter { character, offset });
            }
        }

        Ok(Self(normalized.into_boxed_str()))
    }

    /// Borrow the normalized identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the identifier and return the normalized string.
    pub fn into_string(self) -> String {
        String::from(self.0)
    }
}

/// Whether `byte` is an authored separator, checked before the `-` to `_` fold.
fn is_separator(byte: u8) -> bool {
    byte == b'-' || byte == b'_'
}

impl fmt::Display for RegistryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for RegistryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegistryId({:?})", &*self.0)
    }
}

impl AsRef<str> for RegistryId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for RegistryId {
    fn borrow(&self) -> &str {
        &self.0
    }
}
