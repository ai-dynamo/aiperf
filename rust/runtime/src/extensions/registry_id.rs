// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared string-newtype key for open, dynamically extensible registries.
//!
//! Datasets, samplers, transports, workloads, exporters, and actuators are all
//! registered under a case/hyphen-insensitive name coming from a static Rust
//! trait method or descriptor field, never directly off wire config. This type
//! centralizes the normalization and non-empty validation those registries
//! previously duplicated (`normalize_name` in `dataset/loader/mod.rs` and
//! `dataset/sampler.rs`) behind one comparable, hashable key type instead of a
//! bare `String`. [`crate::endpoints::EndpointId`] is a distinct type: it
//! validates a stricter wire-facing grammar and is part of the public
//! endpoint-binding surface.

use std::borrow::Borrow;
use std::error::Error;
use std::fmt::{self, Display};
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

/// Normalize an identifier for case/separator-insensitive matching: trim,
/// lowercase, and fold `-` to `_`. This is the one shared seam for every
/// registry key and discriminant enum, mirroring Python's
/// `ExtensibleStrEnum._normalize_name` (lowercase + `-`→`_`; note it does *not*
/// strip separators, so `graphir` is a spelling alias, not a normalization of
/// `graph-ir`).
pub fn normalize_ident(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

/// Normalized registry identifier: lowercase, trimmed, `-` folded to `_`.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct RegistryId(Box<str>);

impl RegistryId {
    /// Normalize and validate a registry identifier.
    pub fn new(value: impl AsRef<str>) -> Result<Self, RegistryIdError> {
        let normalized = normalize_ident(value.as_ref());
        if normalized.is_empty() {
            return Err(RegistryIdError {
                value: value.as_ref().to_string(),
            });
        }
        Ok(Self(normalized.into()))
    }

    /// Borrow the normalized identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for RegistryId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl Display for RegistryId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for RegistryId {
    type Err = RegistryIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<&str> for RegistryId {
    type Error = RegistryIdError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for RegistryId {
    type Error = RegistryIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for RegistryId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(&value).map_err(serde::de::Error::custom)
    }
}

/// Invalid or empty registry identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistryIdError {
    value: String,
}

impl RegistryIdError {
    /// Return the rejected spelling.
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl Display for RegistryIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid registry ID {:?}", self.value)
    }
}

impl Error for RegistryIdError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_case_and_hyphens() {
        let id = RegistryId::new(" Random-Pool ").unwrap();
        assert_eq!(id.as_str(), "random_pool");
    }

    #[test]
    fn rejects_empty() {
        assert!(RegistryId::new("   ").is_err());
    }

    #[test]
    fn borrow_str_matches_map_lookup() {
        use std::collections::HashMap;

        let mut map: HashMap<RegistryId, u32> = HashMap::new();
        map.insert(RegistryId::new("Random").unwrap(), 1);
        assert_eq!(map.get("random"), Some(&1));
    }
}
