// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked-in canonical descriptor authorities.

use std::fmt::{self, Display, Formatter};

use crate::{CanonicalJsonError, CanonicalJsonValue, Digest, domain_digest};

/// A checked-in descriptor whose exact canonical bytes are fingerprint authority.
#[derive(Clone, Copy, Debug)]
pub struct CanonicalDescriptor {
    name: &'static str,
    bytes: &'static [u8],
}

impl CanonicalDescriptor {
    /// Constructs a descriptor backed by immutable checked-in bytes.
    #[must_use]
    pub const fn new(name: &'static str, bytes: &'static [u8]) -> Self {
        Self { name, bytes }
    }

    /// Returns the stable descriptor name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Returns the exact fingerprinted bytes.
    #[must_use]
    pub fn bytes(self) -> &'static [u8] {
        self.bytes.strip_suffix(b"\n").unwrap_or(self.bytes)
    }

    /// Computes the descriptor fingerprint.
    #[must_use]
    pub fn fingerprint(self) -> Digest {
        domain_digest(
            "aiperf.archive.descriptor.v1",
            &[self.name.as_bytes(), self.bytes()],
        )
    }

    /// Verifies that checked-in bytes are already canonical and identify this descriptor.
    pub fn validate(self) -> Result<(), DescriptorError> {
        let bytes = self.bytes();
        let decoded = CanonicalJsonValue::parse(bytes).map_err(DescriptorError::Canonical)?;
        if decoded.to_bytes() != bytes {
            return Err(DescriptorError::NotCanonical(self.name));
        }
        let Some(object) = decoded.as_object() else {
            return Err(DescriptorError::NotObject(self.name));
        };
        let actual_name = object.get("name").and_then(CanonicalJsonValue::as_str);
        if actual_name != Some(self.name) {
            return Err(DescriptorError::NameMismatch {
                expected: self.name,
                actual: actual_name.map(ToOwned::to_owned),
            });
        }
        Ok(())
    }
}

/// A checked-in descriptor that is malformed or no longer canonical.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DescriptorError {
    /// Canonical JSON decoding failed.
    Canonical(CanonicalJsonError),
    /// The decoded descriptor is not a JSON object.
    NotObject(&'static str),
    /// Re-encoding did not reproduce the checked-in bytes.
    NotCanonical(&'static str),
    /// The embedded descriptor name disagrees with its registered name.
    NameMismatch {
        /// The registered descriptor name.
        expected: &'static str,
        /// The embedded name, when present and textual.
        actual: Option<String>,
    },
}

impl Display for DescriptorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonical(error) => write!(formatter, "descriptor JSON is invalid: {error}"),
            Self::NotObject(name) => write!(formatter, "descriptor {name} is not an object"),
            Self::NotCanonical(name) => {
                write!(formatter, "descriptor {name} bytes are not canonical")
            }
            Self::NameMismatch { expected, actual } => write!(
                formatter,
                "descriptor name mismatch: expected {expected:?}, found {actual:?}"
            ),
        }
    }
}

impl std::error::Error for DescriptorError {}

/// Canonical JSON profile descriptor.
pub const CANONICAL_JSON_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "canonical-json-v1",
    include_bytes!("../descriptors/canonical-json-v1.json"),
);

/// Frame identity descriptor.
pub const FRAME_IDENTITY_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "frame-identity-v1",
    include_bytes!("../descriptors/frame-identity-v1.json"),
);

/// Canonical logical-row and projection-evidence descriptor.
pub const LOGICAL_ROW_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "logical-row-v1",
    include_bytes!("../descriptors/logical-row-v1.json"),
);

/// WAL frame and segment descriptor.
pub const WAL_V1: CanonicalDescriptor =
    CanonicalDescriptor::new("wal-v1", include_bytes!("../descriptors/wal-v1.json"));

/// Persistent index descriptor.
pub const INDEX_V1: CanonicalDescriptor =
    CanonicalDescriptor::new("index-v1", include_bytes!("../descriptors/index-v1.json"));

/// Receipt journal descriptor.
pub const RECEIPT_V1: CanonicalDescriptor = CanonicalDescriptor::new(
    "receipt-v1",
    include_bytes!("../descriptors/receipt-v1.json"),
);

/// Every durable-core descriptor in deterministic validation order.
pub const ALL_DESCRIPTORS: &[CanonicalDescriptor] = &[
    CANONICAL_JSON_V1,
    FRAME_IDENTITY_V1,
    LOGICAL_ROW_V1,
    WAL_V1,
    INDEX_V1,
    RECEIPT_V1,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_checked_in_descriptor_is_canonical_and_named() {
        for descriptor in ALL_DESCRIPTORS {
            descriptor.validate().unwrap();
        }
    }

    #[test]
    fn descriptor_fingerprints_are_name_bound() {
        let bytes = CANONICAL_JSON_V1.bytes();
        assert_ne!(
            CANONICAL_JSON_V1.fingerprint(),
            CanonicalDescriptor::new("other", bytes).fingerprint()
        );
    }
}
