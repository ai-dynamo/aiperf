// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Archive-key provider seam and independently derived digest subkeys.

use std::fmt::{self, Debug, Display, Formatter};

use crate::Digest;

/// Frozen purpose for one independently derived archive subkey.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveSubkey {
    /// Protected pre-redaction source-series identity.
    SourceSeries,
    /// Optional encoded HTTP entity integrity digest.
    EncodedBody,
    /// Decoded exposition unchanged detection and batch identity.
    DecodedBody,
    /// Exact raw-object content addressing.
    RawObject,
}

impl ArchiveSubkey {
    const fn context(self) -> &'static str {
        match self {
            Self::SourceSeries => "aiperf.archive.subkey.series-source.v1",
            Self::EncodedBody => "aiperf.archive.subkey.body-encoded.v1",
            Self::DecodedBody => "aiperf.archive.subkey.body-decoded.v1",
            Self::RawObject => "aiperf.archive.subkey.raw-object.v1",
        }
    }
}

/// Prepared provider for secret keyed-digest material.
///
/// Only [`Self::provider_id`] and its separately authored secret-free config
/// digest may enter genesis. Returned key bytes are process-local and must
/// never be serialized or included in diagnostics.
pub trait ArchiveKeyProvider: Debug + Send + Sync {
    /// Stable registry identity of the provider implementation/config.
    fn provider_id(&self) -> &str;

    /// Derives one purpose-separated 256-bit subkey.
    fn derive_subkey(&self, purpose: ArchiveSubkey) -> Result<[u8; 32], ArchiveKeyError>;
}

/// In-memory provider used by prepared secret resolvers and deterministic tests.
pub struct Blake3ArchiveKeyProvider {
    provider_id: String,
    master_key: [u8; 32],
}

impl Blake3ArchiveKeyProvider {
    /// Binds already resolved secret bytes to a public provider identity.
    pub fn new(
        provider_id: impl Into<String>,
        master_key: [u8; 32],
    ) -> Result<Self, ArchiveKeyError> {
        let provider_id = provider_id.into();
        validate_provider_id(&provider_id)?;
        Ok(Self {
            provider_id,
            master_key,
        })
    }
}

impl Debug for Blake3ArchiveKeyProvider {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Blake3ArchiveKeyProvider")
            .field("provider_id", &self.provider_id)
            .field("master_key", &"<redacted>")
            .finish()
    }
}

impl ArchiveKeyProvider for Blake3ArchiveKeyProvider {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn derive_subkey(&self, purpose: ArchiveSubkey) -> Result<[u8; 32], ArchiveKeyError> {
        let mut hasher = blake3::Hasher::new_keyed(&self.master_key);
        hasher.update(b"aiperf.archive.key-provider.v1\0");
        let context = purpose.context().as_bytes();
        let length = u64::try_from(context.len()).map_err(|_| ArchiveKeyError::LengthOverflow)?;
        hasher.update(&length.to_be_bytes());
        hasher.update(context);
        Ok(*hasher.finalize().as_bytes())
    }
}

/// Computes the archive's keyed, domain-separated, length-prefixed digest.
#[must_use]
pub fn keyed_domain_digest(key: &[u8; 32], domain: &str, fields: &[&[u8]]) -> Digest {
    let mut hasher = blake3::Hasher::new_keyed(key);
    hasher.update(domain.as_bytes());
    hasher.update(&[0]);
    for field in fields {
        let length =
            u64::try_from(field.len()).expect("usize always fits u64 on supported targets");
        hasher.update(&length.to_be_bytes());
        hasher.update(field);
    }
    Digest::from_bytes(*hasher.finalize().as_bytes())
}

fn validate_provider_id(value: &str) -> Result<(), ArchiveKeyError> {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return Err(ArchiveKeyError::InvalidProviderId);
    };
    if !first.is_ascii_lowercase()
        || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(ArchiveKeyError::InvalidProviderId);
    }
    Ok(())
}

/// Invalid public identity or impossible subkey derivation input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveKeyError {
    /// Provider IDs use the stable lowercase runner-component grammar.
    InvalidProviderId,
    /// A context length could not be represented by the frozen preimage.
    LengthOverflow,
}

impl Display for ArchiveKeyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidProviderId => {
                "archive key provider ID must start with lowercase ASCII and contain only lowercase ASCII, digits, or underscores"
            }
            Self::LengthOverflow => "archive key derivation length overflowed",
        })
    }
}

impl std::error::Error for ArchiveKeyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_digest_purpose_has_an_independent_subkey() {
        let provider = Blake3ArchiveKeyProvider::new("fixture_key", [7; 32]).unwrap();
        let keys = [
            ArchiveSubkey::SourceSeries,
            ArchiveSubkey::EncodedBody,
            ArchiveSubkey::DecodedBody,
            ArchiveSubkey::RawObject,
        ]
        .map(|purpose| provider.derive_subkey(purpose).unwrap());
        for (index, left) in keys.iter().enumerate() {
            for right in &keys[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn debug_output_never_contains_key_material() {
        let provider = Blake3ArchiveKeyProvider::new("fixture_key", [0xab; 32]).unwrap();
        let rendered = format!("{provider:?}");
        assert!(rendered.contains("fixture_key"));
        assert!(!rendered.contains("171"));
        assert!(!rendered.contains("ab"));
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn keyed_fields_and_domains_remain_unambiguous() {
        let key = [11; 32];
        assert_ne!(
            keyed_domain_digest(&key, "aiperf.archive.test.v1", &[b"a", b"bc"]),
            keyed_domain_digest(&key, "aiperf.archive.test.v1", &[b"ab", b"c"]),
        );
        assert_ne!(
            keyed_domain_digest(&key, "aiperf.archive.test.v1", &[b"a"]),
            keyed_domain_digest(&key, "aiperf.archive.other.v1", &[b"a"]),
        );
    }
}
