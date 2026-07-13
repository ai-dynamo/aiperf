// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Domain-separated archive digests.

use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

/// A BLAKE3 digest used by the archive wire formats.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Digest([u8; 32]);

impl Digest {
    /// The number of bytes in every archive digest.
    pub const BYTE_LEN: usize = 32;

    /// Constructs a digest from its exact bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; Self::BYTE_LEN]) -> Self {
        Self(bytes)
    }

    /// Returns the exact digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Self::BYTE_LEN] {
        &self.0
    }

    /// Consumes the digest and returns its exact bytes.
    #[must_use]
    pub const fn into_bytes(self) -> [u8; Self::BYTE_LEN] {
        self.0
    }

    /// Encodes the digest as 64 lowercase hexadecimal characters.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(Self::BYTE_LEN * 2);
        for byte in self.0 {
            output.push(char::from(HEX[usize::from(byte >> 4)]));
            output.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        output
    }

    /// Encodes the digest with the manifest-facing `blake3:` algorithm tag.
    #[must_use]
    pub fn to_tagged_hex(self) -> String {
        format!("blake3:{}", self.to_hex())
    }

    /// Parses either fixed-width lowercase hex or the tagged `blake3:` form.
    pub fn parse(value: &str) -> Result<Self, DigestError> {
        let hex = value.strip_prefix("blake3:").unwrap_or(value);
        if hex.len() != Self::BYTE_LEN * 2 {
            return Err(DigestError::InvalidLength { actual: hex.len() });
        }
        let mut bytes = [0_u8; Self::BYTE_LEN];
        for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
            let high = decode_hex(pair[0]).ok_or(DigestError::InvalidHex { offset: index * 2 })?;
            let low = decode_hex(pair[1]).ok_or(DigestError::InvalidHex {
                offset: index * 2 + 1,
            })?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }
}

impl AsRef<[u8]> for Digest {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl Display for Digest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl FromStr for Digest {
    type Err = DigestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

/// A malformed textual digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DigestError {
    /// The hexadecimal body does not contain exactly 64 characters.
    InvalidLength {
        /// The observed hexadecimal-body length.
        actual: usize,
    },
    /// A character at the given byte offset is not lowercase hexadecimal.
    InvalidHex {
        /// The invalid byte offset within the hexadecimal body.
        offset: usize,
    },
}

impl Display for DigestError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { actual } => write!(
                formatter,
                "archive digest must contain 64 hexadecimal characters, found {actual}"
            ),
            Self::InvalidHex { offset } => write!(
                formatter,
                "archive digest contains non-lowercase-hexadecimal data at byte {offset}"
            ),
        }
    }
}

impl std::error::Error for DigestError {}

/// Computes the archive's length-prefixed, domain-separated BLAKE3 construction.
///
/// The preimage is `domain || 0x00 || each(u64_be(len) || bytes)`. Callers must
/// use a frozen domain constant rather than composing one from runtime input.
#[must_use]
pub fn domain_digest(domain: &str, fields: &[&[u8]]) -> Digest {
    let mut hasher = blake3::Hasher::new();
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

fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_and_bare_hex_round_trip() {
        let digest = domain_digest("aiperf.archive.test.v1", &[b"alpha", b"beta"]);
        assert_eq!(Digest::parse(&digest.to_hex()), Ok(digest));
        assert_eq!(Digest::parse(&digest.to_tagged_hex()), Ok(digest));
        assert!(matches!(
            Digest::parse(&digest.to_hex().to_uppercase()),
            Err(DigestError::InvalidHex { .. })
        ));
    }

    #[test]
    fn field_boundaries_are_unambiguous() {
        assert_ne!(
            domain_digest("aiperf.archive.test.v1", &[b"a", b"bc"]),
            domain_digest("aiperf.archive.test.v1", &[b"ab", b"c"])
        );
        assert_ne!(
            domain_digest("aiperf.archive.test.v1", &[b"a", b"bc"]),
            domain_digest("aiperf.archive.other.v1", &[b"a", b"bc"])
        );
    }
}
