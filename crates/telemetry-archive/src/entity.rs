// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded HTTP content decoding before exposition parsing.
//!
//! Transport receive bounds protect the encoded entity. This module separately
//! proves the decoded byte and expansion-ratio bounds while streaming a
//! supported content coding. It returns no partial body on failure, and the
//! exact encoded bytes remain distinct from the exact decoded bytes for raw
//! retention and keyed unchanged detection.

use std::fmt::{self, Debug, Display, Formatter};
use std::io::{Cursor, Read};

use bytes::Bytes;
use flate2::bufread::GzDecoder;

const MAX_CONTENT_ENCODING_HEADER_BYTES: usize = 256;
const DECODE_CHUNK_BYTES: usize = 16 * 1024;

/// Closed HTTP content-coding vocabulary supported by archive source v1.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ContentEncodingV1 {
    /// No content transformation.
    Identity,
    /// RFC 1952 gzip content coding.
    Gzip,
}

impl ContentEncodingV1 {
    /// Returns the normalized lowercase HTTP token.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Gzip => "gzip",
        }
    }
}

/// Validated response-header presence and normalized wire application order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentEncodingChainV1 {
    header_present: bool,
    encodings: Vec<ContentEncodingV1>,
}

impl ContentEncodingChainV1 {
    /// Represents an absent `Content-Encoding` header.
    #[must_use]
    pub const fn absent() -> Self {
        Self {
            header_present: false,
            encodings: Vec::new(),
        }
    }

    /// Whether the response explicitly carried `Content-Encoding`.
    #[must_use]
    pub const fn header_present(&self) -> bool {
        self.header_present
    }

    /// Returns normalized codings in wire application order.
    #[must_use]
    pub fn encodings(&self) -> &[ContentEncodingV1] {
        &self.encodings
    }

    /// Materializes the bounded lowercase chain required by raw-reference rows.
    #[must_use]
    pub fn normalized_tokens(&self) -> Vec<String> {
        self.encodings
            .iter()
            .map(|encoding| encoding.as_str().to_owned())
            .collect()
    }

    fn explicit(encoding: ContentEncodingV1) -> Self {
        Self {
            header_present: true,
            encodings: vec![encoding],
        }
    }

    fn effective_encoding(&self) -> ContentEncodingV1 {
        self.encodings
            .first()
            .copied()
            .unwrap_or(ContentEncodingV1::Identity)
    }
}

/// Independent receive/decode bounds for one HTTP entity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EntityDecodeLimitsV1 {
    /// Maximum exact content-encoded bytes.
    pub max_encoded_bytes: usize,
    /// Maximum exact content-decoded bytes.
    pub max_decoded_bytes: usize,
    /// Maximum integer decoded-to-encoded byte ratio.
    pub max_expansion_ratio: u64,
}

impl EntityDecodeLimitsV1 {
    /// Validates every bound before source IO begins.
    pub fn validate(self) -> Result<(), EntityDecodeConfigError> {
        if self.max_encoded_bytes == 0 {
            return Err(EntityDecodeConfigError::ZeroLimit("max_encoded_bytes"));
        }
        if self.max_decoded_bytes == 0 {
            return Err(EntityDecodeConfigError::ZeroLimit("max_decoded_bytes"));
        }
        if self.max_expansion_ratio == 0 {
            return Err(EntityDecodeConfigError::ZeroLimit("max_expansion_ratio"));
        }
        Ok(())
    }
}

/// Prepared content-negotiation and entity-bound policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntityDecodePolicyV1 {
    accepted_encodings: Vec<ContentEncodingV1>,
    limits: EntityDecodeLimitsV1,
}

impl EntityDecodePolicyV1 {
    /// Freezes a non-empty, unique supported encoding set and positive limits.
    pub fn new(
        accepted_encodings: impl IntoIterator<Item = ContentEncodingV1>,
        limits: EntityDecodeLimitsV1,
    ) -> Result<Self, EntityDecodeConfigError> {
        limits.validate()?;
        let mut accepted_encodings = accepted_encodings.into_iter().collect::<Vec<_>>();
        if accepted_encodings.is_empty() {
            return Err(EntityDecodeConfigError::NoAcceptedEncoding);
        }
        accepted_encodings.sort_unstable_by_key(|encoding| encoding.as_str());
        if accepted_encodings.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(EntityDecodeConfigError::DuplicateEncoding);
        }
        Ok(Self {
            accepted_encodings,
            limits,
        })
    }

    /// Stock v1 policy accepting exactly gzip and identity.
    pub fn stock(limits: EntityDecodeLimitsV1) -> Result<Self, EntityDecodeConfigError> {
        Self::new(
            [ContentEncodingV1::Gzip, ContentEncodingV1::Identity],
            limits,
        )
    }

    /// Deterministic `Accept-Encoding` value containing only implemented codings.
    #[must_use]
    pub fn accept_encoding_header(&self) -> String {
        self.accepted_encodings
            .iter()
            .map(|encoding| encoding.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Returns the exact prepared encoding set.
    #[must_use]
    pub fn accepted_encodings(&self) -> &[ContentEncodingV1] {
        &self.accepted_encodings
    }

    /// Returns the independent encoded/decoded/ratio bounds.
    #[must_use]
    pub const fn limits(&self) -> EntityDecodeLimitsV1 {
        self.limits
    }

    fn accepts(&self, encoding: ContentEncodingV1) -> bool {
        self.accepted_encodings.contains(&encoding)
    }
}

/// Atomically decoded entity with exact pre/post-coding bytes.
#[derive(Clone, Eq, PartialEq)]
pub struct DecodedHttpEntityV1 {
    /// Validated response content-coding evidence.
    pub content_encoding: ContentEncodingChainV1,
    /// Exact bytes received from HTTP after transfer coding.
    pub encoded_body: Bytes,
    /// Exact bytes after the validated content coding.
    pub decoded_body: Bytes,
}

impl Debug for DecodedHttpEntityV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodedHttpEntityV1")
            .field("content_encoding", &self.content_encoding)
            .field("encoded_bytes", &self.encoded_body.len())
            .field("decoded_bytes", &self.decoded_body.len())
            .field("entity_bytes", &"<redacted>")
            .finish()
    }
}

/// Pure bounded HTTP content-decoding seam.
pub trait BoundedEntityDecoder: Debug + Send + Sync {
    /// Validates one response `Content-Encoding` without decoding the body.
    fn inspect_content_encoding(
        &self,
        content_encoding: Option<&str>,
        policy: &EntityDecodePolicyV1,
    ) -> Result<ContentEncodingChainV1, EntityDecodeError>;

    /// Decodes one complete bounded entity or returns no partial output.
    fn decode(
        &self,
        content_encoding: Option<&str>,
        encoded_body: Bytes,
        policy: &EntityDecodePolicyV1,
    ) -> Result<DecodedHttpEntityV1, EntityDecodeError>;
}

/// Strict v1 decoder supporting only absent/identity and one gzip coding.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityGzipEntityDecoderV1;

impl BoundedEntityDecoder for IdentityGzipEntityDecoderV1 {
    fn inspect_content_encoding(
        &self,
        content_encoding: Option<&str>,
        policy: &EntityDecodePolicyV1,
    ) -> Result<ContentEncodingChainV1, EntityDecodeError> {
        let chain = parse_content_encoding(content_encoding)?;
        let effective = chain.effective_encoding();
        if !policy.accepts(effective) {
            return Err(EntityDecodeError::new(
                EntityDecodeErrorKind::DisallowedEncoding,
                "response Content-Encoding was not accepted by the prepared source policy",
            ));
        }
        Ok(chain)
    }

    fn decode(
        &self,
        content_encoding: Option<&str>,
        encoded_body: Bytes,
        policy: &EntityDecodePolicyV1,
    ) -> Result<DecodedHttpEntityV1, EntityDecodeError> {
        let limits = policy.limits();
        if encoded_body.len() > limits.max_encoded_bytes {
            return Err(EntityDecodeError::new(
                EntityDecodeErrorKind::EncodedBodyLimit,
                format!(
                    "encoded telemetry entity has {} bytes; limit is {}",
                    encoded_body.len(),
                    limits.max_encoded_bytes
                ),
            ));
        }
        let chain = self.inspect_content_encoding(content_encoding, policy)?;
        let decoded_body = match chain.effective_encoding() {
            ContentEncodingV1::Identity => {
                validate_decoded_length(encoded_body.len(), encoded_body.len(), limits)?;
                encoded_body.clone()
            }
            ContentEncodingV1::Gzip => decode_gzip(&encoded_body, limits)?,
        };
        Ok(DecodedHttpEntityV1 {
            content_encoding: chain,
            encoded_body,
            decoded_body,
        })
    }
}

fn parse_content_encoding(
    content_encoding: Option<&str>,
) -> Result<ContentEncodingChainV1, EntityDecodeError> {
    let Some(header) = content_encoding else {
        return Ok(ContentEncodingChainV1::absent());
    };
    if header.is_empty()
        || header.len() > MAX_CONTENT_ENCODING_HEADER_BYTES
        || !header.is_ascii()
        || header
            .bytes()
            .any(|byte| byte.is_ascii_control() && byte != b'\t')
    {
        return Err(EntityDecodeError::new(
            EntityDecodeErrorKind::MalformedContentEncoding,
            "response Content-Encoding is malformed",
        ));
    }
    let parts = header.split(',').collect::<Vec<_>>();
    if parts.len() != 1 {
        let malformed = parts
            .iter()
            .any(|part| trim_ows(part).is_empty() || !valid_token(trim_ows(part)));
        return Err(EntityDecodeError::new(
            if malformed {
                EntityDecodeErrorKind::MalformedContentEncoding
            } else {
                EntityDecodeErrorKind::StackedContentEncoding
            },
            if malformed {
                "response Content-Encoding chain is malformed"
            } else {
                "stacked response Content-Encoding is unsupported"
            },
        ));
    }
    let token = trim_ows(parts[0]);
    if token.is_empty() || !valid_token(token) {
        return Err(EntityDecodeError::new(
            EntityDecodeErrorKind::MalformedContentEncoding,
            "response Content-Encoding token is malformed",
        ));
    }
    let encoding = if token.eq_ignore_ascii_case("identity") {
        ContentEncodingV1::Identity
    } else if token.eq_ignore_ascii_case("gzip") {
        ContentEncodingV1::Gzip
    } else {
        return Err(EntityDecodeError::new(
            EntityDecodeErrorKind::UnsupportedEncoding,
            "response Content-Encoding is not implemented by archive source v1",
        ));
    };
    Ok(ContentEncodingChainV1::explicit(encoding))
}

fn trim_ows(value: &str) -> &str {
    value.trim_matches([' ', '\t'])
}

fn valid_token(value: &str) -> bool {
    value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b'!' | b'#'
                    | b'$'
                    | b'%'
                    | b'&'
                    | b'\''
                    | b'*'
                    | b'+'
                    | b'-'
                    | b'.'
                    | b'^'
                    | b'_'
                    | b'`'
                    | b'|'
                    | b'~'
            )
    })
}

fn decode_gzip(
    encoded_body: &Bytes,
    limits: EntityDecodeLimitsV1,
) -> Result<Bytes, EntityDecodeError> {
    let ratio_limit = ratio_limit(encoded_body.len(), limits.max_expansion_ratio);
    let output_limit = limits.max_decoded_bytes.min(ratio_limit);
    let limit_kind = if limits.max_decoded_bytes <= ratio_limit {
        EntityDecodeErrorKind::DecodedBodyLimit
    } else {
        EntityDecodeErrorKind::ExpansionRatio
    };
    let cursor = Cursor::new(encoded_body.as_ref());
    let mut decoder = GzDecoder::new(cursor);
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_limit.min(DECODE_CHUNK_BYTES))
        .map_err(|_| {
            EntityDecodeError::new(
                EntityDecodeErrorKind::Allocation,
                "bounded gzip output allocation failed",
            )
        })?;
    let mut buffer = [0_u8; DECODE_CHUNK_BYTES];
    loop {
        let remaining = output_limit.saturating_sub(output.len());
        let read_capacity = remaining.min(buffer.len());
        if read_capacity == 0 {
            let mut probe = [0_u8; 1];
            match decoder.read(&mut probe) {
                Ok(0) => break,
                Ok(_) => return Err(entity_limit_error(limit_kind, encoded_body.len(), limits)),
                Err(_) => return Err(malformed_gzip_error()),
            }
        }
        let count = decoder
            .read(&mut buffer[..read_capacity])
            .map_err(|_| malformed_gzip_error())?;
        if count == 0 {
            break;
        }
        output.try_reserve(count).map_err(|_| {
            EntityDecodeError::new(
                EntityDecodeErrorKind::Allocation,
                "bounded gzip output allocation failed",
            )
        })?;
        output.extend_from_slice(&buffer[..count]);
    }
    let consumed = usize::try_from(decoder.into_inner().position()).map_err(|_| {
        EntityDecodeError::new(
            EntityDecodeErrorKind::MalformedGzip,
            "gzip decoder consumed an invalid byte count",
        )
    })?;
    if consumed != encoded_body.len() {
        return Err(EntityDecodeError::new(
            EntityDecodeErrorKind::TrailingGzipData,
            "gzip entity contains trailing or concatenated data",
        ));
    }
    validate_decoded_length(output.len(), encoded_body.len(), limits)?;
    Ok(Bytes::from(output))
}

fn validate_decoded_length(
    decoded_bytes: usize,
    encoded_bytes: usize,
    limits: EntityDecodeLimitsV1,
) -> Result<(), EntityDecodeError> {
    if decoded_bytes > limits.max_decoded_bytes {
        return Err(entity_limit_error(
            EntityDecodeErrorKind::DecodedBodyLimit,
            encoded_bytes,
            limits,
        ));
    }
    if decoded_bytes > ratio_limit(encoded_bytes, limits.max_expansion_ratio) {
        return Err(entity_limit_error(
            EntityDecodeErrorKind::ExpansionRatio,
            encoded_bytes,
            limits,
        ));
    }
    Ok(())
}

fn ratio_limit(encoded_bytes: usize, maximum_ratio: u64) -> usize {
    encoded_bytes
        .max(1)
        .saturating_mul(usize::try_from(maximum_ratio).unwrap_or(usize::MAX))
}

fn entity_limit_error(
    kind: EntityDecodeErrorKind,
    encoded_bytes: usize,
    limits: EntityDecodeLimitsV1,
) -> EntityDecodeError {
    let message = match kind {
        EntityDecodeErrorKind::DecodedBodyLimit => format!(
            "decoded telemetry entity exceeded the {} byte limit",
            limits.max_decoded_bytes
        ),
        EntityDecodeErrorKind::ExpansionRatio => format!(
            "decoded telemetry entity exceeded expansion ratio {} from {encoded_bytes} encoded bytes",
            limits.max_expansion_ratio
        ),
        _ => "telemetry entity exceeded a decode limit".to_owned(),
    };
    EntityDecodeError::new(kind, message)
}

fn malformed_gzip_error() -> EntityDecodeError {
    EntityDecodeError::new(
        EntityDecodeErrorKind::MalformedGzip,
        "response gzip entity is malformed or truncated",
    )
}

/// Invalid entity-decoder policy discovered before source preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EntityDecodeConfigError {
    /// A byte or ratio limit was zero.
    ZeroLimit(&'static str),
    /// No response encoding was accepted.
    NoAcceptedEncoding,
    /// The accepted encoding list repeated a coding.
    DuplicateEncoding,
}

impl Display for EntityDecodeConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLimit(limit) => {
                write!(formatter, "entity decode limit {limit} must be positive")
            }
            Self::NoAcceptedEncoding => {
                formatter.write_str("entity decoder must accept at least one content encoding")
            }
            Self::DuplicateEncoding => {
                formatter.write_str("entity decoder accepted encoding list contains a duplicate")
            }
        }
    }
}

impl std::error::Error for EntityDecodeConfigError {}

/// Stable atomic entity-decoding failure category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntityDecodeErrorKind {
    /// Adapter-supplied or replay decode policy was invalid.
    InvalidPolicy,
    /// Exact encoded bytes exceeded the receive policy.
    EncodedBodyLimit,
    /// Exact decoded bytes exceeded the decoded policy.
    DecodedBodyLimit,
    /// Decoded bytes exceeded the encoded-relative ratio policy.
    ExpansionRatio,
    /// Header syntax was invalid.
    MalformedContentEncoding,
    /// More than one content coding was applied.
    StackedContentEncoding,
    /// A syntactically valid coding is not implemented in v1.
    UnsupportedEncoding,
    /// An implemented coding was excluded by source policy.
    DisallowedEncoding,
    /// Gzip framing, DEFLATE data, checksum, or length was invalid.
    MalformedGzip,
    /// A valid gzip member was followed by another member or trailing bytes.
    TrailingGzipData,
    /// Bounded output allocation failed.
    Allocation,
    /// Adapter-supplied predecoded bytes disagreed with strict decoding.
    PredecodedBodyMismatch,
}

impl EntityDecodeErrorKind {
    /// Returns a stable diagnostic/attempt category.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid_entity_decode_policy",
            Self::EncodedBodyLimit => "encoded_body_limit",
            Self::DecodedBodyLimit => "decoded_body_limit",
            Self::ExpansionRatio => "entity_expansion_ratio",
            Self::MalformedContentEncoding => "malformed_content_encoding",
            Self::StackedContentEncoding => "stacked_content_encoding",
            Self::UnsupportedEncoding => "unsupported_content_encoding",
            Self::DisallowedEncoding => "disallowed_content_encoding",
            Self::MalformedGzip => "malformed_gzip",
            Self::TrailingGzipData => "trailing_gzip_data",
            Self::Allocation => "entity_decode_allocation",
            Self::PredecodedBodyMismatch => "predecoded_body_mismatch",
        }
    }
}

/// One bounded, source-content-free entity decoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntityDecodeError {
    /// Stable failure category.
    pub kind: EntityDecodeErrorKind,
    /// Bounded diagnostic containing no response bytes.
    pub message: String,
}

impl EntityDecodeError {
    /// Constructs a failure from a stable kind and response-content-free detail.
    pub fn new(kind: EntityDecodeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl Display for EntityDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for EntityDecodeError {}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::*;

    fn limits() -> EntityDecodeLimitsV1 {
        EntityDecodeLimitsV1 {
            max_encoded_bytes: 1024,
            max_decoded_bytes: 4096,
            max_expansion_ratio: 100,
        }
    }

    fn policy() -> EntityDecodePolicyV1 {
        EntityDecodePolicyV1::stock(limits()).unwrap()
    }

    fn gzip(bytes: &[u8]) -> Bytes {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(bytes).unwrap();
        Bytes::from(encoder.finish().unwrap())
    }

    #[test]
    fn absent_and_explicit_identity_remain_distinct() {
        let decoder = IdentityGzipEntityDecoderV1;
        let absent = decoder
            .decode(None, Bytes::from_static(b"metric 1\n"), &policy())
            .unwrap();
        let explicit = decoder
            .decode(
                Some(" \tIdEnTiTy\t "),
                Bytes::from_static(b"metric 1\n"),
                &policy(),
            )
            .unwrap();
        assert_eq!(absent.encoded_body, absent.decoded_body);
        assert!(!absent.content_encoding.header_present());
        assert!(absent.content_encoding.encodings().is_empty());
        assert!(explicit.content_encoding.header_present());
        assert_eq!(
            explicit.content_encoding.encodings(),
            &[ContentEncodingV1::Identity]
        );
        assert_eq!(explicit.content_encoding.normalized_tokens(), ["identity"]);
    }

    #[test]
    fn gzip_preserves_exact_encoded_and_decoded_bytes() {
        let body = b"# TYPE requests counter\nrequests_total 17\n";
        let encoded = gzip(body);
        let decoded = IdentityGzipEntityDecoderV1
            .decode(Some("GZip"), encoded.clone(), &policy())
            .unwrap();
        assert_eq!(decoded.encoded_body, encoded);
        assert_eq!(decoded.decoded_body, body.as_slice());
        assert_eq!(
            decoded.content_encoding.encodings(),
            &[ContentEncodingV1::Gzip]
        );
        assert_eq!(decoded.content_encoding.normalized_tokens(), ["gzip"]);
    }

    #[test]
    fn gzip_bomb_fails_at_ratio_or_decoded_bound_without_partial_output() {
        let encoded = gzip(&vec![b'x'; 16 * 1024]);
        let ratio_policy = EntityDecodePolicyV1::stock(EntityDecodeLimitsV1 {
            max_encoded_bytes: 1024,
            max_decoded_bytes: 32 * 1024,
            max_expansion_ratio: 2,
        })
        .unwrap();
        let error = IdentityGzipEntityDecoderV1
            .decode(Some("gzip"), encoded.clone(), &ratio_policy)
            .unwrap_err();
        assert_eq!(error.kind, EntityDecodeErrorKind::ExpansionRatio);

        let decoded_policy = EntityDecodePolicyV1::stock(EntityDecodeLimitsV1 {
            max_encoded_bytes: 1024,
            max_decoded_bytes: 64,
            max_expansion_ratio: 100,
        })
        .unwrap();
        let error = IdentityGzipEntityDecoderV1
            .decode(Some("gzip"), encoded, &decoded_policy)
            .unwrap_err();
        assert_eq!(error.kind, EntityDecodeErrorKind::DecodedBodyLimit);
    }

    #[test]
    fn truncated_and_concatenated_gzip_fail_closed() {
        let mut truncated = gzip(b"metric 1\n").to_vec();
        truncated.truncate(truncated.len() - 4);
        assert_eq!(
            IdentityGzipEntityDecoderV1
                .decode(Some("gzip"), Bytes::from(truncated), &policy())
                .unwrap_err()
                .kind,
            EntityDecodeErrorKind::MalformedGzip
        );

        let mut concatenated = gzip(b"metric 1\n").to_vec();
        concatenated.extend_from_slice(&gzip(b"metric 2\n"));
        assert_eq!(
            IdentityGzipEntityDecoderV1
                .decode(Some("gzip"), Bytes::from(concatenated), &policy())
                .unwrap_err()
                .kind,
            EntityDecodeErrorKind::TrailingGzipData
        );
    }

    #[test]
    fn malformed_stacked_unknown_and_disallowed_headers_are_distinct() {
        let decoder = IdentityGzipEntityDecoderV1;
        for (header, expected) in [
            ("gzip,", EntityDecodeErrorKind::MalformedContentEncoding),
            (
                "gzip, identity",
                EntityDecodeErrorKind::StackedContentEncoding,
            ),
            ("br", EntityDecodeErrorKind::UnsupportedEncoding),
            (
                "gzip; level=1",
                EntityDecodeErrorKind::MalformedContentEncoding,
            ),
        ] {
            assert_eq!(
                decoder
                    .decode(Some(header), Bytes::from_static(b"x"), &policy())
                    .unwrap_err()
                    .kind,
                expected
            );
        }
        let identity_only =
            EntityDecodePolicyV1::new([ContentEncodingV1::Identity], limits()).unwrap();
        assert_eq!(
            decoder
                .decode(Some("gzip"), gzip(b"x"), &identity_only)
                .unwrap_err()
                .kind,
            EntityDecodeErrorKind::DisallowedEncoding
        );
    }

    #[test]
    fn negotiation_contains_only_the_validated_supported_set() {
        assert_eq!(policy().accept_encoding_header(), "gzip, identity");
        let identity_only =
            EntityDecodePolicyV1::new([ContentEncodingV1::Identity], limits()).unwrap();
        assert_eq!(identity_only.accept_encoding_header(), "identity");
        assert_eq!(
            EntityDecodePolicyV1::new(
                [ContentEncodingV1::Gzip, ContentEncodingV1::Gzip],
                limits(),
            )
            .unwrap_err(),
            EntityDecodeConfigError::DuplicateEncoding
        );
    }
}
