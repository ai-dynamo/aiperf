// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded two-stage attempt decoding with strict archive/native separation.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;

use aiperf_prometheus::{
    Exposition, ExpositionFormat, ExpositionParser, ParseError, ParseErrorKind, ParseLimits,
};
use bytes::Bytes;

use crate::{
    ArchiveKeyError, ArchiveKeyProvider, ArchiveSubkey, Digest, SourceOutcome, keyed_domain_digest,
};

/// Resource bounds applied before parser or native compatibility work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeLimits {
    /// Maximum exact encoded entity bytes retained from transport.
    pub max_encoded_bytes: usize,
    /// Maximum content-decoded exposition bytes.
    pub max_decoded_bytes: usize,
    /// Maximum integer decoded-to-encoded expansion ratio.
    pub max_expansion_ratio: u64,
    /// Maximum redaction-safe diagnostic UTF-8 bytes.
    pub max_diagnostic_bytes: usize,
    /// Format parser/cardinality limits.
    pub parse: ParseLimits,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 8 * 1024 * 1024,
            max_decoded_bytes: 32 * 1024 * 1024,
            max_expansion_ratio: 100,
            max_diagnostic_bytes: 4096,
            parse: ParseLimits::default(),
        }
    }
}

impl DecodeLimits {
    /// Rejects zero/inconsistent limits before source IO begins.
    pub fn validate(&self) -> Result<(), DecodeConfigError> {
        if self.max_encoded_bytes == 0 {
            return Err(DecodeConfigError::ZeroLimit("max_encoded_bytes"));
        }
        if self.max_decoded_bytes == 0 {
            return Err(DecodeConfigError::ZeroLimit("max_decoded_bytes"));
        }
        if self.max_expansion_ratio == 0 {
            return Err(DecodeConfigError::ZeroLimit("max_expansion_ratio"));
        }
        if self.max_diagnostic_bytes == 0 {
            return Err(DecodeConfigError::ZeroLimit("max_diagnostic_bytes"));
        }
        if self.parse.max_decoded_bytes != self.max_decoded_bytes {
            return Err(DecodeConfigError::DecodedLimitMismatch {
                decode: self.max_decoded_bytes,
                parser: self.parse.max_decoded_bytes,
            });
        }
        Ok(())
    }
}

/// Exact fetch disposition before any metrics grammar is selected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FetchDisposition {
    /// Complete HTTP response with separately retained encoded/decoded bytes.
    Response {
        /// HTTP status.
        status: u16,
        /// Exact allowlisted `Content-Type` value.
        content_type: Option<String>,
        /// Exact allowlisted `Content-Encoding` value.
        content_encoding: Option<String>,
        /// Bounded entity bytes before content decoding.
        encoded_body: Bytes,
        /// Validated content-decoded entity bytes.
        decoded_body: Bytes,
    },
    /// DNS/TCP/TLS/HTTP transport failure.
    Transport {
        /// Stable transport category.
        kind: String,
        /// Redaction-safe bounded detail.
        message: String,
    },
    /// Absolute Clock deadline expired.
    Timeout {
        /// Whether any network IO began.
        request_started: bool,
    },
    /// Source was terminally disabled before IO.
    Disabled {
        /// Stable disable reason.
        reason: String,
    },
    /// Active or pending source work was closed by shutdown.
    Shutdown,
}

/// One transport-complete source attempt presented to the CPU decode stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchedAttempt {
    /// Stable physical source identity.
    pub source_id: String,
    /// Per-source event sequence.
    pub source_record_seq: u64,
    /// Per-source network attempt sequence when IO began.
    pub request_attempt_seq: Option<u64>,
    /// Optional cadence deadline.
    pub scheduled_ns: Option<i64>,
    /// Clock instant before IO.
    pub request_start_ns: Option<i64>,
    /// Clock instant of first body byte.
    pub first_byte_ns: Option<i64>,
    /// Clock instant representing the fetched source snapshot.
    pub capture_ns: Option<i64>,
    /// Non-negative endpoint latency when available.
    pub latency_ns: Option<i64>,
    /// Terminal transport disposition and exact bytes.
    pub disposition: FetchDisposition,
}

/// Capability-scoped exact body handles reserved for raw/archive projection.
#[derive(Clone, Debug)]
pub struct ExactEntityLease {
    encoded: Bytes,
    decoded: Bytes,
}

impl ExactEntityLease {
    fn new(encoded: Bytes, decoded: Bytes) -> Self {
        Self { encoded, decoded }
    }

    /// Encoded entity byte count without opening the content.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        self.encoded.len()
    }

    /// Decoded entity byte count without opening the content.
    #[must_use]
    pub fn decoded_len(&self) -> usize {
        self.decoded.len()
    }

    /// Computes the protected exact encoded-entity digest without exposing bytes.
    pub fn encoded_digest(
        &self,
        provider: &dyn ArchiveKeyProvider,
    ) -> Result<Digest, ArchiveKeyError> {
        let key = provider.derive_subkey(ArchiveSubkey::EncodedBody)?;
        Ok(keyed_domain_digest(
            &key,
            "aiperf.archive.body-encoded.v1",
            &[&self.encoded],
        ))
    }

    /// Computes the protected exact decoded-entity digest without exposing bytes.
    pub fn decoded_digest(
        &self,
        provider: &dyn ArchiveKeyProvider,
    ) -> Result<Digest, ArchiveKeyError> {
        let key = provider.derive_subkey(ArchiveSubkey::DecodedBody)?;
        Ok(keyed_domain_digest(
            &key,
            "aiperf.archive.body-decoded.v1",
            &[&self.decoded],
        ))
    }

    /// Exact encoded bytes for in-crate raw-envelope projection.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the raw-envelope projection consumes this capability in the next archive increment"
        )
    )]
    pub(crate) fn encoded(&self) -> &[u8] {
        &self.encoded
    }

    /// Exact decoded bytes for in-crate raw-envelope projection.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the raw-envelope projection consumes this capability in the next archive increment"
        )
    )]
    pub(crate) fn decoded(&self) -> &[u8] {
        &self.decoded
    }
}

/// Strict archive parse disposition independent of native compatibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseOutcome {
    /// No grammar was invoked because transport/status/format gates failed first.
    NotAttempted,
    /// The selected declared grammar parsed atomically.
    Success {
        /// Exact grammar selected from `Content-Type`.
        format: ExpositionFormat,
    },
    /// The selected grammar rejected the entire exposition.
    Failed {
        /// Exact grammar selected from `Content-Type`.
        format: ExpositionFormat,
        /// Typed parser failure.
        error: ParseError,
    },
}

/// Borrowed strict result made visible to a separately injected native decoder.
#[derive(Clone, Copy, Debug)]
pub enum StrictParseView<'a> {
    /// No strict grammar ran.
    NotAttempted,
    /// Strict archive entity parsed successfully.
    Success(&'a Exposition),
    /// Strict selected grammar failed atomically.
    Failed {
        /// Selected strict grammar.
        format: ExpositionFormat,
        /// Typed strict failure.
        error: &'a ParseError,
    },
}

/// Explicit native compatibility grammar evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityFallback {
    /// Actual grammar used by the native-only decoder.
    pub format: ExpositionFormat,
    /// Whether that decoder produced a native entity.
    pub succeeded: bool,
    /// Optional bounded native-only diagnostic.
    pub error_message: Option<String>,
}

/// Result of a separately named native decoder/projection.
#[derive(Clone, Debug)]
pub struct NativeDecodeOutcome<NativeEntity> {
    /// Optional successfully decoded native entity.
    pub entity: Option<NativeEntity>,
    /// Compatibility evidence only when a second grammar was intentionally used.
    pub compatibility: Option<CompatibilityFallback>,
}

impl<NativeEntity> NativeDecodeOutcome<NativeEntity> {
    /// Produces no native record and no compatibility grammar evidence.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            entity: None,
            compatibility: None,
        }
    }
}

/// Pure native entity decoder, independent of the strict archive outcome.
pub trait NativeEntityDecoder<NativeEntity>: Debug + Send + Sync {
    /// Decodes/projects native semantics without changing `strict`.
    fn decode(
        &self,
        declared_content_type: Option<&str>,
        decoded_body: &[u8],
        strict: StrictParseView<'_>,
        limits: &DecodeLimits,
    ) -> NativeDecodeOutcome<NativeEntity>;
}

/// Native decoder that deliberately produces no entity.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopNativeEntityDecoder;

impl NativeEntityDecoder<()> for NoopNativeEntityDecoder {
    fn decode(
        &self,
        _declared_content_type: Option<&str>,
        _decoded_body: &[u8],
        _strict: StrictParseView<'_>,
        _limits: &DecodeLimits,
    ) -> NativeDecodeOutcome<()> {
        NativeDecodeOutcome::none()
    }
}

/// Immutable facts produced by transport/status/strict parse classification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttemptFacts {
    /// Stable physical source identity.
    pub source_id: String,
    /// Per-source event sequence.
    pub source_record_seq: u64,
    /// Per-source network attempt sequence when IO began.
    pub request_attempt_seq: Option<u64>,
    /// Terminal archive outcome.
    pub outcome: SourceOutcome,
    /// HTTP status when received.
    pub http_status: Option<u16>,
    /// Exact allowlisted declared media type.
    pub declared_media_type: Option<String>,
    /// Optional cadence deadline.
    pub scheduled_ns: Option<i64>,
    /// Request start Clock instant.
    pub request_start_ns: Option<i64>,
    /// First byte Clock instant.
    pub first_byte_ns: Option<i64>,
    /// Capture Clock instant.
    pub capture_ns: Option<i64>,
    /// Non-negative endpoint latency.
    pub latency_ns: Option<i64>,
    /// Stable error category for non-success outcomes.
    pub error_kind: Option<String>,
    /// Bounded redaction-safe diagnostic.
    pub error_message: Option<String>,
}

/// Pure decode result used by native delivery then archive admission.
#[derive(Clone, Debug)]
pub struct DecodedAttempt<ArchiveEntity, NativeEntity> {
    /// Immutable all-outcome attempt facts.
    pub facts: AttemptFacts,
    /// Strict archive entity only after an atomic successful parse.
    pub strict_archive_entity: Option<ArchiveEntity>,
    /// Separately decoded/projected native entity.
    pub native_entity: Option<NativeEntity>,
    /// Strict selected-format outcome.
    pub strict_parse_outcome: ParseOutcome,
    /// Explicit second-grammar native compatibility evidence.
    pub native_compatibility: Option<CompatibilityFallback>,
    /// Opaque exact bytes reserved for raw/archive projection.
    pub exact_entity: Option<ExactEntityLease>,
}

/// Pure bounded attempt decoder seam.
pub trait AttemptDecoder<ArchiveEntity, NativeEntity>: Debug + Send + Sync {
    /// Produces one terminal all-outcome result and never a partial archive entity.
    fn decode(
        &self,
        fetched: FetchedAttempt,
        limits: &DecodeLimits,
    ) -> DecodedAttempt<ArchiveEntity, NativeEntity>;
}

/// Strict exposition decoder composed with one native-only decoder.
pub struct PrometheusAttemptDecoder<NativeEntity> {
    parser: Arc<dyn ExpositionParser>,
    native: Arc<dyn NativeEntityDecoder<NativeEntity>>,
}

impl<NativeEntity> PrometheusAttemptDecoder<NativeEntity> {
    /// Composes the strict parser and separately named native decoder.
    #[must_use]
    pub fn new(
        parser: Arc<dyn ExpositionParser>,
        native: Arc<dyn NativeEntityDecoder<NativeEntity>>,
    ) -> Self {
        Self { parser, native }
    }
}

impl<NativeEntity> Debug for PrometheusAttemptDecoder<NativeEntity> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrometheusAttemptDecoder")
            .field("parser", &self.parser)
            .field("native", &self.native)
            .finish()
    }
}

impl<NativeEntity> AttemptDecoder<Exposition, NativeEntity>
    for PrometheusAttemptDecoder<NativeEntity>
{
    fn decode(
        &self,
        fetched: FetchedAttempt,
        limits: &DecodeLimits,
    ) -> DecodedAttempt<Exposition, NativeEntity> {
        let FetchedAttempt {
            source_id,
            source_record_seq,
            request_attempt_seq,
            scheduled_ns,
            request_start_ns,
            first_byte_ns,
            capture_ns,
            latency_ns,
            disposition,
        } = fetched;
        let base =
            |outcome, http_status, declared_media_type, error_kind, error_message| AttemptFacts {
                source_id: source_id.clone(),
                source_record_seq,
                request_attempt_seq,
                outcome,
                http_status,
                declared_media_type,
                scheduled_ns,
                request_start_ns,
                first_byte_ns,
                capture_ns,
                latency_ns,
                error_kind,
                error_message,
            };

        let (status, content_type, encoded_body, decoded_body) = match disposition {
            FetchDisposition::Response {
                status,
                content_type,
                content_encoding: _,
                encoded_body,
                decoded_body,
            } => (status, content_type, encoded_body, decoded_body),
            other => {
                let (outcome, kind, message) = match other {
                    FetchDisposition::Transport { kind, message } => {
                        (SourceOutcome::Transport, kind, message)
                    }
                    FetchDisposition::Timeout { request_started: _ } => (
                        SourceOutcome::Timeout,
                        "timeout".to_owned(),
                        "telemetry request exceeded its absolute deadline".to_owned(),
                    ),
                    FetchDisposition::Disabled { reason } => {
                        (SourceOutcome::Disabled, "disabled".to_owned(), reason)
                    }
                    FetchDisposition::Shutdown => (
                        SourceOutcome::Shutdown,
                        "shutdown".to_owned(),
                        "telemetry source stopped during shutdown".to_owned(),
                    ),
                    FetchDisposition::Response { .. } => unreachable!(),
                };
                return DecodedAttempt {
                    facts: base(
                        outcome,
                        None,
                        None,
                        Some(kind),
                        Some(bounded_diagnostic(&message, limits.max_diagnostic_bytes)),
                    ),
                    strict_archive_entity: None,
                    native_entity: None,
                    strict_parse_outcome: ParseOutcome::NotAttempted,
                    native_compatibility: None,
                    exact_entity: None,
                };
            }
        };

        let exact_entity = Some(ExactEntityLease::new(
            encoded_body.clone(),
            decoded_body.clone(),
        ));
        if !(200..300).contains(&status) {
            return DecodedAttempt {
                facts: base(
                    SourceOutcome::Http,
                    Some(status),
                    content_type,
                    Some("http_status".to_owned()),
                    Some(format!("telemetry endpoint returned HTTP {status}")),
                ),
                strict_archive_entity: None,
                native_entity: None,
                strict_parse_outcome: ParseOutcome::NotAttempted,
                native_compatibility: None,
                exact_entity,
            };
        }

        if let Some(error) = check_body_limits(&encoded_body, &decoded_body, limits) {
            return DecodedAttempt {
                facts: base(
                    SourceOutcome::UnsupportedFeature,
                    Some(status),
                    content_type,
                    Some("decode_limit".to_owned()),
                    Some(bounded_diagnostic(&error, limits.max_diagnostic_bytes)),
                ),
                strict_archive_entity: None,
                native_entity: None,
                strict_parse_outcome: ParseOutcome::NotAttempted,
                native_compatibility: None,
                exact_entity,
            };
        }

        let Some(declared) = content_type.as_deref() else {
            return DecodedAttempt {
                facts: base(
                    SourceOutcome::UnsupportedFormat,
                    Some(status),
                    None,
                    Some("missing_content_type".to_owned()),
                    Some("telemetry response omitted Content-Type".to_owned()),
                ),
                strict_archive_entity: None,
                native_entity: None,
                strict_parse_outcome: ParseOutcome::NotAttempted,
                native_compatibility: None,
                exact_entity,
            };
        };
        let format = match ExpositionFormat::from_content_type(declared) {
            Ok(format) => format,
            Err(error) => {
                return DecodedAttempt {
                    facts: base(
                        SourceOutcome::UnsupportedFormat,
                        Some(status),
                        content_type,
                        Some("unsupported_format".to_owned()),
                        Some(bounded_diagnostic(
                            &error.to_string(),
                            limits.max_diagnostic_bytes,
                        )),
                    ),
                    strict_archive_entity: None,
                    native_entity: None,
                    strict_parse_outcome: ParseOutcome::NotAttempted,
                    native_compatibility: None,
                    exact_entity,
                };
            }
        };

        let strict_result = self.parser.parse(format, &decoded_body, &limits.parse);
        let strict_view = match &strict_result {
            Ok(exposition) => StrictParseView::Success(exposition),
            Err(error) => StrictParseView::Failed { format, error },
        };
        let native =
            self.native
                .decode(content_type.as_deref(), &decoded_body, strict_view, limits);
        match strict_result {
            Ok(exposition) => {
                let outcome = if exposition.families.is_empty() && exposition.wire_sample_count == 0
                {
                    SourceOutcome::Empty
                } else {
                    SourceOutcome::Success
                };
                DecodedAttempt {
                    facts: base(outcome, Some(status), content_type, None, None),
                    strict_archive_entity: Some(exposition),
                    native_entity: native.entity,
                    strict_parse_outcome: ParseOutcome::Success { format },
                    native_compatibility: native.compatibility,
                    exact_entity,
                }
            }
            Err(error) => {
                let kind = if matches!(error.kind, ParseErrorKind::UnsupportedFeature) {
                    SourceOutcome::UnsupportedFeature
                } else {
                    SourceOutcome::Parse
                };
                let message = bounded_diagnostic(&error.to_string(), limits.max_diagnostic_bytes);
                DecodedAttempt {
                    facts: base(
                        kind,
                        Some(status),
                        content_type,
                        Some(parse_error_kind(&error).to_owned()),
                        Some(message),
                    ),
                    strict_archive_entity: None,
                    native_entity: native.entity,
                    strict_parse_outcome: ParseOutcome::Failed { format, error },
                    native_compatibility: native.compatibility,
                    exact_entity,
                }
            }
        }
    }
}

fn check_body_limits(encoded: &[u8], decoded: &[u8], limits: &DecodeLimits) -> Option<String> {
    if encoded.len() > limits.max_encoded_bytes {
        return Some(format!(
            "encoded telemetry body has {} bytes; limit is {}",
            encoded.len(),
            limits.max_encoded_bytes
        ));
    }
    if decoded.len() > limits.max_decoded_bytes {
        return Some(format!(
            "decoded telemetry body has {} bytes; limit is {}",
            decoded.len(),
            limits.max_decoded_bytes
        ));
    }
    let base = encoded.len().max(1);
    let allowed =
        base.saturating_mul(usize::try_from(limits.max_expansion_ratio).unwrap_or(usize::MAX));
    if decoded.len() > allowed {
        return Some(format!(
            "decoded telemetry body expansion {}:{} exceeds ratio {}",
            decoded.len(),
            encoded.len(),
            limits.max_expansion_ratio
        ));
    }
    None
}

fn bounded_diagnostic(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

fn parse_error_kind(error: &ParseError) -> &'static str {
    match error.kind {
        ParseErrorKind::InvalidUtf8 => "invalid_utf8",
        ParseErrorKind::LimitExceeded(_) => "parse_limit",
        ParseErrorKind::Syntax => "syntax",
        ParseErrorKind::Metadata => "metadata",
        ParseErrorKind::Number => "number",
        ParseErrorKind::Label => "label",
        ParseErrorKind::Exemplar => "exemplar",
        ParseErrorKind::EndOfFile => "eof",
        ParseErrorKind::Semantic => "semantic",
        ParseErrorKind::UnsupportedFeature => "unsupported_feature",
    }
}

/// Invalid bound profile discovered before source preparation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeConfigError {
    /// A hard bound was configured as zero.
    ZeroLimit(&'static str),
    /// Decoder and parser disagree about decoded entity capacity.
    DecodedLimitMismatch {
        /// Outer decoder bound.
        decode: usize,
        /// Strict parser bound.
        parser: usize,
    },
}

impl Display for DecodeConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLimit(field) => write!(formatter, "decode limit {field} must be positive"),
            Self::DecodedLimitMismatch { decode, parser } => write!(
                formatter,
                "decoded body limit {decode} differs from strict parser limit {parser}"
            ),
        }
    }
}

impl std::error::Error for DecodeConfigError {}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aiperf_prometheus::{StrictExpositionParser, parse_number_lexeme};

    use super::*;

    #[derive(Debug)]
    struct CountingParser {
        calls: Arc<AtomicUsize>,
    }

    impl ExpositionParser for CountingParser {
        fn parse(
            &self,
            format: ExpositionFormat,
            exact_body: &[u8],
            limits: &ParseLimits,
        ) -> Result<Exposition, ParseError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            StrictExpositionParser.parse(format, exact_body, limits)
        }
    }

    fn response(status: u16, content_type: &str, body: &'static [u8]) -> FetchedAttempt {
        FetchedAttempt {
            source_id: "source-a".to_owned(),
            source_record_seq: 0,
            request_attempt_seq: Some(0),
            scheduled_ns: Some(10),
            request_start_ns: Some(10),
            first_byte_ns: Some(11),
            capture_ns: Some(12),
            latency_ns: Some(2),
            disposition: FetchDisposition::Response {
                status,
                content_type: Some(content_type.to_owned()),
                content_encoding: None,
                encoded_body: Bytes::from_static(body),
                decoded_body: Bytes::from_static(body),
            },
        }
    }

    #[test]
    fn metric_looking_non_2xx_body_never_reaches_either_decoder() {
        let calls = Arc::new(AtomicUsize::new(0));
        let decoder = PrometheusAttemptDecoder::new(
            Arc::new(CountingParser {
                calls: calls.clone(),
            }),
            Arc::new(NoopNativeEntityDecoder),
        );
        let decoded = decoder.decode(
            response(
                500,
                "text/plain; version=0.0.4; charset=utf-8",
                b"# TYPE lie gauge\nlie 42\n",
            ),
            &DecodeLimits::default(),
        );
        assert_eq!(decoded.facts.outcome, SourceOutcome::Http);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(decoded.strict_archive_entity.is_none());
        assert!(decoded.native_entity.is_none());
    }

    #[test]
    fn exact_bytes_and_float64_precision_survive_strict_decode() {
        let decoder = PrometheusAttemptDecoder::new(
            Arc::new(StrictExpositionParser),
            Arc::new(NoopNativeEntityDecoder),
        );
        let body = b"# TYPE precise gauge\nprecise 16777217\n";
        let decoded = decoder.decode(
            response(200, "text/plain; version=0.0.4; charset=utf-8", body),
            &DecodeLimits::default(),
        );
        assert_eq!(decoded.facts.outcome, SourceOutcome::Success);
        let lease = decoded.exact_entity.unwrap();
        assert_eq!(lease.encoded(), body);
        assert_eq!(lease.decoded(), body);
        let key = crate::Blake3ArchiveKeyProvider::new("fixture_key", [9; 32]).unwrap();
        assert_ne!(
            lease.encoded_digest(&key).unwrap(),
            lease.decoded_digest(&key).unwrap()
        );
        let point = &decoded.strict_archive_entity.unwrap().families[0].metrics[0].points[0];
        assert_eq!(
            point.wire_samples[0].value,
            parse_number_lexeme(ExpositionFormat::PrometheusText004, "16777217").unwrap()
        );
    }

    #[derive(Debug)]
    struct ClassicFallback;

    impl NativeEntityDecoder<usize> for ClassicFallback {
        fn decode(
            &self,
            _declared_content_type: Option<&str>,
            decoded_body: &[u8],
            strict: StrictParseView<'_>,
            limits: &DecodeLimits,
        ) -> NativeDecodeOutcome<usize> {
            if !matches!(strict, StrictParseView::Failed { .. }) {
                return NativeDecodeOutcome::none();
            }
            match StrictExpositionParser.parse(
                ExpositionFormat::PrometheusText004,
                decoded_body,
                &limits.parse,
            ) {
                Ok(entity) => NativeDecodeOutcome {
                    entity: Some(entity.wire_sample_count),
                    compatibility: Some(CompatibilityFallback {
                        format: ExpositionFormat::PrometheusText004,
                        succeeded: true,
                        error_message: None,
                    }),
                },
                Err(error) => NativeDecodeOutcome {
                    entity: None,
                    compatibility: Some(CompatibilityFallback {
                        format: ExpositionFormat::PrometheusText004,
                        succeeded: false,
                        error_message: Some(error.to_string()),
                    }),
                },
            }
        }
    }

    #[test]
    fn native_fallback_never_reclassifies_strict_openmetrics_failure() {
        let decoder = PrometheusAttemptDecoder::new(
            Arc::new(StrictExpositionParser),
            Arc::new(ClassicFallback),
        );
        let decoded = decoder.decode(
            response(
                200,
                "application/openmetrics-text; version=1.0.0; charset=utf-8",
                b"# TYPE requests counter\nrequests_total 2\n",
            ),
            &DecodeLimits::default(),
        );
        assert_eq!(decoded.facts.outcome, SourceOutcome::Parse);
        assert!(decoded.strict_archive_entity.is_none());
        assert_eq!(decoded.native_entity, Some(1));
        assert_eq!(
            decoded.native_compatibility,
            Some(CompatibilityFallback {
                format: ExpositionFormat::PrometheusText004,
                succeeded: true,
                error_message: None,
            })
        );
    }

    #[test]
    fn oversized_entities_fail_atomically_without_truncation() {
        let calls = Arc::new(AtomicUsize::new(0));
        let decoder = PrometheusAttemptDecoder::new(
            Arc::new(CountingParser {
                calls: calls.clone(),
            }),
            Arc::new(NoopNativeEntityDecoder),
        );
        let limits = DecodeLimits {
            max_encoded_bytes: 4,
            ..DecodeLimits::default()
        };
        let decoded = decoder.decode(
            response(
                200,
                "text/plain; version=0.0.4; charset=utf-8",
                b"metric 1\n",
            ),
            &limits,
        );
        assert_eq!(decoded.facts.outcome, SourceOutcome::UnsupportedFeature);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(decoded.exact_entity.unwrap().decoded(), b"metric 1\n");
    }
}
