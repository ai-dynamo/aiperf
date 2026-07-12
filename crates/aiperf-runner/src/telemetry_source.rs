// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen telemetry source factories for standalone and attached archives.
//!
//! A factory validates its own strict authored object and returns a prepared
//! complete source driver. The registry never recovers concrete entity types
//! through source IDs or `Any`; a validated source object owns its one
//! transition into the generic driver seam.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};
use std::rc::Rc;
use std::sync::Arc;

use aiperf_clock::Clock;
use aiperf_telemetry_archive::{
    ArchiveSourceError, CanonicalJsonValue, ContentEncodingV1, EntityDecodeLimitsV1,
    EntityDecodePolicyV1, FetchDisposition, FetchRequest, FetchedAttempt,
    FixedDeadlineTelemetryDriver, LocalCancellationSignal, PreparedTelemetryDriver,
    TelemetryAttemptConsumer, TelemetryDriverConfig, TelemetryFetcher,
};
use aiperf_transport_http::config::ClientConfig;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use url::Url;

use crate::control_plane_http::{
    ControlPlaneCredentialReference, ControlPlaneHttp, ControlPlaneHttpErrorKind,
    ControlPlaneHttpProvider, ControlPlaneRequest, ControlPlaneTlsReference,
    ValidatedControlPlaneProfile,
};

/// Stable factory capability facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveSourceDescriptor {
    /// Frozen wire ID.
    pub id: &'static str,
    /// Human-readable source description.
    pub description: &'static str,
}

/// Context available during side-effect-free source validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveSourceValidationContext {
    /// Positive outer per-call deadline ceiling.
    pub request_timeout_ns: i64,
}

/// LocalSet-owned preparation context after all source configs validate.
pub struct ArchiveSourcePrepareContext {
    /// Stable run-local physical source identity.
    pub source_id: String,
    /// Positive anchor-relative cadence interval.
    pub interval_ns: i64,
    /// Positive source request lifetime.
    pub request_timeout_ns: i64,
    /// Optional absolute run deadline.
    pub run_deadline_ns: Option<i64>,
    /// The run's one injected Clock.
    pub clock: Rc<dyn Clock>,
    /// Backend-provided isolated control-plane capability.
    pub control_plane: Rc<dyn ControlPlaneHttpProvider>,
    /// Ordered decode/native/archive consumer composed for this source.
    pub consumer: Rc<dyn TelemetryAttemptConsumer>,
}

impl Debug for ArchiveSourcePrepareContext {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArchiveSourcePrepareContext")
            .field("source_id", &self.source_id)
            .field("interval_ns", &self.interval_ns)
            .field("request_timeout_ns", &self.request_timeout_ns)
            .field("run_deadline_ns", &self.run_deadline_ns)
            .field("virtual_clock", &self.clock.is_virtual())
            .field("control_plane", &self.control_plane)
            .field("consumer", &self.consumer)
            .finish()
    }
}

/// Factory-owned validated source ready for one preparation.
pub trait ValidatedArchiveSource: Debug + Send + Sync {
    /// Factory-produced canonical persistent source identity with defaults explicit.
    fn persistent_identity(&self) -> &CanonicalJsonValue;

    /// Worst-case exact content-encoded entity bytes accepted from this source.
    ///
    /// Archive admission uses this static side-effect-free bound before source
    /// preparation so an accepted projection cannot outgrow its reserved input
    /// footprint.
    fn maximum_encoded_entity_bytes(&self) -> usize;

    /// Worst-case exact content-decoded entity bytes produced by this source.
    ///
    /// This remains independent from the encoded bound because compressed
    /// sources reserve both representations through decode and raw projection.
    fn maximum_decoded_entity_bytes(&self) -> usize;

    /// Prepare the complete already-typed driver without a concrete downcast.
    fn prepare(
        self: Box<Self>,
        context: ArchiveSourcePrepareContext,
    ) -> Result<Box<dyn PreparedTelemetryDriver>, ArchiveSourceFactoryError>;
}

/// Strict source validation seam.
pub trait ArchiveSourceFactory: Debug + Send + Sync {
    /// Frozen source identity/capability facts.
    fn descriptor(&self) -> &'static ArchiveSourceDescriptor;

    /// Decode and validate exactly this factory's authored object.
    fn validate(
        &self,
        config: &RawValue,
        context: ArchiveSourceValidationContext,
    ) -> Result<Box<dyn ValidatedArchiveSource>, ArchiveSourceFactoryError>;
}

/// Immutable source-factory universe.
#[derive(Clone)]
pub struct ArchiveSourceFactoryRegistry {
    factories: Arc<BTreeMap<String, Arc<dyn ArchiveSourceFactory>>>,
}

impl ArchiveSourceFactoryRegistry {
    /// Freeze unique, syntactically valid source factory IDs.
    pub fn new(
        factories: impl IntoIterator<Item = Arc<dyn ArchiveSourceFactory>>,
    ) -> Result<Self, ArchiveSourceFactoryError> {
        let mut by_id = BTreeMap::new();
        for factory in factories {
            let id = factory.descriptor().id;
            validate_component_id(id)?;
            if by_id.insert(id.to_owned(), factory).is_some() {
                return Err(ArchiveSourceFactoryError::DuplicateFactory(id.to_owned()));
            }
        }
        Ok(Self {
            factories: Arc::new(by_id),
        })
    }

    /// Stock source registry for this exact runner distribution.
    pub fn stock() -> Self {
        Self::new([Arc::new(PrometheusArchiveSourceFactory) as Arc<dyn ArchiveSourceFactory>])
            .expect("stock telemetry source IDs are valid and unique")
    }

    /// Resolve and strictly validate one authored source config.
    pub fn validate(
        &self,
        source_type: &str,
        config: &RawValue,
        context: ArchiveSourceValidationContext,
    ) -> Result<Box<dyn ValidatedArchiveSource>, ArchiveSourceFactoryError> {
        let factory = self.factories.get(source_type).ok_or_else(|| {
            ArchiveSourceFactoryError::UnknownFactory {
                requested: source_type.to_owned(),
                available: self.factories.keys().cloned().collect(),
            }
        })?;
        factory.validate(config, context)
    }

    /// Deterministic frozen descriptors for capability identity.
    pub fn descriptors(&self) -> impl ExactSizeIterator<Item = &'static ArchiveSourceDescriptor> {
        self.factories.values().map(|factory| factory.descriptor())
    }
}

impl Debug for ArchiveSourceFactoryRegistry {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArchiveSourceFactoryRegistry")
            .field("ids", &self.factories.keys().collect::<Vec<_>>())
            .finish()
    }
}

static PROMETHEUS_SOURCE_DESCRIPTOR: ArchiveSourceDescriptor = ArchiveSourceDescriptor {
    id: "prometheus_http",
    description: "strict Prometheus/OpenMetrics exposition over isolated native HTTP",
};

/// Built-in strict Prometheus/OpenMetrics HTTP factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrometheusArchiveSourceFactory;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PrometheusHttpSourceConfigV2 {
    url: String,
    #[serde(default)]
    credential_provider: Option<String>,
    #[serde(default)]
    tls: TlsProviderSpecV2,
    connect_timeout_ns: i64,
    redirects: DisabledPolicyV2,
    proxy: DisabledPolicyV2,
    accepted_formats: Vec<PrometheusFormatV2>,
    #[serde(default = "default_content_encodings")]
    accepted_content_encodings: Vec<PrometheusContentEncodingV2>,
    max_compressed_bytes: usize,
    max_decompressed_bytes: usize,
    #[serde(default = "default_max_expansion_ratio")]
    max_expansion_ratio: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TlsProviderSpecV2 {
    #[serde(default)]
    trust_provider: Option<String>,
    #[serde(default)]
    mtls_provider: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DisabledPolicyV2 {
    Disabled,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum PrometheusFormatV2 {
    #[serde(rename = "prometheus_text_0_0_4")]
    PrometheusText0_0_4,
    #[serde(rename = "openmetrics_text_1_0_0")]
    OpenmetricsText1_0_0,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum PrometheusContentEncodingV2 {
    Gzip,
    Identity,
}

impl PrometheusContentEncodingV2 {
    const fn archive(self) -> ContentEncodingV1 {
        match self {
            Self::Gzip => ContentEncodingV1::Gzip,
            Self::Identity => ContentEncodingV1::Identity,
        }
    }
}

impl PrometheusFormatV2 {
    const fn media_type(self) -> &'static str {
        match self {
            Self::PrometheusText0_0_4 => "text/plain; version=0.0.4",
            Self::OpenmetricsText1_0_0 => "application/openmetrics-text; version=1.0.0",
        }
    }
}

const fn default_max_expansion_ratio() -> u64 {
    100
}

fn default_content_encodings() -> Vec<PrometheusContentEncodingV2> {
    vec![
        PrometheusContentEncodingV2::Gzip,
        PrometheusContentEncodingV2::Identity,
    ]
}

impl ArchiveSourceFactory for PrometheusArchiveSourceFactory {
    fn descriptor(&self) -> &'static ArchiveSourceDescriptor {
        &PROMETHEUS_SOURCE_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
        context: ArchiveSourceValidationContext,
    ) -> Result<Box<dyn ValidatedArchiveSource>, ArchiveSourceFactoryError> {
        if context.request_timeout_ns <= 0 {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "source request_timeout_ns must be positive".to_owned(),
            ));
        }
        let mut config: PrometheusHttpSourceConfigV2 =
            serde_json::from_str(config.get()).map_err(|error| {
                ArchiveSourceFactoryError::InvalidConfig(format!(
                    "prometheus_http source config: {error}"
                ))
            })?;
        let url = Url::parse(&config.url).map_err(|error| {
            ArchiveSourceFactoryError::InvalidConfig(format!(
                "prometheus_http URL is invalid: {error}"
            ))
        })?;
        if config.connect_timeout_ns <= 0 || config.connect_timeout_ns > context.request_timeout_ns
        {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "prometheus_http connect_timeout_ns must be positive and no greater than request_timeout_ns"
                    .to_owned(),
            ));
        }
        if config.max_compressed_bytes == 0
            || config.max_decompressed_bytes == 0
            || config.max_expansion_ratio == 0
        {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "prometheus_http body limits are zero or inconsistent".to_owned(),
            ));
        }
        if config.accepted_formats.is_empty() {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "prometheus_http accepted_formats cannot be empty".to_owned(),
            ));
        }
        let DisabledPolicyV2::Disabled = config.redirects;
        let DisabledPolicyV2::Disabled = config.proxy;
        let format_count = config.accepted_formats.len();
        let formats = config
            .accepted_formats
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if formats.len() != format_count {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "prometheus_http accepted_formats must be unique".to_owned(),
            ));
        }
        config.accepted_formats = formats.iter().copied().collect();
        if config.accepted_content_encodings.is_empty() {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "prometheus_http accepted_content_encodings cannot be empty".to_owned(),
            ));
        }
        let encoding_count = config.accepted_content_encodings.len();
        let encodings = config
            .accepted_content_encodings
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if encodings.len() != encoding_count {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "prometheus_http accepted_content_encodings must be unique".to_owned(),
            ));
        }
        config.accepted_content_encodings = encodings.iter().copied().collect();
        let entity_policy = EntityDecodePolicyV1::new(
            encodings
                .iter()
                .copied()
                .map(PrometheusContentEncodingV2::archive),
            EntityDecodeLimitsV1 {
                max_encoded_bytes: config.max_compressed_bytes,
                max_decoded_bytes: config.max_decompressed_bytes,
                max_expansion_ratio: config.max_expansion_ratio,
            },
        )
        .map_err(|error| {
            ArchiveSourceFactoryError::InvalidConfig(format!(
                "prometheus_http content decoding: {error}"
            ))
        })?;
        let persistent_identity =
            CanonicalJsonValue::parse(&serde_json::to_vec(&config).map_err(|error| {
                ArchiveSourceFactoryError::InvalidConfig(format!(
                    "prometheus_http canonical source identity: {error}"
                ))
            })?)
            .map_err(|error| {
                ArchiveSourceFactoryError::InvalidConfig(format!(
                    "prometheus_http canonical source identity: {error}"
                ))
            })?;
        let credential = config.credential_provider.map_or(
            ControlPlaneCredentialReference::None,
            ControlPlaneCredentialReference::BearerProvider,
        );
        let tls = ControlPlaneTlsReference {
            trust_provider: config.tls.trust_provider.clone(),
            mtls_provider: config.tls.mtls_provider.clone(),
        };
        let mut client = ClientConfig {
            connect_timeout_ns: Some(config.connect_timeout_ns),
            request_timeout_ns: None,
            total_timeout_ns: None,
            max_response_body_bytes: Some(u64::try_from(config.max_compressed_bytes).map_err(
                |_| {
                    ArchiveSourceFactoryError::InvalidConfig(
                        "prometheus_http max_compressed_bytes exceeds u64".to_owned(),
                    )
                },
            )?),
            max_connections_per_origin: 1,
            collect_trace_chunks: false,
            ..ClientConfig::default()
        };
        client.ssl_verify = true;
        let profile = ValidatedControlPlaneProfile::new(
            url,
            client,
            credential,
            tls,
            formats
                .into_iter()
                .map(|format| format.media_type().to_owned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            entity_policy
                .accepted_encodings()
                .iter()
                .map(|encoding| encoding.as_str().to_owned())
                .collect(),
            config.max_compressed_bytes,
        )
        .map_err(|error| ArchiveSourceFactoryError::InvalidConfig(error.to_string()))?;
        Ok(Box::new(ValidatedPrometheusArchiveSource {
            persistent_identity,
            profile,
            entity_policy,
        }))
    }
}

#[derive(Debug)]
struct ValidatedPrometheusArchiveSource {
    persistent_identity: CanonicalJsonValue,
    profile: ValidatedControlPlaneProfile,
    entity_policy: EntityDecodePolicyV1,
}

impl ValidatedArchiveSource for ValidatedPrometheusArchiveSource {
    fn persistent_identity(&self) -> &CanonicalJsonValue {
        &self.persistent_identity
    }

    fn maximum_encoded_entity_bytes(&self) -> usize {
        self.entity_policy.limits().max_encoded_bytes
    }

    fn maximum_decoded_entity_bytes(&self) -> usize {
        self.entity_policy.limits().max_decoded_bytes
    }

    fn prepare(
        self: Box<Self>,
        context: ArchiveSourcePrepareContext,
    ) -> Result<Box<dyn PreparedTelemetryDriver>, ArchiveSourceFactoryError> {
        if context.interval_ns <= 0 || context.request_timeout_ns <= 0 {
            return Err(ArchiveSourceFactoryError::InvalidConfig(
                "prepared source interval and request timeout must be positive".to_owned(),
            ));
        }
        let control = context
            .control_plane
            .prepare(self.profile)
            .map_err(|error| ArchiveSourceFactoryError::Prepare(error.to_string()))?;
        let fetcher: Rc<dyn TelemetryFetcher> = Rc::new(PrometheusTelemetryFetcher {
            control,
            entity_policy: self.entity_policy,
        });
        let driver = FixedDeadlineTelemetryDriver::new(
            TelemetryDriverConfig {
                source_id: context.source_id,
                interval_ns: context.interval_ns,
                request_timeout_ns: context.request_timeout_ns,
                run_deadline_ns: context.run_deadline_ns,
            },
            context.clock,
            fetcher,
            context.consumer,
        )
        .map_err(|error| ArchiveSourceFactoryError::Prepare(error.to_string()))?;
        Ok(Box::new(driver))
    }
}

struct PrometheusTelemetryFetcher {
    control: Rc<dyn ControlPlaneHttp>,
    entity_policy: EntityDecodePolicyV1,
}

impl Debug for PrometheusTelemetryFetcher {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrometheusTelemetryFetcher")
            .field("control", &self.control)
            .field("entity_policy", &self.entity_policy)
            .finish()
    }
}

#[async_trait(?Send)]
impl TelemetryFetcher for PrometheusTelemetryFetcher {
    async fn fetch(
        &self,
        request: FetchRequest,
        absolute_deadline_ns: i64,
        cancellation: LocalCancellationSignal,
    ) -> FetchedAttempt {
        let source_id = request.source_id.clone();
        let source_record_seq = request.source_record_seq;
        let request_attempt_seq = request.request_attempt_seq;
        let scheduled_ns = match request.kind {
            aiperf_telemetry_archive::SourceAttemptKind::Continuous(deadline) => {
                Some(deadline.scheduled_ns)
            }
            aiperf_telemetry_archive::SourceAttemptKind::Boundary { .. } => None,
        };
        let response = self
            .control
            .execute(
                ControlPlaneRequest {
                    request_id: format!("{source_id}:{source_record_seq}"),
                },
                absolute_deadline_ns,
                cancellation,
            )
            .await;
        match response {
            Ok(response) => {
                let content_type = response.headers.get("content-type").cloned();
                let content_encoding = response.headers.get("content-encoding").cloned();
                let end_ns = response.timings.end_ns;
                let start_ns = response.timings.request_start_ns;
                let first_byte_ns = response.timings.first_byte_ns;
                let disposition = FetchDisposition::EncodedResponse {
                    status: response.status,
                    content_type,
                    content_encoding,
                    encoded_body: response.encoded_body,
                    entity_policy: self.entity_policy.clone(),
                };
                FetchedAttempt {
                    source_id,
                    source_record_seq,
                    request_attempt_seq: Some(request_attempt_seq),
                    scheduled_ns,
                    request_start_ns: Some(start_ns),
                    first_byte_ns,
                    capture_ns: first_byte_ns.or(Some(end_ns)),
                    latency_ns: Some(end_ns.saturating_sub(start_ns).max(0)),
                    disposition,
                }
            }
            Err(error) => {
                let timings = error.timings;
                let request_start_ns = timings.as_ref().map(|value| value.request_start_ns);
                let first_byte_ns = timings.as_ref().and_then(|value| value.first_byte_ns);
                let latency_ns = timings
                    .as_ref()
                    .map(|value| value.end_ns.saturating_sub(value.request_start_ns).max(0));
                let disposition = match error.kind {
                    ControlPlaneHttpErrorKind::Timeout => FetchDisposition::Timeout {
                        request_started: request_start_ns.is_some(),
                    },
                    ControlPlaneHttpErrorKind::Cancelled => FetchDisposition::Shutdown,
                    ControlPlaneHttpErrorKind::InvalidRequest
                    | ControlPlaneHttpErrorKind::Transport
                    | ControlPlaneHttpErrorKind::InvalidResponse
                    | ControlPlaneHttpErrorKind::BodyTooLarge => FetchDisposition::Transport {
                        kind: control_error_kind(error.kind).to_owned(),
                        message: error.message,
                    },
                };
                FetchedAttempt {
                    source_id,
                    source_record_seq,
                    request_attempt_seq: request_start_ns.map(|_| request_attempt_seq),
                    scheduled_ns,
                    request_start_ns,
                    first_byte_ns,
                    capture_ns: None,
                    latency_ns,
                    disposition,
                }
            }
        }
    }

    async fn shutdown(&self) -> Result<(), ArchiveSourceError> {
        Ok(())
    }
}

const fn control_error_kind(kind: ControlPlaneHttpErrorKind) -> &'static str {
    match kind {
        ControlPlaneHttpErrorKind::InvalidRequest => "invalid_request",
        ControlPlaneHttpErrorKind::Transport => "transport",
        ControlPlaneHttpErrorKind::Timeout => "timeout",
        ControlPlaneHttpErrorKind::Cancelled => "cancelled",
        ControlPlaneHttpErrorKind::InvalidResponse => "invalid_response",
        ControlPlaneHttpErrorKind::BodyTooLarge => "encoded_body_limit",
    }
}

fn validate_component_id(value: &str) -> Result<(), ArchiveSourceFactoryError> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(ArchiveSourceFactoryError::InvalidFactoryId(
            value.to_owned(),
        ));
    }
    Ok(())
}

/// Source registry, validation, or preparation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArchiveSourceFactoryError {
    /// Factory descriptor ID is invalid.
    InvalidFactoryId(String),
    /// Two factories registered the same ID.
    DuplicateFactory(String),
    /// Authored source ID is not compiled into this runner.
    UnknownFactory {
        /// Requested wire ID.
        requested: String,
        /// Deterministic available IDs.
        available: Vec<String>,
    },
    /// Factory-owned authored object is invalid.
    InvalidConfig(String),
    /// Side-effectful source preparation failed.
    Prepare(String),
}

impl Display for ArchiveSourceFactoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFactoryId(id) => {
                write!(formatter, "invalid archive source factory ID {id:?}")
            }
            Self::DuplicateFactory(id) => {
                write!(formatter, "duplicate archive source factory ID {id:?}")
            }
            Self::UnknownFactory {
                requested,
                available,
            } => write!(
                formatter,
                "archive source factory {requested:?} is unavailable; compiled factories: {}",
                available.join(", ")
            ),
            Self::InvalidConfig(message) => formatter.write_str(message),
            Self::Prepare(message) => {
                write!(formatter, "preparing archive source failed: {message}")
            }
        }
    }
}

impl std::error::Error for ArchiveSourceFactoryError {}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn raw(value: serde_json::Value) -> Box<RawValue> {
        RawValue::from_string(value.to_string()).unwrap()
    }

    fn config() -> Box<RawValue> {
        raw(json!({
            "url": "http://127.0.0.1:9000/metrics",
            "credential_provider": null,
            "tls": {"trust_provider": null, "mtls_provider": null},
            "connect_timeout_ns": 1_000_000,
            "redirects": "disabled",
            "proxy": "disabled",
            "accepted_formats": ["prometheus_text_0_0_4", "openmetrics_text_1_0_0"],
            "max_compressed_bytes": 1024,
            "max_decompressed_bytes": 4096
        }))
    }

    #[test]
    fn stock_registry_strictly_validates_prometheus_source() {
        let registry = ArchiveSourceFactoryRegistry::stock();
        let validated = registry
            .validate(
                "prometheus_http",
                &config(),
                ArchiveSourceValidationContext {
                    request_timeout_ns: 2_000_000,
                },
            )
            .unwrap();
        let persistent = String::from_utf8(validated.persistent_identity().to_bytes()).unwrap();
        assert!(persistent.contains("\"accepted_content_encodings\":[\"gzip\",\"identity\"]"));
        assert_eq!(validated.maximum_encoded_entity_bytes(), 1024);
        assert_eq!(validated.maximum_decoded_entity_bytes(), 4096);
        assert_eq!(
            registry
                .descriptors()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>(),
            vec!["prometheus_http"]
        );
    }

    #[test]
    fn unknown_fields_and_unsafe_transport_policy_fail_closed() {
        let registry = ArchiveSourceFactoryRegistry::stock();
        let unknown = raw(json!({
            "url": "http://127.0.0.1:9000/metrics",
            "connect_timeout_ns": 1,
            "redirects": "disabled",
            "proxy": "disabled",
            "accepted_formats": ["prometheus_text_0_0_4"],
            "max_compressed_bytes": 1,
            "max_decompressed_bytes": 1,
            "headers": {"authorization": "secret"}
        }));
        assert!(
            registry
                .validate(
                    "prometheus_http",
                    &unknown,
                    ArchiveSourceValidationContext {
                        request_timeout_ns: 2,
                    },
                )
                .is_err()
        );

        for (field, value) in [("redirects", "same_origin"), ("proxy", "ambient")] {
            let mut unsafe_policy: serde_json::Value =
                serde_json::from_str(config().get()).unwrap();
            unsafe_policy[field] = json!(value);
            assert!(
                registry
                    .validate(
                        "prometheus_http",
                        &raw(unsafe_policy),
                        ArchiveSourceValidationContext {
                            request_timeout_ns: 2_000_000,
                        },
                    )
                    .is_err(),
                "unsafe {field} policy unexpectedly validated"
            );
        }

        let mut too_slow: serde_json::Value = serde_json::from_str(config().get()).unwrap();
        too_slow["connect_timeout_ns"] = json!(3_000_000);
        assert!(
            registry
                .validate(
                    "prometheus_http",
                    &raw(too_slow),
                    ArchiveSourceValidationContext {
                        request_timeout_ns: 2_000_000,
                    },
                )
                .is_err()
        );
    }

    #[test]
    fn unknown_factory_lists_only_frozen_choices() {
        let error = ArchiveSourceFactoryRegistry::stock()
            .validate(
                "private_source",
                &config(),
                ArchiveSourceValidationContext {
                    request_timeout_ns: 2_000_000,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("prometheus_http"));
    }

    #[test]
    fn content_encoding_policy_is_closed_unique_and_independently_bounded() {
        let registry = ArchiveSourceFactoryRegistry::stock();
        let mut duplicate: serde_json::Value = serde_json::from_str(config().get()).unwrap();
        duplicate["accepted_content_encodings"] = json!(["gzip", "gzip"]);
        assert!(
            registry
                .validate(
                    "prometheus_http",
                    &raw(duplicate),
                    ArchiveSourceValidationContext {
                        request_timeout_ns: 2_000_000,
                    },
                )
                .is_err()
        );

        let mut unknown: serde_json::Value = serde_json::from_str(config().get()).unwrap();
        unknown["accepted_content_encodings"] = json!(["zstd"]);
        assert!(
            registry
                .validate(
                    "prometheus_http",
                    &raw(unknown),
                    ArchiveSourceValidationContext {
                        request_timeout_ns: 2_000_000,
                    },
                )
                .is_err()
        );

        let mut independent: serde_json::Value = serde_json::from_str(config().get()).unwrap();
        independent["max_compressed_bytes"] = json!(4096);
        independent["max_decompressed_bytes"] = json!(1024);
        assert!(
            registry
                .validate(
                    "prometheus_http",
                    &raw(independent),
                    ArchiveSourceValidationContext {
                        request_timeout_ns: 2_000_000,
                    },
                )
                .is_ok()
        );
    }

    #[test]
    fn tls_and_mtls_provider_ids_are_validated_without_resolving_material() {
        let registry = ArchiveSourceFactoryRegistry::stock();
        let mut secure: serde_json::Value = serde_json::from_str(config().get()).unwrap();
        secure["url"] = json!("https://node-a.example.test/metrics");
        secure["credential_provider"] = json!("node-metrics");
        secure["tls"] = json!({
            "trust_provider": "cluster-ca",
            "mtls_provider": "node-client"
        });
        let validated = registry
            .validate(
                "prometheus_http",
                &raw(secure),
                ArchiveSourceValidationContext {
                    request_timeout_ns: 2_000_000,
                },
            )
            .unwrap();
        let identity = String::from_utf8(validated.persistent_identity().to_bytes()).unwrap();
        assert!(identity.contains("\"trust_provider\":\"cluster-ca\""));
        assert!(identity.contains("\"mtls_provider\":\"node-client\""));

        let mut cleartext: serde_json::Value = serde_json::from_str(config().get()).unwrap();
        cleartext["tls"] = json!({"trust_provider": "cluster-ca", "mtls_provider": null});
        assert!(
            registry
                .validate(
                    "prometheus_http",
                    &raw(cleartext),
                    ArchiveSourceValidationContext {
                        request_timeout_ns: 2_000_000,
                    },
                )
                .is_err()
        );

        let mut invalid: serde_json::Value = serde_json::from_str(config().get()).unwrap();
        invalid["url"] = json!("https://node-a.example.test/metrics");
        invalid["tls"] = json!({"trust_provider": " padded ", "mtls_provider": null});
        assert!(
            registry
                .validate(
                    "prometheus_http",
                    &raw(invalid),
                    ArchiveSourceValidationContext {
                        request_timeout_ns: 2_000_000,
                    },
                )
                .is_err()
        );
    }
}
