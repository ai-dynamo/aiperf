// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile-bound control-plane HTTP over the native Clock-injected transport.
//!
//! Handles created here own a dedicated connection pool and expose only
//! allowlisted response metadata. Endpoint credentials are resolved during
//! preparation and never enter authored DTOs, durable source descriptors, or
//! returned attempt facts.

use std::collections::BTreeMap;
use std::fmt::{self, Debug, Display, Formatter};
use std::rc::Rc;
use std::sync::Arc;

use aiperf_clock::Clock;
use aiperf_telemetry_archive::LocalCancellationSignal;
use aiperf_transport_http::config::ClientConfig;
use aiperf_transport_http::models::{
    ConnectionReuseStrategy, ErrorKind, RequestConfig, Response, TraceData,
};
use aiperf_transport_http::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use bytes::Bytes;
use url::Url;

const ALLOWLISTED_RESPONSE_HEADERS: &[&str] = &[
    "cache-control",
    "content-encoding",
    "content-type",
    "etag",
    "last-modified",
];

/// Secret reference retained in a validated profile without resolving bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControlPlaneCredentialReference {
    /// No request credential is required.
    None,
    /// Provider-owned bearer token selected by stable ID.
    BearerProvider(String),
}

/// Secret bytes whose debug output is always redacted.
#[derive(Clone, Eq, PartialEq)]
pub struct ResolvedSecret(String);

impl ResolvedSecret {
    /// Construct one non-empty provider result.
    pub fn new(value: impl Into<String>) -> Result<Self, SecretResolutionError> {
        let value = value.into();
        if value.is_empty() || value.contains(['\r', '\n', '\0']) {
            return Err(SecretResolutionError::InvalidSecret);
        }
        Ok(Self(value))
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl Debug for ResolvedSecret {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("ResolvedSecret([REDACTED])")
    }
}

/// Deployment-owned credential resolution seam.
pub trait SecretProviderResolver: Debug + Send + Sync {
    /// Resolve one bearer token without logging or serializing its bytes.
    fn resolve_bearer(&self, provider_id: &str) -> Result<ResolvedSecret, SecretResolutionError>;
}

/// Default resolver that permits only credential-free sources.
#[derive(Clone, Copy, Debug, Default)]
pub struct RejectingSecretProviderResolver;

impl SecretProviderResolver for RejectingSecretProviderResolver {
    fn resolve_bearer(&self, provider_id: &str) -> Result<ResolvedSecret, SecretResolutionError> {
        Err(SecretResolutionError::Unavailable(provider_id.to_owned()))
    }
}

/// Strict transport profile after source-factory validation.
#[derive(Clone, Debug)]
pub struct ValidatedControlPlaneProfile {
    url: Url,
    client: ClientConfig,
    credential: ControlPlaneCredentialReference,
    accepted_media_types: Vec<String>,
    max_encoded_bytes: usize,
}

impl ValidatedControlPlaneProfile {
    /// Validate and freeze a secret-free control-plane profile.
    pub fn new(
        url: Url,
        client: ClientConfig,
        credential: ControlPlaneCredentialReference,
        accepted_media_types: Vec<String>,
        max_encoded_bytes: usize,
    ) -> Result<Self, ControlPlanePrepareError> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "control-plane URL scheme must be http or https".to_owned(),
            ));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "control-plane URL must not contain userinfo".to_owned(),
            ));
        }
        if url.host_str().is_none_or(str::is_empty)
            || url.fragment().is_some()
            || url.query().is_some()
        {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "control-plane URL requires a host and forbids query/fragment data".to_owned(),
            ));
        }
        if client.max_connections_per_origin != 1 {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "a physical telemetry source must own exactly one control-plane connection slot"
                    .to_owned(),
            ));
        }
        if client
            .connect_timeout_ns
            .is_some_and(|timeout| timeout <= 0)
        {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "control-plane connect timeout must be positive".to_owned(),
            ));
        }
        if max_encoded_bytes == 0 {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "control-plane encoded-body limit must be positive".to_owned(),
            ));
        }
        if accepted_media_types.is_empty() {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "control-plane profile must accept at least one media type".to_owned(),
            ));
        }
        let mut preceding: Option<&str> = None;
        for media_type in &accepted_media_types {
            if media_type.is_empty()
                || media_type.trim() != media_type
                || media_type.contains(['\r', '\n', '\0'])
            {
                return Err(ControlPlanePrepareError::InvalidProfile(
                    "control-plane accepted media type is invalid".to_owned(),
                ));
            }
            if preceding.is_some_and(|value| value >= media_type.as_str()) {
                return Err(ControlPlanePrepareError::InvalidProfile(
                    "control-plane accepted media types must be sorted and unique".to_owned(),
                ));
            }
            preceding = Some(media_type);
        }
        if let ControlPlaneCredentialReference::BearerProvider(provider_id) = &credential
            && (provider_id.is_empty()
                || provider_id.trim() != provider_id
                || provider_id.chars().any(char::is_control))
        {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "credential provider ID is invalid".to_owned(),
            ));
        }
        Ok(Self {
            url,
            client,
            credential,
            accepted_media_types,
            max_encoded_bytes,
        })
    }

    /// Credential-free normalized endpoint identity.
    #[must_use]
    pub fn display_url(&self) -> &str {
        self.url.as_str()
    }
}

/// Prepared backend capability for isolated control-plane handles.
pub trait ControlPlaneHttpProvider: Debug {
    /// Bind one validated source profile to native transport and provider secrets.
    fn prepare(
        &self,
        profile: ValidatedControlPlaneProfile,
    ) -> Result<Rc<dyn ControlPlaneHttp>, ControlPlanePrepareError>;
}

/// Minimal owned request allowed by a profile-bound control handle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlPlaneRequest {
    /// Stable request identity for transport tracing only.
    pub request_id: String,
}

/// Allowlisted control response and exact entity bytes.
#[derive(Clone, Debug)]
pub struct ControlPlaneResponse {
    /// HTTP status, including non-success status.
    pub status: u16,
    /// Exact allowlisted response metadata with lowercase names.
    pub headers: BTreeMap<String, String>,
    /// Exact content-encoded response entity.
    pub encoded_body: Bytes,
    /// Native transport timing facts on the injected Clock timeline.
    pub timings: ControlPlaneTransportTimings,
}

/// Native transport facts safe for source attempt construction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ControlPlaneTransportTimings {
    /// Request lifecycle start.
    pub request_start_ns: i64,
    /// First response body byte when observed.
    pub first_byte_ns: Option<i64>,
    /// Complete response or failure observation.
    pub end_ns: i64,
    /// Fine-grained native trace without credential bytes.
    pub trace: Option<TraceData>,
}

/// One profile-bound local HTTP handle.
#[async_trait(?Send)]
pub trait ControlPlaneHttp: Debug {
    /// Execute one GET under an absolute, dynamically lowerable Clock deadline.
    async fn execute(
        &self,
        request: ControlPlaneRequest,
        absolute_deadline_ns: i64,
        cancellation: LocalCancellationSignal,
    ) -> Result<ControlPlaneResponse, ControlPlaneHttpError>;
}

/// Native provider composing dedicated per-source transports.
pub struct NativeControlPlaneHttpProvider {
    clock: Rc<dyn Clock>,
    secrets: Arc<dyn SecretProviderResolver>,
}

impl NativeControlPlaneHttpProvider {
    /// Create a provider over the run's one RealClock and deployment resolver.
    #[must_use]
    pub fn new(clock: Rc<dyn Clock>, secrets: Arc<dyn SecretProviderResolver>) -> Self {
        Self { clock, secrets }
    }
}

impl Debug for NativeControlPlaneHttpProvider {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeControlPlaneHttpProvider")
            .field("virtual_clock", &self.clock.is_virtual())
            .field("secrets", &self.secrets)
            .finish()
    }
}

impl ControlPlaneHttpProvider for NativeControlPlaneHttpProvider {
    fn prepare(
        &self,
        profile: ValidatedControlPlaneProfile,
    ) -> Result<Rc<dyn ControlPlaneHttp>, ControlPlanePrepareError> {
        let authorization = match &profile.credential {
            ControlPlaneCredentialReference::None => None,
            ControlPlaneCredentialReference::BearerProvider(provider_id) => Some(
                self.secrets
                    .resolve_bearer(provider_id)
                    .map_err(ControlPlanePrepareError::Secret)?,
            ),
        };
        let transport = Rc::new(HttpTransport::new(
            self.clock.clone(),
            profile.client.clone(),
        ));
        Ok(Rc::new(NativeControlPlaneHttp {
            clock: self.clock.clone(),
            transport,
            url: profile.url.to_string(),
            display_url: profile.url.to_string(),
            authorization,
            accept: profile.accepted_media_types.join(", "),
            max_encoded_bytes: profile.max_encoded_bytes,
        }))
    }
}

struct NativeControlPlaneHttp {
    clock: Rc<dyn Clock>,
    transport: Rc<HttpTransport>,
    url: String,
    display_url: String,
    authorization: Option<ResolvedSecret>,
    accept: String,
    max_encoded_bytes: usize,
}

impl Debug for NativeControlPlaneHttp {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeControlPlaneHttp")
            .field("url", &self.display_url)
            .field("has_authorization", &self.authorization.is_some())
            .field("max_encoded_bytes", &self.max_encoded_bytes)
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl ControlPlaneHttp for NativeControlPlaneHttp {
    async fn execute(
        &self,
        request: ControlPlaneRequest,
        absolute_deadline_ns: i64,
        cancellation: LocalCancellationSignal,
    ) -> Result<ControlPlaneResponse, ControlPlaneHttpError> {
        validate_request_id(&request.request_id)?;
        let mut config = RequestConfig::new(self.url.clone())
            .header("Accept", self.accept.clone())
            .header("Accept-Encoding", "identity")
            .request_id(request.request_id)
            .reuse(ConnectionReuseStrategy::Pooled);
        if let Some(secret) = &self.authorization {
            config = config.header("Authorization", format!("Bearer {}", secret.expose()));
        }

        let get = self.transport.get(&config);
        tokio::pin!(get);
        let mut revision = cancellation.revision();
        let record = loop {
            let effective_deadline_ns = absolute_deadline_ns.min(cancellation.deadline_ns());
            let remaining_ns = effective_deadline_ns.saturating_sub(self.clock.now_ns());
            if remaining_ns <= 0 {
                return Err(ControlPlaneHttpError::deadline(cancellation.is_stopped()));
            }
            let sleep = self.clock.clone().sleep(remaining_ns);
            tokio::pin!(sleep);
            tokio::select! {
                biased;
                record = &mut get => break record,
                next_revision = cancellation.changed(revision) => revision = next_revision,
                () = &mut sleep => {
                    return Err(ControlPlaneHttpError::deadline(cancellation.is_stopped()));
                }
            }
        };

        let timings = ControlPlaneTransportTimings {
            request_start_ns: record.start_ns,
            first_byte_ns: record.recv_start_ns,
            end_ns: record.end_ns.unwrap_or_else(|| self.clock.now_ns()),
            trace: record.trace.clone(),
        };
        if let Some(error) = record.error.as_ref()
            && error.kind != ErrorKind::Http
        {
            return Err(ControlPlaneHttpError {
                kind: match error.kind {
                    ErrorKind::Timeout => ControlPlaneHttpErrorKind::Timeout,
                    ErrorKind::Cancelled => ControlPlaneHttpErrorKind::Cancelled,
                    ErrorKind::Connect | ErrorKind::Sse | ErrorKind::Other => {
                        ControlPlaneHttpErrorKind::Transport
                    }
                    ErrorKind::Http => unreachable!(),
                },
                message: bounded_error(&error.message),
                timings: Some(Box::new(timings)),
            });
        }
        let status = record.status.ok_or_else(|| ControlPlaneHttpError {
            kind: ControlPlaneHttpErrorKind::Transport,
            message: "control-plane response omitted HTTP status".to_owned(),
            timings: Some(Box::new(timings.clone())),
        })?;
        let encoded_body =
            exact_text_body(&record.responses).ok_or_else(|| ControlPlaneHttpError {
                kind: ControlPlaneHttpErrorKind::InvalidResponse,
                message: "control-plane response did not contain one complete entity".to_owned(),
                timings: Some(Box::new(timings.clone())),
            })?;
        if encoded_body.len() > self.max_encoded_bytes {
            return Err(ControlPlaneHttpError {
                kind: ControlPlaneHttpErrorKind::BodyTooLarge,
                message: format!(
                    "control-plane entity exceeded the {} byte encoded-body limit",
                    self.max_encoded_bytes
                ),
                timings: Some(Box::new(timings)),
            });
        }
        let headers = record
            .response_headers
            .into_iter()
            .filter(|(name, _)| ALLOWLISTED_RESPONSE_HEADERS.contains(&name.as_str()))
            .collect();
        Ok(ControlPlaneResponse {
            status,
            headers,
            encoded_body,
            timings,
        })
    }
}

fn exact_text_body(responses: &[Response]) -> Option<Bytes> {
    match responses {
        [Response::Text(response)] => Some(response.body.clone()),
        _ => None,
    }
}

fn validate_request_id(value: &str) -> Result<(), ControlPlaneHttpError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ControlPlaneHttpError {
            kind: ControlPlaneHttpErrorKind::InvalidRequest,
            message: "control-plane request ID is invalid".to_owned(),
            timings: None,
        });
    }
    Ok(())
}

fn bounded_error(message: &str) -> String {
    const MAX_BYTES: usize = 1024;
    if message.len() <= MAX_BYTES {
        return message.to_owned();
    }
    let mut end = MAX_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message[..end].to_owned()
}

/// Prepared-profile or provider-secret failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControlPlanePrepareError {
    /// Secret-free profile is unsafe or inconsistent.
    InvalidProfile(String),
    /// Provider-owned credential resolution failed.
    Secret(SecretResolutionError),
}

impl Display for ControlPlanePrepareError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfile(message) => formatter.write_str(message),
            Self::Secret(error) => {
                write!(formatter, "control-plane secret resolution failed: {error}")
            }
        }
    }
}

impl std::error::Error for ControlPlanePrepareError {}

/// Provider-owned secret resolution failure without secret bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SecretResolutionError {
    /// Selected provider does not exist in this deployment.
    Unavailable(String),
    /// Provider returned an empty or header-unsafe value.
    InvalidSecret,
    /// Provider failed with bounded redaction-safe detail.
    Failed(String),
}

impl Display for SecretResolutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(provider) => {
                write!(formatter, "secret provider {provider:?} is unavailable")
            }
            Self::InvalidSecret => formatter.write_str("secret provider returned an invalid value"),
            Self::Failed(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SecretResolutionError {}

/// Stable control-plane execution category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlPlaneHttpErrorKind {
    /// Request identity/configuration was invalid.
    InvalidRequest,
    /// DNS/TCP/TLS/HTTP transport failed.
    Transport,
    /// Absolute call deadline expired.
    Timeout,
    /// Lifecycle cancellation closed the active call.
    Cancelled,
    /// Response shape was not one complete non-streaming entity.
    InvalidResponse,
    /// Encoded body exceeded its prevalidated bound.
    BodyTooLarge,
}

/// One bounded, credential-free control-plane failure.
#[derive(Clone, Debug)]
pub struct ControlPlaneHttpError {
    /// Stable failure category.
    pub kind: ControlPlaneHttpErrorKind,
    /// Bounded redaction-safe detail.
    pub message: String,
    /// Native timing facts when request IO began.
    pub timings: Option<Box<ControlPlaneTransportTimings>>,
}

impl ControlPlaneHttpError {
    fn deadline(stopped: bool) -> Self {
        Self {
            kind: if stopped {
                ControlPlaneHttpErrorKind::Cancelled
            } else {
                ControlPlaneHttpErrorKind::Timeout
            },
            message: if stopped {
                "control-plane request cancelled by source shutdown"
            } else {
                "control-plane request exceeded its absolute deadline"
            }
            .to_owned(),
            timings: None,
        }
    }
}

impl Display for ControlPlaneHttpError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ControlPlaneHttpError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(url: &str) -> ValidatedControlPlaneProfile {
        let mut client = ClientConfig::default();
        client.max_connections_per_origin = 1;
        ValidatedControlPlaneProfile::new(
            Url::parse(url).unwrap(),
            client,
            ControlPlaneCredentialReference::None,
            vec!["text/plain; version=0.0.4".to_owned()],
            1024,
        )
        .unwrap()
    }

    #[test]
    fn profile_rejects_userinfo_and_nonisolated_connection_capacity() {
        let mut client = ClientConfig::default();
        client.max_connections_per_origin = 1;
        assert!(
            ValidatedControlPlaneProfile::new(
                Url::parse("https://user:secret@example.test/metrics").unwrap(),
                client.clone(),
                ControlPlaneCredentialReference::None,
                vec!["text/plain".to_owned()],
                1,
            )
            .is_err()
        );
        client.max_connections_per_origin = 2;
        assert!(
            ValidatedControlPlaneProfile::new(
                Url::parse("https://example.test/metrics").unwrap(),
                client,
                ControlPlaneCredentialReference::None,
                vec!["text/plain".to_owned()],
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn debug_surfaces_never_expose_secret_bytes() {
        let secret = ResolvedSecret::new("fixture-super-secret").unwrap();
        assert!(!format!("{secret:?}").contains("fixture-super-secret"));
        let _ = profile("http://127.0.0.1:1/metrics");
    }
}
