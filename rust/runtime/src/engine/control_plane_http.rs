// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile-bound control-plane HTTP over the native Clock-injected transport.
//!
//! Handles created here own a dedicated connection pool and expose only
//! allowlisted response metadata. Endpoint credentials are resolved during
//! preparation and never enter authored DTOs, durable source descriptors, or
//! returned attempt facts.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::env;
use std::fmt::{self, Debug, Display, Formatter};
use std::rc::Rc;
use std::sync::Arc;

use crate::clock::Clock;
use crate::transport::core::{ConnectionReuseStrategy, ErrorKind, Response, TraceData};
use crate::transport::http::config::{
    ClientConfig, PreparedTlsClientConfig, PreparedTlsClientConfigError,
};
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::Notify;
use url::Url;

#[derive(Debug)]
struct CancellationState {
    revision: u64,
    deadline_ns: i64,
    stopped: bool,
}

/// LocalSet-owned cancellation and deadline-lowering signal.
///
/// Clones stay on the caller's local thread. An active transport races its own
/// future against [`Self::changed`] and the injected Clock, so shutdown never
/// waits for an originally longer request timeout. This is a general
/// execution-factory seam: any Clock-injected control-plane consumer can hold a
/// signal, lower its deadline, and stop future issuance.
#[derive(Clone, Debug)]
pub struct LocalCancellationSignal {
    state: Rc<RefCell<CancellationState>>,
    notify: Rc<Notify>,
}

impl Default for LocalCancellationSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalCancellationSignal {
    /// Build an open signal with no lowered deadline and no stop request.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(CancellationState {
                revision: 0,
                deadline_ns: i64::MAX,
                stopped: false,
            })),
            notify: Rc::new(Notify::new()),
        }
    }

    /// Close future issuance and lower the effective deadline monotonically.
    pub fn stop(&self, shutdown_deadline_ns: i64) {
        let mut state = self.state.borrow_mut();
        let next_deadline = state.deadline_ns.min(shutdown_deadline_ns);
        if !state.stopped || next_deadline != state.deadline_ns {
            state.stopped = true;
            state.deadline_ns = next_deadline;
            state.revision = state.revision.wrapping_add(1);
            drop(state);
            self.notify.notify_waiters();
        }
    }

    /// Returns the current effective lifecycle cap.
    #[must_use]
    pub fn deadline_ns(&self) -> i64 {
        self.state.borrow().deadline_ns
    }

    /// Whether a stop request has closed future issuance.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.state.borrow().stopped
    }

    /// Monotone local change token used to await a later deadline update.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.state.borrow().revision
    }

    /// Waits until the signal changes beyond `observed_revision`.
    pub async fn changed(&self, observed_revision: u64) -> u64 {
        loop {
            let notified = self.notify.notified();
            let revision = self.state.borrow().revision;
            if revision != observed_revision {
                return revision;
            }
            notified.await;
        }
    }
}

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

/// Provider references for a source's verifying TLS policy.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ControlPlaneTlsReference {
    /// Optional provider whose roots replace the built-in public WebPKI set.
    pub trust_provider: Option<String>,
    /// Optional provider for a client certificate chain and private key.
    pub mtls_provider: Option<String>,
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

/// Provider-resolved trust-root PEM with an always-redacted debug surface.
#[derive(Eq, PartialEq)]
pub struct ResolvedTlsTrustRoots(Vec<u8>);

impl ResolvedTlsTrustRoots {
    /// Construct one non-empty provider result.
    pub fn new(value: impl Into<Vec<u8>>) -> Result<Self, TlsMaterialResolutionError> {
        let value = value.into();
        if value.is_empty() || value.contains(&0) {
            return Err(TlsMaterialResolutionError::InvalidTrustMaterial);
        }
        Ok(Self(value))
    }

    fn expose(&self) -> &[u8] {
        &self.0
    }
}

impl Debug for ResolvedTlsTrustRoots {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("ResolvedTlsTrustRoots([REDACTED])")
    }
}

/// Provider-resolved mTLS identity with an always-redacted debug surface.
#[derive(Eq, PartialEq)]
pub struct ResolvedMtlsIdentity {
    certificate_chain_pem: Vec<u8>,
    private_key_pem: Vec<u8>,
}

impl ResolvedMtlsIdentity {
    /// Construct one non-empty provider result.
    pub fn new(
        certificate_chain_pem: impl Into<Vec<u8>>,
        private_key_pem: impl Into<Vec<u8>>,
    ) -> Result<Self, TlsMaterialResolutionError> {
        let certificate_chain_pem = certificate_chain_pem.into();
        let private_key_pem = private_key_pem.into();
        if certificate_chain_pem.is_empty()
            || private_key_pem.is_empty()
            || certificate_chain_pem.contains(&0)
            || private_key_pem.contains(&0)
        {
            return Err(TlsMaterialResolutionError::InvalidMtlsMaterial);
        }
        Ok(Self {
            certificate_chain_pem,
            private_key_pem,
        })
    }

    fn certificate_chain_pem(&self) -> &[u8] {
        &self.certificate_chain_pem
    }

    fn private_key_pem(&self) -> &[u8] {
        &self.private_key_pem
    }
}

impl Debug for ResolvedMtlsIdentity {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("ResolvedMtlsIdentity([REDACTED])")
    }
}

impl Drop for ResolvedMtlsIdentity {
    fn drop(&mut self) {
        self.certificate_chain_pem.fill(0);
        self.private_key_pem.fill(0);
    }
}

/// Deployment-owned TLS and mTLS material resolution seam.
pub trait TlsMaterialProviderResolver: Debug + Send + Sync {
    /// Resolve one named private trust-root bundle.
    fn resolve_trust(
        &self,
        provider_id: &str,
    ) -> Result<ResolvedTlsTrustRoots, TlsMaterialResolutionError>;

    /// Resolve one named client certificate/private-key identity.
    fn resolve_mtls(
        &self,
        provider_id: &str,
    ) -> Result<ResolvedMtlsIdentity, TlsMaterialResolutionError>;
}

/// Default resolver that permits only the built-in public WebPKI policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct RejectingTlsMaterialProviderResolver;

impl TlsMaterialProviderResolver for RejectingTlsMaterialProviderResolver {
    fn resolve_trust(
        &self,
        provider_id: &str,
    ) -> Result<ResolvedTlsTrustRoots, TlsMaterialResolutionError> {
        Err(TlsMaterialResolutionError::Unavailable(
            provider_id.to_owned(),
        ))
    }

    fn resolve_mtls(
        &self,
        provider_id: &str,
    ) -> Result<ResolvedMtlsIdentity, TlsMaterialResolutionError> {
        Err(TlsMaterialResolutionError::Unavailable(
            provider_id.to_owned(),
        ))
    }
}

/// Default resolver that permits only credential-free sources.
#[derive(Clone, Copy, Debug, Default)]
pub struct RejectingSecretProviderResolver;

impl SecretProviderResolver for RejectingSecretProviderResolver {
    fn resolve_bearer(&self, provider_id: &str) -> Result<ResolvedSecret, SecretResolutionError> {
        Err(SecretResolutionError::Unavailable(provider_id.to_owned()))
    }
}

/// Environment-backed bearer resolver used by the stock runner distribution.
///
/// A public reference such as `node-metrics` maps to
/// `AIPERF_CONTROL_BEARER_NODE_METRICS`. Diagnostics expose only that public
/// variable name and never the token value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvironmentSecretProviderResolver {
    prefix: String,
}

impl Default for EnvironmentSecretProviderResolver {
    fn default() -> Self {
        Self {
            prefix: "AIPERF_CONTROL_BEARER_".to_owned(),
        }
    }
}

impl EnvironmentSecretProviderResolver {
    /// Use an explicit uppercase public environment-variable prefix.
    pub fn new(prefix: impl Into<String>) -> Result<Self, SecretResolutionError> {
        let prefix = prefix.into();
        if prefix.is_empty()
            || prefix.chars().any(|character| {
                !(character.is_ascii_uppercase() || character.is_ascii_digit() || character == '_')
            })
        {
            return Err(SecretResolutionError::InvalidProviderConfiguration);
        }
        Ok(Self { prefix })
    }

    /// Return the public variable name derived from one provider reference.
    pub fn variable_name(&self, provider_id: &str) -> Result<String, SecretResolutionError> {
        provider_variable_name(&self.prefix, provider_id)
            .map_err(|_| SecretResolutionError::Unavailable(provider_id.to_owned()))
    }
}

impl SecretProviderResolver for EnvironmentSecretProviderResolver {
    fn resolve_bearer(&self, provider_id: &str) -> Result<ResolvedSecret, SecretResolutionError> {
        let variable = self.variable_name(provider_id)?;
        let value = env::var(&variable)
            .map_err(|_| SecretResolutionError::Unavailable(provider_id.to_owned()))?;
        ResolvedSecret::new(value)
    }
}

/// Environment-backed provider-held TLS material for the stock distribution.
///
/// Public reference `cluster-ca` maps to
/// `AIPERF_CONTROL_TLS_TRUST_CLUSTER_CA`; mTLS reference `node-client` maps to
/// `AIPERF_CONTROL_TLS_MTLS_CERT_NODE_CLIENT` and
/// `AIPERF_CONTROL_TLS_MTLS_KEY_NODE_CLIENT`. Values contain PEM entities and
/// never appear in diagnostics or debug output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvironmentTlsMaterialProviderResolver {
    trust_prefix: String,
    mtls_certificate_prefix: String,
    mtls_key_prefix: String,
}

impl Default for EnvironmentTlsMaterialProviderResolver {
    fn default() -> Self {
        Self {
            trust_prefix: "AIPERF_CONTROL_TLS_TRUST_".to_owned(),
            mtls_certificate_prefix: "AIPERF_CONTROL_TLS_MTLS_CERT_".to_owned(),
            mtls_key_prefix: "AIPERF_CONTROL_TLS_MTLS_KEY_".to_owned(),
        }
    }
}

impl EnvironmentTlsMaterialProviderResolver {
    fn variable_name(
        &self,
        prefix: &str,
        provider_id: &str,
    ) -> Result<String, TlsMaterialResolutionError> {
        provider_variable_name(prefix, provider_id)
            .map_err(|_| TlsMaterialResolutionError::Unavailable(provider_id.to_owned()))
    }

    fn read_variable(
        &self,
        variable: &str,
        provider_id: &str,
    ) -> Result<Vec<u8>, TlsMaterialResolutionError> {
        env::var(variable)
            .map(String::into_bytes)
            .map_err(|_| TlsMaterialResolutionError::Unavailable(provider_id.to_owned()))
    }
}

impl TlsMaterialProviderResolver for EnvironmentTlsMaterialProviderResolver {
    fn resolve_trust(
        &self,
        provider_id: &str,
    ) -> Result<ResolvedTlsTrustRoots, TlsMaterialResolutionError> {
        let variable = self.variable_name(&self.trust_prefix, provider_id)?;
        ResolvedTlsTrustRoots::new(self.read_variable(&variable, provider_id)?)
    }

    fn resolve_mtls(
        &self,
        provider_id: &str,
    ) -> Result<ResolvedMtlsIdentity, TlsMaterialResolutionError> {
        let certificate_variable =
            self.variable_name(&self.mtls_certificate_prefix, provider_id)?;
        let key_variable = self.variable_name(&self.mtls_key_prefix, provider_id)?;
        ResolvedMtlsIdentity::new(
            self.read_variable(&certificate_variable, provider_id)?,
            self.read_variable(&key_variable, provider_id)?,
        )
    }
}

/// Strict transport profile after source-factory validation.
#[derive(Clone, Debug)]
pub struct ValidatedControlPlaneProfile {
    url: Url,
    client: ClientConfig,
    credential: ControlPlaneCredentialReference,
    tls: ControlPlaneTlsReference,
    accepted_media_types: Vec<String>,
    accepted_content_encodings: Vec<String>,
    max_encoded_bytes: usize,
}

impl ValidatedControlPlaneProfile {
    /// Validate and freeze a secret-free control-plane profile.
    pub fn new(
        url: Url,
        client: ClientConfig,
        credential: ControlPlaneCredentialReference,
        tls: ControlPlaneTlsReference,
        accepted_media_types: Vec<String>,
        accepted_content_encodings: Vec<String>,
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
        if !client.ssl_verify || client.prepared_tls.is_some() {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "control-plane TLS must use provider-prepared certificate verification".to_owned(),
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
        validate_sorted_tokens(
            "control-plane accepted content encodings",
            &accepted_content_encodings,
        )?;
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
            && validate_provider_id(provider_id).is_err()
        {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "credential provider ID is invalid".to_owned(),
            ));
        }
        for provider_id in [tls.trust_provider.as_deref(), tls.mtls_provider.as_deref()]
            .into_iter()
            .flatten()
        {
            validate_provider_id(provider_id).map_err(|_| {
                ControlPlanePrepareError::InvalidProfile(
                    "TLS material provider ID is invalid".to_owned(),
                )
            })?;
        }
        if url.scheme() != "https"
            && (!matches!(credential, ControlPlaneCredentialReference::None)
                || tls.trust_provider.is_some()
                || tls.mtls_provider.is_some())
        {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "credentials and provider TLS material require an https control-plane URL"
                    .to_owned(),
            ));
        }
        Ok(Self {
            url,
            client,
            credential,
            tls,
            accepted_media_types,
            accepted_content_encodings,
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

/// Backend-owned preparation seam for run-local control-plane providers.
///
/// The factory is process-shareable while each returned provider remains
/// LocalSet-owned beside its injected Clock. Remote runner distributions can
/// replace this member without changing telemetry source or workload code.
pub trait ControlPlaneHttpProviderFactory: Debug + Send + Sync {
    /// Prepare one provider for a run's single Clock authority.
    fn prepare(
        &self,
        clock: Rc<dyn Clock>,
        policy: ControlPlaneClientPolicy,
    ) -> Rc<dyn ControlPlaneHttpProvider>;
}

/// Backend-owned ceilings inherited by every prepared source handle.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControlPlaneClientPolicy {
    /// Maximum DNS/TCP/TLS/HTTP-handshake lifetime for any source.
    pub connect_timeout_ns: Option<i64>,
}

impl ControlPlaneClientPolicy {
    /// Rejects a present non-positive backend ceiling.
    pub fn validate(self) -> Result<(), ControlPlanePrepareError> {
        if self.connect_timeout_ns.is_some_and(|timeout| timeout <= 0) {
            return Err(ControlPlanePrepareError::InvalidProfile(
                "backend control-plane connect timeout must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Stock native control-plane provider factory.
#[derive(Clone)]
pub struct NativeControlPlaneHttpProviderFactory {
    secrets: Arc<dyn SecretProviderResolver>,
    tls_materials: Arc<dyn TlsMaterialProviderResolver>,
}

impl NativeControlPlaneHttpProviderFactory {
    /// Bind a deployment-owned secret resolver without resolving any source.
    #[must_use]
    pub fn new(secrets: Arc<dyn SecretProviderResolver>) -> Self {
        Self::with_resolvers(secrets, Arc::new(RejectingTlsMaterialProviderResolver))
    }

    /// Bind independent deployment-owned credential and TLS material resolvers.
    #[must_use]
    pub fn with_resolvers(
        secrets: Arc<dyn SecretProviderResolver>,
        tls_materials: Arc<dyn TlsMaterialProviderResolver>,
    ) -> Self {
        Self {
            secrets,
            tls_materials,
        }
    }
}

impl Default for NativeControlPlaneHttpProviderFactory {
    fn default() -> Self {
        Self::with_resolvers(
            Arc::new(EnvironmentSecretProviderResolver::default()),
            Arc::new(EnvironmentTlsMaterialProviderResolver::default()),
        )
    }
}

impl Debug for NativeControlPlaneHttpProviderFactory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeControlPlaneHttpProviderFactory")
            .field("secrets", &self.secrets)
            .field("tls_materials", &self.tls_materials)
            .finish()
    }
}

impl ControlPlaneHttpProviderFactory for NativeControlPlaneHttpProviderFactory {
    fn prepare(
        &self,
        clock: Rc<dyn Clock>,
        policy: ControlPlaneClientPolicy,
    ) -> Rc<dyn ControlPlaneHttpProvider> {
        Rc::new(
            NativeControlPlaneHttpProvider::with_resolvers_and_client_policy(
                clock,
                Arc::clone(&self.secrets),
                Arc::clone(&self.tls_materials),
                policy,
            ),
        )
    }
}

/// Minimal owned request allowed by a profile-bound control handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlPlaneMethod {
    /// Issue one GET request.
    Get,
    /// Issue one POST request with an empty body.
    Post,
}

/// Minimal owned request allowed by a profile-bound control handle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlPlaneRequest {
    /// Stable request identity for transport tracing only.
    pub request_id: String,
    /// One allowed HTTP verb for this control-plane request.
    pub method: ControlPlaneMethod,
    /// Path resolved against the prepared control-plane origin.
    pub path: String,
}

/// Allowlisted control response and exact entity bytes.
#[derive(Clone)]
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

impl Debug for ControlPlaneResponse {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlPlaneResponse")
            .field("status", &self.status)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("encoded_body_bytes", &self.encoded_body.len())
            .field("encoded_body", &"<redacted>")
            .field("timings", &self.timings)
            .finish()
    }
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
    /// Execute one request under an absolute, dynamically lowerable Clock deadline.
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
    tls_materials: Arc<dyn TlsMaterialProviderResolver>,
    client_policy: ControlPlaneClientPolicy,
}

impl NativeControlPlaneHttpProvider {
    /// Create a provider over the run's one RealClock and deployment resolver.
    #[must_use]
    pub fn new(clock: Rc<dyn Clock>, secrets: Arc<dyn SecretProviderResolver>) -> Self {
        Self::with_client_policy(clock, secrets, ControlPlaneClientPolicy::default())
    }

    /// Create a provider with backend-owned client ceilings.
    #[must_use]
    pub fn with_client_policy(
        clock: Rc<dyn Clock>,
        secrets: Arc<dyn SecretProviderResolver>,
        client_policy: ControlPlaneClientPolicy,
    ) -> Self {
        Self::with_resolvers_and_client_policy(
            clock,
            secrets,
            Arc::new(RejectingTlsMaterialProviderResolver),
            client_policy,
        )
    }

    /// Create a provider with independent material resolvers and client ceilings.
    #[must_use]
    pub fn with_resolvers_and_client_policy(
        clock: Rc<dyn Clock>,
        secrets: Arc<dyn SecretProviderResolver>,
        tls_materials: Arc<dyn TlsMaterialProviderResolver>,
        client_policy: ControlPlaneClientPolicy,
    ) -> Self {
        Self {
            clock,
            secrets,
            tls_materials,
            client_policy,
        }
    }
}

impl Debug for NativeControlPlaneHttpProvider {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeControlPlaneHttpProvider")
            .field("virtual_clock", &self.clock.is_virtual())
            .field("secrets", &self.secrets)
            .field("tls_materials", &self.tls_materials)
            .field("client_policy", &self.client_policy)
            .finish()
    }
}

impl ControlPlaneHttpProvider for NativeControlPlaneHttpProvider {
    fn prepare(
        &self,
        profile: ValidatedControlPlaneProfile,
    ) -> Result<Rc<dyn ControlPlaneHttp>, ControlPlanePrepareError> {
        self.client_policy.validate()?;
        let authorization = match &profile.credential {
            ControlPlaneCredentialReference::None => None,
            ControlPlaneCredentialReference::BearerProvider(provider_id) => Some(
                self.secrets
                    .resolve_bearer(provider_id)
                    .map_err(ControlPlanePrepareError::Secret)?,
            ),
        };
        let trust = profile
            .tls
            .trust_provider
            .as_deref()
            .map(|provider_id| self.tls_materials.resolve_trust(provider_id))
            .transpose()
            .map_err(ControlPlanePrepareError::TlsMaterial)?;
        let mtls = profile
            .tls
            .mtls_provider
            .as_deref()
            .map(|provider_id| self.tls_materials.resolve_mtls(provider_id))
            .transpose()
            .map_err(ControlPlanePrepareError::TlsMaterial)?;
        let mut client = profile.client.clone();
        client.connect_timeout_ns = capped_connect_timeout(
            client.connect_timeout_ns,
            self.client_policy.connect_timeout_ns,
        );
        if trust.is_some() || mtls.is_some() {
            client.prepared_tls = Some(
                PreparedTlsClientConfig::from_provider_pem(
                    trust.as_ref().map(ResolvedTlsTrustRoots::expose),
                    mtls.as_ref()
                        .map(ResolvedMtlsIdentity::certificate_chain_pem),
                    mtls.as_ref().map(ResolvedMtlsIdentity::private_key_pem),
                )
                .map_err(ControlPlanePrepareError::TlsConfig)?,
            );
        }
        let transport = Rc::new(HttpTransport::new(self.clock.clone(), client));
        Ok(Rc::new(NativeControlPlaneHttp {
            clock: self.clock.clone(),
            transport,
            url: profile.url.to_string(),
            display_url: profile.url.to_string(),
            authorization,
            accept: profile.accepted_media_types.join(", "),
            accept_encoding: profile.accepted_content_encodings.join(", "),
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
    accept_encoding: String,
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
        let url = join_relative_path(&self.url, &request.path)?;
        let mut config = RequestConfig::new(url)
            .header("Accept", self.accept.clone())
            .header("Accept-Encoding", self.accept_encoding.clone())
            .request_id(request.request_id)
            .reuse(ConnectionReuseStrategy::Pooled);
        if let Some(secret) = &self.authorization {
            config = config.header("Authorization", format!("Bearer {}", secret.expose()));
        }

        let request_fut = async {
            match request.method {
                ControlPlaneMethod::Get => self.transport.get(&config).await,
                ControlPlaneMethod::Post => {
                    self.transport
                        .send_request_bytes(&config, Bytes::new(), false, |_| {})
                        .await
                }
            }
        };
        tokio::pin!(request_fut);
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
                record = &mut request_fut => break record,
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

fn capped_connect_timeout(source: Option<i64>, backend: Option<i64>) -> Option<i64> {
    match (source, backend) {
        (Some(source), Some(backend)) => Some(source.min(backend)),
        (Some(source), None) => Some(source),
        (None, Some(backend)) => Some(backend),
        (None, None) => None,
    }
}

fn validate_provider_id(provider_id: &str) -> Result<(), ()> {
    if provider_id.is_empty()
        || provider_id.len() > 128
        || !provider_id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || provider_id.starts_with('-')
        || provider_id.ends_with('-')
    {
        return Err(());
    }
    Ok(())
}

fn provider_variable_name(prefix: &str, provider_id: &str) -> Result<String, ()> {
    validate_provider_id(provider_id)?;
    let suffix = provider_id
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() {
                char::from(byte.to_ascii_uppercase())
            } else {
                '_'
            }
        })
        .collect::<String>();
    Ok(format!("{prefix}{suffix}"))
}

fn validate_sorted_tokens(
    field: &'static str,
    values: &[String],
) -> Result<(), ControlPlanePrepareError> {
    if values.is_empty() {
        return Err(ControlPlanePrepareError::InvalidProfile(format!(
            "{field} cannot be empty"
        )));
    }
    let mut preceding: Option<&str> = None;
    for value in values {
        if value.is_empty()
            || !value.is_ascii()
            || value.bytes().any(|byte| {
                !(byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
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
                    ))
            })
        {
            return Err(ControlPlanePrepareError::InvalidProfile(format!(
                "{field} contains an invalid lowercase HTTP token"
            )));
        }
        if preceding.is_some_and(|preceding| preceding >= value.as_str()) {
            return Err(ControlPlanePrepareError::InvalidProfile(format!(
                "{field} must be sorted and unique"
            )));
        }
        preceding = Some(value);
    }
    Ok(())
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

fn join_relative_path(base_url: &str, path: &str) -> Result<String, ControlPlaneHttpError> {
    if path.is_empty()
        || !path.starts_with('/')
        || path.chars().any(char::is_whitespace)
        || path.contains(['?', '#'])
    {
        return Err(ControlPlaneHttpError {
            kind: ControlPlaneHttpErrorKind::InvalidRequest,
            message: "control-plane request path is invalid".to_owned(),
            timings: None,
        });
    }
    let mut url = Url::parse(base_url).map_err(|error| ControlPlaneHttpError {
        kind: ControlPlaneHttpErrorKind::InvalidRequest,
        message: format!("control-plane base URL is invalid: {error}"),
        timings: None,
    })?;
    url.set_path(path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.into())
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
    /// Provider-owned TLS material resolution failed.
    TlsMaterial(TlsMaterialResolutionError),
    /// Resolved TLS material could not form a verifying rustls policy.
    TlsConfig(PreparedTlsClientConfigError),
}

impl Display for ControlPlanePrepareError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfile(message) => formatter.write_str(message),
            Self::Secret(error) => {
                write!(formatter, "control-plane secret resolution failed: {error}")
            }
            Self::TlsMaterial(error) => {
                write!(
                    formatter,
                    "control-plane TLS material resolution failed: {error}"
                )
            }
            Self::TlsConfig(error) => {
                write!(formatter, "control-plane TLS policy failed: {error}")
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
    /// Resolver construction policy is invalid.
    InvalidProviderConfiguration,
}

impl Display for SecretResolutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(provider) => {
                write!(formatter, "secret provider {provider:?} is unavailable")
            }
            Self::InvalidSecret => formatter.write_str("secret provider returned an invalid value"),
            Self::InvalidProviderConfiguration => {
                formatter.write_str("secret provider resolver configuration is invalid")
            }
        }
    }
}

impl std::error::Error for SecretResolutionError {}

/// Provider-owned TLS resolution failure without material bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TlsMaterialResolutionError {
    /// Selected provider does not exist in this deployment.
    Unavailable(String),
    /// Trust provider returned an empty or unsafe entity.
    InvalidTrustMaterial,
    /// mTLS provider returned an empty or unsafe identity entity.
    InvalidMtlsMaterial,
}

impl Display for TlsMaterialResolutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(provider) => {
                write!(
                    formatter,
                    "TLS material provider {provider:?} is unavailable"
                )
            }
            Self::InvalidTrustMaterial => {
                formatter.write_str("TLS trust provider returned invalid material")
            }
            Self::InvalidMtlsMaterial => {
                formatter.write_str("mTLS provider returned invalid identity material")
            }
        }
    }
}

impl std::error::Error for TlsMaterialResolutionError {}

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
    use crate::clock::SimClock;

    #[derive(Debug)]
    struct InvalidPemTlsResolver;

    impl TlsMaterialProviderResolver for InvalidPemTlsResolver {
        fn resolve_trust(
            &self,
            _provider_id: &str,
        ) -> Result<ResolvedTlsTrustRoots, TlsMaterialResolutionError> {
            ResolvedTlsTrustRoots::new(b"fixture-super-secret-invalid-trust".to_vec())
        }

        fn resolve_mtls(
            &self,
            _provider_id: &str,
        ) -> Result<ResolvedMtlsIdentity, TlsMaterialResolutionError> {
            ResolvedMtlsIdentity::new(
                b"fixture-super-secret-invalid-certificate".to_vec(),
                b"fixture-super-secret-invalid-private-key".to_vec(),
            )
        }
    }

    fn profile(url: &str) -> ValidatedControlPlaneProfile {
        let client = ClientConfig {
            max_connections_per_origin: 1,
            ..ClientConfig::default()
        };
        ValidatedControlPlaneProfile::new(
            Url::parse(url).unwrap(),
            client,
            ControlPlaneCredentialReference::None,
            ControlPlaneTlsReference::default(),
            vec!["text/plain; version=0.0.4".to_owned()],
            vec!["gzip".to_owned(), "identity".to_owned()],
            1024,
        )
        .unwrap()
    }

    #[test]
    fn profile_rejects_userinfo_and_nonisolated_connection_capacity() {
        let mut client = ClientConfig {
            max_connections_per_origin: 1,
            ..ClientConfig::default()
        };
        assert!(
            ValidatedControlPlaneProfile::new(
                Url::parse("https://user:secret@example.test/metrics").unwrap(),
                client.clone(),
                ControlPlaneCredentialReference::None,
                ControlPlaneTlsReference::default(),
                vec!["text/plain".to_owned()],
                vec!["identity".to_owned()],
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
                ControlPlaneTlsReference::default(),
                vec!["text/plain".to_owned()],
                vec!["identity".to_owned()],
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn debug_surfaces_never_expose_secret_bytes() {
        let secret = ResolvedSecret::new("fixture-super-secret").unwrap();
        assert!(!format!("{secret:?}").contains("fixture-super-secret"));
        let trust = ResolvedTlsTrustRoots::new(b"fixture-super-secret-trust".to_vec()).unwrap();
        let mtls = ResolvedMtlsIdentity::new(
            b"fixture-super-secret-certificate".to_vec(),
            b"fixture-super-secret-private-key".to_vec(),
        )
        .unwrap();
        let debug = format!("{trust:?} {mtls:?}");
        assert!(!debug.contains("fixture-super-secret"));
        let _ = profile("http://127.0.0.1:1/metrics");
    }

    #[test]
    fn environment_provider_references_have_stable_public_names() {
        let resolver = EnvironmentSecretProviderResolver::default();
        assert_eq!(
            resolver.variable_name("node-metrics").unwrap(),
            "AIPERF_CONTROL_BEARER_NODE_METRICS"
        );
        assert!(resolver.variable_name(" padded ").is_err());
        let tls = EnvironmentTlsMaterialProviderResolver::default();
        assert_eq!(
            tls.variable_name(&tls.trust_prefix, "cluster-ca").unwrap(),
            "AIPERF_CONTROL_TLS_TRUST_CLUSTER_CA"
        );
        assert_eq!(
            tls.variable_name(&tls.mtls_key_prefix, "node-client")
                .unwrap(),
            "AIPERF_CONTROL_TLS_MTLS_KEY_NODE_CLIENT"
        );
    }

    #[test]
    fn provider_tls_is_https_only_and_cannot_be_preinjected() {
        let client = ClientConfig {
            max_connections_per_origin: 1,
            ..ClientConfig::default()
        };
        assert!(
            ValidatedControlPlaneProfile::new(
                Url::parse("http://example.test/metrics").unwrap(),
                client.clone(),
                ControlPlaneCredentialReference::None,
                ControlPlaneTlsReference {
                    trust_provider: Some("cluster-ca".to_owned()),
                    mtls_provider: None,
                },
                vec!["text/plain".to_owned()],
                vec!["identity".to_owned()],
                1024,
            )
            .is_err()
        );
        assert!(
            ValidatedControlPlaneProfile::new(
                Url::parse("http://example.test/metrics").unwrap(),
                client,
                ControlPlaneCredentialReference::BearerProvider("node-token".to_owned()),
                ControlPlaneTlsReference::default(),
                vec!["text/plain".to_owned()],
                vec!["identity".to_owned()],
                1024,
            )
            .is_err()
        );
    }

    #[test]
    fn unknown_tls_provider_fails_preparation_without_material_leakage() {
        let mut tls_profile = profile("https://example.test/metrics");
        tls_profile.tls.trust_provider = Some("missing-private-ca".to_owned());
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider =
            NativeControlPlaneHttpProvider::new(clock, Arc::new(RejectingSecretProviderResolver));
        let error = provider.prepare(tls_profile).unwrap_err();
        assert!(error.to_string().contains("missing-private-ca"));

        let mut invalid = profile("https://example.test/metrics");
        invalid.tls.trust_provider = Some("invalid-private-ca".to_owned());
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider = NativeControlPlaneHttpProvider::with_resolvers_and_client_policy(
            clock,
            Arc::new(RejectingSecretProviderResolver),
            Arc::new(InvalidPemTlsResolver),
            ControlPlaneClientPolicy::default(),
        );
        let error = provider.prepare(invalid).unwrap_err();
        let surfaces = format!("{error:?} {error}");
        assert!(!surfaces.contains("fixture-super-secret"));
        assert!(surfaces.contains("invalid certificate material"));
    }

    #[test]
    fn unknown_bearer_provider_fails_before_transport_construction() {
        let mut credential_profile = profile("https://example.test/metrics");
        credential_profile.credential =
            ControlPlaneCredentialReference::BearerProvider("missing-node-token".to_owned());
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let provider =
            NativeControlPlaneHttpProvider::new(clock, Arc::new(RejectingSecretProviderResolver));
        let error = provider.prepare(credential_profile).unwrap_err();
        assert!(error.to_string().contains("missing-node-token"));
        assert!(!format!("{error:?}").contains("Authorization"));
    }

    #[test]
    fn content_negotiation_and_backend_connect_ceiling_are_exact() {
        let profile = profile("http://127.0.0.1:1/metrics");
        assert_eq!(
            profile.accepted_content_encodings,
            ["gzip".to_owned(), "identity".to_owned()]
        );
        assert_eq!(
            capped_connect_timeout(Some(5_000), Some(3_000)),
            Some(3_000)
        );
        assert_eq!(capped_connect_timeout(Some(2_000), None), Some(2_000));
        assert_eq!(capped_connect_timeout(None, Some(3_000)), Some(3_000));
        assert!(
            ControlPlaneClientPolicy {
                connect_timeout_ns: Some(0),
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn content_encoding_advertisement_rejects_unknown_shape_before_io() {
        let client = ClientConfig {
            max_connections_per_origin: 1,
            ..ClientConfig::default()
        };
        for encodings in [
            vec!["identity".to_owned(), "gzip".to_owned()],
            vec!["gzip".to_owned(), "gzip".to_owned()],
            vec!["GZIP".to_owned()],
            vec!["gzip, identity".to_owned()],
        ] {
            assert!(
                ValidatedControlPlaneProfile::new(
                    Url::parse("https://example.test/metrics").unwrap(),
                    client.clone(),
                    ControlPlaneCredentialReference::None,
                    ControlPlaneTlsReference::default(),
                    vec!["text/plain".to_owned()],
                    encodings,
                    1024,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn control_request_path_replaces_the_prepared_url_path() {
        assert_eq!(
            join_relative_path("https://example.test/metrics?old=1", "/start_profile").unwrap(),
            "https://example.test/start_profile"
        );
        assert!(join_relative_path("https://example.test/metrics", "start_profile").is_err());
        assert!(join_relative_path("https://example.test/metrics", "/bad path").is_err());
    }
}
