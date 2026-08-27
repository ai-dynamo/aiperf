// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared AWS S3 client construction for the streaming data plane.
//!
//! This module is the *only* place in AIPerf that builds an
//! [`aws_sdk_s3::Client`], and [`AwsCredentialProviderAuthority`] is the only
//! AWS credential provider. The streaming S3 source and any later object-store
//! checkpoint backend consume this authority; neither defines a second one.
//!
//! Three constraints shape the construction:
//!
//! * **Proxy.** `aws-smithy-runtime`'s default HTTPS client installs
//!   `ProxyConfig::from_env()` at behavior version `v2025_08_07` and later, and
//!   the underlying hyper-util matcher has no loopback exclusion. An ambient
//!   `HTTP_PROXY` would therefore be applied to a `http://127.0.0.1:9000`
//!   endpoint. AIPerf resolves the proxy itself through
//!   [`crate::transport::http::client::proxy`] — the same authority the HTTP
//!   transport uses, loopback always excluded — and installs an explicit
//!   connector. The SDK default HTTPS client is never used.
//! * **Time.** [`Clock`] is `Rc`-held and not `Send`, while `ProvideCredentials`
//!   requires `Send + Sync`. The authority therefore reads run-clock
//!   nanoseconds from a shared cell that a worker-local [`AwsClockProjection`]
//!   publishes; it never calls `Instant::now`, `SystemTime::now`, or a Tokio
//!   timer. SigV4 request signing keeps the SDK's own time source, because
//!   signing time is remote protocol truth rather than run measurement — a
//!   virtual clock would produce `RequestTimeTooSkewed`. `SimClock` execution
//!   never reaches this module.
//! * **Retry.** SDK retry is disabled so the only backoff in the S3 path is the
//!   source adapter's `Clock`-driven one. The SDK sleeper cannot be backed by
//!   `Clock` (`Clock::sleep` returns a non-`Send` future), so one clocked retry
//!   authority is preferred over two racing ones.
//!
//! Credentials are redacted, not erased: generation one ships no zeroization,
//! per the benchmark security-scope course correction. Redaction is structural —
//! no type here implements `Serialize`, no `Debug` is derived on a
//! secret-bearing type, and the streaming seam's `StreamSourceError` is a
//! stringless `Copy` code, so an SDK error's text never escapes this module.

use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, SystemTime};

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_credential_types::Credentials;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider, future};
use aws_sdk_s3::config::IdentityCache;
use aws_sdk_s3::config::retry::RetryConfig;
use aws_sdk_s3::config::timeout::TimeoutConfig;
use aws_smithy_http_client::proxy::ProxyConfig as SdkProxyConfig;
use aws_smithy_http_client::{Builder as SdkHttpClientBuilder, Connector, tls};
use parking_lot::RwLock;
use url::Url;

use crate::clock::Clock;
use crate::streaming::failure::{SourceFailureCode, StreamSourceError};
use crate::transport::http::client::proxy as host_proxy;

/// Refresh a credential this long before its reported expiry.
const CREDENTIAL_REFRESH_SKEW_NS: i64 = 60_000_000_000;
/// Upper bound on an authored per-operation or connect timeout.
const MAX_OPERATION_TIMEOUT_NS: i64 = 600_000_000_000;
/// Bypass rules always applied to the streaming connector, so a proxied run
/// still reaches a local S3 gateway directly.
const CONNECTOR_NO_PROXY_RULES: &str = "localhost,127.0.0.1,::1";

// ---------------------------------------------------------------------------
// Redaction primitives
// ---------------------------------------------------------------------------

/// An authored secret string that cannot be printed, formatted, or serialized.
///
/// The only way out is `expose_for_sdk`, whose sole call site is the SDK
/// credential handoff in this module.
#[derive(Clone)]
pub struct AwsSecret(String);

impl AwsSecret {
    /// Wrap an authored secret.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the raw value for handoff to the AWS SDK.
    fn expose_for_sdk(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for AwsSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl fmt::Display for AwsSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

/// Opaque, non-invertible identity of one credential source.
///
/// Derived from the source *descriptor* only — kind, optional profile name,
/// optional region. No secret is hashed, and the value is stable across refresh
/// so consumers and provenance can bind to it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AwsCredentialSourceId([u8; 8]);

impl AwsCredentialSourceId {
    fn derive(kind: AwsCredentialSourceKind, profile: Option<&str>, region: Option<&str>) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.streaming.aws-credential-source.v1");
        hasher.update(kind.label().as_bytes());
        hasher.update(&[0]);
        hasher.update(profile.unwrap_or_default().as_bytes());
        hasher.update(&[0]);
        hasher.update(region.unwrap_or_default().as_bytes());
        let mut id = [0_u8; 8];
        id.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
        Self(id)
    }
}

impl fmt::Display for AwsCredentialSourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Closed classification of where credential material originates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AwsCredentialSourceKind {
    /// The SDK default provider chain (environment, profile, IMDS, web identity).
    DefaultChain,
    /// A named shared-config profile resolved through the SDK.
    Profile,
    /// Authored static keys, used by local gateways and tests.
    AuthoredStatic,
}

impl AwsCredentialSourceKind {
    /// Static label safe to place in a `tracing` field or a hash input.
    pub const fn label(self) -> &'static str {
        match self {
            Self::DefaultChain => "default-chain",
            Self::Profile => "profile",
            Self::AuthoredStatic => "authored-static",
        }
    }
}

// ---------------------------------------------------------------------------
// Clock projection
// ---------------------------------------------------------------------------

/// Worker-local publisher of run-clock nanoseconds into the shared cell the
/// `Send + Sync` credential authority reads.
///
/// Callers publish immediately before each S3 operation. The store is `Relaxed`
/// and uncontended, so this adds no synchronization to the request path.
pub struct AwsClockProjection {
    clock: Rc<dyn Clock>,
    cell: Arc<AtomicI64>,
}

impl fmt::Debug for AwsClockProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AwsClockProjection")
            .field("published_ns", &self.cell.load(Ordering::Relaxed))
            .finish()
    }
}

impl AwsClockProjection {
    /// Bind this worker's clock to an authority's shared time cell.
    pub fn new(clock: Rc<dyn Clock>, authority: &AwsCredentialProviderAuthority) -> Self {
        let projection = Self {
            clock,
            cell: authority.wall_ns.clone(),
        };
        projection.publish();
        projection
    }

    /// Publish the current run-clock reading.
    pub fn publish(&self) {
        self.cell.store(self.clock.now_ns(), Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Credential authority
// ---------------------------------------------------------------------------

/// Reads wall time from the SDK's own time source.
///
/// Held as a closure rather than a named `SharedTimeSource` so this module does
/// not take a direct `aws-smithy-async` dependency to name a type it only ever
/// calls once per refresh. AIPerf performs no independent wall-clock read: the
/// value comes from the same time source that produced the credential's expiry.
type SdkWallClock = Arc<dyn Fn() -> SystemTime + Send + Sync>;

/// Cached credential material and its run-relative expiry deadline.
struct CachedCredentials {
    credentials: Credentials,
    /// Run-clock nanoseconds after which the cache must refresh; `None` when the
    /// SDK reported no expiry.
    deadline_ns: Option<i64>,
}

impl fmt::Debug for CachedCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never render `Credentials`: its Debug carries the access key id.
        f.debug_struct("CachedCredentials")
            .field("deadline_ns", &self.deadline_ns)
            .finish()
    }
}

/// The sole AWS credential provider shared by every streaming S3 consumer.
///
/// Refresh-capable: the cached material behind a stable
/// [`AwsCredentialSourceId`] is replaced under the run [`Clock`] without the
/// identity, the frozen object identity, or any consumer binding changing.
pub struct AwsCredentialProviderAuthority {
    kind: AwsCredentialSourceKind,
    source_id: AwsCredentialSourceId,
    delegate: Option<SharedCredentialsProvider>,
    sdk_wall_clock: Option<SdkWallClock>,
    static_credentials: Option<Credentials>,
    cached: RwLock<Option<CachedCredentials>>,
    wall_ns: Arc<AtomicI64>,
}

impl fmt::Debug for AwsCredentialProviderAuthority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AwsCredentialProviderAuthority")
            .field("kind", &self.kind)
            .field("source_id", &self.source_id)
            .field("has_cached", &self.cached.read().is_some())
            .finish()
    }
}

impl AwsCredentialProviderAuthority {
    /// Wrap the provider chain of an already-loaded [`SdkConfig`].
    ///
    /// Private on purpose: taking the chain from a config the caller loaded
    /// separately makes the SDK constructor a prerequisite of the authority and
    /// the authority a prerequisite of the factory, which is a construction
    /// cycle callers cannot satisfy. [`AwsS3ClientFactory::prepare_default_chain`]
    /// loads once and calls this on the result.
    fn from_loaded_chain(sdk_config: &SdkConfig, profile: Option<&str>) -> Option<Self> {
        let delegate = sdk_config.credentials_provider()?;
        let kind = if profile.is_some() {
            AwsCredentialSourceKind::Profile
        } else {
            AwsCredentialSourceKind::DefaultChain
        };
        let region = sdk_config.region().map(Region::as_ref);
        let time_source = sdk_config.time_source();
        Some(Self {
            kind,
            source_id: AwsCredentialSourceId::derive(kind, profile, region),
            delegate: Some(delegate),
            sdk_wall_clock: time_source
                .map(|source| Arc::new(move || source.now()) as SdkWallClock),
            static_credentials: None,
            cached: RwLock::new(None),
            wall_ns: Arc::new(AtomicI64::new(0)),
        })
    }

    /// Build an authority over authored static keys (local gateways, tests).
    ///
    /// Static material carries no expiry, so this path needs no wall clock.
    pub fn from_authored(
        access_key_id: &str,
        secret_access_key: &AwsSecret,
        session_token: Option<&AwsSecret>,
        region: Option<&str>,
    ) -> Self {
        let kind = AwsCredentialSourceKind::AuthoredStatic;
        let credentials = Credentials::new(
            access_key_id,
            secret_access_key.expose_for_sdk(),
            session_token.map(|token| token.expose_for_sdk().to_owned()),
            None,
            "aiperf-streaming-authored",
        );
        Self {
            kind,
            source_id: AwsCredentialSourceId::derive(kind, None, region),
            delegate: None,
            sdk_wall_clock: None,
            static_credentials: Some(credentials),
            cached: RwLock::new(None),
            wall_ns: Arc::new(AtomicI64::new(0)),
        }
    }

    /// Stable opaque identity, safe for errors, logs, and provenance.
    pub fn source_id(&self) -> AwsCredentialSourceId {
        self.source_id
    }

    /// Classification of the credential source.
    pub fn kind(&self) -> AwsCredentialSourceKind {
        self.kind
    }

    /// Drop cached material so the next acquisition refreshes.
    ///
    /// The source adapter calls this after an authentication failure, before its
    /// `Clock`-driven retry. The source identity is unchanged, so a later
    /// successful retry still acquires the same frozen object.
    pub fn invalidate(&self) {
        *self.cached.write() = None;
    }

    /// Cached material, when it is still usable at the published run time.
    fn cached_hit(&self) -> Option<Credentials> {
        let now_ns = self.wall_ns.load(Ordering::Relaxed);
        let guard = self.cached.read();
        let entry = guard.as_ref()?;
        match entry.deadline_ns {
            Some(deadline_ns) if now_ns >= deadline_ns => None,
            _ => Some(entry.credentials.clone()),
        }
    }

    async fn resolve(&self) -> Result<Credentials, CredentialsError> {
        // The read guard is dropped inside `cached_hit`, so no lock is held
        // across the delegate `.await` below.
        if let Some(hit) = self.cached_hit() {
            return Ok(hit);
        }
        if let Some(credentials) = &self.static_credentials {
            return Ok(credentials.clone());
        }
        let Some(delegate) = &self.delegate else {
            return Err(CredentialsError::not_loaded(
                "no streaming AWS credential source is configured",
            ));
        };
        let credentials = delegate.provide_credentials().await?;
        let now_ns = self.wall_ns.load(Ordering::Relaxed);
        let deadline_ns =
            run_relative_deadline_ns(now_ns, credentials.expiry(), self.sdk_wall_clock.as_deref());
        tracing::debug!(
            source_id = %self.source_id,
            kind = self.kind.label(),
            has_expiry = deadline_ns.is_some(),
            "refreshed streaming AWS credentials"
        );
        *self.cached.write() = Some(CachedCredentials {
            credentials: credentials.clone(),
            deadline_ns,
        });
        Ok(credentials)
    }
}

impl ProvideCredentials for AwsCredentialProviderAuthority {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(self.resolve())
    }
}

/// Project an SDK-reported absolute expiry onto the run clock, once.
///
/// `Credentials::expiry` is absolute wall time and cannot be compared against a
/// monotonic run clock. Its *remaining lifetime* can: it is measured against the
/// SDK's own time source (the same source that produced the expiry), and only
/// that duration is added to the run clock. Returns `None` when the SDK reported
/// no expiry or no time source, which caches until an explicit invalidation.
fn run_relative_deadline_ns(
    now_ns: i64,
    expiry: Option<SystemTime>,
    sdk_wall_clock: Option<&(dyn Fn() -> SystemTime + Send + Sync)>,
) -> Option<i64> {
    let expiry = expiry?;
    let sdk_now = sdk_wall_clock?();
    // An already-elapsed expiry yields a zero remaining lifetime, so the very
    // next acquisition refreshes rather than reusing stale material.
    let remaining_ns = expiry
        .duration_since(sdk_now)
        .ok()
        .and_then(|remaining| i64::try_from(remaining.as_nanos()).ok())
        .unwrap_or(0);
    let usable_ns = remaining_ns
        .saturating_sub(CREDENTIAL_REFRESH_SKEW_NS)
        .max(0);
    now_ns.checked_add(usable_ns)
}

// ---------------------------------------------------------------------------
// Authored settings
// ---------------------------------------------------------------------------

/// How the streaming AWS client selects a forward proxy.
///
/// Mirrors the HTTP transport's `--proxy` / `--proxy-from-env` opt-in exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AwsProxySelection {
    /// No proxy; the ambient environment is ignored.
    Disabled,
    /// One authored proxy URL, applied as authored.
    Explicit(String),
    /// Opt into the ambient proxy environment, loopback always excluded.
    FromEnvironment,
}

impl AwsProxySelection {
    /// Static label safe to place in a `tracing` field.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Explicit(_) => "explicit",
            Self::FromEnvironment => "environment",
        }
    }
}

/// Validated, redacted client construction inputs.
pub struct AwsClientSettings {
    /// Authored region; `None` defers to the SDK region chain.
    pub region: Option<String>,
    /// Authored endpoint override (MinIO, S3-compatible gateways).
    pub endpoint_url: Option<String>,
    /// Path-style addressing, required by most S3-compatible gateways.
    pub force_path_style: bool,
    /// Proxy selection.
    pub proxy: AwsProxySelection,
    /// Bounded per-operation-attempt timeout.
    pub operation_timeout_ns: i64,
    /// Bounded connect timeout.
    pub connect_timeout_ns: i64,
}

impl fmt::Debug for AwsClientSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `endpoint_url` may carry userinfo; render host only.
        f.debug_struct("AwsClientSettings")
            .field("region", &self.region)
            .field("endpoint_host", &self.endpoint_host())
            .field("force_path_style", &self.force_path_style)
            .field("proxy", &self.proxy.label())
            .field("operation_timeout_ns", &self.operation_timeout_ns)
            .field("connect_timeout_ns", &self.connect_timeout_ns)
            .finish()
    }
}

impl AwsClientSettings {
    /// Host of the authored endpoint, when one is authored and parses.
    fn endpoint_host(&self) -> Option<String> {
        let raw = self.endpoint_url.as_deref()?;
        Url::parse(raw).ok()?.host_str().map(str::to_owned)
    }

    /// URL used to decide whether the ambient proxy applies.
    ///
    /// An authored endpoint is its own probe; otherwise the default regional S3
    /// host is synthesized so the decision is made before any network call.
    fn proxy_probe_url(&self) -> Result<Url, AwsClientError> {
        if let Some(raw) = self.endpoint_url.as_deref() {
            return Url::parse(raw).map_err(|_| AwsClientError::InvalidEndpoint);
        }
        let region = self.region.as_deref().unwrap_or("us-east-1");
        Url::parse(&format!("https://s3.{region}.amazonaws.com"))
            .map_err(|_| AwsClientError::InvalidEndpoint)
    }

    /// Reject unbounded or unusable authored inputs before any client exists.
    fn validate(&self) -> Result<(), AwsClientError> {
        for value in [self.operation_timeout_ns, self.connect_timeout_ns] {
            if value <= 0 || value > MAX_OPERATION_TIMEOUT_NS {
                return Err(AwsClientError::InvalidTimeout);
            }
        }
        if let Some(raw) = self.endpoint_url.as_deref() {
            let url = Url::parse(raw).map_err(|_| AwsClientError::InvalidEndpoint)?;
            if url.host_str().is_none() {
                return Err(AwsClientError::InvalidEndpoint);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Closed, credential-free client-construction failure.
///
/// Every variant is a fixed classification. No SDK error text, endpoint URL,
/// proxy URL, or credential is ever carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AwsClientError {
    /// The authored endpoint URL is not a usable absolute URL.
    InvalidEndpoint,
    /// The authored proxy URL is not a usable absolute URL.
    InvalidProxy,
    /// An authored timeout is non-positive or exceeds the supported bound.
    InvalidTimeout,
    /// No credential source could be resolved before the run started.
    NoCredentialSource,
}

impl fmt::Display for AwsClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidEndpoint => "the authored S3 endpoint URL is not a usable absolute URL",
            Self::InvalidProxy => "the authored proxy URL is not a usable absolute URL",
            Self::InvalidTimeout => "an authored S3 timeout is non-positive or above the bound",
            Self::NoCredentialSource => "no AWS credential source could be resolved",
        };
        f.write_str(message)
    }
}

impl std::error::Error for AwsClientError {}

impl From<AwsClientError> for StreamSourceError {
    fn from(_: AwsClientError) -> Self {
        // The streaming seam's error is a stringless closed code, which is what
        // structurally guarantees no credential or URL escapes this module.
        StreamSourceError::source(SourceFailureCode::SourceUnavailable)
    }
}

// ---------------------------------------------------------------------------
// Proxy resolution
// ---------------------------------------------------------------------------

/// Translate the AIPerf proxy decision into the SDK connector's vocabulary.
///
/// `SdkProxyConfig::from_env()` is deliberately never used: it delegates
/// matching to hyper-util, which does not exclude loopback, so an ambient
/// `HTTP_PROXY` would be applied to a local S3 gateway. The decision is made
/// here by [`crate::transport::http::client::proxy`], the same authority the
/// HTTP transport uses.
fn resolve_sdk_proxy(
    selection: &AwsProxySelection,
    probe: &Url,
) -> Result<SdkProxyConfig, AwsClientError> {
    let resolved = match selection {
        AwsProxySelection::Disabled => None,
        AwsProxySelection::Explicit(raw) => {
            Some(host_proxy::ProxyConfig::parse(raw).ok_or(AwsClientError::InvalidProxy)?)
        }
        AwsProxySelection::FromEnvironment => host_proxy::ProxyConfig::from_env_for(probe),
    };
    let Some(resolved) = resolved else {
        return Ok(SdkProxyConfig::disabled());
    };
    // Userinfo stays in the typed auth field rather than the URI: the SDK's
    // `ProxyAuth` derives `Debug` over a plaintext password, so the assembled
    // config is never stored in an AIPerf value and never logged.
    let url = format!("http://{}:{}", resolved.host, resolved.port);
    let config = SdkProxyConfig::all(url.as_str())
        .map_err(|_| AwsClientError::InvalidProxy)?
        .no_proxy(CONNECTOR_NO_PROXY_RULES);
    Ok(match decode_basic_auth(resolved.auth_header.as_deref()) {
        Some((user, password)) => config.with_basic_auth(user, password),
        None => config,
    })
}

/// Recover the user/password pair the host authority encoded as a
/// `Proxy-Authorization` header value.
fn decode_basic_auth(header: Option<&str>) -> Option<(String, String)> {
    use base64::Engine;

    let encoded = header?.strip_prefix("Basic ")?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    let (user, password) = decoded.split_once(':')?;
    Some((user.to_owned(), password.to_owned()))
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Constructs worker-local S3 clients from one shared credential authority.
///
/// Exposes construction only: no list, get, put, or compare-and-swap policy
/// lives here. The S3 source owns reconciliation, identity, and pagination
/// policy over its own narrow client trait and merely asks this factory for
/// transport.
pub struct AwsS3ClientFactory {
    settings: AwsClientSettings,
    authority: Arc<AwsCredentialProviderAuthority>,
    sdk_config: SdkConfig,
}

impl fmt::Debug for AwsS3ClientFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `SdkConfig`'s Debug can render a credential provider; render the
        // authority's opaque id instead.
        f.debug_struct("AwsS3ClientFactory")
            .field("settings", &self.settings)
            .field("credential_source_id", &self.authority.source_id())
            .finish()
    }
}

impl AwsS3ClientFactory {
    /// Resolve region, endpoint, proxy, TLS, and timing inputs once against an
    /// authority the caller already owns (authored static credentials).
    ///
    /// Runs during source preparation, before discovery begins, so an authored
    /// mistake is a preflight failure rather than a mid-run fault.
    pub async fn prepare(
        settings: AwsClientSettings,
        authority: Arc<AwsCredentialProviderAuthority>,
    ) -> Result<Self, AwsClientError> {
        let sdk_config = load_sdk_config(&settings).await?;
        Ok(Self::assemble(settings, authority, sdk_config))
    }

    /// Resolve the same inputs and adopt the SDK's own provider chain.
    ///
    /// The chain can only be read from a loaded [`SdkConfig`], and loading it
    /// requires the AIPerf connector this function installs, so the authority is
    /// derived here rather than supplied by the caller. That is what keeps
    /// construction acyclic while leaving exactly one credential authority.
    pub async fn prepare_default_chain(
        settings: AwsClientSettings,
        profile: Option<&str>,
    ) -> Result<Self, AwsClientError> {
        let sdk_config = load_sdk_config(&settings).await?;
        let authority = AwsCredentialProviderAuthority::from_loaded_chain(&sdk_config, profile)
            .ok_or(AwsClientError::NoCredentialSource)?;
        Ok(Self::assemble(settings, Arc::new(authority), sdk_config))
    }

    fn assemble(
        settings: AwsClientSettings,
        authority: Arc<AwsCredentialProviderAuthority>,
        sdk_config: SdkConfig,
    ) -> Self {
        tracing::debug!(
            credential_source_id = %authority.source_id(),
            region = ?settings.region,
            endpoint_host = ?settings.endpoint_host(),
            proxy = settings.proxy.label(),
            "prepared streaming AWS S3 client factory"
        );
        Self {
            settings,
            authority,
            sdk_config,
        }
    }

    /// The shared authority; the S3 source binds to this rather than defining
    /// its own.
    pub fn authority(&self) -> &Arc<AwsCredentialProviderAuthority> {
        &self.authority
    }

    /// Build one worker-local client.
    ///
    /// `clock` binds this worker's run clock to the authority's shared time
    /// cell. The returned projection must outlive the client, and the caller
    /// publishes before each operation.
    pub fn build_client(&self, clock: Rc<dyn Clock>) -> (aws_sdk_s3::Client, AwsClockProjection) {
        let projection = AwsClockProjection::new(clock, &self.authority);
        // There is no blanket `ProvideCredentials for Arc<T>`, only one for
        // `Arc<dyn ProvideCredentials>`, so the shared authority is coerced.
        let provider: Arc<dyn ProvideCredentials> = self.authority.clone();
        let mut builder = aws_sdk_s3::config::Builder::from(&self.sdk_config)
            .credentials_provider(provider)
            .force_path_style(self.settings.force_path_style);
        if let Some(endpoint) = self.settings.endpoint_url.clone() {
            builder = builder.endpoint_url(endpoint);
        }
        (aws_sdk_s3::Client::from_conf(builder.build()), projection)
    }
}

/// Load an [`SdkConfig`] carrying the AIPerf connector, disabled SDK retry, and
/// no SDK identity cache.
async fn load_sdk_config(settings: &AwsClientSettings) -> Result<SdkConfig, AwsClientError> {
    settings.validate()?;
    let probe = settings.proxy_probe_url()?;
    let proxy = resolve_sdk_proxy(&settings.proxy, &probe)?;

    // Mirrors `aws_smithy_runtime::client::http::default_https_client`, with one
    // difference: the proxy config is AIPerf's, never `from_env()`. Re-check
    // this against upstream on any SDK bump.
    let http_client = SdkHttpClientBuilder::new().build_with_connector_fn(
        move |connector_settings, runtime_components| {
            let mut builder = Connector::builder().tls_provider(tls::Provider::Rustls(
                tls::rustls_provider::CryptoMode::AwsLc,
            ));
            builder.set_connector_settings(connector_settings.cloned());
            if let Some(components) = runtime_components {
                builder.set_sleep_impl(components.sleep_impl());
            }
            builder.set_proxy_config(Some(proxy.clone()));
            builder.build()
        },
    );

    let timeouts = TimeoutConfig::builder()
        .operation_attempt_timeout(duration_from_ns(settings.operation_timeout_ns))
        .connect_timeout(duration_from_ns(settings.connect_timeout_ns))
        .build();

    let mut loader = aws_config::defaults(BehaviorVersion::latest())
        .http_client(http_client)
        // The source adapter owns backoff under the injected `Clock`; the SDK
        // sleeper cannot be backed by `Clock::sleep` (non-`Send` future), so
        // there is exactly one retry authority and it is the clocked one.
        .retry_config(RetryConfig::disabled())
        .timeout_config(timeouts)
        // The authority is the single refresh decision point; an SDK identity
        // cache would be a second, unclocked one.
        .identity_cache(IdentityCache::no_cache());
    if let Some(region) = settings.region.clone() {
        loader = loader.region(Region::new(region));
    }
    Ok(loader.load().await)
}

/// Convert validated positive nanoseconds to a `Duration`.
///
/// `AwsClientSettings::validate` has already rejected non-positive and
/// out-of-bound values, so the clamp below can never discard a real value.
fn duration_from_ns(value_ns: i64) -> Duration {
    Duration::from_nanos(value_ns.max(0).unsigned_abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_id_is_stable_and_descriptor_derived() {
        let first = AwsCredentialSourceId::derive(
            AwsCredentialSourceKind::AuthoredStatic,
            None,
            Some("us-east-1"),
        );
        let second = AwsCredentialSourceId::derive(
            AwsCredentialSourceKind::AuthoredStatic,
            None,
            Some("us-east-1"),
        );
        let other = AwsCredentialSourceId::derive(
            AwsCredentialSourceKind::AuthoredStatic,
            None,
            Some("eu-west-1"),
        );
        assert_eq!(first, second);
        assert_ne!(first, other);
        assert_eq!(first.to_string().len(), 16);
    }

    #[test]
    fn expiry_projects_onto_the_run_clock_and_clamps_when_elapsed() {
        let sdk_now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let wall = move || sdk_now;
        // 10 minutes of lifetime, minus the 60s refresh skew.
        let live =
            run_relative_deadline_ns(5_000, Some(sdk_now + Duration::from_secs(600)), Some(&wall));
        assert_eq!(live, Some(5_000 + 540_000_000_000));

        // Already expired: refresh on the very next acquisition.
        let elapsed =
            run_relative_deadline_ns(5_000, Some(sdk_now - Duration::from_secs(1)), Some(&wall));
        assert_eq!(elapsed, Some(5_000));

        // No expiry reported: cache until an explicit invalidation.
        assert_eq!(run_relative_deadline_ns(5_000, None, Some(&wall)), None);
    }

    #[test]
    fn authored_timings_and_endpoints_are_bounded() {
        let mut settings = AwsClientSettings {
            region: Some("us-east-1".to_owned()),
            endpoint_url: Some("http://127.0.0.1:9000".to_owned()),
            force_path_style: true,
            proxy: AwsProxySelection::Disabled,
            operation_timeout_ns: 30_000_000_000,
            connect_timeout_ns: 5_000_000_000,
        };
        assert!(settings.validate().is_ok());

        settings.operation_timeout_ns = 0;
        assert_eq!(settings.validate(), Err(AwsClientError::InvalidTimeout));

        settings.operation_timeout_ns = MAX_OPERATION_TIMEOUT_NS + 1;
        assert_eq!(settings.validate(), Err(AwsClientError::InvalidTimeout));

        settings.operation_timeout_ns = 30_000_000_000;
        settings.endpoint_url = Some("not-a-url".to_owned());
        assert_eq!(settings.validate(), Err(AwsClientError::InvalidEndpoint));
    }
}
