// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Redacted, refresh-capable Hugging Face credential authority and its HTTP client.
//!
//! This module is the single place the native streaming path learns a Hugging
//! Face token. It resolves one *pinned* credential source at preparation, caches
//! the material worker-locally, and hands out generation-stamped leases. When a
//! consumer reports an authentication failure it re-reads that same pinned
//! source under the injected [`Clock`], bounded twice: `max_refresh_attempts`
//! per authentication episode and `max_total_refreshes` per run.
//!
//! Three properties are load-bearing and are enforced by construction rather
//! than by convention:
//!
//! * **Identity is orthogonal to material.** The authority holds no repository,
//!   revision, commit, shard, length, or digest, so a refresh cannot change
//!   which immutable object is being acquired. That is what makes a consumer's
//!   "retry, then partition hole" legal while identity drift stays a fail-run.
//! * **Material cannot reach an observer.** No type here implements `Serialize`,
//!   `Debug` is hand written everywhere a secret is reachable, [`HfSecret`]'s
//!   `Debug`/`Display` render `<redacted>`, and errors carry closed codes
//!   instead of strings.
//! * **The proxy decision belongs to AIPerf.** `hf-hub` 0.4.3's `ureq` backend
//!   sets `try_proxy_from_env(true)` unconditionally, and `ureq` 2.12's
//!   `Proxy::try_from_system` honors neither `NO_PROXY` nor loopback. A local
//!   `HF_ENDPOINT` would therefore be routed through an ambient `HTTP_PROXY`.
//!   Streaming acquisition instead runs over the clock-injected Hyper transport
//!   with the proxy decision made by [`crate::transport::http::client::proxy`],
//!   which excludes every loopback host. [`crate::dataset::hf_hub`] keeps using
//!   `hf-hub` for tokenizer downloads and is unaffected.
//!
//! Generation one redacts credential material; it does not erase it. Secret
//! memory is not zeroized, per the benchmark security-scope course correction,
//! which removed the `zeroize` dependency from the streaming feature graph.

use std::cell::RefCell;
use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::watch;
use tracing::debug;
use url::Url;

use crate::clock::Clock;
use crate::transport::core::RequestRecord;
use crate::transport::http::client::proxy as host_proxy;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;

use super::blocking::{
    BlockingWorkBudget, BlockingWorkClass, BlockingWorkError, StreamingBlockingExecutor,
};

/// Environment variable carrying a Hugging Face access token.
pub const HF_TOKEN_ENV: &str = "HF_TOKEN";
/// Legacy environment variable name still honored by the Python client.
pub const HF_TOKEN_ENV_LEGACY: &str = "HUGGING_FACE_HUB_TOKEN";

/// Largest accepted token, in bytes. Real tokens are well under 128 bytes; the
/// bound exists so a mis-pointed path cannot pull an arbitrary file into memory.
const MAX_TOKEN_BYTES: usize = 4096;

/// Default per-episode refresh attempts before exhaustion.
const DEFAULT_MAX_REFRESH_ATTEMPTS: u32 = 3;
/// Default whole-run refresh ceiling.
const DEFAULT_MAX_TOTAL_REFRESHES: u32 = 64;
/// Default first-attempt backoff, in clock nanoseconds.
const DEFAULT_REFRESH_BACKOFF_BASE_NS: i64 = 250_000_000;
/// Default backoff ceiling, in clock nanoseconds.
const DEFAULT_REFRESH_BACKOFF_CAP_NS: i64 = 5_000_000_000;
/// Largest accepted backoff ceiling, in clock nanoseconds.
const MAX_REFRESH_BACKOFF_CAP_NS: i64 = 60_000_000_000;
/// Default response-body ceiling for hub API and ranged shard reads.
const DEFAULT_MAX_RESPONSE_BODY_BYTES: u64 = 64 * 1024 * 1024;

/// BLAKE3 domain separator for credential-source identity.
const SOURCE_ID_DOMAIN: &[u8] = b"aiperf.streaming.hf.credential-source.v1";
/// BLAKE3 domain separator for material-rotation comparison.
const MATERIAL_DIGEST_DOMAIN: &[u8] = b"aiperf.streaming.hf.credential-material.v1";

// ---------------------------------------------------------------------------
// Secret
// ---------------------------------------------------------------------------

/// One validated Hugging Face bearer token.
///
/// Neither `Debug` nor `Display` renders the material, and the type implements
/// no serialization trait, so it cannot reach a checkpoint payload, an artifact,
/// a provenance record, or a `tracing` field.
pub struct HfSecret {
    bearer: String,
    digest: [u8; 32],
}

impl HfSecret {
    /// Validate and retain one token.
    ///
    /// Rejects empty, oversized, and non-visible-ASCII material. The last check
    /// is not cosmetic: a token file ending in `\r\n` would otherwise become a
    /// header-injection vector at stamping time.
    fn new(raw: &str) -> Result<Self, HfCredentialError> {
        let bearer = raw.trim();
        if bearer.is_empty() || bearer.len() > MAX_TOKEN_BYTES {
            return Err(HfCredentialError::MalformedMaterial);
        }
        if !bearer.bytes().all(|byte| (0x21..=0x7e).contains(&byte)) {
            return Err(HfCredentialError::MalformedMaterial);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(MATERIAL_DIGEST_DOMAIN);
        hasher.update(bearer.as_bytes());
        Ok(Self {
            bearer: bearer.to_owned(),
            digest: *hasher.finalize().as_bytes(),
        })
    }

    /// Borrow the bearer value for the single header-stamping call site.
    fn expose_bearer(&self) -> &str {
        &self.bearer
    }

    /// Borrow the rotation-comparison digest.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

impl fmt::Debug for HfSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HfSecret(<redacted>)")
    }
}

impl fmt::Display for HfSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

// ---------------------------------------------------------------------------
// Source identity
// ---------------------------------------------------------------------------

/// Which pinned place the material is read from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfCredentialSourceKind {
    /// No credential is used; only public repositories are reachable.
    Anonymous,
    /// A named environment variable.
    Environment,
    /// An on-disk token file.
    TokenFile,
}

impl HfCredentialSourceKind {
    /// Stable lowercase label for diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Anonymous => "anonymous",
            Self::Environment => "environment",
            Self::TokenFile => "token_file",
        }
    }
}

/// Opaque identity of one credential source, derived from its descriptor and
/// from no secret input.
///
/// Deliberately not serializable: a consumer that needs the value in its own
/// checkpoint payload copies [`Self::as_bytes`] explicitly.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HfCredentialSourceId([u8; 32]);

impl HfCredentialSourceId {
    /// Borrow the canonical digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Render the 16-hex-character diagnostic prefix.
    #[must_use]
    pub fn short_hex(&self) -> String {
        let mut rendered = String::with_capacity(16);
        for byte in &self.0[..8] {
            // Writing to a String is infallible; the Result is unused rather
            // than unwrapped so this path stays panic-free.
            let _ = write!(rendered, "{byte:02x}");
        }
        rendered
    }
}

impl fmt::Debug for HfCredentialSourceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "HfCredentialSourceId({})", self.short_hex())
    }
}

impl fmt::Display for HfCredentialSourceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.short_hex())
    }
}

/// The frozen description of where material is read from.
#[derive(Clone)]
pub struct HfCredentialSourceDescriptor {
    kind: HfCredentialSourceKind,
    env_var: Option<String>,
    token_path: Option<PathBuf>,
    endpoint_host: String,
}

impl HfCredentialSourceDescriptor {
    /// Describe an anonymous source against `endpoint_host`.
    #[must_use]
    pub fn anonymous(endpoint_host: impl Into<String>) -> Self {
        Self {
            kind: HfCredentialSourceKind::Anonymous,
            env_var: None,
            token_path: None,
            endpoint_host: endpoint_host.into(),
        }
    }

    /// Describe an environment-variable source.
    #[must_use]
    pub fn environment(name: impl Into<String>, endpoint_host: impl Into<String>) -> Self {
        Self {
            kind: HfCredentialSourceKind::Environment,
            env_var: Some(name.into()),
            token_path: None,
            endpoint_host: endpoint_host.into(),
        }
    }

    /// Describe a token-file source.
    #[must_use]
    pub fn token_file(path: impl Into<PathBuf>, endpoint_host: impl Into<String>) -> Self {
        Self {
            kind: HfCredentialSourceKind::TokenFile,
            env_var: None,
            token_path: Some(path.into()),
            endpoint_host: endpoint_host.into(),
        }
    }

    /// Return the pinned source kind.
    #[must_use]
    pub const fn kind(&self) -> HfCredentialSourceKind {
        self.kind
    }

    /// Borrow the pinned environment-variable name, when the kind uses one.
    #[must_use]
    pub fn env_var(&self) -> Option<&str> {
        self.env_var.as_deref()
    }

    /// Borrow the pinned token-file path, when the kind uses one.
    #[must_use]
    pub fn token_path(&self) -> Option<&Path> {
        self.token_path.as_deref()
    }

    /// Borrow the pinned endpoint host the identity is bound to.
    #[must_use]
    pub fn endpoint_host(&self) -> &str {
        &self.endpoint_host
    }

    /// Derive the stable opaque source identity.
    #[must_use]
    pub fn source_id(&self) -> HfCredentialSourceId {
        let mut hasher = blake3::Hasher::new();
        // Length-prefixed fields so no two descriptors can concatenate to the
        // same pre-image.
        let mut field = |bytes: &[u8]| {
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        };
        field(SOURCE_ID_DOMAIN);
        field(self.kind.label().as_bytes());
        field(self.env_var.as_deref().unwrap_or_default().as_bytes());
        field(
            self.token_path
                .as_deref()
                .map(|path| path.as_os_str().as_encoded_bytes())
                .unwrap_or_default(),
        );
        field(self.endpoint_host.as_bytes());
        HfCredentialSourceId(*hasher.finalize().as_bytes())
    }
}

// A path is not a secret, but it is not diagnostics either: only kind, host, and
// the derived id are rendered.
impl fmt::Debug for HfCredentialSourceDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HfCredentialSourceDescriptor")
            .field("kind", &self.kind.label())
            .field("endpoint_host", &self.endpoint_host)
            .field("source_id", &self.source_id())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Closed credential-seam failure classification.
///
/// Every variant renders as its stable code only; no URL, header, path, or
/// material is ever formatted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfCredentialError {
    /// The pinned credential source resolved to nothing.
    SourceUnavailable,
    /// Material was empty, oversized, or contained non-visible-ASCII bytes.
    MalformedMaterial,
    /// The authored credential or HTTP settings are invalid.
    InvalidSettings,
    /// The per-episode refresh attempts are spent.
    RefreshExhausted,
    /// The whole-run refresh ceiling is reached.
    RefreshBudgetExhausted,
    /// The bounded blocking owner refused or failed the read.
    BlockingUnavailable,
    /// Shutdown cancelled the operation.
    Cancelled,
}

impl HfCredentialError {
    /// Return the stable lowercase failure code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::SourceUnavailable => "hf_credential_source_unavailable",
            Self::MalformedMaterial => "hf_credential_malformed_material",
            Self::InvalidSettings => "hf_credential_invalid_settings",
            Self::RefreshExhausted => "hf_credential_refresh_exhausted",
            Self::RefreshBudgetExhausted => "hf_credential_refresh_budget_exhausted",
            Self::BlockingUnavailable => "hf_credential_blocking_unavailable",
            Self::Cancelled => "hf_credential_cancelled",
        }
    }

    /// Whether a consumer may retry after this failure.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::SourceUnavailable | Self::BlockingUnavailable)
    }
}

impl fmt::Display for HfCredentialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

impl std::error::Error for HfCredentialError {}

impl From<BlockingWorkError> for HfCredentialError {
    // The blocking owner's `Join` variant carries a Tokio diagnostic string; it
    // is dropped rather than wrapped so this type stays stringless.
    fn from(error: BlockingWorkError) -> Self {
        match error {
            BlockingWorkError::Cancelled | BlockingWorkError::SubmissionClosed => Self::Cancelled,
            _ => Self::BlockingUnavailable,
        }
    }
}

// ---------------------------------------------------------------------------
// Material reader
// ---------------------------------------------------------------------------

/// Injected reader for the pinned credential source.
///
/// This is the replacement point for a distribution that resolves tokens
/// differently, and the seam that lets tests exercise rotation without mutating
/// process environment (`std::env::set_var` is `unsafe` in edition 2024 and racy
/// under a parallel test binary).
///
/// The implementation runs inside a bounded blocking closure, so it must be
/// `Send + Sync + 'static`; that is the only reason an `Arc` appears in this
/// otherwise worker-local module.
pub trait HfCredentialMaterialReader: fmt::Debug + Send + Sync + 'static {
    /// Read the material for exactly `descriptor`'s pinned source.
    ///
    /// `Ok(None)` means the pinned source is absent; it never means "fall back".
    fn read(
        &self,
        descriptor: &HfCredentialSourceDescriptor,
    ) -> Result<Option<String>, HfCredentialError>;
}

/// Production reader over process environment and the on-disk token file.
#[derive(Debug, Default)]
pub struct ProcessHfCredentialReader;

impl HfCredentialMaterialReader for ProcessHfCredentialReader {
    fn read(
        &self,
        descriptor: &HfCredentialSourceDescriptor,
    ) -> Result<Option<String>, HfCredentialError> {
        match descriptor.kind() {
            HfCredentialSourceKind::Anonymous => Ok(None),
            HfCredentialSourceKind::Environment => {
                let name = descriptor
                    .env_var()
                    .ok_or(HfCredentialError::InvalidSettings)?;
                Ok(std::env::var(name)
                    .ok()
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty()))
            }
            HfCredentialSourceKind::TokenFile => {
                let path = descriptor
                    .token_path()
                    .ok_or(HfCredentialError::InvalidSettings)?;
                match std::fs::read_to_string(path) {
                    Ok(raw) => {
                        let raw = raw.trim().to_owned();
                        Ok((!raw.is_empty()).then_some(raw))
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                    // The path is deliberately not echoed: it can name a home
                    // directory, and this type is stringless by contract.
                    Err(_) => Err(HfCredentialError::SourceUnavailable),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Authored credential settings.
///
/// There is no field that can hold a literal token: only a variable name and a
/// file path. `aiperf config expand` prints the resolved configuration and the
/// projected protocol-v2 request crosses a stdio boundary, so an inline token
/// field would land in both.
#[derive(Clone, Debug)]
pub struct HfCredentialSettings {
    /// Hub endpoint; only its host participates in the source identity.
    pub endpoint: Url,
    /// Authored environment-variable name, when the run pins one.
    pub authored_env_var: Option<String>,
    /// Authored token-file path, when the run pins one.
    pub authored_token_file: Option<PathBuf>,
    /// Whether an absent credential may resolve to anonymous access.
    pub allow_anonymous: bool,
    /// Refresh attempts per authentication episode.
    pub max_refresh_attempts: u32,
    /// Refresh ceiling for the whole run.
    pub max_total_refreshes: u32,
    /// First-attempt backoff, in clock nanoseconds.
    pub refresh_backoff_base_ns: i64,
    /// Backoff ceiling, in clock nanoseconds.
    pub refresh_backoff_cap_ns: i64,
}

impl HfCredentialSettings {
    /// Construct default settings for `endpoint`.
    #[must_use]
    pub fn new(endpoint: Url) -> Self {
        Self {
            endpoint,
            authored_env_var: None,
            authored_token_file: None,
            allow_anonymous: true,
            max_refresh_attempts: DEFAULT_MAX_REFRESH_ATTEMPTS,
            max_total_refreshes: DEFAULT_MAX_TOTAL_REFRESHES,
            refresh_backoff_base_ns: DEFAULT_REFRESH_BACKOFF_BASE_NS,
            refresh_backoff_cap_ns: DEFAULT_REFRESH_BACKOFF_CAP_NS,
        }
    }

    fn endpoint_host(&self) -> Result<String, HfCredentialError> {
        self.endpoint
            .host_str()
            .map(str::to_ascii_lowercase)
            .ok_or(HfCredentialError::InvalidSettings)
    }

    fn validate(&self) -> Result<(), HfCredentialError> {
        let is_backoff_bounded = self.refresh_backoff_base_ns > 0
            && self.refresh_backoff_cap_ns >= self.refresh_backoff_base_ns
            && self.refresh_backoff_cap_ns <= MAX_REFRESH_BACKOFF_CAP_NS;
        if self.max_refresh_attempts == 0 || self.max_total_refreshes == 0 || !is_backoff_bounded {
            return Err(HfCredentialError::InvalidSettings);
        }
        if self
            .authored_env_var
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(HfCredentialError::InvalidSettings);
        }
        Ok(())
    }

    /// Choose the one pinned source, in the order the finite dataset loader
    /// already uses (`crate::dataset::loader`).
    ///
    /// Selection reads only *names and paths*, never material, so it is cheap,
    /// non-blocking, and safe to run before the blocking owner exists.
    fn resolve_descriptor(&self) -> Result<HfCredentialSourceDescriptor, HfCredentialError> {
        let host = self.endpoint_host()?;
        if let Some(name) = self.authored_env_var.as_deref() {
            return Ok(HfCredentialSourceDescriptor::environment(name, host));
        }
        if let Some(path) = self.authored_token_file.as_deref() {
            return Ok(HfCredentialSourceDescriptor::token_file(path, host));
        }
        for name in [HF_TOKEN_ENV, HF_TOKEN_ENV_LEGACY] {
            if std::env::var_os(name).is_some_and(|value| !value.is_empty()) {
                return Ok(HfCredentialSourceDescriptor::environment(name, host));
            }
        }
        let candidate = std::env::var_os("HF_TOKEN_PATH")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("token")))
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|home| PathBuf::from(home).join(".cache/huggingface/token"))
            });
        if let Some(path) = candidate.filter(|path| path.is_file()) {
            return Ok(HfCredentialSourceDescriptor::token_file(path, host));
        }
        if self.allow_anonymous {
            Ok(HfCredentialSourceDescriptor::anonymous(host))
        } else {
            Err(HfCredentialError::SourceUnavailable)
        }
    }
}

// ---------------------------------------------------------------------------
// Lease and provider contract
// ---------------------------------------------------------------------------

/// Monotonic material generation behind one stable source identity.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CredentialGeneration(u64);

impl CredentialGeneration {
    /// Return the raw generation counter.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One consumer-held view of the current material.
#[derive(Clone)]
pub struct HfCredentialLease {
    generation: CredentialGeneration,
    source_id: HfCredentialSourceId,
    secret: Option<Rc<HfSecret>>,
}

impl HfCredentialLease {
    /// Return the generation this lease was taken at.
    #[must_use]
    pub const fn generation(&self) -> CredentialGeneration {
        self.generation
    }

    /// Return the stable source identity.
    #[must_use]
    pub const fn source_id(&self) -> HfCredentialSourceId {
        self.source_id
    }

    /// Whether this lease carries no credential.
    #[must_use]
    pub const fn is_anonymous(&self) -> bool {
        self.secret.is_none()
    }

    fn bearer(&self) -> Option<&str> {
        self.secret.as_deref().map(HfSecret::expose_bearer)
    }
}

impl fmt::Debug for HfCredentialLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HfCredentialLease")
            .field("source_id", &self.source_id)
            .field("generation", &self.generation.0)
            .field(
                "credential",
                &if self.is_anonymous() {
                    "absent"
                } else {
                    "<redacted>"
                },
            )
            .finish()
    }
}

/// Result of one bounded refresh.
#[derive(Clone, Debug)]
pub enum HfRefreshOutcome {
    /// New material replaced the invalidated generation.
    Rotated(HfCredentialLease),
    /// The pinned source still holds byte-identical material.
    Unchanged(HfCredentialLease),
    /// Another caller already refreshed past the reported generation.
    Superseded(HfCredentialLease),
}

impl HfRefreshOutcome {
    /// Borrow the lease every outcome carries.
    #[must_use]
    pub const fn lease(&self) -> &HfCredentialLease {
        match self {
            Self::Rotated(lease) | Self::Unchanged(lease) | Self::Superseded(lease) => lease,
        }
    }
}

/// Injected, redacted, refresh-capable Hugging Face credential seam.
///
/// `?Send` because the authority is worker-local and holds `Rc<dyn Clock>`;
/// `&self` because consumers share it through an [`Rc`] and refresh
/// concurrently.
#[async_trait(?Send)]
pub trait HfCredentialProvider: fmt::Debug {
    /// Return the stable opaque source identity.
    fn source_id(&self) -> HfCredentialSourceId;

    /// Return the pinned source kind.
    fn source_kind(&self) -> HfCredentialSourceKind;

    /// Take a lease on the current material.
    async fn lease(&self) -> Result<HfCredentialLease, HfCredentialError>;

    /// Invalidate `seen` and re-read the pinned source under the host `Clock`.
    async fn refresh(
        &self,
        seen: CredentialGeneration,
    ) -> Result<HfRefreshOutcome, HfCredentialError>;

    /// Record that `generation` produced an authorized response, ending the
    /// current authentication episode.
    fn note_authorized(&self, generation: CredentialGeneration);

    /// Return how many refreshes have completed, for tests and diagnostics.
    fn refresh_count(&self) -> u64;
}

// ---------------------------------------------------------------------------
// Authority
// ---------------------------------------------------------------------------

struct AuthorityState {
    generation: CredentialGeneration,
    secret: Option<Rc<HfSecret>>,
    episode_attempts: u32,
    total_refreshes: u32,
    refresh_count: u64,
    is_refreshing: bool,
}

/// The built-in redacted, refresh-capable credential authority.
pub struct HfCredentialAuthority {
    descriptor: HfCredentialSourceDescriptor,
    source_id: HfCredentialSourceId,
    reader: Arc<dyn HfCredentialMaterialReader>,
    clock: Rc<dyn Clock>,
    executor: StreamingBlockingExecutor,
    settings: HfCredentialSettings,
    state: RefCell<AuthorityState>,
    // Bumped on every completed refresh, rotated or not, so a parked
    // same-generation caller wakes even when nothing changed. `watch` rather
    // than `Notify` because a subscription taken before the `is_refreshing`
    // read cannot miss the wakeup.
    epoch: watch::Sender<u64>,
}

/// Restore `is_refreshing` and wake parked callers even on an early return.
struct RefreshGuard<'authority> {
    authority: &'authority HfCredentialAuthority,
}

impl Drop for RefreshGuard<'_> {
    fn drop(&mut self) {
        self.authority.state.borrow_mut().is_refreshing = false;
        self.authority
            .epoch
            .send_modify(|epoch| *epoch = epoch.wrapping_add(1));
    }
}

impl HfCredentialAuthority {
    /// Resolve the pinned source and read its material once.
    ///
    /// Preparation fails closed: an authored source that resolves to nothing is
    /// a startup refusal, before any request, rather than a mid-run failure.
    pub async fn prepare(
        settings: HfCredentialSettings,
        reader: Arc<dyn HfCredentialMaterialReader>,
        clock: Rc<dyn Clock>,
        executor: StreamingBlockingExecutor,
    ) -> Result<Self, HfCredentialError> {
        settings.validate()?;
        let descriptor = settings.resolve_descriptor()?;
        let source_id = descriptor.source_id();
        let raw = read_material(&executor, &reader, &descriptor).await?;
        let secret = match raw {
            Some(raw) => Some(Rc::new(HfSecret::new(&raw)?)),
            None if descriptor.kind() == HfCredentialSourceKind::Anonymous => None,
            None => return Err(HfCredentialError::SourceUnavailable),
        };
        debug!(
            source_id = %source_id,
            kind = descriptor.kind().label(),
            "prepared Hugging Face credential authority"
        );
        Ok(Self {
            descriptor,
            source_id,
            reader,
            clock,
            executor,
            settings,
            state: RefCell::new(AuthorityState {
                generation: CredentialGeneration(0),
                secret,
                episode_attempts: 0,
                total_refreshes: 0,
                refresh_count: 0,
                is_refreshing: false,
            }),
            epoch: watch::channel(0).0,
        })
    }

    /// Borrow the pinned descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &HfCredentialSourceDescriptor {
        &self.descriptor
    }

    fn lease_from(&self, state: &AuthorityState) -> HfCredentialLease {
        HfCredentialLease {
            generation: state.generation,
            source_id: self.source_id,
            secret: state.secret.clone(),
        }
    }

    /// Exponential backoff with a ceiling; `attempt` is 1-based.
    fn backoff_ns(&self, attempt: u32) -> i64 {
        let shift = attempt.saturating_sub(1).min(62);
        let multiplier = 1_i64.checked_shl(shift).unwrap_or(i64::MAX);
        self.settings
            .refresh_backoff_base_ns
            .saturating_mul(multiplier)
            .min(self.settings.refresh_backoff_cap_ns)
    }
}

/// Read the pinned source on the bounded blocking owner.
///
/// The owner's typed output is only reachable through `Deref`, so the value is
/// cloned out; the reservation covers both the read and that clone.
async fn read_material(
    executor: &StreamingBlockingExecutor,
    reader: &Arc<dyn HfCredentialMaterialReader>,
    descriptor: &HfCredentialSourceDescriptor,
) -> Result<Option<String>, HfCredentialError> {
    let reader = Arc::clone(reader);
    let descriptor = descriptor.clone();
    let output = executor
        .run(
            BlockingWorkClass::Acquisition,
            BlockingWorkBudget {
                input_bytes: 0,
                output_bytes: MAX_TOKEN_BYTES * 2,
            },
            move |cancellation| {
                if cancellation.is_cancelled() {
                    return Err(BlockingWorkError::Cancelled);
                }
                Ok(reader.read(&descriptor))
            },
        )
        .await?;
    (*output).clone()
}

#[async_trait(?Send)]
impl HfCredentialProvider for HfCredentialAuthority {
    fn source_id(&self) -> HfCredentialSourceId {
        self.source_id
    }

    fn source_kind(&self) -> HfCredentialSourceKind {
        self.descriptor.kind()
    }

    async fn lease(&self) -> Result<HfCredentialLease, HfCredentialError> {
        let state = self.state.borrow();
        Ok(self.lease_from(&state))
    }

    async fn refresh(
        &self,
        seen: CredentialGeneration,
    ) -> Result<HfRefreshOutcome, HfCredentialError> {
        // Debounce: a stale reporter never triggers I/O, and a same-generation
        // reporter parks on the in-flight refresh instead of starting a second.
        loop {
            let mut receiver = self.epoch.subscribe();
            let must_wait = {
                let state = self.state.borrow();
                if state.generation != seen {
                    return Ok(HfRefreshOutcome::Superseded(self.lease_from(&state)));
                }
                state.is_refreshing
            };
            if !must_wait {
                break;
            }
            if receiver.changed().await.is_err() {
                return Err(HfCredentialError::Cancelled);
            }
        }

        let attempt = {
            let mut state = self.state.borrow_mut();
            if state.episode_attempts >= self.settings.max_refresh_attempts {
                return Err(HfCredentialError::RefreshExhausted);
            }
            if state.total_refreshes >= self.settings.max_total_refreshes {
                return Err(HfCredentialError::RefreshBudgetExhausted);
            }
            state.is_refreshing = true;
            state.episode_attempts = state.episode_attempts.saturating_add(1);
            state.total_refreshes = state.total_refreshes.saturating_add(1);
            state.episode_attempts
        };
        let _guard = RefreshGuard { authority: self };

        Rc::clone(&self.clock).sleep(self.backoff_ns(attempt)).await;
        let raw = read_material(&self.executor, &self.reader, &self.descriptor).await?;
        let refreshed = match raw {
            Some(raw) => Some(Rc::new(HfSecret::new(&raw)?)),
            None if self.descriptor.kind() == HfCredentialSourceKind::Anonymous => None,
            // Never a runtime downgrade to anonymous: a run authored for gated
            // access must not silently start reading a public object instead.
            None => return Err(HfCredentialError::SourceUnavailable),
        };

        let state = &mut *self.state.borrow_mut();
        state.refresh_count = state.refresh_count.saturating_add(1);
        let is_unchanged = match (state.secret.as_deref(), refreshed.as_deref()) {
            (Some(previous), Some(next)) => previous.digest() == next.digest(),
            (None, None) => true,
            _ => false,
        };
        if is_unchanged {
            debug!(
                source_id = %self.source_id,
                generation = state.generation.0,
                outcome = "unchanged",
                "Hugging Face credential refresh completed"
            );
            return Ok(HfRefreshOutcome::Unchanged(self.lease_from(state)));
        }
        state.secret = refreshed;
        state.generation = CredentialGeneration(state.generation.0.saturating_add(1));
        debug!(
            source_id = %self.source_id,
            generation = state.generation.0,
            outcome = "rotated",
            "Hugging Face credential refresh completed"
        );
        Ok(HfRefreshOutcome::Rotated(self.lease_from(state)))
    }

    fn note_authorized(&self, generation: CredentialGeneration) {
        let mut state = self.state.borrow_mut();
        if state.generation == generation {
            state.episode_attempts = 0;
        }
    }

    fn refresh_count(&self) -> u64 {
        self.state.borrow().refresh_count
    }
}

impl fmt::Debug for HfCredentialAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.borrow();
        formatter
            .debug_struct("HfCredentialAuthority")
            .field("source_id", &self.source_id)
            .field("kind", &self.descriptor.kind().label())
            .field("generation", &state.generation.0)
            .field(
                "credential",
                &if state.secret.is_some() {
                    "<redacted>"
                } else {
                    "absent"
                },
            )
            .field("refreshes", &state.refresh_count)
            .finish_non_exhaustive()
    }
}

/// Construct the built-in authority behind the injected provider contract.
///
/// This is the one call site a Hugging Face source uses, from
/// `PreparedStreamingDatasetSource::open`, which is worker-local and therefore
/// has the `Clock` that `StreamingSourcePrepareContext` does not carry.
pub async fn prepare_hf_credential_authority(
    settings: HfCredentialSettings,
    reader: Arc<dyn HfCredentialMaterialReader>,
    clock: Rc<dyn Clock>,
    executor: StreamingBlockingExecutor,
) -> Result<Rc<dyn HfCredentialProvider>, HfCredentialError> {
    let authority = HfCredentialAuthority::prepare(settings, reader, clock, executor).await?;
    Ok(Rc::new(authority))
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

/// Which proxy decision the host authority produced. A label only: the resolved
/// `ProxyConfig` carries a `Basic <base64>` `Proxy-Authorization` value and is
/// never rendered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfProxySelection {
    /// No proxy applies: loopback endpoint, a `NO_PROXY` hit, or neither opted in.
    Disabled,
    /// An explicitly authored proxy URL.
    Explicit,
    /// The ambient proxy environment, loopback-excluded.
    Environment,
}

impl HfProxySelection {
    /// Stable lowercase label for diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Explicit => "explicit",
            Self::Environment => "environment",
        }
    }
}

impl fmt::Display for HfProxySelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// Authored client settings for hub API and shard reads.
#[derive(Clone, Debug)]
pub struct HfHttpSettings {
    /// Hub endpoint. Its host is the only host the bearer is stamped for.
    pub endpoint: Url,
    /// Explicitly authored proxy URL; wins over the environment when present.
    pub proxy: Option<String>,
    /// Whether the ambient proxy environment applies. Downloads default to
    /// honoring it, matching the finite dataset fetch path.
    pub proxy_from_env: bool,
    /// Verify server certificate and hostname.
    pub ssl_verify: bool,
    /// Response-body ceiling, enforced per received chunk.
    pub max_response_body_bytes: u64,
    /// Connection-establishment deadline, in clock nanoseconds.
    pub connect_timeout_ns: i64,
    /// Send-plus-body deadline, in clock nanoseconds.
    pub request_timeout_ns: i64,
}

impl HfHttpSettings {
    /// Construct default settings for `endpoint`.
    #[must_use]
    pub fn new(endpoint: Url) -> Self {
        Self {
            endpoint,
            proxy: None,
            proxy_from_env: true,
            ssl_verify: true,
            max_response_body_bytes: DEFAULT_MAX_RESPONSE_BODY_BYTES,
            connect_timeout_ns: 10_000_000_000,
            request_timeout_ns: 300_000_000_000,
        }
    }
}

/// Resolved, socket-free client construction plan.
pub struct HfHttpClientFactory {
    endpoint_host: String,
    selection: HfProxySelection,
    proxy: Option<host_proxy::ProxyConfig>,
    settings: HfHttpSettings,
}

impl HfHttpClientFactory {
    /// Make the complete proxy and transport decision once, before any request.
    pub fn resolve(settings: &HfHttpSettings) -> Result<Self, HfCredentialError> {
        let endpoint_host = settings
            .endpoint
            .host_str()
            .map(str::to_ascii_lowercase)
            .ok_or(HfCredentialError::InvalidSettings)?;
        let is_bounded = settings.max_response_body_bytes > 0
            && settings.connect_timeout_ns > 0
            && settings.request_timeout_ns > 0;
        if !is_bounded {
            return Err(HfCredentialError::InvalidSettings);
        }
        // The host authority owns loopback exclusion, `NO_PROXY`, and
        // explicit-wins. This module must never re-derive that policy.
        let proxy = host_proxy::resolve(
            settings.proxy.as_deref(),
            settings.proxy_from_env,
            Some(&settings.endpoint),
        )
        .map_err(|_| HfCredentialError::InvalidSettings)?;
        let selection = match (&proxy, settings.proxy.is_some()) {
            (None, _) => HfProxySelection::Disabled,
            (Some(_), true) => HfProxySelection::Explicit,
            (Some(_), false) => HfProxySelection::Environment,
        };
        debug!(
            endpoint_host = %endpoint_host,
            proxy = %selection,
            "resolved Hugging Face streaming client"
        );
        Ok(Self {
            endpoint_host,
            selection,
            proxy,
            settings: settings.clone(),
        })
    }

    /// Return the resolved proxy decision label.
    #[must_use]
    pub const fn proxy_selection(&self) -> HfProxySelection {
        self.selection
    }

    /// Borrow the pinned endpoint host.
    #[must_use]
    pub fn endpoint_host(&self) -> &str {
        &self.endpoint_host
    }

    /// Build one worker-local client over the injected clock.
    #[must_use]
    pub fn build(&self, clock: Rc<dyn Clock>) -> HfHttpClient {
        let config = ClientConfig {
            proxy: self.proxy.clone(),
            ssl_verify: self.settings.ssl_verify,
            max_response_body_bytes: Some(self.settings.max_response_body_bytes),
            connect_timeout_ns: Some(self.settings.connect_timeout_ns),
            request_timeout_ns: Some(self.settings.request_timeout_ns),
            // This module owns the one retry authority, under the host Clock.
            max_connect_retries: 0,
            ..ClientConfig::default()
        };
        HfHttpClient {
            transport: HttpTransport::new(clock, config).with_user_agent("aiperf-streaming-hf/0"),
            endpoint_host: self.endpoint_host.clone(),
            selection: self.selection,
        }
    }
}

// ClientConfig derives Debug and embeds a ProxyConfig whose `auth_header` is a
// cleartext `Basic <base64>` value, so neither it nor the resolved proxy is
// rendered here.
impl fmt::Debug for HfHttpClientFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HfHttpClientFactory")
            .field("endpoint_host", &self.endpoint_host)
            .field("proxy", &self.selection.label())
            .finish_non_exhaustive()
    }
}

/// One authorized hub response plus whether the bearer was applied.
#[derive(Debug)]
pub struct HfAuthorizedResponse {
    /// Terminal transport record; the consumer owns reduction and classification.
    pub record: RequestRecord,
    /// Whether the request carried the bearer. False for a non-pinned host.
    pub bearer_stamped: bool,
}

/// Worker-local Hugging Face HTTP client.
pub struct HfHttpClient {
    transport: HttpTransport,
    endpoint_host: String,
    selection: HfProxySelection,
}

impl HfHttpClient {
    /// Issue one bounded GET, stamping the bearer only for the pinned host.
    ///
    /// A `302` to a CDN host deliberately loses the credential, matching the
    /// finite dataset fetch path; the consumer follows redirects and calls this
    /// again with the new URL.
    pub async fn authorized_get(
        &self,
        lease: &HfCredentialLease,
        url: &Url,
        extra_headers: &[(&str, &str)],
    ) -> Result<HfAuthorizedResponse, HfCredentialError> {
        let is_pinned_host = url
            .host_str()
            .is_some_and(|host| host.eq_ignore_ascii_case(&self.endpoint_host));
        let mut request = RequestConfig::new(url.as_str());
        for (name, value) in extra_headers {
            request
                .headers
                .insert((*name).to_owned(), (*value).to_owned());
        }
        let mut bearer_stamped = false;
        if is_pinned_host && let Some(bearer) = lease.bearer() {
            request
                .headers
                .insert("Authorization".to_owned(), format!("Bearer {bearer}"));
            bearer_stamped = true;
        }
        let record = self.transport.get(&request).await;
        Ok(HfAuthorizedResponse {
            record,
            bearer_stamped,
        })
    }

    /// Return the proxy decision this client was built with.
    #[must_use]
    pub const fn proxy_selection(&self) -> HfProxySelection {
        self.selection
    }

    /// Borrow the pinned endpoint host.
    #[must_use]
    pub fn endpoint_host(&self) -> &str {
        &self.endpoint_host
    }
}

impl fmt::Debug for HfHttpClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HfHttpClient")
            .field("endpoint_host", &self.endpoint_host)
            .field("proxy", &self.selection.label())
            .finish_non_exhaustive()
    }
}
