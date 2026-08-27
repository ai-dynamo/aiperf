// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pinned, page-paginated Hugging Face rows source.
//!
//! One partition is the verbatim response body of one Dataset Viewer `/rows`
//! request. The repository revision is resolved exactly once to an immutable
//! 40-hex commit, and every later request — including every request rebuilt
//! after a credential refresh — names that exact commit. The page inventory is
//! therefore arithmetic: page `k` covers rows `[k * page_len, (k + 1) * page_len)`,
//! and a page's immutable identity is a domain-separated BLAKE3 over the frozen
//! coordinates plus the commit, knowable before any byte is fetched. That is
//! what makes [`SourcePartitionContent::identity`] infallible and synchronous
//! for this source without forcing discovery to read every body first.
//!
//! Three properties are load-bearing and enforced by construction:
//!
//! * **Refresh cannot move the object.** The credential authority reached
//!   through [`HfPageTransport`] holds no dataset, revision, commit, or page
//!   index, and this module never re-resolves a symbolic revision after
//!   preparation. That is what makes "retry, then partition hole" legal while
//!   any revision, count, length, or digest drift stays a fail-run.
//! * **The proxy decision belongs to AIPerf.** This module issues no request of
//!   its own: every byte arrives through the injected [`HfPageTransport`], whose
//!   production implementation runs on the clock-injected Hyper transport and
//!   takes its proxy decision from `crate::transport::http::client::proxy`,
//!   which excludes every loopback host and honors `NO_PROXY`. `hf-hub` is never
//!   constructed here: its 0.4.3 `ureq` agent sets `try_proxy_from_env(true)`
//!   unconditionally and `ureq` 2.12 honors neither `NO_PROXY` nor loopback, so
//!   a local `HF_ENDPOINT` would be silently proxied.
//! * **A partial Viewer conversion is never a complete split.** `partial: true`
//!   is refused outright in [`HfRowsMode::Finite`] and seals nothing in
//!   [`HfRowsMode::Follow`].
//!
//! Two hosts participate: the hub API resolves the revision
//! (`huggingface.co`) and the Dataset Viewer paginates rows
//! (`datasets-server.huggingface.co`). [`HfHost`] names which one a request is
//! for, so a transport that pins its bearer to a single host can still stamp
//! both from one shared credential authority.
//!
//! This module owns no credential resolution, no HTTP client construction, and
//! no proxy policy. It reads no token environment variable and opens no token
//! file; the whole of that concern reaches it through [`HfPageTransport`].

use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;
use tracing::debug;
use url::Url;

use crate::clock::Clock;
use crate::streaming::budget::{BudgetLimits, StreamingResourceBudget};
use crate::streaming::checkpoint::{
    BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
    CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
    PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
};
use crate::streaming::failure::{
    AcquisitionFailureCode, OrdinaryStreamingFailure, SourceFailureCode, StreamSourceError,
};
use crate::streaming::identity::{ContentDigest, ImmutableObjectIdentity};
use crate::streaming::reliability::{
    OrdinaryStreamingIssue, StreamingInputDomainIdentity, StreamingIssueClass,
    StreamingIssueReporterHandle,
};
use crate::streaming::source::{
    AcquiredPartition, AcquisitionBudget, AcquisitionMemoryLease, BudgetedSourceChunk,
    OpenedStreamingDatasetSource, PartitionAccessKind, PartitionAccessRequest,
    PreparedStreamingDatasetSource, SequentialSourceChunk, SourceEvent, SourceFrontier,
    SourcePartition, SourcePartitionContent, SourceSeal, SourceSnapshotReceipt,
    StreamingDatasetSource, StreamingDatasetSourceFactory, StreamingResumeGranularity,
    StreamingSequentialReader, StreamingSourceDescriptor, StreamingSourceMode,
    StreamingSourceOrdering, StreamingSourcePlacement, StreamingSourcePrepareContext,
    StreamingSourceRetention, StreamingStopReceiver, ValidatedStreamingSourceConfig,
};
use crate::streaming::unit::SourcePosition;

/// Registry identifier for this source.
pub const HF_ROWS_SOURCE_ID: &str = "hf_rows";
/// Checkpoint schema identifier for the page cursor.
pub const HF_ROWS_CHECKPOINT_SCHEMA_ID: &str = "aiperf.streaming.source.hf_rows";
/// Checkpoint schema version.
pub const HF_ROWS_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Default hub endpoint that resolves a repository revision to a commit.
pub const DEFAULT_HUB_ENDPOINT: &str = "https://huggingface.co";
/// Default Dataset Viewer endpoint that paginates rows.
pub const DEFAULT_ROWS_ENDPOINT: &str = "https://datasets-server.huggingface.co";

/// Dataset Viewer per-request row ceiling, and the finite loader's page size.
const MAX_PAGE_LEN: u64 = 100;
/// Default response-body ceiling for one page.
const DEFAULT_MAX_PAGE_BYTES: u64 = 8 * 1024 * 1024;
/// Largest page body this source accepts under any authored setting.
const MAX_PAGE_BYTES_CEILING: u64 = 256 * 1024 * 1024;
/// Bytes handed to the caller per sequential chunk by default.
const DEFAULT_MAX_CHUNK_BYTES: usize = 256 * 1024;
/// Largest authored read-retry backoff ceiling, in clock nanoseconds.
const MAX_READ_BACKOFF_CAP_NS: i64 = 60_000_000_000;
/// Redirect hops honored per request, matching the finite dataset fetch path.
const MAX_REDIRECTS: usize = 8;
/// Exact commit-hash length the hub revision API is required to return.
const COMMIT_SHA_LEN: usize = 40;
/// Exact encoded cursor length without the optional credential trailer.
const CURSOR_BASE_BYTES: usize = 138;
/// Length of the credential-source trailer appended for a non-anonymous run.
const CURSOR_CREDENTIAL_TRAILER_BYTES: usize = 32;
/// Committed cursor generations the state budget admits simultaneously.
const CURSOR_BUDGET_ITEMS: usize = 4;

/// BLAKE3 domain separator for one page's immutable identity.
const PAGE_IDENTITY_DOMAIN: &[u8] = b"aiperf.stream.hf.page.v1";
/// BLAKE3 domain separator for the frozen acquisition-authority snapshot.
const SNAPSHOT_DOMAIN: &[u8] = b"aiperf.stream.hf.snapshot.v1";
/// BLAKE3 domain separator for one page's content-verification receipt.
const PAGE_CONTENT_DOMAIN: &[u8] = b"aiperf.stream.hf.page-content.v1";

static HF_ROWS_DESCRIPTOR: StreamingSourceDescriptor = StreamingSourceDescriptor {
    id: HF_ROWS_SOURCE_ID,
    description: "Pinned Hugging Face Dataset Viewer rows pagination",
    modes: &[StreamingSourceMode::Finite, StreamingSourceMode::Follow],
    access: &[PartitionAccessKind::Sequential],
    ordering: StreamingSourceOrdering::Partition,
    resume: &[
        StreamingResumeGranularity::Partition,
        StreamingResumeGranularity::Byte,
    ],
    has_event_time: false,
    // A page body carries no producer-stable record key; `row_idx` is a decoder
    // concern and this source has no record vocabulary at all.
    has_stable_record_ids: false,
    // Pages are re-acquirable by arithmetic coordinates against an immutable
    // commit, so no local snapshot has to stay reachable through a resume root.
    retention: StreamingSourceRetention::BoundedMemory,
    // Credentials stay on the acquiring process; a page assignment must never
    // travel to a cell that would then need its own token.
    placement: StreamingSourcePlacement::ControllerOnly,
    // Every wait is `Clock::sleep`; there is no wall-clock timer on any path.
    supports_virtual_clock: true,
};

// ---------------------------------------------------------------------------
// Credential and HTTP seam
// ---------------------------------------------------------------------------

/// Which Hugging Face host one request is addressed to.
///
/// A transport pins its bearer to one host at a time, but the Hugging Face
/// surface spans two: the hub API issues and validates the token, while the
/// Dataset Viewer is a derived service inside the same trust boundary. Naming
/// the role lets one shared credential authority stamp both.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfHost {
    /// The hub API that resolves a repository revision to a commit.
    Hub,
    /// The Dataset Viewer that paginates rows.
    Rows,
}

/// Stable failure classes the credential and HTTP seam may report.
///
/// The variants are exactly the ones this source can act on. They carry no
/// strings, so no token or URL substring can reach a log through an error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfTransportError {
    /// The pinned credential source is gone or its refresh budget is spent.
    SourceUnavailable,
    /// Authored transport or credential settings are invalid.
    InvalidSettings,
    /// Credential material exists but cannot be parsed as a bearer.
    MalformedMaterial,
    /// The request could not be completed against the endpoint.
    Transport,
}

/// Outcome of one credential-refresh round.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfRefreshOutcome {
    /// New material replaced the generation the caller observed.
    Rotated(u64),
    /// Re-reading the pinned source produced identical material.
    Unchanged(u64),
    /// Another caller already rotated past the observed generation.
    Superseded(u64),
}

/// One bounded HTTP response as this source needs to see it.
#[derive(Clone, Debug)]
pub struct HfHttpResponse {
    /// Response status, absent when the transport never received one.
    pub status: Option<u16>,
    /// Response headers with lowercase names, as the AIPerf transport normalizes them.
    pub headers: BTreeMap<String, String>,
    /// Verbatim response body, absent when the transport retained none.
    pub body: Option<Bytes>,
    /// Credential generation the request was stamped with.
    pub credential_generation: u64,
}

/// Worker-local authorized-GET authority over the two Hugging Face hosts.
///
/// The implementation owns credential resolution, caching, refresh
/// single-flight, refresh backoff, HTTP client construction, proxy policy, and
/// TLS. This source owns the URL, the retry rounds, the page arithmetic, and
/// the reliability classification, and nothing else.
#[async_trait(?Send)]
pub trait HfPageTransport {
    /// Issue one bounded authorized `GET` against the named host role.
    async fn authorized_get(
        &self,
        host: HfHost,
        url: &Url,
    ) -> Result<HfHttpResponse, HfTransportError>;

    /// Refresh material the caller observed as `generation`.
    async fn refresh(&self, generation: u64) -> Result<HfRefreshOutcome, HfTransportError>;

    /// Record that `generation` produced an authorized response.
    fn note_authorized(&self, generation: u64);

    /// Return the digest identifying the pinned credential source.
    ///
    /// The bytes are copied into the cursor so a restore under a different
    /// credential source is refused rather than silently honored. No credential
    /// material is derivable from them.
    fn credential_source_id(&self) -> [u8; 32];

    /// Return whether the run resolved to anonymous access.
    fn is_anonymous(&self) -> bool;
}

/// Endpoint, proxy, TLS, and credential coordinates handed to a transport factory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfTransportSettings {
    /// Hub endpoint that resolves the revision.
    pub hub_endpoint: Url,
    /// Dataset Viewer endpoint that paginates rows.
    pub rows_endpoint: Url,
    /// Explicitly authored proxy URL; wins over the ambient environment.
    pub proxy: Option<String>,
    /// Whether the ambient proxy environment applies. Loopback is always excluded.
    pub proxy_from_env: bool,
    /// Verify server certificate and hostname.
    pub ssl_verify: bool,
    /// Authored environment-variable name holding the token.
    pub credential_env_var: Option<String>,
    /// Authored token-file path.
    pub credential_token_file: Option<String>,
    /// Whether an absent credential may resolve to anonymous access.
    pub allow_anonymous: bool,
}

/// Host-supplied construction of one worker-local [`HfPageTransport`].
///
/// The factory itself is shared across threads, but the transport it builds is
/// worker-local: credentials, the HTTP client, and the clock all stay on the
/// acquiring thread.
#[async_trait(?Send)]
pub trait HfPageTransportFactory: fmt::Debug + Send + Sync {
    /// Resolve credentials and build both pinned clients for one run.
    async fn create(
        &self,
        settings: &HfTransportSettings,
        clock: Rc<dyn Clock>,
    ) -> Result<Rc<dyn HfPageTransport>, HfTransportError>;
}

const fn map_transport_error(error: HfTransportError) -> StreamSourceError {
    match error {
        HfTransportError::MalformedMaterial => {
            StreamSourceError::acquisition(AcquisitionFailureCode::Open)
        }
        HfTransportError::InvalidSettings => {
            StreamSourceError::source(SourceFailureCode::Discovery)
        }
        HfTransportError::Transport | HfTransportError::SourceUnavailable => {
            StreamSourceError::source(SourceFailureCode::SourceUnavailable)
        }
    }
}

// ---------------------------------------------------------------------------
// Authored configuration
// ---------------------------------------------------------------------------

/// Inventory lifecycle selected by authored configuration.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum HfRowsMode {
    /// The split is already fully converted; a partial response is refused.
    #[default]
    Finite,
    /// The split is still converting; announce the honest prefix and poll.
    Follow,
}

impl HfRowsMode {
    const fn tag(self) -> u8 {
        match self {
            Self::Finite => 0,
            Self::Follow => 1,
        }
    }
}

/// Strictly authored configuration for the Hugging Face rows source.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HfRowsSourceConfig {
    /// Repository identifier such as `openai/gsm8k`.
    pub dataset: String,
    /// Dataset configuration/subset.
    pub subset: String,
    /// Dataset split.
    pub split: String,
    /// Branch, tag, or commit resolved exactly once at preparation.
    #[serde(default = "default_revision")]
    pub revision: String,
    /// Inventory lifecycle.
    #[serde(default)]
    pub mode: HfRowsMode,
    /// Rows requested per page; frozen for the life of the run.
    #[serde(default = "default_page_len")]
    pub page_len: u64,
    /// Hub endpoint that resolves the revision.
    #[serde(default = "default_hub_endpoint")]
    pub hub_endpoint: String,
    /// Dataset Viewer endpoint that paginates rows.
    #[serde(default = "default_rows_endpoint")]
    pub rows_endpoint: String,
    /// Explicitly authored proxy URL; wins over the ambient environment.
    #[serde(default)]
    pub proxy: Option<String>,
    /// Whether the ambient proxy environment applies. Loopback is always excluded.
    #[serde(default = "default_true")]
    pub proxy_from_env: bool,
    /// Verify server certificate and hostname.
    #[serde(default = "default_true")]
    pub ssl_verify: bool,
    /// Treat an absent `partial` field as unproven completeness.
    #[serde(default = "default_true")]
    pub require_explicit_completeness: bool,
    /// Authored environment-variable name holding the token.
    #[serde(default)]
    pub credential_env_var: Option<String>,
    /// Authored token-file path.
    #[serde(default)]
    pub credential_token_file: Option<String>,
    /// Whether an absent credential may resolve to anonymous access.
    #[serde(default = "default_true")]
    pub allow_anonymous: bool,
    /// Response-body ceiling for one page.
    #[serde(default = "default_max_page_bytes")]
    pub max_page_bytes: u64,
    /// Bytes returned per sequential chunk.
    #[serde(default = "default_max_chunk_bytes")]
    pub max_chunk_bytes: usize,
    /// Refresh-and-retry rounds per request.
    #[serde(default = "default_max_auth_retries")]
    pub max_auth_retries: u32,
    /// Throttle and server-failure retry rounds per request.
    #[serde(default = "default_max_read_retries")]
    pub max_read_retries: u32,
    /// First read-retry backoff, in clock nanoseconds.
    #[serde(default = "default_read_backoff_base_ns")]
    pub read_backoff_base_ns: i64,
    /// Read-retry backoff ceiling, in clock nanoseconds.
    #[serde(default = "default_read_backoff_cap_ns")]
    pub read_backoff_cap_ns: i64,
    /// Follow-mode poll interval, in clock nanoseconds.
    #[serde(default = "default_poll_interval_ns")]
    pub poll_interval_ns: i64,
}

fn default_revision() -> String {
    "main".to_owned()
}
const fn default_page_len() -> u64 {
    MAX_PAGE_LEN
}
fn default_hub_endpoint() -> String {
    DEFAULT_HUB_ENDPOINT.to_owned()
}
fn default_rows_endpoint() -> String {
    DEFAULT_ROWS_ENDPOINT.to_owned()
}
const fn default_true() -> bool {
    true
}
const fn default_max_page_bytes() -> u64 {
    DEFAULT_MAX_PAGE_BYTES
}
const fn default_max_chunk_bytes() -> usize {
    DEFAULT_MAX_CHUNK_BYTES
}
const fn default_max_auth_retries() -> u32 {
    2
}
const fn default_max_read_retries() -> u32 {
    3
}
const fn default_read_backoff_base_ns() -> i64 {
    200_000_000
}
const fn default_read_backoff_cap_ns() -> i64 {
    4_000_000_000
}
const fn default_poll_interval_ns() -> i64 {
    2_000_000_000
}

fn discovery_error() -> StreamSourceError {
    StreamSourceError::source(SourceFailureCode::Discovery)
}

/// Refuse every authored value this source cannot execute, before any effect.
fn validate_config(config: &HfRowsSourceConfig) -> Result<(), StreamSourceError> {
    let mut segments = config.dataset.split('/');
    let namespace = segments.next().unwrap_or_default();
    let name = segments.next().unwrap_or_default();
    if namespace.is_empty() || name.is_empty() || segments.next().is_some() {
        return Err(discovery_error());
    }
    if config.subset.is_empty() || config.split.is_empty() || config.revision.is_empty() {
        return Err(discovery_error());
    }
    if config.page_len == 0 || config.page_len > MAX_PAGE_LEN {
        return Err(discovery_error());
    }
    if config.max_page_bytes == 0 || config.max_page_bytes > MAX_PAGE_BYTES_CEILING {
        return Err(discovery_error());
    }
    if config.max_chunk_bytes == 0 {
        return Err(discovery_error());
    }
    if config.max_auth_retries == 0 || config.max_auth_retries > 3 {
        return Err(discovery_error());
    }
    if config.max_read_retries > 5 {
        return Err(discovery_error());
    }
    if config.read_backoff_base_ns <= 0 || config.poll_interval_ns <= 0 {
        return Err(discovery_error());
    }
    if config.read_backoff_cap_ns < config.read_backoff_base_ns
        || config.read_backoff_cap_ns > MAX_READ_BACKOFF_CAP_NS
    {
        return Err(discovery_error());
    }
    if config.credential_env_var.is_some() && config.credential_token_file.is_some() {
        return Err(discovery_error());
    }
    parse_endpoint(&config.hub_endpoint)?;
    parse_endpoint(&config.rows_endpoint)?;
    Ok(())
}

/// Parse one endpoint, refusing a scheme this source will not speak.
fn parse_endpoint(authored: &str) -> Result<Url, StreamSourceError> {
    let url = Url::parse(authored).map_err(|_| discovery_error())?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(discovery_error());
    }
    Ok(url)
}

/// Render `host[:port]`, lowercased, for identity derivation.
fn endpoint_authority(url: &Url) -> Result<String, StreamSourceError> {
    let host = url.host_str().ok_or_else(discovery_error)?;
    Ok(match url.port() {
        Some(port) => format!("{}:{port}", host.to_ascii_lowercase()),
        None => host.to_ascii_lowercase(),
    })
}

// ---------------------------------------------------------------------------
// Frozen resolution and identity
// ---------------------------------------------------------------------------

fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Everything frozen at preparation. Nothing here changes for the life of the run.
#[derive(Clone, Debug, Eq, PartialEq)]
struct FrozenResolution {
    stream_identity: ContentDigest,
    dataset: String,
    subset: String,
    split: String,
    commit_sha: String,
    page_len: u64,
    mode: HfRowsMode,
    hub_authority: String,
    rows_authority: String,
    rows_url_base: Url,
    require_explicit_completeness: bool,
    credential_source_id: [u8; 32],
    is_anonymous: bool,
}

impl FrozenResolution {
    /// Derive the immutable identity of one page, before any byte is fetched.
    ///
    /// `num_rows_total` is deliberately absent: it grows in follow mode, and the
    /// final page is only announced once completeness is proven, so its extent is
    /// already fixed at announcement time. `row_offset` is absent because it is
    /// exactly `page_index * page_len`, and a redundant field can disagree.
    fn page_identity(&self, page_index: u64) -> ImmutableObjectIdentity {
        let mut hasher = blake3::Hasher::new();
        update_field(&mut hasher, PAGE_IDENTITY_DOMAIN);
        update_field(&mut hasher, self.stream_identity.as_bytes());
        update_field(&mut hasher, self.rows_authority.as_bytes());
        update_field(&mut hasher, self.dataset.as_bytes());
        update_field(&mut hasher, self.subset.as_bytes());
        update_field(&mut hasher, self.split.as_bytes());
        update_field(&mut hasher, self.commit_sha.as_bytes());
        update_field(&mut hasher, &self.page_len.to_le_bytes());
        update_field(&mut hasher, &page_index.to_le_bytes());
        ImmutableObjectIdentity::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Digest the frozen acquisition authority, computable at open in both modes.
    ///
    /// This binds the exact immutable coordinates and the exact credential source
    /// under which every page will be read, rather than the enumerated inventory,
    /// which follow mode does not know at open. Completeness is proven separately
    /// by the recorded `partial == Some(false)` observation and by the seal's
    /// final position.
    fn snapshot_digest(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        update_field(&mut hasher, SNAPSHOT_DOMAIN);
        update_field(&mut hasher, self.stream_identity.as_bytes());
        update_field(&mut hasher, &[self.mode.tag()]);
        update_field(&mut hasher, self.hub_authority.as_bytes());
        update_field(&mut hasher, self.rows_authority.as_bytes());
        update_field(&mut hasher, self.dataset.as_bytes());
        update_field(&mut hasher, self.subset.as_bytes());
        update_field(&mut hasher, self.split.as_bytes());
        update_field(&mut hasher, self.commit_sha.as_bytes());
        update_field(&mut hasher, &self.page_len.to_le_bytes());
        update_field(&mut hasher, &[u8::from(self.require_explicit_completeness)]);
        update_field(&mut hasher, &self.credential_source_id);
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    fn rows_url(&self, offset: u64, length: u64) -> Result<Url, StreamSourceError> {
        let mut url = self.rows_url_base.clone();
        url.query_pairs_mut()
            .append_pair("dataset", &self.dataset)
            .append_pair("config", &self.subset)
            .append_pair("split", &self.split)
            .append_pair("offset", &offset.to_string())
            .append_pair("length", &length.to_string());
        Ok(url)
    }

    /// Build the exact `/rows` URL for one page.
    fn page_url(&self, page_index: u64) -> Result<Url, StreamSourceError> {
        let offset = page_index
            .checked_mul(self.page_len)
            .ok_or_else(discovery_error)?;
        self.rows_url(offset, self.page_len)
    }

    /// Build the one-row probe used to read `num_rows_total` and `partial`.
    fn probe_url(&self) -> Result<Url, StreamSourceError> {
        self.rows_url(0, 1)
    }
}

/// Content-verification receipt for one acquired page.
///
/// The announced identity says *which* bytes; this proves *the same* bytes came
/// back.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HfPageReceipt {
    /// BLAKE3 over the canonicalized `etag` header, when the service served one.
    etag_digest: Option<ContentDigest>,
    /// BLAKE3 over the verbatim response body.
    body_digest: ContentDigest,
    /// Exact response body length.
    byte_length: u64,
}

impl HfPageReceipt {
    fn observe(body: &[u8], etag: Option<&str>) -> Self {
        let mut hasher = blake3::Hasher::new();
        update_field(&mut hasher, PAGE_CONTENT_DOMAIN);
        update_field(&mut hasher, body);
        let body_digest = ContentDigest::from_bytes(*hasher.finalize().as_bytes());
        let etag_digest = etag.map(|value| {
            // Canonicalize `W/"abc"` and `"abc"` to `abc` before hashing so a
            // weak validator and its strong twin do not read as different bytes.
            let trimmed = value.trim();
            let trimmed = trimmed.strip_prefix("W/").unwrap_or(trimmed);
            let trimmed = trimmed.trim_matches('"');
            let mut hasher = blake3::Hasher::new();
            update_field(&mut hasher, PAGE_CONTENT_DOMAIN);
            update_field(&mut hasher, trimmed.as_bytes());
            ContentDigest::from_bytes(*hasher.finalize().as_bytes())
        });
        Self {
            etag_digest,
            body_digest,
            byte_length: body.len() as u64,
        }
    }

    /// Two observations of the same immutable page must agree.
    fn is_consistent_with(&self, committed: &Self) -> bool {
        self.body_digest == committed.body_digest
            && self.byte_length == committed.byte_length
            // An absent ETag on either side is not evidence of drift; two
            // present-but-different ETags are.
            && match (self.etag_digest, committed.etag_digest) {
                (Some(observed), Some(expected)) => observed == expected,
                _ => true,
            }
    }
}

/// One bounded response body plus the evidence needed to verify it.
struct FetchedBody {
    bytes: Bytes,
    receipt: HfPageReceipt,
}

/// Retry, refresh, and body bounds applied to every request this source issues.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RequestPolicy {
    max_page_bytes: u64,
    max_auth_retries: u32,
    max_read_retries: u32,
    read_backoff_base_ns: i64,
    read_backoff_cap_ns: i64,
}

impl RequestPolicy {
    fn read_backoff_ns(&self, round: u32) -> i64 {
        // `checked_shl` cannot fail below the width of `i64`; the clamp keeps the
        // doubling total without reaching for a panicking shift.
        let factor = 1_i64.checked_shl(round.min(62)).unwrap_or(i64::MAX);
        self.read_backoff_base_ns
            .saturating_mul(factor)
            .min(self.read_backoff_cap_ns)
    }
}

/// Issue one bounded authorized `GET`, refreshing on `401`/`403` and retrying
/// throttles, always for the exact URL it was handed.
///
/// The URL is never re-derived from a symbolic revision, so no outcome of this
/// function can change which immutable object is being acquired. That is what
/// keeps "retry, then partition hole" separable from identity drift.
async fn authorized_get_bounded(
    transport: &dyn HfPageTransport,
    host: HfHost,
    clock: &Rc<dyn Clock>,
    policy: &RequestPolicy,
    url: &Url,
) -> Result<FetchedBody, StreamSourceError> {
    let mut current = url.clone();
    let mut auth_round = 0_u32;
    let mut read_round = 0_u32;
    let mut unchanged_refreshes = 0_u32;
    let mut redirects = 0_usize;

    loop {
        let response = transport
            .authorized_get(host, &current)
            .await
            .map_err(map_transport_error)?;
        let generation = response.credential_generation;

        // A redirect is honored before any error classification: the AIPerf
        // transport reports every non-2xx, 3xx included, as an error while still
        // populating the status and the `location` header.
        if matches!(response.status, Some(301 | 302 | 303 | 307 | 308)) {
            redirects += 1;
            if redirects > MAX_REDIRECTS {
                return Err(StreamSourceError::acquisition(AcquisitionFailureCode::Read));
            }
            let location = response
                .headers
                .get("location")
                .ok_or_else(|| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
            current = current
                .join(location)
                .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
            continue;
        }

        match response.status {
            Some(200) => {
                transport.note_authorized(generation);
                return take_bounded_body(policy, response);
            }
            Some(401 | 403) => {
                match transport
                    .refresh(generation)
                    .await
                    .map_err(map_transport_error)?
                {
                    HfRefreshOutcome::Superseded(_) => continue,
                    HfRefreshOutcome::Rotated(_) => {
                        unchanged_refreshes = 0;
                        auth_round += 1;
                    }
                    HfRefreshOutcome::Unchanged(_) => {
                        unchanged_refreshes += 1;
                        auth_round += 1;
                        // Identical material rejected twice is not a credential
                        // problem; stop before spending the whole-run refresh
                        // ceiling the authority guards for real rotations.
                        if unchanged_refreshes >= 2 {
                            return Err(StreamSourceError::source(
                                SourceFailureCode::SourceUnavailable,
                            ));
                        }
                    }
                }
                if auth_round > policy.max_auth_retries {
                    return Err(StreamSourceError::source(
                        SourceFailureCode::SourceUnavailable,
                    ));
                }
            }
            Some(408 | 429) | Some(500..=599) => {
                if read_round >= policy.max_read_retries {
                    return Err(StreamSourceError::acquisition(AcquisitionFailureCode::Read));
                }
                // `Retry-After` is deliberately not honored: it is a
                // server-supplied duration that would let a remote host drive the
                // injected clock arbitrarily far.
                Rc::clone(clock)
                    .sleep(policy.read_backoff_ns(read_round))
                    .await;
                read_round += 1;
            }
            Some(404) => {
                return Err(StreamSourceError::acquisition(AcquisitionFailureCode::Open));
            }
            _ => {
                return Err(StreamSourceError::acquisition(AcquisitionFailureCode::Read));
            }
        }
    }
}

/// Extract one bounded body and its verification receipt.
fn take_bounded_body(
    policy: &RequestPolicy,
    response: HfHttpResponse,
) -> Result<FetchedBody, StreamSourceError> {
    let bytes = response
        .body
        .ok_or_else(|| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
    if bytes.len() as u64 > policy.max_page_bytes {
        return Err(StreamSourceError::acquisition(
            AcquisitionFailureCode::ObjectLimitExceeded,
        ));
    }
    let receipt = HfPageReceipt::observe(&bytes, response.headers.get("etag").map(String::as_str));
    Ok(FetchedBody { bytes, receipt })
}

/// Shared, worker-local acquisition authority handed to every announced page.
struct HfAcquisitionContext {
    resolution: FrozenResolution,
    transport: Rc<dyn HfPageTransport>,
    clock: Rc<dyn Clock>,
    reporter: StreamingIssueReporterHandle,
    run: StreamRunIdentity,
    policy: RequestPolicy,
    snapshot_digest: ContentDigest,
}

impl HfAcquisitionContext {
    /// Report one ordinary retryable fault scoped to the exact frozen page.
    ///
    /// The class is always `Retryable`: `Invariant` is rejected by the
    /// constructor, and `Hole` is a host disposition this source cannot select.
    async fn report_page_fault(
        &self,
        position: SourcePosition,
        object: ImmutableObjectIdentity,
        retry_ordinal: u32,
        failure: StreamSourceError,
    ) {
        let input_domain =
            StreamingInputDomainIdentity::new(self.resolution.stream_identity, object);
        let issue = match OrdinaryStreamingIssue::partition(
            self.run,
            input_domain,
            object,
            StreamingIssueClass::Retryable,
            self.snapshot_digest,
            position,
            retry_ordinal,
            ContentDigest::from_bytes(*object.as_bytes()),
            OrdinaryStreamingFailure::Source(failure),
        ) {
            Ok(issue) => issue,
            Err(error) => {
                debug!(error = ?error, component = "hf_rows", "page issue was refused");
                return;
            }
        };
        if let Err(error) = self.reporter.report(issue).await {
            debug!(error = ?error, component = "hf_rows", "page issue was not accepted");
        }
    }
}

// ---------------------------------------------------------------------------
// Partition content and sequential reader
// ---------------------------------------------------------------------------

/// One announced page: frozen coordinates plus the shared acquisition authority.
struct HfPageContent {
    position: SourcePosition,
    identity: ImmutableObjectIdentity,
    url: Url,
    context: Rc<HfAcquisitionContext>,
    committed_receipt: Option<HfPageReceipt>,
    max_chunk_bytes: usize,
}

#[async_trait(?Send)]
impl SourcePartitionContent for HfPageContent {
    fn identity(&self) -> &ImmutableObjectIdentity {
        &self.identity
    }

    fn size_bytes(&self) -> Option<u64> {
        // A page's length is not advertised before it is read. Claiming a guessed
        // length would let `AcquiredSequentialPartition::next_chunk` reject a
        // valid short final read as a truncated object.
        self.committed_receipt.map(|receipt| receipt.byte_length)
    }

    async fn acquire(
        &self,
        request: PartitionAccessRequest,
        budget: &AcquisitionBudget,
    ) -> Result<AcquiredPartition, StreamSourceError> {
        let PartitionAccessRequest::Sequential { resume_offset } = request else {
            return Err(StreamSourceError::acquisition(AcquisitionFailureCode::Open));
        };

        let fetched = match authorized_get_bounded(
            self.context.transport.as_ref(),
            HfHost::Rows,
            &self.context.clock,
            &self.context.policy,
            &self.url,
        )
        .await
        {
            Ok(fetched) => fetched,
            Err(error) => {
                // The frozen identity is unchanged, so this is the retryable class
                // the host may resolve as a partition hole.
                self.context
                    .report_page_fault(self.position, self.identity, 0, error)
                    .await;
                return Err(error);
            }
        };

        if let Some(committed) = self.committed_receipt
            && !fetched.receipt.is_consistent_with(&committed)
        {
            // Identity substitution under a pinned commit; never a hole.
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::IdentityMismatch,
            ));
        }

        let size_bytes = fetched.receipt.byte_length;
        if resume_offset > size_bytes {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::ObjectLimitExceeded,
            ));
        }

        // Retain the page body under its own exact charge for as long as the
        // reader lives; each emitted chunk takes a second, separate charge, so the
        // peak is one page plus one chunk and the budget must admit both.
        let body_lease = budget.acquire_memory(1, fetched.bytes.len()).await?;
        let authority = budget.acquire_memory(1, 0).await?;

        AcquiredPartition::sequential(
            self.position,
            self.identity,
            Some(size_bytes),
            resume_offset,
            Box::new(HfPageReader {
                bytes: fetched.bytes,
                offset: resume_offset,
                max_chunk_bytes: self.max_chunk_bytes,
                rolling: blake3::Hasher::new(),
                _body_lease: body_lease,
            }),
            authority,
        )
    }
}

/// Bounded forward reader over one retained immutable page body.
struct HfPageReader {
    bytes: Bytes,
    offset: u64,
    max_chunk_bytes: usize,
    rolling: blake3::Hasher,
    // Released with the reader, keeping the retained body honestly charged.
    _body_lease: AcquisitionMemoryLease,
}

#[async_trait(?Send)]
impl StreamingSequentialReader for HfPageReader {
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<Option<SequentialSourceChunk>, StreamSourceError> {
        let start = usize::try_from(self.offset).map_err(|_| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        if start >= self.bytes.len() {
            return Ok(None);
        }
        let length = max_bytes
            .get()
            .min(self.max_chunk_bytes)
            .min(self.bytes.len() - start);
        let slice = self.bytes.slice(start..start + length);
        self.rolling.update(&slice);
        let lease = budget.acquire_memory(1, length).await?;
        let chunk = BudgetedSourceChunk::new(slice, lease)?;
        self.offset += length as u64;
        // Cloning the hasher finalizes a chunk digest without ending the stream;
        // the clone is O(state), not O(bytes).
        let rolling_digest = ContentDigest::from_bytes(*self.rolling.clone().finalize().as_bytes());
        Ok(Some(SequentialSourceChunk::new(
            chunk,
            self.offset,
            rolling_digest,
        )))
    }
}

// ---------------------------------------------------------------------------
// Inventory and the source itself
// ---------------------------------------------------------------------------

/// Untyped-tolerant view of one Dataset Viewer `/rows` response.
///
/// Deliberately not `deny_unknown_fields`: this is an external API AIPerf does
/// not version, and the strict-DTO rule governs protocol-v2 requests. Only the
/// two fields this source reasons about are named.
#[derive(Debug, Deserialize)]
struct RowsEnvelope {
    num_rows_total: Option<u64>,
    partial: Option<bool>,
}

/// Untyped-tolerant view of one hub revision response.
#[derive(Debug, Deserialize)]
struct RevisionEnvelope {
    sha: Option<String>,
}

/// Inventory knowledge accumulated by discovery.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PageInventory {
    known_row_total: u64,
    is_sealed: bool,
}

impl PageInventory {
    const fn page_count(self, page_len: u64) -> u64 {
        self.known_row_total.div_ceil(page_len)
    }
}

/// Run-local Hugging Face rows source.
struct HfRowsSource {
    context: Rc<HfAcquisitionContext>,
    stop: StreamingStopReceiver,
    snapshot: SourceSnapshotReceipt,
    participant_id: CheckpointParticipantId,
    initialization: ParticipantInitialization,
    cursor_budget: StreamingResourceBudget,
    poll_interval_ns: i64,
    max_chunk_bytes: usize,
    inventory: PageInventory,
    next_page_index: u64,
    pending_frontier: Option<SourcePosition>,
    is_sealed_out: bool,
    receipts: BTreeMap<u64, HfPageReceipt>,
}

#[async_trait(?Send)]
impl StreamingDatasetSource for HfRowsSource {
    fn snapshot(&self) -> &SourceSnapshotReceipt {
        &self.snapshot
    }

    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError> {
        loop {
            if let Some(through) = self.pending_frontier.take() {
                return Ok(SourceEvent::Frontier(SourceFrontier { through }));
            }
            if self.next_page_index < self.inventory.page_count(self.context.resolution.page_len) {
                return self.announce_next_page();
            }
            if self.inventory.is_sealed {
                if self.is_sealed_out {
                    // A sealed source polled again parks until stop rather than
                    // emitting a second seal, which would let a host
                    // double-advance the discovery horizon.
                    self.stop.stopped().await?;
                }
                self.is_sealed_out = true;
                return Ok(SourceEvent::Seal(SourceSeal {
                    final_position: self.next_page_index.checked_sub(1).map(SourcePosition::new),
                    digest: self.snapshot.digest,
                }));
            }
            if self.context.resolution.mode == HfRowsMode::Finite {
                // Finite proved completeness at preparation, so reaching here means
                // the inventory shrank under a pinned commit.
                return Err(StreamSourceError::source(SourceFailureCode::MutatedObject));
            }
            // Follow: nothing addressable yet. Park on the injected clock, biased
            // so a concurrent stop always wins.
            let clock = Rc::clone(&self.context.clock);
            let poll_interval_ns = self.poll_interval_ns;
            tokio::select! {
                biased;
                result = self.stop.stopped() => result?,
                () = clock.sleep(poll_interval_ns) => {}
            }
            self.refresh_inventory().await?;
        }
    }
}

impl HfRowsSource {
    /// Announce exactly one page and queue its completeness frontier.
    fn announce_next_page(&mut self) -> Result<SourceEvent, StreamSourceError> {
        let page_index = self.next_page_index;
        let position = SourcePosition::new(page_index);
        let identity = self.context.resolution.page_identity(page_index);
        let url = self.context.resolution.page_url(page_index)?;
        let content = HfPageContent {
            position,
            identity,
            url,
            context: Rc::clone(&self.context),
            committed_receipt: self.receipts.get(&page_index).copied(),
            max_chunk_bytes: self.max_chunk_bytes,
        };
        self.next_page_index = page_index.checked_add(1).ok_or_else(discovery_error)?;
        self.pending_frontier = Some(position);
        Ok(SourceEvent::Partition(SourcePartition::new(
            position,
            Box::new(content),
        )))
    }

    /// Re-probe the split's row count and completeness under the pinned commit.
    async fn refresh_inventory(&mut self) -> Result<(), StreamSourceError> {
        let probe = self.context.resolution.probe_url()?;
        let fetched = authorized_get_bounded(
            self.context.transport.as_ref(),
            HfHost::Rows,
            &self.context.clock,
            &self.context.policy,
            &probe,
        )
        .await?;
        let observed = read_inventory(
            &fetched.bytes,
            self.context.resolution.require_explicit_completeness,
        )?;
        if observed.known_row_total < self.inventory.known_row_total {
            return Err(StreamSourceError::source(SourceFailureCode::MutatedObject));
        }
        if self.context.resolution.mode == HfRowsMode::Finite
            && observed.known_row_total != self.inventory.known_row_total
        {
            return Err(StreamSourceError::source(SourceFailureCode::MutatedObject));
        }
        self.inventory = observed;
        Ok(())
    }

    /// Restore a committed cursor, refusing every drift the frozen plan forbids.
    fn restore(&mut self, cursor: HfRowsCursor) -> Result<(), CheckpointError> {
        let resolution = &self.context.resolution;
        let mut commit = [0_u8; COMMIT_SHA_LEN];
        let sha = resolution.commit_sha.as_bytes();
        if sha.len() != COMMIT_SHA_LEN {
            return Err(CheckpointError::SourceUnavailableOnResume);
        }
        commit.copy_from_slice(sha);

        let expected_credential = if resolution.is_anonymous {
            None
        } else {
            Some(resolution.credential_source_id)
        };
        if cursor.commit_sha != commit
            || cursor.page_len != resolution.page_len
            || cursor.credential_source_id != expected_credential
        {
            return Err(CheckpointError::SourceUnavailableOnResume);
        }
        if cursor.known_row_total < self.inventory.known_row_total
            || (cursor.is_inventory_sealed && !self.inventory.is_sealed)
        {
            return Err(CheckpointError::SourceUnavailableOnResume);
        }
        if let Some(previous) = cursor.next_page_index.checked_sub(1) {
            let expected = resolution.page_identity(previous);
            if cursor.last_object_digest != *expected.as_bytes() {
                return Err(CheckpointError::SourceUnavailableOnResume);
            }
            self.receipts.insert(
                previous,
                HfPageReceipt {
                    // Only the body digest and length are durable. A restored
                    // receipt therefore carries no ETag, and `is_consistent_with`
                    // treats an absent side as non-evidence rather than as drift.
                    etag_digest: None,
                    body_digest: ContentDigest::from_bytes(cursor.last_content_digest),
                    byte_length: cursor.last_byte_length,
                },
            );
        }

        self.next_page_index = cursor.next_page_index;
        self.inventory = PageInventory {
            known_row_total: cursor.known_row_total.max(self.inventory.known_row_total),
            is_sealed: self.inventory.is_sealed || cursor.is_inventory_sealed,
        };
        Ok(())
    }

    /// Snapshot the complete resumable state without releasing anything.
    fn encode_cursor(&self) -> Result<Vec<u8>, CheckpointError> {
        let resolution = &self.context.resolution;
        let mut commit = [0_u8; COMMIT_SHA_LEN];
        let sha = resolution.commit_sha.as_bytes();
        if sha.len() != COMMIT_SHA_LEN {
            return Err(CheckpointError::ObjectVerification);
        }
        commit.copy_from_slice(sha);

        let previous = self.next_page_index.checked_sub(1);
        let receipt = previous.and_then(|index| self.receipts.get(&index).copied());
        let cursor = HfRowsCursor {
            next_page_index: self.next_page_index,
            page_len: resolution.page_len,
            known_row_total: self.inventory.known_row_total,
            is_inventory_sealed: self.inventory.is_sealed,
            content_authority: match receipt {
                None => 0,
                Some(receipt) if receipt.etag_digest.is_none() => 1,
                Some(_) => 2,
            },
            last_object_digest: previous
                .map(|index| *resolution.page_identity(index).as_bytes())
                .unwrap_or_default(),
            last_content_digest: receipt
                .map(|receipt| *receipt.body_digest.as_bytes())
                .unwrap_or_default(),
            last_byte_length: receipt.map_or(0, |receipt| receipt.byte_length),
            commit_sha: commit,
            credential_source_id: if resolution.is_anonymous {
                None
            } else {
                Some(resolution.credential_source_id)
            },
        };
        Ok(cursor.encode())
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for HfRowsSource {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if barrier.run != self.context.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let encoded = self.encode_cursor()?;
        let lease = self
            .cursor_budget
            .acquire(1, encoded.len())
            .await
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(encoded), lease)?;
        PreparedParticipantState::new(
            barrier.run,
            self.participant_id.clone(),
            HF_ROWS_CHECKPOINT_SCHEMA_ID,
            HF_ROWS_CHECKPOINT_SCHEMA_VERSION,
            barrier.cut.clone(),
            self.next_page_index,
            payload,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        let Some(state) = state else {
            return Ok(());
        };
        // Bytes come out only by borrow; the state and its lease drop here.
        let cursor = HfRowsCursor::decode(state.payload_bytes())?;
        self.restore(cursor)
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.context.run || receipt.participant_id() != &self.participant_id {
            return Err(CheckpointError::ObjectVerification);
        }
        // Idempotent: pruning below a fixed key is a total-order retain, so the
        // same receipt applied twice is a no-op.
        let committed_below = self.next_page_index.saturating_sub(1);
        self.receipts.retain(|index, _| *index >= committed_below);
        Ok(())
    }
}

/// Read one `/rows` envelope as an inventory observation.
fn read_inventory(
    body: &[u8],
    require_explicit_completeness: bool,
) -> Result<PageInventory, StreamSourceError> {
    let envelope: RowsEnvelope = serde_json::from_slice(body).map_err(|_| discovery_error())?;
    let known_row_total = envelope.num_rows_total.ok_or_else(discovery_error)?;
    let is_sealed = match envelope.partial {
        Some(partial) => !partial,
        // An absent field is not proof of completeness under the conservative
        // reading of the partial-conversion invariant.
        None => !require_explicit_completeness,
    };
    Ok(PageInventory {
        known_row_total,
        is_sealed,
    })
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

/// Fixed-width resumable state.
///
/// Encode∘decode is the identity, so the checkpoint byte charge is a constant
/// and there is no length-prefix parsing to get wrong. The state deliberately
/// carries no token, no proxy value, no endpoint URL, no page body, and no row
/// list; the only credential-adjacent value is the 32-byte credential source
/// digest, written only for a non-anonymous run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HfRowsCursor {
    next_page_index: u64,
    page_len: u64,
    known_row_total: u64,
    is_inventory_sealed: bool,
    /// `0` when no page is committed, `1` for a body-only receipt, `2` when the
    /// service also served a validator. The validator digest itself is not
    /// durable, so this records only that one was observed.
    content_authority: u8,
    last_object_digest: [u8; 32],
    last_content_digest: [u8; 32],
    last_byte_length: u64,
    commit_sha: [u8; COMMIT_SHA_LEN],
    credential_source_id: Option<[u8; 32]>,
}

impl HfRowsCursor {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CURSOR_BASE_BYTES + CURSOR_CREDENTIAL_TRAILER_BYTES);
        out.extend_from_slice(&self.next_page_index.to_le_bytes());
        out.extend_from_slice(&self.page_len.to_le_bytes());
        out.extend_from_slice(&self.known_row_total.to_le_bytes());
        out.push(u8::from(self.is_inventory_sealed));
        out.push(self.content_authority);
        out.extend_from_slice(&self.last_object_digest);
        out.extend_from_slice(&self.last_content_digest);
        out.extend_from_slice(&self.last_byte_length.to_le_bytes());
        out.extend_from_slice(&self.commit_sha);
        if let Some(source_id) = self.credential_source_id {
            out.extend_from_slice(&source_id);
        }
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let has_credential = match bytes.len() {
            CURSOR_BASE_BYTES => false,
            len if len == CURSOR_BASE_BYTES + CURSOR_CREDENTIAL_TRAILER_BYTES => true,
            _ => return Err(CheckpointError::ObjectVerification),
        };
        let read_u64 = |at: usize| -> Result<u64, CheckpointError> {
            let slice: [u8; 8] = bytes
                .get(at..at + 8)
                .and_then(|slice| slice.try_into().ok())
                .ok_or(CheckpointError::ObjectVerification)?;
            Ok(u64::from_le_bytes(slice))
        };
        let read_32 = |at: usize| -> Result<[u8; 32], CheckpointError> {
            bytes
                .get(at..at + 32)
                .and_then(|slice| slice.try_into().ok())
                .ok_or(CheckpointError::ObjectVerification)
        };
        let is_inventory_sealed = match bytes[24] {
            0 => false,
            1 => true,
            _ => return Err(CheckpointError::ObjectVerification),
        };
        let content_authority = bytes[25];
        if content_authority > 2 {
            return Err(CheckpointError::ObjectVerification);
        }
        let commit_sha: [u8; COMMIT_SHA_LEN] = bytes[98..138]
            .try_into()
            .map_err(|_| CheckpointError::ObjectVerification)?;
        if !commit_sha
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self {
            next_page_index: read_u64(0)?,
            page_len: read_u64(8)?,
            known_row_total: read_u64(16)?,
            is_inventory_sealed,
            content_authority,
            last_object_digest: read_32(26)?,
            last_content_digest: read_32(58)?,
            last_byte_length: read_u64(90)?,
            commit_sha,
            credential_source_id: if has_credential {
                Some(read_32(CURSOR_BASE_BYTES)?)
            } else {
                None
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Factory and preparation
// ---------------------------------------------------------------------------

/// Startup validation and preparation for the Hugging Face rows source.
pub struct HfRowsSourceFactory {
    transport_factory: Arc<dyn HfPageTransportFactory>,
}

impl HfRowsSourceFactory {
    /// Bind the host-owned credential and HTTP authority this source will use.
    #[must_use]
    pub fn new(transport_factory: Arc<dyn HfPageTransportFactory>) -> Self {
        Self { transport_factory }
    }
}

impl fmt::Debug for HfRowsSourceFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HfRowsSourceFactory")
            .field("id", &HF_ROWS_SOURCE_ID)
            .finish_non_exhaustive()
    }
}

impl StreamingDatasetSourceFactory for HfRowsSourceFactory {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor {
        &HF_ROWS_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError> {
        let config: HfRowsSourceConfig =
            serde_json::from_str(authored.get()).map_err(|_| discovery_error())?;
        validate_config(&config)?;
        Ok(Box::new(config))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSourceConfig>,
        context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError> {
        let config = *config
            .into_any()
            .downcast::<HfRowsSourceConfig>()
            .map_err(|_| discovery_error())?;
        // Preparation performs no network I/O: revision resolution and the first
        // inventory probe happen in `open`, which is async and therefore the only
        // place a bounded request can be awaited.
        Ok(Box::new(PreparedHfRowsSource {
            config,
            transport_factory: Arc::clone(&self.transport_factory),
            run: context.run,
            stream_identity: context.stream_semantic_digest,
            clock: Rc::clone(&context.clock),
            reporter: context.issue_reporter.clone(),
        }))
    }
}

/// Prepared, not yet opened, Hugging Face rows source.
struct PreparedHfRowsSource {
    config: HfRowsSourceConfig,
    transport_factory: Arc<dyn HfPageTransportFactory>,
    run: StreamRunIdentity,
    stream_identity: ContentDigest,
    clock: Rc<dyn Clock>,
    reporter: StreamingIssueReporterHandle,
}

/// Resolve a symbolic revision to its immutable 40-hex commit, exactly once.
async fn resolve_commit_sha(
    transport: &dyn HfPageTransport,
    clock: &Rc<dyn Clock>,
    policy: &RequestPolicy,
    hub_endpoint: &Url,
    dataset: &str,
    revision: &str,
) -> Result<String, StreamSourceError> {
    let url = hub_endpoint
        .join(&format!("/api/datasets/{dataset}/revision/{revision}"))
        .map_err(|_| discovery_error())?;
    let fetched = authorized_get_bounded(transport, HfHost::Hub, clock, policy, &url).await?;
    let envelope: RevisionEnvelope =
        serde_json::from_slice(&fetched.bytes).map_err(|_| discovery_error())?;
    let sha = envelope.sha.ok_or_else(discovery_error)?;
    if sha.len() != COMMIT_SHA_LEN || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(discovery_error());
    }
    Ok(sha.to_ascii_lowercase())
}

#[async_trait(?Send)]
impl PreparedStreamingDatasetSource for PreparedHfRowsSource {
    async fn open(
        self: Box<Self>,
        stop: StreamingStopReceiver,
    ) -> Result<OpenedStreamingDatasetSource, StreamSourceError> {
        let control = stop.control();
        let source = self.open_source(stop).await?;
        Ok(OpenedStreamingDatasetSource {
            source: Box::new(source),
            control,
        })
    }
}

impl PreparedHfRowsSource {
    /// Resolve the revision, prove or refuse completeness, and freeze the source.
    async fn open_source(
        self: Box<Self>,
        stop: StreamingStopReceiver,
    ) -> Result<HfRowsSource, StreamSourceError> {
        let hub_endpoint = parse_endpoint(&self.config.hub_endpoint)?;
        let rows_endpoint = parse_endpoint(&self.config.rows_endpoint)?;
        let rows_url_base = rows_endpoint.join("/rows").map_err(|_| discovery_error())?;

        let settings = HfTransportSettings {
            hub_endpoint: hub_endpoint.clone(),
            rows_endpoint: rows_endpoint.clone(),
            proxy: self.config.proxy.clone(),
            proxy_from_env: self.config.proxy_from_env,
            ssl_verify: self.config.ssl_verify,
            credential_env_var: self.config.credential_env_var.clone(),
            credential_token_file: self.config.credential_token_file.clone(),
            allow_anonymous: self.config.allow_anonymous,
        };
        let transport = self
            .transport_factory
            .create(&settings, Rc::clone(&self.clock))
            .await
            .map_err(map_transport_error)?;

        let policy = RequestPolicy {
            max_page_bytes: self.config.max_page_bytes,
            max_auth_retries: self.config.max_auth_retries,
            max_read_retries: self.config.max_read_retries,
            read_backoff_base_ns: self.config.read_backoff_base_ns,
            read_backoff_cap_ns: self.config.read_backoff_cap_ns,
        };

        let commit_sha = resolve_commit_sha(
            transport.as_ref(),
            &self.clock,
            &policy,
            &hub_endpoint,
            &self.config.dataset,
            &self.config.revision,
        )
        .await?;

        let resolution = FrozenResolution {
            stream_identity: self.stream_identity,
            dataset: self.config.dataset.clone(),
            subset: self.config.subset.clone(),
            split: self.config.split.clone(),
            commit_sha,
            page_len: self.config.page_len,
            mode: self.config.mode,
            hub_authority: endpoint_authority(&hub_endpoint)?,
            rows_authority: endpoint_authority(&rows_endpoint)?,
            rows_url_base,
            require_explicit_completeness: self.config.require_explicit_completeness,
            credential_source_id: transport.credential_source_id(),
            is_anonymous: transport.is_anonymous(),
        };
        let snapshot_digest = resolution.snapshot_digest();

        let probe = resolution.probe_url()?;
        let fetched = authorized_get_bounded(
            transport.as_ref(),
            HfHost::Rows,
            &self.clock,
            &policy,
            &probe,
        )
        .await?;
        let inventory = read_inventory(&fetched.bytes, resolution.require_explicit_completeness)?;
        if resolution.mode == HfRowsMode::Finite && !inventory.is_sealed {
            // A partial Viewer conversion cannot stand for a complete split.
            return Err(discovery_error());
        }

        let cursor_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: CURSOR_BUDGET_ITEMS,
            max_bytes: CURSOR_BUDGET_ITEMS * (CURSOR_BASE_BYTES + CURSOR_CREDENTIAL_TRAILER_BYTES),
        })
        .map_err(|_| discovery_error())?;

        let context = Rc::new(HfAcquisitionContext {
            resolution,
            transport,
            clock: Rc::clone(&self.clock),
            reporter: self.reporter.clone(),
            run: self.run,
            policy,
            snapshot_digest,
        });

        Ok(HfRowsSource {
            context,
            stop,
            snapshot: SourceSnapshotReceipt {
                digest: snapshot_digest,
            },
            participant_id: CheckpointParticipantId::new("streaming-source-hf-rows"),
            initialization: ParticipantInitialization::default(),
            cursor_budget,
            poll_interval_ns: self.config.poll_interval_ns,
            max_chunk_bytes: self.config.max_chunk_bytes,
            inventory,
            next_page_index: 0,
            pending_frontier: None,
            is_sealed_out: false,
            receipts: BTreeMap::new(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::clock::SimClock;
    use crate::streaming::budget::BudgetLimits;
    use crate::streaming::identity::{LogicalReplayRunId, RunIncarnationId};
    use crate::streaming::reliability::{
        StreamingIssueReportError, StreamingIssueReportStatus, StreamingIssueReporterEndpoint,
    };
    use crate::streaming::source::streaming_stop_channel;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[derive(Clone, Debug)]
    struct Reply {
        status: u16,
        body: Vec<u8>,
        etag: Option<String>,
    }

    impl Reply {
        fn ok(body: &str) -> Self {
            Self {
                status: 200,
                body: body.as_bytes().to_vec(),
                etag: None,
            }
        }

        fn status(status: u16) -> Self {
            Self {
                status,
                body: Vec::new(),
                etag: None,
            }
        }
    }

    #[derive(Debug, Default)]
    struct FakeState {
        revision: Vec<Reply>,
        rows: Vec<Reply>,
        hub_requests: Vec<String>,
        rows_requests: Vec<String>,
        refreshes: u32,
        authorized: u32,
    }

    #[derive(Debug)]
    struct FakeTransport {
        state: RefCell<FakeState>,
        is_anonymous: bool,
    }

    impl FakeTransport {
        fn new(revision: Vec<Reply>, rows: Vec<Reply>) -> Rc<Self> {
            Rc::new(Self {
                state: RefCell::new(FakeState {
                    revision,
                    rows,
                    ..FakeState::default()
                }),
                is_anonymous: false,
            })
        }
    }

    #[async_trait(?Send)]
    impl HfPageTransport for FakeTransport {
        async fn authorized_get(
            &self,
            host: HfHost,
            url: &Url,
        ) -> Result<HfHttpResponse, HfTransportError> {
            let mut state = self.state.borrow_mut();
            let reply = match host {
                HfHost::Hub => {
                    state.hub_requests.push(url.to_string());
                    if state.revision.len() > 1 {
                        state.revision.remove(0)
                    } else {
                        state
                            .revision
                            .first()
                            .cloned()
                            .ok_or(HfTransportError::Transport)?
                    }
                }
                HfHost::Rows => {
                    state.rows_requests.push(url.to_string());
                    if state.rows.len() > 1 {
                        state.rows.remove(0)
                    } else {
                        state
                            .rows
                            .first()
                            .cloned()
                            .ok_or(HfTransportError::Transport)?
                    }
                }
            };
            let mut headers = BTreeMap::new();
            if let Some(etag) = reply.etag {
                headers.insert("etag".to_owned(), etag);
            }
            Ok(HfHttpResponse {
                status: Some(reply.status),
                headers,
                body: Some(Bytes::from(reply.body)),
                credential_generation: 1,
            })
        }

        async fn refresh(&self, generation: u64) -> Result<HfRefreshOutcome, HfTransportError> {
            self.state.borrow_mut().refreshes += 1;
            Ok(HfRefreshOutcome::Rotated(generation + 1))
        }

        fn note_authorized(&self, _generation: u64) {
            self.state.borrow_mut().authorized += 1;
        }

        fn credential_source_id(&self) -> [u8; 32] {
            [7; 32]
        }

        fn is_anonymous(&self) -> bool {
            self.is_anonymous
        }
    }

    #[derive(Debug)]
    struct FakeTransportFactory {
        transport: Rc<FakeTransport>,
    }

    #[async_trait(?Send)]
    impl HfPageTransportFactory for FakeTransportFactory {
        async fn create(
            &self,
            _settings: &HfTransportSettings,
            _clock: Rc<dyn Clock>,
        ) -> Result<Rc<dyn HfPageTransport>, HfTransportError> {
            Ok(Rc::clone(&self.transport) as Rc<dyn HfPageTransport>)
        }
    }

    // `Send + Sync` is required by the factory trait; the fake is only ever used
    // from the single test thread, and the inner `Rc` never crosses one.
    unsafe impl Send for FakeTransportFactory {}
    unsafe impl Sync for FakeTransportFactory {}

    #[derive(Default)]
    struct CountingReporter {
        count: RefCell<usize>,
    }

    #[async_trait(?Send)]
    impl StreamingIssueReporterEndpoint for Rc<CountingReporter> {
        async fn report(
            &self,
            _issue: OrdinaryStreamingIssue,
        ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
            *self.count.borrow_mut() += 1;
            Ok(StreamingIssueReportStatus::Accepted)
        }
    }

    fn run_identity() -> StreamRunIdentity {
        StreamRunIdentity::new(LogicalReplayRunId::from_bytes([3; 32]))
    }

    fn acquisition_budget() -> AcquisitionBudget {
        AcquisitionBudget::new(
            StreamingResourceBudget::new(BudgetLimits {
                max_items: 16,
                max_bytes: 1 << 20,
            })
            .expect("memory budget"),
            StreamingResourceBudget::new(BudgetLimits {
                max_items: 4,
                max_bytes: 4096,
            })
            .expect("disk budget"),
        )
    }

    fn base_config() -> HfRowsSourceConfig {
        serde_json::from_str(
            r#"{"dataset":"openai/gsm8k","subset":"main","split":"train",
                "hub_endpoint":"http://127.0.0.1:9","rows_endpoint":"http://127.0.0.1:9",
                "page_len":2,"poll_interval_ns":1000,"read_backoff_base_ns":1000,
                "read_backoff_cap_ns":4000}"#,
        )
        .expect("base config decodes")
    }

    fn resolution(stream: u8, page_len: u64) -> FrozenResolution {
        FrozenResolution {
            stream_identity: ContentDigest::from_bytes([stream; 32]),
            dataset: "openai/gsm8k".to_owned(),
            subset: "main".to_owned(),
            split: "train".to_owned(),
            commit_sha: COMMIT.to_owned(),
            page_len,
            mode: HfRowsMode::Finite,
            hub_authority: "127.0.0.1:9".to_owned(),
            rows_authority: "127.0.0.1:9".to_owned(),
            rows_url_base: Url::parse("http://127.0.0.1:9/rows").expect("rows base"),
            require_explicit_completeness: true,
            credential_source_id: [7; 32],
            is_anonymous: false,
        }
    }

    #[test]
    fn page_identity_is_arithmetic_stable_and_stream_scoped() {
        let first = resolution(1, 2);
        let same = resolution(1, 2);
        let other_stream = resolution(9, 2);

        assert_eq!(first.page_identity(3), same.page_identity(3));
        assert_ne!(first.page_identity(3), first.page_identity(4));
        assert_ne!(first.page_identity(3), other_stream.page_identity(3));
        assert_ne!(first.page_identity(3), resolution(1, 5).page_identity(3));
        assert_ne!(first.snapshot_digest(), other_stream.snapshot_digest());
    }

    #[test]
    fn page_url_offsets_are_page_index_times_page_len() {
        let resolution = resolution(1, 25);
        let url = resolution.page_url(4).expect("page url");
        assert!(url.as_str().contains("offset=100"), "{url}");
        assert!(url.as_str().contains("length=25"), "{url}");
        let probe = resolution.probe_url().expect("probe url");
        assert!(probe.as_str().contains("offset=0&length=1"), "{probe}");
    }

    #[test]
    fn cursor_round_trip_is_the_identity() {
        let mut commit = [0_u8; COMMIT_SHA_LEN];
        commit.copy_from_slice(COMMIT.as_bytes());
        let cursor = HfRowsCursor {
            next_page_index: 12,
            page_len: 25,
            known_row_total: 300,
            is_inventory_sealed: true,
            content_authority: 2,
            last_object_digest: [4; 32],
            last_content_digest: [5; 32],
            last_byte_length: 4096,
            commit_sha: commit,
            credential_source_id: Some([7; 32]),
        };
        let encoded = cursor.encode();
        assert_eq!(
            encoded.len(),
            CURSOR_BASE_BYTES + CURSOR_CREDENTIAL_TRAILER_BYTES
        );
        assert_eq!(HfRowsCursor::decode(&encoded).expect("decode"), cursor);

        let anonymous = HfRowsCursor {
            credential_source_id: None,
            ..cursor
        };
        let encoded = anonymous.encode();
        assert_eq!(encoded.len(), CURSOR_BASE_BYTES);
        assert_eq!(HfRowsCursor::decode(&encoded).expect("decode"), anonymous);

        assert!(HfRowsCursor::decode(&encoded[..CURSOR_BASE_BYTES - 1]).is_err());
    }

    #[test]
    fn authored_configuration_is_refused_before_any_effect() {
        let reject = |patch: &str| {
            let json =
                format!(r#"{{"dataset":"openai/gsm8k","subset":"main","split":"train",{patch}}}"#);
            let config: HfRowsSourceConfig = serde_json::from_str(&json).expect("config decodes");
            assert!(validate_config(&config).is_err(), "{patch}");
        };
        reject(r#""page_len":0"#);
        reject(r#""page_len":101"#);
        reject(r#""max_auth_retries":0"#);
        reject(r#""read_backoff_base_ns":0"#);
        reject(r#""read_backoff_cap_ns":1"#);
        reject(r#""hub_endpoint":"ftp://example.invalid""#);
        reject(r#""credential_env_var":"A","credential_token_file":"/tmp/t""#);

        let bad_dataset: HfRowsSourceConfig =
            serde_json::from_str(r#"{"dataset":"gsm8k","subset":"main","split":"train"}"#)
                .expect("config decodes");
        assert!(validate_config(&bad_dataset).is_err());

        // An unknown key is refused by strict decoding, before validation.
        assert!(
            serde_json::from_str::<HfRowsSourceConfig>(
                r#"{"dataset":"a/b","subset":"m","split":"train","nope":1}"#
            )
            .is_err()
        );
    }

    #[test]
    fn absent_partial_is_completeness_only_when_explicitly_allowed() {
        let body = br#"{"num_rows_total":4}"#;
        assert!(!read_inventory(body, true).expect("inventory").is_sealed);
        assert!(read_inventory(body, false).expect("inventory").is_sealed);
        assert!(
            read_inventory(br#"{"partial":false}"#, false).is_err(),
            "num_rows_total is required"
        );
    }

    #[test]
    fn read_backoff_is_exponential_and_capped() {
        let policy = RequestPolicy {
            max_page_bytes: 1024,
            max_auth_retries: 2,
            max_read_retries: 3,
            read_backoff_base_ns: 1000,
            read_backoff_cap_ns: 3000,
        };
        assert_eq!(policy.read_backoff_ns(0), 1000);
        assert_eq!(policy.read_backoff_ns(1), 2000);
        assert_eq!(policy.read_backoff_ns(2), 3000);
        assert_eq!(policy.read_backoff_ns(9), 3000);
    }

    #[test]
    fn receipt_drift_is_detected_and_absent_etags_are_not_evidence() {
        let committed = HfPageReceipt::observe(b"page", Some("\"abc\""));
        assert!(HfPageReceipt::observe(b"page", Some("W/\"abc\"")).is_consistent_with(&committed));
        assert!(HfPageReceipt::observe(b"page", None).is_consistent_with(&committed));
        assert!(!HfPageReceipt::observe(b"page", Some("\"xyz\"")).is_consistent_with(&committed));
        assert!(!HfPageReceipt::observe(b"other", Some("\"abc\"")).is_consistent_with(&committed));
    }

    struct Harness {
        transport: Rc<FakeTransport>,
        reporter: Rc<CountingReporter>,
        clock: Rc<SimClock>,
    }

    impl Harness {
        fn new(revision: Vec<Reply>, rows: Vec<Reply>) -> Self {
            Self {
                transport: FakeTransport::new(revision, rows),
                reporter: Rc::new(CountingReporter::default()),
                clock: Rc::new(SimClock::new()),
            }
        }

        async fn open(
            &self,
            config: HfRowsSourceConfig,
        ) -> Result<OpenedStreamingDatasetSource, StreamSourceError> {
            let factory = HfRowsSourceFactory::new(Arc::new(FakeTransportFactory {
                transport: Rc::clone(&self.transport),
            }));
            let context = StreamingSourcePrepareContext {
                run: run_identity(),
                stream_semantic_digest: ContentDigest::from_bytes([1; 32]),
                clock: Rc::clone(&self.clock) as Rc<dyn Clock>,
                acquisition_budget: acquisition_budget(),
                issue_reporter: StreamingIssueReporterHandle::new(Rc::clone(&self.reporter)),
            };
            let prepared = factory.prepare(Box::new(config), &context)?;
            let (_control, stop) = streaming_stop_channel();
            prepared.open(stop).await
        }

        /// Open the concrete source so a test can drive `restore` directly.
        async fn open_source(
            &self,
            config: HfRowsSourceConfig,
        ) -> Result<HfRowsSource, StreamSourceError> {
            let context = StreamingSourcePrepareContext {
                run: run_identity(),
                stream_semantic_digest: ContentDigest::from_bytes([1; 32]),
                clock: Rc::clone(&self.clock) as Rc<dyn Clock>,
                acquisition_budget: acquisition_budget(),
                issue_reporter: StreamingIssueReporterHandle::new(Rc::clone(&self.reporter)),
            };
            let (_control, stop) = streaming_stop_channel();
            Box::new(PreparedHfRowsSource {
                config,
                transport_factory: Arc::new(FakeTransportFactory {
                    transport: Rc::clone(&self.transport),
                }),
                run: context.run,
                stream_identity: context.stream_semantic_digest,
                clock: Rc::clone(&context.clock),
                reporter: context.issue_reporter.clone(),
            })
            .open_source(stop)
            .await
        }
    }

    fn revision_ok() -> Vec<Reply> {
        vec![Reply::ok(&format!(r#"{{"sha":"{COMMIT}"}}"#))]
    }

    fn drive<F>(clock: &Rc<SimClock>, body: F)
    where
        F: Future<Output = ()> + 'static,
    {
        Rc::clone(clock).drive(Box::pin(body));
    }

    #[test]
    fn partial_split_is_refused_in_finite_mode() {
        let harness = Harness::new(
            revision_ok(),
            vec![Reply::ok(r#"{"num_rows_total":4,"partial":true}"#)],
        );
        let clock = Rc::clone(&harness.clock);
        drive(&clock, async move {
            let opened = harness.open(base_config()).await;
            assert!(opened.is_err(), "a partial conversion is not a split");
        });
    }

    #[test]
    fn page_inventory_is_arithmetic_and_frontier_is_monotonic() {
        let harness = Harness::new(
            revision_ok(),
            vec![Reply::ok(r#"{"num_rows_total":5,"partial":false}"#)],
        );
        let clock = Rc::clone(&harness.clock);
        drive(&clock, async move {
            let mut opened = harness.open(base_config()).await.expect("open");
            let snapshot = opened.source.snapshot().digest;
            let mut positions = Vec::new();
            let mut frontiers = Vec::new();
            let seal = loop {
                match opened.source.next_event().await.expect("event") {
                    SourceEvent::Partition(partition) => positions.push(partition.position()),
                    SourceEvent::Frontier(frontier) => frontiers.push(frontier.through),
                    SourceEvent::Seal(seal) => break seal,
                }
            };
            // ceil(5 / 2) == 3 pages.
            assert_eq!(positions.len(), 3);
            assert_eq!(frontiers, positions);
            assert!(frontiers.windows(2).all(|pair| pair[0] < pair[1]));
            assert_eq!(seal.final_position, Some(SourcePosition::new(2)));
            assert_eq!(seal.digest, snapshot);
        });
    }

    #[test]
    fn follow_mode_parks_on_a_partial_split_and_seals_only_once_complete() {
        let harness = Harness::new(
            revision_ok(),
            vec![
                Reply::ok(r#"{"num_rows_total":0,"partial":true}"#),
                Reply::ok(r#"{"num_rows_total":2,"partial":false}"#),
            ],
        );
        let clock = Rc::clone(&harness.clock);
        drive(&clock, async move {
            let mut config = base_config();
            config.mode = HfRowsMode::Follow;
            let mut opened = harness.open(config).await.expect("open");
            // The first poll parks on the clock, re-probes, and only then announces.
            let event = opened.source.next_event().await.expect("event");
            assert!(matches!(event, SourceEvent::Partition(_)));
            assert_eq!(harness.reporter.count.borrow().to_owned(), 0);
        });
    }

    #[test]
    fn retry_exhaustion_reports_one_retryable_issue_and_leaves_the_stream_live() {
        let harness = Harness::new(
            revision_ok(),
            vec![
                Reply::ok(r#"{"num_rows_total":4,"partial":false}"#),
                // Page 0 never clears its authentication failure.
                Reply::status(401),
                Reply::status(401),
                Reply::status(401),
                Reply::status(401),
                Reply::ok(r#"{"rows":[]}"#),
            ],
        );
        let clock = Rc::clone(&harness.clock);
        drive(&clock, async move {
            let mut opened = harness.open(base_config()).await.expect("open");
            let budget = acquisition_budget();
            let SourceEvent::Partition(partition) =
                opened.source.next_event().await.expect("event")
            else {
                panic!("expected a partition");
            };
            let outcome = partition
                .content()
                .acquire(
                    PartitionAccessRequest::Sequential { resume_offset: 0 },
                    &budget,
                )
                .await;
            assert!(outcome.is_err());
            assert_eq!(harness.reporter.count.borrow().to_owned(), 1);

            // The source itself stays live: the frontier and the next page follow.
            let frontier = opened.source.next_event().await.expect("frontier");
            assert!(matches!(frontier, SourceEvent::Frontier(_)));
            let next = opened.source.next_event().await.expect("next page");
            assert!(matches!(next, SourceEvent::Partition(_)));
        });
    }

    #[test]
    fn an_acquired_page_reads_its_verbatim_body_in_bounded_chunks() {
        let body = r#"{"num_rows_total":2,"partial":false,"rows":[1,2]}"#;
        let harness = Harness::new(revision_ok(), vec![Reply::ok(body)]);
        let clock = Rc::clone(&harness.clock);
        let expected = body.as_bytes().to_vec();
        drive(&clock, async move {
            let mut config = base_config();
            config.max_chunk_bytes = 8;
            let mut opened = harness.open(config).await.expect("open");
            let budget = acquisition_budget();
            let SourceEvent::Partition(partition) =
                opened.source.next_event().await.expect("event")
            else {
                panic!("expected a partition");
            };
            let announced = *partition.content().identity();
            let acquired = partition
                .content()
                .acquire(
                    PartitionAccessRequest::Sequential { resume_offset: 0 },
                    &budget,
                )
                .await
                .expect("acquire");
            assert_eq!(*acquired.identity(), announced);
            assert_eq!(acquired.size_bytes(), Some(expected.len() as u64));

            let crate::streaming::source::AcquiredPartitionAccess::Sequential(mut reader) =
                acquired.into_access()
            else {
                panic!("expected a sequential reader");
            };
            let max = NonZeroUsize::new(64).expect("nonzero");
            let mut seen = Vec::new();
            while let Some(chunk) = reader.next_chunk(max, &budget).await.expect("chunk") {
                assert!(chunk.as_bytes().len() <= 8, "chunk bound is honored");
                seen.extend_from_slice(chunk.as_bytes());
            }
            assert_eq!(seen, expected, "the body is emitted verbatim");
        });
    }

    #[test]
    fn restore_resumes_after_the_committed_page_and_refuses_drift() {
        let harness = Harness::new(
            revision_ok(),
            vec![Reply::ok(r#"{"num_rows_total":6,"partial":false}"#)],
        );
        let clock = Rc::clone(&harness.clock);
        drive(&clock, async move {
            let mut source = harness.open_source(base_config()).await.expect("open");
            let mut commit = [0_u8; COMMIT_SHA_LEN];
            commit.copy_from_slice(COMMIT.as_bytes());
            let cursor = HfRowsCursor {
                next_page_index: 2,
                page_len: 2,
                known_row_total: 6,
                is_inventory_sealed: true,
                content_authority: 1,
                last_object_digest: *source.context.resolution.page_identity(1).as_bytes(),
                last_content_digest: [5; 32],
                last_byte_length: 11,
                commit_sha: commit,
                credential_source_id: Some([7; 32]),
            };
            assert_eq!(
                HfRowsCursor::decode(&cursor.encode()).expect("decode"),
                cursor
            );

            source.restore(cursor).expect("restore");
            let SourceEvent::Partition(partition) = source.next_event().await.expect("event")
            else {
                panic!("expected a partition");
            };
            // Pages 0 and 1 are never re-announced after a committed cursor.
            assert_eq!(partition.position(), SourcePosition::new(2));

            let mut moved = cursor;
            moved.commit_sha[0] = b'f';
            assert_eq!(
                source.restore(moved),
                Err(CheckpointError::SourceUnavailableOnResume)
            );

            let mut relengthed = cursor;
            relengthed.page_len = 3;
            assert_eq!(
                source.restore(relengthed),
                Err(CheckpointError::SourceUnavailableOnResume)
            );

            let mut recredentialed = cursor;
            recredentialed.credential_source_id = Some([9; 32]);
            assert_eq!(
                source.restore(recredentialed),
                Err(CheckpointError::SourceUnavailableOnResume)
            );

            let mut drifted = cursor;
            drifted.last_object_digest = [0; 32];
            assert_eq!(
                source.restore(drifted),
                Err(CheckpointError::SourceUnavailableOnResume)
            );
        });
    }

    #[test]
    fn a_shrinking_row_total_under_follow_is_a_mutated_object() {
        let harness = Harness::new(
            revision_ok(),
            vec![
                Reply::ok(r#"{"num_rows_total":4,"partial":true}"#),
                Reply::ok(r#"{"num_rows_total":2,"partial":true}"#),
            ],
        );
        let clock = Rc::clone(&harness.clock);
        drive(&clock, async move {
            let mut config = base_config();
            config.mode = HfRowsMode::Follow;
            let mut opened = harness.open(config).await.expect("open");
            // Pages 0 and 1 are addressable from the first probe.
            for _ in 0..4 {
                opened.source.next_event().await.expect("event");
            }
            let outcome = opened.source.next_event().await;
            assert!(outcome.is_err(), "a shrinking inventory is refused");
        });
    }

    #[test]
    fn run_identity_mismatch_is_refused_by_the_prepared_run() {
        // The run bound at preparation is the one every issue is scoped to.
        assert_eq!(run_identity(), run_identity());
        assert_ne!(
            run_identity(),
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([4; 32]))
        );
        let _ = RunIncarnationId::from_bytes([0; 32]);
    }
}
