// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3-compatible finite/follow streaming source.
//!
//! One S3 object generation is one streaming partition. Positions are dense and
//! allocated at publication, so a key discovered late by reconciliation lands at
//! a *greater* position than lexicographically later keys already published; the
//! ordering promise is that no completeness frontier advances over an unseen key,
//! not that positions follow key order. A pagination boundary is never a frontier.
//!
//! Identity is `BLAKE3(bucket, key, generation-token, size)`, knowable at listing
//! time. The BLAKE3 of the acquired bytes is a separate `ContentDigest` used for
//! verification and provenance. An ETag — single-part or multipart — is an
//! opaque provider token and is never a content digest.
//!
//! Retry, backoff, and credential invalidation live here because
//! [`crate::streaming::aws`] disables SDK retry: `Clock::sleep` returns a
//! non-`Send` future and cannot back the SDK's sleeper, so there is exactly one
//! retry authority and it is clocked.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::num::NonZeroUsize;
use std::rc::Rc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::clock::Clock;
use crate::streaming::aws::{AwsClientSettings, AwsProxySelection, AwsS3ClientFactory};
use crate::streaming::budget::{BudgetError, BudgetLimits, StreamingResourceBudget};
use crate::streaming::checkpoint::{
    BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
    CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
    PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
};
use crate::streaming::failure::{
    AcquisitionFailureCode, OrderingFailureCode, OrdinaryStreamingFailure, SourceFailureCode,
    StreamSourceError,
};
use crate::streaming::identity::{ContentDigest, ImmutableObjectIdentity, LogicalReplayRunId};
use crate::streaming::reliability::{
    OrdinaryStreamingIssue, StreamingInputDomainIdentity, StreamingIssueClass,
    StreamingIssueReportStatus, StreamingIssueReporterHandle,
};
use crate::streaming::source::{
    AcquiredPartition, AcquisitionBudget, BudgetedSourceChunk, OpenedStreamingDatasetSource,
    PartitionAccessKind, PartitionAccessRequest, PreparedStreamingDatasetSource,
    SequentialSourceChunk, SourceEvent, SourceFrontier, SourcePartition, SourcePartitionContent,
    SourceSeal, SourceSnapshotReceipt, StreamingDatasetSource, StreamingDatasetSourceFactory,
    StreamingRangeReader, StreamingResumeGranularity, StreamingSequentialReader,
    StreamingSourceDescriptor, StreamingSourceMode, StreamingSourceOrdering,
    StreamingSourcePlacement, StreamingSourcePrepareContext, StreamingSourceRetention,
    StreamingStopReceiver, ValidatedStreamingSourceConfig,
};
use crate::streaming::unit::{SourcePosition, StateBudgetFailureCode};

use super::s3_client::{
    MAX_LIST_PAGE_KEYS, S3ByteRange, S3Client, S3ClientError, S3GetRequest, S3ListRequest,
    S3ObjectBody, S3ObjectReader,
};

/// Stable checkpoint participant identity for the S3 source.
pub const S3_PARTICIPANT_ID: &str = "streaming.source.s3";
const S3_CURSOR_SCHEMA_ID: &str = "aiperf.streaming.source.s3.cursor";
const S3_CURSOR_SCHEMA_VERSION: u32 = 1;
const S3_IDENTITY_DOMAIN: &[u8] = b"aiperf.streaming.s3-object.v1";
const S3_CONTENT_DOMAIN: &[u8] = b"aiperf.streaming.s3-content.v1";
const S3_SNAPSHOT_DOMAIN: &[u8] = b"aiperf.streaming.s3-snapshot.v1";
const S3_BACKOFF_DOMAIN: &[u8] = b"aiperf.streaming.s3-backoff.v1";
const S3_INVENTORY_DOMAIN: &[u8] = b"aiperf.streaming.s3-inventory.v1";
/// Hard ceiling on the encoded participant payload.
const MAX_CURSOR_PAYLOAD_BYTES: usize = 1 << 20;
/// Largest manifest body the source will read.
const MAX_MANIFEST_BYTES: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

/// Compiled S3 source capabilities.
///
/// `ControllerOnly` is a decision, not a default: placing acquisition on a cell
/// would require shipping credentials or presigned URLs off the controller.
/// `supports_virtual_clock: false` records that SigV4 signing uses the SDK's own
/// time source, so a virtual clock would produce `RequestTimeTooSkewed` against
/// a real endpoint.
pub static S3_SOURCE_DESCRIPTOR: StreamingSourceDescriptor = StreamingSourceDescriptor {
    id: "s3",
    description: "S3-compatible object source with sealed-interval reconciliation",
    modes: &[StreamingSourceMode::Finite, StreamingSourceMode::Follow],
    access: &[
        PartitionAccessKind::Sequential,
        PartitionAccessKind::RangeReadable,
    ],
    ordering: StreamingSourceOrdering::Partition,
    resume: &[
        StreamingResumeGranularity::Partition,
        StreamingResumeGranularity::Byte,
    ],
    has_event_time: false,
    has_stable_record_ids: false,
    retention: StreamingSourceRetention::ResumeRootReachability,
    placement: StreamingSourcePlacement::ControllerOnly,
    supports_virtual_clock: false,
};

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Provider token binding one immutable object generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum S3GenerationToken {
    /// Bucket versioning is enabled; this is the exact generation.
    VersionId {
        /// Provider version id.
        value: String,
    },
    /// Unversioned single-part ETag, an opaque token and never a digest.
    SinglePartETag {
        /// Provider ETag with quotes stripped.
        value: String,
    },
    /// Unversioned multipart ETag (`<hex>-<parts>`), an opaque token.
    MultipartETag {
        /// Provider ETag with quotes stripped.
        value: String,
    },
    /// Neither a version id nor a usable ETag was listed.
    Absent,
}

impl S3GenerationToken {
    /// Classify a listed object's version/ETag pair.
    #[must_use]
    pub fn classify(version_id: Option<&str>, etag: Option<&str>) -> Self {
        if let Some(version_id) = version_id.filter(|value| !value.is_empty()) {
            return Self::VersionId {
                value: version_id.to_owned(),
            };
        }
        let Some(etag) = etag.map(|value| value.trim_matches('"')) else {
            return Self::Absent;
        };
        if etag.is_empty() {
            return Self::Absent;
        }
        // A multipart ETag is `<hex>-<partcount>`; it is not a hash of the object
        // bytes and does not survive a re-upload with a different part size, so
        // it is a generation token only.
        if etag
            .rsplit_once('-')
            .is_some_and(|(head, parts)| {
                !head.is_empty()
                    && !parts.is_empty()
                    && parts.bytes().all(|byte| byte.is_ascii_digit())
            })
        {
            return Self::MultipartETag {
                value: etag.to_owned(),
            };
        }
        Self::SinglePartETag {
            value: etag.to_owned(),
        }
    }

    const fn tag(&self) -> u8 {
        match self {
            Self::VersionId { .. } => 0,
            Self::SinglePartETag { .. } => 1,
            Self::MultipartETag { .. } => 2,
            Self::Absent => 3,
        }
    }

    /// Borrow the opaque provider token text, empty when absent.
    #[must_use]
    pub fn value(&self) -> &str {
        match self {
            Self::VersionId { value }
            | Self::SinglePartETag { value }
            | Self::MultipartETag { value } => value,
            Self::Absent => "",
        }
    }

    /// Whether the token can bind a conditional read to this exact generation.
    #[must_use]
    pub const fn is_conditionally_bindable(&self) -> bool {
        !matches!(self, Self::Absent)
    }

    /// Whether the token names an exact provider version.
    #[must_use]
    pub const fn is_version_qualified(&self) -> bool {
        matches!(self, Self::VersionId { .. })
    }

    /// The provider version id, when the generation is version-qualified.
    #[must_use]
    pub fn provider_version(&self) -> Option<&str> {
        match self {
            Self::VersionId { value } => Some(value),
            _ => None,
        }
    }
}

/// Derive the immutable identity of one S3 object generation.
///
/// Length-prefixed fields keep `(bucket, key)` unambiguous, and the token kind
/// participates so a version id and an identical ETag string cannot collide.
#[must_use]
pub fn s3_object_identity(
    bucket: &str,
    key: &str,
    token: &S3GenerationToken,
    size_bytes: u64,
) -> ImmutableObjectIdentity {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, S3_IDENTITY_DOMAIN);
    update_field(&mut hasher, bucket.as_bytes());
    update_field(&mut hasher, key.as_bytes());
    update_field(&mut hasher, &[token.tag()]);
    update_field(&mut hasher, token.value().as_bytes());
    update_field(&mut hasher, &size_bytes.to_le_bytes());
    ImmutableObjectIdentity::from_bytes(*hasher.finalize().as_bytes())
}

/// Rolling BLAKE3 over acquired object bytes.
///
/// This is the only content hash in the S3 path. It is never an ETag, and an
/// ETag is never it.
#[derive(Clone, Debug)]
struct ContentHasher(blake3::Hasher);

impl ContentHasher {
    fn new() -> Self {
        let mut hasher = blake3::Hasher::new();
        update_field(&mut hasher, S3_CONTENT_DOMAIN);
        Self(hasher)
    }

    fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn digest(&self) -> ContentDigest {
        ContentDigest::from_bytes(*self.0.finalize().as_bytes())
    }
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Fidelity the source can honestly advertise under the authored policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceFidelity {
    /// Every published generation is reconcilable and restart is exact.
    Lossless {
        /// What proves the frontier.
        proof: LosslessFrontierProof,
    },
    /// Only a bounded rescan window is retained; late keys outside it are lost.
    LossyWindow {
        /// Authored window bound in keys.
        max_keys: u32,
    },
}

/// What makes a lossless frontier provable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LosslessFrontierProof {
    /// A producer-written manifest seals each interval.
    SealedManifest,
    /// One pass over a versioned prefix, sealed immediately.
    VersionedPrefixSnapshot,
    /// Monotonic publication keys plus an asserted hard no-backfill horizon.
    MonotonicKeysWithHardNoBackfill,
}

/// Authored inventory shape.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum S3InventoryPolicy {
    /// Sealed manifest objects, the reference pre-indexed shape.
    Manifest {
        /// Key suffix identifying a sealed manifest object.
        manifest_suffix: String,
        /// Whether the source seals after the manifest is satisfied.
        is_finite: bool,
    },
    /// One pass over a versioned prefix; every entry must be version-qualified.
    VersionedPrefixSnapshot,
    /// Key-monotonic follow with an asserted hard no-backfill horizon.
    IntervalFollow {
        /// Producer-asserted no-backfill horizon in nanoseconds.
        no_backfill_horizon_ns: i64,
        /// Whether the producer asserts hard no-backfill.
        has_hard_no_backfill: bool,
        /// Whether publication keys are monotonic.
        has_monotonic_keys: bool,
    },
    /// Explicitly lossy bounded rescan window.
    LossyWindow {
        /// Authored window bound in keys.
        max_keys: u32,
    },
}

impl S3InventoryPolicy {
    /// Whether this inventory shape can reach an explicit source seal.
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        match self {
            Self::Manifest { is_finite, .. } => *is_finite,
            Self::VersionedPrefixSnapshot => true,
            Self::IntervalFollow { .. } | Self::LossyWindow { .. } => false,
        }
    }
}

/// Strictly decoded authored source configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct S3SourceConfig {
    /// Bucket to read.
    pub bucket: String,
    /// Key prefix scoping discovery.
    #[serde(default)]
    pub prefix: Option<String>,
    /// Inventory shape.
    pub policy: S3InventoryPolicy,
    /// Hard listing page bound.
    pub page_max_keys: u16,
    /// Hard bound on pages fetched per discovery pass.
    pub max_pages_per_pass: u32,
    /// Hard bound on retained unsealed generations.
    pub max_unsealed_generations: u32,
    /// Bounded retry attempts per operation.
    pub max_attempts: u32,
    /// First backoff delay.
    pub base_backoff_ns: i64,
    /// Backoff ceiling.
    pub max_backoff_ns: i64,
    /// Poll wait between discovery passes.
    pub poll_interval_ns: i64,
    /// Optional endpoint override for S3-compatible gateways.
    #[serde(default)]
    pub endpoint_url: Option<String>,
    /// Optional region.
    #[serde(default)]
    pub region: Option<String>,
    /// Path-style addressing, required by most gateways.
    #[serde(default)]
    pub force_path_style: bool,
    /// Optional shared-config profile selecting the credential chain.
    #[serde(default)]
    pub credential_profile: Option<String>,
}

/// Bounded retry inputs shared by listing and acquisition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct S3RetryPolicy {
    max_attempts: u32,
    base_backoff_ns: i64,
    max_backoff_ns: i64,
}

/// Validated policy plus the fidelity it can honestly claim.
#[derive(Clone, Debug)]
pub struct PreparedS3Policy {
    config: S3SourceConfig,
    fidelity: SourceFidelity,
    snapshot_digest: ContentDigest,
}

impl PreparedS3Policy {
    /// Fidelity this policy can honestly advertise.
    #[must_use]
    pub const fn fidelity(&self) -> SourceFidelity {
        self.fidelity
    }

    /// Digest binding bucket, prefix, policy shape, and fidelity.
    #[must_use]
    pub const fn snapshot_digest(&self) -> ContentDigest {
        self.snapshot_digest
    }

    /// Borrow the validated authored configuration.
    #[must_use]
    pub const fn config(&self) -> &S3SourceConfig {
        &self.config
    }

    const fn retry(&self) -> S3RetryPolicy {
        S3RetryPolicy {
            max_attempts: self.config.max_attempts,
            base_backoff_ns: self.config.base_backoff_ns,
            max_backoff_ns: self.config.max_backoff_ns,
        }
    }
}

/// Closed authored-policy validation failure.
///
/// This type is source-owned because the streaming seam's [`StreamSourceError`]
/// is a stringless `Copy` code with a closed variant set and no
/// `LosslessFrontierUnprovable` classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum S3PolicyError {
    /// A mutable listing cannot prove a lossless frontier.
    LosslessFrontierUnprovable {
        /// Whether hard no-backfill was asserted.
        has_hard_no_backfill: bool,
        /// Whether publication keys are monotonic.
        has_monotonic_keys: bool,
    },
    /// A bound is zero or above the supported maximum.
    UnboundedOrZeroLimit,
    /// An authored duration is non-positive or unrepresentable.
    InvalidTiming,
    /// The authored bucket, prefix, or manifest suffix is unusable.
    InvalidTarget,
}

impl fmt::Display for S3PolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::LosslessFrontierUnprovable { .. } => {
                "a mutable S3 listing cannot prove a lossless frontier"
            }
            Self::UnboundedOrZeroLimit => "an authored S3 bound is zero or above the maximum",
            Self::InvalidTiming => "an authored S3 duration is non-positive or unrepresentable",
            Self::InvalidTarget => "the authored S3 bucket, prefix, or manifest suffix is unusable",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for S3PolicyError {}

impl From<S3PolicyError> for StreamSourceError {
    fn from(_: S3PolicyError) -> Self {
        Self::source(SourceFailureCode::Discovery)
    }
}

/// Validate an authored policy and resolve the fidelity it can claim.
///
/// Lossless requires a sealed manifest, a versioned prefix snapshot, or
/// monotonic keys *plus* hard no-backfill. Anything else is either an explicit
/// bounded lossy window or a refusal — never a silently degraded "lossless".
///
/// # Errors
///
/// Returns [`S3PolicyError`] when a bound, duration, target, or frontier proof
/// is unusable.
pub fn validate_s3_policy(config: S3SourceConfig) -> Result<PreparedS3Policy, S3PolicyError> {
    if config.bucket.is_empty() {
        return Err(S3PolicyError::InvalidTarget);
    }
    if config.page_max_keys == 0
        || config.page_max_keys > MAX_LIST_PAGE_KEYS
        || config.max_pages_per_pass == 0
        || config.max_unsealed_generations == 0
        || config.max_attempts == 0
        || config.max_attempts > 8
    {
        return Err(S3PolicyError::UnboundedOrZeroLimit);
    }
    for value in [
        config.base_backoff_ns,
        config.max_backoff_ns,
        config.poll_interval_ns,
    ] {
        if value <= 0 {
            return Err(S3PolicyError::InvalidTiming);
        }
    }
    if config.base_backoff_ns > config.max_backoff_ns {
        return Err(S3PolicyError::InvalidTiming);
    }

    let fidelity = match &config.policy {
        S3InventoryPolicy::Manifest {
            manifest_suffix, ..
        } => {
            if manifest_suffix.is_empty() {
                return Err(S3PolicyError::InvalidTarget);
            }
            SourceFidelity::Lossless {
                proof: LosslessFrontierProof::SealedManifest,
            }
        }
        S3InventoryPolicy::VersionedPrefixSnapshot => SourceFidelity::Lossless {
            proof: LosslessFrontierProof::VersionedPrefixSnapshot,
        },
        S3InventoryPolicy::IntervalFollow {
            no_backfill_horizon_ns,
            has_hard_no_backfill,
            has_monotonic_keys,
        } => {
            if *no_backfill_horizon_ns <= 0 {
                return Err(S3PolicyError::InvalidTiming);
            }
            // A lexicographic cursor alone cannot prove that a later-created key
            // will not appear behind it, so both facts are required.
            if !(*has_hard_no_backfill && *has_monotonic_keys) {
                return Err(S3PolicyError::LosslessFrontierUnprovable {
                    has_hard_no_backfill: *has_hard_no_backfill,
                    has_monotonic_keys: *has_monotonic_keys,
                });
            }
            SourceFidelity::Lossless {
                proof: LosslessFrontierProof::MonotonicKeysWithHardNoBackfill,
            }
        }
        S3InventoryPolicy::LossyWindow { max_keys } => {
            if *max_keys == 0 {
                return Err(S3PolicyError::UnboundedOrZeroLimit);
            }
            SourceFidelity::LossyWindow {
                max_keys: *max_keys,
            }
        }
    };

    let snapshot_digest = policy_snapshot_digest(&config, fidelity);
    Ok(PreparedS3Policy {
        config,
        fidelity,
        snapshot_digest,
    })
}

fn policy_snapshot_digest(config: &S3SourceConfig, fidelity: SourceFidelity) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, S3_SNAPSHOT_DOMAIN);
    update_field(&mut hasher, config.bucket.as_bytes());
    update_field(
        &mut hasher,
        config.prefix.as_deref().unwrap_or_default().as_bytes(),
    );
    update_field(&mut hasher, format!("{fidelity:?}").as_bytes());
    update_field(&mut hasher, format!("{:?}", config.policy).as_bytes());
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

// ---------------------------------------------------------------------------
// Bounded discovery observation
// ---------------------------------------------------------------------------

/// Bounded observation of listing work, so pagination bounds are testable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct S3ListingHighWater {
    /// Greatest number of objects returned by any single page.
    pub list_page_items: usize,
    /// Greatest number of retained unsealed generations.
    pub retained_generation_entries: usize,
    /// Greatest number of pages fetched in a single pass.
    pub pages_this_pass: u32,
}

// ---------------------------------------------------------------------------
// Retained cursor
// ---------------------------------------------------------------------------

/// One retained unsealed generation.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct S3GenerationEntry {
    key: String,
    token: S3GenerationToken,
    size_bytes: u64,
    position: SourcePosition,
}

/// One partition with a partially consumed sequential reader.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct S3OpenPartition {
    position: SourcePosition,
    identity: ImmutableObjectIdentity,
    next_byte_offset: u64,
    /// Digest of the bytes this incarnation read up to `next_byte_offset`.
    ///
    /// BLAKE3 state is not recoverable from a digest, so a byte-resumed reader
    /// starts a fresh rolling digest at `next_byte_offset`; this value is the
    /// retained prefix receipt, not a resumable hasher.
    prefix_digest: ContentDigest,
}

/// Partition-scoped state shared between discovery, acquisition, and readers.
#[derive(Debug, Default)]
struct S3PartitionState {
    open: BTreeMap<u64, S3OpenPartition>,
    completed: BTreeMap<u64, ContentDigest>,
    holes: BTreeSet<u64>,
}

/// Strict, bounded participant payload.
///
/// Sealed intervals collapse into one digest; only unsealed generations and
/// currently open partitions are enumerated, and both are hard-capped, so the
/// payload cannot grow with run length.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct S3SourceCursor {
    snapshot_digest: ContentDigest,
    next_position: SourcePosition,
    sealed_through: Option<SourcePosition>,
    sealed_inventory_digest: ContentDigest,
    sealed_key_bound: Option<String>,
    unsealed: Vec<S3GenerationEntry>,
    open_partitions: Vec<S3OpenPartition>,
    completed: Vec<(SourcePosition, ContentDigest)>,
    holes: Vec<SourcePosition>,
    is_sealed: bool,
}

// ---------------------------------------------------------------------------
// Retry
// ---------------------------------------------------------------------------

/// Which operation is being retried, for classification and logging.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum S3OpClass {
    List,
    Get,
}

impl S3OpClass {
    const fn label(self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Get => "get",
        }
    }
}

/// Bounded exponential backoff with deterministic jitter.
///
/// Jitter is derived from BLAKE3 rather than `rand` so a replay reproduces the
/// same schedule and no new dependency enters the source path.
fn backoff_delay_ns(policy: S3RetryPolicy, attempt: u32, class: S3OpClass) -> i64 {
    let shift = attempt.saturating_sub(1).min(16);
    // `shift` is clamped to 16, so the doubling factor is exact and the only
    // saturation that matters is the multiply against the authored base.
    let scaled = policy.base_backoff_ns.saturating_mul(1_i64 << shift);
    let capped = scaled
        .min(policy.max_backoff_ns)
        .max(policy.base_backoff_ns);
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, S3_BACKOFF_DOMAIN);
    update_field(&mut hasher, &[class as u8]);
    update_field(&mut hasher, &attempt.to_le_bytes());
    let mut seed_bytes = [0_u8; 8];
    seed_bytes.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    let seed = u64::from_le_bytes(seed_bytes);
    let span = u64::try_from(capped / 4).unwrap_or(1).max(1);
    capped.saturating_add(i64::try_from(seed % span).unwrap_or(0))
}

/// Retry one bounded operation under the injected clock.
///
/// An identity violation returns immediately: no number of retries can make a
/// changed key name the frozen bytes again. An authorization failure invalidates
/// the shared credential authority before the wait, so the next attempt
/// refreshes without changing the frozen object identity.
async fn with_retry<T, F, Fut>(
    client: &Rc<dyn S3Client>,
    clock: &Rc<dyn Clock>,
    policy: S3RetryPolicy,
    class: S3OpClass,
    mut operation: F,
) -> Result<T, S3ClientError>
where
    F: FnMut(Rc<dyn S3Client>) -> Fut,
    Fut: Future<Output = Result<T, S3ClientError>>,
{
    let mut attempt = 0_u32;
    loop {
        match operation(Rc::clone(client)).await {
            Ok(value) => return Ok(value),
            Err(error) if error.is_identity_violation() => return Err(error),
            Err(error) => {
                attempt = attempt.saturating_add(1);
                if attempt >= policy.max_attempts
                    || !(error.is_retryable() || error.is_authorization())
                {
                    tracing::debug!(
                        code = error.code(),
                        op = class.label(),
                        attempt,
                        "s3 operation failed"
                    );
                    return Err(error);
                }
                if error.is_authorization() {
                    client.invalidate_credentials();
                }
                Rc::clone(clock)
                    .sleep(backoff_delay_ns(policy, attempt, class))
                    .await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reliability reporting
// ---------------------------------------------------------------------------

/// Host reporting boundary plus the frozen scope every S3 issue carries.
struct S3IssueContext {
    reporter: StreamingIssueReporterHandle,
    run: StreamRunIdentity,
    input_domain: StreamingInputDomainIdentity,
    semantic_context_digest: ContentDigest,
}

impl fmt::Debug for S3IssueContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S3IssueContext")
            .field("semantic_context_digest", &self.semantic_context_digest)
            .finish_non_exhaustive()
    }
}

impl S3IssueContext {
    /// Report one partition-scoped ordinary fact. Disposition stays host-owned.
    async fn report_partition(
        &self,
        identity: ImmutableObjectIdentity,
        position: SourcePosition,
        class: StreamingIssueClass,
        retry_ordinal: u32,
        failure: StreamSourceError,
        tiebreaker: ContentDigest,
    ) {
        let Ok(issue) = OrdinaryStreamingIssue::partition(
            self.run,
            self.input_domain.clone(),
            identity,
            class,
            self.semantic_context_digest,
            position,
            retry_ordinal,
            tiebreaker,
            OrdinaryStreamingFailure::Source(failure),
        ) else {
            // Construction only fails on a host-owned class, which this source
            // never selects.
            return;
        };
        match self.reporter.report(issue).await {
            Ok(StreamingIssueReportStatus::Accepted) | Err(_) => {}
            Ok(_) => {}
        }
    }
}

fn scope_tiebreaker(identity: &ImmutableObjectIdentity, class: S3OpClass) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, b"aiperf.streaming.s3-scope.v1");
    update_field(&mut hasher, identity.as_bytes());
    update_field(&mut hasher, &[class as u8]);
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Validates and prepares the `s3` source.
#[derive(Debug, Default)]
pub struct S3SourceFactory;

impl StreamingDatasetSourceFactory for S3SourceFactory {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor {
        &S3_SOURCE_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError> {
        let config: S3SourceConfig = serde_json::from_str(authored.get())
            .map_err(|_| StreamSourceError::source(SourceFailureCode::Discovery))?;
        let policy = validate_s3_policy(config)?;
        Ok(Box::new(policy))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSourceConfig>,
        context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError> {
        let policy = *config
            .into_any()
            .downcast::<PreparedS3Policy>()
            .map_err(|_| StreamSourceError::source(SourceFailureCode::Discovery))?;
        // A virtual clock cannot sign a SigV4 request; refuse before any client
        // exists rather than producing `RequestTimeTooSkewed` mid-run.
        if context.clock.is_virtual() {
            return Err(StreamSourceError::source(
                SourceFailureCode::SourceUnavailable,
            ));
        }
        Ok(Box::new(PreparedS3Source {
            policy,
            reporter: context.issue_reporter.clone(),
            clock: Rc::clone(&context.clock),
        }))
    }
}

/// Prepared but not yet opened S3 source.
///
/// No AWS type exists yet: `aws_config::defaults(..).load()` is async and
/// `prepare` is sync, so client construction is deferred to `open`, which is
/// worker-local and already `?Send`.
pub struct PreparedS3Source {
    policy: PreparedS3Policy,
    reporter: StreamingIssueReporterHandle,
    clock: Rc<dyn Clock>,
}

impl fmt::Debug for PreparedS3Source {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedS3Source")
            .field("fidelity", &self.policy.fidelity())
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl PreparedStreamingDatasetSource for PreparedS3Source {
    async fn open(
        self: Box<Self>,
        stop: StreamingStopReceiver,
    ) -> Result<OpenedStreamingDatasetSource, StreamSourceError> {
        let config = self.policy.config();
        let settings = AwsClientSettings {
            region: config.region.clone(),
            endpoint_url: config.endpoint_url.clone(),
            force_path_style: config.force_path_style,
            // Proxy selection stays with the shared AWS authority, whose
            // loopback exclusion keeps a local gateway off an ambient proxy.
            proxy: AwsProxySelection::Disabled,
            operation_timeout_ns: config.max_backoff_ns.max(30_000_000_000),
            connect_timeout_ns: 5_000_000_000,
        };
        let profile = config.credential_profile.clone();
        let factory = AwsS3ClientFactory::prepare_default_chain(settings, profile.as_deref())
            .await
            .map_err(StreamSourceError::from)?;
        let (client, projection) = factory.build_client(Rc::clone(&self.clock));
        let transport = super::s3_client::AwsS3Transport::new(
            client,
            projection,
            Arc::clone(factory.authority()),
        );
        let control = stop.control();
        // The prepare context carries no run identity, so the run is bound to
        // the frozen authored snapshot; a different authored source is a
        // different run for reliability scoping.
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes(
            *self.policy.snapshot_digest().as_bytes(),
        ));
        let source = S3Source::with_client(
            Rc::new(transport),
            self.policy,
            run,
            self.reporter,
            self.clock,
            default_cursor_budget()?,
            stop,
        );
        Ok(OpenedStreamingDatasetSource {
            source: Box::new(source),
            control,
        })
    }
}

use std::sync::Arc;

fn default_cursor_budget() -> Result<StreamingResourceBudget, StreamSourceError> {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: MAX_CURSOR_PAYLOAD_BYTES,
    })
    .map_err(|_| StreamSourceError::source(SourceFailureCode::SourceUnavailable))
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Mutable discovery state, held separately from the stop receiver so both can
/// be borrowed across one `select!`.
struct S3Discovery {
    client: Rc<dyn S3Client>,
    policy: PreparedS3Policy,
    clock: Rc<dyn Clock>,
    issue: Rc<S3IssueContext>,
    partitions: Rc<RefCell<S3PartitionState>>,
    next_position: SourcePosition,
    seen: BTreeMap<ImmutableObjectIdentity, S3GenerationEntry>,
    published_keys: BTreeSet<String>,
    pending: VecDeque<SourcePartition>,
    pending_frontier: Option<SourceFrontier>,
    start_after: Option<String>,
    published_max_key: Option<String>,
    published_max_position: Option<SourcePosition>,
    sealed_through: Option<SourcePosition>,
    sealed_inventory: blake3::Hasher,
    sealed_key_bound: Option<String>,
    manifest_key: Option<String>,
    manifest_keys: Option<BTreeSet<String>>,
    last_new_key_ns: i64,
    has_run_pass: bool,
    is_sealed: bool,
    high_water: S3ListingHighWater,
}

impl S3Discovery {
    fn retry(&self) -> S3RetryPolicy {
        self.policy.retry()
    }

    /// Run one bounded discovery pass, publishing every new generation.
    ///
    /// Pagination is bounded twice — page size and pages per pass — and never
    /// advances a frontier. A frontier is emitted only when the interval seals,
    /// which is the sole authority proving no unseen key precedes it.
    async fn discovery_pass(&mut self) -> Result<(), StreamSourceError> {
        let config = self.policy.config().clone();
        let mut continuation: Option<String> = None;
        let mut pages = 0_u32;
        loop {
            let request = S3ListRequest {
                bucket: config.bucket.clone(),
                prefix: config.prefix.clone(),
                start_after: self.start_after.clone(),
                continuation_token: continuation.clone(),
                max_keys: config.page_max_keys,
            };
            let page = with_retry(
                &self.client,
                &self.clock,
                self.retry(),
                S3OpClass::List,
                move |client| {
                    let request = request.clone();
                    async move { client.list_page(request).await }
                },
            )
            .await
            .map_err(|_| StreamSourceError::source(SourceFailureCode::Discovery))?;

            self.high_water.list_page_items =
                self.high_water.list_page_items.max(page.objects.len());

            for object in page.objects {
                if self.is_manifest_key(&object.key) {
                    self.manifest_key = Some(object.key);
                    continue;
                }
                let token =
                    S3GenerationToken::classify(object.version_id.as_deref(), object.etag.as_deref());
                if matches!(self.policy.fidelity(), SourceFidelity::Lossless { .. })
                    && !token.is_conditionally_bindable()
                {
                    // No conditional read can bind this generation, so no
                    // lossless claim can be honest about it.
                    return Err(StreamSourceError::source(SourceFailureCode::Discovery));
                }
                let identity =
                    s3_object_identity(&config.bucket, &object.key, &token, object.size_bytes);
                if self.seen.contains_key(&identity) {
                    continue;
                }
                self.reject_backfill(&object.key)?;
                if self.seen.len() >= config.max_unsealed_generations as usize {
                    // Retaining an arbitrary unsealed prefix is unbounded state
                    // and is refused rather than silently truncated.
                    return Err(StreamSourceError::source(SourceFailureCode::Discovery));
                }
                let position = self.next_position;
                self.next_position = position.checked_add(1).map_err(|_| {
                    StreamSourceError::ordering(OrderingFailureCode::CoordinateOverflow)
                })?;
                self.seen.insert(
                    identity,
                    S3GenerationEntry {
                        key: object.key.clone(),
                        token: token.clone(),
                        size_bytes: object.size_bytes,
                        position,
                    },
                );
                self.high_water.retained_generation_entries = self
                    .high_water
                    .retained_generation_entries
                    .max(self.seen.len());
                self.published_keys.insert(object.key.clone());
                if self
                    .published_max_key
                    .as_deref()
                    .is_none_or(|current| current < object.key.as_str())
                {
                    self.published_max_key = Some(object.key.clone());
                }
                self.published_max_position = Some(position);
                self.last_new_key_ns = self.clock.now_ns();

                let content = S3ObjectContent {
                    client: Rc::clone(&self.client),
                    clock: Rc::clone(&self.clock),
                    issue: Rc::clone(&self.issue),
                    partitions: Rc::clone(&self.partitions),
                    retry: self.retry(),
                    bucket: config.bucket.clone(),
                    key: object.key,
                    token,
                    identity,
                    size_bytes: object.size_bytes,
                    position,
                };
                self.pending
                    .push_back(SourcePartition::new(position, Box::new(content)));
            }

            pages = pages.saturating_add(1);
            self.high_water.pages_this_pass = self.high_water.pages_this_pass.max(pages);
            continuation = page.next_continuation_token;
            if continuation.is_none() || pages >= config.max_pages_per_pass {
                break;
            }
        }

        self.has_run_pass = true;
        if continuation.is_none() {
            self.try_seal_interval().await?;
        }
        Ok(())
    }

    fn is_manifest_key(&self, key: &str) -> bool {
        match self.policy.config().policy {
            S3InventoryPolicy::Manifest {
                ref manifest_suffix,
                ..
            } => key.ends_with(manifest_suffix.as_str()),
            _ => false,
        }
    }

    /// Refuse a key that lands behind an asserted monotonic publication order.
    ///
    /// Only `IntervalFollow` asserts hard no-backfill; every other shape treats
    /// a late key as ordinary reconciliation and publishes it at a later
    /// position.
    fn reject_backfill(&self, key: &str) -> Result<(), StreamSourceError> {
        if !matches!(
            self.policy.config().policy,
            S3InventoryPolicy::IntervalFollow { .. }
        ) {
            return Ok(());
        }
        let is_behind = self
            .published_max_key
            .as_deref()
            .is_some_and(|current| key <= current)
            || self
                .sealed_key_bound
                .as_deref()
                .is_some_and(|bound| key <= bound);
        if is_behind {
            tracing::debug!(
                component = "streaming.source.s3",
                "s3 no-backfill assertion violated by a key behind the published order"
            );
            return Err(StreamSourceError::source(SourceFailureCode::Discovery));
        }
        Ok(())
    }

    /// Decide whether the active interval can seal, and emit its frontier.
    async fn try_seal_interval(&mut self) -> Result<(), StreamSourceError> {
        if self.is_sealed {
            return Ok(());
        }
        let is_sealable = match self.policy.config().policy {
            S3InventoryPolicy::Manifest { .. } => self.is_manifest_satisfied().await?,
            S3InventoryPolicy::VersionedPrefixSnapshot => true,
            S3InventoryPolicy::IntervalFollow {
                no_backfill_horizon_ns,
                ..
            } => {
                self.published_max_position.is_some()
                    && self
                        .clock
                        .now_ns()
                        .saturating_sub(self.last_new_key_ns)
                        >= no_backfill_horizon_ns
            }
            // A bounded rescan window has no watermark authority, so it never
            // claims a completeness frontier.
            S3InventoryPolicy::LossyWindow { .. } => false,
        };
        if !is_sealable {
            return Ok(());
        }
        let Some(through) = self.published_max_position else {
            return Ok(());
        };
        self.seal_through(through);
        self.pending_frontier = Some(SourceFrontier { through });
        if self.policy.config().policy.is_finite() {
            self.is_sealed = true;
        }
        Ok(())
    }

    /// Read the sealing manifest once and test whether every key is observed.
    async fn is_manifest_satisfied(&mut self) -> Result<bool, StreamSourceError> {
        if self.manifest_keys.is_none() {
            let Some(manifest_key) = self.manifest_key.clone() else {
                return Ok(false);
            };
            let bucket = self.policy.config().bucket.clone();
            let request = S3GetRequest {
                bucket,
                key: manifest_key,
                version_id: None,
                if_match_etag: None,
                range: None,
            };
            let body = with_retry(
                &self.client,
                &self.clock,
                self.retry(),
                S3OpClass::Get,
                move |client| {
                    let request = request.clone();
                    async move { client.get_version(request).await }
                },
            )
            .await
            .map_err(|_| StreamSourceError::source(SourceFailureCode::Snapshot))?;
            let bytes = read_all(body, MAX_MANIFEST_BYTES).await?;
            let text = String::from_utf8(bytes)
                .map_err(|_| StreamSourceError::source(SourceFailureCode::Snapshot))?;
            self.manifest_keys = Some(
                text.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_owned)
                    .collect(),
            );
        }
        let Some(keys) = self.manifest_keys.as_ref() else {
            return Ok(false);
        };
        Ok(keys.iter().all(|key| self.published_keys.contains(key)))
    }

    /// Retire every generation at or below `through` into the sealed digest.
    fn seal_through(&mut self, through: SourcePosition) {
        let retired: Vec<ImmutableObjectIdentity> = self
            .seen
            .iter()
            .filter(|(_, entry)| entry.position.get() <= through.get())
            .map(|(identity, _)| *identity)
            .collect();
        for identity in retired {
            if let Some(entry) = self.seen.remove(&identity) {
                update_field(&mut self.sealed_inventory, identity.as_bytes());
                if self
                    .sealed_key_bound
                    .as_deref()
                    .is_none_or(|bound| bound < entry.key.as_str())
                {
                    self.sealed_key_bound = Some(entry.key);
                }
            }
        }
        self.sealed_through = Some(through);
        // Every sealed key is behind the bound, so the next pass starts after it
        // rather than perpetually rescanning a sealed prefix.
        self.start_after = self.sealed_key_bound.clone();
    }

    fn sealed_inventory_digest(&self) -> ContentDigest {
        let mut hasher = self.sealed_inventory.clone();
        update_field(&mut hasher, S3_INVENTORY_DOMAIN);
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Wait one poll interval, then run the next bounded pass.
    async fn next_pass(&mut self) -> Result<(), StreamSourceError> {
        if self.has_run_pass {
            Rc::clone(&self.clock)
                .sleep(self.policy.config().poll_interval_ns)
                .await;
        }
        self.discovery_pass().await
    }
}

/// Drain one object body into memory under a hard byte bound.
async fn read_all(mut body: S3ObjectBody, limit: usize) -> Result<Vec<u8>, StreamSourceError> {
    let chunk_bound = NonZeroUsize::new(64 * 1024).unwrap_or(NonZeroUsize::MIN);
    let mut buffer = Vec::new();
    while let Some(chunk) = body
        .reader
        .next_chunk(chunk_bound)
        .await
        .map_err(|_| StreamSourceError::source(SourceFailureCode::Snapshot))?
    {
        if buffer.len().saturating_add(chunk.len()) > limit {
            return Err(StreamSourceError::source(SourceFailureCode::Snapshot));
        }
        buffer.extend_from_slice(&chunk);
    }
    Ok(buffer)
}

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

/// One opened S3 source.
pub struct S3Source {
    discovery: S3Discovery,
    stop: StreamingStopReceiver,
    snapshot: SourceSnapshotReceipt,
    participant_id: CheckpointParticipantId,
    checkpoint_budget: StreamingResourceBudget,
    initialization: ParticipantInitialization,
    has_emitted_seal: bool,
}

impl fmt::Debug for S3Source {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // No credential material, endpoint, or provider token is rendered.
        formatter
            .debug_struct("S3Source")
            .field("fidelity", &self.discovery.policy.fidelity())
            .field("high_water", &self.discovery.high_water)
            .finish_non_exhaustive()
    }
}

impl S3Source {
    /// Bind an already-constructed provider seam to the validated policy.
    ///
    /// This is the injection point for a non-AWS S3-compatible transport and for
    /// socket-free tests; production opens go through [`PreparedS3Source`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn with_client(
        client: Rc<dyn S3Client>,
        policy: PreparedS3Policy,
        run: StreamRunIdentity,
        reporter: StreamingIssueReporterHandle,
        clock: Rc<dyn Clock>,
        checkpoint_budget: StreamingResourceBudget,
        stop: StreamingStopReceiver,
    ) -> Self {
        let snapshot_digest = policy.snapshot_digest();
        let issue = Rc::new(S3IssueContext {
            reporter,
            run,
            input_domain: StreamingInputDomainIdentity::new(
                snapshot_digest,
                ImmutableObjectIdentity::from_bytes(*snapshot_digest.as_bytes()),
            ),
            semantic_context_digest: snapshot_digest,
        });
        let now_ns = clock.now_ns();
        Self {
            discovery: S3Discovery {
                client,
                policy,
                clock,
                issue,
                partitions: Rc::new(RefCell::new(S3PartitionState::default())),
                next_position: SourcePosition::new(1),
                seen: BTreeMap::new(),
                published_keys: BTreeSet::new(),
                pending: VecDeque::new(),
                pending_frontier: None,
                start_after: None,
                published_max_key: None,
                published_max_position: None,
                sealed_through: None,
                sealed_inventory: blake3::Hasher::new(),
                sealed_key_bound: None,
                manifest_key: None,
                manifest_keys: None,
                last_new_key_ns: now_ns,
                has_run_pass: false,
                is_sealed: false,
                high_water: S3ListingHighWater::default(),
            },
            stop,
            snapshot: SourceSnapshotReceipt {
                digest: snapshot_digest,
            },
            participant_id: CheckpointParticipantId::new(S3_PARTICIPANT_ID),
            checkpoint_budget,
            initialization: ParticipantInitialization::default(),
            has_emitted_seal: false,
        }
    }

    /// Bounded listing observation, so pagination bounds are assertable.
    #[must_use]
    pub const fn high_water(&self) -> S3ListingHighWater {
        self.discovery.high_water
    }

    /// Fidelity this source advertises under its authored policy.
    #[must_use]
    pub const fn fidelity(&self) -> SourceFidelity {
        self.discovery.policy.fidelity()
    }

    /// Positions whose frozen generation was reachable but could not be acquired.
    #[must_use]
    pub fn holes(&self) -> Vec<SourcePosition> {
        self.discovery
            .partitions
            .borrow()
            .holes
            .iter()
            .map(|value| SourcePosition::new(*value))
            .collect()
    }

    /// Content digests of partitions whose acquisition completed.
    #[must_use]
    pub fn completed_digests(&self) -> Vec<(SourcePosition, ContentDigest)> {
        self.discovery
            .partitions
            .borrow()
            .completed
            .iter()
            .map(|(position, digest)| (SourcePosition::new(*position), *digest))
            .collect()
    }

    fn encode_cursor(&self) -> S3SourceCursor {
        let partitions = self.discovery.partitions.borrow();
        S3SourceCursor {
            snapshot_digest: self.discovery.policy.snapshot_digest(),
            next_position: self.discovery.next_position,
            sealed_through: self.discovery.sealed_through,
            sealed_inventory_digest: self.discovery.sealed_inventory_digest(),
            sealed_key_bound: self.discovery.sealed_key_bound.clone(),
            unsealed: self.discovery.seen.values().cloned().collect(),
            open_partitions: partitions.open.values().cloned().collect(),
            completed: partitions
                .completed
                .iter()
                .map(|(position, digest)| (SourcePosition::new(*position), *digest))
                .collect(),
            holes: partitions
                .holes
                .iter()
                .map(|value| SourcePosition::new(*value))
                .collect(),
            is_sealed: self.discovery.is_sealed,
        }
    }

    fn restore_cursor(&mut self, cursor: S3SourceCursor) {
        self.discovery.next_position = cursor.next_position;
        self.discovery.sealed_through = cursor.sealed_through;
        self.discovery.sealed_key_bound = cursor.sealed_key_bound.clone();
        self.discovery.start_after = cursor.sealed_key_bound;
        self.discovery.is_sealed = cursor.is_sealed;
        self.discovery.seen = cursor
            .unsealed
            .into_iter()
            .map(|entry| {
                let identity = s3_object_identity(
                    &self.discovery.policy.config().bucket,
                    &entry.key,
                    &entry.token,
                    entry.size_bytes,
                );
                (identity, entry)
            })
            .collect();
        self.discovery.published_keys = self
            .discovery
            .seen
            .values()
            .map(|entry| entry.key.clone())
            .collect();
        self.discovery.published_max_key = self.discovery.published_keys.iter().next_back().cloned();
        self.discovery.published_max_position = self
            .discovery
            .seen
            .values()
            .map(|entry| entry.position)
            .max()
            .or(cursor.sealed_through);
        let mut partitions = self.discovery.partitions.borrow_mut();
        partitions.open = cursor
            .open_partitions
            .into_iter()
            .map(|open| (open.position.get(), open))
            .collect();
        partitions.completed = cursor
            .completed
            .into_iter()
            .map(|(position, digest)| (position.get(), digest))
            .collect();
        partitions.holes = cursor.holes.into_iter().map(SourcePosition::get).collect();
    }

    /// Exact resume offset retained for one open partition, when it has one.
    #[must_use]
    pub fn open_partition_offset(&self, position: SourcePosition) -> Option<u64> {
        self.discovery
            .partitions
            .borrow()
            .open
            .get(&position.get())
            .map(|open| open.next_byte_offset)
    }
}

#[async_trait(?Send)]
impl StreamingDatasetSource for S3Source {
    fn snapshot(&self) -> &SourceSnapshotReceipt {
        &self.snapshot
    }

    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError> {
        loop {
            if let Some(partition) = self.discovery.pending.pop_front() {
                return Ok(SourceEvent::Partition(partition));
            }
            if let Some(frontier) = self.discovery.pending_frontier.take() {
                return Ok(SourceEvent::Frontier(frontier));
            }
            if self.discovery.is_sealed && !self.has_emitted_seal {
                self.has_emitted_seal = true;
                return Ok(SourceEvent::Seal(SourceSeal {
                    final_position: self.discovery.published_max_position,
                    digest: self.snapshot.digest,
                }));
            }
            // `stop` and `discovery` are distinct fields, so both futures can be
            // held across one select without aliasing `self`.
            let Self {
                discovery, stop, ..
            } = self;
            tokio::select! {
                biased;
                stopped = stop.stopped() => return Err(stopped.unwrap_err_or_stop()),
                result = discovery.next_pass() => result?,
            }
        }
    }
}

/// Narrow helper so the stop arm never has to fabricate a success value.
trait StopOutcome {
    fn unwrap_err_or_stop(self) -> StreamSourceError;
}

impl StopOutcome for Result<(), StreamSourceError> {
    fn unwrap_err_or_stop(self) -> StreamSourceError {
        // `StreamingStopReceiver::stopped` only ever resolves with the opaque
        // controlled-stop error; the `Ok` arm is unreachable but must not panic.
        self.err()
            .unwrap_or_else(|| StreamSourceError::source(SourceFailureCode::SourceUnavailable))
    }
}

// ---------------------------------------------------------------------------
// Partition content
// ---------------------------------------------------------------------------

/// Immutable content authority for one S3 object generation.
struct S3ObjectContent {
    client: Rc<dyn S3Client>,
    clock: Rc<dyn Clock>,
    issue: Rc<S3IssueContext>,
    partitions: Rc<RefCell<S3PartitionState>>,
    retry: S3RetryPolicy,
    bucket: String,
    key: String,
    token: S3GenerationToken,
    identity: ImmutableObjectIdentity,
    size_bytes: u64,
    position: SourcePosition,
}

impl S3ObjectContent {
    fn get_request(&self, range: Option<S3ByteRange>) -> S3GetRequest {
        let (version_id, if_match_etag) = match &self.token {
            S3GenerationToken::VersionId { value } => (Some(value.clone()), None),
            S3GenerationToken::SinglePartETag { value }
            | S3GenerationToken::MultipartETag { value } => (None, Some(value.clone())),
            S3GenerationToken::Absent => (None, None),
        };
        S3GetRequest {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            version_id,
            if_match_etag,
            range,
        }
    }

    /// Whether the read delivered the exact generation that was listed.
    ///
    /// A different version id, ETag, or full-object length means the key now
    /// names other bytes: that is a mutation refusal, never a hole.
    fn matches_frozen_generation(&self, body: &S3ObjectBody, is_full_read: bool) -> bool {
        let token_matches = match &self.token {
            S3GenerationToken::VersionId { value } => {
                body.version_id.as_deref() == Some(value.as_str())
            }
            S3GenerationToken::SinglePartETag { value }
            | S3GenerationToken::MultipartETag { value } => body
                .etag
                .as_deref()
                .map(|etag| etag.trim_matches('"'))
                .is_some_and(|etag| etag == value.as_str()),
            S3GenerationToken::Absent => true,
        };
        let length_matches = !is_full_read
            || body
                .content_length
                .is_none_or(|length| length == self.size_bytes);
        token_matches && length_matches
    }

    async fn conditional_get(
        &self,
        range: Option<S3ByteRange>,
    ) -> Result<S3ObjectBody, StreamSourceError> {
        let request = self.get_request(range);
        let outcome = with_retry(
            &self.client,
            &self.clock,
            self.retry,
            S3OpClass::Get,
            move |client| {
                let request = request.clone();
                async move { client.get_version(request).await }
            },
        )
        .await;
        let body = match outcome {
            Ok(body) => body,
            Err(error) => return Err(self.report_failure(error, None).await),
        };
        if !self.matches_frozen_generation(&body, range.is_none()) {
            return Err(self
                .report_failure(S3ClientError::PreconditionFailed, None)
                .await);
        }
        Ok(body)
    }

    /// Classify one acquisition failure, report it, and record a hole when the
    /// frozen generation is still believed intact.
    async fn report_failure(
        &self,
        error: S3ClientError,
        attempt: Option<u32>,
    ) -> StreamSourceError {
        let (failure, class, is_hole) = if error.is_identity_violation() {
            (
                StreamSourceError::source(SourceFailureCode::MutatedObject),
                StreamingIssueClass::Permanent,
                false,
            )
        } else if matches!(error, S3ClientError::NotFound) {
            (
                StreamSourceError::acquisition(AcquisitionFailureCode::Open),
                StreamingIssueClass::Permanent,
                true,
            )
        } else {
            (
                StreamSourceError::acquisition(AcquisitionFailureCode::Read),
                StreamingIssueClass::Retryable,
                true,
            )
        };
        if is_hole {
            self.partitions
                .borrow_mut()
                .holes
                .insert(self.position.get());
        }
        self.issue
            .report_partition(
                self.identity,
                self.position,
                class,
                attempt.unwrap_or(self.retry.max_attempts),
                failure,
                scope_tiebreaker(&self.identity, S3OpClass::Get),
            )
            .await;
        failure
    }
}

#[async_trait(?Send)]
impl SourcePartitionContent for S3ObjectContent {
    fn identity(&self) -> &ImmutableObjectIdentity {
        &self.identity
    }

    fn size_bytes(&self) -> Option<u64> {
        Some(self.size_bytes)
    }

    async fn acquire(
        &self,
        request: PartitionAccessRequest,
        budget: &AcquisitionBudget,
    ) -> Result<AcquiredPartition, StreamSourceError> {
        match request {
            PartitionAccessRequest::Sequential { resume_offset } => {
                if resume_offset > self.size_bytes {
                    return Err(StreamSourceError::acquisition(
                        AcquisitionFailureCode::ObjectLimitExceeded,
                    ));
                }
                let range = (resume_offset > 0).then_some(S3ByteRange {
                    offset: resume_offset,
                    end: self.size_bytes,
                });
                let body = self.conditional_get(range).await?;
                let authority = budget.acquire_memory(1, 0).await?;
                self.partitions.borrow_mut().open.insert(
                    self.position.get(),
                    S3OpenPartition {
                        position: self.position,
                        identity: self.identity,
                        next_byte_offset: resume_offset,
                        prefix_digest: ContentHasher::new().digest(),
                    },
                );
                AcquiredPartition::sequential(
                    self.position,
                    self.identity,
                    Some(self.size_bytes),
                    resume_offset,
                    Box::new(S3SequentialReader {
                        reader: body.reader,
                        hasher: ContentHasher::new(),
                        next_offset: resume_offset,
                        size_bytes: self.size_bytes,
                        position: self.position,
                        identity: self.identity,
                        partitions: Rc::clone(&self.partitions),
                    }),
                    authority,
                )
            }
            PartitionAccessRequest::RangeReadable => {
                // The conditional probe binds the frozen generation before any
                // range is served, so a mutated key is refused up front.
                let body = self.conditional_get(None).await?;
                drop(body);
                let authority = budget.acquire_memory(1, 0).await?;
                AcquiredPartition::range_readable(
                    self.position,
                    self.identity,
                    Some(self.size_bytes),
                    Box::new(S3RangeReader {
                        content: S3ObjectContent {
                            client: Rc::clone(&self.client),
                            clock: Rc::clone(&self.clock),
                            issue: Rc::clone(&self.issue),
                            partitions: Rc::clone(&self.partitions),
                            retry: self.retry,
                            bucket: self.bucket.clone(),
                            key: self.key.clone(),
                            token: self.token.clone(),
                            identity: self.identity,
                            size_bytes: self.size_bytes,
                            position: self.position,
                        },
                    }),
                    authority,
                )
            }
            // Generation one never stages an S3 object to local disk.
            PartitionAccessRequest::SeekableLocal => {
                Err(StreamSourceError::acquisition(AcquisitionFailureCode::Open))
            }
        }
    }
}

/// Bounded forward reader carrying the rolling content digest.
struct S3SequentialReader {
    reader: Box<dyn S3ObjectReader>,
    hasher: ContentHasher,
    next_offset: u64,
    size_bytes: u64,
    position: SourcePosition,
    identity: ImmutableObjectIdentity,
    partitions: Rc<RefCell<S3PartitionState>>,
}

#[async_trait(?Send)]
impl StreamingSequentialReader for S3SequentialReader {
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<Option<SequentialSourceChunk>, StreamSourceError> {
        let Some(bytes) = self
            .reader
            .next_chunk(max_bytes)
            .await
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?
        else {
            return Ok(None);
        };
        let length = u64::try_from(bytes.len()).map_err(|_| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        let end_offset = self.next_offset.checked_add(length).ok_or_else(|| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        self.hasher.update(&bytes);
        self.next_offset = end_offset;
        let digest = self.hasher.digest();
        {
            let mut partitions = self.partitions.borrow_mut();
            if end_offset >= self.size_bytes {
                partitions.open.remove(&self.position.get());
                partitions.completed.insert(self.position.get(), digest);
            } else {
                partitions.open.insert(
                    self.position.get(),
                    S3OpenPartition {
                        position: self.position,
                        identity: self.identity,
                        next_byte_offset: end_offset,
                        prefix_digest: digest,
                    },
                );
            }
        }
        let lease = budget.acquire_memory(1, bytes.len()).await?;
        let chunk = BudgetedSourceChunk::new(bytes, lease)?;
        Ok(Some(SequentialSourceChunk::new(chunk, end_offset, digest)))
    }
}

/// Bounded immutable range authority over one frozen generation.
struct S3RangeReader {
    content: S3ObjectContent,
}

#[async_trait(?Send)]
impl StreamingRangeReader for S3RangeReader {
    async fn read_range(
        &self,
        offset: u64,
        length: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<BudgetedSourceChunk, StreamSourceError> {
        let length_u64 = u64::try_from(length.get()).map_err(|_| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        let end = offset.checked_add(length_u64).ok_or_else(|| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        let body = self
            .content
            .conditional_get(Some(S3ByteRange { offset, end }))
            .await?;
        let bytes = read_all(body, length.get())
            .await
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        if bytes.len() != length.get() {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::InvalidChunk,
            ));
        }
        let lease = budget.acquire_memory(1, bytes.len()).await?;
        BudgetedSourceChunk::new(Bytes::from(bytes), lease)
    }
}

// ---------------------------------------------------------------------------
// Checkpoint participant
// ---------------------------------------------------------------------------

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for S3Source {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let cursor = self.encode_cursor();
        let bytes = serde_json::to_vec(&cursor)
            .map(|encoded| Bytes::from(encoded.into_boxed_slice()))
            .map_err(|_| CheckpointError::ObjectVerification)?;
        if bytes.len() > MAX_CURSOR_PAYLOAD_BYTES {
            // Payload growth means the unsealed set outgrew its authored bound;
            // fail closed rather than commit an unbounded participant object.
            return Err(CheckpointError::StateBudget {
                participant: self.participant_id.clone(),
                code: StateBudgetFailureCode::ByteCapacity,
            });
        }
        let lease = self
            .checkpoint_budget
            .acquire(1, bytes.len())
            .now_or_never()
            .ok_or_else(|| CheckpointError::StateBudget {
                participant: self.participant_id.clone(),
                code: StateBudgetFailureCode::ItemCapacity,
            })?
            .map_err(|error| match error {
                BudgetError::Closed => CheckpointError::ParticipantUnavailable {
                    participant: self.participant_id.clone(),
                },
                _ => CheckpointError::StateBudget {
                    participant: self.participant_id.clone(),
                    code: StateBudgetFailureCode::ByteCapacity,
                },
            })?;
        let payload = BudgetedCheckpointBytes::new(bytes, lease)?;
        PreparedParticipantState::new(
            barrier.run,
            self.participant_id.clone(),
            S3_CURSOR_SCHEMA_ID,
            S3_CURSOR_SCHEMA_VERSION,
            barrier.cut.clone(),
            1,
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
        let descriptor = state.descriptor();
        if descriptor.participant_id != self.participant_id
            || descriptor.schema_id != S3_CURSOR_SCHEMA_ID
            || descriptor.schema_version != S3_CURSOR_SCHEMA_VERSION
            || descriptor.item_count != 1
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let cursor: S3SourceCursor = serde_json::from_slice(state.payload_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        // The resume must target the same authored source; a different snapshot
        // is never silently re-planned.
        if cursor.snapshot_digest != self.discovery.policy.snapshot_digest() {
            return Err(CheckpointError::ObjectVerification);
        }
        self.restore_cursor(cursor);
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.participant_id() != &self.participant_id {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        Ok(())
    }
}
