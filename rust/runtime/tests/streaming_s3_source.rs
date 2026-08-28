// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Socket-free product tests for the native S3 streaming source.
//!
//! Every case drives the real source policy — reconciliation, pagination
//! bounds, generation identity, conditional acquisition, clocked retry, and the
//! checkpoint cursor — against an in-memory [`S3Client`]. No AWS type, network
//! socket, or credential is involved, which is exactly what the narrow client
//! seam exists to make possible.

#![cfg(feature = "streaming-s3")]

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroUsize;
use std::rc::Rc;
use std::task::Poll;

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::streaming::budget::{BudgetLimits, StreamingResourceBudget};
use aiperf_runtime::streaming::checkpoint::{
    AcquisitionHorizon, AdmissionHorizon, CheckpointBarrier, CheckpointCut, CheckpointEpoch,
    DecodeHorizon, DiscoveryHorizon, EventTimeWatermark, OrderedActionHorizon, StreamRunIdentity,
    StreamingCheckpointParticipant, TerminalActionHorizon,
};
use aiperf_runtime::streaming::identity::{
    ContentDigest, GlobalSequence, ImmutableObjectIdentity, LogicalReplayRunId,
    SessionCausalFrontier,
};
use aiperf_runtime::streaming::reliability::{
    HandledIssueCut, OrdinaryStreamingIssue, StreamingIssueReportError, StreamingIssueReportStatus,
    StreamingIssueReporterEndpoint, StreamingIssueReporterHandle,
};
use aiperf_runtime::streaming::source::{
    AcquiredPartitionAccess, AcquisitionBudget, PartitionAccessRequest, SourceEvent, SourceSeal,
    StreamSourceError, StreamingDatasetSource, StreamingSourceControl, streaming_stop_channel,
};
use aiperf_runtime::streaming::sources::s3::{
    LosslessFrontierProof, PreparedS3Policy, S3GenerationToken, S3Source, S3SourceConfig,
    SourceFidelity, S3PolicyError, validate_s3_policy,
};
use aiperf_runtime::streaming::sources::s3_client::{
    S3Client, S3ClientError, S3GetRequest, S3ListPage, S3ListRequest, S3ListedObject, S3ObjectBody,
    S3ObjectReader,
};
use aiperf_runtime::streaming::unit::{EventTimeUtc, SourcePosition};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::json;
use serde_json::value::RawValue;

// ---------------------------------------------------------------------------
// In-memory S3 provider
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FakeGeneration {
    body: Vec<u8>,
    etag: Option<String>,
    version: Option<String>,
}

#[derive(Default)]
struct FakeState {
    current: BTreeMap<String, FakeGeneration>,
    versions: BTreeMap<(String, String), Vec<u8>>,
    page_size: u16,
    get_failures: VecDeque<S3ClientError>,
    get_calls: usize,
    invalidations: usize,
}

/// Deterministic in-memory S3 provider.
///
/// The fake models a versioned inventory: a listing may report a version id,
/// which a real `ListObjectsV2` cannot, because the source treats the listed
/// version as the generation token regardless of which provider surface
/// supplied it.
#[derive(Default)]
struct FakeS3Client {
    state: RefCell<FakeState>,
}

impl std::fmt::Debug for FakeS3Client {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("FakeS3Client").finish()
    }
}

impl FakeS3Client {
    fn new() -> Rc<Self> {
        Rc::new(Self {
            state: RefCell::new(FakeState {
                page_size: 1_000,
                ..FakeState::default()
            }),
        })
    }

    fn with_page_size(page_size: u16) -> Rc<Self> {
        let client = Self::new();
        client.state.borrow_mut().page_size = page_size;
        client
    }

    fn put_versioned(&self, key: &str, body: &[u8], version: &str) {
        let mut state = self.state.borrow_mut();
        state.versions.insert(
            (key.to_owned(), version.to_owned()),
            body.to_vec(),
        );
        state.current.insert(
            key.to_owned(),
            FakeGeneration {
                body: body.to_vec(),
                etag: Some(format!("etag-{version}")),
                version: Some(version.to_owned()),
            },
        );
    }

    fn put_unversioned(&self, key: &str, body: &[u8], etag: &str) {
        self.state.borrow_mut().current.insert(
            key.to_owned(),
            FakeGeneration {
                body: body.to_vec(),
                etag: Some(etag.to_owned()),
                version: None,
            },
        );
    }

    fn delete(&self, key: &str) {
        let mut state = self.state.borrow_mut();
        state.current.remove(key);
        state.versions.retain(|(stored, _), _| stored != key);
    }

    fn seal_manifest(&self, keys: &[&str]) {
        let body = keys.join("\n").into_bytes();
        self.put_unversioned("_MANIFEST", &body, "etag-manifest");
    }

    fn fail_next_gets(&self, count: usize, error: S3ClientError) {
        let mut state = self.state.borrow_mut();
        for _ in 0..count {
            state.get_failures.push_back(error);
        }
    }

    fn invalidations(&self) -> usize {
        self.state.borrow().invalidations
    }

    fn get_calls(&self) -> usize {
        self.state.borrow().get_calls
    }
}

#[async_trait(?Send)]
impl S3Client for FakeS3Client {
    async fn list_page(&self, request: S3ListRequest) -> Result<S3ListPage, S3ClientError> {
        let state = self.state.borrow();
        let after = request
            .continuation_token
            .clone()
            .or_else(|| request.start_after.clone());
        let limit = usize::from(request.max_keys.min(state.page_size)).max(1);
        let mut objects = Vec::new();
        let mut next_continuation_token = None;
        for (key, generation) in &state.current {
            if let Some(prefix) = request.prefix.as_deref()
                && !key.starts_with(prefix)
            {
                continue;
            }
            if after.as_deref().is_some_and(|bound| key.as_str() <= bound) {
                continue;
            }
            if objects.len() == limit {
                next_continuation_token = objects
                    .last()
                    .map(|object: &S3ListedObject| object.key.clone());
                break;
            }
            objects.push(S3ListedObject {
                key: key.clone(),
                size_bytes: generation.body.len() as u64,
                etag: generation.etag.clone(),
                version_id: generation.version.clone(),
            });
        }
        Ok(S3ListPage {
            objects,
            next_continuation_token,
        })
    }

    async fn get_version(&self, request: S3GetRequest) -> Result<S3ObjectBody, S3ClientError> {
        let mut state = self.state.borrow_mut();
        state.get_calls += 1;
        if let Some(error) = state.get_failures.pop_front() {
            return Err(error);
        }
        let (body, etag, version) = if let Some(version) = request.version_id.as_deref() {
            let body = state
                .versions
                .get(&(request.key.clone(), version.to_owned()))
                .cloned()
                .ok_or(S3ClientError::NotFound)?;
            (
                body,
                Some(format!("etag-{version}")),
                Some(version.to_owned()),
            )
        } else {
            let generation = state
                .current
                .get(&request.key)
                .cloned()
                .ok_or(S3ClientError::NotFound)?;
            // S3 compares `If-Match` modulo the ETag's surrounding quotes.
            if let Some(expected) = request.if_match_etag.as_deref()
                && generation
                    .etag
                    .as_deref()
                    .map(|etag| etag.trim_matches('"'))
                    != Some(expected.trim_matches('"'))
            {
                return Err(S3ClientError::PreconditionFailed);
            }
            (generation.body, generation.etag, generation.version)
        };
        let served = match request.range {
            Some(range) => {
                let start = usize::try_from(range.offset).map_err(|_| S3ClientError::Malformed)?;
                let end = usize::try_from(range.end)
                    .map_err(|_| S3ClientError::Malformed)?
                    .min(body.len());
                body.get(start..end).unwrap_or_default().to_vec()
            }
            None => body,
        };
        Ok(S3ObjectBody {
            etag,
            version_id: version,
            content_length: Some(served.len() as u64),
            reader: Box::new(FakeReader {
                remaining: Bytes::from(served),
            }),
        })
    }

    fn invalidate_credentials(&self) {
        self.state.borrow_mut().invalidations += 1;
    }
}

struct FakeReader {
    remaining: Bytes,
}

#[async_trait(?Send)]
impl S3ObjectReader for FakeReader {
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
    ) -> Result<Option<Bytes>, S3ClientError> {
        if self.remaining.is_empty() {
            return Ok(None);
        }
        let take = max_bytes.get().min(self.remaining.len());
        Ok(Some(self.remaining.split_to(take)))
    }
}

// ---------------------------------------------------------------------------
// Reliability reporter
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ReporterState {
    accepted: RefCell<Vec<ContentDigest>>,
    is_closed: Cell<bool>,
}

struct RecordingEndpoint {
    state: Rc<ReporterState>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for RecordingEndpoint {
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        if self.state.is_closed.get() {
            return Err(StreamingIssueReportError::Closed);
        }
        self.state.accepted.borrow_mut().push(issue.issue_id());
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// How long a drain waits for a follow source to produce something new.
const QUIESCENCE: std::time::Duration = std::time::Duration::from_millis(40);

fn run_identity() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x5a; 32]))
}

fn acquisition_budget() -> AcquisitionBudget {
    let limits = BudgetLimits {
        max_items: 256,
        max_bytes: 1 << 20,
    };
    AcquisitionBudget::new(
        StreamingResourceBudget::new(limits).expect("valid memory limits"),
        StreamingResourceBudget::new(limits).expect("valid disk limits"),
    )
}

fn cursor_budget() -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 1 << 20,
    })
    .expect("valid cursor limits")
}

fn policy_from(value: serde_json::Value) -> PreparedS3Policy {
    let config: S3SourceConfig =
        serde_json::from_value(value).expect("authored S3 configuration decodes");
    validate_s3_policy(config).expect("authored S3 policy validates")
}

fn manifest_config() -> serde_json::Value {
    json!({
        "bucket": "bench",
        "policy": { "mode": "manifest", "manifest_suffix": "_MANIFEST", "is_finite": true },
        "page_max_keys": 1000,
        "max_pages_per_pass": 8,
        "max_unsealed_generations": 64,
        "max_attempts": 2,
        "base_backoff_ns": 1_000,
        "max_backoff_ns": 8_000,
        "poll_interval_ns": 1_000_000
    })
}

struct Harness {
    source: S3Source,
    control: StreamingSourceControl,
    reporter: Rc<ReporterState>,
}

fn harness(client: Rc<dyn S3Client>, policy: PreparedS3Policy) -> Harness {
    let reporter = Rc::new(ReporterState::default());
    let handle = StreamingIssueReporterHandle::new(RecordingEndpoint {
        state: Rc::clone(&reporter),
    });
    let (control, stop) = streaming_stop_channel();
    let source = S3Source::with_client(
        client,
        policy,
        run_identity(),
        handle,
        RealClock::new(),
        cursor_budget(),
        stop,
    );
    Harness {
        source,
        control,
        reporter,
    }
}

/// One drained discovery observation.
#[derive(Debug, Default)]
struct Observation {
    keys: Vec<String>,
    positions: Vec<SourcePosition>,
    identities: Vec<ImmutableObjectIdentity>,
    frontiers: Vec<SourcePosition>,
    seal: Option<SourceSeal>,
}

/// Drain every event the source produces until it is quiescent.
///
/// A follow source is never "done", so the loop ends when nothing new arrives
/// within [`QUIESCENCE`] — several authored poll intervals. Parking is the
/// source's honest "nothing more is ready", and it is never a seal.
async fn drain_ready(
    source: &mut S3Source,
    budget: &AcquisitionBudget,
) -> Result<Observation, StreamSourceError> {
    let mut observed = Observation::default();
    loop {
        let event = match tokio::time::timeout(QUIESCENCE, source.next_event()).await {
            Ok(result) => result?,
            Err(_) => break,
        };
        match event {
            SourceEvent::Partition(partition) => {
                observed.positions.push(partition.position());
                observed.identities.push(*partition.content().identity());
                let bytes = read_sequential(partition.content(), budget).await?;
                observed.keys.push(String::from_utf8_lossy(&bytes).into_owned());
            }
            SourceEvent::Frontier(frontier) => observed.frontiers.push(frontier.through),
            SourceEvent::Seal(seal) => {
                observed.seal = Some(seal);
                break;
            }
        }
    }
    Ok(observed)
}

async fn read_sequential(
    content: &dyn aiperf_runtime::streaming::source::SourcePartitionContent,
    budget: &AcquisitionBudget,
) -> Result<Vec<u8>, StreamSourceError> {
    let acquired = content
        .acquire(
            PartitionAccessRequest::Sequential { resume_offset: 0 },
            budget,
        )
        .await?;
    let AcquiredPartitionAccess::Sequential(mut reader) = acquired.into_access() else {
        panic!("the S3 source acquires a sequential reader for a sequential request");
    };
    let bound = NonZeroUsize::new(8).expect("non-zero chunk bound");
    let mut bytes = Vec::new();
    while let Some(chunk) = reader.next_chunk(bound, budget).await? {
        bytes.extend_from_slice(chunk.as_bytes());
    }
    Ok(bytes)
}

fn barrier() -> CheckpointBarrier {
    let event_time = EventTimeUtc::new(1).expect("non-negative event time");
    CheckpointBarrier {
        run: run_identity(),
        epoch: CheckpointEpoch::new(1),
        cut: CheckpointCut {
            discovered: DiscoveryHorizon::new(SourcePosition::new(1)),
            acquired: AcquisitionHorizon::new(SourcePosition::new(1)),
            decoded: DecodeHorizon::new(SourcePosition::new(1)),
            ordered: OrderedActionHorizon::new(GlobalSequence::new(1)),
            admitted: AdmissionHorizon::new(GlobalSequence::new(1)),
            terminal: TerminalActionHorizon::new(GlobalSequence::new(1)),
            handled_issues: HandledIssueCut::empty(),
            event_watermark: EventTimeWatermark::Hard {
                through: event_time,
            },
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(1),
                event_time: Some(event_time),
                digest: ContentDigest::from_bytes([0x71; 32]),
            },
        },
        plan_digest: ContentDigest::from_bytes([0x72; 32]),
    }
}

// ---------------------------------------------------------------------------
// Discovery, ordering, and reconciliation
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn notification_loss_and_late_key_are_recovered_before_interval_seal() {
    let client = FakeS3Client::new();
    client.put_versioned("0002", b"second", "v2");
    let mut harness = harness(Rc::clone(&client) as Rc<dyn S3Client>, policy_from(manifest_config()));
    let budget = acquisition_budget();

    let first = drain_ready(&mut harness.source, &budget)
        .await
        .expect("the first pass discovers the announced key");
    assert_eq!(first.keys, vec!["second".to_owned()]);
    assert!(
        first.frontiers.is_empty(),
        "an unsealed interval never advances a completeness frontier"
    );

    // The producer wrote `0001` without a notification. Reconciliation is the
    // sole discovery authority, so the next pass must find it.
    client.put_versioned("0001", b"first", "v1");
    client.seal_manifest(&["0001", "0002"]);

    let second = drain_ready(&mut harness.source, &budget)
        .await
        .expect("reconciliation recovers the late key");
    assert_eq!(second.keys, vec!["first".to_owned()]);
    // Positions are allocated at publication, so the lexicographically earlier
    // key lands at the *greater* position. The promise is that no frontier
    // advanced over an unseen key, not that positions follow key order.
    assert_eq!(second.positions, vec![SourcePosition::new(2)]);
    assert_eq!(second.frontiers, vec![SourcePosition::new(2)]);
    assert!(
        second.seal.is_some(),
        "the manifest is the seal authority and it is now satisfied"
    );

    harness.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn pagination_never_advances_the_frontier() {
    let client = FakeS3Client::with_page_size(2);
    for index in 0..5 {
        client.put_versioned(&format!("{index:04}"), b"body", &format!("v{index}"));
    }
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );
    let budget = acquisition_budget();

    let observed = drain_ready(&mut harness.source, &budget)
        .await
        .expect("a paginated pass publishes every listed generation");
    assert_eq!(observed.positions.len(), 5);
    assert!(
        observed.frontiers.is_empty() && observed.seal.is_none(),
        "a continuation token proves nothing about what a producer writes behind the cursor"
    );
    assert!(harness.source.high_water().list_page_items <= 2);
    assert!(harness.source.high_water().pages_this_pass >= 3);

    harness.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn unsealed_generation_bound_fails_closed() {
    let client = FakeS3Client::new();
    client.put_versioned("0001", b"a", "v1");
    client.put_versioned("0002", b"b", "v2");
    let mut config = manifest_config();
    config["max_unsealed_generations"] = json!(1);
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(config),
    );

    let error = drain_ready(&mut harness.source, &acquisition_budget())
        .await
        .expect_err("an unbounded unsealed prefix is refused rather than truncated");
    assert_eq!(
        error,
        StreamSourceError::source(
            aiperf_runtime::streaming::failure::SourceFailureCode::Discovery
        )
    );

    harness.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn hard_no_backfill_violation_fails_before_seal() {
    let client = FakeS3Client::new();
    client.put_versioned("0005", b"late", "v5");
    let config = json!({
        "bucket": "bench",
        "policy": {
            "mode": "interval_follow",
            "no_backfill_horizon_ns": 1_000_000_000_i64,
            "has_hard_no_backfill": true,
            "has_monotonic_keys": true
        },
        "page_max_keys": 1000,
        "max_pages_per_pass": 8,
        "max_unsealed_generations": 64,
        "max_attempts": 2,
        "base_backoff_ns": 1_000,
        "max_backoff_ns": 8_000,
        "poll_interval_ns": 1_000_000
    });
    let mut harness = harness(Rc::clone(&client) as Rc<dyn S3Client>, policy_from(config));
    let budget = acquisition_budget();

    let first = drain_ready(&mut harness.source, &budget)
        .await
        .expect("the monotonic key publishes normally");
    assert_eq!(first.positions.len(), 1);
    assert!(first.frontiers.is_empty());

    // `0003` lands behind the asserted publication order, which contradicts the
    // hard-no-backfill claim the lossless frontier rests on.
    client.put_versioned("0003", b"backfill", "v3");
    let error = drain_ready(&mut harness.source, &budget)
        .await
        .expect_err("a backfilled key refutes the authored no-backfill assertion");
    assert_eq!(
        error,
        StreamSourceError::source(
            aiperf_runtime::streaming::failure::SourceFailureCode::Discovery
        )
    );

    harness.control.stop();
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn pagination_and_identity_rules_are_explicit() {
    let client = FakeS3Client::with_page_size(2);
    client.put_versioned("0001", b"alpha", "version-1");
    client.put_unversioned("0002", b"beta", "\"d41d8cd9-3\"");
    client.seal_manifest(&["0001", "0002"]);
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );
    let budget = acquisition_budget();

    let observed = drain_ready(&mut harness.source, &budget)
        .await
        .expect("both generations are discovered and acquired");
    assert_eq!(observed.keys, vec!["alpha".to_owned(), "beta".to_owned()]);
    assert_eq!(
        observed.identities.len(),
        2,
        "each object generation is exactly one partition"
    );
    assert_ne!(observed.identities[0], observed.identities[1]);
    assert!(harness.source.high_water().list_page_items <= 2);

    let versioned = S3GenerationToken::classify(Some("version-1"), Some("etag-version-1"));
    assert_eq!(versioned.provider_version(), Some("version-1"));
    assert!(versioned.is_version_qualified());

    harness.control.stop();
}

#[test]
fn multipart_etag_is_never_a_content_digest() {
    let multipart = S3GenerationToken::classify(None, Some("\"9bb58f26192e4ba00f01e2e7b136bbd8-3\""));
    assert!(matches!(
        multipart,
        S3GenerationToken::MultipartETag { .. }
    ));
    assert!(multipart.provider_version().is_none());
    assert!(
        multipart.is_conditionally_bindable(),
        "a multipart ETag still binds a conditional read even though it hashes nothing"
    );

    let single = S3GenerationToken::classify(None, Some("\"d41d8cd98f00b204e9800998ecf8427e\""));
    assert!(matches!(single, S3GenerationToken::SinglePartETag { .. }));

    assert_eq!(S3GenerationToken::classify(None, None), S3GenerationToken::Absent);
    assert!(!S3GenerationToken::Absent.is_conditionally_bindable());
}

#[tokio::test(flavor = "current_thread")]
async fn versioned_identity_survives_overwrite() {
    let client = FakeS3Client::new();
    client.put_versioned("0001", b"original", "v1");
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );
    let budget = acquisition_budget();

    let first = {
        let next = harness.source.next_event();
        tokio::pin!(next);
        match futures::poll!(&mut next) {
            Poll::Ready(Ok(SourceEvent::Partition(partition))) => partition,
            _ => panic!("the first event is the discovered partition"),
        }
    };
    let frozen = *first.content().identity();

    // A later PUT mints a new version, so it is a different generation at a
    // later position; the frozen partition still reads its own bytes.
    client.put_versioned("0001", b"rewritten", "v2");
    let bytes = read_sequential(first.content(), &budget)
        .await
        .expect("the frozen version is still reachable after an overwrite");
    assert_eq!(bytes, b"original");

    let observed = drain_ready(&mut harness.source, &budget)
        .await
        .expect("the new version is discovered as a new partition");
    assert_eq!(observed.keys, vec!["rewritten".to_owned()]);
    assert_ne!(observed.identities[0], frozen);
    assert_eq!(observed.positions, vec![SourcePosition::new(2)]);

    harness.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn unversioned_overwrite_is_refused() {
    let client = FakeS3Client::new();
    client.put_unversioned("0001", b"original", "etag-a");
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );

    let partition = {
        let next = harness.source.next_event();
        tokio::pin!(next);
        match futures::poll!(&mut next) {
            Poll::Ready(Ok(SourceEvent::Partition(partition))) => partition,
            _ => panic!("the first event is the discovered partition"),
        }
    };

    client.put_unversioned("0001", b"rewritten", "etag-b");
    let error = read_sequential(partition.content(), &acquisition_budget())
        .await
        .expect_err("an If-Match read against a rewritten key is refused");
    assert_eq!(
        error,
        StreamSourceError::source(
            aiperf_runtime::streaming::failure::SourceFailureCode::MutatedObject
        )
    );
    assert!(
        harness.source.holes().is_empty(),
        "identity substitution is a refusal, never a hole"
    );

    harness.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn s3_identity_substitution_is_refused_not_holed() {
    let client = FakeS3Client::new();
    client.put_versioned("0001", b"original", "v1");
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );

    let partition = {
        let next = harness.source.next_event();
        tokio::pin!(next);
        match futures::poll!(&mut next) {
            Poll::Ready(Ok(SourceEvent::Partition(partition))) => partition,
            _ => panic!("the first event is the discovered partition"),
        }
    };

    // The pinned version is gone and only different bytes remain under the key.
    // Binding those bytes to the frozen identity is the substitution the source
    // must refuse; the fake reports the frozen version as absent.
    client.delete("0001");
    client.put_versioned("0001", b"substituted", "v2");
    let error = read_sequential(partition.content(), &acquisition_budget())
        .await
        .expect_err("the frozen generation cannot be satisfied by other bytes");
    assert_eq!(
        error,
        StreamSourceError::acquisition(
            aiperf_runtime::streaming::failure::AcquisitionFailureCode::Open
        ),
        "a vanished pinned version is an open failure, never a silent substitution"
    );
    assert_eq!(
        harness.reporter.accepted.borrow().len(),
        1,
        "exactly one partition-scoped fact is reported"
    );

    harness.control.stop();
}

// ---------------------------------------------------------------------------
// Reliability: holes, retry, and credential refresh
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn s3_retry_exhaustion_records_hole_and_reconciliation_continues() {
    let client = FakeS3Client::new();
    client.put_versioned("0001", b"first", "v1");
    client.put_versioned("0002", b"second", "v2");
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );
    let budget = acquisition_budget();

    let mut partitions = Vec::new();
    for _ in 0..2 {
        let next = harness.source.next_event();
        tokio::pin!(next);
        match futures::poll!(&mut next) {
            Poll::Ready(Ok(SourceEvent::Partition(partition))) => partitions.push(partition),
            _ => panic!("both generations are discovered"),
        }
    }

    // `max_attempts` is two, so two throttles exhaust the bounded retry.
    client.fail_next_gets(2, S3ClientError::Throttled);
    let error = read_sequential(partitions[0].content(), &budget)
        .await
        .expect_err("retry exhaustion surfaces an acquisition read failure");
    assert_eq!(
        error,
        StreamSourceError::acquisition(
            aiperf_runtime::streaming::failure::AcquisitionFailureCode::Read
        )
    );
    assert_eq!(client.get_calls(), 2, "the retry bound is honored exactly");
    assert_eq!(
        harness.source.holes(),
        vec![SourcePosition::new(1)],
        "the unreachable position enters the bounded exception set"
    );
    assert_eq!(harness.reporter.accepted.borrow().len(), 1);

    // The run is not failed: the next partition acquires normally.
    let bytes = read_sequential(partitions[1].content(), &budget)
        .await
        .expect("a later partition continues after a hole");
    assert_eq!(bytes, b"second");
    assert_eq!(harness.source.completed_digests().len(), 1);

    harness.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn deleted_object_after_listing_becomes_a_hole_not_a_refusal() {
    let client = FakeS3Client::new();
    client.put_versioned("0001", b"doomed", "v1");
    client.put_versioned("0002", b"survivor", "v2");
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );
    let budget = acquisition_budget();

    let mut partitions = Vec::new();
    for _ in 0..2 {
        let next = harness.source.next_event();
        tokio::pin!(next);
        match futures::poll!(&mut next) {
            Poll::Ready(Ok(SourceEvent::Partition(partition))) => partitions.push(partition),
            _ => panic!("both generations are discovered"),
        }
    }

    client.delete("0001");
    let error = read_sequential(partitions[0].content(), &budget)
        .await
        .expect_err("a deleted generation cannot be acquired");
    assert_eq!(
        error,
        StreamSourceError::acquisition(
            aiperf_runtime::streaming::failure::AcquisitionFailureCode::Open
        )
    );
    assert_eq!(harness.source.holes(), vec![SourcePosition::new(1)]);

    let bytes = read_sequential(partitions[1].content(), &budget)
        .await
        .expect("later partitions continue past the hole");
    assert_eq!(bytes, b"survivor");

    harness.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn credential_refresh_retry_keeps_the_frozen_identity() {
    let client = FakeS3Client::new();
    client.put_versioned("0001", b"guarded", "v1");
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(manifest_config()),
    );
    let budget = acquisition_budget();

    let partition = {
        let next = harness.source.next_event();
        tokio::pin!(next);
        match futures::poll!(&mut next) {
            Poll::Ready(Ok(SourceEvent::Partition(partition))) => partition,
            _ => panic!("the first event is the discovered partition"),
        }
    };
    let frozen = *partition.content().identity();

    client.fail_next_gets(1, S3ClientError::Unauthorized);
    let bytes = read_sequential(partition.content(), &budget)
        .await
        .expect("a refreshed credential acquires the same frozen generation");
    assert_eq!(bytes, b"guarded");
    assert_eq!(
        client.invalidations(),
        1,
        "an authorization failure invalidates the shared authority exactly once"
    );
    assert_eq!(
        *partition.content().identity(),
        frozen,
        "a credential refresh never changes the frozen object identity"
    );
    assert!(harness.reporter.accepted.borrow().is_empty());

    harness.control.stop();
}

// ---------------------------------------------------------------------------
// Policy fidelity
// ---------------------------------------------------------------------------

#[test]
fn lossless_and_lossy_policies_fail_or_label_honestly() {
    let mutable = json!({
        "bucket": "bench",
        "policy": {
            "mode": "interval_follow",
            "no_backfill_horizon_ns": 1_000_000_000_i64,
            "has_hard_no_backfill": false,
            "has_monotonic_keys": true
        },
        "page_max_keys": 1000,
        "max_pages_per_pass": 8,
        "max_unsealed_generations": 64,
        "max_attempts": 2,
        "base_backoff_ns": 1_000,
        "max_backoff_ns": 8_000,
        "poll_interval_ns": 1_000_000
    });
    let config: S3SourceConfig =
        serde_json::from_value(mutable).expect("authored configuration decodes");
    assert_eq!(
        validate_s3_policy(config).err(),
        Some(S3PolicyError::LosslessFrontierUnprovable {
            has_hard_no_backfill: false,
            has_monotonic_keys: true,
        })
    );

    let mut lossy = manifest_config();
    lossy["policy"] = json!({ "mode": "lossy_window", "max_keys": 128 });
    assert_eq!(
        policy_from(lossy).fidelity(),
        SourceFidelity::LossyWindow { max_keys: 128 },
        "a bounded rescan window is labelled lossy rather than silently degraded"
    );

    assert_eq!(
        policy_from(manifest_config()).fidelity(),
        SourceFidelity::Lossless {
            proof: LosslessFrontierProof::SealedManifest
        }
    );
}

#[test]
fn authored_configuration_is_strictly_decoded() {
    let mut unknown = manifest_config();
    unknown["surprise"] = json!(true);
    let config: Result<S3SourceConfig, _> = serde_json::from_value(unknown);
    assert!(config.is_err(), "unknown authored fields are refused");

    let mut zero_pages = manifest_config();
    zero_pages["max_pages_per_pass"] = json!(0);
    let config: S3SourceConfig =
        serde_json::from_value(zero_pages).expect("configuration decodes");
    assert_eq!(
        validate_s3_policy(config).err(),
        Some(S3PolicyError::UnboundedOrZeroLimit)
    );
}

// ---------------------------------------------------------------------------
// Checkpoint cursor
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_cursor_retains_exact_byte_offsets_and_no_secrets() {
    let client = FakeS3Client::new();
    client.put_versioned("0001", b"0123456789abcdef", "v1");
    let mut config = manifest_config();
    config["endpoint_url"] = json!("http://minio:sekrit-value@127.0.0.1:9000");
    let mut harness = harness(
        Rc::clone(&client) as Rc<dyn S3Client>,
        policy_from(config),
    );
    let budget = acquisition_budget();

    let partition = {
        let next = harness.source.next_event();
        tokio::pin!(next);
        match futures::poll!(&mut next) {
            Poll::Ready(Ok(SourceEvent::Partition(partition))) => partition,
            _ => panic!("the first event is the discovered partition"),
        }
    };

    // Read exactly one bounded chunk so the partition stays open mid-object.
    let acquired = partition
        .content()
        .acquire(
            PartitionAccessRequest::Sequential { resume_offset: 0 },
            &budget,
        )
        .await
        .expect("the frozen generation acquires");
    let AcquiredPartitionAccess::Sequential(mut reader) = acquired.into_access() else {
        panic!("a sequential request yields a sequential reader");
    };
    let bound = NonZeroUsize::new(4).expect("non-zero bound");
    let chunk = reader
        .next_chunk(bound, &budget)
        .await
        .expect("a bounded chunk reads")
        .expect("the object is not empty");
    assert_eq!(chunk.end_offset(), 4);
    assert_eq!(
        harness.source.open_partition_offset(SourcePosition::new(1)),
        Some(4),
        "the source retains the exact resume offset of an open partition"
    );

    harness
        .source
        .initialize(None)
        .await
        .expect("fresh participant initialization");
    let prepared = harness
        .source
        .checkpoint_view(&barrier())
        .await
        .expect("a non-destructive participant view");
    let payload = String::from_utf8(prepared.payload_bytes().to_vec())
        .expect("the cursor payload is canonical JSON text");
    assert!(
        payload.contains("\"next_byte_offset\":4"),
        "the cursor is byte-exact: {payload}"
    );
    assert!(
        !payload.contains("sekrit-value") && !payload.contains("127.0.0.1"),
        "no endpoint or credential material reaches the retained cursor"
    );
    let rendered = format!("{:?}", harness.source);
    assert!(
        !rendered.contains("sekrit-value"),
        "no credential material reaches the source Debug rendering"
    );

    harness.control.stop();
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn sim_clock_is_refused_at_prepare() {
    use aiperf_runtime::clock::SimClock;
    use aiperf_runtime::streaming::source::{
        StreamingDatasetSourceFactory, StreamingSourcePrepareContext,
    };
    use aiperf_runtime::streaming::sources::s3::S3SourceFactory;

    let factory = S3SourceFactory;
    assert_eq!(factory.descriptor().id, "s3");
    assert!(!factory.descriptor().supports_virtual_clock);

    let authored = RawValue::from_string(manifest_config().to_string())
        .expect("authored configuration is valid JSON");
    let reporter = Rc::new(ReporterState::default());
    let context = StreamingSourcePrepareContext {
        acquisition_budget: acquisition_budget(),
        issue_reporter: StreamingIssueReporterHandle::new(RecordingEndpoint { state: reporter }),
        clock: Rc::new(SimClock::new()),
    };
    let validated = factory
        .validate(&authored)
        .expect("authored configuration validates");
    let error = factory
        .prepare(validated, &context)
        .err()
        .expect("a virtual clock cannot sign a SigV4 request");
    assert_eq!(
        error,
        StreamSourceError::source(
            aiperf_runtime::streaming::failure::SourceFailureCode::SourceUnavailable
        )
    );
}
