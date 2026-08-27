// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavior tests for the Baseten Parquet streaming decoder.
//!
//! These live inside the crate because the Arrow and Parquet readers are
//! normal (feature-gated) dependencies rather than dev dependencies, so an
//! integration-test binary could not write the immutable Parquet fixtures the
//! decoder consumes without adding them a second time.
//!
//! Every test drives the real acquisition seam: a fixture is written to bytes,
//! bound to an [`AcquiredPartition`] through a no-follow seekable-local
//! snapshot, and decoded through the registered factory.

use std::num::NonZeroUsize;

use arrow::array::{Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use serde_json::value::RawValue;

use super::*;
use crate::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    failure::{StableStreamingFailure, StreamingIssueReportError, StreamingIssueReportStatus},
    identity::LogicalReplayRunId,
    reliability::StreamingIssueReporterEndpoint,
    source::{
        AcquiredPartition, BudgetedSourceChunk, StreamSourceError, StreamingResumeGranularity,
        StreamingSeekableLocalSnapshot, StreamingSourceMode, StreamingSourceOrdering,
        StreamingSourcePlacement, StreamingSourceRetention,
    },
};

// ---------------------------------------------------------------------------
// Pure projection tests
// ---------------------------------------------------------------------------

fn config() -> BasetenFormatConfig {
    BasetenFormatConfig {
        session_column: BasetenSessionColumn::Provided,
        omit_kv_hints: false,
        force_min_tokens: false,
        max_output_tokens: None,
        session_sample_ratio: None,
        max_row_group_bytes: 1 << 20,
        max_prompt_bytes: 1 << 16,
    }
}

fn projected_row() -> BasetenRowProjection {
    BasetenRowProjection {
        row_in_group: 0,
        absolute_row: 0,
        timestamp_start_unix_ms: 1_750_000_000_123,
        prompt: "hello".to_owned(),
        input_tokens: 812,
        output_tokens: 128,
        duration_e2e_ms: Some(842.5),
        duration_ttft_ms: Some(91.0),
        cached_tokens_reference: Some(768),
        total_hashes: vec![11, 12, 13],
        block_size: Some(16),
        session_id: Some("s-1".to_owned()),
    }
}

#[test]
fn cursor_round_trips_byte_for_byte() {
    let cursor = BasetenCursor {
        row_group: 3,
        row_in_group: 17,
        rows_emitted: 4_096,
        schema_digest: ContentDigest::from_bytes([0x5a; 32]),
    };
    let encoded = cursor.encode();
    assert_eq!(encoded.len(), CURSOR_BYTES);
    assert_eq!(BasetenCursor::decode(&encoded), Ok(cursor));
    let mut smuggled = encoded;
    smuggled[2] = 1;
    assert!(
        BasetenCursor::decode(&smuggled).is_err(),
        "reserved cursor bytes cannot smuggle an unset field"
    );
}

#[test]
fn replay_parameters_are_canonical_and_omit_absent_fields() {
    let payload = canonical_replay_parameters(&projected_row(), &config());
    assert_eq!(
        String::from_utf8(payload).as_deref(),
        Ok(
            "{\"block_size\":16,\"hash_ids\":[11,12,13],\"input_tokens\":812,\
             \"max_tokens\":128,\"recorded\":{\"cached_tokens_reference\":768,\
             \"duration_e2e_ms\":842.5,\"duration_ttft_ms\":91},\
             \"recorded_start_unix_ms\":1750000000123,\
             \"schema\":\"aiperf.baseten.replay-turn.v1\"}"
        )
    );

    let mut bare = projected_row();
    bare.total_hashes.clear();
    bare.block_size = None;
    bare.duration_e2e_ms = None;
    bare.duration_ttft_ms = None;
    bare.cached_tokens_reference = None;
    let payload = canonical_replay_parameters(&bare, &config());
    assert_eq!(
        String::from_utf8(payload).as_deref(),
        Ok(
            "{\"input_tokens\":812,\"max_tokens\":128,\
             \"recorded_start_unix_ms\":1750000000123,\
             \"schema\":\"aiperf.baseten.replay-turn.v1\"}"
        )
    );
}

#[test]
fn kv_hints_and_min_tokens_follow_the_authored_gates() {
    let mut authored = config();
    authored.omit_kv_hints = true;
    authored.force_min_tokens = true;
    authored.max_output_tokens = Some(64);
    let payload = canonical_replay_parameters(&projected_row(), &authored);
    let payload = String::from_utf8(payload).unwrap_or_default();
    assert!(!payload.contains("hash_ids"), "{payload}");
    assert!(!payload.contains("block_size"), "{payload}");
    assert!(payload.contains("\"max_tokens\":64"), "{payload}");
    assert!(payload.contains("\"min_tokens\":64"), "{payload}");
}

#[test]
fn output_length_floors_at_one_and_honors_the_cap() {
    assert_eq!(resolved_output_length(0, None), 1);
    assert_eq!(resolved_output_length(512, Some(64)), 64);
    assert_eq!(resolved_output_length(32, Some(64)), 32);
}

#[test]
fn session_column_falls_back_to_the_other_recorded_column() {
    use BasetenSessionColumn::{PoorMan, Provided};
    assert_eq!(
        resolve_session_column(Provided, true, true),
        Some(COL_SESSION)
    );
    assert_eq!(
        resolve_session_column(Provided, false, true),
        Some(COL_POOR_MAN_SESSION)
    );
    assert_eq!(
        resolve_session_column(PoorMan, true, false),
        Some(COL_SESSION)
    );
    assert_eq!(resolve_session_column(Provided, false, false), None);
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

static TEST_SOURCE_DESCRIPTOR: StreamingSourceDescriptor = StreamingSourceDescriptor {
    id: "test_local",
    description: "Test-local immutable snapshot source",
    modes: &[StreamingSourceMode::Finite],
    access: &[
        PartitionAccessKind::Sequential,
        PartitionAccessKind::SeekableLocal,
    ],
    ordering: StreamingSourceOrdering::Partition,
    resume: &[
        StreamingResumeGranularity::Partition,
        StreamingResumeGranularity::RowGroup,
    ],
    has_event_time: true,
    has_stable_record_ids: true,
    retention: StreamingSourceRetention::ResumeRootReachability,
    placement: StreamingSourcePlacement::ImmutablePartitionAssignment,
    supports_virtual_clock: true,
};

/// One authored fixture row.
struct FixtureRow {
    timestamp_start_unix_ms: i64,
    prompt: Option<String>,
    input_tokens: i64,
    output_tokens: i64,
    duration_e2e_ms: Option<f64>,
    session_id: Option<&'static str>,
}

impl FixtureRow {
    fn new(timestamp_start_unix_ms: i64, prompt: &str, session_id: Option<&'static str>) -> Self {
        Self {
            timestamp_start_unix_ms,
            prompt: Some(prompt.to_owned()),
            input_tokens: 12,
            output_tokens: 34,
            duration_e2e_ms: Some(100.5),
            session_id,
        }
    }
}

/// Write one immutable Parquet fixture with the requested row-group size.
///
/// `narrow_input_tokens` writes `input_tokens` as `Int32`, which is the
/// projected-schema drift the frozen schema digest must reject.
fn fixture_bytes(rows: &[FixtureRow], rows_per_group: usize, narrow_input_tokens: bool) -> Bytes {
    let input_type = if narrow_input_tokens {
        DataType::Int32
    } else {
        DataType::Int64
    };
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new(COL_TIME, DataType::Int64, false),
        Field::new(COL_PROMPT, DataType::Utf8, true),
        Field::new(COL_INPUT_TOKENS, input_type, false),
        Field::new(COL_OUTPUT_TOKENS, DataType::Int64, false),
        Field::new(COL_DURATION_E2E, DataType::Float64, true),
        Field::new(COL_SESSION, DataType::Utf8, true),
    ]));

    let timestamps = Int64Array::from(
        rows.iter()
            .map(|row| row.timestamp_start_unix_ms)
            .collect::<Vec<_>>(),
    );
    let prompts = StringArray::from(
        rows.iter()
            .map(|row| row.prompt.clone())
            .collect::<Vec<_>>(),
    );
    let input_tokens: std::sync::Arc<dyn arrow::array::Array> = if narrow_input_tokens {
        std::sync::Arc::new(Int32Array::from(
            rows.iter()
                .map(|row| row.input_tokens as i32)
                .collect::<Vec<_>>(),
        ))
    } else {
        std::sync::Arc::new(Int64Array::from(
            rows.iter().map(|row| row.input_tokens).collect::<Vec<_>>(),
        ))
    };
    let output_tokens =
        Int64Array::from(rows.iter().map(|row| row.output_tokens).collect::<Vec<_>>());
    let durations = Float64Array::from(
        rows.iter()
            .map(|row| row.duration_e2e_ms)
            .collect::<Vec<_>>(),
    );
    let sessions = StringArray::from(rows.iter().map(|row| row.session_id).collect::<Vec<_>>());

    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(timestamps),
            std::sync::Arc::new(prompts),
            input_tokens,
            std::sync::Arc::new(output_tokens),
            std::sync::Arc::new(durations),
            std::sync::Arc::new(sessions),
        ],
    )
    .unwrap_or_else(|error| panic!("fixture batch: {error}"));

    let properties = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_size(rows_per_group)
        .build();
    let mut buffer = Vec::new();
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(&mut buffer, schema, Some(properties))
            .unwrap_or_else(|error| panic!("fixture writer: {error}"));
    writer
        .write(&batch)
        .unwrap_or_else(|error| panic!("fixture write: {error}"));
    writer
        .close()
        .unwrap_or_else(|error| panic!("fixture close: {error}"));
    Bytes::from(buffer)
}

/// In-memory no-follow snapshot standing in for an acquired local object.
struct MemorySnapshot {
    bytes: Bytes,
    reads: Rc<Cell<u64>>,
}

#[async_trait(?Send)]
impl StreamingSeekableLocalSnapshot for MemorySnapshot {
    async fn read_at(
        &self,
        offset: u64,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<BudgetedSourceChunk, StreamSourceError> {
        self.reads.set(self.reads.get().saturating_add(1));
        let start = usize::try_from(offset).unwrap_or(self.bytes.len());
        let start = start.min(self.bytes.len());
        let end = start.saturating_add(max_bytes.get()).min(self.bytes.len());
        let slice = self.bytes.slice(start..end);
        let lease = budget.acquire_memory(1, slice.len()).await?;
        BudgetedSourceChunk::new(slice, lease)
    }
}

async fn acquire(
    bytes: Bytes,
    position: u64,
    identity: ImmutableObjectIdentity,
    budget: &AcquisitionBudget,
    reads: Rc<Cell<u64>>,
) -> AcquiredPartition {
    let size = bytes.len();
    let lease = budget
        .acquire_disk(1, size)
        .await
        .unwrap_or_else(|error| panic!("disk lease: {error}"));
    AcquiredPartition::seekable_local(
        SourcePosition::new(position),
        identity,
        size as u64,
        Box::new(MemorySnapshot { bytes, reads }),
        lease,
    )
    .unwrap_or_else(|error| panic!("acquired partition: {error}"))
}

#[derive(Default)]
struct CountingEndpoint {
    accepted: Rc<Cell<u64>>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for CountingEndpoint {
    async fn report(
        &self,
        _issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        self.accepted.set(self.accepted.get().saturating_add(1));
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

struct Harness {
    format: Box<dyn StreamingDatasetFormat>,
    fragment_budget: StreamingResourceBudget,
    acquisition_budget: AcquisitionBudget,
    issues: Rc<Cell<u64>>,
    reads: Rc<Cell<u64>>,
}

fn authored(value: serde_json::Value) -> Box<RawValue> {
    RawValue::from_string(value.to_string()).unwrap_or_else(|error| panic!("authored: {error}"))
}

fn harness(value: serde_json::Value, fragment_items: usize) -> Harness {
    let fragment_budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: fragment_items,
        max_bytes: 1 << 20,
    })
    .unwrap_or_else(|error| panic!("fragment budget: {error}"));
    let acquisition_budget = AcquisitionBudget::new(
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 8,
            max_bytes: 1 << 22,
        })
        .unwrap_or_else(|error| panic!("memory budget: {error}")),
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 8,
            max_bytes: 1 << 22,
        })
        .unwrap_or_else(|error| panic!("disk budget: {error}")),
    );
    let issues = Rc::new(Cell::new(0));
    let reporter = StreamingIssueReporterHandle::new(CountingEndpoint {
        accepted: Rc::clone(&issues),
    });
    let factory = BasetenFormatFactory;
    let validated = factory
        .validate(&authored(value), &TEST_SOURCE_DESCRIPTOR)
        .unwrap_or_else(|error| panic!("authored configuration validates: {error}"));
    let context = StreamingFormatPrepareContext {
        run: StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x11; 32])),
        stream_semantic_digest: ContentDigest::from_bytes([0x71; 32]),
        issue_reporter: reporter,
        fragment_budget: fragment_budget.clone(),
        acquisition_budget: acquisition_budget.clone(),
    };
    let format = factory
        .prepare(validated, &context)
        .unwrap_or_else(|error| panic!("format preparation: {error}"));
    Harness {
        format,
        fragment_budget,
        acquisition_budget,
        issues,
        reads: Rc::new(Cell::new(0)),
    }
}

fn base_config() -> serde_json::Value {
    serde_json::json!({
        "max_row_group_bytes": 1 << 20,
        "max_prompt_bytes": 1 << 16,
    })
}

fn turn_content(fragment: &StreamingSessionFragment) -> String {
    match &fragment.mutation {
        SessionMutationV1::ConversationTurn(turn) => {
            assert_eq!(turn.role, "user");
            String::from_utf8(turn.content.clone()).unwrap_or_default()
        }
        other => panic!("expected a conversation turn, got {other:?}"),
    }
}

fn event_payload(fragment: &StreamingSessionFragment) -> String {
    match &fragment.mutation {
        SessionMutationV1::AgentEvent(event) => {
            assert_eq!(event.event_kind, BASETEN_REPLAY_TURN_EVENT_KIND);
            String::from_utf8(event.payload.clone()).unwrap_or_default()
        }
        other => panic!("expected an agent event, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Decode-path tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rows_become_causal_turn_and_replay_parameter_pairs() {
    let mut harness = harness(base_config(), 64);
    let bytes = fixture_bytes(
        &[
            FixtureRow::new(1_750_000_000_000, "first prompt", Some("s-1")),
            FixtureRow::new(1_750_000_000_500, "second prompt", Some("s-1")),
        ],
        128,
        false,
    );
    let partition = acquire(
        bytes,
        0,
        ImmutableObjectIdentity::from_bytes([0x21; 32]),
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;

    let mut decoder = harness
        .format
        .begin_partition(partition, None)
        .await
        .unwrap_or_else(|error| panic!("decoder begins: {error}"));
    let batch = match decoder
        .next_batch(DecodeBatchBudget {
            max_fragments: 8,
            max_bytes: 1 << 16,
        })
        .await
        .unwrap_or_else(|error| panic!("first pull: {error}"))
    {
        DecodeStep::Batch(batch) => batch,
        DecodeStep::End(_) => panic!("a nonempty fixture yields a batch first"),
    };

    assert_eq!(batch.fragments.len(), 4, "two rows emit two causal pairs");
    assert_eq!(turn_content(&batch.fragments[0]), "first prompt");
    assert_eq!(turn_content(&batch.fragments[2]), "second prompt");
    assert!(
        event_payload(&batch.fragments[1]).contains("\"input_tokens\":12"),
        "the replay-parameter payload carries the recorded ISL"
    );
    assert!(
        event_payload(&batch.fragments[1]).contains("\"duration_e2e_ms\":100.5"),
        "the recorded outcome is carried verbatim"
    );
    assert_eq!(
        batch.fragments[1].predecessors.as_slice(),
        &[batch.fragments[0].record_id],
        "the parameter fragment names its turn as a predecessor"
    );
    assert_eq!(
        batch.fragments[0].event_time.map(EventTimeUtc::get),
        Some(1_750_000_000_000 * NANOS_PER_MILLI),
        "event time is the absolute recorded request start"
    );
    assert_eq!(
        batch.fragments[0].session_key, batch.fragments[2].session_key,
        "rows sharing a recorded session id share a stable session key"
    );

    drop(batch);
    let receipt = match decoder
        .next_batch(DecodeBatchBudget {
            max_fragments: 8,
            max_bytes: 1 << 16,
        })
        .await
        .unwrap_or_else(|error| panic!("second pull: {error}"))
    {
        DecodeStep::End(receipt) => receipt,
        DecodeStep::Batch(_) => panic!("the fixture is exhausted after one batch"),
    };
    assert_eq!(receipt.fragment_count, 4);
    assert_eq!(harness.issues.get(), 0);
}

#[tokio::test]
async fn one_session_joins_across_three_shards() {
    let mut harness = harness(base_config(), 64);
    let mut keys = Vec::new();
    for shard in 0..3_u8 {
        let bytes = fixture_bytes(
            &[FixtureRow::new(
                1_750_000_000_000 + i64::from(shard),
                "shard prompt",
                Some("s-shared"),
            )],
            128,
            false,
        );
        let partition = acquire(
            bytes,
            u64::from(shard),
            ImmutableObjectIdentity::from_bytes([0x30 + shard; 32]),
            &harness.acquisition_budget,
            Rc::clone(&harness.reads),
        )
        .await;
        let mut decoder = harness
            .format
            .begin_partition(partition, None)
            .await
            .unwrap_or_else(|error| panic!("decoder begins: {error}"));
        let batch = match decoder
            .next_batch(DecodeBatchBudget {
                max_fragments: 8,
                max_bytes: 1 << 16,
            })
            .await
            .unwrap_or_else(|error| panic!("shard pull: {error}"))
        {
            DecodeStep::Batch(batch) => batch,
            DecodeStep::End(_) => panic!("each shard yields one pair"),
        };
        keys.push(batch.fragments[0].session_key);
        assert_ne!(
            batch.fragments[0].source_partition,
            ImmutableObjectIdentity::from_bytes([0; 32])
        );
    }
    assert_eq!(keys[0], keys[1]);
    assert_eq!(keys[1], keys[2]);
}

#[tokio::test]
async fn blocked_output_parks_until_the_batch_is_released() {
    // Exactly one pair plus one cursor fits, so the second pull cannot make
    // progress until the issued batch releases its charge.
    let mut harness = harness(base_config(), 3);
    let bytes = fixture_bytes(
        &[FixtureRow::new(1_750_000_000_000, "only prompt", Some("s-1"))],
        128,
        false,
    );
    let partition = acquire(
        bytes,
        0,
        ImmutableObjectIdentity::from_bytes([0x21; 32]),
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;
    let mut decoder = harness
        .format
        .begin_partition(partition, None)
        .await
        .unwrap_or_else(|error| panic!("decoder begins: {error}"));
    let budget = DecodeBatchBudget {
        max_fragments: 8,
        max_bytes: 1 << 16,
    };
    let batch = match decoder
        .next_batch(budget)
        .await
        .unwrap_or_else(|error| panic!("first pull: {error}"))
    {
        DecodeStep::Batch(batch) => batch,
        DecodeStep::End(_) => panic!("a nonempty fixture yields a batch first"),
    };
    assert!(harness.fragment_budget.snapshot().used_items > 0);
    let reads_before = harness.reads.get();

    {
        let blocked = decoder.next_batch(budget);
        tokio::pin!(blocked);
        assert!(
            futures::poll!(&mut blocked).is_pending(),
            "a decoder parks rather than exceeding the budget its output holds"
        );
    }
    assert_eq!(
        harness.reads.get(),
        reads_before,
        "a parked decoder reads no further source bytes"
    );

    drop(batch);
    assert_eq!(harness.fragment_budget.snapshot().used_items, 0);
    let receipt = match decoder
        .next_batch(budget)
        .await
        .unwrap_or_else(|error| panic!("resumed pull: {error}"))
    {
        DecodeStep::End(receipt) => receipt,
        DecodeStep::Batch(_) => panic!("the fixture is exhausted"),
    };
    assert_eq!(receipt.fragment_count, 2);
}

#[tokio::test]
async fn cursor_restore_resumes_at_the_next_row() {
    let mut harness = harness(base_config(), 64);
    let rows = [
        FixtureRow::new(1_750_000_000_000, "row zero", Some("s-1")),
        FixtureRow::new(1_750_000_000_100, "row one", Some("s-1")),
    ];
    let identity = ImmutableObjectIdentity::from_bytes([0x21; 32]);
    let bytes = fixture_bytes(&rows, 1, false);
    let partition = acquire(
        bytes.clone(),
        0,
        identity,
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;

    let mut decoder = harness
        .format
        .begin_partition(partition, None)
        .await
        .unwrap_or_else(|error| panic!("decoder begins: {error}"));
    let batch = match decoder
        .next_batch(DecodeBatchBudget {
            max_fragments: 2,
            max_bytes: 1 << 16,
        })
        .await
        .unwrap_or_else(|error| panic!("first pull: {error}"))
    {
        DecodeStep::Batch(batch) => batch,
        DecodeStep::End(_) => panic!("a nonempty fixture yields a batch first"),
    };
    assert_eq!(batch.fragments.len(), 2);
    assert_eq!(turn_content(&batch.fragments[0]), "row zero");
    let first_ids: Vec<StableRecordId> = batch
        .fragments
        .iter()
        .map(|fragment| fragment.record_id)
        .collect();
    let cursor = batch.resume_after.as_bytes().to_vec();
    drop(batch);
    drop(decoder);

    let lease = harness
        .fragment_budget
        .acquire(1, cursor.len())
        .await
        .unwrap_or_else(|error| panic!("cursor lease: {error}"));
    let state = DecoderResumeState::new(Bytes::from(cursor.clone()), lease)
        .unwrap_or_else(|error| panic!("cursor state: {error}"));
    let resumed_partition = acquire(
        bytes,
        0,
        identity,
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;
    let mut resumed = harness
        .format
        .begin_partition(
            resumed_partition,
            Some(DecoderCheckpoint {
                partition: identity,
                format_semantic_digest: BASETEN_SEMANTIC_DIGEST,
                state,
            }),
        )
        .await
        .unwrap_or_else(|error| panic!("decoder resumes: {error}"));
    assert_eq!(
        resumed
            .resume_state()
            .unwrap_or_else(|error| panic!("resumed cursor: {error}"))
            .as_bytes(),
        cursor.as_slice(),
        "resumption restores the exact retained cursor byte for byte"
    );

    let batch = match resumed
        .next_batch(DecodeBatchBudget {
            max_fragments: 2,
            max_bytes: 1 << 16,
        })
        .await
        .unwrap_or_else(|error| panic!("resumed pull: {error}"))
    {
        DecodeStep::Batch(batch) => batch,
        DecodeStep::End(_) => panic!("the successor row is still pending"),
    };
    assert_eq!(turn_content(&batch.fragments[0]), "row one");
    for fragment in &batch.fragments {
        assert!(
            !first_ids.contains(&fragment.record_id),
            "resumption neither duplicates nor skips a record"
        );
    }
}

#[tokio::test]
async fn oversized_row_group_is_refused_before_allocation() {
    let mut harness = harness(
        serde_json::json!({
            "max_row_group_bytes": 4_000,
            "max_prompt_bytes": 1 << 16,
        }),
        64,
    );
    let prompt = "x".repeat(8_000);
    let bytes = fixture_bytes(
        &[FixtureRow::new(1_750_000_000_000, &prompt, Some("s-1"))],
        128,
        false,
    );
    let partition = acquire(
        bytes,
        0,
        ImmutableObjectIdentity::from_bytes([0x21; 32]),
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;
    let mut decoder = harness
        .format
        .begin_partition(partition, None)
        .await
        .unwrap_or_else(|error| panic!("footer fits the authored bound: {error}"));
    let error = decoder
        .next_batch(DecodeBatchBudget {
            max_fragments: 8,
            max_bytes: 1 << 16,
        })
        .await
        .err()
        .expect("an oversized row group is refused");
    assert_eq!(error.code(), "oversized_record");
}

#[tokio::test]
async fn bad_row_quarantines_only_its_record() {
    let mut harness = harness(base_config(), 64);
    let mut invalid = FixtureRow::new(1_750_000_000_100, "unused", Some("s-1"));
    invalid.prompt = None;
    let bytes = fixture_bytes(
        &[
            FixtureRow::new(1_750_000_000_000, "good zero", Some("s-1")),
            invalid,
            FixtureRow::new(1_750_000_000_200, "good two", Some("s-1")),
        ],
        128,
        false,
    );
    let partition = acquire(
        bytes,
        0,
        ImmutableObjectIdentity::from_bytes([0x21; 32]),
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;
    let mut decoder = harness
        .format
        .begin_partition(partition, None)
        .await
        .unwrap_or_else(|error| panic!("decoder begins: {error}"));
    let batch = match decoder
        .next_batch(DecodeBatchBudget {
            max_fragments: 16,
            max_bytes: 1 << 16,
        })
        .await
        .unwrap_or_else(|error| panic!("first pull: {error}"))
    {
        DecodeStep::Batch(batch) => batch,
        DecodeStep::End(_) => panic!("the valid neighbours still decode"),
    };
    assert_eq!(batch.fragments.len(), 4, "only the invalid row is excluded");
    assert_eq!(turn_content(&batch.fragments[0]), "good zero");
    assert_eq!(turn_content(&batch.fragments[2]), "good two");
    assert_eq!(harness.issues.get(), 1, "exactly one record-scoped receipt");
}

#[tokio::test]
async fn projected_schema_drift_is_terminal_before_fragment_output() {
    let mut harness = harness(base_config(), 64);
    let first = acquire(
        fixture_bytes(
            &[FixtureRow::new(1_750_000_000_000, "row", Some("s-1"))],
            128,
            false,
        ),
        0,
        ImmutableObjectIdentity::from_bytes([0x21; 32]),
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;
    harness
        .format
        .begin_partition(first, None)
        .await
        .unwrap_or_else(|error| panic!("first partition freezes the digest: {error}"));

    let drifted = acquire(
        fixture_bytes(
            &[FixtureRow::new(1_750_000_000_000, "row", Some("s-1"))],
            128,
            true,
        ),
        1,
        ImmutableObjectIdentity::from_bytes([0x22; 32]),
        &harness.acquisition_budget,
        Rc::clone(&harness.reads),
    )
    .await;
    let error = harness
        .format
        .begin_partition(drifted, None)
        .await
        .err()
        .expect("drifted projected schema is refused before any fragment");
    assert_eq!(error.code(), "schema");
    assert_eq!(
        harness.issues.get(),
        0,
        "drift returns the typed refusal; partition scope has no format-owned receipt"
    );
}

#[tokio::test]
async fn session_sample_ratio_is_a_stable_whole_session_admission() {
    for (ratio, expected_fragments) in [(0.0_f64, 0_usize), (1.0, 4)] {
        let mut harness = harness(
            serde_json::json!({
                "max_row_group_bytes": 1 << 20,
                "max_prompt_bytes": 1 << 16,
                "session_sample_ratio": ratio,
            }),
            64,
        );
        let bytes = fixture_bytes(
            &[
                FixtureRow::new(1_750_000_000_000, "zero", Some("s-1")),
                FixtureRow::new(1_750_000_000_100, "one", Some("s-1")),
            ],
            128,
            false,
        );
        let partition = acquire(
            bytes,
            0,
            ImmutableObjectIdentity::from_bytes([0x21; 32]),
            &harness.acquisition_budget,
            Rc::clone(&harness.reads),
        )
        .await;
        let mut decoder = harness
            .format
            .begin_partition(partition, None)
            .await
            .unwrap_or_else(|error| panic!("decoder begins: {error}"));
        let observed = match decoder
            .next_batch(DecodeBatchBudget {
                max_fragments: 16,
                max_bytes: 1 << 16,
            })
            .await
            .unwrap_or_else(|error| panic!("sampled pull: {error}"))
        {
            DecodeStep::Batch(batch) => batch.fragments.len(),
            DecodeStep::End(receipt) => {
                assert_eq!(receipt.fragment_count, 0);
                0
            }
        };
        assert_eq!(observed, expected_fragments, "ratio {ratio}");
    }
}

#[test]
fn validation_refuses_unknown_fields_and_incompatible_access() {
    let factory = BasetenFormatFactory;
    assert!(
        factory
            .validate(&authored(base_config()), &TEST_SOURCE_DESCRIPTOR)
            .is_ok()
    );
    assert!(
        factory
            .validate(
                &authored(serde_json::json!({
                    "max_row_group_bytes": 1 << 20,
                    "max_prompt_bytes": 1 << 16,
                    "extra": true,
                })),
                &TEST_SOURCE_DESCRIPTOR,
            )
            .is_err(),
        "unknown configuration fields are refused before preparation"
    );
    assert!(
        factory
            .validate(
                &authored(serde_json::json!({
                    "max_row_group_bytes": 1 << 20,
                    "max_prompt_bytes": 1 << 16,
                    "session_sample_ratio": 1.5,
                })),
                &TEST_SOURCE_DESCRIPTOR,
            )
            .is_err(),
        "an out-of-range admission fraction is refused"
    );

    static SEQUENTIAL_ONLY: StreamingSourceDescriptor = StreamingSourceDescriptor {
        id: "test_sequential_only",
        description: "Test-local sequential-only source",
        modes: &[StreamingSourceMode::Finite],
        access: &[PartitionAccessKind::Sequential],
        ordering: StreamingSourceOrdering::Partition,
        resume: &[StreamingResumeGranularity::Partition],
        has_event_time: true,
        has_stable_record_ids: true,
        retention: StreamingSourceRetention::BoundedMemory,
        placement: StreamingSourcePlacement::ControllerOnly,
        supports_virtual_clock: true,
    };
    assert!(
        factory
            .validate(&authored(base_config()), &SEQUENTIAL_ONLY)
            .is_err(),
        "a source that cannot serve seekable local access is refused"
    );
}
