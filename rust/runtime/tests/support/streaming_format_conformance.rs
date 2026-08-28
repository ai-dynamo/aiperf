// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable conformance assertions for `StreamingDatasetFormatFactory`.
//!
//! One implementation of the streaming format contract is exercised end to end:
//! strict configuration validation against the paired source descriptor, one
//! leased fragment batch, backpressure while an output lease is held, resumption
//! from the exact decoder cursor, duplicate replay determinism, explicit
//! partition exhaustion, frontier and seal translation, and idempotent
//! post-commit notification.
//!
//! The caller constructs the reliability reporter and moves it in. No adapter
//! owns it. Every borrow of the reporter is released before the next format,
//! decoder, or checkpoint `await`.
//!
//! Loaded by an integration test with
//! `#[path = "support/streaming_format_conformance.rs"] mod …;`.

use std::rc::Rc;

use aiperf_runtime::streaming::{
    budget::StreamingResourceBudget,
    checkpoint::StreamRunIdentity,
    failure::{StreamFormatError, StreamingIssueReporter},
    format::{
        DecodeBatchBudget, DecodeStep, DecoderCheckpoint, DecoderResumeState, FormatEvent,
        FormatEventSink, StreamingDatasetFormat, StreamingDatasetFormatFactory,
        StreamingFormatPrepareContext,
    },
    identity::{ContentDigest, ImmutableObjectIdentity, StableRecordId},
    source::{
        AcquiredPartition, AcquisitionBudget, SourceFrontier, SourceSeal, StreamingSourceDescriptor,
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::value::RawValue;

/// Decoder-scoped hook that releases exactly one parked decode step.
///
/// A conformant decoder parks rather than exceeding its caller-supplied budget
/// while an already-issued output lease is outstanding. The harness proves the
/// park, drops the lease, and then calls this hook so the scripted decoder can
/// finish. Decoders that never park supply a no-op.
pub type FormatAdvance = Rc<dyn Fn()>;

/// Everything one format implementation contributes to the shared harness.
pub struct FormatConformanceCases {
    /// Logical run bound into the prepare context.
    pub run: StreamRunIdentity,
    /// Strictly authored configuration the factory must accept.
    pub authored: Box<RawValue>,
    /// Authored configuration the factory must refuse before any effect.
    pub rejected_authored: Box<RawValue>,
    /// Source descriptor the format is validated against.
    pub source_descriptor: &'static StreamingSourceDescriptor,
    /// Semantic namespace bound into the prepare context.
    pub stream_semantic_digest: ContentDigest,
    /// Two acquisitions of the *same* immutable partition generation: the first
    /// is decoded from the start, the second is resumed from the exact cursor.
    pub partitions: Vec<AcquiredPartition>,
    /// Exact immutable generation both acquisitions name.
    pub partition_identity: ImmutableObjectIdentity,
    /// Bound applied to every decoder pull.
    pub decode_budget: DecodeBatchBudget,
    /// The budget the decoder's output leases are charged against.
    pub fragment_budget: StreamingResourceBudget,
    /// The budget bounding decoder reads from the acquired partition.
    pub acquisition_budget: AcquisitionBudget,
    /// Fragments the script emits before exhaustion.
    pub expected_fragment_count: u64,
    /// Frontier translated by the format.
    pub frontier: SourceFrontier,
    /// Seal accepted by the format.
    pub seal: SourceSeal,
    /// Ordinary issues the script reports through the injected handle.
    pub expected_issue_count: u64,
    /// Hook releasing one parked decode step.
    pub advance: FormatAdvance,
}

/// Canonical events captured from the format under test.
#[derive(Default)]
pub struct CapturingFormatEventSink {
    /// Fragment record identities in emission order.
    pub fragments: Vec<StableRecordId>,
    /// Session frontier digests in emission order.
    pub session_frontiers: Vec<ContentDigest>,
}

#[async_trait(?Send)]
impl FormatEventSink for CapturingFormatEventSink {
    async fn send(&mut self, event: FormatEvent) -> Result<(), StreamFormatError> {
        match event {
            FormatEvent::Fragment(fragment) => self.fragments.push(fragment.record_id),
            FormatEvent::SessionFrontier(watermark) => {
                self.session_frontiers.push(watermark.digest);
            }
        }
        Ok(())
    }
}

/// Assert the complete streaming format contract for one factory.
///
/// # Panics
///
/// Panics with a described failure on any contract violation.
pub async fn assert_format_conformance(
    factory: &dyn StreamingDatasetFormatFactory,
    reporter: Box<dyn StreamingIssueReporter>,
    mut cases: FormatConformanceCases,
) {
    assert_strict_validation(factory, &cases);

    let handle = reporter.handle();
    // Borrow of the owned reporter ends here, before every await below.
    let context = StreamingFormatPrepareContext {
        run: cases.run,
        stream_semantic_digest: cases.stream_semantic_digest,
        issue_reporter: handle,
        fragment_budget: cases.fragment_budget.clone(),
        acquisition_budget: cases.acquisition_budget.clone(),
    };
    let validated = factory
        .validate(&cases.authored, cases.source_descriptor)
        .expect("authored format configuration validates");
    let mut format = factory
        .prepare(validated, &context)
        .expect("format preparation succeeds");

    format
        .initialize(None)
        .await
        .expect("fresh participant initialization");

    assert_eq!(
        cases.partitions.len(),
        2,
        "the harness decodes one partition twice: once fresh, once resumed"
    );
    let replay_partition = cases.partitions.pop().expect("replay acquisition");
    let first_partition = cases.partitions.pop().expect("initial acquisition");

    let (first_ids, cursor_bytes, cursor_charge) =
        assert_leased_batch_then_backpressure_then_end(format.as_mut(), first_partition, &cases)
            .await;

    let replay_ids = assert_exact_cursor_restore(
        factory,
        format.as_mut(),
        replay_partition,
        cursor_bytes,
        &cases,
    )
    .await;
    assert_eq!(
        first_ids, replay_ids,
        "resuming from the exact cursor replays the identical fragment identities"
    );
    assert!(
        cursor_charge > 0,
        "a retained decoder cursor carries an exact nonzero byte charge"
    );

    assert_frontier_and_seal(format.as_mut(), &cases).await;

    let total = reporter
        .summary()
        .expect("reporter summary is available after conformance")
        .total;
    assert_eq!(
        total, cases.expected_issue_count,
        "scripted ordinary faults are the only reporter receipts"
    );
}

fn assert_strict_validation(
    factory: &dyn StreamingDatasetFormatFactory,
    cases: &FormatConformanceCases,
) {
    let descriptor = factory.descriptor();
    assert!(
        !descriptor.id.is_empty(),
        "a registered format declares a stable identifier"
    );
    assert!(
        cases
            .source_descriptor
            .access
            .contains(&descriptor.required_access),
        "a conformance pairing agrees on partition access before decoding"
    );
    factory
        .validate(&cases.authored, cases.source_descriptor)
        .expect("authored format configuration validates");
    assert!(
        factory
            .validate(&cases.rejected_authored, cases.source_descriptor)
            .is_err(),
        "unknown or malformed format configuration is refused before preparation"
    );
}

/// One leased batch, a parked second pull while the lease is held, resumption
/// after release, and explicit exhaustion.
async fn assert_leased_batch_then_backpressure_then_end(
    format: &mut dyn StreamingDatasetFormat,
    partition: AcquiredPartition,
    cases: &FormatConformanceCases,
) -> (Vec<StableRecordId>, Vec<u8>, usize) {
    let mut decoder = format
        .begin_partition(partition, None)
        .await
        .expect("decoder begins on a fresh partition");

    let batch = match decoder
        .next_batch(cases.decode_budget)
        .await
        .expect("first decoder pull succeeds")
    {
        DecodeStep::Batch(batch) => batch,
        DecodeStep::End(_) => panic!("a nonempty scripted partition yields a batch first"),
    };
    assert!(
        !batch.fragments.is_empty(),
        "a returned batch is never empty"
    );
    assert!(
        batch.fragments.len() <= cases.decode_budget.max_fragments,
        "a decoder never exceeds its caller-supplied fragment bound"
    );
    let ids: Vec<StableRecordId> = batch
        .fragments
        .iter()
        .map(|fragment| fragment.record_id)
        .collect();
    let cursor_bytes = batch.resume_after.as_bytes().to_vec();
    let cursor_charge = batch.resume_after.charged_bytes();

    // Lease lifetime: the batch still owns its fragment leases here.
    let held = cases.fragment_budget.snapshot().used_items;
    assert!(
        held > 0,
        "an issued batch holds its exact output charge for as long as it lives"
    );

    // Backpressure: with the output lease outstanding the decoder parks.
    {
        let blocked = decoder.next_batch(cases.decode_budget);
        tokio::pin!(blocked);
        assert!(
            futures::poll!(&mut blocked).is_pending(),
            "a decoder parks rather than exceeding the budget its output still holds"
        );
    }

    drop(batch);
    assert_eq!(
        cases.fragment_budget.snapshot().used_items,
        0,
        "dropping a batch releases its exact output charge"
    );

    (cases.advance)();
    let receipt = match decoder
        .next_batch(cases.decode_budget)
        .await
        .expect("the decoder resumes once its output lease is released")
    {
        DecodeStep::End(receipt) => receipt,
        DecodeStep::Batch(_) => panic!("the scripted partition emits exactly one batch"),
    };
    assert_eq!(receipt.partition, cases.partition_identity);
    assert_eq!(receipt.fragment_count, cases.expected_fragment_count);

    (ids, cursor_bytes, cursor_charge)
}

/// Resuming with a host-retained checkpoint restores the exact decoder cursor.
async fn assert_exact_cursor_restore(
    factory: &dyn StreamingDatasetFormatFactory,
    format: &mut dyn StreamingDatasetFormat,
    partition: AcquiredPartition,
    cursor_bytes: Vec<u8>,
    cases: &FormatConformanceCases,
) -> Vec<StableRecordId> {
    let lease = cases
        .fragment_budget
        .acquire(1, cursor_bytes.len())
        .await
        .expect("cursor charge fits the harness budget");
    let state = DecoderResumeState::new(Bytes::from(cursor_bytes.clone()), lease)
        .expect("exact cursor charge");
    let checkpoint = DecoderCheckpoint {
        partition: cases.partition_identity,
        format_semantic_digest: factory.descriptor().semantic_digest,
        state,
    };

    let mut decoder = format
        .begin_partition(partition, Some(checkpoint))
        .await
        .expect("decoder resumes from a host-retained cursor");
    let restored = decoder
        .resume_state()
        .expect("a resumed decoder reports its cursor");
    assert_eq!(
        restored.as_bytes(),
        cursor_bytes.as_slice(),
        "resumption restores the exact retained cursor, byte for byte"
    );
    // The reported cursor holds a real charge against the same output budget the
    // next pull draws from; release it before asking the decoder to continue.
    drop(restored);

    (cases.advance)();
    match decoder
        .next_batch(cases.decode_budget)
        .await
        .expect("resumed decoder pull succeeds")
    {
        DecodeStep::Batch(batch) => batch
            .fragments
            .iter()
            .map(|fragment| fragment.record_id)
            .collect(),
        DecodeStep::End(_) => Vec::new(),
    }
}

async fn assert_frontier_and_seal(
    format: &mut dyn StreamingDatasetFormat,
    cases: &FormatConformanceCases,
) {
    let mut sink = CapturingFormatEventSink::default();
    format
        .advance_source_frontier(cases.frontier.clone(), &mut sink)
        .await
        .expect("format translates a source frontier");
    let receipt = format
        .seal(cases.seal.clone(), &mut sink)
        .await
        .expect("format accepts an explicit source seal");
    assert_eq!(
        receipt.partition_count, 1,
        "the seal receipt counts exactly the partitions the harness exhausted"
    );
    assert_ne!(
        receipt.digest,
        ContentDigest::from_bytes([0; 32]),
        "a seal receipt binds a real digest"
    );
    assert!(
        !sink.session_frontiers.is_empty(),
        "a format with event time contributes session completeness"
    );
}
