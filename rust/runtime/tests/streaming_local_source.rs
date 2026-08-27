// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executable contract of the built-in immutable `local` streaming source.

#[allow(dead_code)]
#[path = "support/streaming_local_fixture.rs"]
mod streaming_local_fixture;

#[allow(dead_code)]
#[path = "support/streaming_source_conformance.rs"]
mod streaming_source_conformance;

use std::cell::Cell;
use std::rc::Rc;

use aiperf_runtime::streaming::{
    budget::BudgetLimits,
    checkpoint::CheckpointError,
    failure::{StableStreamingFailure, StreamingInputDomainIdentity},
    identity::{ContentDigest, ImmutableObjectIdentity},
    reliability::StreamingIssueReporter,
    source::{
        PartitionAccessRequest, SourceEvent, StreamingDatasetSourceFactory,
        StreamingSourceMode,
    },
    sources::local::{LOCAL_SOURCE_ID, LocalSourceFactory},
};

use streaming_local_fixture::{
    CountingReporter, LocalFixture, checkpoint_and_commit, expect_partition, raw, read_partition,
};
use streaming_source_conformance::{SourceConformanceCases, assert_source_conformance};

const FIRST: &[u8] = br#"{"role":"user","content":"a"}"#;
const SECOND: &[u8] = br#"{"role":"user","content":"b"}"#;
const THIRD: &[u8] = br#"{"role":"user","content":"c"}"#;

#[test]
fn descriptor_advertises_both_local_inventory_lifecycles() {
    let descriptor = LocalSourceFactory.descriptor();
    assert_eq!(descriptor.id, LOCAL_SOURCE_ID);
    assert!(descriptor.modes.contains(&StreamingSourceMode::Finite));
    assert!(descriptor.modes.contains(&StreamingSourceMode::Follow));
    assert!(
        LocalSourceFactory
            .validate(&raw(serde_json::json!({
                "root": "relative/not/absolute",
                "mode": { "kind": "finite" },
                "max_partition_bytes": 1,
                "max_scan_entries": 1,
            })))
            .is_err(),
        "a non-absolute root is refused before any filesystem effect"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn finite_scan_is_byte_sorted_and_seals_once() {
    let fixture = LocalFixture::new("finite-sorted");
    // Published out of order: the source must impose raw byte order on names.
    fixture.publish("002.jsonl", THIRD);
    fixture.publish("000.jsonl", FIRST);
    fixture.publish("001.jsonl", SECOND);

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture.open(&fixture.finite_config(), reporter.handle()).await;
    let snapshot_digest = opened.source.snapshot().digest;

    let mut ordered = Vec::new();
    let mut frontiers = 0_usize;
    let seal = loop {
        match opened.source.next_event().await.expect("finite source drains") {
            SourceEvent::Partition(partition) => {
                let bytes = read_partition(&partition).await.expect("partition acquires");
                ordered.push((partition.position().get(), bytes));
            }
            SourceEvent::Frontier(_) => frontiers += 1,
            SourceEvent::Seal(seal) => break seal,
        }
    };

    assert_eq!(
        ordered,
        vec![
            (0, FIRST.to_vec()),
            (1, SECOND.to_vec()),
            (2, THIRD.to_vec()),
        ],
        "finite positions follow raw byte order of the root-relative path"
    );
    assert_eq!(frontiers, 1, "a finite inventory publishes one frontier");
    assert_eq!(seal.final_position.map(|position| position.get()), Some(2));
    assert_eq!(
        seal.digest, snapshot_digest,
        "the seal binds the same immutable inventory as the open snapshot"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn follow_parks_while_quiet_and_never_seals_without_marker() {
    let fixture = LocalFixture::new("follow-parks");
    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture
        .open(&fixture.follow_config(Some("_SEALED")), reporter.handle())
        .await;

    {
        let pending = opened.source.next_event();
        tokio::pin!(pending);
        assert!(
            futures::poll!(&mut pending).is_pending(),
            "an empty follow root parks instead of sealing"
        );
        fixture.publish("000.jsonl", FIRST);
        let event = pending.await.expect("publication wakes the parked source");
        let SourceEvent::Partition(partition) = event else {
            panic!("publication announces a partition, not a seal");
        };
        assert_eq!(
            read_partition(&partition).await.expect("acquires"),
            FIRST.to_vec()
        );
    }

    // The frontier follows, then the source parks again: no marker, no seal.
    let SourceEvent::Frontier(frontier) = opened.source.next_event().await.expect("frontier") else {
        panic!("a drained follow batch publishes a frontier");
    };
    assert_eq!(frontier.through.get(), 0);
    {
        let pending = opened.source.next_event();
        tokio::pin!(pending);
        assert!(
            futures::poll!(&mut pending).is_pending(),
            "a quiet follow source without a marker never seals"
        );
        fixture.publish("_SEALED", b"");
        let event = pending.await.expect("the marker completes the inventory");
        let SourceEvent::Seal(seal) = event else {
            panic!("the authored marker seals the inventory");
        };
        assert_eq!(seal.final_position.map(|position| position.get()), Some(0));
    }
    opened.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn restore_resumes_after_committed_object_without_duplicates() {
    let fixture = LocalFixture::new("restore-follow");
    fixture.publish("000.jsonl", FIRST);

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture
        .open(&fixture.follow_config(None), reporter.handle())
        .await;
    opened.source.initialize(None).await.expect("fresh start");
    let (position, first_identity, bytes) = expect_partition(opened.source.as_mut()).await;
    assert_eq!(position.get(), 0);
    assert_eq!(bytes, FIRST.to_vec());
    let committed = checkpoint_and_commit(opened.source.as_mut(), fixture.run).await;
    opened.control.stop();
    drop(opened);

    fixture.publish("001.jsonl", SECOND);
    let restored_reporter = CountingReporter::new(fixture.run);
    let mut restored = fixture
        .open(&fixture.follow_config(None), restored_reporter.handle())
        .await;
    restored
        .source
        .initialize(Some(committed))
        .await
        .expect("committed state restores");
    let (next_position, next_identity, next_bytes) = expect_partition(restored.source.as_mut()).await;
    assert_eq!(
        next_bytes,
        SECOND.to_vec(),
        "a restored follow source resumes after the committed cursor without duplicates"
    );
    assert_eq!(next_position.get(), 1, "positions continue past the cursor");
    assert_ne!(next_identity, first_identity);
    restored.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn absent_committed_partition_refuses_restore() {
    let fixture = LocalFixture::new("restore-absent");
    fixture.publish("000.jsonl", FIRST);

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture
        .open(&fixture.follow_config(None), reporter.handle())
        .await;
    opened.source.initialize(None).await.expect("fresh start");
    let _ = expect_partition(opened.source.as_mut()).await;
    let committed = checkpoint_and_commit(opened.source.as_mut(), fixture.run).await;
    opened.control.stop();
    drop(opened);

    fixture.unlink("000.jsonl");
    let restored_reporter = CountingReporter::new(fixture.run);
    let mut restored = fixture
        .open(&fixture.follow_config(None), restored_reporter.handle())
        .await;
    assert_eq!(
        restored.source.initialize(Some(committed)).await,
        Err(CheckpointError::SourceUnavailableOnResume),
        "an unreachable committed partition has no truthful continuation"
    );
    restored.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn in_place_mutation_after_discovery_is_refused() {
    let fixture = LocalFixture::new("mutation");
    fixture.publish("000.jsonl", FIRST);

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture
        .open(&fixture.follow_config(None), reporter.handle())
        .await;
    let SourceEvent::Partition(partition) = opened.source.next_event().await.expect("partition")
    else {
        panic!("the published object is announced");
    };
    // Same inode, different content: the frozen generation no longer holds.
    fixture.rewrite_in_place("000.jsonl", b"mutated-in-place-after-discovery");

    let error = read_partition(&partition)
        .await
        .expect_err("a mutated object is never acquired");
    assert_eq!(error.code(), "mutated_object");
    assert_eq!(
        reporter.accepted(),
        0,
        "a mutated object is an invariant, never a retried hole"
    );
    opened.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn missing_partition_reports_one_retryable_issue_and_follow_continues() {
    let fixture = LocalFixture::new("hole");
    fixture.publish("000.jsonl", FIRST);

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture
        .open(&fixture.follow_config(None), reporter.handle())
        .await;
    let SourceEvent::Partition(partition) = opened.source.next_event().await.expect("partition")
    else {
        panic!("the published object is announced");
    };
    fixture.unlink("000.jsonl");

    let error = read_partition(&partition)
        .await
        .expect_err("a vanished object cannot be acquired");
    assert_eq!(error.code(), "open");
    assert_eq!(
        reporter.accepted(),
        1,
        "bounded reopen exhaustion reports exactly one ordinary partition issue"
    );
    drop(partition);

    // The stream stays live: a later publication receives the next ordinal.
    let SourceEvent::Frontier(_) = opened.source.next_event().await.expect("frontier") else {
        panic!("the announced position still advanced the frontier");
    };
    fixture.publish("001.jsonl", SECOND);
    let (position, _, bytes) = expect_partition(opened.source.as_mut()).await;
    assert_eq!(position.get(), 1);
    assert_eq!(bytes, SECOND.to_vec());
    opened.control.stop();
}

#[tokio::test(flavor = "current_thread")]
async fn symlinked_partition_is_never_discovered() {
    let fixture = LocalFixture::new("symlink");
    fixture.publish("000.jsonl", FIRST);
    fixture.symlink("999.jsonl", "000.jsonl");

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture.open(&fixture.finite_config(), reporter.handle()).await;
    let mut discovered = 0_usize;
    loop {
        match opened.source.next_event().await.expect("finite source drains") {
            SourceEvent::Partition(_) => discovered += 1,
            SourceEvent::Frontier(_) => {}
            SourceEvent::Seal(_) => break,
        }
    }
    assert_eq!(
        discovered, 1,
        "an in-root symlink is refused by O_NOFOLLOW discovery, never followed"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn reference_manifest_orders_and_verifies_declared_digests() {
    let fixture = LocalFixture::new("reference");
    fixture.publish("000.jsonl", FIRST);
    fixture.publish("001.jsonl", SECOND);
    let first_digest = hex(blake3::hash(FIRST).as_bytes());
    let second_digest = hex(blake3::hash(SECOND).as_bytes());
    // Manifest order, not name order, is the ordering authority.
    fixture.publish(
        "index.manifest",
        format!(
            "{{\"path\":\"001.jsonl\",\"size_bytes\":{},\"digest\":\"{second_digest}\"}}\n\
             {{\"path\":\"000.jsonl\",\"size_bytes\":{},\"digest\":\"{first_digest}\"}}\n",
            SECOND.len(),
            FIRST.len()
        )
        .as_bytes(),
    );

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture
        .open(&fixture.reference_config("index.manifest"), reporter.handle())
        .await;
    let (first_position, _, first_bytes) = expect_partition(opened.source.as_mut()).await;
    assert_eq!(first_position.get(), 0);
    assert_eq!(first_bytes, SECOND.to_vec());

    // A declared digest that the streamed bytes cannot reproduce is an
    // identity mismatch, never a hole.
    let mismatched = LocalFixture::new("reference-mismatch");
    mismatched.publish("000.jsonl", FIRST);
    mismatched.publish(
        "index.manifest",
        format!(
            "{{\"path\":\"000.jsonl\",\"size_bytes\":{},\"digest\":\"{}\"}}\n",
            FIRST.len(),
            hex(&[0x00; 32])
        )
        .as_bytes(),
    );
    let mismatch_reporter = CountingReporter::new(mismatched.run);
    let mut opened_mismatch = mismatched
        .open(
            &mismatched.reference_config("index.manifest"),
            mismatch_reporter.handle(),
        )
        .await;
    let SourceEvent::Partition(partition) = opened_mismatch
        .source
        .next_event()
        .await
        .expect("manifest partition")
    else {
        panic!("the manifest record is announced");
    };
    let error = read_partition(&partition)
        .await
        .expect_err("a declared-digest mismatch refuses the object");
    assert_eq!(error.code(), "identity_mismatch");

    // An escaping manifest path is refused before any event.
    let escaping = LocalFixture::new("reference-escape");
    escaping.publish(
        "index.manifest",
        format!(
            "{{\"path\":\"../outside.jsonl\",\"size_bytes\":1,\"digest\":\"{}\"}}\n",
            hex(&[0x00; 32])
        )
        .as_bytes(),
    );
    let escape_reporter = CountingReporter::new(escaping.run);
    assert!(
        escaping
            .try_open(
                &escaping.reference_config("index.manifest"),
                escape_reporter.handle()
            )
            .await
            .is_err(),
        "a non-normal manifest path is refused before discovery emits anything"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn seekable_local_access_reads_exact_immutable_ranges() {
    let fixture = LocalFixture::new("seekable");
    fixture.publish("000.jsonl", FIRST);

    let reporter = CountingReporter::new(fixture.run);
    let mut opened = fixture.open(&fixture.finite_config(), reporter.handle()).await;
    let SourceEvent::Partition(partition) = opened.source.next_event().await.expect("partition")
    else {
        panic!("the published object is announced");
    };
    let budget = streaming_local_fixture::acquisition_budget();
    let acquired = partition
        .content()
        .acquire(PartitionAccessRequest::SeekableLocal, &budget)
        .await
        .expect("a local snapshot is acquirable");
    let aiperf_runtime::streaming::source::AcquiredPartitionAccess::SeekableLocal(snapshot) =
        acquired.into_access()
    else {
        panic!("a seekable request returns seekable access");
    };
    let chunk = snapshot
        .read_at(
            5,
            std::num::NonZeroUsize::new(4).expect("nonzero bound"),
            &budget,
        )
        .await
        .expect("an in-object range reads");
    assert_eq!(chunk.as_bytes(), &FIRST[5..9]);
}

#[tokio::test(flavor = "current_thread")]
async fn same_object_under_two_streams_has_distinct_input_domains() {
    let fixture = LocalFixture::new("two-streams");
    fixture.publish("000.jsonl", FIRST);

    let reporter = CountingReporter::new(fixture.run);
    let mut first = fixture.open(&fixture.finite_config(), reporter.handle()).await;
    let (_, first_identity, _) = expect_partition(first.source.as_mut()).await;

    let mut other = LocalFixture::new("two-streams-b");
    other.stream_identity = ContentDigest::from_bytes([0x52; 32]);
    other.publish("000.jsonl", FIRST);
    let other_reporter = CountingReporter::new(other.run);
    let mut second = other
        .open(&other.finite_config(), other_reporter.handle())
        .await;
    let (_, other_identity, _) = expect_partition(second.source.as_mut()).await;

    assert_ne!(
        first_identity, other_identity,
        "identity is bound under the stream semantic digest"
    );
    assert_ne!(
        domain_bytes(fixture.stream_identity, first_identity),
        domain_bytes(other.stream_identity, other_identity),
        "the same record path under two streams is two distinct input domains"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn local_source_satisfies_the_shared_source_conformance_harness() {
    let fixture = LocalFixture::new("conformance");
    let reporter = CountingReporter::new(fixture.run);
    let step = Rc::new(Cell::new(0_usize));
    let root = fixture.root().to_path_buf();
    let advance = {
        let step = Rc::clone(&step);
        Rc::new(move || {
            let current = step.get();
            step.set(current + 1);
            match current {
                0 => publish_at(&root, "000.jsonl", FIRST),
                1 => publish_at(&root, "001.jsonl", SECOND),
                2 => publish_at(&root, "_SEALED", b""),
                _ => {}
            }
        })
    };

    assert_source_conformance(
        &LocalSourceFactory,
        Box::new(reporter),
        SourceConformanceCases {
            authored: fixture.follow_config(Some("_SEALED")),
            rejected_authored: raw(serde_json::json!({
                "root": fixture.root(),
                "mode": { "kind": "follow" },
                "max_partition_bytes": 1,
                "max_scan_entries": 1,
                "unknown_field": true,
            })),
            memory_limits: BudgetLimits {
                max_items: 32,
                max_bytes: 1 << 20,
            },
            disk_limits: BudgetLimits {
                max_items: 8,
                max_bytes: 1 << 20,
            },
            expected_partition_count: 2,
            // `announce` deduplicates before assigning a position, so the
            // source provably never re-announces a discovered position.
            expected_duplicate_count: 0,
            expects_frontier: true,
            expected_issue_count: 0,
            run: fixture.run,
            stream_semantic_digest: fixture.stream_identity,
            advance,
        },
    )
    .await;
}

fn publish_at(root: &std::path::Path, name: &str, bytes: &[u8]) {
    let staged = root.join(format!("{name}.part"));
    std::fs::write(&staged, bytes).expect("staged partition is writable");
    std::fs::rename(&staged, root.join(name)).expect("publication by rename succeeds");
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn domain_bytes(stream: ContentDigest, object: ImmutableObjectIdentity) -> [u8; 64] {
    let domain = StreamingInputDomainIdentity::new(stream, object);
    let mut bytes = [0_u8; 64];
    bytes[..32].copy_from_slice(domain.stream_identity().as_bytes());
    bytes[32..].copy_from_slice(domain.source_identity().as_bytes());
    bytes
}
