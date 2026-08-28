// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-binary socket-free product coverage for native streaming shadow replay.
//!
//! Every row drives the pinned `AIPERF_DRY_RUN_BIN` over a committed Config-v2
//! fixture through the `dry_run` transport, so nothing here opens a socket.
//!
//! ## Why one row is `#[ignore]`d and one row guards it
//!
//! `shadow_replay` is a registered workload identity: the CLI authors it, the
//! protocol-v2 layer projects it onto the shadow-replay workload DTO, and the
//! static registry resolves its selected source, format, session program, and
//! checkpoint backend against the compiled inventory. The one component it
//! cannot resolve is the endpoint-reaching action sink: `scheduled_request`
//! holds worker-local `Rc` handles, so it is neither `Send` nor `Sync` and is
//! constructed by a run's composition root rather than registered at startup.
//! `shadow_replay` has no such composition root yet, so capability agreement
//! refuses the binding and the product cannot reach the streaming pipeline from
//! a profile invocation.
//!
//! [`restart_from_checkpoint_matches_sealed_reference`] is therefore ignored
//! and [`shadow_replay_profile_refuses_before_any_source_or_endpoint_effect`]
//! asserts *exactly why*. The pair cannot rot: the moment the action sink is
//! constructed and registered, the guard row fails and forces the restart row
//! to be un-ignored, rather than leaving a permanently-skipped test behind.

#[path = "support/streaming_product.rs"]
mod support;

use support::{LEAK_NEEDLES, ProductRun, StreamingProductFixture};

/// The restart contract: a resumed run and a single uninterrupted sealed
/// reference agree on the logical record multiset and the compacted metric
/// store, exactly.
///
/// Partition B is published *between* the kill and the resume — without it the
/// resumed run would have nothing new to consume and could equal the sealed
/// reference trivially.
#[test]
#[ignore = "shadow_replay has no composition root that registers the endpoint-reaching action \
            sink; guarded by shadow_replay_profile_refuses_before_any_source_or_endpoint_effect"]
fn restart_from_checkpoint_matches_sealed_reference() {
    let fixture = StreamingProductFixture::local_follow_cross_chunk_graph();
    let first = fixture
        .run_until_checkpoint_then_kill()
        .expect("first incarnation commits a generation");
    let locator = first.stdout.clone();

    fixture.publish_partition_b();
    fixture.publish_seal();
    let resumed = fixture.resume(&locator);

    let sealed = StreamingProductFixture::local_finite_conversation().run_sealed_reference();

    assert_eq!(
        resumed.logical_record_multiset(),
        sealed.logical_record_multiset(),
        "a resumed run must reach the same logical membership as one uninterrupted pass"
    );
    assert_eq!(
        resumed.compacted_metric_store(),
        sealed.compacted_metric_store(),
        "a resumed run must fold to the same compacted metric store"
    );
}

/// A `shadow_replay` profile refuses before any source, checkpoint, or endpoint
/// effect.
///
/// This is the current product truth and the guard on the ignored restart row.
/// The refusal must name the unresolved binding and the compiled inventory, and
/// it must leave the source root, the checkpoint root, and every measurement
/// artifact untouched: a workload that cannot execute must not first acquire a
/// partition, commit a generation, or emit a partial report. The run's own log
/// file is not a measurement artifact — the logger is installed before the
/// registry resolves anything — so it is excluded by name.
#[test]
fn shadow_replay_profile_refuses_before_any_source_or_endpoint_effect() {
    let fixture = StreamingProductFixture::local_follow_cross_chunk_graph();
    let run = fixture.profile();

    run.assert_refused_naming(&["scheduled_request", "available:"]);
    assert_eq!(
        ProductRun::generation(fixture.checkpoint_root()),
        None,
        "a refused workload must not commit a checkpoint generation"
    );
    let measurement: Vec<_> = run
        .artifact_files()
        .into_iter()
        .filter(|path| !path.starts_with("logs"))
        .collect();
    assert!(
        measurement.is_empty(),
        "a refused workload must not emit artifacts, found {measurement:?}"
    );
    assert!(
        std::fs::read_dir(fixture.checkpoint_root())
            .expect("checkpoint root readable")
            .next()
            .is_none(),
        "a refused workload must not write into the checkpoint root"
    );
}

/// Unregistered streaming identifiers fail closed against the compiled
/// inventory, naming both the request and the alternatives.
///
/// This is the static registry stage: it runs without opening a socket, a
/// dataset, or a stream, so an authoring mistake is a validation error rather
/// than a mid-run surprise.
#[test]
fn unregistered_streaming_component_fails_closed_against_compiled_inventory() {
    let fixture = StreamingProductFixture::unregistered_components();
    let run = fixture.validate();

    run.assert_refused_naming(&["available:"]);
    assert!(
        run.artifact_files().is_empty(),
        "validation must not emit artifacts"
    );
    assert!(
        std::fs::read_dir(fixture.source_root())
            .expect("source root readable")
            .next()
            .is_none(),
        "validation must not touch the source root"
    );
}

/// The registered composition normalizes and resolves without Python, a socket,
/// or a secret in any emitted diagnostic.
///
/// `config validate` is the one streaming surface that is complete today, so it
/// carries the Config-v2 authoring proof for the whole section.
#[test]
fn registered_streaming_composition_validates_without_python_or_socket() {
    let fixture = StreamingProductFixture::local_finite_conversation();
    let run = fixture.validate();

    let output = run.combined_output();
    assert!(
        !output.contains("Traceback") && !output.contains("ModuleNotFoundError"),
        "streaming validation must be native, not delegated to Python:\n{output}"
    );
    // A refusal is acceptable here — what is not acceptable is a refusal that
    // blames an identifier the fixture never authored.
    for absent in ["no_such_source", "no_such_format", "no_such_session_program"] {
        assert!(
            !output.contains(absent),
            "validation named an identifier this fixture never authored: {absent}"
        );
    }
    run.assert_no_raw_or_secret_leak(LEAK_NEEDLES);
}

/// Raw source bytes never reach an artifact or a child stream on any row.
///
/// Normalized request messages may legitimately appear in raw-record artifacts;
/// the acquired partition's own bytes may not, in any disposition, including a
/// refusal that has already read the fixture from disk.
#[test]
fn raw_source_bytes_and_secrets_never_reach_artifacts() {
    for fixture in [
        StreamingProductFixture::local_follow_cross_chunk_graph(),
        StreamingProductFixture::local_finite_conversation(),
    ] {
        let run = fixture.profile();
        run.assert_no_raw_or_secret_leak(LEAK_NEEDLES);
    }
}
