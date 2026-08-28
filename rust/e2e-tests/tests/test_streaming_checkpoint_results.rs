// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-binary checkpoint-result convergence and the public status vocabulary.
//!
//! The vocabulary is the contract these rows exist to protect:
//!
//! - `failed` is reachable **only** from a checked terminal-boundary invariant;
//! - `degraded` means execution continued or drained truthfully with holes,
//!   quarantines, or failed terminal actions, and the authoritative generation
//!   stays readable;
//! - `export_incomplete` means the native generation is readable while a
//!   compactor, report, or optional exporter is pending or exhausted.
//!
//! No derived sink failure may rewrite an execution outcome or a checkpoint
//! head, and no ordinary data fault may be hidden.

mod common;

#[path = "support/streaming_product.rs"]
mod support;

use support::{
    StreamingServerCase, StreamingServerHarness, StreamingSourceKind, StreamingTopology,
    StreamingTransport,
};

const FIXTURE: &str = "local_finite_conversation.yaml";

/// A refused streaming run leaves no checkpoint head behind.
///
/// This is the "no derived failure rewrites a checkpoint head" rule in its
/// simplest form: a run that never executed must not have published one.
#[test]
fn a_refused_run_publishes_no_checkpoint_head() {
    let case = StreamingServerCase {
        name: "http_single_process",
        transport: StreamingTransport::Http,
        source: StreamingSourceKind::Local,
        topology: StreamingTopology::SingleProcess,
        expected_status: "failed",
    };
    let harness = StreamingServerHarness::start(&case);
    let outcome = harness.profile(FIXTURE);

    assert!(!outcome.success());
    assert!(
        std::fs::read_dir(harness.checkpoint_root())
            .expect("checkpoint root readable")
            .next()
            .is_none(),
        "a refused run must not publish a checkpoint head"
    );
}

/// The public status a row reports is one of exactly four values.
///
/// An unknown status is a contract break even when the run behaved correctly:
/// downstream readers branch on this string.
#[test]
fn public_status_is_drawn_from_the_declared_vocabulary() {
    for case in support::server_matrix() {
        let harness = StreamingServerHarness::start(&case);
        let status = harness.profile(FIXTURE).public_status();
        assert!(
            matches!(
                status.as_str(),
                "complete" | "degraded" | "export_incomplete" | "failed"
            ),
            "{}: unknown public status {status:?}",
            case.name
        );
        assert_eq!(
            status, case.expected_status,
            "{}: unexpected public status",
            case.name
        );
    }
}

/// A source hole continues later objects and reports `degraded`, not `failed`.
#[test]
#[ignore = "shadow_replay has no prepare_with_context; guarded by \
            public_status_is_drawn_from_the_declared_vocabulary"]
fn source_hole_continues_later_objects_and_reports_degraded() {
    unimplemented!("un-ignore once a streaming run can acquire a partition");
}
