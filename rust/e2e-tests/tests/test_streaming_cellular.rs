// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-binary cellular streaming product evidence.
//!
//! The plan places the *no-early-issue* proof here rather than in the V4A
//! socket-free suite for a specific reason: "nothing was issued" is only
//! meaningful against a real endpoint that can report what it observed. A
//! socket-free fixture cannot distinguish a run that refused before issuing
//! from a run that issued and discarded the reply.

mod common;

#[path = "support/streaming_product.rs"]
mod support;

use support::{
    StreamingServerCase, StreamingServerHarness, StreamingSourceKind, StreamingTopology,
    StreamingTransport,
};

const FIXTURE: &str = "local_finite_conversation.yaml";

/// A bound cellular streaming run refuses before any prepare acknowledgement,
/// capacity release, or endpoint issue.
///
/// The refusal has to precede all three: a controller that acknowledged a
/// prepare or released capacity before failing would leave the cell's admission
/// accounting inconsistent with a run that never started.
#[test]
fn bound_synthesis_profile_mismatch_refuses_before_any_prepare_or_issue() {
    let case = StreamingServerCase {
        name: "cellular_synthesis_profile_mismatch",
        transport: StreamingTransport::Http,
        source: StreamingSourceKind::Local,
        topology: StreamingTopology::Cellular { cells: 2 },
        expected_status: "failed",
    };
    let harness = StreamingServerHarness::start(&case);
    let outcome = harness.profile(FIXTURE);

    assert_eq!(outcome.public_status(), "failed");
    assert_eq!(
        outcome.prepare_acknowledgements(),
        0,
        "nothing may be prepared before the refusal"
    );
    assert_eq!(
        outcome.releases(),
        0,
        "nothing may be released before the refusal"
    );
    assert_eq!(
        outcome.endpoint_issues(),
        0,
        "nothing may be issued before the refusal"
    );
}

/// A cellular run and a single-process run over the same fixture reach the same
/// outcome: the topology overlay must not change what the product decides.
#[test]
fn cellular_and_single_process_agree_on_the_streaming_outcome() {
    let single = StreamingServerHarness::start(&StreamingServerCase {
        name: "http_single_process",
        transport: StreamingTransport::Http,
        source: StreamingSourceKind::Local,
        topology: StreamingTopology::SingleProcess,
        expected_status: "failed",
    })
    .profile(FIXTURE);
    let cellular = StreamingServerHarness::start(&StreamingServerCase {
        name: "http_cellular_two_cells",
        transport: StreamingTransport::Http,
        source: StreamingSourceKind::Local,
        topology: StreamingTopology::Cellular { cells: 2 },
        expected_status: "failed",
    })
    .profile(FIXTURE);

    assert_eq!(
        single.public_status(),
        cellular.public_status(),
        "the topology overlay must not change the streaming outcome"
    );
    assert_eq!(single.logical_membership(), cellular.logical_membership());
    assert_eq!(single.final_report_order(), cellular.final_report_order());
}

/// A controller restart and a cell restart converge on one checkpoint result.
#[test]
#[ignore = "shadow_replay has no prepare_with_context; guarded by \
            cellular_and_single_process_agree_on_the_streaming_outcome"]
fn controller_and_cell_restart_converge_on_one_checkpoint_result() {
    unimplemented!("un-ignore once a cellular streaming run can commit a generation");
}
