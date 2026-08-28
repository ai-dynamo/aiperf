// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-binary cellular streaming product coverage.
//!
//! The companion suite `test_streaming_shadow_replay.rs` proves that a single
//! cellular row refuses before any endpoint issue. These rows prove the
//! *invariance* half: the topology overlay is not allowed to change the product
//! decision, the reason for it, or the side effects it leaves behind.
//!
//! That distinction matters because a controller partitions work and launches
//! cell processes. A product that resolved its stream bindings per-cell instead
//! of once at the controller could easily reach a different decision at
//! `--cells 4` than at `--cells 2`, or leave one cell's scratch state behind
//! after the controller refused.

mod common;

#[path = "support/streaming_product.rs"]
mod support;

use support::{LEAK_NEEDLES, StreamingServerCase, StreamingServerHarness, StreamingTransport};

/// A cellular run and a single-process run reach the same decision for the same
/// reason.
///
/// The stable refusal strips ports and scratch paths, so an equal comparison
/// means the two topologies agreed on *why* the product refused rather than
/// merely on the fact that both exited nonzero.
#[tokio::test]
async fn cellular_and_single_process_agree_on_the_streaming_decision() {
    let single = StreamingServerHarness::start(StreamingServerCase::single_process(
        "single_process",
        StreamingTransport::Http,
    ));
    let single_outcome = single.profile();
    single_outcome.assert_refused_naming("single_process", &["scheduled_request"]);

    let cellular = StreamingServerHarness::start(StreamingServerCase::cellular(
        "cellular",
        StreamingTransport::Http,
        2,
    ));
    let cellular_outcome = cellular.profile();
    cellular_outcome.assert_refused_naming("cellular", &["scheduled_request"]);

    assert_eq!(
        single_outcome.stable_refusal(),
        cellular_outcome.stable_refusal(),
        "the topology overlay must not change the product decision or its reason"
    );
    assert_eq!(single.endpoint_issues().await, 0);
    assert_eq!(cellular.endpoint_issues().await, 0);
}

/// Widening the cell count changes nothing a caller can observe.
///
/// If bindings were resolved per-cell rather than once, a wider fan-out would
/// be the row that exposed it: more cells means more chances for one of them to
/// reach the endpoint before the controller's refusal propagated.
#[tokio::test]
async fn a_wider_cell_fan_out_reaches_the_same_decision_without_issuing() {
    let narrow = StreamingServerHarness::start(StreamingServerCase::cellular(
        "cellular_two",
        StreamingTransport::Http,
        2,
    ));
    let narrow_outcome = narrow.profile();

    let wide = StreamingServerHarness::start(StreamingServerCase::cellular(
        "cellular_four",
        StreamingTransport::Http,
        4,
    ));
    let wide_outcome = wide.profile();

    assert_eq!(
        narrow_outcome.stable_refusal(),
        wide_outcome.stable_refusal(),
        "the cell count must not change the product decision"
    );
    assert_eq!(
        wide.endpoint_issues().await,
        0,
        "no cell may issue a request before the controller resolves its bindings"
    );
    assert_eq!(
        wide.captured_inference_requests(),
        0,
        "the mock retained an inference body, so a cell reached the endpoint"
    );
    for outcome in [&narrow_outcome, &wide_outcome] {
        assert!(
            outcome.measurement_artifacts().is_empty(),
            "a refused cellular run must emit no measurement artifact, found {:?}",
            outcome.measurement_artifacts()
        );
        outcome.assert_no_raw_or_secret_leak(LEAK_NEEDLES);
    }
}
