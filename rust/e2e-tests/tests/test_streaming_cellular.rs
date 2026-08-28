// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-binary cellular streaming product coverage.
//!
//! ## Where a cellular streaming run stops today, and why that is the assertion
//!
//! A single-process streaming run reaches streaming capability agreement and is
//! refused there, because the endpoint-reaching `scheduled_request` action sink
//! holds worker-local `Rc` handles and has no composition root to construct it.
//!
//! A cellular run stops *earlier and elsewhere*: the controller's partitioner
//! reads a `datasets` array that a `dataset_streams` run does not author, and
//! refuses with `run cfg has no datasets array` before it launches a cell. So
//! the two topologies do **not** agree on the reason, and a row asserting they
//! do would be asserting a property the product does not have.
//!
//! What they do agree on is the part the plan actually requires: both fail
//! closed, and neither prepares, releases, or issues anything. A mock server is
//! running and reachable in every row, so a controller that partitioned work
//! and let a cell issue optimistically before resolving its stream bindings
//! would be caught here.
//!
//! These rows cannot rot. When cellular streaming partitioning lands, the
//! `no datasets array` assertion fails and forces this file to be rewritten
//! against whatever the controller then does.

mod common;

#[path = "support/streaming_product.rs"]
mod support;

use support::{LEAK_NEEDLES, StreamingServerCase, StreamingServerHarness, StreamingTransport};

/// The refusal a cellular streaming run reaches before it launches a cell.
const CELLULAR_REFUSAL: &str = "run cfg has no datasets array";

/// A cellular streaming run fails closed before any endpoint issue, and stops
/// at the controller rather than inside a cell.
#[tokio::test]
async fn cellular_streaming_fails_closed_at_the_controller_without_issuing() {
    let harness = StreamingServerHarness::start(StreamingServerCase::cellular(
        "cellular",
        StreamingTransport::Http,
        2,
    ));
    let outcome = harness.profile();

    outcome.assert_refused_naming("cellular", &[CELLULAR_REFUSAL]);
    assert_eq!(
        harness.endpoint_issues().await,
        0,
        "cellular: no cell may issue a request before the controller resolves its bindings"
    );
    assert_eq!(
        harness.captured_inference_requests(),
        0,
        "cellular: the mock retained an inference body, so a cell reached the endpoint"
    );
    assert!(
        std::fs::read_dir(harness.checkpoint_root())
            .expect("checkpoint root readable")
            .next()
            .is_none(),
        "cellular: a refused run must not write into the checkpoint root"
    );
    assert!(
        outcome.measurement_artifacts().is_empty(),
        "cellular: a refused run must emit no measurement artifact, found {:?}",
        outcome.measurement_artifacts()
    );
    outcome.assert_no_raw_or_secret_leak(LEAK_NEEDLES);
}

/// The cellular and single-process paths stop for different reasons.
///
/// This row pins the *gap* rather than papering over it: while it passes, the
/// topology overlay changes where a streaming run stops, and any claim of
/// cellular streaming parity is false. When the two paths converge, this row
/// fails and forces the parity claim to be written explicitly.
#[tokio::test]
async fn cellular_and_single_process_stop_at_different_stages() {
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
    cellular_outcome.assert_refused_naming("cellular", &[CELLULAR_REFUSAL]);

    assert_ne!(
        single_outcome.stable_refusal(),
        cellular_outcome.stable_refusal(),
        "the two paths have converged; rewrite this file against the new cellular behavior"
    );
    assert_eq!(single.endpoint_issues().await, 0);
    assert_eq!(cellular.endpoint_issues().await, 0);
}

/// Widening the cell count changes nothing a caller can observe.
///
/// If the controller resolved its configuration per-cell rather than once, a
/// wider fan-out would be the row that exposed it: more cells means more
/// chances for one of them to reach the endpoint before the controller's
/// refusal propagated.
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
