// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-binary checkpoint-head and result-artifact coverage.
//!
//! The rule these rows protect is that no derived sink failure and no refusal
//! may rewrite an execution outcome or publish a checkpoint head. A run that
//! never executed must leave the durable checkpoint root exactly as it found
//! it, and must not emit a measurement artifact that a later reader would
//! mistake for a partial result.
//!
//! The checkpoint root is created empty by the harness and its path is authored
//! into the fixture, so "the product wrote nothing here" is directly
//! observable rather than inferred from the absence of a log line.

mod common;

#[path = "support/streaming_product.rs"]
mod support;

use support::{StreamingServerCase, StreamingServerHarness, StreamingTransport};

/// A refused run publishes no checkpoint head.
#[tokio::test]
async fn a_refused_run_publishes_no_checkpoint_head() {
    let harness = StreamingServerHarness::start(StreamingServerCase::single_process(
        "checkpoint_head",
        StreamingTransport::Http,
    ));
    let outcome = harness.profile();

    outcome.assert_refused_naming("checkpoint_head", &["scheduled_request"]);
    assert!(
        std::fs::read_dir(harness.checkpoint_root())
            .expect("checkpoint root readable")
            .next()
            .is_none(),
        "a refused run must not publish a checkpoint head"
    );
}

/// Static validation touches neither the checkpoint root nor the source root.
///
/// Validation resolves the same authored checkpoint backend a profile run
/// would, so a backend that eagerly created its root at construction rather
/// than at first commit would be caught here.
#[tokio::test]
async fn validation_touches_neither_the_checkpoint_root_nor_the_source_root() {
    let harness = StreamingServerHarness::start(StreamingServerCase::single_process(
        "validate_no_effect",
        StreamingTransport::Http,
    ));
    let before = source_entries(&harness);
    let outcome = harness.validate();

    assert!(
        std::fs::read_dir(harness.checkpoint_root())
            .expect("checkpoint root readable")
            .next()
            .is_none(),
        "validation must not write into the checkpoint root"
    );
    assert_eq!(
        source_entries(&harness),
        before,
        "validation must not add to or remove from the source root"
    );
    assert!(
        outcome.measurement_artifacts().is_empty(),
        "validation must emit no measurement artifact, found {:?}",
        outcome.measurement_artifacts()
    );
}

/// A refused run leaves no artifact a later reader could mistake for a result.
///
/// The run's own log tree is excluded by name: the logger is installed before
/// the registry resolves anything, so its presence is not an execution effect.
#[tokio::test]
async fn a_refused_run_emits_no_partial_result_artifact() {
    let harness = StreamingServerHarness::start(StreamingServerCase::single_process(
        "no_partial_result",
        StreamingTransport::Http,
    ));
    let outcome = harness.profile();

    assert!(!outcome.success());
    assert!(
        outcome.measurement_artifacts().is_empty(),
        "a refused run must emit no measurement artifact, found {:?}",
        outcome.measurement_artifacts()
    );
    assert_eq!(
        harness.endpoint_issues().await,
        0,
        "a refused run must not reach the endpoint"
    );
}

/// Names published under the source root, sorted.
fn source_entries(harness: &StreamingServerHarness) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(harness.source_root())
        .expect("source root readable")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}
