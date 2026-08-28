// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-binary server and cellular product coverage for native streaming
//! shadow replay.
//!
//! Every row drives the pinned `AIPERF_E2E_BIN` over a committed Config-v2
//! fixture against a live in-repo `aiperf-mock-server`, so unlike the
//! socket-free dry-run suite these rows prove behavior *with a reachable
//! endpoint present*.
//!
//! ## What the product does today, and why that is the assertion
//!
//! `shadow_replay` resolves its selected source, format, session program, and
//! checkpoint backend against the compiled inventory. The one component it
//! cannot resolve is the endpoint-reaching action sink: `scheduled_request`
//! holds worker-local `Rc` handles, so it is neither `Send` nor `Sync` and must
//! be constructed by a run's composition root rather than registered at
//! startup. `shadow_replay` has no such composition root yet, so streaming
//! capability agreement refuses the binding.
//!
//! That refusal is exactly the property the plan asks these rows to prove: the
//! run must fail closed **before any endpoint issue**, and it must do so
//! identically over HTTP, over gRPC, and across a cellular topology. A mock
//! server is running and reachable in every row, so a product that issued
//! optimistically before resolving its bindings would be caught here and
//! nowhere else.
//!
//! The rows are written so they cannot rot: the moment the action sink is
//! constructed and registered, every `assert_refused_naming` row fails and
//! forces the executing coverage to be written in its place.

mod common;

#[path = "support/streaming_product.rs"]
mod support;

use support::{LEAK_NEEDLES, StreamingServerCase, StreamingServerHarness, StreamingTransport};

/// The transport-invariance contract: HTTP and gRPC reach the same product
/// decision, for the same reason, with the same absence of side effects.
///
/// The two fixtures differ only in their `transport` and `endpoint` blocks, so
/// a difference in the stable refusal text is attributable to the transport and
/// nothing else.
#[tokio::test]
async fn http_and_grpc_rows_agree_on_outcome_and_leave_the_endpoint_untouched() {
    let mut refusals = Vec::new();
    for transport in [StreamingTransport::Http, StreamingTransport::Grpc] {
        let case = StreamingServerCase::single_process("single_process", transport);
        let name = format!("{}/{transport:?}", case.name);
        let harness = StreamingServerHarness::start(case);
        let outcome = harness.profile();

        outcome.assert_refused_naming(&name, &["scheduled_request", "available:"]);
        assert_eq!(
            harness.endpoint_issues().await,
            0,
            "{name}: a run refused at capability agreement must issue no request"
        );
        assert!(
            outcome.measurement_artifacts().is_empty(),
            "{name}: a refused run must emit no measurement artifact, found {:?}",
            outcome.measurement_artifacts()
        );
        outcome.assert_no_raw_or_secret_leak(LEAK_NEEDLES);
        refusals.push(outcome.stable_refusal());
    }

    assert_eq!(
        refusals[0], refusals[1],
        "HTTP and gRPC must reach the same product decision for the same reason"
    );
}

/// A cellular topology refuses before any prepare, release, or endpoint issue.
///
/// This is the product evidence the plan places here rather than in the
/// socket-free suite: a controller that partitioned work and launched cells
/// before resolving its stream resource would leave prepared state and issued
/// requests behind. Neither may exist, and the checkpoint root must stay empty.
///
/// The cellular refusal today is *earlier* than the single-process one — the
/// partitioner reads an authored `datasets` array that a stream run does not
/// have, so it refuses before streaming capability agreement is ever reached.
/// The row therefore asserts the invariant (nothing prepared, nothing issued)
/// rather than a specific message, and pins only that the refusal names the
/// resource it could not partition. When cellular stream partitioning lands,
/// this row converges on the same `scheduled_request` refusal as its
/// single-process twin.
#[tokio::test]
async fn cellular_topology_refuses_before_any_prepare_or_endpoint_issue() {
    let case = StreamingServerCase::cellular("cellular", StreamingTransport::Http, 2);
    let harness = StreamingServerHarness::start(case);
    let outcome = harness.profile();

    outcome.assert_refused_naming("cellular", &["datasets"]);
    assert_eq!(
        harness.endpoint_issues().await,
        0,
        "cellular: no cell may issue a request before the controller resolves its bindings"
    );
    assert_eq!(
        harness.captured_inference_requests(),
        0,
        "cellular: the mock retained a request body, so something reached the endpoint"
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
}

/// The registered composition normalizes and resolves natively, against a live
/// endpoint, without Python and without naming an identifier it never authored.
///
/// `config validate` is the streaming surface that is complete today; running
/// it with a reachable server present proves the static registry stage opens no
/// socket of its own.
#[tokio::test]
async fn streaming_validation_is_native_and_opens_no_socket() {
    let case = StreamingServerCase::single_process("validate", StreamingTransport::Http);
    let harness = StreamingServerHarness::start(case);
    let outcome = harness.validate();

    let output = outcome.combined_output();
    assert!(
        !output.contains("Traceback") && !output.contains("ModuleNotFoundError"),
        "streaming validation must be native, not delegated to Python:\n{output}"
    );
    assert_eq!(
        harness.endpoint_issues().await,
        0,
        "validation must not contact the endpoint"
    );
    assert!(
        outcome.measurement_artifacts().is_empty(),
        "validation must emit no measurement artifact, found {:?}",
        outcome.measurement_artifacts()
    );
    outcome.assert_no_raw_or_secret_leak(LEAK_NEEDLES);
}
