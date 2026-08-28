// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Streaming capability inventory integration tests.
//!
//! The stock distribution registers no streaming action sink on this head, so
//! both invariants below are stated over the registered inventory rather than
//! over one named sink: they hold vacuously today and arm themselves the moment
//! a stock sink is registered, which is exactly when they must start failing a
//! wrong descriptor.

use aiperf_runtime::extensions::AIPerfRegistry;
use aiperf_runtime::streaming::action::EndpointRetrySafety;

#[test]
fn stock_action_sinks_accept_the_dry_run_transport() {
    let registry = AIPerfRegistry::builtin().expect("the stock registry composes");
    for descriptor in registry.stream_action_sink_descriptors() {
        assert!(
            descriptor.transport_ids.contains(&"dry_run"),
            "stock action sink {:?} must accept the dry_run transport; got {:?}",
            descriptor.id,
            descriptor.transport_ids
        );
        assert!(
            descriptor.supports_virtual_clock,
            "stock action sink {:?} must support the simulated clock --dry-run selects",
            descriptor.id
        );
    }
}

#[test]
fn stock_action_sinks_do_not_claim_endpoint_retry_safety() {
    let registry = AIPerfRegistry::builtin().expect("the stock registry composes");
    for descriptor in registry.stream_action_sink_descriptors() {
        assert_eq!(
            descriptor.endpoint_retry_safety,
            EndpointRetrySafety::Unproven,
            "stock action sink {:?} must prove retry safety before claiming it",
            descriptor.id
        );
    }
}

#[test]
fn endpoint_retry_safety_defaults_to_the_refusing_variant() {
    assert_eq!(
        EndpointRetrySafety::default(),
        EndpointRetrySafety::Unproven,
        "an unannotated sink must not be retryable by omission"
    );
}
