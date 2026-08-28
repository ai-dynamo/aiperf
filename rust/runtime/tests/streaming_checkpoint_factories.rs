// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in checkpoint backend factory inventory and `none` refusal behavior.

use aiperf_runtime::{
    clock::{Clock, RealClock},
    extensions::{AIPerfRegistry, AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory},
    streaming::{
        checkpoint::{CheckpointError, StreamRunIdentity},
        checkpoint_backend::{
            CheckpointBackendPrepareContext, CheckpointBackendRequirements, CheckpointRetention,
            StreamingCheckpointBackend, StreamingCheckpointBackendFactory,
        },
        checkpoint_factories::{LOCAL_CHECKPOINT_BACKEND_ID, NONE_CHECKPOINT_BACKEND_ID},
    },
};
use serde_json::value::RawValue;
use std::rc::Rc;

/// Virtual clock every prepared backend reads lease deadlines from.
fn test_clock() -> Rc<dyn aiperf_runtime::clock::Clock> {
    Rc::new(aiperf_runtime::clock::SimClock::new())
}

#[path = "support/streaming_checkpoint.rs"]
mod support;

fn frozen_streaming_registry() -> AIPerfRegistry {
    BuiltinAIPerfRegistryFactory
        .build()
        .expect("built-in registry")
}

fn no_durability_requirements() -> CheckpointBackendRequirements {
    CheckpointBackendRequirements {
        needs_restartable_execution: false,
        needs_durable_partial_results: false,
    }
}

fn none_backend() -> Box<dyn StreamingCheckpointBackend> {
    let registry = frozen_streaming_registry();
    let factory = registry
        .stream_checkpoint_backend_factory(NONE_CHECKPOINT_BACKEND_ID)
        .expect("built-in none checkpoint backend");
    let authored = RawValue::from_string("{}".to_string()).expect("authored object");
    let config = factory
        .validate(&authored, &no_durability_requirements())
        .expect("checkpoint mode none needs no durability");
    factory
        .prepare(
            config,
            &CheckpointBackendPrepareContext {
                run: support::run_id(7),
                clock: test_clock(),
            },
        )
        .expect("prepared none backend")
}

#[test]
fn local_and_none_factories_are_registered() {
    let registry = frozen_streaming_registry();
    assert!(
        registry
            .stream_checkpoint_backend_factory("local")
            .is_some()
    );
    assert!(registry.stream_checkpoint_backend_factory("none").is_some());
    assert_eq!(LOCAL_CHECKPOINT_BACKEND_ID, "local");
    assert_eq!(NONE_CHECKPOINT_BACKEND_ID, "none");
}

#[tokio::test(flavor = "current_thread")]
async fn none_backend_refuses_begin_generation() {
    let backend = none_backend();
    let run: StreamRunIdentity = support::run_id(7);
    let error = backend
        .begin_generation(run, None, support::expectations(run))
        .await
        .err()
        .expect("none backend publishes no generation");
    assert!(matches!(error, CheckpointError::Storage { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn none_backend_advertises_no_resume_capability() {
    let registry = frozen_streaming_registry();
    let factory = registry
        .stream_checkpoint_backend_factory(NONE_CHECKPOINT_BACKEND_ID)
        .expect("built-in none checkpoint backend");
    let descriptor = factory.descriptor();
    assert!(!descriptor.is_durable);
    assert!(!descriptor.has_leased_readers);
    assert!(!descriptor.has_atomic_generations);
    assert!(!descriptor.has_result_segments);
    assert_eq!(descriptor.retention, CheckpointRetention::Ephemeral);

    // A run that must resume or retain partial results cannot select `none`.
    let authored = RawValue::from_string("{}".to_string()).expect("authored object");
    assert!(
        factory
            .validate(
                &authored,
                &CheckpointBackendRequirements {
                    needs_restartable_execution: true,
                    needs_durable_partial_results: false,
                },
            )
            .is_err()
    );

    // Opening the latest generation yields no resume claim rather than an error.
    let backend = none_backend();
    let run = support::run_id(7);
    let opened = backend
        .open_latest(&run, &support::expectations(run))
        .await
        .expect("none backend opens cleanly");
    assert!(opened.is_none());
}

#[test]
fn none_backend_advertises_no_encrypted_state_capability() {
    let registry = frozen_streaming_registry();
    let factory = registry
        .stream_checkpoint_backend_factory(NONE_CHECKPOINT_BACKEND_ID)
        .expect("built-in none checkpoint backend");
    assert!(!factory.descriptor().protects_sensitive_state);

    let authored = RawValue::from_string("{}".to_string()).expect("authored object");
    assert!(
        factory
            .validate(
                &authored,
                &CheckpointBackendRequirements {
                    needs_restartable_execution: false,
                    needs_durable_partial_results: true,
                },
            )
            .is_err()
    );
}

#[test]
fn local_factory_advertises_durable_generation_reachability() {
    let registry = frozen_streaming_registry();
    let descriptor = registry
        .stream_checkpoint_backend_factory(LOCAL_CHECKPOINT_BACKEND_ID)
        .expect("built-in local checkpoint backend")
        .descriptor();
    assert!(descriptor.is_durable);
    assert!(descriptor.has_leased_readers);
    assert!(descriptor.has_atomic_generations);
    assert!(descriptor.has_result_segments);
    assert_eq!(
        descriptor.retention,
        CheckpointRetention::GenerationReachability
    );
}

#[test]
fn local_factory_refuses_a_relative_store_root() {
    let registry = frozen_streaming_registry();
    let factory = registry
        .stream_checkpoint_backend_factory(LOCAL_CHECKPOINT_BACKEND_ID)
        .expect("built-in local checkpoint backend");
    let authored = RawValue::from_string("{\"root\":\"relative/store\"}".to_string())
        .expect("authored object");
    assert!(
        factory
            .validate(&authored, &no_durability_requirements())
            .is_err()
    );
}

#[test]
fn local_factory_prepares_storage_for_an_absolute_root() {
    let temporary = tempfile::tempdir().expect("temporary store root");
    let registry = frozen_streaming_registry();
    let factory = registry
        .stream_checkpoint_backend_factory(LOCAL_CHECKPOINT_BACKEND_ID)
        .expect("built-in local checkpoint backend");
    let authored =
        RawValue::from_string(serde_json::json!({ "root": temporary.path() }).to_string())
            .expect("authored object");
    let config = factory
        .validate(&authored, &no_durability_requirements())
        .expect("valid local configuration");
    let prepared = factory.prepare(
        config,
        &CheckpointBackendPrepareContext {
            run: support::run_id(3),
            clock: test_clock(),
        },
    );
    assert!(prepared.is_ok());
}
