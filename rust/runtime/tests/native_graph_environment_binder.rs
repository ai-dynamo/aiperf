// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph rollout environment-binder selection contracts.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    fs,
    num::NonZeroUsize,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use aiperf_runtime::graph::{
    agent::{LiveAgentPolicyDecisionReader, LiveAgentPolicyDecisionRequest},
    tools::{
        EnvironmentResetRequest, EnvironmentSessionAuthority, EnvironmentStepRequest,
        EnvironmentStepper, EnvironmentStepperBinding, EnvironmentStepperError,
        EnvironmentStepperFactory,
    },
};
use aiperf_runtime::{
    eval::{
        AdapterExit, AdapterLifecycleDeadlines, AdapterProtocolConfig, AdapterProtocolFactory,
        AdapterRole, AdapterRuntimeFactory, AdapterSpawnRequest, AdapterSpawner,
        AdapterSupervisionError, AgentVariantRef, ArtifactDigest, ArtifactQuota, CancelReason,
        HarborImporter, HarborSource, ModelBindingId, ModelIdentity, MoveV1ActionEncoderFactory,
        NativeGraphAdapterRuntimeProvider, NativeGraphAdapterRuntimeResolution,
        NativeGraphEnvironmentStepperFactory, NativeGraphFactoryError,
        NativeGraphModelDecisionError, NativeGraphPolicyModelRuntime, NativeGraphSuiteManifest,
        NativeSourceAcquirer, PolicyIdentity, ProtocolCapability, ProtocolLimits,
        ResourceLeaseRequest, RuntimeIdentity, StrictAdapterProtocolFactory, SuiteRunId,
        SuiteTrialSpec, SupervisedAdapter, SupervisedEnvironmentStepperBinder, TrialBudget,
        TrialSpec, bind_native_graph_environment_stepper,
    },
    extensions::{AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory},
};
use async_trait::async_trait;

#[test]
fn builtin_registry_resolves_every_sealed_rollout_environment_selector() {
    let imported = import_rollout_fixture();
    let rollout = imported
        .package
        .native_graph()
        .and_then(|native| native.rollout())
        .expect("fixture imports one sealed rollout selection");
    let environment = rollout.environment();
    let registry = BuiltinAIPerfRegistryFactory
        .build()
        .expect("the built-in registry is available");

    assert!(
        registry
            .native_graph_protocol(environment.protocol_factory_id().as_str())
            .is_some(),
        "the sealed protocol selector must resolve from the frozen registry"
    );
    assert!(
        registry
            .native_graph_adapter_runtime(environment.runtime_provider_id().as_str())
            .is_some(),
        "the sealed runtime selector must resolve from the frozen registry"
    );
    assert!(
        registry
            .native_graph_environment_stepper(environment.stepper_factory_id().as_str())
            .is_some(),
        "the sealed stepper selector must resolve from the frozen registry"
    );
    assert!(
        registry
            .native_graph_action_encoder(environment.action_encoder_id().as_str())
            .is_some(),
        "the sealed action encoder selector must resolve from the frozen registry"
    );
}

#[test]
fn unknown_rollout_selector_refuses_before_any_adapter_start() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported.clone());
    let registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();

    let error = bind_native_graph_environment_stepper(&registry, &trial)
        .err()
        .expect("an unregistered protocol selection must fail closed");

    assert!(error.to_string().contains("protocol factory"));
}

#[test]
fn incompatible_runtime_admission_refuses_before_stepper_or_adapter_start() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let runtime_binds = Arc::new(AtomicUsize::new(0));
    let stepper_binds = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(WrongRoleRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
                binds: Arc::clone(&runtime_binds),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(CountingStepperBinder {
                binds: Arc::clone(&stepper_binds),
            }),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let error = bind_native_graph_environment_stepper(&registry, &trial)
        .err()
        .expect("a non-environment runtime configuration must be rejected");

    assert!(error.to_string().contains("runtime protocol configuration"));
    assert_eq!(runtime_starts.load(Ordering::Relaxed), 0);
    assert_eq!(runtime_binds.load(Ordering::Relaxed), 0);
    assert_eq!(stepper_binds.load(Ordering::Relaxed), 0);
}

#[test]
fn missing_selected_action_encoder_refuses_before_runtime_or_stepper_binding() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let stepper_binds = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
                sends: Arc::new(AtomicUsize::new(0)),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(CountingStepperBinder {
                binds: Arc::clone(&stepper_binds),
            }),
        )
        .expect("test stepper registration succeeds");

    let error = bind_native_graph_environment_stepper(&registry, &trial)
        .err()
        .expect("a missing package-selected action encoder must fail closed");

    assert!(error.to_string().contains("action encoder"));
    assert_eq!(runtime_starts.load(Ordering::Relaxed), 0);
    assert_eq!(stepper_binds.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn selected_worker_local_components_preserve_rollout_admission_through_stepper_start() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported.clone());
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
                sends: Arc::new(AtomicUsize::new(0)),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(SupervisedEnvironmentStepperBinder),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let spawner = Rc::new(CountingSpawner::default());
    let bound = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the selected components bind one worker-local environment stepper");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("the package-selected store quota is valid"),
    ));
    let started = bound
        .start(
            "root",
            Rc::clone(&store),
            Rc::clone(&spawner) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the selected runtime and supervised stepper start with sealed admission");

    assert_eq!(bound.adapter().id.as_str(), "environment-adapter");
    assert_eq!(bound.package_identity(), &imported.task.digest);
    assert_eq!(bound.action_encoder_id().as_str(), "move_v1");
    assert_eq!(bound.action_encoder().id().as_str(), "move_v1");
    assert_eq!(bound.action_encoding_limits().max_decision_bytes(), 256);
    assert_eq!(bound.action_encoding_limits().max_action_bytes(), 3_072);
    assert_eq!(bound.operation_deadline(), Duration::from_secs(5));
    assert_eq!(runtime_starts.load(Ordering::Relaxed), 1);
    assert_eq!(spawner.starts.get(), 0);
    assert_eq!(
        store
            .borrow()
            .read_frozen(started.reset_input().artifact())
            .expect("reset input is frozen through the Rust-owned store"),
        b"{\"seed\":7}\n"
    );
    drop(started.into_stepper());
}

#[tokio::test]
async fn foreign_same_selector_session_decision_is_refused_before_stepper_dispatch() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let adapter_sends = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
                sends: Arc::clone(&adapter_sends),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(SupervisedEnvironmentStepperBinder),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let target = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the target worker-local session binds");
    let foreign = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the foreign worker-local session binds");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("the package-selected store quota is valid"),
    ));
    let started = target
        .start(
            "root",
            Rc::clone(&store),
            Rc::new(CountingSpawner::default()) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the target stepper starts");
    let foreign_prepared = foreign
        .prepare_live_rollout_coordinator(Box::new(ForeignSessionRuntime::new()))
        .expect("the foreign immutable rollout prepares its selected model runtime");
    let error = match started.bind_live_rollout_coordinator(foreign_prepared) {
        Ok(_) => {
            panic!("a foreign immutable binding must not attach a coordinator to this stepper")
        }
        Err(error) => error,
    };

    assert!(
        error.to_string().contains("another environment binding"),
        "unexpected foreign-session rejection: {error}"
    );
    assert_eq!(adapter_sends.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn one_bound_stepper_mints_distinct_authority_for_each_started_session() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let stepper_dispatches = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::new(AtomicUsize::new(0)),
                sends: Arc::new(AtomicUsize::new(0)),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(DispatchRecordingStepperBinder {
                dispatches: Arc::clone(&stepper_dispatches),
            }),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let bound = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the immutable package admission binds once");
    let first_prepared = bound
        .prepare_live_rollout_coordinator(Box::new(ForeignSessionRuntime::new()))
        .expect("the first selected model runtime prepares against the immutable binding");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let first_store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("first package-selected store opens"),
    ));
    let second_store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("second package-selected store opens"),
    ));
    let first = bound
        .start(
            "first",
            Rc::clone(&first_store),
            Rc::new(CountingSpawner::default()) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the first worker session starts");
    let mut second = bound
        .start(
            "second",
            Rc::clone(&second_store),
            Rc::new(CountingSpawner::default()) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the second worker session starts");
    let mut first_coordinator = first
        .bind_live_rollout_coordinator(first_prepared)
        .expect("the prepared model runtime binds only to the first started session");
    let observation = freeze_observation(&mut first_store.borrow_mut());
    let decision = first_coordinator
        .decide_policy_decision(&observation, &mut first_store.borrow_mut())
        .await
        .expect("the first session receives one sealed decision");

    let error = second
        .step_policy_decision("cross-session", decision)
        .await
        .expect_err("a decision from the first start must not reach the second start");

    assert!(
        error
            .to_string()
            .contains("another worker-local rollout session"),
        "cross-start decision must fail with session authority: {error}"
    );
    assert_eq!(
        stepper_dispatches.load(Ordering::Relaxed),
        0,
        "a cross-session decision must be rejected before the selected stepper dispatches"
    );
}

#[tokio::test]
async fn started_session_binds_only_one_live_rollout_coordinator() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let stepper_dispatches = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::new(AtomicUsize::new(0)),
                sends: Arc::new(AtomicUsize::new(0)),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(DispatchRecordingStepperBinder {
                dispatches: Arc::clone(&stepper_dispatches),
            }),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let bound = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the immutable package admission binds once");
    let first_prepared = bound
        .prepare_live_rollout_coordinator(Box::new(ForeignSessionRuntime::new()))
        .expect("the first selected model runtime prepares against the immutable binding");
    let second_prepared = bound
        .prepare_live_rollout_coordinator(Box::new(ForeignSessionRuntime::new()))
        .expect("the second selected model runtime prepares against the immutable binding");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("package-selected store opens"),
    ));
    let started = bound
        .start(
            "root",
            store,
            Rc::new(CountingSpawner::default()) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the selected worker session starts");
    let _first = started
        .bind_live_rollout_coordinator(first_prepared)
        .expect("the first prepared coordinator binds once");
    let error = match started.bind_live_rollout_coordinator(second_prepared) {
        Ok(_) => panic!("a started worker session must reject a second live rollout coordinator"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("already has a live rollout coordinator"),
        "unexpected repeated-bind rejection: {error}"
    );
    assert_eq!(
        stepper_dispatches.load(Ordering::Relaxed),
        0,
        "a repeated coordinator bind must fail before environment-stepper dispatch"
    );
}

#[tokio::test]
async fn raw_encoder_action_cannot_bypass_the_started_session_boundary() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let adapter_sends = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::new(AtomicUsize::new(0)),
                sends: Arc::clone(&adapter_sends),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(SupervisedEnvironmentStepperBinder),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let bound = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the immutable package admission binds once");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("package-selected store opens"),
    ));
    let started = bound
        .start(
            "root",
            Rc::clone(&store),
            Rc::new(CountingSpawner::default()) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the selected environment stepper starts");
    let raw_action = bound
        .action_encoder()
        .admit(
            aiperf_runtime::eval::DeclaredPolicyDecision::from_json_bytes(
                bound.action_encoder_id().clone(),
                br#"{"kind":"move","direction":"north"}"#,
                bound.action_encoding_limits(),
            )
            .expect("raw same-selector decision parses"),
            &mut store.borrow_mut(),
            bound.action_encoding_limits(),
        )
        .expect("the public raw encoder path can mint a pre-start action");

    let mut raw_stepper = started.into_stepper();
    let error = raw_stepper
        .step(EnvironmentStepRequest::admitted("raw-bypass", raw_action))
        .await
        .expect_err("a raw admitted action must not bypass the started session authority");

    assert!(
        matches!(error, EnvironmentStepperError::ActionSessionMismatch),
        "unexpected raw-action bypass result: {error}"
    );
    assert_eq!(
        adapter_sends.load(Ordering::Relaxed),
        0,
        "a raw encoder capability must be refused before adapter dispatch"
    );
}

#[tokio::test]
async fn same_session_coordinator_decision_reaches_the_selected_stepper_once() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let runtime_starts = Arc::new(AtomicUsize::new(0));
    let stepper_dispatches = Arc::new(AtomicUsize::new(0));
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(RecordingRuntimeProvider {
                starts: Arc::clone(&runtime_starts),
                sends: Arc::new(AtomicUsize::new(0)),
            }),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(DispatchRecordingStepperBinder {
                dispatches: Arc::clone(&stepper_dispatches),
            }),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let bound = bind_native_graph_environment_stepper(&registry, &trial)
        .expect("the worker-local session binds");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("the package-selected store quota is valid"),
    ));
    let prepared = bound
        .prepare_live_rollout_coordinator(Box::new(ForeignSessionRuntime::new()))
        .expect("the selected model runtime prepares against this immutable binding");
    let mut started = bound
        .start(
            "root",
            Rc::clone(&store),
            Rc::new(CountingSpawner::default()) as Rc<dyn AdapterSpawner>,
        )
        .await
        .expect("the selected stepper starts");
    let mut coordinator = started
        .bind_live_rollout_coordinator(prepared)
        .expect("the prepared model runtime binds to this started session");
    let observation = {
        let mut artifacts = store.borrow_mut();
        let upload = artifacts.begin_upload(14).expect("observation reserves");
        artifacts
            .write_upload(&upload, &mut std::io::Cursor::new(br#"{"position":0}"#))
            .expect("observation writes");
        artifacts
            .commit_upload(&upload)
            .expect("observation freezes")
    };
    let decision = coordinator
        .decide_policy_decision(&observation, &mut store.borrow_mut())
        .await
        .expect("the coordinator issues a decision for this session");

    started
        .step_policy_decision("step-0", decision)
        .await
        .expect_err("the recording adapter ends after observing one dispatched request");

    assert_eq!(runtime_starts.load(Ordering::Relaxed), 0);
    assert_eq!(stepper_dispatches.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn sealed_start_mints_adapter_request_without_caller_argv_or_model_secret() {
    let imported = import_rollout_fixture();
    let trial = resolve_rollout_trial(imported);
    let mut registry = aiperf_runtime::extensions::AIPerfRegistry::empty_or_base();
    registry
        .register_native_graph_protocol("strict_jsonl", Arc::new(StrictAdapterProtocolFactory))
        .expect("test protocol registration succeeds");
    registry
        .register_native_graph_adapter_runtime(
            "strict_supervised",
            Arc::new(CapturingRuntimeProvider),
        )
        .expect("test runtime registration succeeds");
    registry
        .register_native_graph_environment_stepper(
            "supervised_environment",
            Arc::new(SupervisedEnvironmentStepperBinder),
        )
        .expect("test stepper registration succeeds");
    registry
        .register_native_graph_action_encoder("move_v1", Arc::new(MoveV1ActionEncoderFactory))
        .expect("test action encoder registration succeeds");
    let spawner = Rc::new(CapturingSpawner::default());
    let bound =
        bind_native_graph_environment_stepper(&registry, &trial).expect("selected components bind");
    let store_root = tempfile::tempdir().expect("artifact-store root");
    let store = Rc::new(RefCell::new(
        aiperf_runtime::eval::EpisodeArtifactStore::new(
            store_root.path(),
            ArtifactQuota {
                max_artifacts: 8,
                max_total_bytes: 16_384,
                max_artifact_bytes: 3_072,
                max_download_handles: 4,
            },
        )
        .expect("store opens"),
    ));
    let error = match bound
        .start("root", store, Rc::clone(&spawner) as Rc<dyn AdapterSpawner>)
        .await
    {
        Ok(_) => panic!("the capturing runtime must refuse the test start"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("capturing test spawner"));
    let requests = spawner.requests.borrow();
    assert_eq!(requests.len(), 1);
    let request = requests.first().expect("the sealed request is retained");
    assert_eq!(request.argv(), ["tools/environment.sh"]);
    assert!(request.environment().is_empty());
    assert!(!request.environment().contains_key("MODEL_SECRET"));
    assert_eq!(request.deadlines().operation(), Duration::from_secs(5));
}

#[derive(Default)]
struct CountingSpawner {
    starts: Cell<usize>,
}

impl AdapterSpawner for CountingSpawner {
    fn begin_spawn(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn aiperf_runtime::eval::AdapterSpawnTransaction>, AdapterSupervisionError>
    {
        self.starts.set(self.starts.get() + 1);
        Err(AdapterSupervisionError::Process(
            "test spawner must not start".to_owned(),
        ))
    }
}

#[derive(Default)]
struct CapturingSpawner {
    requests: RefCell<Vec<AdapterSpawnRequest>>,
}

impl AdapterSpawner for CapturingSpawner {
    fn begin_spawn(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn aiperf_runtime::eval::AdapterSpawnTransaction>, AdapterSupervisionError>
    {
        self.requests.borrow_mut().push(request);
        Err(AdapterSupervisionError::Process(
            "capturing test spawner refuses launch".to_owned(),
        ))
    }
}

struct RecordingRuntimeProvider {
    starts: Arc<AtomicUsize>,
    sends: Arc<AtomicUsize>,
}

struct CapturingRuntimeProvider;

impl NativeGraphAdapterRuntimeProvider for CapturingRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        _: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        Ok(Rc::new(CapturingRuntimeResolution { config }))
    }
}

struct CapturingRuntimeResolution {
    config: AdapterProtocolConfig,
}

impl NativeGraphAdapterRuntimeResolution for CapturingRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(CapturingRuntime {
            config: self.config.clone(),
            spawner,
        }))
    }
}

struct CapturingRuntime {
    config: AdapterProtocolConfig,
    spawner: Rc<dyn AdapterSpawner>,
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for CapturingRuntime {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        let _ = self.spawner.begin_spawn(request)?;
        Err(AdapterSupervisionError::Process(
            "capturing test transaction unexpectedly started".to_owned(),
        ))
    }
}

impl NativeGraphAdapterRuntimeProvider for RecordingRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        _: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        Ok(Rc::new(RecordingRuntimeResolution {
            config,
            starts: Arc::clone(&self.starts),
            sends: Arc::clone(&self.sends),
        }))
    }
}

struct RecordingRuntimeResolution {
    config: AdapterProtocolConfig,
    starts: Arc<AtomicUsize>,
    sends: Arc<AtomicUsize>,
}

impl NativeGraphAdapterRuntimeResolution for RecordingRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        _: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(RecordingRuntime {
            config: self.config.clone(),
            starts: Arc::clone(&self.starts),
            sends: Arc::clone(&self.sends),
        }))
    }
}

struct WrongRoleRuntimeProvider {
    starts: Arc<AtomicUsize>,
    binds: Arc<AtomicUsize>,
}

impl NativeGraphAdapterRuntimeProvider for WrongRoleRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        _: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        let config = AdapterProtocolConfig::new(
            AdapterRole::Tool,
            config.episode(),
            [ProtocolCapability::Tool].into_iter().collect(),
            BTreeSet::new(),
            ProtocolLimits::default(),
        )
        .map_err(|error| NativeGraphFactoryError::new(error.to_string()))?;
        Ok(Rc::new(WrongRoleRuntimeResolution {
            config,
            starts: Arc::clone(&self.starts),
            binds: Arc::clone(&self.binds),
        }))
    }
}

struct WrongRoleRuntimeResolution {
    config: AdapterProtocolConfig,
    starts: Arc<AtomicUsize>,
    binds: Arc<AtomicUsize>,
}

impl NativeGraphAdapterRuntimeResolution for WrongRoleRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        self.binds.fetch_add(1, Ordering::Relaxed);
        let _ = spawner.begin_spawn(adapter_spawn_request(Duration::from_secs(5)));
        Ok(Rc::new(RecordingRuntime {
            config: self.config.clone(),
            starts: Arc::clone(&self.starts),
            sends: Arc::new(AtomicUsize::new(0)),
        }))
    }
}

struct RecordingRuntime {
    config: AdapterProtocolConfig,
    starts: Arc<AtomicUsize>,
    sends: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for RecordingRuntime {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        self.starts.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(RecordingAdapter {
            sends: Arc::clone(&self.sends),
        }))
    }
}

struct RecordingAdapter {
    sends: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl SupervisedAdapter for RecordingAdapter {
    async fn send(
        &mut self,
        _: aiperf_runtime::eval::HostEnvelope,
    ) -> Result<(), AdapterSupervisionError> {
        self.sends.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn receive(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        Err(AdapterSupervisionError::EndOfStream)
    }

    async fn receive_heartbeat(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn receive_idle(
        &mut self,
    ) -> Result<aiperf_runtime::eval::AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn reset(
        &mut self,
        _: aiperf_runtime::eval::HostEnvelope,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    fn release_download_handle(
        &mut self,
        _: &aiperf_runtime::eval::ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn cancel_and_reap(
        &mut self,
        _: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        Ok(AdapterExit::Reaped)
    }
}

struct ForeignSessionRuntime {
    binding: ModelBindingId,
}

impl ForeignSessionRuntime {
    fn new() -> Self {
        Self {
            binding: serde_json::from_str("\"primary\"").expect("fixture binding is valid"),
        }
    }
}

#[async_trait(?Send)]
impl NativeGraphPolicyModelRuntime for ForeignSessionRuntime {
    fn binding(&self) -> &ModelBindingId {
        &self.binding
    }

    async fn open_decision(
        &mut self,
        _: &LiveAgentPolicyDecisionRequest,
    ) -> Result<Box<dyn LiveAgentPolicyDecisionReader>, NativeGraphModelDecisionError> {
        Ok(Box::new(StaticDecisionReader {
            bytes: br#"{"kind":"move","direction":"north"}"#,
            offset: 0,
        }))
    }
}

struct StaticDecisionReader {
    bytes: &'static [u8],
    offset: usize,
}

#[async_trait(?Send)]
impl LiveAgentPolicyDecisionReader for StaticDecisionReader {
    async fn read(
        &mut self,
        destination: &mut [u8],
    ) -> Result<usize, aiperf_runtime::graph::agent::AgentLoopError> {
        let remaining = &self.bytes[self.offset..];
        let count = remaining.len().min(destination.len());
        destination[..count].copy_from_slice(&remaining[..count]);
        self.offset += count;
        Ok(count)
    }
}

struct CountingStepperBinder {
    binds: Arc<AtomicUsize>,
}

impl NativeGraphEnvironmentStepperFactory for CountingStepperBinder {
    fn bind(
        &self,
        _: Rc<dyn AdapterRuntimeFactory>,
    ) -> Result<Rc<dyn EnvironmentStepperFactory>, NativeGraphFactoryError> {
        self.binds.fetch_add(1, Ordering::Relaxed);
        Err(NativeGraphFactoryError::new(
            "test stepper binder must not run",
        ))
    }
}

struct DispatchRecordingStepperBinder {
    dispatches: Arc<AtomicUsize>,
}

impl NativeGraphEnvironmentStepperFactory for DispatchRecordingStepperBinder {
    fn bind(
        &self,
        _: Rc<dyn AdapterRuntimeFactory>,
    ) -> Result<Rc<dyn EnvironmentStepperFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(DispatchRecordingStepperFactory {
            dispatches: Arc::clone(&self.dispatches),
        }))
    }
}

struct DispatchRecordingStepperFactory {
    dispatches: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl EnvironmentStepperFactory for DispatchRecordingStepperFactory {
    async fn start(
        &self,
        _: EnvironmentStepperBinding,
        _: EnvironmentSessionAuthority,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn EnvironmentStepper>, EnvironmentStepperError> {
        Ok(Box::new(DispatchRecordingStepper {
            dispatches: Arc::clone(&self.dispatches),
        }))
    }
}

struct DispatchRecordingStepper {
    dispatches: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl EnvironmentStepper for DispatchRecordingStepper {
    async fn reset(
        &mut self,
        _: EnvironmentResetRequest,
    ) -> Result<aiperf_runtime::graph::tools::EnvironmentResetRecord, EnvironmentStepperError> {
        Err(EnvironmentStepperError::AlreadyReset)
    }

    async fn step(
        &mut self,
        _: EnvironmentStepRequest,
    ) -> Result<aiperf_runtime::eval::EnvironmentTransitionRecord, EnvironmentStepperError> {
        self.dispatches.fetch_add(1, Ordering::Relaxed);
        Err(EnvironmentStepperError::EpisodeTerminal)
    }

    async fn cancel_and_reap(&mut self) -> Result<(), EnvironmentStepperError> {
        // This in-memory fake owns no adapter child.
        Ok(())
    }
}

fn adapter_spawn_request(operation: Duration) -> AdapterSpawnRequest {
    let deadlines = AdapterLifecycleDeadlines::new(
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        operation,
        Duration::from_secs(1),
        Duration::from_secs(1),
    )
    .expect("test deadlines are valid");
    AdapterSpawnRequest::for_non_model_adapter(
        ["environment-adapter".to_owned()],
        BTreeMap::new(),
        deadlines,
    )
    .expect("test spawn request is valid")
}

fn freeze_observation(
    store: &mut aiperf_runtime::eval::EpisodeArtifactStore,
) -> aiperf_runtime::eval::FrozenArtifact {
    let bytes = br#"{\"position\":0}"#;
    let upload = store
        .begin_upload(u64::try_from(bytes.len()).expect("fixture bytes fit u64"))
        .expect("observation reserves");
    store
        .write_upload(&upload, &mut std::io::Cursor::new(bytes))
        .expect("observation writes");
    store.commit_upload(&upload).expect("observation freezes")
}

fn resolve_rollout_trial(
    imported: aiperf_runtime::eval::ImportedTask,
) -> aiperf_runtime::eval::ResolvedEpisodeTrial {
    let trial = TrialSpec::new(
        imported.task.clone(),
        AgentVariantRef::new("native-graph").expect("fixture agent is valid"),
        ModelIdentity::new("provider-default", "example-model").expect("fixture model is valid"),
        7,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        TrialBudget::new(30.0, 30.0).expect("fixture budget is valid"),
        ArtifactDigest::from_bytes(b"environment"),
        ArtifactDigest::from_bytes(b"verifier"),
        RuntimeIdentity::new("native").expect("fixture runtime is valid"),
    )
    .expect("fixture trial is valid");
    NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported,
            trial,
            NonZeroUsize::new(1).expect("one repetition is nonzero"),
            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).expect("fixture resources are valid"),
        )
        .expect("fixture suite trial resolves"),
    ])
    .expect("fixture suite manifest resolves")
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        b"environment-binder-run",
    )))
    .expect("fixture suite expands")
    .trials()
    .first()
    .expect("fixture contains exactly one resolved trial")
    .clone()
}

fn import_rollout_fixture() -> aiperf_runtime::eval::ImportedTask {
    let task = tempfile::tempdir().expect("temporary task root");
    fs::create_dir_all(task.path().join("environment")).expect("task environment directory");
    fs::create_dir_all(task.path().join("tests")).expect("task tests directory");
    fs::create_dir_all(task.path().join("tools")).expect("adapter directory");
    fs::create_dir_all(task.path().join("rollout")).expect("rollout directory");
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .expect("task Dockerfile");
    fs::write(task.path().join("instruction.md"), b"Do work.\n").expect("task instruction");
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("task verifier");
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-environment-binder"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("task manifest");
    fs::write(
        task.path().join("agent_graph.json"),
        br#"{
  "schema_version": "1.0", "trace_id": "environment-binder", "stage_bound": 1,
  "channels": { "output": { "type": "messages", "reducer": "add_messages" } },
  "nodes": [{ "id": "model", "kind": "model", "binding": "primary", "output": "output" }],
  "edges": [{ "source": "START", "target": "model" }, { "source": "model", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .expect("graph source");
    fs::write(
        task.path().join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
request_timeout_ms = 30000
capture = "none"

[model_bindings.tokenizer]
type = "local"
name = "tiktoken"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
"#,
    )
    .expect("model bindings");
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "environment-adapter"
role = "environment"
argv = ["tools/environment.sh"]
executable = "tools/environment.sh"
"#,
    )
    .expect("adapter manifest");
    fs::write(
        task.path().join("tools/environment.sh"),
        b"#!/bin/sh\nexit 0\n",
    )
    .expect("adapter executable");
    fs::write(task.path().join("rollout/reset.json"), b"{\"seed\":7}\n").expect("reset source");
    fs::write(
        task.path().join("rollout/policy.json"),
        b"{\"instruction\":\"choose a move\"}\n",
    )
    .expect("policy prompt source");
    fs::write(
        task.path().join("rollout.toml"),
        r#"[environment]
adapter_id = "environment-adapter"
protocol_factory_id = "strict_jsonl"
runtime_provider_id = "strict_supervised"
stepper_factory_id = "supervised_environment"
action_encoder_id = "move_v1"
operation_deadline_ms = 5000
reset_source = "rollout/reset.json"
max_frame_bytes = 4096
max_identifier_bytes = 128
max_json_bytes = 2048
max_json_depth = 4
max_json_array_entries = 8
max_json_object_entries = 8
max_operation_ledger_entries = 16
max_model_call_lineage_entries = 4
max_session_model_call_lineage_entries = 16
max_session_model_call_lineage_bytes = 2048
max_artifact_handles = 4
max_artifact_bytes = 4096

[artifacts]
max_artifacts = 8
max_total_bytes = 16384
max_artifact_bytes = 3072
max_download_handles = 4

[policy]
environment = "counter-v1"
model_binding_id = "primary"
prompt_source = "rollout/policy.json"
max_decision_bytes = 256
horizon = 4
gamma = 0.75

[limits]
max_environment_bytes = 256
max_horizon = 8
max_prompt_bytes = 256

[workspace_patch]
mutable_paths = ["result.txt"]
max_patches = 4
max_patch_bytes = 4096
max_total_patch_bytes = 8192
"#,
    )
    .expect("rollout manifest");
    let source = HarborSource::local(task.path().to_string_lossy())
        .expect("temporary task path is a valid source");
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .expect("fixture imports")
}
