// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow, independently selectable factories for NativeGraph-only behavior.
//!
//! Endpoint, transport, tokenizer, graph placement, clock, segment, observer,
//! and exporter selection remain in their existing AIPerf registries. This
//! module contains only NativeGraph behavior that needs an explicit extension
//! seam of its own.

use std::{
    cell::RefCell,
    collections::BTreeSet,
    fmt::{self, Display, Formatter},
    io::Cursor,
    rc::Rc,
    sync::Arc,
    time::Duration,
};

#[cfg(feature = "engine")]
use crate::extensions::AIPerfRegistry;
use crate::graph::tools::{
    EnvironmentArtifactBindings, EnvironmentEpisodeIdentity, EnvironmentResetRequest,
    EnvironmentSessionAuthority, EnvironmentStepRequest, EnvironmentStepper,
    EnvironmentStepperBinding, EnvironmentStepperFactory as WorkerEnvironmentStepperFactory,
    SupervisedEnvironmentStepperFactory,
};
use crate::{
    eval::semantic::GraphLowererFactory,
    eval::{AdapterSpec, ArtifactDigest, ProviderRecovery},
};
use async_trait::async_trait;
use bytes::Bytes;

use super::action_encoder::{
    ActionEncodingLimits, ActionSessionAuthority, BoundNativeGraphActionEncoder,
    EpisodeActionEncodingError, MoveV1ActionEncoder,
};
#[cfg(feature = "engine")]
use super::model_runtime::{
    IssuedNativeGraphPolicyDecision, NativeGraphLiveRolloutBindingAuthority,
    NativeGraphLiveRolloutCoordinator, NativeGraphLiveRolloutError,
    NativeGraphLiveRolloutSessionAuthority, NativeGraphPolicyModelRuntime,
    PreparedNativeGraphLiveRolloutCoordinator,
};
use super::package::ActionEncoderFactoryId;
#[cfg(feature = "engine")]
use super::package::{
    NativeGraphRolloutEnvironment, NativeGraphRolloutPlan, NativeGraphRolloutResetSource,
};
use super::{
    AdapterLifecycleDeadlines, AdapterProtocolConfig, AdapterProtocolFactory,
    AdapterRuntimeFactory, AdapterSpawnRequest, AdapterSpawner, ArtifactError, ArtifactQuota,
    FrozenArtifact, FrozenArtifactReference, FrozenRolloutEvidence, NativeGraphLowererFactory,
    NativeGraphLoweringReport, NativeGraphPackagePlan, NativeGraphProfile,
    NativeGraphRolloutReceipt, NativeGraphRolloutTransitionReceipt, ProtocolAdapterRuntimeFactory,
    ProtocolCapability, ProtocolLimits, RlEvaluationPolicy,
};
use super::{CompatibilityTerminalReceipt, ResolvedEpisodeTrial};

/// Failure while selecting or binding one NativeGraph-only extension seam.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGraphFactoryError(String);

impl NativeGraphFactoryError {
    /// Creates a stable, redacted factory-selection failure.
    pub fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }
}

impl Display for NativeGraphFactoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for NativeGraphFactoryError {}

/// Stateless provider that binds a lowerer only after package import freezes.
pub trait NativeGraphLowererProvider: Send + Sync {
    /// Binds one lowerer to the imported package snapshot and declared authority.
    fn bind(
        &self,
        package: &NativeGraphPackagePlan,
    ) -> Result<Arc<dyn GraphLowererFactory>, NativeGraphFactoryError>;
}

/// Built-in package-bound lowerer provider for schema-1.1 NativeGraph sources.
#[derive(Clone, Copy, Debug, Default)]
pub struct PackageNativeGraphLowererProvider;

impl NativeGraphLowererProvider for PackageNativeGraphLowererProvider {
    fn bind(
        &self,
        package: &NativeGraphPackagePlan,
    ) -> Result<Arc<dyn GraphLowererFactory>, NativeGraphFactoryError> {
        Ok(Arc::new(NativeGraphLowererFactory::new(package)))
    }
}

/// Factory that resolves one package-scoped, protocol-validated adapter runtime.
///
/// Resolution receives no spawn authority. The resolved worker-local object receives the
/// task-owned spawner only after the sealed package admission has succeeded.
pub trait NativeGraphAdapterRuntimeProvider {
    /// Resolves the provider's exact protocol admission without spawn authority.
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        protocol_factory: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError>;
}

/// One pure provider resolution which can receive task-owned spawn authority only at start.
pub trait NativeGraphAdapterRuntimeResolution {
    /// Returns the exact immutable protocol configuration admitted during resolution.
    fn protocol_config(&self) -> &AdapterProtocolConfig;

    /// Binds the task-owned spawner after the caller has completed all admission checks.
    fn bind(
        &self,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError>;
}

/// Built-in constructor for the existing strict supervised adapter runtime.
#[derive(Clone, Copy, Debug, Default)]
pub struct StrictAdapterRuntimeProvider;

impl NativeGraphAdapterRuntimeProvider for StrictAdapterRuntimeProvider {
    fn resolve(
        &self,
        config: AdapterProtocolConfig,
        protocol_factory: Rc<dyn AdapterProtocolFactory>,
    ) -> Result<Rc<dyn NativeGraphAdapterRuntimeResolution>, NativeGraphFactoryError> {
        Ok(Rc::new(StrictAdapterRuntimeResolution {
            config,
            protocol_factory,
        }))
    }
}

struct StrictAdapterRuntimeResolution {
    config: AdapterProtocolConfig,
    protocol_factory: Rc<dyn AdapterProtocolFactory>,
}

impl NativeGraphAdapterRuntimeResolution for StrictAdapterRuntimeResolution {
    fn protocol_config(&self) -> &AdapterProtocolConfig {
        &self.config
    }

    fn bind(
        &self,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<Rc<dyn AdapterRuntimeFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(ProtocolAdapterRuntimeFactory::new(
            self.config.clone(),
            Rc::clone(&self.protocol_factory),
            spawner,
        )))
    }
}

/// Stateless Rust-owned action-encoder implementation selected by an imported package.
pub trait NativeGraphActionEncoderFactory: Send + Sync {
    /// Returns the canonical selector this factory permits.
    fn id(&self) -> &str;

    /// Binds one fresh encoder after exact sealed-selector resolution and before adapter start.
    fn bind(
        &self,
        selection: &ActionEncoderFactoryId,
    ) -> Result<BoundNativeGraphActionEncoder, NativeGraphFactoryError>;
}

/// Built-in factory for the schema-1 `move_v1` decision/action contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveV1ActionEncoderFactory;

impl NativeGraphActionEncoderFactory for MoveV1ActionEncoderFactory {
    fn id(&self) -> &str {
        "move_v1"
    }

    fn bind(
        &self,
        selection: &ActionEncoderFactoryId,
    ) -> Result<BoundNativeGraphActionEncoder, NativeGraphFactoryError> {
        BoundNativeGraphActionEncoder::new(selection.clone(), Box::new(MoveV1ActionEncoder))
            .map_err(action_encoder_factory_error)
    }
}

fn action_encoder_factory_error(error: EpisodeActionEncodingError) -> NativeGraphFactoryError {
    NativeGraphFactoryError::new(format!(
        "NativeGraph action encoder binding failed: {error}"
    ))
}

/// Factory that binds a selected adapter runtime to one worker-local environment stepper.
pub trait NativeGraphEnvironmentStepperFactory: Send + Sync {
    /// Creates a worker-local environment-stepper factory over the selected runtime.
    fn bind(
        &self,
        runtime: Rc<dyn AdapterRuntimeFactory>,
    ) -> Result<Rc<dyn WorkerEnvironmentStepperFactory>, NativeGraphFactoryError>;
}

/// Host-owned stepper for legacy environment-adapter integrations.
///
/// New package-selected rollouts use [`NativeGraphEnvironmentStepperFactory`] to obtain the
/// strict worker-local stepper seam. This trait remains available for existing integrations that
/// own their own already-authorized stepping boundary.
pub trait NativeGraphEnvironmentStepper {
    /// Advances the environment by one already-authorized operation.
    fn step(&mut self) -> Result<(), NativeGraphFactoryError>;
}

/// Built-in binder for the strict supervised environment stepper.
#[derive(Clone, Copy, Debug, Default)]
pub struct SupervisedEnvironmentStepperBinder;

impl NativeGraphEnvironmentStepperFactory for SupervisedEnvironmentStepperBinder {
    fn bind(
        &self,
        runtime: Rc<dyn AdapterRuntimeFactory>,
    ) -> Result<Rc<dyn WorkerEnvironmentStepperFactory>, NativeGraphFactoryError> {
        Ok(Rc::new(SupervisedEnvironmentStepperFactory::new(runtime)))
    }
}

/// Explicit refusal for compositions that deliberately disable environment stepping.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefusingEnvironmentStepperFactory;

impl NativeGraphEnvironmentStepperFactory for RefusingEnvironmentStepperFactory {
    fn bind(
        &self,
        _: Rc<dyn AdapterRuntimeFactory>,
    ) -> Result<Rc<dyn WorkerEnvironmentStepperFactory>, NativeGraphFactoryError> {
        Err(NativeGraphFactoryError::new(
            "NativeGraph environment stepping is unavailable in this product composition",
        ))
    }
}

/// Immutable worker-local environment admission assembled from one package snapshot.
#[cfg(feature = "engine")]
pub struct BoundNativeGraphEnvironmentStepper {
    adapter: AdapterSpec,
    package_identity: ArtifactDigest,
    protocol: AdapterProtocolConfig,
    action_encoder_id: ActionEncoderFactoryId,
    action_encoder: BoundNativeGraphActionEncoder,
    action_encoding_limits: ActionEncodingLimits,
    policy: RlEvaluationPolicy,
    artifact_quota: ArtifactQuota,
    operation_deadline: Duration,
    reset_source: NativeGraphRolloutResetSource,
    rollout: NativeGraphRolloutPlan,
    rollout_binding: NativeGraphLiveRolloutBindingAuthority,
    runtime: Rc<dyn NativeGraphAdapterRuntimeResolution>,
    stepper_factory: Arc<dyn NativeGraphEnvironmentStepperFactory>,
}

#[cfg(feature = "engine")]
impl BoundNativeGraphEnvironmentStepper {
    /// Returns the exact environment adapter declared by the imported package.
    pub fn adapter(&self) -> &AdapterSpec {
        &self.adapter
    }

    /// Returns the imported package identity bound to the worker-local store session.
    pub fn package_identity(&self) -> &ArtifactDigest {
        &self.package_identity
    }

    /// Returns the exact protocol admission configuration selected by the package.
    pub fn protocol(&self) -> &AdapterProtocolConfig {
        &self.protocol
    }

    /// Returns the exact action-encoder selector retained from the imported package.
    pub fn action_encoder_id(&self) -> &ActionEncoderFactoryId {
        &self.action_encoder_id
    }

    /// Borrows the exact real action encoder bound before spawn authority exists.
    pub fn action_encoder(&self) -> &BoundNativeGraphActionEncoder {
        &self.action_encoder
    }

    /// Returns the exact decision and action byte limits derived from the sealed package.
    pub const fn action_encoding_limits(&self) -> ActionEncodingLimits {
        self.action_encoding_limits
    }

    /// Returns the sealed environment operation deadline.
    pub const fn operation_deadline(&self) -> Duration {
        self.operation_deadline
    }

    /// Returns the exact artifact quota sealed into the imported rollout selection.
    pub const fn artifact_quota(&self) -> ArtifactQuota {
        self.artifact_quota
    }

    /// Prepares a selected model runtime for this exact imported rollout before a worker starts.
    pub fn prepare_live_rollout_coordinator(
        &self,
        runtime: Box<dyn NativeGraphPolicyModelRuntime>,
    ) -> Result<PreparedNativeGraphLiveRolloutCoordinator, NativeGraphFactoryError> {
        NativeGraphLiveRolloutCoordinator::prepare(
            &self.rollout,
            self.rollout_binding.clone(),
            runtime,
        )
        .map_err(live_rollout_factory_error)
    }

    /// Starts the selected worker-local stepper and returns its only reset-input capability.
    pub async fn start(
        &self,
        span: impl Into<String>,
        store: Rc<RefCell<super::EpisodeArtifactStore>>,
        spawner: Rc<dyn AdapterSpawner>,
    ) -> Result<StartedNativeGraphEnvironmentStepper, NativeGraphFactoryError> {
        let request = self.sealed_spawn_request()?;
        self.start_with_request(span, store, spawner, request, false)
            .await
    }

    /// Starts the selected stepper from the backend's exact authorization-minted request.
    ///
    /// The backend retains the only request bearing the NativeGraph exact-spawn token; this
    /// worker seam verifies its immutable package-facing fields but never mints a replacement.
    pub(crate) async fn start_with_authorized_request(
        &self,
        span: impl Into<String>,
        store: Rc<RefCell<super::EpisodeArtifactStore>>,
        spawner: Rc<dyn AdapterSpawner>,
        request: AdapterSpawnRequest,
    ) -> Result<StartedNativeGraphEnvironmentStepper, NativeGraphFactoryError> {
        self.start_with_request(span, store, spawner, request, true)
            .await
    }

    async fn start_with_request(
        &self,
        span: impl Into<String>,
        store: Rc<RefCell<super::EpisodeArtifactStore>>,
        spawner: Rc<dyn AdapterSpawner>,
        request: AdapterSpawnRequest,
        defer_terminal_cleanup: bool,
    ) -> Result<StartedNativeGraphEnvironmentStepper, NativeGraphFactoryError> {
        let expected_argv = if defer_terminal_cleanup {
            self.adapter.container_argv()
        } else {
            self.adapter.argv.clone()
        };
        if request.argv() != expected_argv.as_slice() || !request.environment().is_empty() {
            return Err(NativeGraphFactoryError::new(
                "NativeGraph authorized environment spawn request does not match the imported adapter selection",
            ));
        }
        let rollout_session = NativeGraphLiveRolloutSessionAuthority::new();
        let action_session = ActionSessionAuthority::new();
        let reset_input = {
            let mut store = store.try_borrow_mut().map_err(|_| {
                NativeGraphFactoryError::new(
                    "NativeGraph environment artifact store is already borrowed",
                )
            })?;
            if store.quota() != self.artifact_quota {
                return Err(NativeGraphFactoryError::new(
                    "NativeGraph environment artifact store does not match the sealed quota",
                ));
            }
            self.freeze_reset_input(&mut store)?
        };
        let identity = EnvironmentEpisodeIdentity::new(
            self.package_identity.clone(),
            self.protocol.episode(),
            self.operation_deadline,
        )
        .map_err(|error| NativeGraphFactoryError::new(error.to_string()));
        let binding = match identity.and_then(|identity| {
            EnvironmentStepperBinding::new(
                self.protocol.clone(),
                identity,
                span,
                self.policy.clone(),
                EnvironmentArtifactBindings::new([reset_input.clone()])
                    .map_err(|error| NativeGraphFactoryError::new(error.to_string()))?,
            )
            .map(|binding| {
                binding.with_selected_action_encoder_session(
                    &self.action_encoder,
                    action_session.clone(),
                )
            })
            .map(|binding| {
                if defer_terminal_cleanup {
                    binding.with_deferred_terminal_cleanup()
                } else {
                    binding
                }
            })
            .map_err(|error| NativeGraphFactoryError::new(error.to_string()))
        }) {
            Ok(binding) => binding,
            Err(primary) => return Err(revoke_reset_reference(&store, &reset_input, primary)),
        };
        let authority =
            EnvironmentSessionAuthority::new(self.package_identity.clone(), Rc::clone(&store));
        let runtime = match self.runtime.bind(spawner) {
            Ok(runtime) if runtime.protocol_config() == Some(&self.protocol) => runtime,
            Ok(_) => {
                return Err(revoke_reset_reference(
                    &store,
                    &reset_input,
                    NativeGraphFactoryError::new(
                        "NativeGraph environment runtime protocol configuration does not match the sealed role and capabilities",
                    ),
                ));
            }
            Err(error) => {
                return Err(revoke_reset_reference(
                    &store,
                    &reset_input,
                    NativeGraphFactoryError::new(format!(
                        "NativeGraph environment runtime provider binding failed: {error}"
                    )),
                ));
            }
        };
        let factory = match self.stepper_factory.bind(runtime) {
            Ok(factory) => factory,
            Err(error) => {
                return Err(revoke_reset_reference(
                    &store,
                    &reset_input,
                    NativeGraphFactoryError::new(format!(
                        "NativeGraph environment stepper factory binding failed: {error}"
                    )),
                ));
            }
        };
        match factory.start(binding, authority, request).await {
            Ok(stepper) => Ok(StartedNativeGraphEnvironmentStepper {
                stepper,
                reset_input,
                action_encoder: self.action_encoder.clone(),
                action_encoding_limits: self.action_encoding_limits,
                policy: self.policy.clone(),
                rollout_binding: self.rollout_binding.clone(),
                rollout_session,
                coordinator_bind_token: RefCell::new(Some(())),
                action_session,
                artifacts: store,
            }),
            Err(error) => Err(revoke_reset_reference(
                &store,
                &reset_input,
                NativeGraphFactoryError::new(error.to_string()),
            )),
        }
    }

    fn freeze_reset_input(
        &self,
        store: &mut super::EpisodeArtifactStore,
    ) -> Result<FrozenArtifactReference, NativeGraphFactoryError> {
        let declared_bytes = u64::try_from(self.reset_source.bytes().len()).map_err(|_| {
            NativeGraphFactoryError::new("NativeGraph rollout reset source length does not fit u64")
        })?;
        store
            .preflight_reference()
            .map_err(|error| artifact_factory_error("preflight reset capability", error))?;
        let upload = store
            .begin_upload(declared_bytes)
            .map_err(|error| artifact_factory_error("reserve reset source", error))?;
        let write = store.write_upload(&upload, &mut Cursor::new(self.reset_source.bytes()));
        if let Err(primary) = write {
            return Err(abort_reset_upload(
                store,
                &upload,
                artifact_factory_error("write reset source", primary),
            ));
        }
        let artifact = match store.commit_upload(&upload) {
            Ok(artifact) => artifact,
            Err(primary) => {
                return Err(abort_reset_upload(
                    store,
                    &upload,
                    artifact_factory_error("freeze reset source", primary),
                ));
            }
        };
        store
            .issue_reference(&artifact)
            .map_err(|error| artifact_factory_error("grant reset capability", error))
    }

    fn sealed_spawn_request(&self) -> Result<AdapterSpawnRequest, NativeGraphFactoryError> {
        let defaults = AdapterLifecycleDeadlines::default();
        let deadlines = AdapterLifecycleDeadlines::new(
            defaults.startup(),
            defaults.reset(),
            defaults.heartbeat(),
            defaults.idle(),
            self.operation_deadline,
            defaults.cancel(),
            defaults.reap(),
        )
        .map_err(|error| NativeGraphFactoryError::new(error.to_string()))?;
        AdapterSpawnRequest::for_non_model_adapter(
            self.adapter.argv.clone(),
            Default::default(),
            deadlines,
        )
        .map_err(|error| NativeGraphFactoryError::new(error.to_string()))
    }
}

/// One started worker-local stepper paired with the reset-input capability Rust admitted.
#[cfg(feature = "engine")]
pub struct StartedNativeGraphEnvironmentStepper {
    stepper: Box<dyn EnvironmentStepper>,
    reset_input: FrozenArtifactReference,
    action_encoder: BoundNativeGraphActionEncoder,
    action_encoding_limits: ActionEncodingLimits,
    policy: RlEvaluationPolicy,
    rollout_binding: NativeGraphLiveRolloutBindingAuthority,
    rollout_session: NativeGraphLiveRolloutSessionAuthority,
    coordinator_bind_token: RefCell<Option<()>>,
    action_session: ActionSessionAuthority,
    artifacts: Rc<RefCell<super::EpisodeArtifactStore>>,
}

#[cfg(feature = "engine")]
impl StartedNativeGraphEnvironmentStepper {
    /// Borrows the one-shot reset-input capability frozen from the imported package snapshot.
    pub fn reset_input(&self) -> &FrozenArtifactReference {
        &self.reset_input
    }

    /// Starts a descriptor-only receipt under this session's immutable rollout policy.
    pub fn new_rollout_receipt(&self) -> NativeGraphRolloutReceipt {
        NativeGraphRolloutReceipt::new(self.policy.clone())
    }

    /// Resets the selected environment and returns only its frozen observation descriptor.
    pub async fn reset_live_rollout(
        &mut self,
        operation: impl Into<String>,
    ) -> Result<FrozenArtifact, NativeGraphFactoryError> {
        self.stepper
            .reset(EnvironmentResetRequest::new(
                operation,
                self.reset_input.clone(),
            ))
            .await
            .map(|record| record.observation().clone())
            .map_err(|error| NativeGraphFactoryError::new(error.to_string()))
    }

    /// Consumes exact prepared package facts to bind one coordinator to this started session.
    ///
    /// A successful bind consumes the worker-local bind token. A rejected prepared coordinator
    /// leaves that token available so the started session may still bind its matching coordinator.
    pub fn bind_live_rollout_coordinator(
        &self,
        prepared: PreparedNativeGraphLiveRolloutCoordinator,
    ) -> Result<NativeGraphLiveRolloutCoordinator, NativeGraphFactoryError> {
        let mut token = self.coordinator_bind_token.try_borrow_mut().map_err(|_| {
            NativeGraphFactoryError::new(
                "NativeGraph started worker session coordinator binding is already in progress",
            )
        })?;
        let Some(bind_token) = token.take() else {
            return Err(NativeGraphFactoryError::new(
                "NativeGraph started worker session already has a live rollout coordinator",
            ));
        };
        match prepared.into_coordinator(&self.rollout_binding, self.rollout_session.clone()) {
            Ok(coordinator) => Ok(coordinator),
            Err(error) => {
                *token = Some(bind_token);
                Err(live_rollout_factory_error(error))
            }
        }
    }

    /// Admits a sealed policy decision through this stepper's own selected encoder before
    /// dispatching its single-use action capability.
    pub async fn step_policy_decision(
        &mut self,
        operation: impl Into<String>,
        decision: IssuedNativeGraphPolicyDecision,
    ) -> Result<super::EnvironmentTransitionRecord, NativeGraphFactoryError> {
        if !decision.belongs_to(&self.rollout_session) {
            return Err(NativeGraphFactoryError::new(
                "NativeGraph policy decision belongs to another worker-local rollout session",
            ));
        }
        let action = {
            let mut artifacts = self.artifacts.try_borrow_mut().map_err(|_| {
                NativeGraphFactoryError::new(
                    "NativeGraph environment artifact store is already borrowed",
                )
            })?;
            self.action_encoder
                .admit_for_session(
                    decision.into_decision(),
                    &mut artifacts,
                    self.action_encoding_limits,
                    &self.action_session,
                )
                .map_err(action_encoder_factory_error)?
        };
        self.stepper
            .step(EnvironmentStepRequest::admitted(operation, action))
            .await
            .map_err(|error| NativeGraphFactoryError::new(error.to_string()))
    }

    /// Admits and dispatches one sealed policy decision while retaining a capability-free receipt.
    pub async fn step_policy_decision_receipt(
        &mut self,
        operation: impl Into<String>,
        decision: IssuedNativeGraphPolicyDecision,
    ) -> Result<NativeGraphRolloutTransitionReceipt, NativeGraphFactoryError> {
        if !decision.belongs_to(&self.rollout_session) {
            return Err(NativeGraphFactoryError::new(
                "NativeGraph policy decision belongs to another worker-local rollout session",
            ));
        }
        let action = {
            let mut artifacts = self.artifacts.try_borrow_mut().map_err(|_| {
                NativeGraphFactoryError::new(
                    "NativeGraph environment artifact store is already borrowed",
                )
            })?;
            self.action_encoder
                .admit_for_session(
                    decision.into_decision(),
                    &mut artifacts,
                    self.action_encoding_limits,
                    &self.action_session,
                )
                .map_err(action_encoder_factory_error)?
        };
        let action_descriptor = action.reference().artifact().clone();
        let transition = self
            .stepper
            .step(EnvironmentStepRequest::admitted(operation, action))
            .await
            .map_err(|error| NativeGraphFactoryError::new(error.to_string()))?;
        Ok(NativeGraphRolloutTransitionReceipt::from_stepper(
            action_descriptor,
            transition,
        ))
    }

    /// Uses the matching coordinator to decide and dispatch one descriptor-only rollout step.
    pub(crate) async fn step_live_rollout(
        &mut self,
        coordinator: &mut NativeGraphLiveRolloutCoordinator,
        operation: impl Into<String>,
        observation: &FrozenArtifact,
    ) -> Result<NativeGraphRolloutTransitionReceipt, NativeGraphFactoryError> {
        let observation = {
            let artifacts = self.artifacts.try_borrow().map_err(|_| {
                NativeGraphFactoryError::new(
                    "NativeGraph environment artifact store is already borrowed",
                )
            })?;
            artifacts
                .read_frozen(observation)
                .map(Bytes::from)
                .map_err(|error| NativeGraphFactoryError::new(error.to_string()))?
        };
        let decision = coordinator
            .decide_policy_decision_bytes(observation)
            .await
            .map_err(live_rollout_factory_error)?;
        self.step_policy_decision_receipt(operation, decision).await
    }

    /// Reaps a nonterminal environment adapter after callback or collection completion.
    pub async fn cancel_and_reap(&mut self) -> Result<(), NativeGraphFactoryError> {
        self.stepper
            .cancel_and_reap()
            .await
            .map_err(|error| NativeGraphFactoryError::new(error.to_string()))
    }

    /// Freezes this session's descriptor-only receipt against its private trusted artifact store.
    pub(crate) fn freeze_rollout_receipt(
        &self,
        receipt: NativeGraphRolloutReceipt,
        identity: super::RolloutEvidenceIdentity,
    ) -> Result<FrozenRolloutEvidence, NativeGraphFactoryError> {
        let artifacts = self.artifacts.try_borrow().map_err(|_| {
            NativeGraphFactoryError::new(
                "NativeGraph environment artifact store is already borrowed",
            )
        })?;
        receipt
            .freeze(identity, &artifacts)
            .map_err(|error| NativeGraphFactoryError::new(error.to_string()))
    }

    /// Transfers ownership of the selected worker-local environment stepper.
    pub fn into_stepper(self) -> Box<dyn EnvironmentStepper> {
        self.stepper
    }
}

/// Resolves the sealed rollout selectors into one worker-local supervised environment stepper.
#[cfg(feature = "engine")]
pub fn bind_native_graph_environment_stepper(
    registry: &AIPerfRegistry,
    trial: &super::ResolvedEpisodeTrial,
) -> Result<BoundNativeGraphEnvironmentStepper, NativeGraphFactoryError> {
    let package = trial.imported().package.native_graph().ok_or_else(|| {
        NativeGraphFactoryError::new(
            "NativeGraph environment stepping requires an imported NativeGraph package",
        )
    })?;
    if package.profile() != NativeGraphProfile::NativeGraph {
        return Err(NativeGraphFactoryError::new(
            "NativeGraph environment stepping requires the native_graph profile",
        ));
    }
    let rollout = package.rollout().ok_or_else(|| {
        NativeGraphFactoryError::new(
            "NativeGraph environment stepping requires a sealed rollout selection",
        )
    })?;
    let environment = rollout.environment();
    let adapter = package
        .adapters()
        .iter()
        .find(|adapter| adapter.id == *environment.adapter_id())
        .ok_or_else(|| {
            NativeGraphFactoryError::new(
                "NativeGraph rollout selected an adapter absent from the imported package",
            )
        })?;
    if adapter.role != super::AdapterRole::Environment {
        return Err(NativeGraphFactoryError::new(
            "NativeGraph rollout selected an adapter without the environment role",
        ));
    }
    let protocol_factory = registry
        .native_graph_protocol(environment.protocol_factory_id().as_str())
        .ok_or_else(|| {
            NativeGraphFactoryError::new(format!(
                "NativeGraph rollout selected unknown protocol factory {:?}",
                environment.protocol_factory_id().as_str()
            ))
        })?;
    let runtime_provider = registry
        .native_graph_adapter_runtime(environment.runtime_provider_id().as_str())
        .ok_or_else(|| {
            NativeGraphFactoryError::new(format!(
                "NativeGraph rollout selected unknown adapter runtime provider {:?}",
                environment.runtime_provider_id().as_str()
            ))
        })?;
    let stepper_factory = registry
        .native_graph_environment_stepper(environment.stepper_factory_id().as_str())
        .ok_or_else(|| {
            NativeGraphFactoryError::new(format!(
                "NativeGraph rollout selected unknown environment stepper factory {:?}",
                environment.stepper_factory_id().as_str()
            ))
        })?;
    let action_encoder_factory = registry
        .native_graph_action_encoder(environment.action_encoder_id().as_str())
        .ok_or_else(|| {
            NativeGraphFactoryError::new(format!(
                "NativeGraph rollout selected unknown action encoder {:?}",
                environment.action_encoder_id().as_str()
            ))
        })?;
    if action_encoder_factory.id() != environment.action_encoder_id().as_str() {
        return Err(NativeGraphFactoryError::new(
            "NativeGraph action encoder registration does not match the sealed selector",
        ));
    }
    let action_encoder = action_encoder_factory.bind(environment.action_encoder_id())?;
    if action_encoder.id() != environment.action_encoder_id() {
        return Err(NativeGraphFactoryError::new(
            "NativeGraph action encoder binding does not match the sealed selector",
        ));
    }
    let protocol = environment_protocol_config(
        environment,
        ArtifactDigest::from_bytes(trial.attempt_id().as_str().as_bytes())
            .as_str()
            .to_owned(),
    )?;
    let artifact_quota = environment_artifact_quota(environment)?;
    let action_encoding_limits = ActionEncodingLimits::new(
        environment_limit_usize(
            rollout.policy().max_decision_bytes(),
            "rollout.policy.max_decision_bytes",
        )?,
        environment_limit_usize(
            artifact_quota.max_artifact_bytes,
            "rollout.artifacts.max_artifact_bytes",
        )?,
    )
    .map_err(action_encoder_factory_error)?;
    let operation_deadline = Duration::from_millis(environment.operation_deadline_ms().get());
    let protocol_factory = Rc::new(WorkerLocalProtocolFactory {
        inner: Arc::clone(protocol_factory),
    });
    let resolved = runtime_provider
        .resolve(protocol.clone(), protocol_factory.clone())
        .map_err(|error| {
            NativeGraphFactoryError::new(format!(
                "NativeGraph environment runtime provider resolution failed: {error}"
            ))
        })?;
    if resolved.protocol_config() != &protocol {
        return Err(NativeGraphFactoryError::new(
            "NativeGraph environment runtime protocol configuration does not match the sealed role and capabilities",
        ));
    }
    Ok(BoundNativeGraphEnvironmentStepper {
        adapter: adapter.clone(),
        package_identity: trial.imported().task.digest.clone(),
        protocol,
        action_encoder_id: environment.action_encoder_id().clone(),
        action_encoder,
        action_encoding_limits,
        policy: RlEvaluationPolicy::new(
            rollout.policy().environment(),
            rollout.policy().horizon(),
            rollout.policy().gamma(),
        )
        .map_err(|error| NativeGraphFactoryError::new(error.to_string()))?,
        artifact_quota,
        operation_deadline,
        reset_source: environment.reset_source().clone(),
        rollout: rollout.clone(),
        rollout_binding: NativeGraphLiveRolloutBindingAuthority::new(),
        runtime: resolved,
        stepper_factory: Arc::clone(stepper_factory),
    })
}

#[cfg(feature = "engine")]
fn live_rollout_factory_error(error: NativeGraphLiveRolloutError) -> NativeGraphFactoryError {
    NativeGraphFactoryError::new(format!(
        "NativeGraph live rollout coordinator binding failed: {error}"
    ))
}

#[cfg(feature = "engine")]
struct WorkerLocalProtocolFactory {
    inner: Arc<dyn AdapterProtocolFactory + Send + Sync>,
}

#[cfg(feature = "engine")]
impl AdapterProtocolFactory for WorkerLocalProtocolFactory {
    fn create(
        &self,
        config: AdapterProtocolConfig,
    ) -> Result<Box<dyn super::AdapterProtocol>, super::ProtocolError> {
        self.inner.create(config)
    }
}

#[cfg(feature = "engine")]
fn environment_protocol_config(
    environment: &NativeGraphRolloutEnvironment,
    episode: String,
) -> Result<AdapterProtocolConfig, NativeGraphFactoryError> {
    let limits = environment.protocol_limits();
    let protocol_limits = ProtocolLimits {
        max_frame_bytes: environment_limit_usize(limits.max_frame_bytes(), "max_frame_bytes")?,
        max_identifier_bytes: environment_limit_usize(
            limits.max_identifier_bytes(),
            "max_identifier_bytes",
        )?,
        max_json_bytes: environment_limit_usize(limits.max_json_bytes(), "max_json_bytes")?,
        max_json_depth: environment_limit_usize(limits.max_json_depth(), "max_json_depth")?,
        max_json_array_entries: environment_limit_usize(
            limits.max_json_array_entries(),
            "max_json_array_entries",
        )?,
        max_json_object_entries: environment_limit_usize(
            limits.max_json_object_entries(),
            "max_json_object_entries",
        )?,
        max_operation_ledger_entries: environment_limit_usize(
            limits.max_operation_ledger_entries(),
            "max_operation_ledger_entries",
        )?,
        max_model_call_lineage_entries: environment_limit_usize(
            limits.max_model_call_lineage_entries(),
            "max_model_call_lineage_entries",
        )?,
        max_session_model_call_lineage_entries: environment_limit_usize(
            limits.max_session_model_call_lineage_entries(),
            "max_session_model_call_lineage_entries",
        )?,
        max_session_model_call_lineage_bytes: environment_limit_usize(
            limits.max_session_model_call_lineage_bytes(),
            "max_session_model_call_lineage_bytes",
        )?,
        max_artifact_handles: environment_limit_usize(
            limits.max_artifact_handles(),
            "max_artifact_handles",
        )?,
        max_artifact_bytes: limits.max_artifact_bytes(),
    };
    AdapterProtocolConfig::new(
        super::AdapterRole::Environment,
        episode,
        [
            ProtocolCapability::Environment,
            ProtocolCapability::Artifacts,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        BTreeSet::new(),
        protocol_limits,
    )
    .map_err(|error| NativeGraphFactoryError::new(error.to_string()))
}

#[cfg(feature = "engine")]
fn environment_artifact_quota(
    environment: &NativeGraphRolloutEnvironment,
) -> Result<ArtifactQuota, NativeGraphFactoryError> {
    let limits = environment.artifact_limits();
    Ok(ArtifactQuota {
        max_artifacts: environment_limit_usize(limits.max_artifacts(), "max_artifacts")?,
        max_total_bytes: limits.max_total_bytes(),
        max_artifact_bytes: limits.max_artifact_bytes(),
        max_download_handles: environment_limit_usize(
            limits.max_download_handles(),
            "max_download_handles",
        )?,
    })
}

#[cfg(feature = "engine")]
fn environment_limit_usize(
    value: u64,
    field: &'static str,
) -> Result<usize, NativeGraphFactoryError> {
    usize::try_from(value).map_err(|_| {
        NativeGraphFactoryError::new(format!(
            "NativeGraph environment limit {field} does not fit this platform"
        ))
    })
}

#[cfg(feature = "engine")]
fn artifact_factory_error(stage: &'static str, error: ArtifactError) -> NativeGraphFactoryError {
    NativeGraphFactoryError::new(format!("NativeGraph environment cannot {stage}: {error}"))
}

#[cfg(feature = "engine")]
fn abort_reset_upload(
    store: &mut super::EpisodeArtifactStore,
    upload: &super::ArtifactUploadHandle,
    primary: NativeGraphFactoryError,
) -> NativeGraphFactoryError {
    match store.abort_upload(upload) {
        Ok(()) => primary,
        Err(cleanup) => NativeGraphFactoryError::new(format!(
            "{primary}; NativeGraph environment reset upload cleanup failed: {cleanup}"
        )),
    }
}

#[cfg(feature = "engine")]
fn revoke_reset_reference(
    store: &Rc<RefCell<super::EpisodeArtifactStore>>,
    reset_input: &FrozenArtifactReference,
    primary: NativeGraphFactoryError,
) -> NativeGraphFactoryError {
    match store.try_borrow_mut() {
        Ok(mut store) => match store.revoke_reference(reset_input) {
            Ok(()) => primary,
            Err(cleanup) => NativeGraphFactoryError::new(format!(
                "{primary}; NativeGraph environment reset capability cleanup failed: {cleanup}"
            )),
        },
        Err(_) => NativeGraphFactoryError::new(format!(
            "{primary}; NativeGraph environment reset capability cleanup could not borrow the store"
        )),
    }
}

/// Capability-limited session supplied only after external compatibility preparation succeeds.
///
/// This marker keeps preparation independent of processes, environments, secrets, and native
/// execution authority.
pub trait ExternalDriverSession {}

/// Redacted failure category for compatibility-driver preparation and terminal exchange.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalDriverError {
    /// The selected external driver is not available in this product build.
    Unavailable,
    /// The selected package or trial did not satisfy compatibility preparation requirements.
    PreparationRejected,
    /// The supervised session did not return an admissible terminal receipt.
    TerminalReceiptRejected,
}

impl Display for ExternalDriverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => {
                formatter.write_str("external compatibility driver is unavailable")
            }
            Self::PreparationRejected => {
                formatter.write_str("external compatibility driver preparation was rejected")
            }
            Self::TerminalReceiptRejected => {
                formatter.write_str("external compatibility terminal receipt was rejected")
            }
        }
    }
}

impl std::error::Error for ExternalDriverError {}

/// Prepared externally driven compatibility work bound to one exact package and trial.
#[async_trait(?Send)]
pub trait PreparedExternalDriver {
    /// Runs one bounded compatibility interaction through the externally authorized session.
    async fn run(
        &mut self,
        session: &mut dyn ExternalDriverSession,
    ) -> Result<CompatibilityTerminalReceipt, ExternalDriverError>;
}

/// Factory for compatibility-only externally driven package support.
pub trait NativeGraphExternalDriverFactory: Send + Sync {
    /// Returns the canonical immutable selector this factory permits.
    fn id(&self) -> &str;

    /// Prepares an external driver from exact immutable package and trial authority.
    fn prepare(
        &self,
        package: &NativeGraphPackagePlan,
        trial: &ResolvedEpisodeTrial,
    ) -> Result<Box<dyn PreparedExternalDriver>, ExternalDriverError>;
}

/// Explicit refusal until the externally driven compatibility slice is enabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefusingExternalDriverFactory;

impl NativeGraphExternalDriverFactory for RefusingExternalDriverFactory {
    fn id(&self) -> &str {
        "refuse"
    }

    fn prepare(
        &self,
        _: &NativeGraphPackagePlan,
        _: &ResolvedEpisodeTrial,
    ) -> Result<Box<dyn PreparedExternalDriver>, ExternalDriverError> {
        Err(ExternalDriverError::Unavailable)
    }
}

/// Resolves an external compatibility driver only from its immutable package selector.
///
/// This preflight intentionally receives neither a task environment nor spawn authority. A
/// later compatibility runner may bind the selected factory only after this exact selection has
/// succeeded.
#[cfg(feature = "engine")]
pub fn select_native_graph_external_driver(
    registry: &AIPerfRegistry,
    package: &NativeGraphPackagePlan,
) -> Result<Arc<dyn NativeGraphExternalDriverFactory>, NativeGraphFactoryError> {
    if package.profile() != NativeGraphProfile::ExternallyDriven {
        return Err(NativeGraphFactoryError::new(
            "NativeGraph external driver selection requires the externally_driven profile",
        ));
    }
    let selector = package.external_driver_factory_id().ok_or_else(|| {
        NativeGraphFactoryError::new(
            "externally driven NativeGraph package has no immutable external driver factory selector",
        )
    })?;
    let factory = registry
        .native_graph_external_driver(selector.as_str())
        .ok_or_else(|| {
            NativeGraphFactoryError::new(format!(
                "NativeGraph selected unknown external driver factory {:?}",
                selector.as_str()
            ))
        })?;
    if factory.id() != selector.as_str() {
        return Err(NativeGraphFactoryError::new(
            "NativeGraph external driver registration does not match the sealed selector",
        ));
    }
    Ok(Arc::clone(factory))
}

/// Observer that validates the fidelity account emitted by source lowering.
pub trait NativeGraphFidelityObserver {
    /// Records or refuses one immutable lowering report before dispatch.
    fn observe(&self, report: &NativeGraphLoweringReport) -> Result<(), NativeGraphFactoryError>;
}

/// Factory for a fidelity observer selected through application composition.
pub trait NativeGraphFidelityObserverFactory: Send + Sync {
    /// Creates one observer for an immutable episode preparation.
    fn create(&self) -> Rc<dyn NativeGraphFidelityObserver>;
}

/// Built-in observer requiring exact lowering for every source node.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactNativeGraphFidelityObserverFactory;

impl NativeGraphFidelityObserverFactory for ExactNativeGraphFidelityObserverFactory {
    fn create(&self) -> Rc<dyn NativeGraphFidelityObserver> {
        Rc::new(ExactNativeGraphFidelityObserver)
    }
}

struct ExactNativeGraphFidelityObserver;

impl NativeGraphFidelityObserver for ExactNativeGraphFidelityObserver {
    fn observe(&self, report: &NativeGraphLoweringReport) -> Result<(), NativeGraphFactoryError> {
        if report.nodes().all(super::NativeGraphNodeLowering::is_exact) {
            Ok(())
        } else {
            Err(NativeGraphFactoryError::new(
                "NativeGraph lowering contains a non-exact source node",
            ))
        }
    }
}

/// Resolves a provider's terminal recovery fact without inventing cleanup state.
pub trait NativeGraphProviderRecoveryFactory: Send + Sync {
    /// Validates a provider-owned recovery disposition at an episode boundary.
    fn resolve(&self, recovery: ProviderRecovery) -> Result<(), NativeGraphFactoryError>;
}

/// Built-in recovery policy preserving only provider-confirmed terminal facts.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConfirmedNativeGraphProviderRecoveryFactory;

impl NativeGraphProviderRecoveryFactory for ConfirmedNativeGraphProviderRecoveryFactory {
    fn resolve(&self, recovery: ProviderRecovery) -> Result<(), NativeGraphFactoryError> {
        match recovery {
            ProviderRecovery::Recovered | ProviderRecovery::RequiresFreshAdapter => Ok(()),
            ProviderRecovery::Failed { reason } => Err(NativeGraphFactoryError::new(format!(
                "NativeGraph provider did not establish terminal recovery: {reason}"
            ))),
        }
    }
}
