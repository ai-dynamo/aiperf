// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local reset and transition operations for supervised RL environments.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Display, Formatter},
    io::Cursor,
    rc::Rc,
    time::Duration,
};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};

use crate::eval::{
    ActionAdmissionAuthority, ActionEncoderFactoryId, ActionSessionAuthority, AdapterEnvelope,
    AdapterMessage, AdapterProtocolConfig, AdapterRole, AdapterRuntimeFactory, AdapterSpawnRequest,
    AdapterSupervisionError, AdmittedEnvironmentAction, ArtifactDigest, ArtifactError,
    BoundNativeGraphActionEncoder, CancelReason, EnvironmentTransitionRecord, EpisodeArtifactStore,
    FrozenArtifact, FrozenArtifactReference, HostEnvelope, HostMessage, PROTOCOL_VERSION,
    ProtocolCapability, RlEvaluationPolicy, RlRolloutError, SupervisedAdapter,
};

/// Immutable host request for the initial environment observation.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentResetRequest {
    operation: String,
    input_ref: FrozenArtifactReference,
}

impl EnvironmentResetRequest {
    /// Creates one Rust-authorized reset operation.
    pub fn new(operation: impl Into<String>, input_ref: FrozenArtifactReference) -> Self {
        Self {
            operation: operation.into(),
            input_ref,
        }
    }
}

/// Immutable host request for one environment transition.
#[derive(Debug)]
pub struct EnvironmentStepRequest {
    operation: String,
    action: EnvironmentStepAction,
}

#[derive(Debug)]
enum EnvironmentStepAction {
    Legacy(FrozenArtifactReference),
    Admitted(AdmittedEnvironmentAction),
}

impl EnvironmentStepRequest {
    /// Creates a legacy caller-owned action request.
    ///
    /// A package-selected stepper rejects this form before dispatch; its actions must use
    /// [`Self::admitted`]. The constructor remains for existing non-package integrations.
    pub fn new(operation: impl Into<String>, action_ref: FrozenArtifactReference) -> Self {
        Self {
            operation: operation.into(),
            action: EnvironmentStepAction::Legacy(action_ref),
        }
    }

    /// Creates the only request form a package-selected stepper may dispatch.
    pub fn admitted(operation: impl Into<String>, action: AdmittedEnvironmentAction) -> Self {
        Self {
            operation: operation.into(),
            action: EnvironmentStepAction::Admitted(action),
        }
    }
}

/// Initial environment observation admitted from the reset operation.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentResetRecord {
    observation: FrozenArtifact,
}

impl EnvironmentResetRecord {
    /// Borrows the protocol-admitted initial observation descriptor.
    pub fn observation(&self) -> &FrozenArtifact {
        &self.observation
    }
}

/// Immutable package, episode, and operation-deadline identity for one environment stepper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvironmentEpisodeIdentity {
    package: ArtifactDigest,
    episode: String,
    operation_deadline: Duration,
}

impl EnvironmentEpisodeIdentity {
    /// Creates a package-bound identity with the exact selected operation deadline.
    pub fn new(
        package: ArtifactDigest,
        episode: impl Into<String>,
        operation_deadline: Duration,
    ) -> Result<Self, EnvironmentStepperError> {
        let episode = episode.into();
        if episode.is_empty() {
            return Err(EnvironmentStepperError::InvalidEpisodeIdentity);
        }
        if operation_deadline.is_zero() {
            return Err(EnvironmentStepperError::InvalidOperationDeadline);
        }
        Ok(Self {
            package,
            episode,
            operation_deadline,
        })
    }

    /// Borrows the immutable package content identity.
    pub fn package(&self) -> &ArtifactDigest {
        &self.package
    }

    /// Borrows the immutable protocol episode identity.
    pub fn episode(&self) -> &str {
        &self.episode
    }

    /// Returns the operation deadline selected by the package plan.
    pub const fn operation_deadline(&self) -> Duration {
        self.operation_deadline
    }
}

/// Actual Rust-owned package and artifact-store authority for one environment episode.
pub struct EnvironmentSessionAuthority {
    package: ArtifactDigest,
    artifacts: Rc<RefCell<EpisodeArtifactStore>>,
}

impl EnvironmentSessionAuthority {
    /// Binds an imported package digest to its single worker-local artifact store.
    pub fn new(package: ArtifactDigest, artifacts: Rc<RefCell<EpisodeArtifactStore>>) -> Self {
        Self { package, artifacts }
    }
}

/// Immutable artifact capabilities declared for one environment episode.
#[derive(Clone, Debug)]
pub struct EnvironmentArtifactBindings {
    inputs: BTreeSet<FrozenArtifactReference>,
}

impl EnvironmentArtifactBindings {
    /// Creates the nonempty Rust-issued input references for one environment episode.
    pub fn new(
        inputs: impl IntoIterator<Item = FrozenArtifactReference>,
    ) -> Result<Self, EnvironmentStepperError> {
        let inputs = inputs.into_iter().collect::<BTreeSet<_>>();
        if inputs.is_empty() {
            return Err(EnvironmentStepperError::MissingArtifactBindings);
        }
        Ok(Self { inputs })
    }

    fn contains_input(&self, reference: &FrozenArtifactReference) -> bool {
        self.inputs.contains(reference)
    }

    fn inputs(&self) -> impl Iterator<Item = &FrozenArtifactReference> {
        self.inputs.iter()
    }

    fn consume_input(&mut self, reference: &FrozenArtifactReference) -> bool {
        self.inputs.remove(reference)
    }
}

/// Immutable worker-local binding for one supervised environment episode.
#[derive(Clone, Debug)]
pub struct EnvironmentStepperBinding {
    protocol: AdapterProtocolConfig,
    identity: EnvironmentEpisodeIdentity,
    span: String,
    horizon: u32,
    artifacts: EnvironmentArtifactBindings,
    selected_action_encoder: Option<SelectedActionEncoder>,
    defer_terminal_cleanup: bool,
}

#[derive(Clone, Debug)]
struct SelectedActionEncoder {
    id: ActionEncoderFactoryId,
    authority: ActionAdmissionAuthority,
    session: Option<ActionSessionAuthority>,
}

impl EnvironmentStepperBinding {
    /// Validates a role-pinned environment binding before an adapter starts.
    pub fn new(
        protocol: AdapterProtocolConfig,
        identity: EnvironmentEpisodeIdentity,
        span: impl Into<String>,
        policy: RlEvaluationPolicy,
        artifacts: EnvironmentArtifactBindings,
    ) -> Result<Self, EnvironmentStepperError> {
        if protocol.role() != AdapterRole::Environment {
            return Err(EnvironmentStepperError::AdapterRole);
        }
        if !protocol
            .capabilities()
            .contains(&ProtocolCapability::Environment)
        {
            return Err(EnvironmentStepperError::EnvironmentCapability);
        }
        if identity.episode() != protocol.episode() {
            return Err(EnvironmentStepperError::EpisodeIdentityMismatch);
        }
        Ok(Self {
            protocol,
            identity,
            span: span.into(),
            horizon: policy.horizon(),
            artifacts,
            selected_action_encoder: None,
            defer_terminal_cleanup: false,
        })
    }

    /// Pins one package-selected action encoder to this worker-local stepper.
    pub fn with_selected_action_encoder(mut self, encoder: &BoundNativeGraphActionEncoder) -> Self {
        self.selected_action_encoder = Some(SelectedActionEncoder {
            id: encoder.id().clone(),
            authority: encoder.authority(),
            session: None,
        });
        self
    }

    /// Pins one selected encoder to the exact started live-rollout session that may consume it.
    pub(crate) fn with_selected_action_encoder_session(
        mut self,
        encoder: &BoundNativeGraphActionEncoder,
        session: ActionSessionAuthority,
    ) -> Self {
        self.selected_action_encoder = Some(SelectedActionEncoder {
            id: encoder.id().clone(),
            authority: encoder.authority(),
            session: Some(session),
        });
        self
    }

    /// Defers terminal adapter cleanup to the owner of the surrounding task lease.
    ///
    /// Docker uses this only after it has admitted its exact task-minted spawn request, so
    /// declared artifact collection and verification can complete before the adapter client
    /// is cancelled and reaped. Ordinary worker-local steppers retain immediate terminal
    /// cleanup.
    pub(crate) fn with_deferred_terminal_cleanup(mut self) -> Self {
        self.defer_terminal_cleanup = true;
        self
    }
}

/// Worker-local environment reset and transition operations.
#[async_trait(?Send)]
pub trait EnvironmentStepper {
    /// Requests and retains the initial observation for this fresh episode.
    async fn reset(
        &mut self,
        request: EnvironmentResetRequest,
    ) -> Result<EnvironmentResetRecord, EnvironmentStepperError>;

    /// Advances exactly one environment action and records its authoritative reward facts.
    async fn step(
        &mut self,
        request: EnvironmentStepRequest,
    ) -> Result<EnvironmentTransitionRecord, EnvironmentStepperError>;

    /// Reaps a still-live child when the host ends the rollout before its terminal transition.
    async fn cancel_and_reap(&mut self) -> Result<(), EnvironmentStepperError>;
}

/// Factory for one worker-local supervised environment stepper.
#[async_trait(?Send)]
pub trait EnvironmentStepperFactory {
    /// Starts the pinned adapter and returns its episode-local reset/step boundary.
    async fn start(
        &self,
        binding: EnvironmentStepperBinding,
        authority: EnvironmentSessionAuthority,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn EnvironmentStepper>, EnvironmentStepperError>;
}

/// Task 5 supervision composition for worker-local environment stepping.
pub struct SupervisedEnvironmentStepperFactory {
    runtime: Rc<dyn AdapterRuntimeFactory>,
}

impl SupervisedEnvironmentStepperFactory {
    /// Creates a factory over one selected protocol-validated adapter runtime.
    pub fn new(runtime: Rc<dyn AdapterRuntimeFactory>) -> Self {
        Self { runtime }
    }
}

#[async_trait(?Send)]
impl EnvironmentStepperFactory for SupervisedEnvironmentStepperFactory {
    async fn start(
        &self,
        binding: EnvironmentStepperBinding,
        authority: EnvironmentSessionAuthority,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn EnvironmentStepper>, EnvironmentStepperError> {
        if authority.package != *binding.identity.package() {
            return Err(EnvironmentStepperError::PackageIdentityMismatch);
        }
        if self.runtime.protocol_config() != Some(&binding.protocol) {
            return Err(EnvironmentStepperError::ProtocolConfigMismatch);
        }
        if request.deadlines().operation() != binding.identity.operation_deadline() {
            return Err(EnvironmentStepperError::OperationDeadlineMismatch);
        }
        for input in binding.artifacts.inputs() {
            authority
                .artifacts
                .borrow()
                .validate_reference(input)
                .map_err(EnvironmentStepperError::Artifact)?;
        }
        let adapter = self
            .runtime
            .start(request)
            .await
            .map_err(EnvironmentStepperError::Supervision)?;
        let defer_terminal_cleanup = binding.defer_terminal_cleanup;
        Ok(Box::new(SupervisedEnvironmentStepper {
            adapter,
            binding,
            artifacts: authority.artifacts,
            outputs: BTreeMap::new(),
            operations: BTreeSet::new(),
            next_host_sequence: 1,
            next_adapter_sequence: 1,
            step_count: 0,
            is_reset: false,
            is_terminal: false,
            is_invalidated: false,
            cleanup: CleanupState::Active,
            defer_terminal_cleanup,
        }))
    }
}

struct SupervisedEnvironmentStepper {
    adapter: Box<dyn SupervisedAdapter>,
    binding: EnvironmentStepperBinding,
    artifacts: Rc<RefCell<EpisodeArtifactStore>>,
    outputs: BTreeMap<String, BTreeSet<FrozenArtifactReference>>,
    operations: BTreeSet<String>,
    next_host_sequence: u64,
    next_adapter_sequence: u64,
    step_count: u32,
    is_reset: bool,
    is_terminal: bool,
    is_invalidated: bool,
    cleanup: CleanupState,
    defer_terminal_cleanup: bool,
}

enum EnvironmentInput<'a> {
    Declared(&'a FrozenArtifactReference),
    Admitted(&'a AdmittedEnvironmentAction),
}

impl EnvironmentInput<'_> {
    fn reference(&self) -> &FrozenArtifactReference {
        match self {
            Self::Declared(reference) => reference,
            Self::Admitted(action) => action.reference(),
        }
    }
}

/// Persistent supervision-cleanup ownership for one environment episode.
#[derive(Clone, Copy)]
enum CleanupState {
    Active,
    Pending(CancelReason),
    Reaped,
}

impl SupervisedEnvironmentStepper {
    async fn finish_cleanup(
        &mut self,
        requested_reason: CancelReason,
    ) -> Result<(), AdapterSupervisionError> {
        let reason = match self.cleanup {
            CleanupState::Active => {
                self.cleanup = CleanupState::Pending(requested_reason);
                requested_reason
            }
            CleanupState::Pending(reason) => reason,
            CleanupState::Reaped => return Ok(()),
        };
        self.adapter.cancel_and_reap(reason).await?;
        self.cleanup = CleanupState::Reaped;
        Ok(())
    }

    async fn fail_after_cleanup<T>(
        &mut self,
        reason: CancelReason,
        primary: EnvironmentStepperError,
    ) -> Result<T, EnvironmentStepperError> {
        match self.finish_cleanup(reason).await {
            Ok(()) => Err(primary),
            Err(cleanup) => Err(EnvironmentStepperError::Cleanup {
                primary: Box::new(primary),
                cleanup,
            }),
        }
    }

    async fn invalidate<T>(
        &mut self,
        reason: CancelReason,
        primary: EnvironmentStepperError,
    ) -> Result<T, EnvironmentStepperError> {
        self.is_invalidated = true;
        self.fail_after_cleanup(reason, primary).await
    }

    async fn refuse_invalidated<T>(&mut self) -> Result<T, EnvironmentStepperError> {
        self.fail_after_cleanup(
            CancelReason::IntegrityViolation,
            EnvironmentStepperError::ProtocolInvalidated,
        )
        .await
    }

    async fn refuse_terminal<T>(&mut self) -> Result<T, EnvironmentStepperError> {
        self.fail_after_cleanup(
            CancelReason::HostShutdown,
            EnvironmentStepperError::EpisodeTerminal,
        )
        .await
    }

    async fn refuse_closed<T>(&mut self) -> Option<Result<T, EnvironmentStepperError>> {
        if self.is_invalidated {
            return Some(self.refuse_invalidated().await);
        }
        if self.is_terminal {
            return Some(self.refuse_terminal().await);
        }
        None
    }

    async fn exchange(
        &mut self,
        operation: String,
        message: HostMessage,
        input: EnvironmentInput<'_>,
    ) -> Result<AdapterEnvelope, EnvironmentStepperError> {
        if self.is_invalidated {
            return self.refuse_invalidated().await;
        }
        if !self.operations.insert(operation.clone()) {
            return self
                .reject_undispatched_input(
                    input,
                    CancelReason::IntegrityViolation,
                    EnvironmentStepperError::OperationReused,
                )
                .await;
        }
        if matches!(&input, EnvironmentInput::Admitted(_)) {
            let validation = {
                let artifacts = self.artifacts.borrow();
                artifacts.validate_reference(input.reference())
            };
            if let Err(error) = validation {
                return self
                    .reject_undispatched_input(
                        input,
                        CancelReason::IntegrityViolation,
                        EnvironmentStepperError::Artifact(error),
                    )
                    .await;
            }
        }
        let envelope = match self.send_host(&operation, message).await {
            Ok(envelope) => envelope,
            Err(error) => {
                return self
                    .reject_undispatched_input(input, CancelReason::OperationFailure, error)
                    .await;
            }
        };
        if let Err(error) = self.consume_dispatched_input(input) {
            return self
                .invalidate(CancelReason::IntegrityViolation, error)
                .await;
        }
        loop {
            let response = match self.receive_adapter(&envelope).await {
                Ok(response) => response,
                Err(error) => {
                    let reason = if matches!(
                        error,
                        EnvironmentStepperError::Supervision(_)
                            | EnvironmentStepperError::SequenceExhausted
                    ) {
                        CancelReason::OperationFailure
                    } else {
                        CancelReason::IntegrityViolation
                    };
                    return self.invalidate(reason, error).await;
                }
            };
            if matches!(response.message, AdapterMessage::PutArtifactRequest { .. }) {
                if let Err(error) = self.complete_output_upload(&envelope, response).await {
                    return self
                        .invalidate(CancelReason::IntegrityViolation, error)
                        .await;
                }
                continue;
            }
            if let Err(error) = self.validate_response(&envelope, &response) {
                return self
                    .invalidate(CancelReason::IntegrityViolation, error)
                    .await;
            }
            return Ok(response);
        }
    }

    fn consume_dispatched_input(
        &mut self,
        input: EnvironmentInput<'_>,
    ) -> Result<(), EnvironmentStepperError> {
        if matches!(&input, EnvironmentInput::Declared(_))
            && !self.binding.artifacts.consume_input(input.reference())
        {
            return Err(EnvironmentStepperError::UndeclaredInput);
        }
        self.artifacts
            .borrow_mut()
            .revoke_reference(input.reference())
            .map_err(EnvironmentStepperError::Artifact)
    }

    async fn reject_undispatched_input<T>(
        &mut self,
        input: EnvironmentInput<'_>,
        reason: CancelReason,
        primary: EnvironmentStepperError,
    ) -> Result<T, EnvironmentStepperError> {
        if matches!(&input, EnvironmentInput::Admitted(_)) {
            let revocation = {
                let mut artifacts = self.artifacts.borrow_mut();
                artifacts.revoke_reference(input.reference())
            };
            if let Err(error) = revocation {
                return self
                    .invalidate(
                        CancelReason::IntegrityViolation,
                        EnvironmentStepperError::Artifact(error),
                    )
                    .await;
            }
        }
        self.invalidate(reason, primary).await
    }

    async fn send_host(
        &mut self,
        operation: &str,
        message: HostMessage,
    ) -> Result<HostEnvelope, EnvironmentStepperError> {
        let envelope = HostEnvelope::new(
            self.binding.protocol.episode(),
            &self.binding.span,
            self.next_host_sequence,
            operation,
            message,
        );
        let Some(next_host_sequence) = self.next_host_sequence.checked_add(1) else {
            return Err(EnvironmentStepperError::SequenceExhausted);
        };
        self.next_host_sequence = next_host_sequence;
        self.adapter
            .send(envelope.clone())
            .await
            .map_err(EnvironmentStepperError::Supervision)?;
        Ok(envelope)
    }

    async fn receive_adapter(
        &mut self,
        request: &HostEnvelope,
    ) -> Result<AdapterEnvelope, EnvironmentStepperError> {
        let response = self
            .adapter
            .receive()
            .await
            .map_err(EnvironmentStepperError::Supervision)?;
        self.validate_common_response(request, &response)?;
        let Some(next_adapter_sequence) = self.next_adapter_sequence.checked_add(1) else {
            return Err(EnvironmentStepperError::SequenceExhausted);
        };
        self.next_adapter_sequence = next_adapter_sequence;
        Ok(response)
    }

    async fn complete_output_upload(
        &mut self,
        root: &HostEnvelope,
        request: AdapterEnvelope,
    ) -> Result<(), EnvironmentStepperError> {
        let AdapterMessage::PutArtifactRequest {
            parent_operation,
            declared_bytes,
        } = request.message
        else {
            return Err(EnvironmentStepperError::UnexpectedResponse(
                "put_artifact_request",
            ));
        };
        if parent_operation != root.operation {
            return Err(EnvironmentStepperError::Correlation("artifact parent"));
        }
        if !self.operations.insert(request.operation.clone()) {
            return Err(EnvironmentStepperError::OperationReused);
        }
        let upload = self
            .artifacts
            .borrow_mut()
            .begin_upload(declared_bytes)
            .map_err(EnvironmentStepperError::Artifact)?;
        self.send_host(
            &request.operation,
            HostMessage::PutArtifactHandle {
                upload: upload.clone(),
                declared_bytes,
            },
        )
        .await?;
        loop {
            let response = self.receive_adapter(root).await?;
            if response.operation != request.operation {
                return Err(EnvironmentStepperError::Correlation("artifact operation"));
            }
            match response.message {
                AdapterMessage::ArtifactUploadChunk {
                    upload: chunk_upload,
                    bytes_base64,
                } => {
                    if chunk_upload != upload {
                        return Err(EnvironmentStepperError::ArtifactUploadMismatch);
                    }
                    if bytes_base64.len() > self.binding.protocol.max_frame_bytes() {
                        return Err(EnvironmentStepperError::ArtifactUploadChunkTooLarge {
                            limit: self.binding.protocol.max_frame_bytes(),
                            actual: bytes_base64.len(),
                        });
                    }
                    let bytes = STANDARD
                        .decode(bytes_base64)
                        .map_err(|_| EnvironmentStepperError::InvalidArtifactUploadEncoding)?;
                    if bytes.is_empty() {
                        return Err(EnvironmentStepperError::EmptyArtifactUploadChunk);
                    }
                    self.artifacts
                        .borrow_mut()
                        .write_upload(&upload, &mut Cursor::new(bytes))
                        .map_err(EnvironmentStepperError::Artifact)?;
                }
                AdapterMessage::ArtifactUploadComplete {
                    upload: completed_upload,
                } => {
                    if completed_upload != upload {
                        return Err(EnvironmentStepperError::ArtifactUploadMismatch);
                    }
                    break;
                }
                _ => {
                    return Err(EnvironmentStepperError::UnexpectedResponse(
                        "artifact_upload_chunk_or_complete",
                    ));
                }
            }
        }
        let reference = {
            let mut artifacts = self.artifacts.borrow_mut();
            let artifact = artifacts
                .commit_upload(&upload)
                .map_err(EnvironmentStepperError::Artifact)?;
            artifacts
                .issue_reference(&artifact)
                .map_err(EnvironmentStepperError::Artifact)?
        };
        self.send_host(
            &request.operation,
            HostMessage::ArtifactCommitted {
                upload,
                reference: reference.clone(),
            },
        )
        .await?;
        self.outputs
            .entry(root.operation.clone())
            .or_default()
            .insert(reference);
        Ok(())
    }

    fn validate_response(
        &self,
        request: &HostEnvelope,
        response: &AdapterEnvelope,
    ) -> Result<(), EnvironmentStepperError> {
        if response.operation != request.operation {
            return Err(EnvironmentStepperError::Correlation("operation"));
        }
        Ok(())
    }

    fn validate_common_response(
        &self,
        request: &HostEnvelope,
        response: &AdapterEnvelope,
    ) -> Result<(), EnvironmentStepperError> {
        if response.version != PROTOCOL_VERSION {
            return Err(EnvironmentStepperError::Correlation("version"));
        }
        if response.episode != request.episode {
            return Err(EnvironmentStepperError::Correlation("episode"));
        }
        if response.span != request.span {
            return Err(EnvironmentStepperError::Correlation("span"));
        }
        if response.sequence != self.next_adapter_sequence {
            return Err(EnvironmentStepperError::Correlation("sequence"));
        }
        Ok(())
    }

    fn release_output(
        &mut self,
        operation: &str,
        reference: &FrozenArtifactReference,
    ) -> Result<FrozenArtifact, EnvironmentStepperError> {
        if !self
            .outputs
            .get_mut(operation)
            .is_some_and(|outputs| outputs.remove(reference))
        {
            return Err(EnvironmentStepperError::UndeclaredOutput);
        }
        let protocol_release = self.adapter.release_download_handle(reference.download());
        let store_release = self.artifacts.borrow_mut().revoke_reference(reference);
        match (protocol_release, store_release) {
            (Ok(()), Ok(())) => Ok(reference.artifact().clone()),
            (Err(error), _) => Err(EnvironmentStepperError::Supervision(error)),
            (Ok(()), Err(error)) => Err(EnvironmentStepperError::Artifact(error)),
        }
    }
}

#[async_trait(?Send)]
impl EnvironmentStepper for SupervisedEnvironmentStepper {
    async fn reset(
        &mut self,
        request: EnvironmentResetRequest,
    ) -> Result<EnvironmentResetRecord, EnvironmentStepperError> {
        if let Some(result) = self.refuse_closed().await {
            return result;
        }
        if self.is_reset {
            return Err(EnvironmentStepperError::AlreadyReset);
        }
        if !self.binding.artifacts.contains_input(&request.input_ref) {
            return Err(EnvironmentStepperError::UndeclaredInput);
        }
        let response = self
            .exchange(
                request.operation.clone(),
                HostMessage::ResetEnvironment {
                    input_ref: request.input_ref.clone(),
                },
                EnvironmentInput::Declared(&request.input_ref),
            )
            .await?;
        let AdapterMessage::EnvironmentReset { observation_ref } = response.message else {
            return self
                .invalidate(
                    CancelReason::IntegrityViolation,
                    EnvironmentStepperError::UnexpectedResponse("environment_reset"),
                )
                .await;
        };
        let observation = match self.release_output(&request.operation, &observation_ref) {
            Ok(observation) => observation,
            Err(error) => {
                return self
                    .invalidate(CancelReason::IntegrityViolation, error)
                    .await;
            }
        };
        self.is_reset = true;
        Ok(EnvironmentResetRecord { observation })
    }

    async fn step(
        &mut self,
        request: EnvironmentStepRequest,
    ) -> Result<EnvironmentTransitionRecord, EnvironmentStepperError> {
        let EnvironmentStepRequest { operation, action } = request;
        let response = match action {
            EnvironmentStepAction::Legacy(action_ref) => {
                if let Some(result) = self.refuse_closed().await {
                    return result;
                }
                if self.binding.selected_action_encoder.is_some() {
                    return Err(EnvironmentStepperError::AdmittedActionRequired);
                }
                if !self.is_reset {
                    return Err(EnvironmentStepperError::ResetRequired);
                }
                if !self.binding.artifacts.contains_input(&action_ref) {
                    return Err(EnvironmentStepperError::UndeclaredInput);
                }
                self.exchange(
                    operation.clone(),
                    HostMessage::StepEnvironment {
                        action_ref: action_ref.clone(),
                    },
                    EnvironmentInput::Declared(&action_ref),
                )
                .await?
            }
            EnvironmentStepAction::Admitted(action) => {
                if self.is_invalidated {
                    return self
                        .reject_undispatched_input(
                            EnvironmentInput::Admitted(&action),
                            CancelReason::IntegrityViolation,
                            EnvironmentStepperError::ProtocolInvalidated,
                        )
                        .await;
                }
                if self.is_terminal {
                    return self
                        .reject_undispatched_input(
                            EnvironmentInput::Admitted(&action),
                            CancelReason::HostShutdown,
                            EnvironmentStepperError::EpisodeTerminal,
                        )
                        .await;
                }
                let Some(selected_encoder) = self.binding.selected_action_encoder.as_ref() else {
                    return self
                        .reject_undispatched_input(
                            EnvironmentInput::Admitted(&action),
                            CancelReason::IntegrityViolation,
                            EnvironmentStepperError::AdmittedActionRequired,
                        )
                        .await;
                };
                if action.encoder() != &selected_encoder.id
                    || !action.matches_authority(&selected_encoder.authority)
                {
                    return self
                        .reject_undispatched_input(
                            EnvironmentInput::Admitted(&action),
                            CancelReason::IntegrityViolation,
                            EnvironmentStepperError::ActionEncoderMismatch,
                        )
                        .await;
                }
                if selected_encoder
                    .session
                    .as_ref()
                    .is_some_and(|session| !action.matches_session(session))
                {
                    return self
                        .reject_undispatched_input(
                            EnvironmentInput::Admitted(&action),
                            CancelReason::IntegrityViolation,
                            EnvironmentStepperError::ActionSessionMismatch,
                        )
                        .await;
                }
                if !self.is_reset {
                    return self
                        .reject_undispatched_input(
                            EnvironmentInput::Admitted(&action),
                            CancelReason::IntegrityViolation,
                            EnvironmentStepperError::ResetRequired,
                        )
                        .await;
                }
                self.exchange(
                    operation.clone(),
                    HostMessage::StepEnvironment {
                        action_ref: action.reference().clone(),
                    },
                    EnvironmentInput::Admitted(&action),
                )
                .await?
            }
        };
        let AdapterMessage::Transition {
            observation_ref,
            reward,
            terminated,
            truncated,
            info_ref,
            ..
        } = response.message
        else {
            return self
                .invalidate(
                    CancelReason::IntegrityViolation,
                    EnvironmentStepperError::UnexpectedResponse("transition"),
                )
                .await;
        };
        let observation = match self.release_output(&operation, &observation_ref) {
            Ok(observation) => observation,
            Err(error) => {
                return self
                    .invalidate(CancelReason::IntegrityViolation, error)
                    .await;
            }
        };
        let info = match self.release_output(&operation, &info_ref) {
            Ok(info) => info,
            Err(error) => {
                return self
                    .invalidate(CancelReason::IntegrityViolation, error)
                    .await;
            }
        };
        let reaches_horizon = self.step_count.saturating_add(1) == self.binding.horizon;
        let transition = match EnvironmentTransitionRecord::new(
            self.step_count,
            observation,
            reward,
            terminated,
            truncated || (reaches_horizon && !terminated),
            info,
        ) {
            Ok(transition) => transition,
            Err(error) => {
                return self
                    .invalidate(
                        CancelReason::IntegrityViolation,
                        EnvironmentStepperError::Rollout(error),
                    )
                    .await;
            }
        };
        let Some(next_step_count) = self.step_count.checked_add(1) else {
            return self
                .invalidate(
                    CancelReason::OperationFailure,
                    EnvironmentStepperError::HorizonOverflow,
                )
                .await;
        };
        self.step_count = next_step_count;
        self.is_terminal = terminated || transition.is_truncated();
        if self.is_terminal && !self.defer_terminal_cleanup {
            if let Err(cleanup) = self.finish_cleanup(CancelReason::HostShutdown).await {
                return Err(EnvironmentStepperError::Cleanup {
                    primary: Box::new(EnvironmentStepperError::EpisodeTerminal),
                    cleanup,
                });
            }
        }
        Ok(transition)
    }

    async fn cancel_and_reap(&mut self) -> Result<(), EnvironmentStepperError> {
        self.finish_cleanup(CancelReason::HostShutdown)
            .await
            .map_err(EnvironmentStepperError::Supervision)
    }
}

/// Typed refusal from the worker-local environment stepping boundary.
#[derive(Debug)]
pub enum EnvironmentStepperError {
    /// The runtime protocol configuration is not pinned to an environment role.
    AdapterRole,
    /// The runtime protocol configuration omitted the environment capability.
    EnvironmentCapability,
    /// The runtime cannot prove it will spawn the binding's immutable protocol session.
    ProtocolConfigMismatch,
    /// The binding had no immutable input reference declarations.
    MissingArtifactBindings,
    /// The immutable environment episode identity had no protocol episode correlation.
    InvalidEpisodeIdentity,
    /// The immutable environment binding selected a zero operation deadline.
    InvalidOperationDeadline,
    /// The binding episode did not match its exact protocol configuration.
    EpisodeIdentityMismatch,
    /// The binding package did not match the actual imported/store session package.
    PackageIdentityMismatch,
    /// The adapter spawn request selected a different operation deadline.
    OperationDeadlineMismatch,
    /// A request attempted to bypass the declared immutable input references.
    UndeclaredInput,
    /// A package-selected stepper requires a selected encoder-minted action capability.
    AdmittedActionRequired,
    /// An admitted action was created by a different selected encoder than this stepper binds.
    ActionEncoderMismatch,
    /// An admitted action was not issued for this exact started environment session.
    ActionSessionMismatch,
    /// An adapter response named no immutable output reference Rust generated for this operation.
    UndeclaredOutput,
    /// The adapter completed a different upload capability than Rust granted.
    ArtifactUploadMismatch,
    /// A base64 upload fragment exceeded the pre-decode frame bound.
    ArtifactUploadChunkTooLarge {
        /// Maximum permitted encoded bytes.
        limit: usize,
        /// Actual encoded bytes supplied by the adapter.
        actual: usize,
    },
    /// An upload fragment was not valid base64.
    InvalidArtifactUploadEncoding,
    /// An upload fragment made no forward progress.
    EmptyArtifactUploadChunk,
    /// The Rust-owned artifact authority rejected an input or output capability.
    Artifact(ArtifactError),
    /// A step arrived before the required initial reset.
    ResetRequired,
    /// A fresh worker-local stepper accepted more than one initial reset.
    AlreadyReset,
    /// The episode already terminated or truncated.
    EpisodeTerminal,
    /// The caller attempted to reuse an operation correlation.
    OperationReused,
    /// A response did not match the request's Task 4 correlation fields.
    Correlation(&'static str),
    /// A previous protocol or transition violation closed this worker-local episode.
    ProtocolInvalidated,
    /// The required terminal cleanup failed after a primary environment failure.
    Cleanup {
        /// The operation failure that required cleanup.
        primary: Box<EnvironmentStepperError>,
        /// The cleanup failure returned by supervision.
        cleanup: AdapterSupervisionError,
    },
    /// The adapter returned a different message kind for this operation.
    UnexpectedResponse(&'static str),
    /// The adapter-runtime supervision boundary failed.
    Supervision(AdapterSupervisionError),
    /// Batch 1 rejected the transition facts.
    Rollout(RlRolloutError),
    /// A protocol sequence cannot be advanced further.
    SequenceExhausted,
    /// The bounded local step index cannot be advanced further.
    HorizonOverflow,
}

impl Display for EnvironmentStepperError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterRole => {
                formatter.write_str("environment stepper requires environment role")
            }
            Self::EnvironmentCapability => {
                formatter.write_str("environment stepper requires environment capability")
            }
            Self::ProtocolConfigMismatch => formatter
                .write_str("environment runtime protocol configuration does not match binding"),
            Self::MissingArtifactBindings => {
                formatter.write_str("environment binding requires immutable input references")
            }
            Self::InvalidEpisodeIdentity => {
                formatter.write_str("environment episode identity must not be empty")
            }
            Self::InvalidOperationDeadline => {
                formatter.write_str("environment operation deadline must be positive")
            }
            Self::EpisodeIdentityMismatch => formatter
                .write_str("environment episode identity does not match protocol configuration"),
            Self::PackageIdentityMismatch => {
                formatter.write_str("environment package identity does not match session authority")
            }
            Self::OperationDeadlineMismatch => formatter
                .write_str("environment operation deadline does not match adapter spawn request"),
            Self::UndeclaredInput => {
                formatter.write_str("environment request reference was not declared at binding")
            }
            Self::AdmittedActionRequired => {
                formatter.write_str("environment step requires an admitted selected action")
            }
            Self::ActionEncoderMismatch => {
                formatter.write_str("environment action capability was minted by another encoder")
            }
            Self::ActionSessionMismatch => formatter
                .write_str("environment action capability was not issued for this started session"),
            Self::UndeclaredOutput => {
                formatter.write_str("environment response reference was not generated by Rust")
            }
            Self::ArtifactUploadMismatch => {
                formatter.write_str("environment adapter completed a different artifact upload")
            }
            Self::ArtifactUploadChunkTooLarge { limit, actual } => {
                write!(
                    formatter,
                    "environment upload chunk {actual} exceeds encoded limit {limit}"
                )
            }
            Self::InvalidArtifactUploadEncoding => {
                formatter.write_str("environment upload chunk is not valid base64")
            }
            Self::EmptyArtifactUploadChunk => {
                formatter.write_str("environment upload chunk must not be empty")
            }
            Self::Artifact(error) => {
                write!(formatter, "environment artifact authority failed: {error}")
            }
            Self::ResetRequired => formatter.write_str("environment step requires initial reset"),
            Self::AlreadyReset => formatter.write_str("environment reset was already performed"),
            Self::EpisodeTerminal => formatter.write_str("environment episode is terminal"),
            Self::OperationReused => formatter.write_str("environment operation was already used"),
            Self::Correlation(field) => {
                write!(
                    formatter,
                    "environment response correlation mismatch: {field}"
                )
            }
            Self::ProtocolInvalidated => {
                formatter.write_str("environment protocol episode is invalidated")
            }
            Self::Cleanup { primary, cleanup } => {
                write!(
                    formatter,
                    "environment failure {primary}; cleanup failed: {cleanup}"
                )
            }
            Self::UnexpectedResponse(expected) => {
                write!(formatter, "environment adapter did not return {expected}")
            }
            Self::Supervision(error) => {
                write!(formatter, "environment supervision failed: {error}")
            }
            Self::Rollout(error) => write!(formatter, "invalid environment transition: {error}"),
            Self::SequenceExhausted => {
                formatter.write_str("environment protocol sequence exhausted")
            }
            Self::HorizonOverflow => formatter.write_str("environment horizon counter overflowed"),
        }
    }
}

impl std::error::Error for EnvironmentStepperError {}
