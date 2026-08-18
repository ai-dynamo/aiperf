// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict directional JSONL protocol for supervised NativeGraph adapters.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Display, Formatter},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::eval::ArtifactDigest;

use super::{
    AdapterRole, ModelBindingId,
    artifacts::{ArtifactDownloadHandle, ArtifactUploadHandle, FrozenArtifactReference},
};

/// The first supported NativeGraph adapter wire version.
pub const PROTOCOL_VERSION: u16 = 1;

/// One optional operation an adapter may expose to the host.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolCapability {
    /// The adapter accepts Rust-authorized tool invocations.
    Tool,
    /// The adapter can make bounded decisions and model-call intents.
    Policy,
    /// The adapter can reset and step an external environment.
    Environment,
    /// The adapter can make a bounded heuristic decision.
    Heuristic,
    /// The compatibility-only adapter may respond to a terminal request.
    Driver,
    /// The adapter can request Rust-owned artifact capabilities.
    Artifacts,
    /// The adapter can emit nonterminal operation checkpoints.
    Checkpoint,
}

/// Bounds enforced before adapter-provided values reach a runner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtocolLimits {
    /// Maximum bytes in one JSONL frame including its terminator.
    pub max_frame_bytes: usize,
    /// Maximum bytes in a typed correlation or capability identifier.
    pub max_identifier_bytes: usize,
    /// Maximum serialized bytes of an arbitrary JSON payload.
    pub max_json_bytes: usize,
    /// Maximum nesting depth of arbitrary JSON payloads.
    pub max_json_depth: usize,
    /// Maximum entries in any arbitrary JSON array.
    pub max_json_array_entries: usize,
    /// Maximum entries in any arbitrary JSON object.
    pub max_json_object_entries: usize,
    /// Maximum lifetime operation ledger entries; correlations are never reused.
    pub max_operation_ledger_entries: usize,
    /// Maximum active plus completed model-call correlations retained per pending policy decision.
    pub max_model_call_lineage_entries: usize,
    /// Maximum active plus completed model-call correlations retained across the session.
    pub max_session_model_call_lineage_entries: usize,
    /// Maximum UTF-8 bytes retained by model-call correlations across the session.
    pub max_session_model_call_lineage_bytes: usize,
    /// Maximum outstanding artifact references carried by one protocol session.
    pub max_artifact_handles: usize,
    /// Maximum bytes granted for one protocol artifact operation.
    pub max_artifact_bytes: u64,
}

impl Default for ProtocolLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 256 * 1024,
            max_identifier_bytes: 16 * 1024,
            max_json_bytes: 128 * 1024,
            max_json_depth: 16,
            max_json_array_entries: 1024,
            max_json_object_entries: 1024,
            max_operation_ledger_entries: 1024,
            max_model_call_lineage_entries: 64,
            max_session_model_call_lineage_entries: 1024,
            max_session_model_call_lineage_bytes: 1024 * 1024,
            max_artifact_handles: 64,
            max_artifact_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Immutable admission parameters for one supervised adapter session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterProtocolConfig {
    role: AdapterRole,
    episode: String,
    capabilities: BTreeSet<ProtocolCapability>,
    allowed_model_bindings: BTreeSet<ModelBindingId>,
    limits: ProtocolLimits,
}

impl AdapterProtocolConfig {
    /// Validates a role-pinned protocol admission configuration.
    pub fn new(
        role: AdapterRole,
        episode: impl Into<String>,
        capabilities: BTreeSet<ProtocolCapability>,
        allowed_model_bindings: BTreeSet<ModelBindingId>,
        limits: ProtocolLimits,
    ) -> Result<Self, ProtocolError> {
        let episode = episode.into();
        validate_limits(&limits)?;
        validate_identifier(&episode, &limits, "episode")?;
        if !capabilities.contains(&role_capability(role)) {
            return Err(ProtocolError::CapabilityNotDeclared(role_capability(role)));
        }
        Ok(Self {
            role,
            episode,
            capabilities,
            allowed_model_bindings,
            limits,
        })
    }

    /// Returns the Rust-pinned adapter role selected for this session.
    pub const fn role(&self) -> AdapterRole {
        self.role
    }

    /// Returns the Rust-assigned episode correlation for this session.
    pub fn episode(&self) -> &str {
        &self.episode
    }

    /// Returns the exact capability set Rust selected for this session.
    pub fn capabilities(&self) -> &BTreeSet<ProtocolCapability> {
        &self.capabilities
    }

    /// Returns the immutable maximum wire frame selected for this session.
    pub const fn max_frame_bytes(&self) -> usize {
        self.limits.max_frame_bytes
    }
}

/// One Rust-to-adapter JSONL envelope.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostEnvelope {
    /// Strict protocol version.
    pub version: u16,
    /// Rust-assigned episode correlation.
    pub episode: String,
    /// Rust-assigned causal span correlation.
    pub span: String,
    /// Monotonic host sequence for this adapter session.
    pub sequence: u64,
    /// Never-reused operation correlation.
    pub operation: String,
    /// Directional host message.
    pub message: HostMessage,
}

impl HostEnvelope {
    /// Constructs a current-version envelope before protocol admission.
    pub fn new(
        episode: impl Into<String>,
        span: impl Into<String>,
        sequence: u64,
        operation: impl Into<String>,
        message: HostMessage,
    ) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            episode: episode.into(),
            span: span.into(),
            sequence,
            operation: operation.into(),
            message,
        }
    }
}

/// One adapter-to-Rust JSONL envelope.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterEnvelope {
    /// Strict protocol version.
    pub version: u16,
    /// Rust-assigned episode correlation.
    pub episode: String,
    /// Rust-assigned causal span correlation.
    pub span: String,
    /// Monotonic adapter sequence for this session.
    pub sequence: u64,
    /// Never-reused operation correlation.
    pub operation: String,
    /// Directional adapter message.
    pub message: AdapterMessage,
}

impl AdapterEnvelope {
    /// Constructs a current-version adapter envelope before host admission.
    pub fn new(
        episode: impl Into<String>,
        span: impl Into<String>,
        sequence: u64,
        operation: impl Into<String>,
        message: AdapterMessage,
    ) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            episode: episode.into(),
            span: span.into(),
            sequence,
            operation: operation.into(),
            message,
        }
    }
}

/// Rust-only messages sent to one supervised adapter.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HostMessage {
    /// Selects one role and exact capability set before the first reset.
    Hello {
        /// Wire versions the host can speak in preference order.
        supported_versions: Vec<u16>,
        /// Role Rust provisioned for this child.
        adapter_role: AdapterRole,
        /// Exact capabilities Rust selected for this child.
        capabilities: Vec<ProtocolCapability>,
    },
    /// Resets the adapter to a Rust-owned seed and frozen identities.
    Reset {
        /// Rust-owned deterministic seed.
        seed: u64,
        /// Opaque frozen artifact identities selected by Rust.
        identities: Vec<ArtifactDigest>,
    },
    /// Invokes one pinned tool operation.
    InvokeTool {
        /// Arbitrary but bounded Rust-validated tool input.
        input: Value,
    },
    /// Requests a policy or heuristic decision.
    Decide {
        /// Arbitrary but bounded decision context.
        input: Value,
    },
    /// Delivers the result of a Rust-dispatched model call.
    DeliverModelResult {
        /// Exact model-call correlation accepted earlier for this operation.
        model_call: String,
        /// Arbitrary but bounded Rust-captured model output.
        output: Value,
    },
    /// Requests the first external environment observation.
    ResetEnvironment {
        /// Rust-issued immutable reset input reference.
        input_ref: FrozenArtifactReference,
    },
    /// Requests exactly one external environment transition.
    StepEnvironment {
        /// Rust-issued immutable action reference.
        action_ref: FrozenArtifactReference,
    },
    /// Opens an externally-driven terminal candidate operation.
    RequestEpisodeTerminal {
        /// Arbitrary but bounded compatibility-driver terminal context.
        input: Value,
    },
    /// Grants one upload capability for the adapter's matching byte reservation.
    PutArtifactHandle {
        /// Store-issued opaque upload capability.
        upload: ArtifactUploadHandle,
        /// Exact bytes selected by the adapter request and store grant.
        declared_bytes: u64,
    },
    /// Closes an upload request after Rust freezes the exact granted bytes.
    ArtifactCommitted {
        /// Same store-issued upload capability Rust granted.
        upload: ArtifactUploadHandle,
        /// Rust-issued immutable reference after exact streaming commit.
        reference: FrozenArtifactReference,
    },
    /// Answers an adapter artifact-read request with a bounded read capability.
    GetArtifactHandle {
        /// Store-issued opaque read capability.
        download: ArtifactDownloadHandle,
        /// Exact frozen length the read stream will carry.
        length: u64,
    },
    /// Cancels exactly one live target operation.
    Cancel {
        /// Existing operation that transitions to cancelling until acknowledged.
        target_operation: String,
    },
    /// Ends an idle protocol session after all live operations close.
    Shutdown,
}

/// Adapter-only messages received from one supervised child.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterMessage {
    /// Acknowledges the protocol version and exact host-selected capabilities.
    Ready {
        /// The selected supported version.
        protocol_version: u16,
        /// Capabilities implemented by the pinned child for this session.
        capabilities: Vec<ProtocolCapability>,
        /// Pinned adapter implementation identity.
        implementation_digest: ArtifactDigest,
    },
    /// Acknowledges a Rust-owned reset.
    ResetAck {
        /// The applied seed.
        effective_seed: u64,
        /// Implementation identity after reset.
        implementation_digest: ArtifactDigest,
    },
    /// Returns a tool result.
    ToolResult {
        /// Arbitrary but bounded tool result.
        output: Value,
    },
    /// Asks Rust to dispatch a pinned model binding.
    ModelIntent {
        /// Live correlation retained until exact result delivery or terminal cleanup.
        model_call: String,
        /// Rust-pinned binding selection; never a network origin or credential.
        binding: ModelBindingId,
        /// Arbitrary but bounded model input.
        input: Value,
    },
    /// Returns a policy or heuristic decision.
    Decision {
        /// Arbitrary but bounded decision output.
        output: Value,
    },
    /// Returns the first external environment observation.
    EnvironmentReset {
        /// Rust-issued immutable initial-observation reference.
        observation_ref: FrozenArtifactReference,
    },
    /// Returns exactly one external environment transition.
    Transition {
        /// Rust-issued immutable observation reference.
        observation_ref: FrozenArtifactReference,
        /// Finite source-reported reward retained as evidence.
        reward: f64,
        /// Whether the environment terminally ended.
        terminated: bool,
        /// Whether the environment truncated without terminal completion.
        truncated: bool,
        /// Rust-issued immutable environment diagnostics reference.
        info_ref: FrozenArtifactReference,
    },
    /// Confirms that a previously granted upload now contains its exact declared bytes.
    ArtifactUploadComplete {
        /// Same store-issued upload capability Rust granted for this operation.
        upload: ArtifactUploadHandle,
    },
    /// Streams one bounded base64 fragment into the exact Rust-issued upload capability.
    ArtifactUploadChunk {
        /// Same store-issued upload capability Rust granted for this operation.
        upload: ArtifactUploadHandle,
        /// One nonempty base64 fragment decoded only by the Rust-owned artifact authority.
        bytes_base64: String,
    },
    /// Starts a store-owned bounded artifact upload request.
    PutArtifactRequest {
        /// Existing Rust-origin operation that supplies this request's causal span.
        parent_operation: String,
        /// Exact requested byte count before Rust grants a capability.
        declared_bytes: u64,
    },
    /// Requests a Rust-owned artifact read capability.
    GetArtifactRequest {
        /// Existing Rust-origin operation that supplies this request's causal span.
        parent_operation: String,
        /// Arbitrary but bounded host-interpreted request selector.
        request: Value,
    },
    /// Emits a nonterminal snapshot for the current live operation.
    Checkpoint {
        /// Arbitrary but bounded checkpoint data.
        data: Value,
    },
    /// Acknowledges cancellation of its named target operation.
    CancelAck {
        /// Exact cancellation target from the host request.
        target_operation: String,
    },
    /// Acknowledges an idle-session shutdown.
    ShutdownAck,
    /// Closes a compatibility-driver terminal candidate operation.
    EpisodeTerminalCandidate {
        /// Arbitrary but bounded terminal output proposal.
        output: Value,
    },
    /// Fails one existing operation without inventing an episode terminal result.
    OperationFailed {
        /// Bounded machine-readable failure code.
        code: String,
        /// Arbitrary but bounded failure evidence.
        details: Value,
    },
}

/// A host message that passed version, bound, direction, and transition checks.
#[derive(Clone, Debug)]
pub struct ValidatedHostMessage(HostEnvelope);

impl ValidatedHostMessage {
    /// Borrows the admitted host envelope.
    pub fn envelope(&self) -> &HostEnvelope {
        &self.0
    }
}

/// An adapter message that passed version, bound, direction, and transition checks.
#[derive(Clone, Debug)]
pub struct ValidatedAdapterMessage(AdapterEnvelope);

impl ValidatedAdapterMessage {
    /// Borrows the admitted adapter envelope.
    pub fn envelope(&self) -> &AdapterEnvelope {
        &self.0
    }
}

/// Object-safe codec and protocol-state seam consumed by later supervision.
pub trait AdapterProtocol {
    /// Encodes one already-admitted host message as exactly one bounded JSONL frame.
    fn encode_host_frame(&self, message: &ValidatedHostMessage) -> Result<Vec<u8>, ProtocolError>;

    /// Decodes exactly one bounded adapter JSONL frame without admitting its state transition.
    fn decode_adapter_frame(&self, frame: &[u8]) -> Result<AdapterEnvelope, ProtocolError>;

    /// Admits one host transition and returns only the validated message.
    fn accept_host(
        &mut self,
        envelope: HostEnvelope,
    ) -> Result<ValidatedHostMessage, ProtocolError>;

    /// Admits one adapter transition and returns only the validated message.
    fn accept_adapter(
        &mut self,
        envelope: AdapterEnvelope,
    ) -> Result<ValidatedAdapterMessage, ProtocolError>;

    /// Decodes and admits one adapter JSONL frame.
    fn accept_adapter_frame(
        &mut self,
        frame: &[u8],
    ) -> Result<ValidatedAdapterMessage, ProtocolError> {
        let envelope = self.decode_adapter_frame(frame)?;
        self.accept_adapter(envelope)
    }

    /// Releases a consumed or revoked download capability from the protocol ledger.
    fn release_download_handle(
        &mut self,
        download: &ArtifactDownloadHandle,
    ) -> Result<(), ProtocolError>;

    /// Returns the terminal session disposition.
    fn session_state(&self) -> ProtocolSessionState;

    /// Returns one permanently recorded operation disposition.
    fn operation_state(&self, operation: &str) -> Option<ProtocolOperationState>;
}

/// Object-safe constructor seam for a role-pinned adapter protocol.
pub trait AdapterProtocolFactory {
    /// Creates a new private protocol codec and state machine.
    fn create(
        &self,
        config: AdapterProtocolConfig,
    ) -> Result<Box<dyn AdapterProtocol>, ProtocolError>;
}

/// Built-in strict JSONL codec and protocol implementation factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct StrictAdapterProtocolFactory;

impl AdapterProtocolFactory for StrictAdapterProtocolFactory {
    fn create(
        &self,
        config: AdapterProtocolConfig,
    ) -> Result<Box<dyn AdapterProtocol>, ProtocolError> {
        Ok(Box::new(StrictAdapterProtocol::new(config)))
    }
}

/// Session-level protocol disposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolSessionState {
    /// Waiting for the host hello and adapter ready exchange.
    Negotiating,
    /// Ready to receive or issue bounded operations.
    Ready,
    /// Cleanly closed after a shutdown acknowledgement.
    Closed,
    /// Failed closed after a remote protocol violation.
    Failed,
}

/// Permanently retained operation disposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolOperationState {
    /// Waiting for a correlated response or host continuation.
    Pending,
    /// The host requested cancellation and awaits acknowledgement.
    Cancelling,
    /// The operation completed normally or was cancelled.
    Closed,
    /// The adapter explicitly failed the operation.
    Failed,
}

struct StrictAdapterProtocol {
    config: AdapterProtocolConfig,
    session: ProtocolSessionState,
    next_host_sequence: u64,
    next_adapter_sequence: u64,
    operations: BTreeMap<String, OperationRecord>,
    model_calls: BTreeMap<String, ModelCallLineage>,
    model_call_lineage_entries: usize,
    model_call_lineage_bytes: usize,
    active_upload_handles: BTreeSet<String>,
    active_download_handles: BTreeSet<String>,
}

struct OperationRecord {
    span: String,
    origin: OperationOrigin,
    state: OperationState,
}

#[derive(Default)]
struct ModelCallLineage {
    active: Option<String>,
    completed: BTreeSet<String>,
    entries: usize,
    bytes: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum OperationOrigin {
    Host,
    Adapter,
}

enum OperationState {
    AwaitReady {
        capabilities: BTreeSet<ProtocolCapability>,
    },
    AwaitResetAck,
    AwaitToolResult,
    AwaitDecision,
    AwaitModelResult {
        model_call: String,
    },
    AwaitEnvironmentReset,
    AwaitTransition,
    AwaitTerminalCandidate,
    AwaitArtifactGrant {
        declared_bytes: u64,
    },
    AwaitArtifactCommit {
        upload: ArtifactUploadHandle,
        declared_bytes: u64,
    },
    AwaitDownloadGrant,
    AwaitCancelAck {
        target_operation: String,
    },
    AwaitShutdownAck,
    Cancelling,
    Closed,
    Failed,
}

impl StrictAdapterProtocol {
    fn new(config: AdapterProtocolConfig) -> Self {
        Self {
            config,
            session: ProtocolSessionState::Negotiating,
            next_host_sequence: 0,
            next_adapter_sequence: 0,
            operations: BTreeMap::new(),
            model_calls: BTreeMap::new(),
            model_call_lineage_entries: 0,
            model_call_lineage_bytes: 0,
            active_upload_handles: BTreeSet::new(),
            active_download_handles: BTreeSet::new(),
        }
    }

    fn accept_host_inner(&mut self, envelope: &HostEnvelope) -> Result<(), ProtocolError> {
        self.validate_host_envelope(envelope)?;
        if matches!(
            &envelope.message,
            HostMessage::DeliverModelResult { .. }
                | HostMessage::PutArtifactHandle { .. }
                | HostMessage::ArtifactCommitted { .. }
                | HostMessage::GetArtifactHandle { .. }
        ) {
            self.require_operation_span(&envelope.operation, &envelope.span)?;
        }
        let message = envelope.message.clone();
        match message {
            HostMessage::Hello {
                supported_versions,
                adapter_role,
                capabilities,
            } => {
                self.require_new_operation(&envelope.operation)?;
                if self.session != ProtocolSessionState::Negotiating {
                    return Err(ProtocolError::SessionState(self.session));
                }
                if adapter_role != self.config.role {
                    return Err(ProtocolError::RoleMismatch {
                        expected: self.config.role,
                        actual: adapter_role,
                    });
                }
                if !supported_versions.contains(&PROTOCOL_VERSION) {
                    return Err(ProtocolError::UnsupportedVersion(PROTOCOL_VERSION));
                }
                let capabilities = capability_set(capabilities)?;
                if capabilities != self.config.capabilities {
                    return Err(ProtocolError::ReadyCapabilitiesMismatch);
                }
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitReady { capabilities },
                )?;
            }
            HostMessage::Reset { identities, .. } => {
                self.require_ready()?;
                self.require_new_operation(&envelope.operation)?;
                if identities.len() > self.config.limits.max_artifact_handles {
                    return Err(ProtocolError::ArtifactHandleLimit {
                        limit: self.config.limits.max_artifact_handles,
                    });
                }
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitResetAck,
                )?;
            }
            HostMessage::InvokeTool { input } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Tool)?;
                self.require_role(AdapterRole::Tool)?;
                self.require_new_operation(&envelope.operation)?;
                validate_json(&input, &self.config.limits)?;
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitToolResult,
                )?;
            }
            HostMessage::Decide { input } => {
                self.require_ready()?;
                if self.config.role != AdapterRole::Policy
                    && self.config.role != AdapterRole::Heuristic
                {
                    return Err(ProtocolError::MessageForbiddenForRole(self.config.role));
                }
                self.require_capability(if self.config.role == AdapterRole::Policy {
                    ProtocolCapability::Policy
                } else {
                    ProtocolCapability::Heuristic
                })?;
                self.require_new_operation(&envelope.operation)?;
                validate_json(&input, &self.config.limits)?;
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitDecision,
                )?;
                if self.config.role == AdapterRole::Policy {
                    self.model_calls
                        .insert(envelope.operation.clone(), ModelCallLineage::default());
                }
            }
            HostMessage::DeliverModelResult { model_call, output } => {
                self.require_ready()?;
                validate_identifier(&model_call, &self.config.limits, "model call")?;
                validate_json(&output, &self.config.limits)?;
                let expected = match &self.require_operation(&envelope.operation)?.state {
                    OperationState::AwaitModelResult { model_call } => model_call.clone(),
                    state => {
                        return Err(ProtocolError::OperationState {
                            operation: envelope.operation.clone(),
                            state: operation_disposition(state),
                        });
                    }
                };
                if expected != model_call {
                    return Err(ProtocolError::ModelCallMismatch {
                        expected,
                        actual: model_call,
                    });
                }
                self.complete_model_call(&model_call, &envelope.operation)?;
                self.require_operation_mut(&envelope.operation)?.state =
                    OperationState::AwaitDecision;
            }
            HostMessage::ResetEnvironment { input_ref } => {
                self.require_ready()?;
                self.require_role(AdapterRole::Environment)?;
                self.require_capability(ProtocolCapability::Environment)?;
                self.require_new_operation(&envelope.operation)?;
                self.validate_frozen_reference(&input_ref)?;
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitEnvironmentReset,
                )?;
            }
            HostMessage::StepEnvironment { action_ref } => {
                self.require_ready()?;
                self.require_role(AdapterRole::Environment)?;
                self.require_capability(ProtocolCapability::Environment)?;
                self.require_new_operation(&envelope.operation)?;
                self.validate_frozen_reference(&action_ref)?;
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitTransition,
                )?;
            }
            HostMessage::RequestEpisodeTerminal { input } => {
                self.require_ready()?;
                self.require_role(AdapterRole::Driver)?;
                self.require_capability(ProtocolCapability::Driver)?;
                self.require_new_operation(&envelope.operation)?;
                validate_json(&input, &self.config.limits)?;
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitTerminalCandidate,
                )?;
            }
            HostMessage::PutArtifactHandle {
                upload,
                declared_bytes,
            } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Artifacts)?;
                validate_identifier(upload.as_str(), &self.config.limits, "upload capability")?;
                self.validate_artifact_length(declared_bytes)?;
                let expected = match &self.require_operation(&envelope.operation)?.state {
                    OperationState::AwaitArtifactGrant { declared_bytes } => *declared_bytes,
                    state => {
                        return Err(ProtocolError::OperationState {
                            operation: envelope.operation.clone(),
                            state: operation_disposition(state),
                        });
                    }
                };
                if expected != declared_bytes {
                    return Err(ProtocolError::ArtifactLengthMismatch {
                        expected,
                        actual: declared_bytes,
                    });
                }
                self.reserve_upload_handle(upload.as_str())?;
                self.require_operation_mut(&envelope.operation)?.state =
                    OperationState::AwaitArtifactCommit {
                        upload,
                        declared_bytes,
                    };
            }
            HostMessage::ArtifactCommitted { upload, reference } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Artifacts)?;
                validate_identifier(upload.as_str(), &self.config.limits, "upload capability")?;
                validate_identifier(
                    reference.download().as_str(),
                    &self.config.limits,
                    "download capability",
                )?;
                self.validate_artifact_length(reference.artifact().length())?;
                let (expected_upload, declared_bytes) =
                    match &self.require_operation(&envelope.operation)?.state {
                        OperationState::AwaitArtifactCommit {
                            upload,
                            declared_bytes,
                        } => (upload.clone(), *declared_bytes),
                        state => {
                            return Err(ProtocolError::OperationState {
                                operation: envelope.operation.clone(),
                                state: operation_disposition(state),
                            });
                        }
                    };
                if expected_upload != upload {
                    return Err(ProtocolError::ArtifactUploadMismatch {
                        expected: expected_upload.as_str().to_owned(),
                        actual: upload.as_str().to_owned(),
                    });
                }
                if declared_bytes != reference.artifact().length() {
                    return Err(ProtocolError::ArtifactLengthMismatch {
                        expected: declared_bytes,
                        actual: reference.artifact().length(),
                    });
                }
                self.replace_upload_with_download(upload.as_str(), reference.download().as_str())?;
                self.require_operation_mut(&envelope.operation)?.state = OperationState::Closed;
            }
            HostMessage::GetArtifactHandle { download, length } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Artifacts)?;
                validate_identifier(
                    download.as_str(),
                    &self.config.limits,
                    "download capability",
                )?;
                self.validate_artifact_length(length)?;
                let state = &self.require_operation(&envelope.operation)?.state;
                if !matches!(state, OperationState::AwaitDownloadGrant) {
                    return Err(ProtocolError::OperationState {
                        operation: envelope.operation.clone(),
                        state: operation_disposition(state),
                    });
                }
                self.reserve_download_handle(download.as_str())?;
                self.require_operation_mut(&envelope.operation)?.state = OperationState::Closed;
            }
            HostMessage::Cancel { target_operation } => {
                self.require_ready()?;
                self.require_new_operation(&envelope.operation)?;
                validate_identifier(&target_operation, &self.config.limits, "cancel target")?;
                if target_operation == envelope.operation {
                    return Err(ProtocolError::CancelTargetInvalid(target_operation));
                }
                {
                    let target = self.require_operation(&target_operation)?;
                    if !is_cancellable(&target.state) {
                        return Err(ProtocolError::CancelTargetInvalid(target_operation));
                    }
                }
                self.ensure_operation_capacity()?;
                self.clear_model_call_lineage(&target_operation)?;
                self.require_operation_mut(&target_operation)?.state = OperationState::Cancelling;
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitCancelAck { target_operation },
                )?;
            }
            HostMessage::Shutdown => {
                self.require_ready()?;
                self.require_new_operation(&envelope.operation)?;
                if self
                    .operations
                    .values()
                    .any(|record| is_pending(&record.state))
                {
                    return Err(ProtocolError::ShutdownWithLiveOperations);
                }
                self.insert_operation(
                    envelope.operation.clone(),
                    envelope.span.clone(),
                    OperationState::AwaitShutdownAck,
                )?;
            }
        }
        Ok(())
    }

    fn accept_adapter_inner(&mut self, envelope: &AdapterEnvelope) -> Result<(), ProtocolError> {
        self.validate_adapter_envelope(envelope)?;
        if matches!(
            &envelope.message,
            AdapterMessage::EpisodeTerminalCandidate { .. }
        ) {
            self.require_role(AdapterRole::Driver)?;
            self.require_capability(ProtocolCapability::Driver)?;
        }
        if !matches!(
            &envelope.message,
            AdapterMessage::PutArtifactRequest { .. } | AdapterMessage::GetArtifactRequest { .. }
        ) {
            self.require_operation_span(&envelope.operation, &envelope.span)?;
        }
        let message = envelope.message.clone();
        match message {
            AdapterMessage::Ready {
                protocol_version,
                capabilities,
                ..
            } => {
                if protocol_version != PROTOCOL_VERSION {
                    return Err(ProtocolError::UnsupportedVersion(protocol_version));
                }
                let capabilities = capability_set(capabilities)?;
                let record = self.require_operation_mut(&envelope.operation)?;
                let OperationState::AwaitReady {
                    capabilities: expected,
                } = &record.state
                else {
                    return Err(ProtocolError::OperationState {
                        operation: envelope.operation.clone(),
                        state: operation_disposition(&record.state),
                    });
                };
                if expected != &capabilities {
                    return Err(ProtocolError::ReadyCapabilitiesMismatch);
                }
                record.state = OperationState::Closed;
                self.session = ProtocolSessionState::Ready;
            }
            AdapterMessage::ResetAck { .. } => {
                self.close_response(&envelope.operation, |state| {
                    matches!(state, OperationState::AwaitResetAck)
                })?;
            }
            AdapterMessage::ToolResult { output } => {
                self.require_role(AdapterRole::Tool)?;
                self.require_capability(ProtocolCapability::Tool)?;
                validate_json(&output, &self.config.limits)?;
                self.close_response(&envelope.operation, |state| {
                    matches!(state, OperationState::AwaitToolResult)
                })?;
            }
            AdapterMessage::ModelIntent {
                model_call,
                binding,
                input,
            } => {
                self.require_role(AdapterRole::Policy)?;
                self.require_capability(ProtocolCapability::Policy)?;
                validate_identifier(&model_call, &self.config.limits, "model call")?;
                validate_json(&input, &self.config.limits)?;
                if !self.config.allowed_model_bindings.contains(&binding) {
                    return Err(ProtocolError::ModelBindingNotAllowed(
                        binding.as_str().to_owned(),
                    ));
                }
                let state = &self.require_operation(&envelope.operation)?.state;
                if !matches!(state, OperationState::AwaitDecision) {
                    return Err(ProtocolError::OperationState {
                        operation: envelope.operation.clone(),
                        state: operation_disposition(state),
                    });
                }
                self.reserve_model_call(&model_call, &envelope.operation)?;
                self.require_operation_mut(&envelope.operation)?.state =
                    OperationState::AwaitModelResult { model_call };
            }
            AdapterMessage::Decision { output } => {
                if self.config.role != AdapterRole::Policy
                    && self.config.role != AdapterRole::Heuristic
                {
                    return Err(ProtocolError::MessageForbiddenForRole(self.config.role));
                }
                self.require_capability(if self.config.role == AdapterRole::Policy {
                    ProtocolCapability::Policy
                } else {
                    ProtocolCapability::Heuristic
                })?;
                validate_json(&output, &self.config.limits)?;
                self.close_response(&envelope.operation, |state| {
                    matches!(state, OperationState::AwaitDecision)
                })?;
                self.clear_model_call_lineage(&envelope.operation)?;
            }
            AdapterMessage::EnvironmentReset { observation_ref } => {
                self.require_role(AdapterRole::Environment)?;
                self.require_capability(ProtocolCapability::Environment)?;
                self.validate_granted_frozen_reference(&observation_ref)?;
                self.close_response(&envelope.operation, |state| {
                    matches!(state, OperationState::AwaitEnvironmentReset)
                })?;
            }
            AdapterMessage::Transition {
                observation_ref,
                reward,
                info_ref,
                ..
            } => {
                self.require_role(AdapterRole::Environment)?;
                self.require_capability(ProtocolCapability::Environment)?;
                if !reward.is_finite() {
                    return Err(ProtocolError::NonFiniteReward);
                }
                self.validate_granted_frozen_reference(&observation_ref)?;
                self.validate_granted_frozen_reference(&info_ref)?;
                self.close_response(&envelope.operation, |state| {
                    matches!(state, OperationState::AwaitTransition)
                })?;
            }
            AdapterMessage::PutArtifactRequest {
                parent_operation,
                declared_bytes,
            } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Artifacts)?;
                self.require_new_operation(&envelope.operation)?;
                self.validate_artifact_length(declared_bytes)?;
                let parent_span =
                    self.require_host_parent_span(&parent_operation, &envelope.span)?;
                self.insert_adapter_operation(
                    envelope.operation.clone(),
                    parent_span,
                    OperationState::AwaitArtifactGrant { declared_bytes },
                )?;
            }
            AdapterMessage::ArtifactUploadComplete { upload } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Artifacts)?;
                validate_identifier(upload.as_str(), &self.config.limits, "upload capability")?;
                let state = &self.require_operation(&envelope.operation)?.state;
                let OperationState::AwaitArtifactCommit {
                    upload: expected_upload,
                    ..
                } = state
                else {
                    return Err(ProtocolError::OperationState {
                        operation: envelope.operation.clone(),
                        state: operation_disposition(state),
                    });
                };
                if expected_upload != &upload {
                    return Err(ProtocolError::ArtifactUploadMismatch {
                        expected: expected_upload.as_str().to_owned(),
                        actual: upload.as_str().to_owned(),
                    });
                }
            }
            AdapterMessage::ArtifactUploadChunk {
                upload,
                bytes_base64,
            } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Artifacts)?;
                validate_identifier(upload.as_str(), &self.config.limits, "upload capability")?;
                if bytes_base64.is_empty() {
                    return Err(ProtocolError::ArtifactUploadChunkEmpty);
                }
                if bytes_base64.len() > self.config.limits.max_frame_bytes {
                    return Err(ProtocolError::FrameTooLarge {
                        limit: self.config.limits.max_frame_bytes,
                        actual: bytes_base64.len(),
                    });
                }
                let state = &self.require_operation(&envelope.operation)?.state;
                let OperationState::AwaitArtifactCommit {
                    upload: expected_upload,
                    ..
                } = state
                else {
                    return Err(ProtocolError::OperationState {
                        operation: envelope.operation.clone(),
                        state: operation_disposition(state),
                    });
                };
                if expected_upload != &upload {
                    return Err(ProtocolError::ArtifactUploadMismatch {
                        expected: expected_upload.as_str().to_owned(),
                        actual: upload.as_str().to_owned(),
                    });
                }
            }
            AdapterMessage::GetArtifactRequest {
                parent_operation,
                request,
            } => {
                self.require_ready()?;
                self.require_capability(ProtocolCapability::Artifacts)?;
                self.require_new_operation(&envelope.operation)?;
                validate_json(&request, &self.config.limits)?;
                let parent_span =
                    self.require_host_parent_span(&parent_operation, &envelope.span)?;
                self.insert_adapter_operation(
                    envelope.operation.clone(),
                    parent_span,
                    OperationState::AwaitDownloadGrant,
                )?;
            }
            AdapterMessage::Checkpoint { data } => {
                self.require_capability(ProtocolCapability::Checkpoint)?;
                validate_json(&data, &self.config.limits)?;
                let record = self.require_operation(&envelope.operation)?;
                if !is_checkpointable(&record.state) {
                    return Err(ProtocolError::OperationState {
                        operation: envelope.operation.clone(),
                        state: operation_disposition(&record.state),
                    });
                }
            }
            AdapterMessage::CancelAck { target_operation } => {
                validate_identifier(&target_operation, &self.config.limits, "cancel target")?;
                let expected_target = {
                    let record = self.require_operation(&envelope.operation)?;
                    let OperationState::AwaitCancelAck { target_operation } = &record.state else {
                        return Err(ProtocolError::OperationState {
                            operation: envelope.operation.clone(),
                            state: operation_disposition(&record.state),
                        });
                    };
                    target_operation.clone()
                };
                if expected_target != target_operation {
                    return Err(ProtocolError::CancelTargetMismatch {
                        expected: expected_target,
                        actual: target_operation,
                    });
                }
                let target = self.require_operation_mut(&expected_target)?;
                if !matches!(target.state, OperationState::Cancelling) {
                    return Err(ProtocolError::CancelTargetInvalid(expected_target));
                }
                target.state = OperationState::Closed;
                self.require_operation_mut(&envelope.operation)?.state = OperationState::Closed;
            }
            AdapterMessage::ShutdownAck => {
                self.close_response(&envelope.operation, |state| {
                    matches!(state, OperationState::AwaitShutdownAck)
                })?;
                self.session = ProtocolSessionState::Closed;
            }
            AdapterMessage::EpisodeTerminalCandidate { output } => {
                self.require_role(AdapterRole::Driver)?;
                self.require_capability(ProtocolCapability::Driver)?;
                validate_json(&output, &self.config.limits)?;
                self.close_response(&envelope.operation, |state| {
                    matches!(state, OperationState::AwaitTerminalCandidate)
                })?;
            }
            AdapterMessage::OperationFailed { code, details } => {
                validate_identifier(&code, &self.config.limits, "failure code")?;
                validate_json(&details, &self.config.limits)?;
                let state = &self.require_operation(&envelope.operation)?.state;
                if !is_pending(state) {
                    return Err(ProtocolError::OperationState {
                        operation: envelope.operation.clone(),
                        state: operation_disposition(state),
                    });
                }
                self.clear_model_call_lineage(&envelope.operation)?;
                self.require_operation_mut(&envelope.operation)?.state = OperationState::Failed;
            }
        }
        Ok(())
    }

    fn validate_host_envelope(&mut self, envelope: &HostEnvelope) -> Result<(), ProtocolError> {
        self.validate_common(
            envelope.version,
            &envelope.episode,
            &envelope.span,
            &envelope.operation,
        )?;
        self.validate_sequence(envelope.sequence, true)
    }

    fn validate_adapter_envelope(
        &mut self,
        envelope: &AdapterEnvelope,
    ) -> Result<(), ProtocolError> {
        self.validate_common(
            envelope.version,
            &envelope.episode,
            &envelope.span,
            &envelope.operation,
        )?;
        self.validate_sequence(envelope.sequence, false)
    }

    fn validate_common(
        &self,
        version: u16,
        episode: &str,
        span: &str,
        operation: &str,
    ) -> Result<(), ProtocolError> {
        if version != PROTOCOL_VERSION {
            return Err(ProtocolError::UnsupportedVersion(version));
        }
        if episode != self.config.episode {
            return Err(ProtocolError::EpisodeMismatch {
                expected: self.config.episode.clone(),
                actual: episode.to_owned(),
            });
        }
        validate_identifier(span, &self.config.limits, "span")?;
        validate_identifier(operation, &self.config.limits, "operation")
    }

    fn validate_sequence(&mut self, sequence: u64, host: bool) -> Result<(), ProtocolError> {
        let expected = if host {
            &mut self.next_host_sequence
        } else {
            &mut self.next_adapter_sequence
        };
        if sequence != *expected {
            return Err(ProtocolError::SequenceOutOfOrder {
                expected: *expected,
                actual: sequence,
            });
        }
        *expected = expected
            .checked_add(1)
            .ok_or(ProtocolError::SequenceExhausted)?;
        Ok(())
    }

    fn require_ready(&self) -> Result<(), ProtocolError> {
        if self.session == ProtocolSessionState::Ready {
            Ok(())
        } else {
            Err(ProtocolError::SessionState(self.session))
        }
    }

    fn require_role(&self, role: AdapterRole) -> Result<(), ProtocolError> {
        if self.config.role == role {
            Ok(())
        } else {
            Err(ProtocolError::MessageForbiddenForRole(self.config.role))
        }
    }

    fn require_capability(&self, capability: ProtocolCapability) -> Result<(), ProtocolError> {
        if self.config.capabilities.contains(&capability) {
            Ok(())
        } else {
            Err(ProtocolError::CapabilityNotDeclared(capability))
        }
    }

    fn require_new_operation(&self, operation: &str) -> Result<(), ProtocolError> {
        if self.operations.contains_key(operation) {
            Err(ProtocolError::OperationAlreadyUsed(operation.to_owned()))
        } else {
            Ok(())
        }
    }

    fn require_operation(&self, operation: &str) -> Result<&OperationRecord, ProtocolError> {
        self.operations
            .get(operation)
            .ok_or_else(|| ProtocolError::UnknownOperation(operation.to_owned()))
    }

    fn require_operation_mut(
        &mut self,
        operation: &str,
    ) -> Result<&mut OperationRecord, ProtocolError> {
        self.operations
            .get_mut(operation)
            .ok_or_else(|| ProtocolError::UnknownOperation(operation.to_owned()))
    }

    fn require_operation_span(&self, operation: &str, span: &str) -> Result<(), ProtocolError> {
        let record = self.require_operation(operation)?;
        if record.span == span {
            Ok(())
        } else {
            Err(ProtocolError::SpanMismatch {
                operation: operation.to_owned(),
                expected: record.span.clone(),
                actual: span.to_owned(),
            })
        }
    }

    fn require_host_parent_span(
        &self,
        parent_operation: &str,
        adapter_span: &str,
    ) -> Result<String, ProtocolError> {
        validate_identifier(
            parent_operation,
            &self.config.limits,
            "artifact parent operation",
        )?;
        let parent = self.require_operation(parent_operation)?;
        if parent.origin != OperationOrigin::Host || !is_pending(&parent.state) {
            return Err(ProtocolError::ArtifactParentInvalid(
                parent_operation.to_owned(),
            ));
        }
        if parent.span != adapter_span {
            return Err(ProtocolError::SpanMismatch {
                operation: parent_operation.to_owned(),
                expected: parent.span.clone(),
                actual: adapter_span.to_owned(),
            });
        }
        Ok(parent.span.clone())
    }

    fn insert_operation(
        &mut self,
        operation: String,
        span: String,
        state: OperationState,
    ) -> Result<(), ProtocolError> {
        self.insert_operation_with_origin(operation, span, OperationOrigin::Host, state)
    }

    fn insert_adapter_operation(
        &mut self,
        operation: String,
        span: String,
        state: OperationState,
    ) -> Result<(), ProtocolError> {
        self.insert_operation_with_origin(operation, span, OperationOrigin::Adapter, state)
    }

    fn insert_operation_with_origin(
        &mut self,
        operation: String,
        span: String,
        origin: OperationOrigin,
        state: OperationState,
    ) -> Result<(), ProtocolError> {
        self.ensure_operation_capacity()?;
        self.operations.insert(
            operation,
            OperationRecord {
                span,
                origin,
                state,
            },
        );
        Ok(())
    }

    fn ensure_operation_capacity(&self) -> Result<(), ProtocolError> {
        if self.operations.len() >= self.config.limits.max_operation_ledger_entries {
            return Err(ProtocolError::OperationLedgerLimit {
                limit: self.config.limits.max_operation_ledger_entries,
            });
        }
        Ok(())
    }

    fn close_response(
        &mut self,
        operation: &str,
        is_expected: impl FnOnce(&OperationState) -> bool,
    ) -> Result<(), ProtocolError> {
        let record = self.require_operation_mut(operation)?;
        if !is_expected(&record.state) {
            return Err(ProtocolError::OperationState {
                operation: operation.to_owned(),
                state: operation_disposition(&record.state),
            });
        }
        record.state = OperationState::Closed;
        Ok(())
    }

    fn validate_artifact_length(&self, length: u64) -> Result<(), ProtocolError> {
        if length > self.config.limits.max_artifact_bytes {
            Err(ProtocolError::ArtifactLengthTooLarge {
                limit: self.config.limits.max_artifact_bytes,
                actual: length,
            })
        } else {
            Ok(())
        }
    }

    fn validate_frozen_reference(
        &self,
        reference: &FrozenArtifactReference,
    ) -> Result<(), ProtocolError> {
        validate_identifier(
            reference.download().as_str(),
            &self.config.limits,
            "download capability",
        )?;
        // Environment bindings preauthorize immutable references before the
        // initial reset; the session ledger tracks only request/response grants.
        self.validate_artifact_length(reference.artifact().length())
    }

    fn validate_granted_frozen_reference(
        &self,
        reference: &FrozenArtifactReference,
    ) -> Result<(), ProtocolError> {
        self.validate_frozen_reference(reference)?;
        if self
            .active_download_handles
            .contains(reference.download().as_str())
        {
            Ok(())
        } else {
            Err(ProtocolError::UnknownArtifactDownloadHandle(
                reference.download().as_str().to_owned(),
            ))
        }
    }

    fn reserve_model_call(
        &mut self,
        model_call: &str,
        operation: &str,
    ) -> Result<(), ProtocolError> {
        let lineage = self
            .model_calls
            .get_mut(operation)
            .ok_or_else(|| ProtocolError::ModelCallNotActive(model_call.to_owned()))?;
        if lineage.active.is_some() {
            return Err(ProtocolError::ModelCallNotActive(model_call.to_owned()));
        }
        if lineage.completed.contains(model_call) {
            return Err(ProtocolError::ModelCallAlreadyUsed(model_call.to_owned()));
        }
        if lineage.entries >= self.config.limits.max_model_call_lineage_entries {
            return Err(ProtocolError::ModelCallLimit {
                limit: self.config.limits.max_model_call_lineage_entries,
            });
        }
        let next_lineage_entries = lineage
            .entries
            .checked_add(1)
            .ok_or(ProtocolError::ModelCallLineageAccounting)?;
        let next_lineage_bytes = lineage
            .bytes
            .checked_add(model_call.len())
            .ok_or(ProtocolError::ModelCallLineageAccounting)?;
        let next_entries = self.model_call_lineage_entries.checked_add(1).ok_or(
            ProtocolError::ModelCallSessionEntryLimit {
                limit: self.config.limits.max_session_model_call_lineage_entries,
            },
        )?;
        if next_entries > self.config.limits.max_session_model_call_lineage_entries {
            return Err(ProtocolError::ModelCallSessionEntryLimit {
                limit: self.config.limits.max_session_model_call_lineage_entries,
            });
        }
        let next_bytes = self
            .model_call_lineage_bytes
            .checked_add(model_call.len())
            .ok_or(ProtocolError::ModelCallSessionByteLimit {
                limit: self.config.limits.max_session_model_call_lineage_bytes,
            })?;
        if next_bytes > self.config.limits.max_session_model_call_lineage_bytes {
            return Err(ProtocolError::ModelCallSessionByteLimit {
                limit: self.config.limits.max_session_model_call_lineage_bytes,
            });
        }
        let lineage = self
            .model_calls
            .get_mut(operation)
            .ok_or_else(|| ProtocolError::ModelCallNotActive(model_call.to_owned()))?;
        lineage.active = Some(model_call.to_owned());
        lineage.entries = next_lineage_entries;
        lineage.bytes = next_lineage_bytes;
        self.model_call_lineage_entries = next_entries;
        self.model_call_lineage_bytes = next_bytes;
        Ok(())
    }

    fn complete_model_call(
        &mut self,
        model_call: &str,
        operation: &str,
    ) -> Result<(), ProtocolError> {
        let lineage = self
            .model_calls
            .get_mut(operation)
            .ok_or_else(|| ProtocolError::ModelCallNotActive(model_call.to_owned()))?;
        if lineage.active.as_deref() != Some(model_call) {
            return Err(ProtocolError::ModelCallNotActive(model_call.to_owned()));
        }
        lineage.active = None;
        lineage.completed.insert(model_call.to_owned());
        Ok(())
    }

    fn clear_model_call_lineage(&mut self, operation: &str) -> Result<(), ProtocolError> {
        let Some(lineage) = self.model_calls.get(operation) else {
            return Ok(());
        };
        let remaining_entries = self
            .model_call_lineage_entries
            .checked_sub(lineage.entries)
            .ok_or(ProtocolError::ModelCallLineageAccounting)?;
        let remaining_bytes = self
            .model_call_lineage_bytes
            .checked_sub(lineage.bytes)
            .ok_or(ProtocolError::ModelCallLineageAccounting)?;
        self.model_calls.remove(operation);
        self.model_call_lineage_entries = remaining_entries;
        self.model_call_lineage_bytes = remaining_bytes;
        Ok(())
    }

    fn ensure_artifact_handle_available(&self, handle: &str) -> Result<(), ProtocolError> {
        if self.active_upload_handles.contains(handle)
            || self.active_download_handles.contains(handle)
        {
            return Err(ProtocolError::ArtifactHandleAlreadyActive(
                handle.to_owned(),
            ));
        }
        if self
            .active_upload_handles
            .len()
            .checked_add(self.active_download_handles.len())
            .ok_or(ProtocolError::ArtifactHandleLimit {
                limit: self.config.limits.max_artifact_handles,
            })?
            >= self.config.limits.max_artifact_handles
        {
            return Err(ProtocolError::ArtifactHandleLimit {
                limit: self.config.limits.max_artifact_handles,
            });
        }
        Ok(())
    }

    fn reserve_upload_handle(&mut self, handle: &str) -> Result<(), ProtocolError> {
        self.ensure_artifact_handle_available(handle)?;
        self.active_upload_handles.insert(handle.to_owned());
        Ok(())
    }

    fn reserve_download_handle(&mut self, handle: &str) -> Result<(), ProtocolError> {
        self.ensure_artifact_handle_available(handle)?;
        self.active_download_handles.insert(handle.to_owned());
        Ok(())
    }

    fn replace_upload_with_download(
        &mut self,
        upload: &str,
        download: &str,
    ) -> Result<(), ProtocolError> {
        if !self.active_upload_handles.contains(upload) {
            return Err(ProtocolError::UnknownArtifactUploadHandle(
                upload.to_owned(),
            ));
        }
        if self.active_upload_handles.contains(download)
            || self.active_download_handles.contains(download)
        {
            return Err(ProtocolError::ArtifactHandleAlreadyActive(
                download.to_owned(),
            ));
        }
        let active_after_exchange = self
            .active_upload_handles
            .len()
            .checked_add(self.active_download_handles.len())
            .and_then(|count| count.checked_sub(1))
            .ok_or(ProtocolError::ArtifactHandleLimit {
                limit: self.config.limits.max_artifact_handles,
            })?;
        if active_after_exchange >= self.config.limits.max_artifact_handles {
            return Err(ProtocolError::ArtifactHandleLimit {
                limit: self.config.limits.max_artifact_handles,
            });
        }
        self.active_upload_handles.remove(upload);
        self.active_download_handles.insert(download.to_owned());
        Ok(())
    }
}

impl AdapterProtocol for StrictAdapterProtocol {
    fn encode_host_frame(&self, message: &ValidatedHostMessage) -> Result<Vec<u8>, ProtocolError> {
        let mut frame = serde_json::to_vec(message.envelope())
            .map_err(|error| ProtocolError::InvalidJson(error.to_string()))?;
        frame.push(b'\n');
        if frame.len() > self.config.limits.max_frame_bytes {
            return Err(ProtocolError::FrameTooLarge {
                limit: self.config.limits.max_frame_bytes,
                actual: frame.len(),
            });
        }
        Ok(frame)
    }

    fn decode_adapter_frame(&self, frame: &[u8]) -> Result<AdapterEnvelope, ProtocolError> {
        if frame.len() > self.config.limits.max_frame_bytes {
            return Err(ProtocolError::FrameTooLarge {
                limit: self.config.limits.max_frame_bytes,
                actual: frame.len(),
            });
        }
        let Some(body) = frame.strip_suffix(b"\n") else {
            return Err(ProtocolError::MissingFrameTerminator);
        };
        if body.contains(&b'\n') {
            return Err(ProtocolError::MultipleFrames);
        }
        serde_json::from_slice(body).map_err(|error| ProtocolError::InvalidJson(error.to_string()))
    }

    fn accept_host(
        &mut self,
        envelope: HostEnvelope,
    ) -> Result<ValidatedHostMessage, ProtocolError> {
        self.accept_host_inner(&envelope)?;
        Ok(ValidatedHostMessage(envelope))
    }

    fn accept_adapter(
        &mut self,
        envelope: AdapterEnvelope,
    ) -> Result<ValidatedAdapterMessage, ProtocolError> {
        match self.accept_adapter_inner(&envelope) {
            Ok(()) => Ok(ValidatedAdapterMessage(envelope)),
            Err(error) => {
                self.session = ProtocolSessionState::Failed;
                Err(error)
            }
        }
    }

    fn accept_adapter_frame(
        &mut self,
        frame: &[u8],
    ) -> Result<ValidatedAdapterMessage, ProtocolError> {
        let envelope = match self.decode_adapter_frame(frame) {
            Ok(envelope) => envelope,
            Err(error) => {
                self.session = ProtocolSessionState::Failed;
                return Err(error);
            }
        };
        self.accept_adapter(envelope)
    }

    fn release_download_handle(
        &mut self,
        download: &ArtifactDownloadHandle,
    ) -> Result<(), ProtocolError> {
        if self.active_download_handles.remove(download.as_str()) {
            Ok(())
        } else {
            Err(ProtocolError::UnknownArtifactDownloadHandle(
                download.as_str().to_owned(),
            ))
        }
    }

    fn session_state(&self) -> ProtocolSessionState {
        self.session
    }

    fn operation_state(&self, operation: &str) -> Option<ProtocolOperationState> {
        self.operations
            .get(operation)
            .map(|record| operation_disposition(&record.state))
    }
}

/// NativeGraph protocol admission failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    /// A frame exceeded its byte cap before it could be parsed.
    FrameTooLarge { limit: usize, actual: usize },
    /// A JSONL frame lacked the required trailing newline.
    MissingFrameTerminator,
    /// A byte sequence attempted to carry more than one frame.
    MultipleFrames,
    /// Serde rejected the strict envelope or message shape.
    InvalidJson(String),
    /// The peer selected an unsupported version.
    UnsupportedVersion(u16),
    /// The envelope carried an unexpected episode correlation.
    EpisodeMismatch { expected: String, actual: String },
    /// A continuation named a causal span other than its operation's span.
    SpanMismatch {
        operation: String,
        expected: String,
        actual: String,
    },
    /// A typed identifier was empty or exceeded its byte bound.
    InvalidIdentifier { field: &'static str },
    /// A sequence was not exactly the next expected directional value.
    SequenceOutOfOrder { expected: u64, actual: u64 },
    /// A directional sequence cannot advance without wrapping.
    SequenceExhausted,
    /// The session could not admit a message in its current state.
    SessionState(ProtocolSessionState),
    /// The message declared a role other than the Rust-pinned role.
    RoleMismatch {
        expected: AdapterRole,
        actual: AdapterRole,
    },
    /// A role cannot emit or receive the attempted message.
    MessageForbiddenForRole(AdapterRole),
    /// The selected capabilities omitted a message's required capability.
    CapabilityNotDeclared(ProtocolCapability),
    /// Ready failed to acknowledge the exact host-selected capability set.
    ReadyCapabilitiesMismatch,
    /// A correlation was already recorded and cannot be recycled.
    OperationAlreadyUsed(String),
    /// A response named no recorded operation.
    UnknownOperation(String),
    /// The permanent operation ledger reached its bounded maximum.
    OperationLedgerLimit { limit: usize },
    /// A message did not match its operation's required transition state.
    OperationState {
        operation: String,
        state: ProtocolOperationState,
    },
    /// A model binding is not in the Rust-pinned allowlist.
    ModelBindingNotAllowed(String),
    /// A model-call correlation was reused within one pending decision lineage.
    ModelCallAlreadyUsed(String),
    /// A pending decision lineage reached its configured model-call bound.
    ModelCallLimit { limit: usize },
    /// All pending decision lineages reached their configured model-call entry bound.
    ModelCallSessionEntryLimit { limit: usize },
    /// All pending decision lineages reached their configured model-call byte bound.
    ModelCallSessionByteLimit { limit: usize },
    /// A state transition named no active model-call correlation in its lineage.
    ModelCallNotActive(String),
    /// Internal model-call lineage accounting was inconsistent.
    ModelCallLineageAccounting,
    /// Rust delivered a result for a different accepted model call.
    ModelCallMismatch { expected: String, actual: String },
    /// A cancel target was absent, terminal, or self-referential.
    CancelTargetInvalid(String),
    /// A cancel acknowledgement named a different target.
    CancelTargetMismatch { expected: String, actual: String },
    /// Shutdown was attempted while an operation remained live.
    ShutdownWithLiveOperations,
    /// An artifact operation exceeded its byte cap.
    ArtifactLengthTooLarge { limit: u64, actual: u64 },
    /// An artifact continuation changed its original length.
    ArtifactLengthMismatch { expected: u64, actual: u64 },
    /// An artifact upload chunk was empty and could not advance the bounded stream.
    ArtifactUploadChunkEmpty,
    /// An artifact commit changed its originally granted upload capability.
    ArtifactUploadMismatch { expected: String, actual: String },
    /// An adapter artifact request did not name a live Rust-origin parent operation.
    ArtifactParentInvalid(String),
    /// An artifact handle was granted twice while still active.
    ArtifactHandleAlreadyActive(String),
    /// An artifact commit named no active upload capability.
    UnknownArtifactUploadHandle(String),
    /// Rust attempted to release no active download capability.
    UnknownArtifactDownloadHandle(String),
    /// Active artifact references exceeded their bound.
    ArtifactHandleLimit { limit: usize },
    /// An arbitrary JSON payload exceeded a byte cap.
    JsonTooLarge { limit: usize, actual: usize },
    /// An arbitrary JSON payload exceeded a nesting cap.
    JsonTooDeep { limit: usize, actual: usize },
    /// An arbitrary JSON array exceeded its entry cap.
    JsonArrayTooLarge { limit: usize, actual: usize },
    /// An arbitrary JSON object exceeded its entry cap.
    JsonObjectTooLarge { limit: usize, actual: usize },
    /// An environment transition supplied a nonfinite reward.
    NonFiniteReward,
}

impl Display for ProtocolError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameTooLarge { limit, actual } => {
                write!(formatter, "protocol frame {actual} exceeds limit {limit}")
            }
            Self::MissingFrameTerminator => {
                formatter.write_str("protocol frame lacks JSONL terminator")
            }
            Self::MultipleFrames => {
                formatter.write_str("protocol input contains multiple JSONL frames")
            }
            Self::InvalidJson(error) => write!(formatter, "invalid protocol JSON: {error}"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported protocol version {version}")
            }
            Self::EpisodeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "protocol episode {actual:?} does not match {expected:?}"
                )
            }
            Self::SpanMismatch {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "protocol operation {operation:?} span {actual:?} does not match {expected:?}"
            ),
            Self::InvalidIdentifier { field } => write!(formatter, "invalid protocol {field}"),
            Self::SequenceOutOfOrder { expected, actual } => {
                write!(
                    formatter,
                    "protocol sequence {actual} does not match {expected}"
                )
            }
            Self::SequenceExhausted => formatter.write_str("protocol sequence exhausted"),
            Self::SessionState(state) => write!(formatter, "protocol session is {state:?}"),
            Self::RoleMismatch { expected, actual } => {
                write!(
                    formatter,
                    "adapter role {actual:?} does not match {expected:?}"
                )
            }
            Self::MessageForbiddenForRole(role) => {
                write!(formatter, "message is forbidden for adapter role {role:?}")
            }
            Self::CapabilityNotDeclared(capability) => {
                write!(
                    formatter,
                    "protocol capability {capability:?} was not declared"
                )
            }
            Self::ReadyCapabilitiesMismatch => {
                formatter.write_str("ready capabilities do not match host selection")
            }
            Self::OperationAlreadyUsed(operation) => {
                write!(formatter, "operation {operation:?} was already used")
            }
            Self::UnknownOperation(operation) => {
                write!(formatter, "unknown operation {operation:?}")
            }
            Self::OperationLedgerLimit { limit } => {
                write!(formatter, "operation ledger exceeds limit {limit}")
            }
            Self::OperationState { operation, state } => {
                write!(formatter, "operation {operation:?} is {state:?}")
            }
            Self::ModelBindingNotAllowed(binding) => {
                write!(formatter, "model binding {binding:?} is not allowed")
            }
            Self::ModelCallAlreadyUsed(model_call) => {
                write!(formatter, "model call {model_call:?} was already used")
            }
            Self::ModelCallLimit { limit } => {
                write!(formatter, "model call lineage exceeds limit {limit}")
            }
            Self::ModelCallSessionEntryLimit { limit } => {
                write!(
                    formatter,
                    "session model call lineage exceeds entry limit {limit}"
                )
            }
            Self::ModelCallSessionByteLimit { limit } => {
                write!(
                    formatter,
                    "session model call lineage exceeds byte limit {limit}"
                )
            }
            Self::ModelCallNotActive(model_call) => {
                write!(formatter, "model call {model_call:?} is not active")
            }
            Self::ModelCallLineageAccounting => {
                formatter.write_str("model call lineage accounting is inconsistent")
            }
            Self::ModelCallMismatch { expected, actual } => {
                write!(
                    formatter,
                    "model call {actual:?} does not match {expected:?}"
                )
            }
            Self::CancelTargetInvalid(operation) => {
                write!(formatter, "invalid cancel target {operation:?}")
            }
            Self::CancelTargetMismatch { expected, actual } => {
                write!(
                    formatter,
                    "cancel target {actual:?} does not match {expected:?}"
                )
            }
            Self::ShutdownWithLiveOperations => formatter.write_str("shutdown has live operations"),
            Self::ArtifactLengthTooLarge { limit, actual } => {
                write!(formatter, "artifact length {actual} exceeds limit {limit}")
            }
            Self::ArtifactLengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "artifact length {actual} does not match {expected}"
                )
            }
            Self::ArtifactUploadChunkEmpty => {
                formatter.write_str("artifact upload chunk must not be empty")
            }
            Self::ArtifactUploadMismatch { expected, actual } => {
                write!(
                    formatter,
                    "artifact upload {actual:?} does not match {expected:?}"
                )
            }
            Self::ArtifactParentInvalid(operation) => {
                write!(formatter, "invalid artifact parent operation {operation:?}")
            }
            Self::ArtifactHandleAlreadyActive(handle) => {
                write!(formatter, "artifact handle {handle:?} is active")
            }
            Self::UnknownArtifactUploadHandle(handle) => {
                write!(
                    formatter,
                    "unknown active artifact upload handle {handle:?}"
                )
            }
            Self::UnknownArtifactDownloadHandle(handle) => {
                write!(
                    formatter,
                    "unknown active artifact download handle {handle:?}"
                )
            }
            Self::ArtifactHandleLimit { limit } => {
                write!(formatter, "artifact handle limit {limit} exceeded")
            }
            Self::JsonTooLarge { limit, actual } => {
                write!(formatter, "JSON bytes {actual} exceed limit {limit}")
            }
            Self::JsonTooDeep { limit, actual } => {
                write!(formatter, "JSON depth {actual} exceeds limit {limit}")
            }
            Self::JsonArrayTooLarge { limit, actual } => {
                write!(formatter, "JSON array {actual} exceeds limit {limit}")
            }
            Self::JsonObjectTooLarge { limit, actual } => {
                write!(formatter, "JSON object {actual} exceeds limit {limit}")
            }
            Self::NonFiniteReward => formatter.write_str("transition reward must be finite"),
        }
    }
}

impl std::error::Error for ProtocolError {}

fn validate_limits(limits: &ProtocolLimits) -> Result<(), ProtocolError> {
    if limits.max_frame_bytes == 0
        || limits.max_identifier_bytes == 0
        || limits.max_json_bytes == 0
        || limits.max_json_depth == 0
        || limits.max_json_array_entries == 0
        || limits.max_json_object_entries == 0
        || limits.max_operation_ledger_entries == 0
        || limits.max_model_call_lineage_entries == 0
        || limits.max_session_model_call_lineage_entries == 0
        || limits.max_session_model_call_lineage_bytes == 0
        || limits.max_artifact_handles == 0
        || limits.max_artifact_bytes == 0
    {
        return Err(ProtocolError::InvalidIdentifier { field: "limits" });
    }
    Ok(())
}

fn role_capability(role: AdapterRole) -> ProtocolCapability {
    match role {
        AdapterRole::Tool => ProtocolCapability::Tool,
        AdapterRole::Policy => ProtocolCapability::Policy,
        AdapterRole::Environment => ProtocolCapability::Environment,
        AdapterRole::Heuristic => ProtocolCapability::Heuristic,
        AdapterRole::Driver => ProtocolCapability::Driver,
    }
}

fn capability_set(
    capabilities: Vec<ProtocolCapability>,
) -> Result<BTreeSet<ProtocolCapability>, ProtocolError> {
    let count = capabilities.len();
    let capabilities = capabilities.into_iter().collect::<BTreeSet<_>>();
    if capabilities.len() != count {
        return Err(ProtocolError::ReadyCapabilitiesMismatch);
    }
    Ok(capabilities)
}

fn validate_identifier(
    value: &str,
    limits: &ProtocolLimits,
    field: &'static str,
) -> Result<(), ProtocolError> {
    if value.is_empty() || value.len() > limits.max_identifier_bytes {
        Err(ProtocolError::InvalidIdentifier { field })
    } else {
        Ok(())
    }
}

fn validate_json(value: &Value, limits: &ProtocolLimits) -> Result<(), ProtocolError> {
    let bytes =
        serde_json::to_vec(value).map_err(|error| ProtocolError::InvalidJson(error.to_string()))?;
    if bytes.len() > limits.max_json_bytes {
        return Err(ProtocolError::JsonTooLarge {
            limit: limits.max_json_bytes,
            actual: bytes.len(),
        });
    }
    validate_json_shape(value, limits, 1)
}

fn validate_json_shape(
    value: &Value,
    limits: &ProtocolLimits,
    depth: usize,
) -> Result<(), ProtocolError> {
    if depth > limits.max_json_depth {
        return Err(ProtocolError::JsonTooDeep {
            limit: limits.max_json_depth,
            actual: depth,
        });
    }
    match value {
        Value::Array(values) => {
            if values.len() > limits.max_json_array_entries {
                return Err(ProtocolError::JsonArrayTooLarge {
                    limit: limits.max_json_array_entries,
                    actual: values.len(),
                });
            }
            for value in values {
                validate_json_shape(value, limits, depth.saturating_add(1))?;
            }
        }
        Value::Object(values) => {
            if values.len() > limits.max_json_object_entries {
                return Err(ProtocolError::JsonObjectTooLarge {
                    limit: limits.max_json_object_entries,
                    actual: values.len(),
                });
            }
            for value in values.values() {
                validate_json_shape(value, limits, depth.saturating_add(1))?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
    Ok(())
}

fn is_pending(state: &OperationState) -> bool {
    !matches!(state, OperationState::Closed | OperationState::Failed)
}

fn is_cancellable(state: &OperationState) -> bool {
    is_pending(state)
        && !matches!(
            state,
            OperationState::Cancelling | OperationState::AwaitCancelAck { .. }
        )
}

fn is_checkpointable(state: &OperationState) -> bool {
    matches!(
        state,
        OperationState::AwaitToolResult
            | OperationState::AwaitDecision
            | OperationState::AwaitModelResult { .. }
            | OperationState::AwaitEnvironmentReset
            | OperationState::AwaitTransition
            | OperationState::AwaitTerminalCandidate
    )
}

fn operation_disposition(state: &OperationState) -> ProtocolOperationState {
    match state {
        OperationState::Cancelling | OperationState::AwaitCancelAck { .. } => {
            ProtocolOperationState::Cancelling
        }
        OperationState::Closed => ProtocolOperationState::Closed,
        OperationState::Failed => ProtocolOperationState::Failed,
        OperationState::AwaitReady { .. }
        | OperationState::AwaitResetAck
        | OperationState::AwaitToolResult
        | OperationState::AwaitDecision
        | OperationState::AwaitModelResult { .. }
        | OperationState::AwaitEnvironmentReset
        | OperationState::AwaitTransition
        | OperationState::AwaitTerminalCandidate
        | OperationState::AwaitArtifactGrant { .. }
        | OperationState::AwaitArtifactCommit { .. }
        | OperationState::AwaitDownloadGrant
        | OperationState::AwaitShutdownAck => ProtocolOperationState::Pending,
    }
}
