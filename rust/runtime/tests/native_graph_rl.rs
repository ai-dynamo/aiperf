// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable NativeGraph RL rollout and environment-stepper contracts.

use std::{
    cell::RefCell,
    collections::{BTreeSet, VecDeque},
    io::Cursor,
    rc::Rc,
    time::Duration,
};

use aiperf_runtime::{
    eval::{
        AdapterEnvelope, AdapterExit, AdapterLifecycleDeadlines, AdapterMessage, AdapterProcess,
        AdapterProtocolConfig, AdapterRole, AdapterRuntimeFactory, AdapterSpawnRequest,
        AdapterSpawnTransaction, AdapterSpawner, AdapterSupervisionError, ArtifactDigest,
        ArtifactDownloadHandle, ArtifactError, ArtifactQuota, AttemptId, CancelReason,
        EnvironmentTransitionRecord, EpisodeArtifactStore, EvidenceKind, FrozenArtifact,
        FrozenArtifactReference, FrozenRolloutEvidence, HostEnvelope, HostMessage,
        ProtocolAdapterRuntimeFactory, ProtocolCapability, ProtocolError, ProtocolLimits,
        RlEvaluationLimits, RlEvaluationPolicy, RlRolloutError, RolloutAdmissionError,
        RolloutEvidenceError, RolloutEvidenceIdentity, RolloutEvidenceLimits,
        RolloutReturnAgreementError, RolloutVerifierDecodeError, RolloutVerifierInput,
        StrictAdapterProtocolFactory, SupervisedAdapter,
    },
    graph::tools::{
        EnvironmentArtifactBindings, EnvironmentEpisodeIdentity, EnvironmentResetRequest,
        EnvironmentSessionAuthority, EnvironmentStepRequest, EnvironmentStepperBinding,
        EnvironmentStepperError, EnvironmentStepperFactory, SupervisedEnvironmentStepperFactory,
    },
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;
use tokio::sync::Notify;

enum FakeReapBehavior {
    Complete,
    Fail,
    Block {
        started: Rc<Notify>,
        release: Rc<Notify>,
    },
}

#[derive(Default)]
struct FakeEnvironmentAdapterState {
    received: VecDeque<AdapterEnvelope>,
    sent: Vec<HostEnvelope>,
    starts: usize,
    reap_reasons: Vec<CancelReason>,
    successful_reaps: usize,
    reap_behaviors: VecDeque<FakeReapBehavior>,
    send_failures: VecDeque<AdapterSupervisionError>,
    receive_failures: VecDeque<AdapterSupervisionError>,
}

struct FakeEnvironmentAdapter {
    state: Rc<RefCell<FakeEnvironmentAdapterState>>,
}

struct FakeEnvironmentRuntime {
    config: AdapterProtocolConfig,
    state: Rc<RefCell<FakeEnvironmentAdapterState>>,
}

struct StoreBackedEnvironmentState {
    plans: VecDeque<StoreBackedEnvironmentPlan>,
    active: Option<StoreBackedEnvironmentPlan>,
    pending_bytes: Option<Vec<u8>>,
    upload: Option<aiperf_runtime::eval::ArtifactUploadHandle>,
    references: Vec<FrozenArtifactReference>,
    stage: StoreBackedEnvironmentStage,
    next_adapter_sequence: u64,
    is_tampered_output: bool,
    sent: Vec<HostEnvelope>,
    starts: usize,
    reap_reasons: Vec<CancelReason>,
    successful_reaps: usize,
    reap_behaviors: VecDeque<FakeReapBehavior>,
    send_failures: VecDeque<AdapterSupervisionError>,
    receive_failures: VecDeque<AdapterSupervisionError>,
}

struct StoreBackedEnvironmentPlan {
    operation: String,
    response_operation: String,
    response: StoreBackedEnvironmentResponse,
    outputs: VecDeque<Vec<u8>>,
}

enum StoreBackedEnvironmentResponse {
    Reset,
    Transition {
        reward: f64,
        terminated: bool,
        truncated: bool,
    },
}

enum StoreBackedEnvironmentStage {
    AwaitRoot,
    RequestUpload,
    UploadChunk,
    CompleteUpload,
    AwaitCommit,
    ReturnResponse,
    Done,
}

struct StoreBackedEnvironmentAdapter {
    state: Rc<RefCell<StoreBackedEnvironmentState>>,
}

struct StoreBackedEnvironmentRuntime {
    config: AdapterProtocolConfig,
    state: Rc<RefCell<StoreBackedEnvironmentState>>,
}

/// JSONL child fixture used to prove dynamic output bytes cross the strict adapter boundary.
struct StrictUploadChild {
    state: Rc<RefCell<StrictUploadChildState>>,
}

struct StrictUploadChildState {
    stdout: VecDeque<Vec<u8>>,
    reaps: usize,
    upload_fragment: StrictUploadFragment,
    mode: StrictUploadMode,
    next_adapter_sequence: u64,
    active_parent: Option<String>,
    queued_outputs: VecDeque<Vec<u8>>,
    pending_operation: Option<String>,
    pending_bytes: Option<Vec<u8>>,
    response_references: Vec<FrozenArtifactReference>,
    committed_references: Vec<FrozenArtifactReference>,
}

impl Default for StrictUploadChildState {
    fn default() -> Self {
        Self {
            stdout: VecDeque::new(),
            reaps: 0,
            upload_fragment: StrictUploadFragment::Valid,
            mode: StrictUploadMode::ResetOnly,
            next_adapter_sequence: 0,
            active_parent: None,
            queued_outputs: VecDeque::new(),
            pending_operation: None,
            pending_bytes: None,
            response_references: Vec::new(),
            committed_references: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StrictUploadMode {
    ResetOnly,
    ResetThenTerminalStep,
}

#[derive(Clone, Copy)]
enum StrictUploadFragment {
    Valid,
    MalformedBase64,
    WrongCapability,
    Overflow,
    OversizedFrame,
}

struct StrictUploadSpawner {
    state: Rc<RefCell<StrictUploadChildState>>,
}

struct StrictUploadTransaction {
    child: Option<Box<dyn AdapterProcess>>,
}

impl AdapterSpawner for StrictUploadSpawner {
    fn begin_spawn(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        Ok(Box::new(StrictUploadTransaction {
            child: Some(Box::new(StrictUploadChild {
                state: Rc::clone(&self.state),
            })),
        }))
    }
}

#[async_trait(?Send)]
impl AdapterSpawnTransaction for StrictUploadTransaction {
    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
        self.child
            .take()
            .ok_or(AdapterSupervisionError::AlreadyReaped)
    }

    async fn abort(&mut self, _: Duration) -> Result<(), AdapterSupervisionError> {
        self.child.take();
        Ok(())
    }

    fn fence(&mut self) {}
}

#[async_trait(?Send)]
impl AdapterProcess for StrictUploadChild {
    async fn write_frame(
        &mut self,
        frame: &[u8],
        _: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let host: HostEnvelope = serde_json::from_slice(frame).map_err(|error| {
            AdapterSupervisionError::Process(format!("fixture host frame is invalid: {error}"))
        })?;
        let mut state = self.state.borrow_mut();
        match host.message {
            HostMessage::Hello { .. } => {
                let ready = strict_upload_next(
                    &mut state,
                    "startup",
                    "hello",
                    AdapterMessage::Ready {
                        protocol_version: 1,
                        capabilities: vec![
                            ProtocolCapability::Environment,
                            ProtocolCapability::Artifacts,
                        ],
                        implementation_digest: ArtifactDigest::from_bytes(b"strict-output-child"),
                    },
                );
                strict_upload_push(&mut state, ready)?;
            }
            HostMessage::ResetEnvironment { .. } => {
                strict_upload_begin_parent(
                    &mut state,
                    host.operation,
                    [b"strict-observation".to_vec()],
                )?;
            }
            HostMessage::StepEnvironment { .. }
                if state.mode == StrictUploadMode::ResetThenTerminalStep =>
            {
                strict_upload_begin_parent(
                    &mut state,
                    host.operation,
                    [
                        b"strict-transition-observation".to_vec(),
                        b"strict-transition-info".to_vec(),
                    ],
                )?;
            }
            HostMessage::PutArtifactHandle { upload, .. } => {
                if state.pending_operation.as_deref() != Some(host.operation.as_str()) {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                let (chunk_upload, bytes_base64) = match state.upload_fragment {
                    StrictUploadFragment::Valid => (
                        serde_json::to_value(&upload).expect("fixture upload serializes"),
                        STANDARD.encode(
                            state
                                .pending_bytes
                                .as_deref()
                                .ok_or(AdapterSupervisionError::EndOfStream)?,
                        ),
                    ),
                    StrictUploadFragment::MalformedBase64 => (
                        serde_json::to_value(&upload).expect("fixture upload serializes"),
                        "not valid base64!".to_owned(),
                    ),
                    StrictUploadFragment::WrongCapability => (
                        json!("forged-upload-capability"),
                        STANDARD.encode(b"strict-observation"),
                    ),
                    StrictUploadFragment::Overflow => (
                        serde_json::to_value(&upload).expect("fixture upload serializes"),
                        STANDARD.encode(b"strict-observation!"),
                    ),
                    StrictUploadFragment::OversizedFrame => (
                        serde_json::to_value(&upload).expect("fixture upload serializes"),
                        STANDARD.encode(vec![0_u8; 192 * 1024]),
                    ),
                };
                let sequence = state.next_adapter_sequence;
                state.next_adapter_sequence += 1;
                strict_upload_push_raw(
                    &mut state,
                    json!({
                        "version": 1,
                        "episode": "episode-1",
                        "span": "root",
                        "sequence": sequence,
                        "operation": host.operation,
                        "message": {
                            "type": "artifact_upload_chunk",
                            "upload": chunk_upload,
                            "bytes_base64": bytes_base64,
                        },
                    }),
                )?;
                let complete = strict_upload_next(
                    &mut state,
                    "root",
                    host.operation,
                    AdapterMessage::ArtifactUploadComplete { upload },
                );
                strict_upload_push(&mut state, complete)?;
            }
            HostMessage::ArtifactCommitted { reference, .. } => {
                if state.pending_operation.as_deref() != Some(host.operation.as_str()) {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                state.pending_operation = None;
                state.pending_bytes = None;
                state.response_references.push(reference.clone());
                state.committed_references.push(reference);
                if state.queued_outputs.is_empty() {
                    strict_upload_finish_parent(&mut state)?;
                } else {
                    strict_upload_request_output(&mut state)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    async fn read_stdout_frame(
        &mut self,
        _: usize,
        _: Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        self.state
            .borrow_mut()
            .stdout
            .pop_front()
            .ok_or(AdapterSupervisionError::EndOfStream)
    }

    async fn drain_stderr(&mut self, _: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        Ok(Vec::new())
    }

    async fn cancel(
        &mut self,
        _: CancelReason,
        _: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn reap(&mut self, _: Duration) -> Result<AdapterExit, AdapterSupervisionError> {
        self.state.borrow_mut().reaps += 1;
        Ok(AdapterExit::Reaped)
    }

    fn fence(&mut self) {}
}

fn strict_upload_next(
    state: &mut StrictUploadChildState,
    span: impl Into<String>,
    operation: impl Into<String>,
    message: AdapterMessage,
) -> AdapterEnvelope {
    let envelope = AdapterEnvelope::new(
        "episode-1",
        span,
        state.next_adapter_sequence,
        operation,
        message,
    );
    state.next_adapter_sequence += 1;
    envelope
}

fn strict_upload_begin_parent(
    state: &mut StrictUploadChildState,
    parent: String,
    outputs: impl IntoIterator<Item = Vec<u8>>,
) -> Result<(), AdapterSupervisionError> {
    if state.active_parent.is_some() {
        return Err(AdapterSupervisionError::InvalidResetTransition);
    }
    state.active_parent = Some(parent);
    state.queued_outputs = outputs.into_iter().collect();
    state.response_references.clear();
    strict_upload_request_output(state)
}

fn strict_upload_request_output(
    state: &mut StrictUploadChildState,
) -> Result<(), AdapterSupervisionError> {
    let parent = state
        .active_parent
        .clone()
        .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
    let bytes = state
        .queued_outputs
        .pop_front()
        .ok_or(AdapterSupervisionError::EndOfStream)?;
    let operation = format!("{parent}-output-{}", state.response_references.len());
    let declared_bytes = u64::try_from(bytes.len())
        .map_err(|_| AdapterSupervisionError::Process("fixture output is too large".to_owned()))?;
    state.pending_operation = Some(operation.clone());
    state.pending_bytes = Some(bytes);
    let request = strict_upload_next(
        state,
        "root",
        operation,
        AdapterMessage::PutArtifactRequest {
            parent_operation: parent,
            declared_bytes,
        },
    );
    strict_upload_push(state, request)
}

fn strict_upload_finish_parent(
    state: &mut StrictUploadChildState,
) -> Result<(), AdapterSupervisionError> {
    let parent = state
        .active_parent
        .take()
        .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
    let references = std::mem::take(&mut state.response_references);
    let message = if parent == "reset-1" {
        let [observation]: [FrozenArtifactReference; 1] = references
            .try_into()
            .map_err(|_| AdapterSupervisionError::InvalidResetTransition)?;
        AdapterMessage::EnvironmentReset {
            observation_ref: observation,
        }
    } else if parent == "step-1" && state.mode == StrictUploadMode::ResetThenTerminalStep {
        let [observation, info]: [FrozenArtifactReference; 2] = references
            .try_into()
            .map_err(|_| AdapterSupervisionError::InvalidResetTransition)?;
        AdapterMessage::Transition {
            observation_ref: observation,
            reward: 1.0,
            terminated: true,
            truncated: false,
            info_ref: info,
        }
    } else {
        return Err(AdapterSupervisionError::InvalidResetTransition);
    };
    let response = strict_upload_next(state, "root", parent, message);
    strict_upload_push(state, response)
}

fn strict_upload_push(
    state: &mut StrictUploadChildState,
    message: AdapterEnvelope,
) -> Result<(), AdapterSupervisionError> {
    strict_upload_push_raw(
        state,
        serde_json::to_value(message).map_err(|error| {
            AdapterSupervisionError::Process(format!("fixture adapter frame is invalid: {error}"))
        })?,
    )
}

fn strict_upload_push_raw(
    state: &mut StrictUploadChildState,
    message: serde_json::Value,
) -> Result<(), AdapterSupervisionError> {
    let mut frame = serde_json::to_vec(&message).map_err(|error| {
        AdapterSupervisionError::Process(format!("fixture adapter frame is invalid: {error}"))
    })?;
    frame.push(b'\n');
    state.stdout.push_back(frame);
    Ok(())
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for StoreBackedEnvironmentRuntime {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        self.state.borrow_mut().starts += 1;
        Ok(Box::new(StoreBackedEnvironmentAdapter {
            state: Rc::clone(&self.state),
        }))
    }
}

#[async_trait(?Send)]
impl SupervisedAdapter for StoreBackedEnvironmentAdapter {
    async fn send(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.send_failures.pop_front() {
            return Err(error);
        }
        match &message.message {
            HostMessage::ResetEnvironment { .. } | HostMessage::StepEnvironment { .. } => {
                if !matches!(state.stage, StoreBackedEnvironmentStage::AwaitRoot) {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                let plan = state
                    .plans
                    .pop_front()
                    .ok_or(AdapterSupervisionError::EndOfStream)?;
                if plan.operation != message.operation {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                state.active = Some(plan);
                state.stage = StoreBackedEnvironmentStage::RequestUpload;
            }
            HostMessage::PutArtifactHandle { upload, .. } => {
                state.upload = Some(upload.clone());
                state.stage = StoreBackedEnvironmentStage::UploadChunk;
            }
            HostMessage::ArtifactCommitted { upload, reference } => {
                assert_eq!(state.upload.as_ref(), Some(upload));
                state.references.push(reference.clone());
                state.upload = None;
                state.pending_bytes = None;
                let active = state
                    .active
                    .as_ref()
                    .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
                state.stage = if active.outputs.is_empty() {
                    StoreBackedEnvironmentStage::ReturnResponse
                } else {
                    StoreBackedEnvironmentStage::RequestUpload
                };
            }
            _ => {}
        }
        state.sent.push(message);
        Ok(())
    }

    async fn receive(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.receive_failures.pop_front() {
            return Err(error);
        }
        match state.stage {
            StoreBackedEnvironmentStage::AwaitRoot | StoreBackedEnvironmentStage::AwaitCommit => {
                Err(AdapterSupervisionError::InvalidResetTransition)
            }
            StoreBackedEnvironmentStage::RequestUpload => {
                let output_index = state.references.len();
                let (parent_operation, output_index, bytes) = {
                    let active = state
                        .active
                        .as_mut()
                        .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
                    let bytes = active
                        .outputs
                        .pop_front()
                        .ok_or(AdapterSupervisionError::EndOfStream)?;
                    (active.operation.clone(), output_index, bytes)
                };
                let declared_bytes = bytes.len() as u64;
                state.pending_bytes = Some(bytes);
                state.stage = StoreBackedEnvironmentStage::CompleteUpload;
                let envelope = AdapterEnvelope::new(
                    "episode-1",
                    "root",
                    state.next_adapter_sequence,
                    format!("{parent_operation}-output-{output_index}"),
                    AdapterMessage::PutArtifactRequest {
                        parent_operation,
                        declared_bytes,
                    },
                );
                state.next_adapter_sequence += 1;
                Ok(envelope)
            }
            StoreBackedEnvironmentStage::UploadChunk => {
                state.stage = StoreBackedEnvironmentStage::CompleteUpload;
                let upload = state
                    .upload
                    .clone()
                    .ok_or(AdapterSupervisionError::EndOfStream)?;
                let bytes = state
                    .pending_bytes
                    .as_deref()
                    .ok_or(AdapterSupervisionError::EndOfStream)?;
                let operation = format!(
                    "{}-output-{}",
                    state
                        .active
                        .as_ref()
                        .ok_or(AdapterSupervisionError::InvalidResetTransition)?
                        .operation,
                    state.references.len()
                );
                let envelope = AdapterEnvelope::new(
                    "episode-1",
                    "root",
                    state.next_adapter_sequence,
                    operation,
                    AdapterMessage::ArtifactUploadChunk {
                        upload,
                        bytes_base64: STANDARD.encode(bytes),
                    },
                );
                state.next_adapter_sequence += 1;
                Ok(envelope)
            }
            StoreBackedEnvironmentStage::CompleteUpload => {
                state.stage = StoreBackedEnvironmentStage::AwaitCommit;
                let upload = state
                    .upload
                    .clone()
                    .ok_or(AdapterSupervisionError::EndOfStream)?;
                let operation = format!(
                    "{}-output-{}",
                    state
                        .active
                        .as_ref()
                        .ok_or(AdapterSupervisionError::InvalidResetTransition)?
                        .operation,
                    state.references.len()
                );
                let envelope = AdapterEnvelope::new(
                    "episode-1",
                    "root",
                    state.next_adapter_sequence,
                    operation,
                    AdapterMessage::ArtifactUploadComplete { upload },
                );
                state.next_adapter_sequence += 1;
                Ok(envelope)
            }
            StoreBackedEnvironmentStage::ReturnResponse => {
                let plan = state
                    .active
                    .take()
                    .ok_or(AdapterSupervisionError::InvalidResetTransition)?;
                let mut references = std::mem::take(&mut state.references).into_iter();
                let first = references
                    .next()
                    .ok_or(AdapterSupervisionError::EndOfStream)?;
                let first = if state.is_tampered_output {
                    tamper_reference(&first)
                } else {
                    first
                };
                let message = match plan.response {
                    StoreBackedEnvironmentResponse::Reset => {
                        if references.next().is_some() {
                            return Err(AdapterSupervisionError::InvalidResetTransition);
                        }
                        AdapterMessage::EnvironmentReset {
                            observation_ref: first,
                        }
                    }
                    StoreBackedEnvironmentResponse::Transition {
                        reward,
                        terminated,
                        truncated,
                    } => AdapterMessage::Transition {
                        observation_ref: first,
                        reward,
                        terminated,
                        truncated,
                        info_ref: references
                            .next()
                            .ok_or(AdapterSupervisionError::EndOfStream)?,
                    },
                };
                if references.next().is_some() {
                    return Err(AdapterSupervisionError::InvalidResetTransition);
                }
                state.stage = if state.plans.is_empty() {
                    StoreBackedEnvironmentStage::Done
                } else {
                    StoreBackedEnvironmentStage::AwaitRoot
                };
                let envelope = AdapterEnvelope::new(
                    "episode-1",
                    "root",
                    state.next_adapter_sequence,
                    plan.response_operation,
                    message,
                );
                state.next_adapter_sequence += 1;
                Ok(envelope)
            }
            StoreBackedEnvironmentStage::Done => Err(AdapterSupervisionError::EndOfStream),
        }
    }

    async fn receive_heartbeat(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn receive_idle(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn reset(&mut self, _: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        Err(AdapterSupervisionError::InvalidResetTransition)
    }

    fn release_download_handle(
        &mut self,
        _: &ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn cancel_and_reap(
        &mut self,
        reason: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        let behavior = {
            let mut state = self.state.borrow_mut();
            state.reap_reasons.push(reason);
            state
                .reap_behaviors
                .pop_front()
                .unwrap_or(FakeReapBehavior::Complete)
        };
        match behavior {
            FakeReapBehavior::Complete => {
                self.state.borrow_mut().successful_reaps += 1;
                Ok(AdapterExit::Reaped)
            }
            FakeReapBehavior::Fail => Err(AdapterSupervisionError::Process(
                "fixture reaper failure".to_owned(),
            )),
            FakeReapBehavior::Block { started, release } => {
                started.notify_one();
                release.notified().await;
                self.state.borrow_mut().successful_reaps += 1;
                Ok(AdapterExit::Reaped)
            }
        }
    }
}

#[async_trait(?Send)]
impl AdapterRuntimeFactory for FakeEnvironmentRuntime {
    fn protocol_config(&self) -> Option<&AdapterProtocolConfig> {
        Some(&self.config)
    }

    async fn start(
        &self,
        _: AdapterSpawnRequest,
    ) -> Result<Box<dyn SupervisedAdapter>, AdapterSupervisionError> {
        self.state.borrow_mut().starts += 1;
        Ok(Box::new(FakeEnvironmentAdapter {
            state: Rc::clone(&self.state),
        }))
    }
}

#[async_trait(?Send)]
impl SupervisedAdapter for FakeEnvironmentAdapter {
    async fn send(&mut self, message: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        if let Some(error) = self.state.borrow_mut().send_failures.pop_front() {
            return Err(error);
        }
        self.state.borrow_mut().sent.push(message);
        Ok(())
    }

    async fn receive(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        if let Some(error) = self.state.borrow_mut().receive_failures.pop_front() {
            return Err(error);
        }
        self.state
            .borrow_mut()
            .received
            .pop_front()
            .ok_or(AdapterSupervisionError::EndOfStream)
    }

    async fn receive_heartbeat(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn receive_idle(&mut self) -> Result<AdapterEnvelope, AdapterSupervisionError> {
        self.receive().await
    }

    async fn reset(&mut self, _: HostEnvelope) -> Result<(), AdapterSupervisionError> {
        Err(AdapterSupervisionError::InvalidResetTransition)
    }

    fn release_download_handle(
        &mut self,
        _: &ArtifactDownloadHandle,
    ) -> Result<(), AdapterSupervisionError> {
        Ok(())
    }

    async fn cancel_and_reap(
        &mut self,
        reason: CancelReason,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        let behavior = {
            let mut state = self.state.borrow_mut();
            state.reap_reasons.push(reason);
            state
                .reap_behaviors
                .pop_front()
                .unwrap_or(FakeReapBehavior::Complete)
        };
        match behavior {
            FakeReapBehavior::Complete => {
                self.state.borrow_mut().successful_reaps += 1;
                Ok(AdapterExit::Reaped)
            }
            FakeReapBehavior::Fail => Err(AdapterSupervisionError::Process(
                "fixture reaper failure".to_owned(),
            )),
            FakeReapBehavior::Block { started, release } => {
                started.notify_one();
                release.notified().await;
                self.state.borrow_mut().successful_reaps += 1;
                Ok(AdapterExit::Reaped)
            }
        }
    }
}

fn environment_config(role: AdapterRole) -> AdapterProtocolConfig {
    environment_config_for(role, "episode-1")
}

fn environment_config_for(role: AdapterRole, episode: &str) -> AdapterProtocolConfig {
    let capability = match role {
        AdapterRole::Environment => ProtocolCapability::Environment,
        AdapterRole::Tool => ProtocolCapability::Tool,
        AdapterRole::Policy => ProtocolCapability::Policy,
        AdapterRole::Heuristic => ProtocolCapability::Heuristic,
        AdapterRole::Driver => ProtocolCapability::Driver,
    };
    AdapterProtocolConfig::new(
        role,
        episode,
        [capability].into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::new(),
        ProtocolLimits::default(),
    )
    .expect("fixture protocol configuration is valid")
}

fn spawn_request() -> AdapterSpawnRequest {
    AdapterSpawnRequest::for_non_model_adapter(
        ["environment-adapter".to_owned()],
        Default::default(),
        lifecycle_deadlines(Duration::from_secs(7)),
    )
    .expect("fixture spawn request is valid")
}

async fn start_stepper(
    horizon: u32,
    received: impl IntoIterator<Item = AdapterEnvelope>,
) -> (
    Box<dyn aiperf_runtime::graph::tools::EnvironmentStepper>,
    Rc<RefCell<StoreBackedEnvironmentState>>,
    EnvironmentTestInputs,
) {
    start_stepper_with_reap_behaviors(horizon, received, VecDeque::new()).await
}

async fn start_stepper_with_reap_behaviors(
    horizon: u32,
    received: impl IntoIterator<Item = AdapterEnvelope>,
    reap_behaviors: VecDeque<FakeReapBehavior>,
) -> (
    Box<dyn aiperf_runtime::graph::tools::EnvironmentStepper>,
    Rc<RefCell<StoreBackedEnvironmentState>>,
    EnvironmentTestInputs,
) {
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(&std::env::temp_dir(), artifact_quota())
            .expect("fixture artifact store is valid"),
    ));
    let reset = freeze_reference(&mut store.borrow_mut(), b"reset-input");
    let action = freeze_reference(&mut store.borrow_mut(), b"action");
    let inputs = EnvironmentTestInputs { reset, action };
    let plans = received
        .into_iter()
        .enumerate()
        .map(|(index, response)| {
            store_backed_plan_from_response(if index == 0 { "reset-1" } else { "step-1" }, response)
        })
        .collect::<Vec<_>>();
    let state = Rc::new(RefCell::new(store_backed_state(plans)));
    state.borrow_mut().reap_behaviors = reap_behaviors;
    let binding = environment_binding_with_inputs(
        environment_config(AdapterRole::Environment),
        horizon,
        [inputs.reset.clone(), inputs.action.clone()],
    );
    let factory =
        SupervisedEnvironmentStepperFactory::new(Rc::new(StoreBackedEnvironmentRuntime {
            config: environment_config(AdapterRole::Environment),
            state: Rc::clone(&state),
        }));
    let stepper = factory
        .start(
            binding,
            EnvironmentSessionAuthority::new(ArtifactDigest::from_bytes(b"package"), store),
            spawn_request(),
        )
        .await
        .expect("supervised environment stepper binds");
    (stepper, state, inputs)
}

#[derive(Clone)]
struct EnvironmentTestInputs {
    reset: FrozenArtifactReference,
    action: FrozenArtifactReference,
}

fn reset_response() -> AdapterEnvelope {
    AdapterEnvelope::new(
        "episode-1",
        "root",
        1,
        "reset-1",
        AdapterMessage::EnvironmentReset {
            observation_ref: reset_observation_reference(),
        },
    )
}

fn transition_response(operation: &str, terminated: bool, truncated: bool) -> AdapterEnvelope {
    transition_response_with_reward(operation, 1.0, terminated, truncated)
}

fn transition_response_with_reward(
    operation: &str,
    reward: f64,
    terminated: bool,
    truncated: bool,
) -> AdapterEnvelope {
    AdapterEnvelope::new(
        "episode-1",
        "root",
        2,
        operation,
        AdapterMessage::Transition {
            observation_ref: transition_observation_reference(),
            reward,
            terminated,
            truncated,
            info_ref: transition_info_reference(),
        },
    )
}

fn immutable_reference(label: &str) -> FrozenArtifactReference {
    let download: ArtifactDownloadHandle = serde_json::from_value(json!(format!("{label}-handle")))
        .expect("fixture download capability deserializes");
    let frozen: FrozenArtifact = serde_json::from_value(json!({
        "digest": ArtifactDigest::from_bytes(label.as_bytes()),
        "length": label.len(),
    }))
    .expect("fixture frozen artifact deserializes");
    FrozenArtifactReference::new(download, frozen)
}

fn reset_input_reference() -> FrozenArtifactReference {
    immutable_reference("reset-input")
}

fn action_reference() -> FrozenArtifactReference {
    immutable_reference("action")
}

fn reset_observation_reference() -> FrozenArtifactReference {
    immutable_reference("reset-observation")
}

fn transition_observation_reference() -> FrozenArtifactReference {
    immutable_reference("transition-observation")
}

fn transition_info_reference() -> FrozenArtifactReference {
    immutable_reference("transition-info")
}

fn environment_binding(protocol: AdapterProtocolConfig, horizon: u32) -> EnvironmentStepperBinding {
    environment_binding_with_inputs(
        protocol,
        horizon,
        [reset_input_reference(), action_reference()],
    )
}

fn environment_binding_with_inputs(
    protocol: AdapterProtocolConfig,
    horizon: u32,
    inputs: impl IntoIterator<Item = FrozenArtifactReference>,
) -> EnvironmentStepperBinding {
    let episode = protocol.episode().to_owned();
    EnvironmentStepperBinding::new(
        protocol,
        EnvironmentEpisodeIdentity::new(
            ArtifactDigest::from_bytes(b"package"),
            episode,
            Duration::from_secs(7),
        )
        .expect("fixture identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", horizon, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new(inputs).expect("fixture artifact bindings are valid"),
    )
    .expect("environment binding is valid")
}

fn transition_record(
    step: u32,
    reward: f64,
    terminated: bool,
    truncated: bool,
) -> Result<EnvironmentTransitionRecord, RlRolloutError> {
    EnvironmentTransitionRecord::new(
        step,
        transition_observation_reference().artifact().clone(),
        reward,
        terminated,
        truncated,
        transition_info_reference().artifact().clone(),
    )
}

fn lifecycle_deadlines(operation: Duration) -> AdapterLifecycleDeadlines {
    AdapterLifecycleDeadlines::new(
        Duration::from_secs(1),
        Duration::from_secs(2),
        Duration::from_secs(3),
        Duration::from_secs(4),
        operation,
        Duration::from_secs(5),
        Duration::from_secs(6),
    )
    .expect("fixture deadlines are valid")
}

fn artifact_quota() -> ArtifactQuota {
    ArtifactQuota {
        max_artifacts: 8,
        max_total_bytes: 1024,
        max_artifact_bytes: 256,
        max_download_handles: 8,
    }
}

fn session_authority(package: ArtifactDigest) -> EnvironmentSessionAuthority {
    let store = EpisodeArtifactStore::new(&std::env::temp_dir(), artifact_quota())
        .expect("episode artifact store is valid");
    EnvironmentSessionAuthority::new(package, Rc::new(RefCell::new(store)))
}

fn freeze_reference(store: &mut EpisodeArtifactStore, bytes: &[u8]) -> FrozenArtifactReference {
    let upload = store
        .begin_upload(bytes.len() as u64)
        .expect("fixture upload capability is issued");
    store
        .write_upload(&upload, &mut Cursor::new(bytes.to_vec()))
        .expect("fixture bytes are uploaded");
    let artifact = store
        .commit_upload(&upload)
        .expect("fixture artifact is frozen");
    store
        .issue_reference(&artifact)
        .expect("fixture reference is issued")
}

fn store_backed_state(
    plans: impl IntoIterator<Item = StoreBackedEnvironmentPlan>,
) -> StoreBackedEnvironmentState {
    StoreBackedEnvironmentState {
        plans: plans.into_iter().collect(),
        active: None,
        pending_bytes: None,
        upload: None,
        references: Vec::new(),
        stage: StoreBackedEnvironmentStage::AwaitRoot,
        next_adapter_sequence: 1,
        is_tampered_output: false,
        sent: Vec::new(),
        starts: 0,
        reap_reasons: Vec::new(),
        successful_reaps: 0,
        reap_behaviors: VecDeque::new(),
        send_failures: VecDeque::new(),
        receive_failures: VecDeque::new(),
    }
}

fn store_backed_plan(
    operation: &str,
    response: StoreBackedEnvironmentResponse,
    outputs: impl IntoIterator<Item = &'static [u8]>,
) -> StoreBackedEnvironmentPlan {
    StoreBackedEnvironmentPlan {
        operation: operation.to_owned(),
        response_operation: operation.to_owned(),
        response,
        outputs: outputs.into_iter().map(ToOwned::to_owned).collect(),
    }
}

fn store_backed_plan_from_response(
    operation: &str,
    response: AdapterEnvelope,
) -> StoreBackedEnvironmentPlan {
    let response_operation = response.operation.clone();
    let (response, outputs) = match response.message {
        AdapterMessage::EnvironmentReset { .. } => (
            StoreBackedEnvironmentResponse::Reset,
            VecDeque::from([b"dynamic-reset-observation".to_vec()]),
        ),
        AdapterMessage::Transition {
            reward,
            terminated,
            truncated,
            ..
        } => (
            StoreBackedEnvironmentResponse::Transition {
                reward,
                terminated,
                truncated,
            },
            VecDeque::from([
                b"dynamic-transition-observation".to_vec(),
                b"dynamic-transition-info".to_vec(),
            ]),
        ),
        _ => panic!("fixture only scripts environment responses"),
    };
    StoreBackedEnvironmentPlan {
        operation: operation.to_owned(),
        response_operation,
        response,
        outputs,
    }
}

fn tamper_reference(reference: &FrozenArtifactReference) -> FrozenArtifactReference {
    let mut value = serde_json::to_value(reference).expect("fixture reference serializes");
    value["download"] = json!("forged-output-capability");
    serde_json::from_value(value).expect("fixture reference mutation deserializes")
}

fn environment_requests(sent: &[HostEnvelope]) -> Vec<&HostEnvelope> {
    sent.iter()
        .filter(|envelope| {
            matches!(
                envelope.message,
                HostMessage::ResetEnvironment { .. } | HostMessage::StepEnvironment { .. }
            )
        })
        .collect()
}

#[test]
fn environment_protocol_carries_only_frozen_artifact_references() {
    let input = immutable_reference("input");
    let observation = immutable_reference("observation");
    let info = immutable_reference("info");

    let reset = serde_json::to_value(HostMessage::ResetEnvironment {
        input_ref: input.clone(),
    })
    .expect("reset serializes");
    assert!(reset.get("input").is_none());
    assert_eq!(reset["input_ref"], serde_json::to_value(&input).unwrap());

    let step = serde_json::to_value(HostMessage::StepEnvironment {
        action_ref: input.clone(),
    })
    .expect("step serializes");
    assert!(step.get("action").is_none());
    assert_eq!(step["action_ref"], serde_json::to_value(&input).unwrap());

    let transition = serde_json::to_value(AdapterMessage::Transition {
        observation_ref: observation.clone(),
        reward: 1.0,
        terminated: false,
        truncated: false,
        info_ref: info.clone(),
    })
    .expect("transition serializes");
    assert!(transition.get("observation").is_none());
    assert!(transition.get("info").is_none());
    assert_eq!(
        transition["observation_ref"],
        serde_json::to_value(observation).unwrap()
    );
    assert_eq!(transition["info_ref"], serde_json::to_value(info).unwrap());
}

#[test]
fn environment_protocol_refuses_raw_json_payloads() {
    assert!(
        serde_json::from_value::<HostMessage>(json!({
            "type": "reset_environment",
            "input": {"untrusted": "raw-json"},
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<AdapterMessage>(json!({
            "type": "transition",
            "observation": {"untrusted": "raw-json"},
            "reward": 1.0,
            "terminated": true,
            "truncated": false,
            "info": {"untrusted": "raw-json"},
        }))
        .is_err()
    );
}

#[tokio::test]
async fn environment_stepper_refuses_an_unbound_artifact_input_before_dispatch() {
    let state = Rc::new(RefCell::new(FakeEnvironmentAdapterState::default()));
    let store_root = tempfile::tempdir().expect("temporary artifact root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(store_root.path(), artifact_quota())
            .expect("episode artifact store is valid"),
    ));
    let expected_input = freeze_reference(&mut store.borrow_mut(), b"expected-input");
    let deadlines = lifecycle_deadlines(Duration::from_secs(7));
    let binding = EnvironmentStepperBinding::new(
        environment_config(AdapterRole::Environment),
        EnvironmentEpisodeIdentity::new(
            ArtifactDigest::from_bytes(b"package"),
            "episode-1",
            deadlines.operation(),
        )
        .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([expected_input]).expect("artifact bindings are valid"),
    )
    .expect("environment binding is valid");
    let factory = SupervisedEnvironmentStepperFactory::new(Rc::new(FakeEnvironmentRuntime {
        config: environment_config(AdapterRole::Environment),
        state: Rc::clone(&state),
    }));
    let mut stepper = factory
        .start(
            binding,
            EnvironmentSessionAuthority::new(ArtifactDigest::from_bytes(b"package"), store),
            AdapterSpawnRequest::for_non_model_adapter(
                ["environment-adapter".to_owned()],
                Default::default(),
                deadlines,
            )
            .expect("fixture spawn request is valid"),
        )
        .await
        .expect("stepper starts for a matching identity");

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new(
                "reset-1",
                immutable_reference("undeclared-input"),
            ))
            .await,
        Err(EnvironmentStepperError::UndeclaredInput)
    ));
    assert!(state.borrow().sent.is_empty());
}

#[tokio::test]
async fn environment_stepper_factory_refuses_an_operation_deadline_mismatch_before_spawn() {
    let state = Rc::new(RefCell::new(FakeEnvironmentAdapterState::default()));
    let binding_deadlines = lifecycle_deadlines(Duration::from_secs(7));
    let request_deadlines = lifecycle_deadlines(Duration::from_secs(8));
    let binding = EnvironmentStepperBinding::new(
        environment_config(AdapterRole::Environment),
        EnvironmentEpisodeIdentity::new(
            ArtifactDigest::from_bytes(b"package"),
            "episode-1",
            binding_deadlines.operation(),
        )
        .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([immutable_reference("input")])
            .expect("artifact bindings are valid"),
    )
    .expect("environment binding is valid");
    let factory = SupervisedEnvironmentStepperFactory::new(Rc::new(FakeEnvironmentRuntime {
        config: environment_config(AdapterRole::Environment),
        state: Rc::clone(&state),
    }));

    assert!(matches!(
        factory
            .start(
                binding,
                session_authority(ArtifactDigest::from_bytes(b"package")),
                AdapterSpawnRequest::for_non_model_adapter(
                    ["environment-adapter".to_owned()],
                    Default::default(),
                    request_deadlines,
                )
                .expect("fixture spawn request is valid"),
            )
            .await,
        Err(EnvironmentStepperError::OperationDeadlineMismatch)
    ));
    assert_eq!(state.borrow().starts, 0);
}

#[tokio::test]
async fn environment_stepper_factory_refuses_a_package_mismatch_before_spawn() {
    let state = Rc::new(RefCell::new(FakeEnvironmentAdapterState::default()));
    let store_root = tempfile::tempdir().expect("temporary artifact root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(store_root.path(), artifact_quota())
            .expect("episode artifact store is valid"),
    ));
    let binding = EnvironmentStepperBinding::new(
        environment_config(AdapterRole::Environment),
        EnvironmentEpisodeIdentity::new(
            ArtifactDigest::from_bytes(b"binding-package"),
            "episode-1",
            Duration::from_secs(7),
        )
        .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([immutable_reference("input")])
            .expect("artifact bindings are valid"),
    )
    .expect("environment binding is valid");
    let authority = EnvironmentSessionAuthority::new(
        ArtifactDigest::from_bytes(b"actual-imported-package"),
        store,
    );
    let factory = SupervisedEnvironmentStepperFactory::new(Rc::new(FakeEnvironmentRuntime {
        config: environment_config(AdapterRole::Environment),
        state: Rc::clone(&state),
    }));

    assert!(matches!(
        factory.start(binding, authority, spawn_request()).await,
        Err(EnvironmentStepperError::PackageIdentityMismatch)
    ));
    assert_eq!(state.borrow().starts, 0);
}

#[tokio::test]
async fn environment_stepper_freezes_a_dynamic_reset_output_from_the_granted_upload() {
    let package = ArtifactDigest::from_bytes(b"imported-package");
    let store_root = tempfile::tempdir().expect("temporary artifact root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(store_root.path(), artifact_quota())
            .expect("episode artifact store is valid"),
    ));
    let input_ref = freeze_reference(&mut store.borrow_mut(), b"initial-input");
    let binding = EnvironmentStepperBinding::new(
        environment_config(AdapterRole::Environment),
        EnvironmentEpisodeIdentity::new(package.clone(), "episode-1", Duration::from_secs(7))
            .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([input_ref.clone()])
            .expect("artifact input binding is valid"),
    )
    .expect("environment binding is valid");
    let state = Rc::new(RefCell::new(store_backed_state([store_backed_plan(
        "reset-1",
        StoreBackedEnvironmentResponse::Reset,
        [b"observation".as_slice()],
    )])));
    let factory =
        SupervisedEnvironmentStepperFactory::new(Rc::new(StoreBackedEnvironmentRuntime {
            config: environment_config(AdapterRole::Environment),
            state,
        }));
    let mut stepper = factory
        .start(
            binding,
            EnvironmentSessionAuthority::new(package, Rc::clone(&store)),
            spawn_request(),
        )
        .await
        .expect("store-bound stepper starts");

    let reset = stepper
        .reset(EnvironmentResetRequest::new("reset-1", input_ref.clone()))
        .await
        .expect("the uploaded observation is committed by Rust");
    assert_eq!(
        store
            .borrow()
            .read_frozen(reset.observation())
            .expect("committed observation remains readable"),
        b"observation"
    );
    assert!(matches!(
        store.borrow().validate_reference(&input_ref),
        Err(ArtifactError::UnknownDownloadHandle)
    ));
}

async fn start_strict_upload_stepper(
    upload_fragment: StrictUploadFragment,
) -> (
    Box<dyn aiperf_runtime::graph::tools::EnvironmentStepper>,
    Rc<RefCell<EpisodeArtifactStore>>,
    Rc<RefCell<StrictUploadChildState>>,
    FrozenArtifactReference,
    tempfile::TempDir,
) {
    let package = ArtifactDigest::from_bytes(b"imported-package");
    let store_root = tempfile::tempdir().expect("temporary artifact root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(store_root.path(), artifact_quota())
            .expect("episode artifact store is valid"),
    ));
    let input_ref = freeze_reference(&mut store.borrow_mut(), b"initial-input");
    let config = AdapterProtocolConfig::new(
        AdapterRole::Environment,
        "episode-1",
        [
            ProtocolCapability::Environment,
            ProtocolCapability::Artifacts,
        ]
        .into_iter()
        .collect(),
        BTreeSet::new(),
        ProtocolLimits::default(),
    )
    .expect("strict environment protocol configuration is valid");
    let binding = EnvironmentStepperBinding::new(
        config.clone(),
        EnvironmentEpisodeIdentity::new(package.clone(), "episode-1", Duration::from_secs(7))
            .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([input_ref.clone()])
            .expect("artifact input binding is valid"),
    )
    .expect("environment binding is valid");
    let child = Rc::new(RefCell::new(StrictUploadChildState {
        upload_fragment,
        ..Default::default()
    }));
    let runtime = ProtocolAdapterRuntimeFactory::new(
        config,
        Rc::new(StrictAdapterProtocolFactory),
        Rc::new(StrictUploadSpawner {
            state: Rc::clone(&child),
        }),
    );
    let factory = SupervisedEnvironmentStepperFactory::new(Rc::new(runtime));
    let stepper = factory
        .start(
            binding,
            EnvironmentSessionAuthority::new(package, Rc::clone(&store)),
            spawn_request(),
        )
        .await
        .expect("strict JSONL adapter starts");

    (stepper, store, child, input_ref, store_root)
}

async fn start_strict_terminal_rollout_stepper() -> (
    Box<dyn aiperf_runtime::graph::tools::EnvironmentStepper>,
    Rc<RefCell<EpisodeArtifactStore>>,
    Rc<RefCell<StrictUploadChildState>>,
    EnvironmentTestInputs,
    tempfile::TempDir,
) {
    let package = ArtifactDigest::from_bytes(b"imported-package");
    let store_root = tempfile::tempdir().expect("temporary artifact root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(store_root.path(), artifact_quota())
            .expect("episode artifact store is valid"),
    ));
    let reset = freeze_reference(&mut store.borrow_mut(), b"initial-input");
    let action = freeze_reference(&mut store.borrow_mut(), b"action");
    let inputs = EnvironmentTestInputs { reset, action };
    let mut limits = ProtocolLimits::default();
    limits.max_artifact_handles = 2;
    let config = AdapterProtocolConfig::new(
        AdapterRole::Environment,
        "episode-1",
        [
            ProtocolCapability::Environment,
            ProtocolCapability::Artifacts,
        ]
        .into_iter()
        .collect(),
        BTreeSet::new(),
        limits,
    )
    .expect("strict environment protocol configuration is valid");
    let binding = EnvironmentStepperBinding::new(
        config.clone(),
        EnvironmentEpisodeIdentity::new(package.clone(), "episode-1", Duration::from_secs(7))
            .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 1, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([inputs.reset.clone(), inputs.action.clone()])
            .expect("artifact input binding is valid"),
    )
    .expect("environment binding is valid");
    let child = Rc::new(RefCell::new(StrictUploadChildState {
        mode: StrictUploadMode::ResetThenTerminalStep,
        ..Default::default()
    }));
    let runtime = ProtocolAdapterRuntimeFactory::new(
        config,
        Rc::new(StrictAdapterProtocolFactory),
        Rc::new(StrictUploadSpawner {
            state: Rc::clone(&child),
        }),
    );
    let stepper = SupervisedEnvironmentStepperFactory::new(Rc::new(runtime))
        .start(
            binding,
            EnvironmentSessionAuthority::new(package, Rc::clone(&store)),
            spawn_request(),
        )
        .await
        .expect("strict JSONL rollout adapter starts");
    (stepper, store, child, inputs, store_root)
}

#[tokio::test]
async fn strict_adapter_upload_chunk_freezes_and_grants_a_dynamic_environment_output() {
    let (mut stepper, store, child, input_ref, _store_root) =
        start_strict_upload_stepper(StrictUploadFragment::Valid).await;

    let reset = stepper
        .reset(EnvironmentResetRequest::new("reset-1", input_ref))
        .await
        .expect("the strict child uploads bytes through Rust's artifact authority");
    assert_eq!(
        store
            .borrow()
            .read_frozen(reset.observation())
            .expect("Rust froze the strict child output"),
        b"strict-observation"
    );
    assert_eq!(child.borrow().reaps, 0);
}

#[tokio::test]
async fn strict_rollout_releases_each_admitted_output_before_the_next_bounded_grant() {
    let (mut stepper, store, child, inputs, _store_root) =
        start_strict_terminal_rollout_stepper().await;

    let reset = stepper
        .reset(EnvironmentResetRequest::new("reset-1", inputs.reset))
        .await
        .expect("the reset output is admitted and released");
    let transition = stepper
        .step(EnvironmentStepRequest::new("step-1", inputs.action))
        .await
        .expect("two terminal outputs fit only after the reset grant is released");

    assert_eq!(
        store
            .borrow()
            .read_frozen(reset.observation())
            .expect("reset descriptor remains frozen"),
        b"strict-observation"
    );
    assert_eq!(
        store
            .borrow()
            .read_frozen(transition.observation())
            .expect("transition observation descriptor remains frozen"),
        b"strict-transition-observation"
    );
    assert_eq!(
        store
            .borrow()
            .read_frozen(transition.info())
            .expect("transition info descriptor remains frozen"),
        b"strict-transition-info"
    );
    let committed = child.borrow().committed_references.clone();
    assert_eq!(committed.len(), 3);
    for reference in committed {
        assert!(matches!(
            store.borrow().validate_reference(&reference),
            Err(ArtifactError::UnknownDownloadHandle)
        ));
    }
    assert_eq!(child.borrow().reaps, 1);
}

#[tokio::test]
async fn malformed_strict_upload_chunk_reaps_the_child() {
    let (mut stepper, _, child, input_ref, _store_root) =
        start_strict_upload_stepper(StrictUploadFragment::MalformedBase64).await;

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new("reset-1", input_ref))
            .await,
        Err(EnvironmentStepperError::InvalidArtifactUploadEncoding)
    ));
    assert_eq!(child.borrow().reaps, 1);
}

#[tokio::test]
async fn wrong_strict_upload_capability_reaps_the_child() {
    let (mut stepper, _, child, input_ref, _store_root) =
        start_strict_upload_stepper(StrictUploadFragment::WrongCapability).await;

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new("reset-1", input_ref))
            .await,
        Err(EnvironmentStepperError::Supervision(
            AdapterSupervisionError::Protocol(ProtocolError::ArtifactUploadMismatch { .. })
        ))
    ));
    assert_eq!(child.borrow().reaps, 1);
}

#[tokio::test]
async fn overflowing_strict_upload_chunk_reaps_the_child() {
    let (mut stepper, _, child, input_ref, _store_root) =
        start_strict_upload_stepper(StrictUploadFragment::Overflow).await;

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new("reset-1", input_ref))
            .await,
        Err(EnvironmentStepperError::Artifact(
            ArtifactError::UploadLengthExceeded {
                expected: 18,
                actual: 19,
            }
        ))
    ));
    assert_eq!(child.borrow().reaps, 1);
}

#[tokio::test]
async fn oversized_strict_upload_chunk_reaps_before_json_or_base64_decode() {
    let (mut stepper, _, child, input_ref, _store_root) =
        start_strict_upload_stepper(StrictUploadFragment::OversizedFrame).await;

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new("reset-1", input_ref))
            .await,
        Err(EnvironmentStepperError::Supervision(
            AdapterSupervisionError::StdoutFrameLimit { actual, limit }
        )) if actual > limit
    ));
    assert_eq!(child.borrow().reaps, 1);
}

#[tokio::test]
async fn environment_stepper_refuses_a_replayed_action_capability_before_dispatch() {
    let package = ArtifactDigest::from_bytes(b"imported-package");
    let store_root = tempfile::tempdir().expect("temporary artifact root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(store_root.path(), artifact_quota())
            .expect("episode artifact store is valid"),
    ));
    let reset_input = freeze_reference(&mut store.borrow_mut(), b"initial-input");
    let action = freeze_reference(&mut store.borrow_mut(), b"action");
    let binding = EnvironmentStepperBinding::new(
        environment_config(AdapterRole::Environment),
        EnvironmentEpisodeIdentity::new(package.clone(), "episode-1", Duration::from_secs(7))
            .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 3, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([reset_input.clone(), action.clone()])
            .expect("artifact input bindings are valid"),
    )
    .expect("environment binding is valid");
    let state = Rc::new(RefCell::new(store_backed_state([
        store_backed_plan(
            "reset-1",
            StoreBackedEnvironmentResponse::Reset,
            [b"initial-observation".as_slice()],
        ),
        store_backed_plan(
            "step-1",
            StoreBackedEnvironmentResponse::Transition {
                reward: 1.0,
                terminated: false,
                truncated: false,
            },
            [
                b"next-observation".as_slice(),
                b"transition-info".as_slice(),
            ],
        ),
    ])));
    let factory =
        SupervisedEnvironmentStepperFactory::new(Rc::new(StoreBackedEnvironmentRuntime {
            config: environment_config(AdapterRole::Environment),
            state: Rc::clone(&state),
        }));
    let mut stepper = factory
        .start(
            binding,
            EnvironmentSessionAuthority::new(package, Rc::clone(&store)),
            spawn_request(),
        )
        .await
        .expect("store-bound stepper starts");

    stepper
        .reset(EnvironmentResetRequest::new("reset-1", reset_input))
        .await
        .expect("dynamic reset is admitted");
    stepper
        .step(EnvironmentStepRequest::new("step-1", action.clone()))
        .await
        .expect("first action is admitted");
    assert!(matches!(
        store.borrow().validate_reference(&action),
        Err(ArtifactError::UnknownDownloadHandle)
    ));
    let dispatched = state.borrow().sent.len();

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-2", action))
            .await,
        Err(EnvironmentStepperError::UndeclaredInput)
    ));
    assert_eq!(state.borrow().sent.len(), dispatched);
}

#[tokio::test]
async fn environment_stepper_refuses_a_mutated_adapter_output_capability() {
    let package = ArtifactDigest::from_bytes(b"imported-package");
    let store_root = tempfile::tempdir().expect("temporary artifact root");
    let store = Rc::new(RefCell::new(
        EpisodeArtifactStore::new(store_root.path(), artifact_quota())
            .expect("episode artifact store is valid"),
    ));
    let input_ref = freeze_reference(&mut store.borrow_mut(), b"initial-input");
    let binding = EnvironmentStepperBinding::new(
        environment_config(AdapterRole::Environment),
        EnvironmentEpisodeIdentity::new(package.clone(), "episode-1", Duration::from_secs(7))
            .expect("episode identity is valid"),
        "root",
        RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy"),
        EnvironmentArtifactBindings::new([input_ref.clone()])
            .expect("artifact input binding is valid"),
    )
    .expect("environment binding is valid");
    let mut fake = store_backed_state([store_backed_plan(
        "reset-1",
        StoreBackedEnvironmentResponse::Reset,
        [b"observation".as_slice()],
    )]);
    fake.is_tampered_output = true;
    let state = Rc::new(RefCell::new(fake));
    let factory =
        SupervisedEnvironmentStepperFactory::new(Rc::new(StoreBackedEnvironmentRuntime {
            config: environment_config(AdapterRole::Environment),
            state: Rc::clone(&state),
        }));
    let mut stepper = factory
        .start(
            binding,
            EnvironmentSessionAuthority::new(package, Rc::clone(&store)),
            spawn_request(),
        )
        .await
        .expect("store-bound stepper starts");

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new("reset-1", input_ref))
            .await,
        Err(EnvironmentStepperError::UndeclaredOutput)
    ));
    assert!(
        state
            .borrow()
            .sent
            .iter()
            .any(|envelope| matches!(envelope.message, HostMessage::ArtifactCommitted { .. }))
    );
}

#[tokio::test]
async fn supervised_stepper_resets_then_advances_one_correlated_transition() {
    let (mut stepper, state, inputs) = start_stepper(
        2,
        [
            reset_response(),
            transition_response_with_reward("step-1", 2.5, false, false),
        ],
    )
    .await;

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new(
                "step-before-reset",
                inputs.action.clone(),
            ))
            .await,
        Err(EnvironmentStepperError::ResetRequired)
    ));

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");

    let transition = stepper
        .step(EnvironmentStepRequest::new("step-1", inputs.action.clone()))
        .await
        .expect("transition response is correlated");
    assert_eq!(transition.reward(), 2.5);
    assert!(!transition.is_terminated());
    assert!(!transition.is_truncated());

    let borrowed = state.borrow();
    let sent = environment_requests(&borrowed.sent);
    assert!(matches!(
        &sent[..],
        [
            HostEnvelope {
                sequence: 1,
                operation,
                message: HostMessage::ResetEnvironment { input_ref },
                ..
            },
            HostEnvelope {
                sequence: 4,
                operation: step_operation,
                message: HostMessage::StepEnvironment { action_ref },
                ..
            },
        ] if operation == "reset-1"
            && input_ref == &inputs.reset
            && step_operation == "step-1"
            && action_ref == &inputs.action
    ));
}

#[tokio::test]
async fn stale_transition_invalidates_stepper_before_a_later_action_dispatches() {
    let (mut stepper, state, inputs) = start_stepper(
        2,
        [
            reset_response(),
            transition_response("stale-step", false, false),
        ],
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new("reset-1", inputs.reset))
        .await
        .expect("reset response is correlated");
    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-1", inputs.action.clone()))
            .await,
        Err(EnvironmentStepperError::Correlation("operation"))
    ));
    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-2", inputs.action))
            .await,
        Err(EnvironmentStepperError::ProtocolInvalidated)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![CancelReason::IntegrityViolation]
    );
}

#[tokio::test]
async fn invalidated_stepper_retries_a_failed_integrity_reap_before_refusing_later_work() {
    let (mut stepper, state, inputs) = start_stepper_with_reap_behaviors(
        2,
        [
            reset_response(),
            transition_response("stale-step", false, false),
        ],
        VecDeque::from([FakeReapBehavior::Fail, FakeReapBehavior::Complete]),
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-1", inputs.action.clone()))
            .await,
        Err(EnvironmentStepperError::Cleanup {
            primary,
            cleanup: AdapterSupervisionError::Process(_),
        }) if matches!(*primary, EnvironmentStepperError::Correlation("operation"))
    ));

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-2", inputs.action))
            .await,
        Err(EnvironmentStepperError::ProtocolInvalidated)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![
            CancelReason::IntegrityViolation,
            CancelReason::IntegrityViolation,
        ]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[tokio::test]
async fn interrupted_integrity_reap_is_retried_before_later_work_is_refused() {
    let started = Rc::new(Notify::new());
    let release = Rc::new(Notify::new());
    let (mut stepper, state, inputs) = start_stepper_with_reap_behaviors(
        2,
        [
            reset_response(),
            transition_response("stale-step", false, false),
        ],
        VecDeque::from([
            FakeReapBehavior::Block {
                started: Rc::clone(&started),
                release,
            },
            FakeReapBehavior::Complete,
        ]),
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    {
        let operation = stepper.step(EnvironmentStepRequest::new("step-1", inputs.action.clone()));
        tokio::pin!(operation);
        tokio::select! {
            _ = &mut operation => panic!("the fixture reaper must block"),
            _ = started.notified() => {},
        }
    }

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-2", inputs.action))
            .await,
        Err(EnvironmentStepperError::ProtocolInvalidated)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![
            CancelReason::IntegrityViolation,
            CancelReason::IntegrityViolation,
        ]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[tokio::test]
async fn send_failure_reaps_with_an_operation_failure_reason() {
    let (mut stepper, state, inputs) = start_stepper(2, [reset_response()]).await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    state
        .borrow_mut()
        .send_failures
        .push_back(AdapterSupervisionError::Process(
            "fixture send failure".to_owned(),
        ));

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-1", inputs.action))
            .await,
        Err(EnvironmentStepperError::Supervision(
            AdapterSupervisionError::Process(_)
        ))
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 1);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![CancelReason::OperationFailure]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[tokio::test]
async fn receive_failure_reaps_with_an_operation_failure_reason() {
    let (mut stepper, state, inputs) = start_stepper(
        2,
        [
            reset_response(),
            transition_response("step-1", false, false),
        ],
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    state
        .borrow_mut()
        .receive_failures
        .push_back(AdapterSupervisionError::Process(
            "fixture receive failure".to_owned(),
        ));

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-1", inputs.action))
            .await,
        Err(EnvironmentStepperError::Supervision(
            AdapterSupervisionError::Process(_)
        ))
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![CancelReason::OperationFailure]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[tokio::test]
async fn invalidated_reset_retries_pending_cleanup_before_refusing_a_step() {
    let (mut stepper, state, inputs) = start_stepper_with_reap_behaviors(
        2,
        [AdapterEnvelope::new(
            "episode-1",
            "root",
            1,
            "reset-1",
            AdapterMessage::Transition {
                observation_ref: transition_observation_reference(),
                reward: 1.0,
                terminated: false,
                truncated: false,
                info_ref: transition_info_reference(),
            },
        )],
        VecDeque::from([FakeReapBehavior::Fail, FakeReapBehavior::Complete]),
    )
    .await;

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new(
                "reset-1",
                inputs.reset.clone(),
            ))
            .await,
        Err(EnvironmentStepperError::Cleanup {
            primary,
            cleanup: AdapterSupervisionError::Process(_),
        }) if matches!(*primary, EnvironmentStepperError::UnexpectedResponse("environment_reset"))
    ));

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-1", inputs.action.clone()))
            .await,
        Err(EnvironmentStepperError::ProtocolInvalidated)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 1);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![
            CancelReason::IntegrityViolation,
            CancelReason::IntegrityViolation,
        ]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[tokio::test]
async fn failed_terminal_reap_is_retried_before_episode_refusal() {
    let (mut stepper, state, inputs) = start_stepper_with_reap_behaviors(
        2,
        [reset_response(), transition_response("step-1", true, false)],
        VecDeque::from([FakeReapBehavior::Fail, FakeReapBehavior::Complete]),
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-1", inputs.action.clone()))
            .await,
        Err(EnvironmentStepperError::Cleanup {
            primary,
            cleanup: AdapterSupervisionError::Process(_),
        }) if matches!(*primary, EnvironmentStepperError::EpisodeTerminal)
    ));

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-2", inputs.action))
            .await,
        Err(EnvironmentStepperError::EpisodeTerminal)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![CancelReason::HostShutdown, CancelReason::HostShutdown]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[tokio::test]
async fn terminal_pending_cleanup_is_retried_before_reset_is_refused() {
    let (mut stepper, state, inputs) = start_stepper_with_reap_behaviors(
        2,
        [reset_response(), transition_response("step-1", true, false)],
        VecDeque::from([FakeReapBehavior::Fail, FakeReapBehavior::Complete]),
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-1", inputs.action.clone()))
            .await,
        Err(EnvironmentStepperError::Cleanup {
            primary,
            cleanup: AdapterSupervisionError::Process(_),
        }) if matches!(*primary, EnvironmentStepperError::EpisodeTerminal)
    ));

    assert!(matches!(
        stepper
            .reset(EnvironmentResetRequest::new("reset-2", inputs.reset,))
            .await,
        Err(EnvironmentStepperError::EpisodeTerminal)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![CancelReason::HostShutdown, CancelReason::HostShutdown]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[tokio::test]
async fn interrupted_terminal_reap_is_retried_before_episode_refusal() {
    let started = Rc::new(Notify::new());
    let release = Rc::new(Notify::new());
    let (mut stepper, state, inputs) = start_stepper_with_reap_behaviors(
        2,
        [reset_response(), transition_response("step-1", true, false)],
        VecDeque::from([
            FakeReapBehavior::Block {
                started: Rc::clone(&started),
                release,
            },
            FakeReapBehavior::Complete,
        ]),
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    {
        let operation = stepper.step(EnvironmentStepRequest::new("step-1", inputs.action.clone()));
        tokio::pin!(operation);
        tokio::select! {
            _ = &mut operation => panic!("the fixture reaper must block"),
            _ = started.notified() => {},
        }
    }

    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-2", inputs.action))
            .await,
        Err(EnvironmentStepperError::EpisodeTerminal)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![CancelReason::HostShutdown, CancelReason::HostShutdown]
    );
    assert_eq!(state.borrow().successful_reaps, 1);
}

#[test]
fn environment_stepper_requires_task4_environment_role_and_capability() {
    assert!(matches!(
        EnvironmentStepperBinding::new(
            environment_config(AdapterRole::Tool),
            EnvironmentEpisodeIdentity::new(
                ArtifactDigest::from_bytes(b"package"),
                "episode-1",
                Duration::from_secs(7),
            )
            .expect("fixture identity is valid"),
            "root",
            RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy"),
            EnvironmentArtifactBindings::new([reset_input_reference()])
                .expect("fixture artifact bindings are valid"),
        ),
        Err(EnvironmentStepperError::AdapterRole)
    ));
    assert!(matches!(
        AdapterProtocolConfig::new(
            AdapterRole::Environment,
            "episode-1",
            BTreeSet::new(),
            BTreeSet::new(),
            ProtocolLimits::default(),
        ),
        Err(ProtocolError::CapabilityNotDeclared(
            ProtocolCapability::Environment
        ))
    ));
}

#[tokio::test]
async fn environment_stepper_factory_refuses_a_binding_from_another_protocol_session() {
    let state = Rc::new(RefCell::new(FakeEnvironmentAdapterState::default()));
    let binding = environment_binding(
        environment_config_for(AdapterRole::Environment, "binding-episode"),
        2,
    );
    let factory = SupervisedEnvironmentStepperFactory::new(Rc::new(FakeEnvironmentRuntime {
        config: environment_config_for(AdapterRole::Environment, "runtime-episode"),
        state: Rc::clone(&state),
    }));

    assert!(matches!(
        factory
            .start(
                binding,
                session_authority(ArtifactDigest::from_bytes(b"package")),
                spawn_request(),
            )
            .await,
        Err(EnvironmentStepperError::ProtocolConfigMismatch)
    ));
    assert_eq!(state.borrow().starts, 0);
}

#[tokio::test]
async fn terminal_transition_refuses_the_next_environment_action() {
    let (mut stepper, state, inputs) = start_stepper(
        2,
        [reset_response(), transition_response("step-1", true, false)],
    )
    .await;

    stepper
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    let transition = stepper
        .step(EnvironmentStepRequest::new("step-1", inputs.action.clone()))
        .await
        .expect("terminal transition is valid");
    assert_eq!(transition.step(), 0);
    assert_eq!(transition.reward(), 1.0);
    assert!(transition.is_terminated());
    assert!(!transition.is_truncated());
    assert!(matches!(
        stepper
            .step(EnvironmentStepRequest::new("step-2", inputs.action))
            .await,
        Err(EnvironmentStepperError::EpisodeTerminal)
    ));
    assert_eq!(environment_requests(&state.borrow().sent).len(), 2);
    assert_eq!(
        state.borrow().reap_reasons,
        vec![CancelReason::HostShutdown]
    );
}

#[tokio::test]
async fn truncation_and_horizon_refuse_the_next_environment_action() {
    let (mut adapter_truncated, truncated_state, truncated_inputs) = start_stepper(
        2,
        [reset_response(), transition_response("step-1", false, true)],
    )
    .await;
    adapter_truncated
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            truncated_inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    adapter_truncated
        .step(EnvironmentStepRequest::new(
            "step-1",
            truncated_inputs.action.clone(),
        ))
        .await
        .expect("adapter truncation is valid");
    assert!(matches!(
        adapter_truncated
            .step(EnvironmentStepRequest::new(
                "step-2",
                truncated_inputs.action
            ))
            .await,
        Err(EnvironmentStepperError::EpisodeTerminal)
    ));
    assert_eq!(
        environment_requests(&truncated_state.borrow().sent).len(),
        2
    );
    assert_eq!(
        truncated_state.borrow().reap_reasons,
        vec![CancelReason::HostShutdown]
    );

    let (mut horizon_truncated, horizon_state, horizon_inputs) = start_stepper(
        1,
        [
            reset_response(),
            transition_response("step-1", false, false),
        ],
    )
    .await;
    horizon_truncated
        .reset(EnvironmentResetRequest::new(
            "reset-1",
            horizon_inputs.reset.clone(),
        ))
        .await
        .expect("reset response is correlated");
    let transition = horizon_truncated
        .step(EnvironmentStepRequest::new(
            "step-1",
            horizon_inputs.action.clone(),
        ))
        .await
        .expect("Rust applies horizon truncation");
    assert_eq!(transition.step(), 0);
    assert_eq!(transition.reward(), 1.0);
    assert!(!transition.is_terminated());
    assert!(transition.is_truncated());
    assert!(matches!(
        horizon_truncated
            .step(EnvironmentStepRequest::new("step-2", horizon_inputs.action))
            .await,
        Err(EnvironmentStepperError::EpisodeTerminal)
    ));
    assert_eq!(environment_requests(&horizon_state.borrow().sent).len(), 2);
    assert_eq!(
        horizon_state.borrow().reap_reasons,
        vec![CancelReason::HostShutdown]
    );
}

#[test]
fn derives_authoritative_discounted_and_undiscounted_returns() {
    let policy = RlEvaluationPolicy::new("env:v1", 3, 0.5).expect("valid policy");
    let trajectory = policy
        .trajectory([
            transition_record(0, 2.0, false, false).unwrap(),
            transition_record(1, 4.0, true, false).unwrap(),
        ])
        .expect("valid terminal trajectory");
    assert_eq!(trajectory.undiscounted_return(), 6.0);
    assert_eq!(trajectory.discounted_return(), 4.0);
}

#[test]
fn trajectory_rejects_horizon_overflow_without_collecting_the_rest_of_the_stream() {
    let policy = RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy");
    let mut next = 0;
    let transitions = std::iter::from_fn(move || {
        let step = next;
        next += 1;
        match step {
            0 | 1 | 2 => Some(
                transition_record(step, 1.0, false, false).expect("fixture transition is valid"),
            ),
            _ => panic!("trajectory consumed past the first horizon overflow"),
        }
    });

    assert!(matches!(
        policy.trajectory(transitions),
        Err(RlRolloutError::HorizonExceeded)
    ));
}

#[test]
fn trajectory_requires_a_terminal_transition_at_its_nonempty_end() {
    let policy = RlEvaluationPolicy::new("env:v1", 2, 0.5).expect("valid policy");
    assert!(matches!(
        policy.trajectory([]),
        Err(RlRolloutError::MissingTerminal)
    ));
    assert!(matches!(
        policy.trajectory([
            transition_record(0, 1.0, false, false).expect("fixture transition is valid")
        ],),
        Err(RlRolloutError::MissingTerminal)
    ));
}

#[test]
fn trajectory_rejects_nonfinite_derived_returns() {
    let policy = RlEvaluationPolicy::new("env:v1", 2, 1.0).expect("valid policy");
    assert!(matches!(
        policy.trajectory([
            transition_record(0, f64::MAX, false, false).expect("fixture transition is valid"),
            transition_record(1, f64::MAX, true, false).expect("fixture transition is valid"),
        ],),
        Err(RlRolloutError::NonFiniteReturn)
    ));
}

#[test]
fn rejects_illegal_terminal_and_reward_facts() {
    assert!(matches!(
        transition_record(0, f64::NAN, false, false),
        Err(RlRolloutError::NonFiniteReward)
    ));
    assert!(matches!(
        transition_record(0, 1.0, true, true),
        Err(RlRolloutError::AmbiguousTerminal)
    ));
}

fn frozen_rollout_evidence(source: &[u8], action: &[u8]) -> FrozenRolloutEvidence {
    let root = tempfile::tempdir().expect("temporary evidence root");
    let mut store =
        EpisodeArtifactStore::new(root.path(), artifact_quota()).expect("artifact store is valid");
    let reset = freeze_reference(&mut store, b"reset-observation");
    let action = freeze_reference(&mut store, action);
    let observation = freeze_reference(&mut store, b"terminal-observation");
    let info = freeze_reference(&mut store, b"terminal-info");
    let policy = RlEvaluationPolicy::new("environment:v1", 2, 0.5).expect("policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            4.0,
            true,
            false,
            info.artifact().clone(),
        )
        .expect("terminal transition is valid")])
        .expect("trajectory is valid");

    FrozenRolloutEvidence::freeze(
        RolloutEvidenceIdentity::new(
            ArtifactDigest::from_bytes(source),
            ArtifactDigest::from_bytes(b"task"),
            ArtifactDigest::from_bytes(b"environment-implementation"),
        ),
        reset,
        &[action],
        trajectory,
        &store,
    )
    .expect("validated trajectory freezes into path-free evidence")
}

#[test]
fn frozen_rollout_evidence_strips_capabilities_and_agrees_after_store_close() {
    let evidence = frozen_rollout_evidence(b"source", b"action");
    let input = evidence.verifier_input().clone();

    assert_eq!(input.manifest().artifacts().len(), 4);
    assert_eq!(input.transitions().len(), 1);
    assert!(input.transitions()[0].is_terminated());
    assert_eq!(input.transitions()[0].reward(), 4.0);
    input
        .verify_return_agreement()
        .expect("a verifier needs no store or live workspace to check returns");
    let serialized = serde_json::to_string(&input).expect("input serializes");
    assert!(!serialized.contains("download"));
    assert!(!serialized.contains("path"));
}

#[test]
fn frozen_rollout_evidence_identity_binds_source_and_action_artifacts() {
    let baseline = frozen_rollout_evidence(b"source-a", b"action-a");
    let source_changed = frozen_rollout_evidence(b"source-b", b"action-a");
    let action_changed = frozen_rollout_evidence(b"source-a", b"action-b");

    assert_ne!(baseline.identity_digest(), source_changed.identity_digest());
    assert_ne!(baseline.identity_digest(), action_changed.identity_digest());
}

#[test]
fn rollout_verifier_input_rejects_a_tampered_derived_return() {
    let evidence = frozen_rollout_evidence(b"source", b"action");
    let mut tampered = serde_json::to_value(evidence.verifier_input())
        .expect("verifier input serializes for an independent verifier");
    tampered["returns"]["discounted_return"] = json!(f64::from_bits(4.0_f64.to_bits() + 1));
    let tampered = serde_json::to_vec(&tampered).expect("tampered input serializes");
    let limits = rollout_evidence_limits(RlEvaluationLimits::default(), artifact_quota());

    let result = RolloutVerifierInput::decode_bounded(&tampered, &limits);
    assert!(
        matches!(
            result,
            Err(RolloutVerifierDecodeError::Agreement(
                RolloutReturnAgreementError::DiscountedReturnMismatch { .. }
            ))
        ),
        "unexpected bounded decoder result: {result:?}"
    );
}

#[test]
fn rollout_evidence_projects_to_harbor_lifecycle_without_becoming_verifier_input() {
    let evidence = frozen_rollout_evidence(b"source", b"action");
    let verifier_input = evidence.verifier_input().clone();
    let event = evidence.lifecycle_evidence(
        AttemptId::new("rollout-attempt").expect("attempt identity is valid"),
        3,
        None,
    );

    assert_eq!(event.kind, EvidenceKind::Artifact);
    assert_eq!(event.payload, evidence.identity_digest());
    assert_eq!(verifier_input, *evidence.verifier_input());
}

fn rollout_evidence_limits(
    policy: RlEvaluationLimits,
    quota: ArtifactQuota,
) -> RolloutEvidenceLimits {
    RolloutEvidenceLimits::new(8 * 1024, 128, policy, quota)
        .expect("rollout evidence limits are valid")
}

#[test]
fn rollout_policy_rejects_selected_horizon_and_environment_before_trajectory_consumption() {
    let limits = RlEvaluationLimits::new(4, 2).expect("selected RL limits are valid");
    assert!(matches!(
        RlEvaluationPolicy::new_with_limits("environment", 1, 0.5, limits),
        Err(RlRolloutError::EnvironmentTooLong { .. })
    ));
    assert!(matches!(
        RlEvaluationPolicy::new_with_limits("env", 3, 0.5, limits),
        Err(RlRolloutError::HorizonLimitExceeded { .. })
    ));
}

#[test]
fn rollout_verifier_decode_rejects_an_oversized_document_before_json_parsing() {
    let limits = rollout_evidence_limits(
        RlEvaluationLimits::new(64, 2).expect("selected RL limits are valid"),
        artifact_quota(),
    );
    let oversized = vec![b'{'; limits.max_document_bytes() + 1];

    assert!(matches!(
        RolloutVerifierInput::decode_bounded(&oversized, &limits),
        Err(RolloutVerifierDecodeError::DocumentTooLarge { .. })
    ));
}

#[test]
fn rollout_verifier_decode_bounds_strings_transitions_and_manifest_before_retention() {
    let evidence = frozen_rollout_evidence(b"source", b"action");
    let strict_limits = rollout_evidence_limits(
        RlEvaluationLimits::new(4, 1).expect("selected RL limits are valid"),
        ArtifactQuota {
            max_artifacts: 3,
            ..artifact_quota()
        },
    );

    let mut oversized_environment =
        serde_json::to_value(evidence.verifier_input()).expect("verifier input serializes");
    oversized_environment["policy"]["environment"] = json!("environment");
    let oversized_environment =
        serde_json::to_vec(&oversized_environment).expect("oversized environment serializes");
    assert!(matches!(
        RolloutVerifierInput::decode_bounded(&oversized_environment, &strict_limits),
        Err(RolloutVerifierDecodeError::InvalidDocument(_))
    ));

    let mut oversized_transitions =
        serde_json::to_value(evidence.verifier_input()).expect("verifier input serializes");
    let transition = oversized_transitions["transitions"][0].clone();
    oversized_transitions["transitions"] = json!([transition.clone(), transition]);
    let oversized_transitions =
        serde_json::to_vec(&oversized_transitions).expect("oversized transitions serialize");
    assert!(matches!(
        RolloutVerifierInput::decode_bounded(&oversized_transitions, &strict_limits),
        Err(RolloutVerifierDecodeError::InvalidDocument(_))
    ));

    let oversized_manifest =
        serde_json::to_vec(evidence.verifier_input()).expect("oversized manifest serializes");
    assert!(matches!(
        RolloutVerifierInput::decode_bounded(&oversized_manifest, &strict_limits),
        Err(RolloutVerifierDecodeError::InvalidDocument(_))
    ));
}

#[test]
fn rollout_freeze_pre_admits_policy_before_retaining_actions() {
    let root = tempfile::tempdir().expect("temporary evidence root");
    let mut store =
        EpisodeArtifactStore::new(root.path(), artifact_quota()).expect("artifact store is valid");
    let reset = freeze_reference(&mut store, b"reset-observation");
    let observation = freeze_reference(&mut store, b"terminal-observation");
    let info = freeze_reference(&mut store, b"terminal-info");
    let policy = RlEvaluationPolicy::new("environment:v1", 1, 0.5).expect("policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            4.0,
            true,
            false,
            info.artifact().clone(),
        )
        .expect("terminal transition is valid")])
        .expect("trajectory is valid");
    let selected_limits = rollout_evidence_limits(
        RlEvaluationLimits::new(64, 1).expect("selected RL limits are valid"),
        artifact_quota(),
    );
    let action = freeze_reference(&mut store, b"action");

    assert!(matches!(
        FrozenRolloutEvidence::freeze_with_limits(
            RolloutEvidenceIdentity::new(
                ArtifactDigest::from_bytes(b"source"),
                ArtifactDigest::from_bytes(b"task"),
                ArtifactDigest::from_bytes(b"environment-implementation"),
            ),
            reset,
            &[action],
            trajectory,
            &selected_limits,
            &store,
        ),
        Err(RolloutEvidenceError::Admission(_))
    ));
}

#[test]
fn trajectory_keeps_its_strict_policy_limits_when_freeze_offers_looser_limits() {
    let root = tempfile::tempdir().expect("temporary evidence root");
    let mut store =
        EpisodeArtifactStore::new(root.path(), artifact_quota()).expect("artifact store is valid");
    let reset = freeze_reference(&mut store, b"reset");
    let action = freeze_reference(&mut store, b"action");
    let observation = freeze_reference(&mut store, b"observation");
    let info = freeze_reference(&mut store, b"info");
    let strict_limits = RlEvaluationLimits::new(64, 1).expect("strict policy limits are valid");
    let policy = RlEvaluationPolicy::new_with_limits("env:v1", 1, 0.5, strict_limits)
        .expect("strict policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            1.0,
            true,
            false,
            info.artifact().clone(),
        )
        .expect("terminal transition is valid")])
        .expect("trajectory is valid under its selected limits");
    let looser_limits = rollout_evidence_limits(
        RlEvaluationLimits::new(128, 2).expect("looser limits are valid"),
        artifact_quota(),
    );

    assert!(matches!(
        FrozenRolloutEvidence::freeze_with_limits(
            RolloutEvidenceIdentity::new(
                ArtifactDigest::from_bytes(b"source"),
                ArtifactDigest::from_bytes(b"task"),
                ArtifactDigest::from_bytes(b"environment-implementation"),
            ),
            reset,
            &[action],
            trajectory,
            &looser_limits,
            &store,
        ),
        Err(RolloutEvidenceError::Admission(
            RolloutAdmissionError::PolicyLimitsMismatch
        ))
    ));
}

#[test]
fn rollout_freeze_rejects_aggregate_descriptor_bytes_before_retaining_actions() {
    let root = tempfile::tempdir().expect("temporary evidence root");
    let mut store =
        EpisodeArtifactStore::new(root.path(), artifact_quota()).expect("artifact store is valid");
    let reset = freeze_reference(&mut store, b"reset");
    let action = freeze_reference(&mut store, b"action");
    let observation = freeze_reference(&mut store, b"ob");
    let info = freeze_reference(&mut store, b"in");
    let policy_limits = RlEvaluationLimits::new(64, 1).expect("policy limits are valid");
    let policy = RlEvaluationPolicy::new_with_limits("env:v1", 1, 0.5, policy_limits)
        .expect("policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            1.0,
            true,
            false,
            info.artifact().clone(),
        )
        .expect("terminal transition is valid")])
        .expect("trajectory is valid");
    let selected_limits = rollout_evidence_limits(
        policy_limits,
        ArtifactQuota {
            max_total_bytes: 14,
            ..artifact_quota()
        },
    );

    assert!(matches!(
        FrozenRolloutEvidence::freeze_with_limits(
            RolloutEvidenceIdentity::new(
                ArtifactDigest::from_bytes(b"source"),
                ArtifactDigest::from_bytes(b"task"),
                ArtifactDigest::from_bytes(b"environment-implementation"),
            ),
            reset,
            &[action],
            trajectory,
            &selected_limits,
            &store,
        ),
        Err(RolloutEvidenceError::Admission(
            RolloutAdmissionError::ArtifactTotalBytesExceeded {
                requested: 15,
                limit: 14,
            }
        ))
    ));
}

#[test]
fn rollout_verifier_decode_rejects_repeated_descriptor_bytes_above_total_quota() {
    let root = tempfile::tempdir().expect("temporary evidence root");
    let mut store =
        EpisodeArtifactStore::new(root.path(), artifact_quota()).expect("artifact store is valid");
    let reset = freeze_reference(&mut store, b"repeat");
    let descriptor = reset.artifact().clone();
    let action = store
        .issue_reference(&descriptor)
        .expect("action reference is issued");
    let observation = store
        .issue_reference(&descriptor)
        .expect("observation reference is issued");
    let info = store
        .issue_reference(&descriptor)
        .expect("info reference is issued");
    let policy = RlEvaluationPolicy::new("env:v1", 1, 0.5).expect("policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            1.0,
            true,
            false,
            info.artifact().clone(),
        )
        .expect("terminal transition is valid")])
        .expect("trajectory is valid");
    let evidence = FrozenRolloutEvidence::freeze(
        RolloutEvidenceIdentity::new(
            ArtifactDigest::from_bytes(b"source"),
            ArtifactDigest::from_bytes(b"task"),
            ArtifactDigest::from_bytes(b"environment-implementation"),
        ),
        reset,
        &[action],
        trajectory,
        &store,
    )
    .expect("default evidence quota admits the repeated descriptors");
    let document = serde_json::to_vec(evidence.verifier_input()).expect("evidence serializes");
    let strict_limits = rollout_evidence_limits(
        RlEvaluationLimits::default(),
        ArtifactQuota {
            max_total_bytes: 6,
            ..artifact_quota()
        },
    );

    let result = RolloutVerifierInput::decode_bounded(&document, &strict_limits);
    assert!(
        matches!(
            result,
            Err(RolloutVerifierDecodeError::Admission(
                RolloutAdmissionError::ArtifactTotalBytesExceeded {
                    requested: 12,
                    limit: 6,
                }
            ))
        ),
        "unexpected bounded decoder result: {result:?}"
    );
}

#[test]
fn rollout_freeze_refuses_a_verifier_document_above_the_selected_document_limit() {
    let root = tempfile::tempdir().expect("temporary evidence root");
    let mut store =
        EpisodeArtifactStore::new(root.path(), artifact_quota()).expect("artifact store is valid");
    let reset = freeze_reference(&mut store, b"reset");
    let action = freeze_reference(&mut store, b"action");
    let observation = freeze_reference(&mut store, b"observation");
    let info = freeze_reference(&mut store, b"info");
    let policy = RlEvaluationPolicy::new("env:v1", 1, 0.5).expect("policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            1.0,
            true,
            false,
            info.artifact().clone(),
        )
        .expect("terminal transition is valid")])
        .expect("trajectory is valid");
    let limits =
        RolloutEvidenceLimits::new(1, 128, RlEvaluationLimits::default(), artifact_quota())
            .expect("selected evidence limits are valid");

    assert!(matches!(
        FrozenRolloutEvidence::freeze_with_limits(
            RolloutEvidenceIdentity::new(
                ArtifactDigest::from_bytes(b"source"),
                ArtifactDigest::from_bytes(b"task"),
                ArtifactDigest::from_bytes(b"environment-implementation"),
            ),
            reset,
            &[action],
            trajectory,
            &limits,
            &store,
        ),
        Err(RolloutEvidenceError::VerifierDocumentTooLarge { limit: 1 })
    ));
}

#[test]
fn rollout_verifier_decode_rejects_an_oversized_escaped_string_before_serde() {
    let evidence = frozen_rollout_evidence(b"source", b"action");
    let document =
        serde_json::to_string(evidence.verifier_input()).expect("verifier document serializes");
    let escaped_environment = r"\u0065".repeat(129);
    let replacement = format!(r#""environment":"{escaped_environment}""#);
    let document = document.replacen(r#""environment":"environment:v1""#, &replacement, 1);
    let limits = rollout_evidence_limits(RlEvaluationLimits::default(), artifact_quota());

    assert!(matches!(
        RolloutVerifierInput::decode_bounded(document.as_bytes(), &limits),
        Err(RolloutVerifierDecodeError::StringTooLong { limit: 128 })
    ));
}

#[test]
fn rollout_verifier_decode_counts_escaped_strings_by_decoded_bytes() {
    let root = tempfile::tempdir().expect("temporary evidence root");
    let mut store =
        EpisodeArtifactStore::new(root.path(), artifact_quota()).expect("artifact store is valid");
    let reset = freeze_reference(&mut store, b"reset");
    let action = freeze_reference(&mut store, b"action");
    let observation = freeze_reference(&mut store, b"observation");
    let info = freeze_reference(&mut store, b"info");
    let environment = "e".repeat(100);
    let policy_limits = RlEvaluationLimits::new(128, 1).expect("policy limits are valid");
    let policy = RlEvaluationPolicy::new_with_limits(&environment, 1, 0.5, policy_limits)
        .expect("policy is valid");
    let trajectory = policy
        .trajectory([EnvironmentTransitionRecord::new(
            0,
            observation.artifact().clone(),
            1.0,
            true,
            false,
            info.artifact().clone(),
        )
        .expect("terminal transition is valid")])
        .expect("trajectory is valid");
    let limits = RolloutEvidenceLimits::new(8 * 1024, 128, policy_limits, artifact_quota())
        .expect("selected evidence limits are valid");
    let evidence = FrozenRolloutEvidence::freeze_with_limits(
        RolloutEvidenceIdentity::new(
            ArtifactDigest::from_bytes(b"source"),
            ArtifactDigest::from_bytes(b"task"),
            ArtifactDigest::from_bytes(b"environment-implementation"),
        ),
        reset,
        &[action],
        trajectory,
        &limits,
        &store,
    )
    .expect("evidence is valid");
    let document =
        serde_json::to_string(evidence.verifier_input()).expect("verifier document serializes");
    let escaped_environment = r"\u0065".repeat(100);
    let expected = format!(r#""environment":"{environment}""#);
    let replacement = format!(r#""environment":"{escaped_environment}""#);
    let document = document.replacen(&expected, &replacement, 1);

    let decoded = RolloutVerifierInput::decode_bounded(document.as_bytes(), &limits)
        .expect("escaped source bytes below the decoded string limit are admissible");
    assert_eq!(decoded.policy().environment(), environment);
}
