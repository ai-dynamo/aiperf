// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned admission credentials for cellular registrations.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, ensure};
use bytes::Bytes;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::TryRngCore;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::cellular::transport::connect::ControllerPeerBinding;
use crate::cellular::transport::{CellRegister, CellRegistrationProof, HANDLER_STORE_PARTITION};
use crate::engine::cellular_bootstrap::CellularRole;

const REGISTRATION_PROTOCOL_VERSION: u8 = 1;
const TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-registration-v1\0";
const AUTHENTICATED_FRAME_TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-frame-v1\0";
pub(crate) const ADMISSION_PURPOSE_COUNT: usize = 12;
const VELO_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES: usize = 22;
pub(crate) const MAX_AUTHENTICATED_FRAME_BYTES: usize =
    VELO_MAX_FRAME_BYTES - VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES - HANDLER_STORE_PARTITION.len();
const REGISTER_REPLY_TRANSCRIPT_DOMAIN: &[u8] = b"aiperf-cellular-register-reply-v1\0";

/// One public role key in the controller's fixed authorization roster.
#[derive(Clone, Copy)]
pub(crate) struct RoleVerifyingKey {
    pub(crate) role: CellularRole,
    pub(crate) verifier: VerifyingKey,
}

/// The sole private cellular authority owned by one process.
pub(crate) struct CellSecurityContext {
    run_nonce: [u8; 32],
    session_nonce: [u8; 32],
    authority: ProcessSecurityAuthority,
    send_sequences: [AtomicU64; ADMISSION_PURPOSE_COUNT],
}

enum ProcessSecurityAuthority {
    Controller {
        signer: SigningKey,
        role_verifiers: Box<[RoleVerifyingKey]>,
    },
    Worker {
        role: CellularRole,
        signer: SigningKey,
        controller_verifier: VerifyingKey,
    },
}

impl CellSecurityContext {
    pub(crate) fn controller(
        run_nonce: [u8; 32],
        signer: SigningKey,
        role_verifiers: Box<[RoleVerifyingKey]>,
    ) -> Result<Self> {
        ensure!(
            !role_verifiers.is_empty(),
            "controller security roster is empty"
        );
        Ok(Self {
            run_nonce,
            session_nonce: random_nonce("process session nonce")?,
            authority: ProcessSecurityAuthority::Controller {
                signer,
                role_verifiers,
            },
            send_sequences: std::array::from_fn(|_| AtomicU64::new(1)),
        })
    }

    pub(crate) fn worker(
        run_nonce: [u8; 32],
        role: CellularRole,
        signer: SigningKey,
        controller_verifier: VerifyingKey,
    ) -> Result<Self> {
        Ok(Self {
            run_nonce,
            session_nonce: random_nonce("process session nonce")?,
            authority: ProcessSecurityAuthority::Worker {
                role,
                signer,
                controller_verifier,
            },
            send_sequences: std::array::from_fn(|_| AtomicU64::new(1)),
        })
    }

    pub(crate) fn run_nonce(&self) -> [u8; 32] {
        self.run_nonce
    }

    pub(crate) fn role(&self) -> Option<CellularRole> {
        match self.authority {
            ProcessSecurityAuthority::Controller { .. } => None,
            ProcessSecurityAuthority::Worker { role, .. } => Some(role),
        }
    }

    pub(crate) fn role_verifiers(&self) -> Result<Box<[RoleVerifyingKey]>> {
        match &self.authority {
            ProcessSecurityAuthority::Controller { role_verifiers, .. } => {
                Ok(role_verifiers.to_vec().into_boxed_slice())
            }
            ProcessSecurityAuthority::Worker { .. } => {
                anyhow::bail!("worker security context has no controller roster")
            }
        }
    }

    pub(crate) fn controller_verifier(&self) -> Result<VerifyingKey> {
        match &self.authority {
            ProcessSecurityAuthority::Worker {
                controller_verifier,
                ..
            } => Ok(*controller_verifier),
            ProcessSecurityAuthority::Controller { .. } => {
                anyhow::bail!("controller security context has no controller verifier")
            }
        }
    }

    fn sign_worker(&self, transcript: &[u8]) -> Result<Signature> {
        match &self.authority {
            ProcessSecurityAuthority::Worker { signer, .. } => Ok(signer.sign(transcript)),
            ProcessSecurityAuthority::Controller { .. } => {
                anyhow::bail!("controller security context cannot sign worker admission")
            }
        }
    }

    fn sign_controller(&self, transcript: &[u8]) -> Result<Signature> {
        match &self.authority {
            ProcessSecurityAuthority::Controller { signer, .. } => Ok(signer.sign(transcript)),
            ProcessSecurityAuthority::Worker { .. } => {
                anyhow::bail!("worker security context cannot attest controller replies")
            }
        }
    }

    pub(crate) fn seal(
        &self,
        purpose: AdmissionPurpose,
        peer: &velo::PeerInfo,
        payload: Vec<u8>,
    ) -> Result<AuthenticatedFrame> {
        let role = self
            .role()
            .ok_or_else(|| anyhow::anyhow!("controller cannot seal worker application frames"))?;
        ensure!(
            purpose.supports(role),
            "cellular role cannot use this admission purpose"
        );
        let sequence = next_sequence(&self.send_sequences[purpose.index()])?;
        let peer_info = rmp_serde::to_vec(peer)
            .map_err(|error| anyhow::anyhow!("encode authenticated frame peer: {error}"))?;
        let transcript = authenticated_frame_transcript(
            self.run_nonce,
            role,
            purpose,
            self.session_nonce,
            sequence,
            &peer_info,
            &payload,
        );
        Ok(AuthenticatedFrame {
            version: REGISTRATION_PROTOCOL_VERSION,
            role,
            session_nonce: self.session_nonce,
            sequence,
            peer_info,
            payload,
            signature: self.sign_worker(&transcript)?.to_bytes().to_vec(),
        })
    }

    pub(crate) fn registration_credential(self: &Arc<Self>) -> Result<CellRegistrationCredential> {
        let Some(CellularRole::Cell(cell_id)) = self.role() else {
            anyhow::bail!("security context has no cell credential");
        };
        Ok(CellRegistrationCredential {
            cell_id,
            context: Arc::clone(self),
        })
    }

    pub(crate) fn registration_authority(self: &Arc<Self>) -> Result<CellRegistrationAuthority> {
        CellRegistrationAuthority::from_controller_context(self)
    }

    pub(crate) fn reply_attestor(self: &Arc<Self>) -> Result<ControllerRegisterAttestor> {
        ensure!(
            matches!(self.authority, ProcessSecurityAuthority::Controller { .. }),
            "worker has no controller reply authority"
        );
        Ok(ControllerRegisterAttestor {
            context: Arc::clone(self),
        })
    }
}

fn next_sequence(sequence: &AtomicU64) -> Result<u64> {
    let mut current = sequence.load(Ordering::Relaxed);
    loop {
        ensure!(current != 0, "cellular admission sequence is exhausted");
        let next = if current == u64::MAX { 0 } else { current + 1 };
        match sequence.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Ok(current),
            Err(observed) => current = observed,
        }
    }
}

fn random_nonce(class: &str) -> Result<[u8; 32]> {
    let mut nonce = [0_u8; 32];
    rand::rngs::OsRng
        .try_fill_bytes(&mut nonce)
        .map_err(|_| anyhow::anyhow!("OS RNG could not mint cellular {class}"))?;
    Ok(nonce)
}

/// One authenticated cell-to-controller application operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum AdmissionPurpose {
    /// Registers one launched role and commits its process session nonce.
    Register = 1,
    /// Reports the envelope-local replay preflight result.
    Preflight = 2,
    /// Delivers one cell metrics heartbeat.
    Heartbeat = 3,
    /// Delivers one controller-owned phase barrier signal.
    PhaseSignal = 4,
    /// Delivers a records shard to the controller.
    Partition = 5,
    /// Delivers a folded column store to the controller.
    StorePartition = 6,
    /// Subscribes to controller-owned phase transitions.
    PhaserSubscribe = 7,
    /// Subscribes to controller-owned dataset chunks.
    DatasetSubscribe = 8,
    /// Opens a Velo artifact stream.
    ArtifactOpen = 9,
    /// Waits for an exact artifact path to commit.
    ArtifactClose = 10,
    /// Marks one cell's artifact stream terminal.
    ArtifactDone = 11,
    /// Delivers an aggregator-owned upstream partition.
    AggregatorStorePartition = 12,
}

impl AdmissionPurpose {
    const fn index(self) -> usize {
        self as usize - 1
    }

    const fn supports(self, role: CellularRole) -> bool {
        match self {
            Self::AggregatorStorePartition => matches!(role, CellularRole::Aggregator { .. }),
            _ => matches!(role, CellularRole::Cell(_)),
        }
    }
}

/// One signed, raw-payload application frame.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AuthenticatedFrame {
    version: u8,
    role: CellularRole,
    session_nonce: [u8; 32],
    sequence: u64,
    peer_info: Vec<u8>,
    payload: Vec<u8>,
    signature: Vec<u8>,
}

impl AuthenticatedFrame {
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, AdmissionRejection> {
        if bytes.len() > MAX_AUTHENTICATED_FRAME_BYTES {
            return Err(AdmissionRejection::Oversized);
        }
        rmp_serde::from_slice(bytes).map_err(|_| AdmissionRejection::Malformed)
    }

    pub(crate) fn role(&self) -> CellularRole {
        self.role
    }

    pub(crate) fn peer_info(&self) -> &[u8] {
        &self.peer_info
    }

    pub(crate) fn payload(&self) -> &[u8] {
        &self.payload
    }

    #[cfg(test)]
    pub(crate) fn session_nonce(&self) -> [u8; 32] {
        self.session_nonce
    }

    #[cfg(test)]
    pub(crate) fn sequence(&self) -> u64 {
        self.sequence
    }
}

/// A fixed-class admission failure. Wire handlers expose only `AdmissionRejected`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionRejection {
    Oversized,
    Malformed,
    Role,
    Signature,
    Session,
    Replay,
}

impl std::fmt::Display for AdmissionRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("AdmissionRejected")
    }
}

impl std::error::Error for AdmissionRejection {}

/// A signature-verified frame whose raw payload is still undecoded.
pub(crate) struct VerifiedFrame {
    role: CellularRole,
    session_nonce: [u8; 32],
    sequence: u64,
    peer_info: Vec<u8>,
    payload: Vec<u8>,
    encoded: Bytes,
    fingerprint: [u8; 32],
}

impl VerifiedFrame {
    pub(crate) fn role(&self) -> CellularRole {
        self.role
    }

    pub(crate) fn session_nonce(&self) -> [u8; 32] {
        self.session_nonce
    }

    pub(crate) fn peer_info(&self) -> &[u8] {
        &self.peer_info
    }

    pub(crate) fn encoded(&self) -> &Bytes {
        &self.encoded
    }

    pub(crate) fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    pub(crate) fn decode_payload<T: DeserializeOwned>(&self) -> Result<T> {
        rmp_serde::from_slice(&self.payload)
            .map_err(|_| anyhow::anyhow!("authenticated registration payload is malformed"))
    }
}

/// A verified frame after its route-specific payload has been decoded.
pub(crate) struct VerifiedPayload<T> {
    role: CellularRole,
    session_nonce: [u8; 32],
    peer_info: Vec<u8>,
    payload: T,
}

impl<T> VerifiedPayload<T> {
    pub(crate) fn role(&self) -> CellularRole {
        self.role
    }

    pub(crate) fn into_payload(self) -> T {
        self.payload
    }

    pub(crate) fn into_parts(self) -> (CellularRole, [u8; 32], Vec<u8>, T) {
        (self.role, self.session_nonce, self.peer_info, self.payload)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ReplayWindow {
    highest: u64,
    seen: u64,
}

impl ReplayWindow {
    fn accept(&mut self, sequence: u64) -> bool {
        if self.seen == 0 {
            self.highest = sequence;
            self.seen = 1;
            return true;
        }
        if sequence > self.highest {
            let distance = sequence - self.highest;
            self.highest = sequence;
            self.seen = if distance >= 64 {
                1
            } else {
                (self.seen << distance) | 1
            };
            return true;
        }
        let distance = self.highest - sequence;
        if distance >= 64 {
            return false;
        }
        let mask = 1_u64 << distance;
        if self.seen & mask != 0 {
            return false;
        }
        self.seen |= mask;
        true
    }
}

#[derive(Default)]
struct RoleAdmissionState {
    session_nonce: Option<[u8; 32]>,
    replay: [ReplayWindow; ADMISSION_PURPOSE_COUNT],
}

struct RoleAdmissionSlot {
    role: CellularRole,
    verifier: VerifyingKey,
    state: parking_lot::Mutex<RoleAdmissionState>,
}

/// Fixed per-purpose rejection counters.
pub(crate) struct AdmissionDropCounters {
    counts: [AtomicU64; ADMISSION_PURPOSE_COUNT],
}

impl AdmissionDropCounters {
    fn new() -> Self {
        Self {
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    fn increment(&self, purpose: AdmissionPurpose) {
        self.counts[purpose.index()].fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn get(&self, purpose: AdmissionPurpose) -> u64 {
        self.counts[purpose.index()].load(Ordering::Relaxed)
    }
}

/// Controller-owned fixed role slots and replay windows.
pub(crate) struct AdmissionLedger {
    run_nonce: [u8; 32],
    slots: Box<[RoleAdmissionSlot]>,
    drops: AdmissionDropCounters,
}

impl AdmissionLedger {
    fn new(run_nonce: [u8; 32], roster: &[RoleVerifyingKey]) -> Self {
        let slots = roster
            .iter()
            .map(|entry| RoleAdmissionSlot {
                role: entry.role,
                verifier: entry.verifier,
                state: parking_lot::Mutex::new(RoleAdmissionState::default()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            run_nonce,
            slots,
            drops: AdmissionDropCounters::new(),
        }
    }

    pub(crate) fn open(
        &self,
        purpose: AdmissionPurpose,
        frame: AuthenticatedFrame,
    ) -> Result<VerifiedFrame, AdmissionRejection> {
        self.open_inner(purpose, frame).map_err(|rejection| {
            self.drops.increment(purpose);
            rejection
        })
    }

    fn open_inner(
        &self,
        purpose: AdmissionPurpose,
        frame: AuthenticatedFrame,
    ) -> Result<VerifiedFrame, AdmissionRejection> {
        let verified = self.authenticate_inner(purpose, frame, Bytes::new(), [0; 32])?;
        {
            let mut state = self
                .slots
                .iter()
                .find(|slot| slot.role == verified.role)
                .ok_or(AdmissionRejection::Role)?
                .state
                .lock();
            if purpose == AdmissionPurpose::Register {
                if state.session_nonce != Some(verified.session_nonce) {
                    *state = RoleAdmissionState {
                        session_nonce: Some(verified.session_nonce),
                        replay: [ReplayWindow::default(); ADMISSION_PURPOSE_COUNT],
                    };
                }
            } else if state.session_nonce != Some(verified.session_nonce) {
                return Err(AdmissionRejection::Session);
            }
            if !state.replay[purpose.index()].accept(verified.sequence) {
                return Err(AdmissionRejection::Replay);
            }
        }
        Ok(verified)
    }

    fn authenticate(
        &self,
        purpose: AdmissionPurpose,
        frame: AuthenticatedFrame,
        encoded: Bytes,
    ) -> Result<VerifiedFrame, AdmissionRejection> {
        let fingerprint = *blake3::hash(&encoded).as_bytes();
        self.authenticate_inner(purpose, frame, encoded, fingerprint)
            .map_err(|rejection| {
                self.drops.increment(purpose);
                rejection
            })
    }

    fn authenticate_inner(
        &self,
        purpose: AdmissionPurpose,
        frame: AuthenticatedFrame,
        encoded: Bytes,
        fingerprint: [u8; 32],
    ) -> Result<VerifiedFrame, AdmissionRejection> {
        if frame.version != REGISTRATION_PROTOCOL_VERSION || !purpose.supports(frame.role) {
            return Err(AdmissionRejection::Role);
        }
        if frame.signature.len() != Signature::BYTE_SIZE {
            return Err(AdmissionRejection::Malformed);
        }
        let slot = self
            .slots
            .iter()
            .find(|slot| slot.role == frame.role)
            .ok_or(AdmissionRejection::Role)?;
        let signature =
            Signature::from_slice(&frame.signature).map_err(|_| AdmissionRejection::Malformed)?;
        slot.verifier
            .verify(
                &authenticated_frame_transcript(
                    self.run_nonce,
                    frame.role,
                    purpose,
                    frame.session_nonce,
                    frame.sequence,
                    &frame.peer_info,
                    &frame.payload,
                ),
                &signature,
            )
            .map_err(|_| AdmissionRejection::Signature)?;

        Ok(VerifiedFrame {
            role: frame.role,
            session_nonce: frame.session_nonce,
            sequence: frame.sequence,
            peer_info: frame.peer_info,
            payload: frame.payload,
            encoded,
            fingerprint,
        })
    }

    fn reject<T>(
        &self,
        purpose: AdmissionPurpose,
        rejection: AdmissionRejection,
    ) -> Result<T, AdmissionRejection> {
        self.drops.increment(purpose);
        Err(rejection)
    }

    fn commit_session(&self, role: CellularRole, session_nonce: [u8; 32]) {
        if let Some(slot) = self.slots.iter().find(|slot| slot.role == role) {
            let mut state = slot.state.lock();
            if state.session_nonce != Some(session_nonce) {
                *state = RoleAdmissionState {
                    session_nonce: Some(session_nonce),
                    replay: [ReplayWindow::default(); ADMISSION_PURPOSE_COUNT],
                };
            }
        }
    }

    #[cfg(test)]
    fn replay_slot_count(&self) -> usize {
        self.slots.len() * ADMISSION_PURPOSE_COUNT
    }
}

/// Controller-owned cell verifiers, run nonce, and provisioned reply-attestation capability.
pub(crate) struct CellRegistrationAuthority {
    run_nonce: [u8; 32],
    role_verifiers: Box<[RoleVerifyingKey]>,
    reply_attestor: ControllerRegisterAttestor,
    admission_ledger: AdmissionLedger,
}

/// The private, cell-specific signing key delivered only by a trusted launcher.
#[cfg_attr(test, derive(Clone))]
pub(crate) struct CellRegistrationCredential {
    cell_id: u32,
    context: Arc<CellSecurityContext>,
}

/// Signed controller evidence for one complete registration reply.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ControllerRegisterAttestation {
    /// Fixed protocol version for the signed reply transcript.
    pub version: u8,
    /// Per-run nonce selected by the deployment or local controller.
    pub run_nonce: [u8; 32],
    /// Ed25519 signature over controller peer, registration, and reply material.
    pub signature: Vec<u8>,
}

/// Controller-only signing authority for registration replies.
#[derive(Clone)]
pub(crate) struct ControllerRegisterAttestor {
    context: Arc<CellSecurityContext>,
}

/// Role-local verifier for controller registration replies.
#[derive(Clone)]
pub(crate) struct ControllerRegisterVerifier {
    run_nonce: [u8; 32],
    verifying_key: VerifyingKey,
}

/// A controller-verified registration identity.
#[derive(Clone, Copy)]
pub(crate) struct VerifiedCellRegistration {
    cell_id: u32,
}

impl VerifiedCellRegistration {
    pub(crate) fn cell_id(self) -> u32 {
        self.cell_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegistrationBusy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RegistrationConflict {
    ChangedTranscript,
    Busy(RegistrationBusy),
}

impl std::fmt::Display for RegistrationConflict {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChangedTranscript => formatter.write_str("ChangedTranscript"),
            Self::Busy(_) => formatter.write_str("RegistrationBusy"),
        }
    }
}

impl std::error::Error for RegistrationConflict {}

enum RegistrationSlot {
    Vacant,
    Preparing {
        fingerprint: [u8; 32],
    },
    Committed {
        fingerprint: [u8; 32],
        session_nonce: [u8; 32],
        reply: Bytes,
    },
}

struct RegistrationLedgerState {
    slots: Box<[RegistrationSlot]>,
}

#[derive(Clone)]
pub(crate) struct RegistrationLedger {
    state: Arc<parking_lot::Mutex<RegistrationLedgerState>>,
    authority: Arc<CellRegistrationAuthority>,
    expected_cell_count: u32,
}

pub(crate) enum RegistrationBegin {
    ExactRetry { reply: Bytes },
    Prepared(PreparedRegistration),
}

pub(crate) struct PreparedRegistration {
    ledger: RegistrationLedger,
    cell_id: usize,
    role: CellularRole,
    session_nonce: [u8; 32],
    fingerprint: [u8; 32],
    plan: Option<crate::cellular::transport::velo_transport::CellRegistrationPlan>,
    is_active: bool,
    #[cfg(test)]
    drop_pause: Option<PreparedRegistrationDropPause>,
}

#[cfg(test)]
struct PreparedRegistrationDropPause {
    slot_is_vacant: Arc<std::sync::Barrier>,
    release: Arc<std::sync::Barrier>,
}

pub(crate) struct RegistrationCommit {
    should_advance_barrier: bool,
}

impl RegistrationCommit {
    pub(crate) fn should_advance_barrier(&self) -> bool {
        self.should_advance_barrier
    }
}

impl RegistrationLedger {
    pub(crate) fn new(authority: Arc<CellRegistrationAuthority>, expected_cell_count: u32) -> Self {
        let capacity = authority.planned_cell_capacity();
        Self {
            state: Arc::new(parking_lot::Mutex::new(RegistrationLedgerState {
                slots: (0..capacity)
                    .map(|_| RegistrationSlot::Vacant)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            })),
            authority,
            expected_cell_count,
        }
    }

    pub(crate) fn cached_reply(
        &self,
        verified: &VerifiedFrame,
    ) -> Result<Option<Bytes>, RegistrationConflict> {
        let cell_id = match verified.role() {
            CellularRole::Cell(cell_id) => cell_id as usize,
            CellularRole::Aggregator { .. } => {
                return Err(RegistrationConflict::ChangedTranscript);
            }
        };
        let state = self.state.lock();
        let slot = state
            .slots
            .get(cell_id)
            .ok_or(RegistrationConflict::ChangedTranscript)?;
        match slot {
            RegistrationSlot::Vacant => Ok(None),
            RegistrationSlot::Preparing { fingerprint, .. }
                if *fingerprint == verified.fingerprint() =>
            {
                Err(RegistrationConflict::Busy(RegistrationBusy))
            }
            RegistrationSlot::Preparing { .. } => Err(RegistrationConflict::ChangedTranscript),
            RegistrationSlot::Committed {
                fingerprint,
                session_nonce,
                reply,
                ..
            } if *fingerprint == verified.fingerprint()
                && *session_nonce == verified.session_nonce() =>
            {
                Ok(Some(reply.clone()))
            }
            RegistrationSlot::Committed { .. } => Err(RegistrationConflict::ChangedTranscript),
        }
    }

    pub(crate) fn begin(
        &self,
        verified: &VerifiedFrame,
    ) -> Result<RegistrationBegin, RegistrationConflict> {
        if let Some(reply) = self.cached_reply(&verified)? {
            return Ok(RegistrationBegin::ExactRetry { reply });
        }
        let cell_id = match verified.role() {
            CellularRole::Cell(cell_id) => cell_id as usize,
            CellularRole::Aggregator { .. } => {
                return Err(RegistrationConflict::ChangedTranscript);
            }
        };
        let mut state = self.state.lock();
        let slot = state
            .slots
            .get_mut(cell_id)
            .ok_or(RegistrationConflict::ChangedTranscript)?;
        match slot {
            RegistrationSlot::Vacant => {
                *slot = RegistrationSlot::Preparing {
                    fingerprint: verified.fingerprint(),
                };
            }
            RegistrationSlot::Preparing { fingerprint, .. }
                if *fingerprint == verified.fingerprint() =>
            {
                return Err(RegistrationConflict::Busy(RegistrationBusy));
            }
            RegistrationSlot::Committed {
                fingerprint, reply, ..
            } if *fingerprint == verified.fingerprint() => {
                return Ok(RegistrationBegin::ExactRetry {
                    reply: reply.clone(),
                });
            }
            RegistrationSlot::Preparing { .. } | RegistrationSlot::Committed { .. } => {
                return Err(RegistrationConflict::ChangedTranscript);
            }
        }
        Ok(RegistrationBegin::Prepared(PreparedRegistration {
            ledger: self.clone(),
            cell_id,
            role: verified.role(),
            session_nonce: verified.session_nonce(),
            fingerprint: verified.fingerprint(),
            plan: None,
            is_active: true,
            #[cfg(test)]
            drop_pause: None,
        }))
    }
}

impl PreparedRegistration {
    pub(crate) fn install_plan(
        &mut self,
        plan: crate::cellular::transport::velo_transport::CellRegistrationPlan,
    ) {
        self.plan = Some(plan);
    }

    pub(crate) fn envelope(&self) -> &[u8] {
        self.plan
            .as_ref()
            .map_or(&[], |plan| plan.envelope.as_slice())
    }

    pub(crate) fn artifact_channel(
        &self,
    ) -> Option<crate::cellular::transport::ArtifactChannelServerConfig> {
        self.plan
            .as_ref()
            .and_then(|plan| plan.artifact.as_ref())
            .map(|artifact| artifact.server_config())
    }

    pub(crate) fn commit(mut self, reply: Bytes) -> RegistrationCommit {
        self.ledger
            .authority
            .commit_registration_session(self.role, self.session_nonce);
        if let Some(artifact) = self.plan.take().and_then(|plan| plan.artifact) {
            artifact.commit();
        }
        let should_advance_barrier = {
            let mut state = self.ledger.state.lock();
            state.slots[self.cell_id] = RegistrationSlot::Committed {
                fingerprint: self.fingerprint,
                session_nonce: self.session_nonce,
                reply,
            };
            state
                .slots
                .iter()
                .filter(|slot| matches!(slot, RegistrationSlot::Committed { .. }))
                .count()
                == self.ledger.expected_cell_count as usize
        };
        self.is_active = false;
        RegistrationCommit {
            should_advance_barrier,
        }
    }
}

impl Drop for PreparedRegistration {
    fn drop(&mut self) {
        if !self.is_active {
            return;
        }
        drop(self.plan.take());
        let mut state = self.ledger.state.lock();
        if matches!(
            state.slots.get(self.cell_id),
            Some(RegistrationSlot::Preparing { fingerprint, .. })
                if *fingerprint == self.fingerprint
        ) {
            state.slots[self.cell_id] = RegistrationSlot::Vacant;
        }
        #[cfg(test)]
        if let Some(pause) = self.drop_pause.as_ref() {
            drop(state);
            pause.slot_is_vacant.wait();
            pause.release.wait();
        }
    }
}

impl CellRegistrationAuthority {
    #[cfg(test)]
    pub(crate) fn mint(cell_count: u32) -> Result<(Self, Vec<CellRegistrationCredential>)> {
        ensure!(
            cell_count > 0,
            "cell registration requires at least one cell"
        );
        let mut run_nonce = [0_u8; 32];
        rand::rngs::OsRng
            .try_fill_bytes(&mut run_nonce)
            .map_err(|_| anyhow::anyhow!("OS RNG could not mint cellular registration nonce"))?;
        let controller_signer = SigningKey::from_bytes(&random_nonce("controller reply key")?);
        let controller_verifier = controller_signer.verifying_key();
        let mut role_verifiers = Vec::with_capacity(cell_count as usize);
        let mut credentials = Vec::with_capacity(cell_count as usize);
        for cell_id in 0..cell_count {
            let seed = random_nonce("registration key")?;
            let signing_key = SigningKey::from_bytes(&seed);
            role_verifiers.push(RoleVerifyingKey {
                role: CellularRole::Cell(cell_id),
                verifier: signing_key.verifying_key(),
            });
            let context = Arc::new(CellSecurityContext::worker(
                run_nonce,
                CellularRole::Cell(cell_id),
                signing_key,
                controller_verifier,
            )?);
            credentials.push(CellRegistrationCredential { cell_id, context });
        }
        let context = Arc::new(CellSecurityContext::controller(
            run_nonce,
            controller_signer,
            role_verifiers.into_boxed_slice(),
        )?);
        let authority = context.registration_authority()?;
        for credential in &credentials {
            authority.admission_ledger.commit_session(
                CellularRole::Cell(credential.cell_id),
                credential.context.session_nonce,
            );
        }
        Ok((authority, credentials))
    }

    #[cfg(test)]
    pub(crate) fn run_nonce(&self) -> [u8; 32] {
        self.run_nonce
    }

    fn from_controller_context(context: &Arc<CellSecurityContext>) -> Result<Self> {
        let role_verifiers = context.role_verifiers()?;
        ensure!(
            !role_verifiers.is_empty(),
            "cell registration roster requires at least one public key"
        );
        Ok(Self {
            run_nonce: context.run_nonce,
            admission_ledger: AdmissionLedger::new(context.run_nonce, &role_verifiers),
            role_verifiers,
            reply_attestor: context.reply_attestor()?,
        })
    }

    pub(crate) fn reply_attestor(&self) -> ControllerRegisterAttestor {
        self.reply_attestor.clone()
    }

    pub(crate) fn open_payload<T: DeserializeOwned>(
        &self,
        purpose: AdmissionPurpose,
        bytes: &[u8],
    ) -> Result<VerifiedPayload<T>, AdmissionRejection> {
        if bytes.len() > MAX_AUTHENTICATED_FRAME_BYTES {
            return self
                .admission_ledger
                .reject(purpose, AdmissionRejection::Oversized);
        }
        let frame: AuthenticatedFrame = match rmp_serde::from_slice(bytes) {
            Ok(frame) => frame,
            Err(_) => {
                return self
                    .admission_ledger
                    .reject(purpose, AdmissionRejection::Malformed);
            }
        };
        let verified = self.admission_ledger.open(purpose, frame)?;
        let payload = match rmp_serde::from_slice(&verified.payload) {
            Ok(payload) => payload,
            Err(_) => {
                return self
                    .admission_ledger
                    .reject(purpose, AdmissionRejection::Malformed);
            }
        };
        Ok(VerifiedPayload {
            role: verified.role,
            session_nonce: verified.session_nonce,
            peer_info: verified.peer_info,
            payload,
        })
    }

    pub(crate) fn verify_registration_frame(
        &self,
        bytes: Bytes,
    ) -> Result<VerifiedFrame, AdmissionRejection> {
        if bytes.len() > MAX_AUTHENTICATED_FRAME_BYTES {
            return self
                .admission_ledger
                .reject(AdmissionPurpose::Register, AdmissionRejection::Oversized);
        }
        let frame = match AuthenticatedFrame::decode(&bytes) {
            Ok(frame) => frame,
            Err(rejection) => {
                return self
                    .admission_ledger
                    .reject(AdmissionPurpose::Register, rejection);
            }
        };
        self.admission_ledger
            .authenticate(AdmissionPurpose::Register, frame, bytes)
    }

    pub(crate) fn commit_registration_session(&self, role: CellularRole, session_nonce: [u8; 32]) {
        self.admission_ledger.commit_session(role, session_nonce);
    }

    #[cfg(test)]
    pub(crate) fn invalid_count(&self, purpose: AdmissionPurpose) -> u64 {
        self.admission_ledger.drops.get(purpose)
    }

    #[cfg(test)]
    pub(crate) fn replay_slot_count(&self) -> usize {
        self.admission_ledger.replay_slot_count()
    }

    pub(crate) fn planned_cell_capacity(&self) -> usize {
        self.role_verifiers
            .iter()
            .filter_map(|entry| match entry.role {
                CellularRole::Cell(cell_id) => Some(cell_id as usize + 1),
                CellularRole::Aggregator { .. } => None,
            })
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn verify(
        &self,
        registration: &CellRegister,
        controller_peer: &velo::PeerInfo,
    ) -> Result<VerifiedCellRegistration> {
        let proof = registration
            .registration_proof
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cell registration proof is missing"))?;
        ensure!(
            proof.version == REGISTRATION_PROTOCOL_VERSION && proof.run_nonce == self.run_nonce,
            "cell registration proof does not belong to this run"
        );
        ensure!(
            proof.controller_binding.matches_peer(controller_peer)?,
            "cell registration names a different controller peer"
        );
        let key = self
            .role_verifiers
            .iter()
            .find(|entry| entry.role == CellularRole::Cell(registration.cell_id))
            .map(|entry| &entry.verifier)
            .ok_or_else(|| anyhow::anyhow!("cell registration id is out of range"))?;
        let signature = Signature::from_slice(&proof.signature)
            .map_err(|_| anyhow::anyhow!("cell registration proof is malformed"))?;
        key.verify(
            &registration_transcript(
                registration.cell_id,
                &registration.cell_peer,
                registration.artifact_capability_digest,
                &proof.controller_binding,
                proof.version,
                proof.run_nonce,
            ),
            &signature,
        )
        .map_err(|_| anyhow::anyhow!("cell registration proof is invalid"))?;
        Ok(VerifiedCellRegistration {
            cell_id: registration.cell_id,
        })
    }
}

impl ControllerRegisterAttestor {
    #[cfg(test)]
    pub(crate) fn mint(run_nonce: [u8; 32]) -> Result<Self> {
        let signer = SigningKey::from_bytes(&random_nonce("controller reply key")?);
        let placeholder = RoleVerifyingKey {
            role: CellularRole::Cell(0),
            verifier: signer.verifying_key(),
        };
        Ok(Self {
            context: Arc::new(CellSecurityContext::controller(
                run_nonce,
                signer,
                vec![placeholder].into_boxed_slice(),
            )?),
        })
    }

    #[cfg(test)]
    pub(crate) fn verifier(&self) -> ControllerRegisterVerifier {
        ControllerRegisterVerifier {
            run_nonce: self.context.run_nonce,
            verifying_key: match &self.context.authority {
                ProcessSecurityAuthority::Controller { signer, .. } => signer.verifying_key(),
                ProcessSecurityAuthority::Worker { .. } => unreachable!("attestor is controller"),
            },
        }
    }

    pub(crate) fn attest(
        &self,
        controller_binding: &ControllerPeerBinding,
        registration_frame: &[u8],
        reply_payload: &[u8],
    ) -> Result<ControllerRegisterAttestation> {
        let transcript = register_reply_transcript(
            controller_binding,
            registration_frame,
            reply_payload,
            REGISTRATION_PROTOCOL_VERSION,
            self.context.run_nonce,
        );
        Ok(ControllerRegisterAttestation {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.context.run_nonce,
            signature: self
                .context
                .sign_controller(&transcript)?
                .to_bytes()
                .to_vec(),
        })
    }
}

impl ControllerRegisterVerifier {
    pub(crate) fn from_public_key(run_nonce: [u8; 32], verifying_key: [u8; 32]) -> Result<Self> {
        Ok(Self {
            run_nonce,
            verifying_key: VerifyingKey::from_bytes(&verifying_key)
                .map_err(|_| anyhow::anyhow!("controller verification key is malformed"))?,
        })
    }
    pub(crate) fn verify(
        &self,
        controller_binding: &ControllerPeerBinding,
        registration_frame: &[u8],
        reply_payload: &[u8],
        attestation: &ControllerRegisterAttestation,
    ) -> Result<()> {
        ensure!(
            attestation.version == REGISTRATION_PROTOCOL_VERSION
                && attestation.run_nonce == self.run_nonce,
            "controller registration attestation does not belong to this run"
        );
        let signature = Signature::from_slice(&attestation.signature)
            .map_err(|_| anyhow::anyhow!("controller registration attestation is malformed"))?;
        self.verifying_key
            .verify(
                &register_reply_transcript(
                    controller_binding,
                    registration_frame,
                    reply_payload,
                    attestation.version,
                    attestation.run_nonce,
                ),
                &signature,
            )
            .map_err(|_| anyhow::anyhow!("controller registration attestation is invalid"))
    }
}

impl CellRegistrationCredential {
    pub(crate) fn cell_id(&self) -> u32 {
        self.cell_id
    }
    pub(crate) fn sign_register(
        &self,
        cell_peer: &[u8],
        artifact_capability_digest: Option<[u8; 32]>,
        controller_binding: ControllerPeerBinding,
    ) -> Result<CellRegistrationProof> {
        let transcript = registration_transcript(
            self.cell_id(),
            cell_peer,
            artifact_capability_digest,
            &controller_binding,
            REGISTRATION_PROTOCOL_VERSION,
            self.context.run_nonce,
        );
        Ok(CellRegistrationProof {
            version: REGISTRATION_PROTOCOL_VERSION,
            run_nonce: self.context.run_nonce,
            controller_binding,
            signature: self.context.sign_worker(&transcript)?.to_bytes().to_vec(),
        })
    }

    pub(crate) fn seal_payload<T: Serialize>(
        &self,
        purpose: AdmissionPurpose,
        peer: &velo::PeerInfo,
        payload: &T,
    ) -> Result<Vec<u8>> {
        let payload = rmp_serde::to_vec(payload)
            .map_err(|error| anyhow::anyhow!("encode authenticated payload: {error}"))?;
        let frame = self.context.seal(purpose, peer, payload)?;
        let encoded = rmp_serde::to_vec(&frame)
            .map_err(|error| anyhow::anyhow!("encode authenticated frame: {error}"))?;
        ensure!(
            encoded.len() <= MAX_AUTHENTICATED_FRAME_BYTES,
            "authenticated frame exceeds the Velo frame limit"
        );
        Ok(encoded)
    }
}

fn registration_transcript(
    cell_id: u32,
    cell_peer: &[u8],
    artifact_capability_digest: Option<[u8; 32]>,
    controller_binding: &ControllerPeerBinding,
    version: u8,
    run_nonce: [u8; 32],
) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(TRANSCRIPT_DOMAIN.len() + 1 + 32 + 4 + 32 + 1 + 32);
    transcript.extend_from_slice(TRANSCRIPT_DOMAIN);
    transcript.push(version);
    transcript.extend_from_slice(&run_nonce);
    transcript.extend_from_slice(&cell_id.to_le_bytes());
    transcript.extend_from_slice(blake3::hash(cell_peer).as_bytes());
    controller_binding.append_transcript(&mut transcript);
    match artifact_capability_digest {
        Some(digest) => {
            transcript.push(1);
            transcript.extend_from_slice(&digest);
        }
        None => transcript.push(0),
    }
    transcript
}

fn authenticated_frame_transcript(
    run_nonce: [u8; 32],
    role: CellularRole,
    purpose: AdmissionPurpose,
    session_nonce: [u8; 32],
    sequence: u64,
    peer_info: &[u8],
    payload: &[u8],
) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(
        AUTHENTICATED_FRAME_TRANSCRIPT_DOMAIN.len() + 1 + 32 + 9 + 1 + 32 + 8 + 64,
    );
    transcript.extend_from_slice(AUTHENTICATED_FRAME_TRANSCRIPT_DOMAIN);
    transcript.push(REGISTRATION_PROTOCOL_VERSION);
    transcript.extend_from_slice(&run_nonce);
    match role {
        CellularRole::Cell(cell_id) => {
            transcript.push(1);
            transcript.extend_from_slice(&cell_id.to_le_bytes());
        }
        CellularRole::Aggregator { tier, id } => {
            transcript.push(2);
            transcript.extend_from_slice(&tier.to_le_bytes());
            transcript.extend_from_slice(&id.to_le_bytes());
        }
    }
    transcript.push(purpose as u8);
    transcript.extend_from_slice(&session_nonce);
    transcript.extend_from_slice(&sequence.to_le_bytes());
    transcript.extend_from_slice(blake3::hash(peer_info).as_bytes());
    transcript.extend_from_slice(blake3::hash(payload).as_bytes());
    transcript
}

fn register_reply_transcript(
    controller_binding: &ControllerPeerBinding,
    registration_frame: &[u8],
    reply_payload: &[u8],
    version: u8,
    run_nonce: [u8; 32],
) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(REGISTER_REPLY_TRANSCRIPT_DOMAIN.len() + 1 + 32 + 96);
    transcript.extend_from_slice(REGISTER_REPLY_TRANSCRIPT_DOMAIN);
    transcript.push(version);
    transcript.extend_from_slice(&run_nonce);
    controller_binding.append_transcript(&mut transcript);
    transcript.extend_from_slice(blake3::hash(registration_frame).as_bytes());
    transcript.extend_from_slice(blake3::hash(reply_payload).as_bytes());
    transcript
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use crate::cellular::transport::CellRegister;
    use crate::cellular::transport::HANDLER_STORE_PARTITION;
    use crate::cellular::transport::connect::{ControllerPeerBinding, DialedControllerAddress};
    use crate::cellular::transport::velo_transport::CellRegistrationPlan;
    use crate::engine::artifact_shipping::{ArtifactBearer, ArtifactUploadServer};

    use super::{
        AdmissionPurpose, AdmissionRejection, AuthenticatedFrame, CellRegistrationAuthority,
        ControllerRegisterAttestor, MAX_AUTHENTICATED_FRAME_BYTES, PreparedRegistrationDropPause,
        RegistrationBegin, RegistrationLedger, ReplayWindow,
    };

    fn application_peer(marker: u8) -> velo::PeerInfo {
        velo::PeerInfo::new(
            velo::InstanceId::new_v4(),
            velo::WorkerAddress::from_encoded(vec![marker]),
        )
    }

    #[tokio::test]
    async fn prepared_drop_releases_artifact_before_publishing_ledger_vacancy() {
        let temporary = tempfile::tempdir().unwrap();
        let artifact_server = ArtifactUploadServer::start(
            "127.0.0.1:0".parse().unwrap(),
            temporary.path().join("landed"),
            HashSet::new(),
            1,
        )
        .await
        .unwrap();
        let registrar = artifact_server.registrar();
        let digest = ArtifactBearer::from_test_bytes([0x47; 32]).digest_bytes();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let authority = Arc::new(authority);
        let ledger = RegistrationLedger::new(Arc::clone(&authority), 1);
        let peer = application_peer(0x47);
        let frame = bytes::Bytes::from(
            credentials[0]
                .seal_payload(AdmissionPurpose::Register, &peer, &0_u8)
                .unwrap(),
        );
        let verified = authority.verify_registration_frame(frame).unwrap();
        let mut abandoned = match ledger.begin(&verified).unwrap() {
            RegistrationBegin::Prepared(prepared) => prepared,
            RegistrationBegin::ExactRetry { .. } => panic!("fresh slot returned a retry"),
        };
        abandoned.install_plan(CellRegistrationPlan {
            envelope: Vec::new(),
            artifact: Some(registrar.prepare(0, digest).unwrap()),
        });
        let slot_is_vacant = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        abandoned.drop_pause = Some(PreparedRegistrationDropPause {
            slot_is_vacant: Arc::clone(&slot_is_vacant),
            release: Arc::clone(&release),
        });

        std::thread::scope(|scope| {
            let dropping = scope.spawn(move || drop(abandoned));
            slot_is_vacant.wait();
            let retry = ledger.begin(&verified);
            let artifact = registrar.prepare(0, digest);
            release.wait();
            dropping.join().unwrap();
            let mut retry = match retry.unwrap() {
                RegistrationBegin::Prepared(prepared) => prepared,
                RegistrationBegin::ExactRetry { .. } => panic!("vacant slot returned a retry"),
            };
            retry.install_plan(CellRegistrationPlan {
                envelope: Vec::new(),
                artifact: Some(artifact.expect("artifact reservation must precede vacancy")),
            });
        });
        artifact_server.shutdown().await.unwrap();
    }

    fn encoded_frame_with_len(
        credential: &super::CellRegistrationCredential,
        peer: &velo::PeerInfo,
        target_len: usize,
    ) -> Vec<u8> {
        let mut payload_len = target_len - 256;
        for _ in 0..8 {
            let payload = rmp_serde::to_vec(&vec![0_u8; payload_len]).unwrap();
            let mut frame = credential
                .context
                .seal(AdmissionPurpose::StorePartition, peer, payload)
                .unwrap();
            frame.signature.fill(0);
            let encoded = rmp_serde::to_vec(&frame).unwrap();
            if encoded.len() == target_len {
                return encoded;
            }
            payload_len = if encoded.len() < target_len {
                payload_len + target_len - encoded.len()
            } else {
                payload_len - (encoded.len() - target_len)
            };
        }
        panic!("could not construct an authenticated frame of length {target_len}");
    }

    #[test]
    fn authenticated_frame_limit_fits_velo_store_partition_route() {
        const VELO_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
        const VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES: usize = 22;

        let (_, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let peer = application_peer(0x84);
        let accepted =
            encoded_frame_with_len(&credentials[0], &peer, MAX_AUTHENTICATED_FRAME_BYTES);
        assert!(AuthenticatedFrame::decode(&accepted).is_ok());
        assert!(
            accepted.len() + VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES + HANDLER_STORE_PARTITION.len()
                <= VELO_MAX_FRAME_BYTES,
            "AIPerf accepted a payload that Velo cannot frame on the longest route"
        );

        let rejected =
            encoded_frame_with_len(&credentials[0], &peer, MAX_AUTHENTICATED_FRAME_BYTES + 1);
        assert!(matches!(
            AuthenticatedFrame::decode(&rejected),
            Err(AdmissionRejection::Oversized)
        ));
    }

    #[test]
    fn authenticated_frame_binds_payload_peer_purpose_and_sequence() {
        let (authority, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let peer = application_peer(0x80);
        let frame = credentials[0]
            .context
            .seal(AdmissionPurpose::PhaseSignal, &peer, b"warmup".to_vec())
            .unwrap();

        assert!(
            authority
                .admission_ledger
                .open(AdmissionPurpose::PhaseSignal, frame.clone())
                .is_ok()
        );
        assert!(matches!(
            authority
                .admission_ledger
                .open(AdmissionPurpose::PhaseSignal, frame.clone()),
            Err(AdmissionRejection::Replay)
        ));

        let mut tampered_payload = frame.clone();
        tampered_payload.payload = b"profiling".to_vec();
        assert!(
            authority
                .admission_ledger
                .open(AdmissionPurpose::PhaseSignal, tampered_payload)
                .is_err()
        );

        let mut tampered_peer = frame.clone();
        tampered_peer.peer_info = rmp_serde::to_vec(&application_peer(0x81)).unwrap();
        assert!(
            authority
                .admission_ledger
                .open(AdmissionPurpose::PhaseSignal, tampered_peer)
                .is_err()
        );

        assert!(
            authority
                .admission_ledger
                .open(AdmissionPurpose::Heartbeat, frame)
                .is_err()
        );
    }

    #[test]
    fn replay_window_handles_zero_sixty_three_sixty_four_and_max() {
        let mut window = ReplayWindow::default();
        assert!(window.accept(0));
        assert!(window.accept(64));
        assert!(window.accept(1));
        assert!(!window.accept(0));

        let mut high = ReplayWindow::default();
        assert!(high.accept(u64::MAX));
        assert!(high.accept(u64::MAX - 63));
        assert!(!high.accept(u64::MAX - 64));
        assert!(!high.accept(u64::MAX));
    }

    #[test]
    fn concurrent_sealing_allocates_unique_sequences() {
        let (_, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let context = Arc::clone(&credentials[0].context);
        let peer = application_peer(0x84);
        let mut threads = Vec::new();
        for _ in 0..8 {
            let context = Arc::clone(&context);
            let peer = peer.clone();
            threads.push(std::thread::spawn(move || {
                (0..128)
                    .map(|_| {
                        context
                            .seal(AdmissionPurpose::Heartbeat, &peer, Vec::new())
                            .unwrap()
                            .sequence
                    })
                    .collect::<Vec<_>>()
            }));
        }
        let sequences = threads
            .into_iter()
            .flat_map(|thread| thread.join().unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(sequences.len(), 1024);
    }

    fn controller_binding() -> (velo::PeerInfo, ControllerPeerBinding) {
        let peer = velo::PeerInfo::new(
            velo::InstanceId::new_v4(),
            velo::WorkerAddress::from_encoded(vec![0x80]),
        );
        let binding = ControllerPeerBinding::new(
            &peer,
            DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
        )
        .unwrap();
        (peer, binding)
    }

    #[test]
    fn registration_proof_binds_controller_cell_peer_and_capability_digest() {
        let (authority, credentials) = CellRegistrationAuthority::mint(2).unwrap();
        let credential = &credentials[1];
        let peer = b"encoded-peer";
        let digest = [0x11; 32];
        let (controller_peer, binding) = controller_binding();
        let proof = credential
            .sign_register(peer, Some(digest), binding)
            .unwrap();
        let register = CellRegister {
            cell_id: 1,
            cell_peer: peer.to_vec(),
            artifact_capability_digest: Some(digest),
            registration_proof: Some(proof),
        };

        assert!(authority.verify(&register, &controller_peer).is_ok());
        let changed_controller_peer = velo::PeerInfo::new(
            controller_peer.instance_id(),
            velo::WorkerAddress::from_encoded(vec![0x81]),
        );
        assert!(
            authority
                .verify(&register, &changed_controller_peer)
                .is_err()
        );
        let mut changed_digest = register.clone();
        changed_digest.artifact_capability_digest = Some([0x22; 32]);
        assert!(authority.verify(&changed_digest, &controller_peer).is_err());
    }

    // Catches a cell signature that authenticates only structural peer equality,
    // rather than the exact controller binding carried by the proof.
    #[test]
    fn registration_signature_rejects_consistently_replaced_controller_binding() {
        let (authority, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let (controller_peer, binding) = controller_binding();
        let proof = credentials[0]
            .sign_register(b"cell-peer", None, binding)
            .unwrap();
        let mut register = CellRegister {
            cell_id: 0,
            cell_peer: b"cell-peer".to_vec(),
            artifact_capability_digest: None,
            registration_proof: Some(proof),
        };
        let changed_controller_peer = velo::PeerInfo::new(
            controller_peer.instance_id(),
            velo::WorkerAddress::from_encoded(vec![0x81]),
        );
        register
            .registration_proof
            .as_mut()
            .unwrap()
            .controller_binding = ControllerPeerBinding::new(
            &changed_controller_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
        )
        .unwrap();

        assert!(
            authority
                .verify(&register, &changed_controller_peer)
                .is_err(),
            "the original signature must not authorize a replacement binding"
        );
    }

    // Catches a cell signature that omits only the resolved dial target.
    #[test]
    fn registration_signature_rejects_dial_only_replacement() {
        let (authority, credentials) = CellRegistrationAuthority::mint(1).unwrap();
        let (controller_peer, binding) = controller_binding();
        let proof = credentials[0]
            .sign_register(b"cell-peer", None, binding)
            .unwrap();
        let mut register = CellRegister {
            cell_id: 0,
            cell_peer: b"cell-peer".to_vec(),
            artifact_capability_digest: None,
            registration_proof: Some(proof),
        };
        register
            .registration_proof
            .as_mut()
            .unwrap()
            .controller_binding = ControllerPeerBinding::new(
            &controller_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9501".parse().unwrap()),
        )
        .unwrap();

        assert!(
            authority.verify(&register, &controller_peer).is_err(),
            "the original cell signature must not authorize a changed dial target"
        );
    }

    // Catches controller attestations that omit any exact registration-reply input.
    #[test]
    fn controller_attestation_covers_binding_registration_frame_and_reply_payload() {
        let attestor = ControllerRegisterAttestor::mint([0x42; 32]).unwrap();
        let verifier = attestor.verifier();
        let (_, binding) = controller_binding();
        let changed_peer = velo::PeerInfo::new(
            velo::InstanceId::new_v4(),
            velo::WorkerAddress::from_encoded(vec![0x82]),
        );
        let changed_binding = ControllerPeerBinding::new(
            &changed_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9501".parse().unwrap()),
        )
        .unwrap();
        let registration_frame = b"exact registration frame";
        let reply_payload = b"exact reply payload";
        let attestation = attestor
            .attest(&binding, registration_frame, reply_payload)
            .unwrap();

        assert!(
            verifier
                .verify(
                    &changed_binding,
                    registration_frame,
                    reply_payload,
                    &attestation,
                )
                .is_err(),
            "the controller binding must be covered"
        );
        assert!(
            verifier
                .verify(
                    &binding,
                    b"changed registration frame",
                    reply_payload,
                    &attestation,
                )
                .is_err(),
            "the exact registration frame must be covered"
        );
        assert!(
            verifier
                .verify(
                    &binding,
                    registration_frame,
                    b"changed reply payload",
                    &attestation,
                )
                .is_err(),
            "the exact reply payload must be covered"
        );
    }

    // Catches a controller attestation that omits only the resolved dial target.
    #[test]
    fn controller_attestation_rejects_dial_only_replacement() {
        let attestor = ControllerRegisterAttestor::mint([0x42; 32]).unwrap();
        let verifier = attestor.verifier();
        let (controller_peer, binding) = controller_binding();
        let changed_binding = ControllerPeerBinding::new(
            &controller_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9501".parse().unwrap()),
        )
        .unwrap();
        let attestation = attestor
            .attest(&binding, b"registration", b"reply")
            .unwrap();

        assert!(
            verifier
                .verify(&changed_binding, b"registration", b"reply", &attestation,)
                .is_err(),
            "the original controller attestation must not authorize a changed dial target"
        );
    }
}
