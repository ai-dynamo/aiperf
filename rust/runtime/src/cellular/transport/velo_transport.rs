// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The velo-backed cell↔controller transport.
//!
//! Realizes the [`CellClient`] / [`ControllerTransport`] seam over the Velo 0.5
//! messaging framework. Six named handlers on the controller carry the whole
//! protocol:
//!
//! - [`HANDLER_REGISTER`] (unary): a cell sends its [`CellRegister`] (its own
//!   `PeerInfo` + `cell_id`); the controller `register_peer`s it (so replies and
//!   later messages route back) and returns that cell's `RegisterReply`, whose
//!   `envelope` is the cell's sliced execute envelope.
//! - [`HANDLER_PREFLIGHT`] (fire-and-forget): a cell's
//!   [`CellMessage::Preflight`] capability result, which ticks the controller's
//!   pre-START preflight barrier instead of reaching the application channel.
//! - [`HANDLER_HEARTBEAT`] (fire-and-forget): a cell's terminal
//!   [`CellMessage::Heartbeat`].
//! - [`HANDLER_PHASE_SIGNAL`] (fire-and-forget): a cell's named
//!   [`CellMessage::PhaseSignal`] barrier notification.
//! - [`HANDLER_PARTITION`] (unary): a cell's final [`RecordsShardPartition`]; the
//!   reply is a [`CellAck`].
//! - [`HANDLER_STORE_PARTITION`] (unary): a cell's final
//!   [`ColumnStorePartition`](crate::cellular::shard::ColumnStorePartition); the
//!   reply is a [`CellAck`].
//!
//! All bodies are `rmp-serde` carried as velo **raw** payloads (see the module
//! docs on `super`) so t-digest `+inf` and NaN metric values survive the wire.

use std::sync::Arc;

use anyhow::ensure;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::sync::mpsc;
use velo::{Context, EventHandle, Handler, PeerInfo, Velo};

#[cfg(test)]
use super::CellPhaseSignal;
#[cfg(test)]
use super::connect::DialedControllerAddress;
use super::connect::{
    ConnectedController, ControllerPeerBinding, RegistrationDeadline, await_handler_until,
};
use super::{
    ArtifactChannelServerConfig, CellAck, CellClient, CellMessage, CellPartitionShip, CellRegister,
    CellStorePartitionShip, CellTransportError, ControllerTransport, HANDLER_HEARTBEAT,
    HANDLER_PARTITION, HANDLER_PHASE_SIGNAL, HANDLER_PREFLIGHT, HANDLER_REGISTER,
    HANDLER_STORE_PARTITION,
};
use crate::engine::artifact_shipping::ArtifactRegistrationPlan;
use crate::engine::cellular_registration::{
    AdmissionPurpose, AuthenticatedFrame, CellRegistrationAuthority, CellRegistrationCredential,
    ControllerRegisterAttestation, ControllerRegisterAttestor, ControllerRegisterVerifier,
    RegistrationBegin, RegistrationLedger, VerifiedFrame,
};

pub(crate) struct CellRegistrationPlan {
    pub(crate) envelope: Vec<u8>,
    pub(crate) artifact: Option<ArtifactRegistrationPlan>,
}

pub(crate) type PlanRegistration =
    Arc<dyn Fn(&VerifiedFrame) -> anyhow::Result<Option<CellRegistrationPlan>> + Send + Sync>;

/// The controller's reply to a cell's registration: the cell's sliced execute
/// envelope plus the handle of the run-wide **START** event. The cell awaits that
/// event before dispatching, so every cell begins the benchmark together once the
/// controller has seen all `cell_count` registrations and every cell's preflight
/// report (synchronized start).
#[derive(Debug, Clone)]
pub struct RegisterReply {
    /// The cell's sliced execute envelope (protocol-v2 JSON bytes).
    pub envelope: Vec<u8>,
    /// The controller-owned START event handle the cell awaits before dispatching.
    pub start_event: EventHandle,
    /// The public artifact TLS certificate, when cross-host transfer is enabled.
    pub artifact_channel: Option<ArtifactChannelServerConfig>,
    /// Controller signature binding this reply to the connected controller and request.
    attestation: ControllerRegisterAttestation,
    registration_frame: Bytes,
    reply_payload: Bytes,
}

#[derive(Serialize, Deserialize)]
struct RegisterReplyPayload {
    envelope: Vec<u8>,
    start_event: EventHandle,
    artifact_channel: Option<ArtifactChannelServerConfig>,
}

#[derive(Serialize)]
struct RegisterReplyPayloadRef<'a> {
    envelope: &'a [u8],
    start_event: EventHandle,
    artifact_channel: &'a Option<ArtifactChannelServerConfig>,
}

#[derive(Serialize, Deserialize)]
struct AttestedRegisterReply {
    payload: Vec<u8>,
    attestation: ControllerRegisterAttestation,
}

fn encode_attested_register_reply(
    payload: Vec<u8>,
    attestation: ControllerRegisterAttestation,
) -> Result<Bytes, CellTransportError> {
    rmp_serde::to_vec(&AttestedRegisterReply {
        payload,
        attestation,
    })
    .map(Bytes::from)
    .map_err(encode)
}

#[allow(clippy::too_many_arguments)]
fn execute_registration_transaction<A, E, R, N>(
    registration_authority: &CellRegistrationAuthority,
    registration_ledger: &RegistrationLedger,
    plan_registration: &PlanRegistration,
    controller_peer: &PeerInfo,
    start_event: EventHandle,
    registration_frame: Bytes,
    attest: A,
    encode_reply: E,
    register_peer: R,
    notify_registered: N,
) -> anyhow::Result<Bytes>
where
    A: FnOnce(
        &ControllerPeerBinding,
        &[u8],
        &[u8],
    ) -> anyhow::Result<ControllerRegisterAttestation>,
    E: FnOnce(Vec<u8>, ControllerRegisterAttestation) -> anyhow::Result<Bytes>,
    R: FnOnce(PeerInfo) -> anyhow::Result<()>,
    N: FnOnce(),
{
    let verified = registration_authority
        .verify_registration_frame(registration_frame)
        .map_err(anyhow::Error::new)?;
    if let Some(reply) = registration_ledger
        .cached_reply(&verified)
        .map_err(anyhow::Error::new)?
    {
        return Ok(reply);
    }
    let register: CellRegister = verified.decode_payload()?;
    ensure!(
        verified.role() == crate::engine::cellular_bootstrap::CellularRole::Cell(register.cell_id)
            && verified.peer_info() == register.cell_peer,
        "registration identity mismatch"
    );
    let registration = registration_authority.verify(&register, controller_peer)?;
    ensure!(
        registration.cell_id() == register.cell_id,
        "registration cell mismatch"
    );
    let peer: PeerInfo = rmp_serde::from_slice(verified.peer_info())
        .map_err(|_| anyhow::anyhow!("registration peer is malformed"))?;
    let mut prepared = match registration_ledger
        .begin(&verified)
        .map_err(anyhow::Error::new)?
    {
        RegistrationBegin::ExactRetry { reply } => return Ok(reply),
        RegistrationBegin::Prepared(prepared) => prepared,
    };
    let plan = plan_registration(&verified)?
        .ok_or_else(|| anyhow::anyhow!("no launch plan for cell {}", register.cell_id))?;
    prepared.install_plan(plan);
    let artifact_channel = prepared.artifact_channel();
    let reply_payload = encode_reply_payload(prepared.envelope(), start_event, &artifact_channel)
        .map_err(anyhow::Error::new)?;
    let binding = &register
        .registration_proof
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("verified registration proof is missing"))?
        .controller_binding;
    let attestation = attest(
        binding,
        verified.encoded().as_ref(),
        reply_payload.as_slice(),
    )?;
    let reply = encode_reply(reply_payload, attestation)?;
    // Trust boundary: registered benchmark cells are trusted routing-plane participants after
    // authenticated admission. There is deliberately no per-push route authenticity or replay protection;
    // preserve normal replay/live delivery and revisit if cellular topology admits untrusted participants.
    register_peer(peer)?;
    let commit = prepared.commit(reply.clone());
    if commit.should_advance_barrier() {
        notify_registered();
    }
    Ok(reply)
}

fn encode_reply_payload(
    envelope: &[u8],
    start_event: EventHandle,
    artifact_channel: &Option<ArtifactChannelServerConfig>,
) -> Result<Vec<u8>, CellTransportError> {
    rmp_serde::to_vec(&RegisterReplyPayloadRef {
        envelope,
        start_event,
        artifact_channel,
    })
    .map_err(encode)
}

pub(crate) fn decode_reply(
    bytes: &[u8],
    registration_frame: Bytes,
) -> Result<RegisterReply, CellTransportError> {
    let wire: AttestedRegisterReply = rmp_serde::from_slice(bytes).map_err(decode)?;
    let payload: RegisterReplyPayload = rmp_serde::from_slice(&wire.payload).map_err(decode)?;
    Ok(RegisterReply {
        envelope: payload.envelope,
        start_event: payload.start_event,
        artifact_channel: payload.artifact_channel,
        attestation: wire.attestation,
        registration_frame,
        reply_payload: Bytes::from(wire.payload),
    })
}

pub(crate) fn verify_reply(
    controller: &ConnectedController,
    verifier: &ControllerRegisterVerifier,
    registration: &CellRegister,
    reply: &RegisterReply,
    cell_id: u32,
) -> Result<(), CellTransportError> {
    if registration.cell_id != cell_id {
        return Err(CellTransportError::Authentication(
            "controller reply has the wrong cell identity",
        ));
    }
    let proof =
        registration
            .registration_proof
            .as_ref()
            .ok_or(CellTransportError::Authentication(
                "controller binding is missing",
            ))?;
    if proof.controller_binding != controller.binding()? {
        return Err(CellTransportError::Authentication(
            "controller binding does not match the connection",
        ));
    }
    let encoded_payload =
        encode_reply_payload(&reply.envelope, reply.start_event, &reply.artifact_channel)?;
    if encoded_payload.as_slice() != reply.reply_payload.as_ref() {
        return Err(CellTransportError::Authentication(
            "controller reply payload is inconsistent",
        ));
    }
    let encoded_registration = rmp_serde::to_vec(registration).map_err(encode)?;
    let registration_frame =
        AuthenticatedFrame::decode(&reply.registration_frame).map_err(|_| {
            CellTransportError::Authentication("controller registration frame is invalid")
        })?;
    if registration_frame.role() != crate::engine::cellular_bootstrap::CellularRole::Cell(cell_id)
        || registration_frame.peer_info() != registration.cell_peer
        || encoded_registration.as_slice() != registration_frame.payload()
    {
        return Err(CellTransportError::Authentication(
            "controller registration frame is inconsistent",
        ));
    }
    verifier
        .verify(
            &proof.controller_binding,
            &reply.registration_frame,
            &reply.reply_payload,
            &reply.attestation,
        )
        .map_err(|_| CellTransportError::Authentication("controller reply attestation is invalid"))
}

fn encode(error: impl std::fmt::Display) -> CellTransportError {
    CellTransportError::Encode(error.to_string())
}
fn decode(error: impl std::fmt::Display) -> CellTransportError {
    CellTransportError::Decode(error.to_string())
}
fn io(error: impl std::fmt::Display) -> CellTransportError {
    CellTransportError::Io(error.to_string())
}

/// The controller's velo endpoint: registers the control/partition handlers and exposes a
/// merged [`ControllerTransport::recv`] stream of every cell's decoded messages.
pub struct VeloControllerTransport {
    /// Held so the velo instance (and thus its registered handlers) outlives the
    /// transport; dropping it tears the messaging plane down.
    _velo: Arc<Velo>,
    receiver: mpsc::Receiver<Result<CellMessage, CellTransportError>>,
    /// Notified once every `cell_count` cell has registered — the controller then
    /// awaits the preflight barrier before triggering the START event
    /// (synchronized start).
    all_registered: Arc<Notify>,
    preflight: Arc<crate::graph::supplement::GraphCellPreflightBarrier>,
}

impl VeloControllerTransport {
    /// Register the register/control/partition handlers on `velo` and return the
    /// controller transport. `plan_registration` validates each registering cell
    /// and prepares its launch material; `start_event` is the run-wide START handle
    /// returned to each cell. The barrier fires once every cell has registered.
    pub(crate) fn bind_controller(
        velo: Arc<Velo>,
        registration_authority: Arc<CellRegistrationAuthority>,
        plan_registration: PlanRegistration,
        cell_count: u32,
        start_event: EventHandle,
    ) -> Result<Self, CellTransportError> {
        let reply_attestor = registration_authority.reply_attestor();
        Self::bind_controller_inner(
            velo,
            registration_authority,
            reply_attestor,
            plan_registration,
            cell_count,
            start_event,
        )
    }

    fn bind_controller_inner(
        velo: Arc<Velo>,
        registration_authority: Arc<CellRegistrationAuthority>,
        reply_attestor: ControllerRegisterAttestor,
        plan_registration: PlanRegistration,
        cell_count: u32,
        start_event: EventHandle,
    ) -> Result<Self, CellTransportError> {
        let (sender, receiver) = mpsc::channel(1024);
        let all_registered = Arc::new(Notify::new());
        let preflight = Arc::new(crate::graph::supplement::GraphCellPreflightBarrier::new(
            cell_count,
        ));
        let registration_ledger =
            RegistrationLedger::new(Arc::clone(&registration_authority), cell_count);
        let register_authority = registration_authority.clone();
        let partition_authority = registration_authority.clone();
        let store_partition_authority = registration_authority.clone();
        // `_hello` publishes the messenger peer. `Velo::peer_info()` augments it
        // with streaming addresses the connected cell never observed.
        let controller_peer = velo.messenger().peer_info();

        // register (unary): learn the cell, count it toward the start barrier, and
        // return its launch material + the START handle it must await before dispatching.
        let reg_notify = all_registered.clone();
        let registration_state = registration_ledger.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_REGISTER, move |ctx: Context| {
                let plan_registration = plan_registration.clone();
                let registration_authority = register_authority.clone();
                let registration_ledger = registration_state.clone();
                let reg_notify = reg_notify.clone();
                let reply_attestor = reply_attestor.clone();
                let controller_peer = controller_peer.clone();
                async move {
                    execute_registration_transaction(
                        &registration_authority,
                        &registration_ledger,
                        &plan_registration,
                        &controller_peer,
                        start_event,
                        ctx.payload.clone(),
                        |binding, registration, payload| {
                            reply_attestor.attest(binding, registration, payload)
                        },
                        |payload, attestation| {
                            encode_attested_register_reply(payload, attestation)
                                .map_err(anyhow::Error::new)
                        },
                        |peer| {
                            ctx.msg
                                .register_peer(peer)
                                .map_err(|error| anyhow::anyhow!("register_peer cell: {error}"))
                        },
                        || reg_notify.notify_one(),
                    )
                    .map(Some)
                    .map_err(|_| anyhow::anyhow!("AdmissionRejected"))
                }
            })
            .build(),
        )
        .map_err(io)?;

        let preflight_barrier = preflight.clone();
        let preflight_authority = registration_authority.clone();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_PREFLIGHT, move |ctx: Context| {
                let preflight_barrier = preflight_barrier.clone();
                let registration_authority = preflight_authority.clone();
                async move {
                    let Ok(opened) = registration_authority
                        .open_payload::<CellMessage>(AdmissionPurpose::Preflight, &ctx.payload)
                    else {
                        return Ok(());
                    };
                    let role = opened.role();
                    if let CellMessage::Preflight { cell_id, result } = opened.into_payload()
                        && role == crate::engine::cellular_bootstrap::CellularRole::Cell(cell_id)
                    {
                        preflight_barrier.report(cell_id, result);
                    }
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(io)?;

        // heartbeat (fire-and-forget): push the decoded CellMessage::Heartbeat.
        let heartbeat_sender = sender.clone();
        let heartbeat_authority = registration_authority.clone();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_HEARTBEAT, move |ctx: Context| {
                let sender = heartbeat_sender.clone();
                let registration_authority = heartbeat_authority.clone();
                async move {
                    let Ok(opened) = registration_authority
                        .open_payload::<CellMessage>(AdmissionPurpose::Heartbeat, &ctx.payload)
                    else {
                        return Ok(());
                    };
                    let role = opened.role();
                    let message = opened.into_payload();
                    if let CellMessage::Heartbeat { cell_id, .. } = &message
                        && role == crate::engine::cellular_bootstrap::CellularRole::Cell(*cell_id)
                    {
                        let _ = sender.send(Ok(message)).await;
                    }
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(io)?;

        // phase_signal (fire-and-forget): push the decoded CellMessage::PhaseSignal.
        let phase_signal_sender = sender.clone();
        let phase_signal_authority = registration_authority.clone();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_PHASE_SIGNAL, move |ctx: Context| {
                let sender = phase_signal_sender.clone();
                let registration_authority = phase_signal_authority.clone();
                async move {
                    let Ok(opened) = registration_authority
                        .open_payload::<CellMessage>(AdmissionPurpose::PhaseSignal, &ctx.payload)
                    else {
                        return Ok(());
                    };
                    let role = opened.role();
                    let message = opened.into_payload();
                    if let CellMessage::PhaseSignal { cell_id, .. } = &message
                        && role == crate::engine::cellular_bootstrap::CellularRole::Cell(*cell_id)
                    {
                        let _ = sender.send(Ok(message)).await;
                    }
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(io)?;

        // partition (unary): push the decoded partition, reply with an ack.
        let partition_sender = sender.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_PARTITION, move |ctx: Context| {
                let sender = partition_sender.clone();
                let registration_authority = partition_authority.clone();
                async move {
                    let opened = registration_authority
                        .open_payload::<CellPartitionShip>(
                            AdmissionPurpose::Partition,
                            &ctx.payload,
                        )
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    let (role, _, authenticated_peer, ship) = opened.into_parts();
                    ensure!(
                        ship.partition.cell_id() == ship.cell_id
                            && role
                                == crate::engine::cellular_bootstrap::CellularRole::Cell(
                                    ship.cell_id,
                                ),
                        "AdmissionRejected"
                    );
                    let peer: PeerInfo = rmp_serde::from_slice(&authenticated_peer)
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    ctx.msg
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer shipper: {error}"))?;
                    let _ = sender
                        .send(Ok(CellMessage::Partition(ship.partition)))
                        .await;
                    let ack = rmp_serde::to_vec(&CellAck { ok: true })
                        .map_err(|error| anyhow::anyhow!("encode ack: {error}"))?;
                    Ok(Some(Bytes::from(ack)))
                }
            })
            .build(),
        )
        .map_err(io)?;

        // store_partition is the exact-fold sibling of `partition`; push
        // the decoded folded column-store partition, reply with an ack.
        let store_partition_sender = sender;
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_STORE_PARTITION, move |ctx: Context| {
                let sender = store_partition_sender.clone();
                let registration_authority = store_partition_authority.clone();
                async move {
                    let opened = registration_authority
                        .open_payload::<CellStorePartitionShip>(
                            AdmissionPurpose::StorePartition,
                            &ctx.payload,
                        )
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    let (role, _, authenticated_peer, ship) = opened.into_parts();
                    ensure!(
                        ship.partition.cell_id() == ship.cell_id
                            && role
                                == crate::engine::cellular_bootstrap::CellularRole::Cell(
                                    ship.cell_id,
                                ),
                        "AdmissionRejected"
                    );
                    let peer: PeerInfo = rmp_serde::from_slice(&authenticated_peer)
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    ctx.msg
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer store shipper: {error}"))?;
                    let _ = sender
                        .send(Ok(CellMessage::StorePartition(Box::new(ship.partition))))
                        .await;
                    let ack = rmp_serde::to_vec(&CellAck { ok: true })
                        .map_err(|error| anyhow::anyhow!("encode store ack: {error}"))?;
                    Ok(Some(Bytes::from(ack)))
                }
            })
            .build(),
        )
        .map_err(io)?;

        Ok(Self {
            _velo: velo,
            receiver,
            all_registered,
            preflight,
        })
    }

    /// Resolves once every `cell_count` cell has registered. The controller awaits
    /// this (with a deadline) before triggering the START event.
    pub async fn await_all_registered(&self) {
        self.all_registered.notified().await;
    }

    /// Wait for every cell to pass its envelope-local preflight before START.
    pub async fn await_all_preflight(
        &self,
    ) -> Result<(), crate::graph::supplement::GraphSupplementError> {
        self.preflight.await_all().await
    }
}

#[async_trait::async_trait]
impl ControllerTransport for VeloControllerTransport {
    async fn recv(&mut self) -> Result<Option<CellMessage>, CellTransportError> {
        match self.receiver.recv().await {
            Some(Ok(message)) => Ok(Some(message)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }
}

/// A cell's velo client to the controller. Built from a velo instance and the
/// controller's resolved [`PeerInfo`] (obtained via the bootstrap in
/// [`connect`](crate::cellular::transport::connect)); [`register`](Self::register)
/// fetches the cell's launch envelope, and [`CellClient::send`] ships control messages and
/// the final partition.
pub struct VeloCellClient {
    velo: Arc<Velo>,
    controller: PeerInfo,
    registration_peer: PeerInfo,
    credential: Option<Arc<CellRegistrationCredential>>,
}

impl VeloCellClient {
    /// Register the controller peer so the cell can address it, and return the client.
    pub fn connect(velo: Arc<Velo>, controller: PeerInfo) -> Result<Self, CellTransportError> {
        velo.register_peer(controller.clone()).map_err(io)?;
        let registration_peer = velo.peer_info();
        Ok(Self {
            velo,
            controller,
            registration_peer,
            credential: None,
        })
    }

    /// Build a client with the per-cell credential needed for authenticated frames.
    pub(crate) fn connect_authenticated(
        velo: Arc<Velo>,
        controller: PeerInfo,
        credential: Arc<CellRegistrationCredential>,
    ) -> Result<Self, CellTransportError> {
        let mut client = Self::connect(velo, controller)?;
        client.credential = Some(credential);
        Ok(client)
    }

    /// Send the registration request and return the controller's [`RegisterReply`]
    /// (the cell's sliced execute envelope + the START event handle to await).
    pub async fn register(&self, cell_id: u32) -> Result<RegisterReply, CellTransportError> {
        self.register_with_artifact_capability(cell_id, None).await
    }

    /// Register with the digest of the cell-local artifact bearer.
    pub async fn register_with_artifact_capability(
        &self,
        cell_id: u32,
        artifact_capability_digest: Option<[u8; 32]>,
    ) -> Result<RegisterReply, CellTransportError> {
        self.register_with_registration_proof(cell_id, artifact_capability_digest, None)
            .await
    }

    /// Register with controller-verified launch proof material.
    pub(crate) async fn register_with_registration_proof(
        &self,
        cell_id: u32,
        artifact_capability_digest: Option<[u8; 32]>,
        registration_proof: Option<super::CellRegistrationProof>,
    ) -> Result<RegisterReply, CellTransportError> {
        self.send_registration(
            cell_id,
            rmp_serde::to_vec(&self.registration_peer).map_err(encode)?,
            artifact_capability_digest,
            registration_proof,
        )
        .await
    }

    async fn send_registration(
        &self,
        cell_id: u32,
        cell_peer: Vec<u8>,
        artifact_capability_digest: Option<[u8; 32]>,
        registration_proof: Option<super::CellRegistrationProof>,
    ) -> Result<RegisterReply, CellTransportError> {
        let registration = CellRegister {
            cell_id,
            cell_peer,
            artifact_capability_digest,
            registration_proof,
        };
        self.register_request(&registration).await
    }

    async fn send_registration_frame(
        &self,
        registration_frame: Bytes,
    ) -> Result<RegisterReply, CellTransportError> {
        let reply: Bytes = self
            .velo
            .unary(HANDLER_REGISTER)
            .map_err(io)?
            .raw_payload(registration_frame.clone())
            .instance(self.controller.instance_id())
            .send()
            .await
            .map_err(io)?;
        decode_reply(&reply, registration_frame)
    }

    /// Build the exact signed request whose reply will be verified by the cell.
    #[cfg(test)]
    pub(crate) fn signed_registration(
        &self,
        cell_id: u32,
        artifact_capability_digest: Option<[u8; 32]>,
        credential: &CellRegistrationCredential,
    ) -> Result<CellRegister, CellTransportError> {
        let binding = ControllerPeerBinding::new(
            &self.controller,
            DialedControllerAddress::Tcp("127.0.0.1:0".parse().map_err(io)?),
        )?;
        self.signed_registration_for_controller(
            cell_id,
            artifact_capability_digest,
            credential,
            binding,
        )
    }

    /// Build a request signed for the exact connected controller and dial target.
    pub(crate) fn signed_registration_for_controller(
        &self,
        cell_id: u32,
        artifact_capability_digest: Option<[u8; 32]>,
        credential: &CellRegistrationCredential,
        controller_binding: ControllerPeerBinding,
    ) -> Result<CellRegister, CellTransportError> {
        let cell_peer = rmp_serde::to_vec(&self.registration_peer).map_err(encode)?;
        let registration_proof = credential
            .sign_register(&cell_peer, artifact_capability_digest, controller_binding)
            .map_err(encode)?;
        Ok(CellRegister {
            cell_id,
            cell_peer,
            artifact_capability_digest,
            registration_proof: Some(registration_proof),
        })
    }

    /// Send a caller-retained request so it can verify the signed reply transcript.
    pub(crate) async fn register_request(
        &self,
        registration: &CellRegister,
    ) -> Result<RegisterReply, CellTransportError> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            CellTransportError::Authentication("cell registration credential is missing")
        })?;
        if credential.cell_id() != registration.cell_id {
            return Err(CellTransportError::Authentication(
                "cell registration credential has the wrong identity",
            ));
        }
        let frame = credential
            .seal_payload(
                AdmissionPurpose::Register,
                &self.registration_peer,
                registration,
            )
            .map_err(encode)?;
        self.send_registration_frame(Bytes::from(frame)).await
    }

    /// Await typed handler publication and send under the caller's one deadline.
    pub(crate) async fn register_request_until(
        &self,
        controller: &ConnectedController,
        registration: &CellRegister,
        deadline: RegistrationDeadline,
    ) -> Result<RegisterReply, CellTransportError> {
        await_handler_until(&self.velo, controller, HANDLER_REGISTER, deadline).await?;
        tokio::time::timeout_at(deadline.instant(), self.register_request(registration))
            .await
            .map_err(|_| {
                CellTransportError::Io("controller registration deadline elapsed".to_owned())
            })?
    }

    /// Sign the exact peer bytes and capability digest before registration.
    #[cfg(test)]
    pub(crate) async fn register_with_credential(
        &self,
        cell_id: u32,
        artifact_capability_digest: Option<[u8; 32]>,
        credential: &CellRegistrationCredential,
    ) -> Result<RegisterReply, CellTransportError> {
        let registration =
            self.signed_registration(cell_id, artifact_capability_digest, credential)?;
        let frame = credential
            .seal_payload(
                AdmissionPurpose::Register,
                &self.registration_peer,
                &registration,
            )
            .map_err(encode)?;
        self.send_registration_frame(Bytes::from(frame)).await
    }

    /// Block until the controller triggers the run-wide START event (a synchronized
    /// start: every cell resumes together once all cells have registered and passed
    /// preflight). A poisoned event (the controller aborted before starting)
    /// surfaces as an error.
    pub async fn await_start(&self, start_event: EventHandle) -> Result<(), CellTransportError> {
        self.velo
            .event_manager()
            .awaiter(start_event)
            .map_err(io)?
            .await
            .map_err(io)
    }

    fn seal_payload<T: Serialize>(
        &self,
        purpose: AdmissionPurpose,
        payload: &T,
    ) -> Result<Bytes, CellTransportError> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            CellTransportError::Authentication("cell application credential is missing")
        })?;
        credential
            .seal_payload(purpose, &self.registration_peer, payload)
            .map(Bytes::from)
            .map_err(encode)
    }
}

#[async_trait::async_trait]
impl CellClient for VeloCellClient {
    async fn send(&mut self, message: &CellMessage) -> Result<(), CellTransportError> {
        match message {
            CellMessage::Preflight { .. } => {
                let body = self.seal_payload(AdmissionPurpose::Preflight, message)?;
                self.velo
                    .am_send(HANDLER_PREFLIGHT)
                    .map_err(io)?
                    .raw_payload(body)
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
            }
            CellMessage::Heartbeat { .. } => {
                // Fire-and-forget: no ack, so the controller needs no return route.
                let body = self.seal_payload(AdmissionPurpose::Heartbeat, message)?;
                self.velo
                    .am_send(HANDLER_HEARTBEAT)
                    .map_err(io)?
                    .raw_payload(body)
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
            }
            CellMessage::PhaseSignal { .. } => {
                let body = self.seal_payload(AdmissionPurpose::PhaseSignal, message)?;
                self.velo
                    .am_send(HANDLER_PHASE_SIGNAL)
                    .map_err(io)?
                    .raw_payload(body)
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
            }
            CellMessage::Partition(partition) => {
                // Ship carries this instance's PeerInfo so the controller can ack back.
                let credential = self.credential.as_ref().ok_or_else(|| {
                    CellTransportError::Io(
                        "partition shipping requires an authenticated cell credential".to_owned(),
                    )
                })?;
                if credential.cell_id() != partition.cell_id() {
                    return Err(CellTransportError::Io(
                        "partition credential does not match the cell identity".to_owned(),
                    ));
                }
                let ship = CellPartitionShip {
                    cell_id: partition.cell_id(),
                    partition: partition.clone(),
                };
                let body = self.seal_payload(AdmissionPurpose::Partition, &ship)?;
                let reply: Bytes = self
                    .velo
                    .unary(HANDLER_PARTITION)
                    .map_err(io)?
                    .raw_payload(body)
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
                let ack: CellAck = rmp_serde::from_slice(&reply).map_err(decode)?;
                if !ack.ok {
                    return Err(CellTransportError::Io(
                        "controller nacked partition".to_owned(),
                    ));
                }
            }
            CellMessage::StorePartition(partition) => {
                // Same unary+ack+peer path as `Partition`, over the store handler.
                let credential = self.credential.as_ref().ok_or_else(|| {
                    CellTransportError::Io(
                        "store partition shipping requires an authenticated cell credential"
                            .to_owned(),
                    )
                })?;
                if credential.cell_id() != partition.cell_id() {
                    return Err(CellTransportError::Io(
                        "store partition credential does not match the cell identity".to_owned(),
                    ));
                }
                let ship = CellStorePartitionShip {
                    cell_id: partition.cell_id(),
                    partition: (**partition).clone(),
                };
                let body = self.seal_payload(AdmissionPurpose::StorePartition, &ship)?;
                let reply: Bytes = self
                    .velo
                    .unary(HANDLER_STORE_PARTITION)
                    .map_err(io)?
                    .raw_payload(body)
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
                let ack: CellAck = rmp_serde::from_slice(&reply).map_err(decode)?;
                if !ack.ok {
                    return Err(CellTransportError::Io(
                        "controller nacked store partition".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::process::Command;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::cellular::dataset_session::DatasetPublisher;
    use crate::cellular::phaser::Phaser;
    use crate::cellular::shard::ColumnStorePartition;
    use crate::cellular::transport::dataset_velo::{
        DatasetServer, HANDLER_DATASET_SUBSCRIBE, WirePayload,
    };
    use crate::cellular::transport::phaser_velo::{HANDLER_PHASER_SUBSCRIBE, PhaserServer};
    use crate::engine::artifact_shipping::{
        ArtifactBearer, ArtifactChannelClient, ArtifactChannelRegistrar, ArtifactUploadServer,
        ship_cell_artifacts,
    };
    use crate::engine::artifact_stream_velo::{
        ArtifactVeloReceiver, HANDLER_ARTIFACT_CLOSE, HANDLER_ARTIFACT_DONE, HANDLER_ARTIFACT_OPEN,
    };
    use crate::engine::cellular_bootstrap::CellularRole;
    use crate::engine::cellular_registration::{
        ADMISSION_PURPOSE_COUNT, AdmissionRejection, CellRegistrationAuthority,
        CellRegistrationCredential, CellSecurityContext, MAX_AUTHENTICATED_FRAME_BYTES,
        RegistrationBegin, RegistrationConflict, RegistrationLedger, RoleVerifyingKey,
    };
    use crate::metrics_core::accumulator::MetricsAccumulator;

    const PRODUCTION_ROUTE_CHILD_ENV: &str = "AIPERF_PRODUCTION_ROUTE_AUTH_CHILD";
    const PRODUCTION_ROUTE_TEST: &str = "cellular::transport::velo_transport::tests::production_handlers_authenticate_payloads_and_reject_replay";
    const VELO_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
    const VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES: usize = 22;

    #[test]
    fn route_registration_trust_boundary_is_disclosed() {
        let production_module = include_str!("velo_transport.rs")
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("module source has a test boundary");

        assert!(production_module.contains("registered benchmark cells are trusted"));
        assert!(production_module.contains("per-push route authenticity or replay protection"));
        assert!(production_module.contains("untrusted participants"));
    }

    #[derive(Serialize, Deserialize)]
    struct TestAuthenticatedFrame {
        version: u8,
        role: CellularRole,
        session_nonce: [u8; 32],
        sequence: u64,
        peer_info: Vec<u8>,
        payload: Vec<u8>,
        signature: Vec<u8>,
    }

    #[derive(Serialize)]
    struct CellIdRequest {
        cell_id: u32,
    }

    #[derive(Serialize)]
    struct ArtifactPathRequest {
        cell_id: u32,
        rel: String,
    }

    #[derive(Deserialize)]
    struct ArtifactAck {
        ok: bool,
        error: Option<String>,
    }

    #[derive(Deserialize)]
    struct ArtifactOpenReply {
        handle: velo::StreamAnchorHandle,
        controller_peer: Vec<u8>,
    }

    struct CountingSubscriber(Arc<AtomicUsize>);

    impl tracing::Subscriber for CountingSubscriber {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            if event.metadata().target().starts_with("aiperf") {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        fn enter(&self, _: &tracing::span::Id) {}

        fn exit(&self, _: &tracing::span::Id) {}
    }

    fn malformed_frame(encoded: &[u8], has_malformed_peer: bool) -> Bytes {
        let mut frame: TestAuthenticatedFrame =
            rmp_serde::from_slice(encoded).expect("decode authenticated test frame");
        if has_malformed_peer {
            frame.peer_info = vec![0xC1];
        } else {
            frame.payload = vec![0xC1];
        }
        Bytes::from(rmp_serde::to_vec(&frame).expect("encode tampered test frame"))
    }

    async fn wait_for_invalid_count(
        authority: &CellRegistrationAuthority,
        purpose: AdmissionPurpose,
        expected: u64,
    ) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while authority.invalid_count(purpose) != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "timed out waiting for {purpose:?} invalid count {expected}; observed {}",
                authority.invalid_count(purpose)
            )
        });
    }

    async fn send_fire(cell: &Velo, controller: &PeerInfo, handler: &str, body: Bytes) {
        cell.am_send(handler)
            .expect("fire builder")
            .raw_payload(body)
            .instance(controller.instance_id())
            .send()
            .await
            .expect("fire send");
    }

    async fn send_unary(
        cell: &Velo,
        controller: &PeerInfo,
        handler: &str,
        body: Bytes,
    ) -> anyhow::Result<Bytes> {
        cell.unary(handler)?
            .raw_payload(body)
            .instance(controller.instance_id())
            .send()
            .await
    }

    async fn unary_rejection(
        cell: &Velo,
        controller: &PeerInfo,
        handler: &str,
        body: Bytes,
        returns_ack: bool,
    ) -> String {
        match send_unary(cell, controller, handler, body).await {
            Ok(reply) if returns_ack => {
                let ack: ArtifactAck =
                    rmp_serde::from_slice(&reply).expect("decode artifact rejection ack");
                assert!(!ack.ok, "{handler} accepted an invalid frame");
                ack.error.expect("artifact rejection reason")
            }
            Ok(_) => panic!("{handler} accepted an invalid frame"),
            Err(error) => error.to_string(),
        }
    }

    async fn reject_invalid_frames(
        cell: &Velo,
        controller: &PeerInfo,
        authority: &CellRegistrationAuthority,
        trace_events: &AtomicUsize,
        handler: &str,
        purpose: AdmissionPurpose,
        valid: &[u8],
        is_fire: bool,
        has_peer_decode: bool,
        returns_ack: bool,
    ) {
        let baseline = authority.invalid_count(purpose);
        let trace_baseline = trace_events.load(Ordering::Relaxed);
        let mut invalids = if purpose == AdmissionPurpose::Register {
            vec![malformed_frame(valid, false)]
        } else {
            vec![Bytes::copy_from_slice(valid), malformed_frame(valid, false)]
        };
        if has_peer_decode {
            invalids.push(malformed_frame(valid, true));
        }
        let rejected_frames = invalids.len();

        if is_fire {
            for body in &invalids {
                send_fire(cell, controller, handler, body.clone()).await;
            }
            if purpose == AdmissionPurpose::Heartbeat {
                for _ in invalids.len()..10_000 {
                    send_fire(cell, controller, handler, invalids[1].clone()).await;
                }
            }
        } else {
            let mut rejections = Vec::with_capacity(invalids.len());
            for body in invalids {
                rejections
                    .push(unary_rejection(cell, controller, handler, body, returns_ack).await);
            }
            assert!(
                rejections
                    .iter()
                    .all(|rejection| rejection == &rejections[0]),
                "{handler} exposed non-constant rejection details: {rejections:?}"
            );
            assert!(
                rejections[0].contains("AdmissionRejected"),
                "{handler} did not use the constant admission rejection: {rejections:?}"
            );
        }

        let rejected = if purpose == AdmissionPurpose::Heartbeat {
            10_000
        } else {
            rejected_frames as u64
        };
        wait_for_invalid_count(authority, purpose, baseline + rejected).await;
        assert_eq!(
            trace_events.load(Ordering::Relaxed),
            trace_baseline,
            "{handler} traced an invalid frame"
        );
    }

    async fn assert_no_controller_message(controller: &mut VeloControllerTransport) {
        assert!(
            tokio::time::timeout(Duration::from_millis(50), controller.recv())
                .await
                .is_err(),
            "an invalid frame reached the controller application channel"
        );
    }

    async fn open_artifact_stream(
        cell: &Velo,
        controller: &PeerInfo,
        credential: &CellRegistrationCredential,
        rel: &str,
    ) -> (Vec<u8>, velo::StreamSender<Vec<u8>>) {
        let request = ArtifactPathRequest {
            cell_id: 0,
            rel: rel.to_owned(),
        };
        let body = credential
            .seal_payload(AdmissionPurpose::ArtifactOpen, &cell.peer_info(), &request)
            .expect("seal artifact open");
        let reply = send_unary(
            cell,
            controller,
            HANDLER_ARTIFACT_OPEN,
            Bytes::copy_from_slice(&body),
        )
        .await
        .expect("valid artifact open");
        let open: ArtifactOpenReply =
            rmp_serde::from_slice(&reply).expect("decode artifact open reply");
        let controller_full: PeerInfo =
            rmp_serde::from_slice(&open.controller_peer).expect("decode full controller peer");
        cell.register_peer(controller_full)
            .expect("register full controller peer");
        let sender = cell
            .attach_anchor::<Vec<u8>>(open.handle)
            .await
            .expect("attach artifact anchor");
        (body, sender)
    }

    #[test]
    fn production_handlers_authenticate_payloads_and_reject_replay() {
        if std::env::var_os(PRODUCTION_ROUTE_CHILD_ENV).is_none() {
            let output = Command::new(std::env::current_exe().expect("current test binary"))
                .arg("--exact")
                .arg(PRODUCTION_ROUTE_TEST)
                .arg("--nocapture")
                .env(PRODUCTION_ROUTE_CHILD_ENV, "1")
                .env("RUST_TEST_THREADS", "1")
                .output()
                .expect("spawn isolated production-route test");
            assert!(
                output.status.success(),
                "isolated production-route child failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let trace_events = Arc::new(AtomicUsize::new(0));
        tracing::subscriber::set_global_default(CountingSubscriber(Arc::clone(&trace_events)))
            .expect("install isolated trace subscriber");
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("production-route runtime");
        runtime.block_on(async {
            production_handlers_authenticate_payloads_and_reject_replay_impl(trace_events).await;
        });
    }

    async fn production_handlers_authenticate_payloads_and_reject_replay_impl(
        trace_events: Arc<AtomicUsize>,
    ) {
        let temp = tempfile::tempdir().expect("route fixture tempdir");
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.peer_info();
        let controller_messenger_peer = controller_velo.messenger().peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let authority = Arc::new(authority);
        let plan_calls = Arc::new(AtomicUsize::new(0));
        let seen_plan_calls = Arc::clone(&plan_calls);
        let plan_registration: PlanRegistration = Arc::new(move |_| {
            seen_plan_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(CellRegistrationPlan {
                envelope: vec![0xA5],
                artifact: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            Arc::clone(&controller_velo),
            Arc::clone(&authority),
            plan_registration,
            1,
            start.handle(),
        )
        .expect("bind controller handlers");
        let _phaser = PhaserServer::bind(
            Arc::clone(&controller_velo),
            Phaser::new(),
            Arc::clone(&authority),
        )
        .expect("bind phaser handler");
        let _dataset = DatasetServer::bind(
            Arc::clone(&controller_velo),
            DatasetPublisher::<WirePayload>::new(),
            Arc::clone(&authority),
        )
        .expect("bind dataset handler");
        let allowed: HashSet<String> = [
            "inventory-open.bin".to_owned(),
            "inventory-close.bin".to_owned(),
        ]
        .into_iter()
        .collect();
        let artifact = ArtifactVeloReceiver::register(
            Arc::clone(&controller_velo),
            temp.path().to_path_buf(),
            allowed,
            Arc::clone(&authority),
        )
        .expect("bind artifact handlers");

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        cell_velo
            .register_peer(controller_peer.clone())
            .expect("register controller peer");
        let credential = &credentials[0];
        let cell_peer = cell_velo.peer_info();

        let mut handlers: Vec<String> = controller_velo
            .list_local_handlers()
            .into_iter()
            .filter(|handler| handler.starts_with("aiperf."))
            .collect();
        handlers.sort();
        assert_eq!(
            handlers.len(),
            ADMISSION_PURPOSE_COUNT - 1,
            "the live controller handler inventory changed without a production-route security test adapter"
        );
        for handler in &handlers {
            assert!(
                MAX_AUTHENTICATED_FRAME_BYTES
                    + VELO_ACTIVE_MESSAGE_FIXED_HEADER_BYTES
                    + handler.len()
                    <= VELO_MAX_FRAME_BYTES,
                "the authenticated-frame ceiling no longer fits production route {handler}"
            );
        }
        for handler in handlers {
            match handler.as_str() {
                HANDLER_REGISTER => {
                    let connected = ConnectedController::from_parts(
                        controller_messenger_peer.clone(),
                        DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
                    );
                    let cell_peer_bytes = rmp_serde::to_vec(&cell_peer).unwrap();
                    let register = CellRegister {
                        cell_id: 0,
                        cell_peer: cell_peer_bytes.clone(),
                        artifact_capability_digest: None,
                        registration_proof: Some(
                            credential
                                .sign_register(
                                    &cell_peer_bytes,
                                    None,
                                    connected.binding().expect("controller binding"),
                                )
                                .expect("registration proof"),
                        ),
                    };
                    let body = credential
                        .seal_payload(AdmissionPurpose::Register, &cell_peer, &register)
                        .expect("seal register");
                    let first = send_unary(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await
                    .expect("valid register");
                    let retry = send_unary(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await
                    .expect("exact registration retry");
                    assert_eq!(first, retry);
                    assert_eq!(plan_calls.load(Ordering::Relaxed), 1);
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::Register,
                        &body,
                        false,
                        true,
                        false,
                    )
                    .await;
                    assert_eq!(plan_calls.load(Ordering::Relaxed), 1);
                }
                HANDLER_PREFLIGHT => {
                    let body = credential
                        .seal_payload(
                            AdmissionPurpose::Preflight,
                            &cell_peer,
                            &CellMessage::Preflight {
                                cell_id: 0,
                                result: Ok(()),
                            },
                        )
                        .expect("seal preflight");
                    send_fire(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await;
                    controller
                        .await_all_preflight()
                        .await
                        .expect("valid preflight effect");
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::Preflight,
                        &body,
                        true,
                        false,
                        false,
                    )
                    .await;
                }
                HANDLER_HEARTBEAT => {
                    let body = credential
                        .seal_payload(
                            AdmissionPurpose::Heartbeat,
                            &cell_peer,
                            &CellMessage::Heartbeat {
                                cell_id: 0,
                                heartbeat: Box::new(sample_heartbeat()),
                            },
                        )
                        .expect("seal heartbeat");
                    send_fire(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await;
                    assert!(matches!(
                        tokio::time::timeout(Duration::from_secs(2), controller.recv())
                            .await
                            .expect("valid heartbeat effect timed out")
                            .expect("heartbeat recv"),
                        Some(CellMessage::Heartbeat { cell_id: 0, .. })
                    ));
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::Heartbeat,
                        &body,
                        true,
                        false,
                        false,
                    )
                    .await;
                    assert_no_controller_message(&mut controller).await;
                }
                HANDLER_PHASE_SIGNAL => {
                    let body = credential
                        .seal_payload(
                            AdmissionPurpose::PhaseSignal,
                            &cell_peer,
                            &CellMessage::PhaseSignal {
                                cell_id: 0,
                                phase: "profiling".to_owned(),
                                signal: CellPhaseSignal::Complete,
                            },
                        )
                        .expect("seal phase signal");
                    send_fire(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await;
                    assert!(matches!(
                        tokio::time::timeout(Duration::from_secs(2), controller.recv())
                            .await
                            .expect("valid phase effect timed out")
                            .expect("phase recv"),
                        Some(CellMessage::PhaseSignal { cell_id: 0, .. })
                    ));
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::PhaseSignal,
                        &body,
                        true,
                        false,
                        false,
                    )
                    .await;
                    assert_no_controller_message(&mut controller).await;
                }
                HANDLER_PARTITION => {
                    let ship = CellPartitionShip {
                        cell_id: 0,
                        partition: sample_partition(0),
                    };
                    let body = credential
                        .seal_payload(AdmissionPurpose::Partition, &cell_peer, &ship)
                        .expect("seal partition");
                    let reply = send_unary(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await
                    .expect("valid partition");
                    let ack: CellAck = rmp_serde::from_slice(&reply).expect("partition ack");
                    assert!(ack.ok);
                    assert!(matches!(
                        tokio::time::timeout(Duration::from_secs(2), controller.recv())
                            .await
                            .expect("valid partition effect timed out")
                            .expect("partition recv"),
                        Some(CellMessage::Partition(_))
                    ));
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::Partition,
                        &body,
                        false,
                        true,
                        false,
                    )
                    .await;
                    assert_no_controller_message(&mut controller).await;
                }
                HANDLER_STORE_PARTITION => {
                    let mut accumulator = MetricsAccumulator::new();
                    accumulator.process_record(&RecordIngest::minimal(
                        1_000,
                        5_000,
                        Phase::Profiling,
                    ));
                    let ship = CellStorePartitionShip {
                        cell_id: 0,
                        partition: ColumnStorePartition::from_accumulator(0, &accumulator),
                    };
                    let body = credential
                        .seal_payload(AdmissionPurpose::StorePartition, &cell_peer, &ship)
                        .expect("seal store partition");
                    let reply = send_unary(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await
                    .expect("valid store partition");
                    let ack: CellAck = rmp_serde::from_slice(&reply).expect("store ack");
                    assert!(ack.ok);
                    assert!(matches!(
                        tokio::time::timeout(Duration::from_secs(2), controller.recv())
                            .await
                            .expect("valid store-partition effect timed out")
                            .expect("store recv"),
                        Some(CellMessage::StorePartition(_))
                    ));
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::StorePartition,
                        &body,
                        false,
                        true,
                        false,
                    )
                    .await;
                    assert_no_controller_message(&mut controller).await;
                }
                HANDLER_PHASER_SUBSCRIBE => {
                    let body = credential
                        .seal_payload(
                            AdmissionPurpose::PhaserSubscribe,
                            &cell_peer,
                            &CellIdRequest { cell_id: 0 },
                        )
                        .expect("seal phaser subscribe");
                    assert!(
                        !send_unary(
                            &cell_velo,
                            &controller_peer,
                            &handler,
                            Bytes::copy_from_slice(&body),
                        )
                        .await
                        .expect("valid phaser subscribe")
                        .is_empty()
                    );
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::PhaserSubscribe,
                        &body,
                        false,
                        true,
                        false,
                    )
                    .await;
                }
                HANDLER_DATASET_SUBSCRIBE => {
                    let body = credential
                        .seal_payload(
                            AdmissionPurpose::DatasetSubscribe,
                            &cell_peer,
                            &CellIdRequest { cell_id: 0 },
                        )
                        .expect("seal dataset subscribe");
                    assert!(
                        !send_unary(
                            &cell_velo,
                            &controller_peer,
                            &handler,
                            Bytes::copy_from_slice(&body),
                        )
                        .await
                        .expect("valid dataset subscribe")
                        .is_empty()
                    );
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::DatasetSubscribe,
                        &body,
                        false,
                        true,
                        false,
                    )
                    .await;
                }
                HANDLER_ARTIFACT_OPEN => {
                    let (body, _sender) = open_artifact_stream(
                        &cell_velo,
                        &controller_peer,
                        credential,
                        "inventory-open.bin",
                    )
                    .await;
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::ArtifactOpen,
                        &body,
                        false,
                        true,
                        true,
                    )
                    .await;
                }
                HANDLER_ARTIFACT_CLOSE => {
                    let (_, sender) = open_artifact_stream(
                        &cell_velo,
                        &controller_peer,
                        credential,
                        "inventory-close.bin",
                    )
                    .await;
                    sender
                        .send(zstd::encode_all(b"route".as_slice(), 3).unwrap())
                        .await
                        .expect("send compressed artifact");
                    sender.finalize().expect("finalize artifact");
                    let close = ArtifactPathRequest {
                        cell_id: 0,
                        rel: "inventory-close.bin".to_owned(),
                    };
                    let body = credential
                        .seal_payload(AdmissionPurpose::ArtifactClose, &cell_peer, &close)
                        .expect("seal artifact close");
                    let reply = send_unary(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await
                    .expect("valid artifact close");
                    let ack: ArtifactAck =
                        rmp_serde::from_slice(&reply).expect("artifact close ack");
                    assert!(ack.ok, "artifact close failed: {:?}", ack.error);
                    assert_eq!(
                        std::fs::read(temp.path().join("cell-0/inventory-close.bin")).unwrap(),
                        b"route"
                    );
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::ArtifactClose,
                        &body,
                        false,
                        false,
                        true,
                    )
                    .await;
                }
                HANDLER_ARTIFACT_DONE => {
                    let body = credential
                        .seal_payload(
                            AdmissionPurpose::ArtifactDone,
                            &cell_peer,
                            &CellIdRequest { cell_id: 0 },
                        )
                        .expect("seal artifact done");
                    let reply = send_unary(
                        &cell_velo,
                        &controller_peer,
                        &handler,
                        Bytes::copy_from_slice(&body),
                    )
                    .await
                    .expect("valid artifact done");
                    let ack: ArtifactAck =
                        rmp_serde::from_slice(&reply).expect("artifact done ack");
                    assert!(ack.ok);
                    artifact
                        .wait_for_cells(1, Duration::from_secs(1))
                        .await
                        .expect("artifact done effect");
                    reject_invalid_frames(
                        &cell_velo,
                        &controller_peer,
                        &authority,
                        &trace_events,
                        &handler,
                        AdmissionPurpose::ArtifactDone,
                        &body,
                        false,
                        false,
                        true,
                    )
                    .await;
                }
                _ => panic!(
                    "live production handler {handler} has no authentication behavior adapter"
                ),
            }
        }

        assert_eq!(authority.replay_slot_count(), ADMISSION_PURPOSE_COUNT);
    }

    struct AuthenticatedRegisterFixture {
        connected: ConnectedController,
        verifier: ControllerRegisterVerifier,
        registration: CellRegister,
        credential: CellRegistrationCredential,
        cell_peer: PeerInfo,
        attestor: ControllerRegisterAttestor,
        start_event: EventHandle,
    }

    impl AuthenticatedRegisterFixture {
        fn signed_reply(&self, cell_id: u32, envelope: &[u8]) -> RegisterReply {
            let reply_payload = encode_reply_payload(envelope, self.start_event, &None)
                .expect("encode reply payload");
            let registration_frame = Bytes::from(
                self.credential
                    .seal_payload(
                        AdmissionPurpose::Register,
                        &self.cell_peer,
                        &self.registration,
                    )
                    .expect("encode registration frame"),
            );
            let attestation = self
                .attestor
                .attest(
                    &self.connected.binding().expect("controller binding"),
                    &registration_frame,
                    &reply_payload,
                )
                .expect("attest reply");
            assert_eq!(self.registration.cell_id, cell_id);
            RegisterReply {
                envelope: envelope.to_vec(),
                start_event: self.start_event,
                artifact_channel: None,
                attestation,
                registration_frame,
                reply_payload: Bytes::from(reply_payload),
            }
        }

        fn with_envelope(&self, mut reply: RegisterReply, envelope: &[u8]) -> RegisterReply {
            reply.envelope = envelope.to_vec();
            reply
        }

        fn connected_with_changed_worker_address(&self) -> ConnectedController {
            ConnectedController::from_parts(
                PeerInfo::new(
                    self.connected.peer().instance_id(),
                    velo::WorkerAddress::from_encoded(vec![0x80]),
                ),
                DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
            )
        }

        fn with_dial_port(&self, port_offset: u16) -> ConnectedController {
            ConnectedController::from_parts(
                self.connected.peer().clone(),
                DialedControllerAddress::Tcp(
                    format!("127.0.0.1:{}", 9500 + port_offset).parse().unwrap(),
                ),
            )
        }
    }

    async fn authenticated_register_fixture() -> AuthenticatedRegisterFixture {
        let controller = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let controller_peer = controller.peer_info();
        let attestor = ControllerRegisterAttestor::mint(authority.run_nonce()).expect("attestor");
        let connected = ConnectedController::from_parts(
            controller_peer,
            DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
        );
        let binding = connected.binding().expect("controller binding");
        let cell_peer = PeerInfo::new(
            velo::InstanceId::new_v4(),
            velo::WorkerAddress::from_encoded(vec![0x81]),
        );
        let encoded_cell_peer = rmp_serde::to_vec(&cell_peer).expect("cell peer");
        let registration = CellRegister {
            cell_id: 0,
            cell_peer: encoded_cell_peer.clone(),
            artifact_capability_digest: None,
            registration_proof: Some(
                credentials[0]
                    .sign_register(&encoded_cell_peer, None, binding)
                    .expect("registration proof"),
            ),
        };
        AuthenticatedRegisterFixture {
            connected,
            verifier: attestor.verifier(),
            registration,
            credential: credentials[0].clone(),
            cell_peer,
            attestor,
            start_event: controller
                .event_manager()
                .new_event()
                .expect("start event")
                .handle(),
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(u8)]
    enum FailurePoint {
        None,
        Plan,
        Attest,
        Encode,
        RegisterPeer,
    }

    struct PlanningReleaseGuard {
        release: Arc<std::sync::Barrier>,
        is_armed: bool,
    }

    impl PlanningReleaseGuard {
        fn new(release: Arc<std::sync::Barrier>) -> Self {
            Self {
                release,
                is_armed: true,
            }
        }

        fn release(mut self) {
            self.release.wait();
            self.is_armed = false;
        }
    }

    impl Drop for PlanningReleaseGuard {
        fn drop(&mut self) {
            if self.is_armed {
                self.release.wait();
            }
        }
    }

    struct RegistrationTransactionFixture {
        _temporary: tempfile::TempDir,
        _cell: Arc<Velo>,
        controller: Arc<Velo>,
        controller_peer: PeerInfo,
        cell_peer: PeerInfo,
        authority: Arc<CellRegistrationAuthority>,
        credentials: [CellRegistrationCredential; 2],
        attestor: ControllerRegisterAttestor,
        ledger: RegistrationLedger,
        planner: PlanRegistration,
        registrar: ArtifactChannelRegistrar,
        artifact_authority: String,
        _artifact_server: ArtifactUploadServer,
        start_event: EventHandle,
        failure: Arc<AtomicU8>,
        plan_calls: Arc<AtomicUsize>,
        barrier_count: Arc<AtomicUsize>,
    }

    impl RegistrationTransactionFixture {
        async fn new(failure_point: FailurePoint) -> Self {
            let temporary = tempfile::tempdir().expect("registration fixture temporary directory");
            let artifact_server = ArtifactUploadServer::start(
                "127.0.0.1:0".parse().unwrap(),
                temporary.path().join("landed"),
                HashSet::new(),
                1,
            )
            .await
            .expect("artifact server");
            let registrar = artifact_server.registrar();
            let artifact_authority = artifact_server.local_addr().to_string();
            let controller = build_velo(BindSpec::TcpLoopback)
                .await
                .expect("controller velo");
            let cell = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
            let controller_peer = controller.messenger().peer_info();
            let cell_peer = cell.peer_info();
            let run_nonce = [0x41; 32];
            let controller_signer = ed25519_dalek::SigningKey::from_bytes(&[0x42; 32]);
            let cell_signer = ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]);
            let controller_context = Arc::new(
                CellSecurityContext::controller(
                    run_nonce,
                    controller_signer.clone(),
                    vec![RoleVerifyingKey {
                        role: CellularRole::Cell(0),
                        verifier: cell_signer.verifying_key(),
                    }]
                    .into_boxed_slice(),
                )
                .expect("controller security context"),
            );
            let worker = |signer| {
                Arc::new(
                    CellSecurityContext::worker(
                        run_nonce,
                        CellularRole::Cell(0),
                        signer,
                        controller_signer.verifying_key(),
                    )
                    .expect("worker security context"),
                )
                .registration_credential()
                .expect("registration credential")
            };
            let credentials = [worker(cell_signer.clone()), worker(cell_signer)];
            let authority = Arc::new(
                controller_context
                    .registration_authority()
                    .expect("registration authority"),
            );
            let attestor = controller_context
                .reply_attestor()
                .expect("registration reply attestor");
            let ledger = RegistrationLedger::new(Arc::clone(&authority), 1);
            let failure = Arc::new(AtomicU8::new(failure_point as u8));
            let plan_calls = Arc::new(AtomicUsize::new(0));
            let planner: PlanRegistration = {
                let failure = Arc::clone(&failure);
                let plan_calls = Arc::clone(&plan_calls);
                let registrar = registrar.clone();
                Arc::new(move |verified| {
                    plan_calls.fetch_add(1, Ordering::Relaxed);
                    if failure.load(Ordering::Relaxed) == FailurePoint::Plan as u8 {
                        anyhow::bail!("injected registration planning failure");
                    }
                    let register: CellRegister = verified.decode_payload()?;
                    let digest = register
                        .artifact_capability_digest
                        .ok_or_else(|| anyhow::anyhow!("missing artifact capability"))?;
                    Ok(Some(CellRegistrationPlan {
                        envelope: vec![0xA5],
                        artifact: Some(registrar.prepare(register.cell_id, digest)?),
                    }))
                })
            };
            let start_event = controller
                .event_manager()
                .new_event()
                .expect("start event")
                .handle();
            Self {
                _temporary: temporary,
                _cell: cell,
                controller,
                controller_peer,
                cell_peer,
                authority,
                credentials,
                attestor,
                ledger,
                planner,
                registrar,
                artifact_authority,
                _artifact_server: artifact_server,
                start_event,
                failure,
                plan_calls,
                barrier_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn frame(&self, credential_index: usize, peer: &PeerInfo, bearer_byte: u8) -> Bytes {
            let credential = &self.credentials[credential_index];
            let cell_peer = rmp_serde::to_vec(peer).expect("encode cell peer");
            let digest = ArtifactBearer::from_test_bytes([bearer_byte; 32]).digest_bytes();
            let binding = ControllerPeerBinding::new(
                &self.controller_peer,
                DialedControllerAddress::Tcp("127.0.0.1:9500".parse().unwrap()),
            )
            .expect("controller binding");
            let register = CellRegister {
                cell_id: 0,
                cell_peer,
                artifact_capability_digest: Some(digest),
                registration_proof: Some(
                    credential
                        .sign_register(
                            &rmp_serde::to_vec(peer).expect("encode proof peer"),
                            Some(digest),
                            binding,
                        )
                        .expect("registration proof"),
                ),
            };
            Bytes::from(
                credential
                    .seal_payload(AdmissionPurpose::Register, peer, &register)
                    .expect("authenticated registration frame"),
            )
        }

        fn exact_frame(&self) -> Bytes {
            self.frame(0, &self.cell_peer, 0x31)
        }

        fn heartbeat_frame(&self) -> Bytes {
            Bytes::from(
                self.credentials[0]
                    .seal_payload(
                        AdmissionPurpose::Heartbeat,
                        &self.cell_peer,
                        &CellMessage::Heartbeat {
                            cell_id: 0,
                            heartbeat: Box::new(sample_heartbeat()),
                        },
                    )
                    .expect("authenticated heartbeat frame"),
            )
        }

        fn register_frame(&self, frame: Bytes) -> anyhow::Result<Bytes> {
            self.register_frame_with_planner(frame, &self.planner)
        }

        fn register_frame_with_planner(
            &self,
            frame: Bytes,
            planner: &PlanRegistration,
        ) -> anyhow::Result<Bytes> {
            let failure = Arc::clone(&self.failure);
            let barrier_count = Arc::clone(&self.barrier_count);
            execute_registration_transaction(
                &self.authority,
                &self.ledger,
                planner,
                &self.controller_peer,
                self.start_event,
                frame,
                |binding, registration, payload| {
                    if failure.load(Ordering::Relaxed) == FailurePoint::Attest as u8 {
                        anyhow::bail!("injected registration attestation failure");
                    }
                    self.attestor.attest(binding, registration, payload)
                },
                |payload, attestation| {
                    if failure.load(Ordering::Relaxed) == FailurePoint::Encode as u8 {
                        anyhow::bail!("injected registration encoding failure");
                    }
                    encode_attested_register_reply(payload, attestation).map_err(anyhow::Error::new)
                },
                |peer| {
                    if failure.load(Ordering::Relaxed) == FailurePoint::RegisterPeer as u8 {
                        anyhow::bail!("injected reverse peer registration failure");
                    }
                    self.controller
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register peer: {error}"))
                },
                move || {
                    barrier_count.fetch_add(1, Ordering::Relaxed);
                },
            )
        }

        fn register(&self) -> anyhow::Result<Bytes> {
            self.register_frame(self.exact_frame())
        }

        fn clear_failure(&self) {
            self.failure
                .store(FailurePoint::None as u8, Ordering::Relaxed);
        }

        async fn artifact_authorized(&self, bearer_byte: u8) -> bool {
            let client = ArtifactChannelClient::new(
                self.artifact_authority.clone(),
                self.registrar.server_config(),
                ArtifactBearer::from_test_bytes([bearer_byte; 32]),
            )
            .expect("artifact client");
            ship_cell_artifacts(&client, 0, self._temporary.path(), &[])
                .await
                .is_ok()
        }

        fn artifact_publication_count(&self) -> usize {
            self.registrar.authorization_publication_count()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn registration_transaction_failure_rolls_back_every_application_side_effect() {
        for failure in [
            FailurePoint::Plan,
            FailurePoint::Attest,
            FailurePoint::Encode,
            FailurePoint::RegisterPeer,
        ] {
            let fixture = RegistrationTransactionFixture::new(failure).await;
            let frame = fixture.exact_frame();
            let heartbeat = fixture.heartbeat_frame();
            assert!(
                fixture.register_frame(frame.clone()).is_err(),
                "failure point {failure:?}"
            );
            assert_eq!(fixture.barrier_count.load(Ordering::Relaxed), 0);
            assert_eq!(fixture.artifact_publication_count(), 0);
            assert!(!fixture.artifact_authorized(0x31).await);
            assert!(matches!(
                fixture
                    .authority
                    .open_payload::<CellMessage>(AdmissionPurpose::Heartbeat, heartbeat.as_ref(),),
                Err(AdmissionRejection::Session)
            ));

            let plan_calls_after_failure = fixture.plan_calls.load(Ordering::Relaxed);
            fixture.clear_failure();
            let reply = fixture
                .register_frame(frame.clone())
                .unwrap_or_else(|error| panic!("retry after {failure:?}: {error}"));
            assert_eq!(
                fixture.plan_calls.load(Ordering::Relaxed),
                plan_calls_after_failure + 1,
                "retry after {failure:?} must replan instead of reading a cache"
            );
            assert_eq!(decode_reply(&reply, frame).unwrap().envelope, vec![0xA5]);
            assert_eq!(fixture.barrier_count.load(Ordering::Relaxed), 1);
            assert_eq!(fixture.artifact_publication_count(), 1);
            assert!(fixture.artifact_authorized(0x31).await);
            assert!(
                fixture
                    .authority
                    .open_payload::<CellMessage>(AdmissionPurpose::Heartbeat, heartbeat.as_ref(),)
                    .is_ok(),
                "retry after {failure:?} must install the session baseline"
            );
            assert!(matches!(
                fixture
                    .authority
                    .open_payload::<CellMessage>(AdmissionPurpose::Heartbeat, heartbeat.as_ref(),),
                Err(AdmissionRejection::Replay)
            ));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exact_registration_transaction_retry_returns_cached_bytes_and_counts_once() {
        let fixture = RegistrationTransactionFixture::new(FailurePoint::None).await;
        let frame = fixture.exact_frame();
        let first = fixture.register_frame(frame.clone()).unwrap();
        let retry = fixture.register_frame(frame);

        assert_eq!(fixture.plan_calls.load(Ordering::Relaxed), 1);
        let retry = retry.unwrap();
        assert_eq!(first, retry);
        assert_eq!(fixture.barrier_count.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.artifact_publication_count(), 1);
        assert!(fixture.artifact_authorized(0x31).await);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn changed_registration_transaction_cannot_observe_or_replace_committed_material() {
        for changed in ["session", "peer", "capability"] {
            let fixture = RegistrationTransactionFixture::new(FailurePoint::None).await;
            fixture.register().unwrap();
            let changed_frame = match changed {
                "session" => fixture.frame(1, &fixture.cell_peer, 0x31),
                "peer" => fixture.frame(
                    0,
                    &PeerInfo::new(
                        velo::InstanceId::new_v4(),
                        velo::WorkerAddress::from_encoded(vec![0x91]),
                    ),
                    0x31,
                ),
                "capability" => fixture.frame(0, &fixture.cell_peer, 0x32),
                _ => unreachable!(),
            };
            let error = fixture.register_frame(changed_frame).unwrap_err();
            assert!(matches!(
                error.downcast_ref::<RegistrationConflict>(),
                Some(RegistrationConflict::ChangedTranscript)
            ));
            assert_eq!(fixture.plan_calls.load(Ordering::Relaxed), 1);
            assert_eq!(fixture.barrier_count.load(Ordering::Relaxed), 1);
            assert!(fixture.artifact_authorized(0x31).await);
            if changed == "capability" {
                assert!(!fixture.artifact_authorized(0x32).await);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn registration_transaction_concurrent_exact_is_busy_and_drop_restores_vacancy() {
        let fixture = RegistrationTransactionFixture::new(FailurePoint::None).await;
        let frame = fixture.exact_frame();
        let planning_started = Arc::new(std::sync::Barrier::new(2));
        let release_planning = Arc::new(std::sync::Barrier::new(2));
        let held_planner: PlanRegistration = {
            let planner = Arc::clone(&fixture.planner);
            let planning_started = Arc::clone(&planning_started);
            let release_planning = Arc::clone(&release_planning);
            Arc::new(move |verified| {
                let plan = planner(verified)?;
                planning_started.wait();
                release_planning.wait();
                Ok(plan)
            })
        };

        std::thread::scope(|scope| {
            let held_frame = frame.clone();
            let held =
                scope.spawn(|| fixture.register_frame_with_planner(held_frame, &held_planner));
            planning_started.wait();
            let release_guard = PlanningReleaseGuard::new(Arc::clone(&release_planning));
            let changed_frame = fixture.frame(0, &fixture.cell_peer, 0x32);
            let exact = scope.spawn(|| fixture.register_frame(frame.clone()));
            let changed = scope.spawn(|| fixture.register_frame(changed_frame));
            let exact = exact.join().unwrap();
            let changed = changed.join().unwrap();
            release_guard.release();
            let held = held.join().unwrap();
            assert!(matches!(
                exact.unwrap_err().downcast_ref(),
                Some(RegistrationConflict::Busy(_))
            ));
            assert!(matches!(
                changed.unwrap_err().downcast_ref(),
                Some(RegistrationConflict::ChangedTranscript)
            ));
            assert!(held.is_ok());
        });
        assert_eq!(fixture.plan_calls.load(Ordering::Relaxed), 1);
        assert_eq!(fixture.barrier_count.load(Ordering::Relaxed), 1);

        let replacement = RegistrationTransactionFixture::new(FailurePoint::None).await;
        let frame = replacement.exact_frame();
        let verified = replacement
            .authority
            .verify_registration_frame(frame.clone())
            .unwrap();
        let mut prepared = match replacement.ledger.begin(&verified).unwrap() {
            RegistrationBegin::Prepared(prepared) => prepared,
            RegistrationBegin::ExactRetry { .. } => panic!("fresh slot returned a retry"),
        };
        let plan = (replacement.planner)(&verified).unwrap().unwrap();
        prepared.install_plan(plan);
        drop(prepared);
        assert!(replacement.register_frame(frame).is_ok());
        assert_eq!(replacement.barrier_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn cell_refuses_envelope_without_matching_controller_attestation() {
        let fixture = authenticated_register_fixture().await;
        let reply = fixture.signed_reply(0, b"a");
        assert!(
            verify_reply(
                &fixture.connected,
                &fixture.verifier,
                &fixture.registration,
                &reply,
                0,
            )
            .is_ok()
        );
        assert!(
            verify_reply(
                &fixture.connected,
                &fixture.verifier,
                &fixture.registration,
                &fixture.with_envelope(reply, b"b"),
                0,
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn controller_attestation_rejects_same_instance_with_changed_worker_address() {
        let fixture = authenticated_register_fixture().await;
        let reply = fixture.signed_reply(0, b"a");

        assert!(
            verify_reply(
                &fixture.connected_with_changed_worker_address(),
                &fixture.verifier,
                &fixture.registration,
                &reply,
                0,
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn controller_attestation_rejects_changed_dial_address_or_reply_payload() {
        let fixture = authenticated_register_fixture().await;
        let reply = fixture.signed_reply(0, b"a");
        assert!(
            verify_reply(
                &fixture.with_dial_port(1),
                &fixture.verifier,
                &fixture.registration,
                &reply,
                0,
            )
            .is_err()
        );
        assert!(
            verify_reply(
                &fixture.connected,
                &fixture.verifier,
                &fixture.registration,
                &fixture.with_envelope(reply, b"changed"),
                0,
            )
            .is_err()
        );
    }
    use crate::cellular::heartbeat::HeartbeatAccumulator;
    use crate::cellular::shard::RecordsShardPartition;
    use crate::cellular::transport::connect::{BindSpec, build_velo};
    use crate::metrics_core::ingest::RecordIngest;
    use crate::metrics_core::window::Phase;

    fn sample_partition(cell_id: u32) -> RecordsShardPartition {
        let mut record = RecordIngest::minimal(1_000, 5_000, Phase::Profiling);
        record.request_index = Some(cell_id as usize);
        RecordsShardPartition::new(cell_id, vec![record])
    }

    fn sample_heartbeat() -> crate::cellular::heartbeat::MetricsHeartbeat {
        let mut accumulator = HeartbeatAccumulator::new();
        accumulator.observe(Some(20.0), Some(5.0), Some(50.0));
        accumulator.snapshot(1, Default::default(), Default::default())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cell_registers_ships_heartbeat_and_partition() {
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.messenger().peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(4).expect("authority");
        let plan_registration: PlanRegistration = Arc::new(|verified| {
            let register: CellRegister = verified.decode_payload()?;
            Ok(Some(CellRegistrationPlan {
                envelope: vec![register.cell_id as u8, 0xAB],
                artifact: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            plan_registration,
            1,
            start_handle,
        )
        .expect("bind");

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect_authenticated(
            cell_velo,
            controller_peer,
            Arc::new(credentials[3].clone()),
        )
        .expect("connect");

        let reply = cell
            .register_with_credential(3, None, &credentials[3])
            .await
            .expect("register");
        assert_eq!(reply.envelope, vec![3_u8, 0xAB]);
        assert_eq!(reply.start_event, start_handle);

        cell.send(&CellMessage::Heartbeat {
            cell_id: 3,
            heartbeat: Box::new(sample_heartbeat()),
        })
        .await
        .expect("ship heartbeat");
        cell.send(&CellMessage::Partition(sample_partition(3)))
            .await
            .expect("ship partition");

        let mut heartbeats = 0;
        let mut partitions = 0;
        for _ in 0..2 {
            match controller.recv().await.expect("recv").expect("some") {
                CellMessage::Preflight { .. } => {
                    panic!("preflight uses its dedicated controller barrier");
                }
                CellMessage::Heartbeat { cell_id, .. } => {
                    assert_eq!(cell_id, 3);
                    heartbeats += 1;
                }
                CellMessage::Partition(partition) => {
                    assert_eq!(partition.len(), 1);
                    partitions += 1;
                }
                CellMessage::StorePartition(partition) => {
                    panic!("this test ships a records partition, not a store: {partition:?}");
                }
                CellMessage::PhaseSignal { phase, signal, .. } => {
                    panic!(
                        "this test ships a heartbeat and a records partition, not a phase signal: \
                         {phase} {signal:?}"
                    );
                }
            }
        }
        assert_eq!((heartbeats, partitions), (1, 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn artifact_registration_requires_controller_minted_proof_before_planning() {
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.messenger().peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let plan_calls = Arc::new(AtomicUsize::new(0));
        let seen = plan_calls.clone();
        let plan_registration: PlanRegistration = Arc::new(move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Ok(Some(CellRegistrationPlan {
                envelope: vec![0xA5],
                artifact: None,
            }))
        });
        let _controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            plan_registration,
            1,
            start.handle(),
        )
        .expect("bind");

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let cell = VeloCellClient::connect(cell_velo, controller_peer).expect("connect");
        assert!(
            cell.register_with_artifact_capability(0, Some([0x11; 32]))
                .await
                .is_err()
        );
        assert_eq!(plan_calls.load(Ordering::Relaxed), 0);

        let reply = cell
            .register_with_credential(0, Some([0x11; 32]), &credentials[0])
            .await
            .expect("controller-minted proof registers");
        assert_eq!(reply.envelope, vec![0xA5]);
        assert_eq!(plan_calls.load(Ordering::Relaxed), 1);
        assert!(
            cell.register_with_credential(0, Some([0x11; 32]), &credentials[0])
                .await
                .is_err(),
            "a newly sealed frame is a changed transcript, not an exact wire retry"
        );
        assert_eq!(plan_calls.load(Ordering::Relaxed), 1);
        assert!(
            cell.register_with_credential(0, Some([0x22; 32]), &credentials[0])
                .await
                .is_err()
        );
        assert_eq!(plan_calls.load(Ordering::Relaxed), 1);
    }

    // A cell ships its partition from a DIFFERENT velo instance than the one it
    // registered with (as in a real cell, the registration instance is gone by
    // ship time). The partition ship carries its peer, so the controller can ack
    // the fresh instance even though it only saw the register instance.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ship_from_a_fresh_instance_is_acked() {
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.messenger().peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let plan_registration: PlanRegistration = Arc::new(|_| {
            Ok(Some(CellRegistrationPlan {
                envelope: vec![1_u8],
                artifact: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            plan_registration,
            1,
            start_handle,
        )
        .expect("bind");

        let cell_a = build_velo(BindSpec::TcpLoopback).await.expect("cell A");
        let client_a = VeloCellClient::connect(cell_a, controller_peer.clone()).expect("connect A");
        client_a
            .register_with_credential(0, None, &credentials[0])
            .await
            .expect("register");

        let cell_b = build_velo(BindSpec::TcpLoopback).await.expect("cell B");
        let mut client_b = VeloCellClient::connect_authenticated(
            cell_b,
            controller_peer,
            Arc::new(credentials[0].clone()),
        )
        .expect("connect B");
        client_b
            .send(&CellMessage::Partition(sample_partition(0)))
            .await
            .expect("ship from fresh instance");

        match controller.recv().await.expect("recv").expect("some") {
            CellMessage::Partition(partition) => assert_eq!(partition.len(), 1),
            other => panic!("expected partition, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn forged_partition_ship_is_rejected_before_delivery() {
        #[derive(Serialize)]
        struct UnprovenPartitionShip {
            cell_id: u32,
            cell_peer: Vec<u8>,
            partition: RecordsShardPartition,
        }

        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo.event_manager().new_event().expect("start");
        let (authority, _) = CellRegistrationAuthority::mint(1).expect("authority");
        let plan_registration: PlanRegistration = Arc::new(|_| Ok(None));
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            plan_registration,
            1,
            start.handle(),
        )
        .expect("bind");

        let attacker = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("attacker velo");
        attacker
            .register_peer(controller_peer.clone())
            .expect("trusted controller route");
        let body = rmp_serde::to_vec(&UnprovenPartitionShip {
            cell_id: 0,
            cell_peer: rmp_serde::to_vec(&attacker.peer_info()).expect("peer"),
            partition: sample_partition(0),
        })
        .expect("encode forged ship");
        assert!(
            attacker
                .unary(HANDLER_PARTITION)
                .expect("unary")
                .raw_payload(Bytes::from(body))
                .instance(controller_peer.instance_id())
                .send()
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), controller.recv())
                .await
                .is_err()
        );
    }

    // A metrics-only cell ships a folded `StorePartition` through
    // `CellMessage::StorePartition` → rmp raw payload →
    // `HANDLER_STORE_PARTITION` → ack, preserving its record count for append-merge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cell_ships_folded_store_partition() {
        use crate::cellular::shard::ColumnStorePartition;
        use crate::metrics_core::accumulator::MetricsAccumulator;

        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.messenger().peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let plan_registration: PlanRegistration = Arc::new(|_| {
            Ok(Some(CellRegistrationPlan {
                envelope: vec![7_u8],
                artifact: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            plan_registration,
            1,
            start_handle,
        )
        .expect("bind");

        // A folded store: a handful of completed records processed into an accumulator.
        let mut accumulator = MetricsAccumulator::new();
        for idx in 0..5u64 {
            let mut record =
                RecordIngest::minimal(1_000 + idx as i64 * 10, 5_000, Phase::Profiling);
            record.request_index = None;
            accumulator.process_record(&record);
        }
        let partition = ColumnStorePartition::from_accumulator(0, &accumulator);

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect_authenticated(
            cell_velo,
            controller_peer,
            Arc::new(credentials[0].clone()),
        )
        .expect("connect");
        cell.register_with_credential(0, None, &credentials[0])
            .await
            .expect("register");
        cell.send(&CellMessage::StorePartition(Box::new(partition)))
            .await
            .expect("ship folded store");

        match controller.recv().await.expect("recv").expect("some") {
            CellMessage::StorePartition(partition) => {
                assert_eq!(partition.cell_id(), 0);
                assert_eq!(partition.record_count(), 5);
            }
            other => panic!("expected store partition, got {other:?}"),
        }
    }

    // The synchronized-start barrier: two cells register, the controller's
    // `await_all_registered` releases once both have, and both cells' `await_start`
    // resolve after the controller triggers the START event.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn synchronized_start_releases_all_cells_together() {
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.messenger().peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(2).expect("authority");
        let plan_registration: PlanRegistration = Arc::new(|_| {
            Ok(Some(CellRegistrationPlan {
                envelope: vec![9_u8],
                artifact: None,
            }))
        });
        let controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            plan_registration,
            2,
            start_handle,
        )
        .expect("bind");

        let cell_a_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell A");
        let cell_a = VeloCellClient::connect(cell_a_velo, controller_peer.clone()).expect("A");
        let cell_b_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell B");
        let cell_b = VeloCellClient::connect(cell_b_velo, controller_peer).expect("B");

        let reply_a = cell_a
            .register_with_credential(0, None, &credentials[0])
            .await
            .expect("register A");
        let reply_b = cell_b
            .register_with_credential(1, None, &credentials[1])
            .await
            .expect("register B");
        assert_eq!(reply_a.start_event, start_handle);

        // Both cells registered, so the barrier is released immediately; trigger START.
        controller.await_all_registered().await;
        start.trigger().expect("trigger start");

        cell_a
            .await_start(reply_a.start_event)
            .await
            .expect("A start");
        cell_b
            .await_start(reply_b.start_event)
            .await
            .expect("B start");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controller_receives_cell_phase_complete_notification() {
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.messenger().peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let plan_registration: PlanRegistration = Arc::new(|_| {
            Ok(Some(CellRegistrationPlan {
                envelope: vec![0xCD],
                artifact: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            plan_registration,
            1,
            start_handle,
        )
        .expect("bind");

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect_authenticated(
            cell_velo,
            controller_peer,
            Arc::new(credentials[0].clone()),
        )
        .expect("connect");

        cell.send(&CellMessage::PhaseSignal {
            cell_id: 0,
            phase: "profiling".to_owned(),
            signal: CellPhaseSignal::Complete,
        })
        .await
        .expect("ship phase signal");

        match controller.recv().await.expect("recv").expect("some") {
            CellMessage::PhaseSignal {
                cell_id,
                phase,
                signal,
            } => {
                assert_eq!(cell_id, 0);
                assert_eq!(phase, "profiling");
                assert_eq!(signal, CellPhaseSignal::Complete);
            }
            other => panic!("expected phase signal, got {other:?}"),
        }
    }
}
