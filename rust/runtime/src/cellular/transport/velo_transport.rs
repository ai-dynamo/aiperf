// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The velo-backed cell↔controller transport.
//!
//! Realizes the [`CellClient`] / [`ControllerTransport`] seam over the Velo 0.5
//! messaging framework. Five named handlers on the
//! controller carry the whole protocol:
//!
//! - [`HANDLER_REGISTER`] (unary): a cell sends its [`CellRegister`] (its own
//!   `PeerInfo` + `cell_id`); the controller `register_peer`s it (so replies and
//!   later messages route back) and returns that cell's serialized `CellLaunchSpec`.
//! - [`HANDLER_HEARTBEAT`] (fire-and-forget): a cell's periodic
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
use super::{
    ArtifactChannelServerConfig, CellAck, CellClient, CellMessage, CellPartitionShip, CellRegister,
    CellStorePartitionShip, CellTransportError, ControllerTransport, HANDLER_HEARTBEAT,
    HANDLER_PARTITION, HANDLER_PHASE_SIGNAL, HANDLER_PREFLIGHT, HANDLER_REGISTER,
    HANDLER_STORE_PARTITION,
};
use crate::engine::cellular_registration::{
    CellPeerAdmissionPurpose, CellRegistrationAuthority, CellRegistrationCredential,
};

/// Per-cell material returned only after the controller accepts registration.
#[derive(Clone)]
pub struct CellRegistrationSpec {
    /// The cell's sliced execute envelope.
    pub envelope: Vec<u8>,
    /// The public certificate for the authenticated TLS artifact channel.
    pub artifact_channel: Option<ArtifactChannelServerConfig>,
}

/// Validates a complete registration request and returns its per-cell material.
pub type SpecFor =
    Arc<dyn Fn(&CellRegister) -> anyhow::Result<Option<CellRegistrationSpec>> + Send + Sync>;

/// The controller's reply to a cell's registration: the cell's sliced execute
/// envelope plus the handle of the run-wide **START** event. The cell awaits that
/// event before dispatching, so every cell begins the benchmark together once the
/// controller has seen all `cell_count` registrations (synchronized start).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterReply {
    /// The cell's sliced execute envelope (protocol-v2 JSON bytes).
    pub envelope: Vec<u8>,
    /// The controller-owned START event handle the cell awaits before dispatching.
    pub start_event: EventHandle,
    /// The public artifact TLS certificate, when cross-host transfer is enabled.
    pub artifact_channel: Option<ArtifactChannelServerConfig>,
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
    /// triggers the START event (synchronized start).
    all_registered: Arc<Notify>,
    preflight: Arc<crate::graph::supplement::GraphCellPreflightBarrier>,
}

impl VeloControllerTransport {
    /// Register the register/control/partition handlers on `velo` and return the
    /// controller transport. `spec_for` supplies each registering cell's
    /// `CellLaunchSpec` bytes; `start_event` is the run-wide START handle returned
    /// to each cell; the barrier fires once all `cell_count` cells have registered.
    pub(crate) fn bind_controller(
        velo: Arc<Velo>,
        registration_authority: Arc<CellRegistrationAuthority>,
        spec_for: SpecFor,
        cell_count: u32,
        start_event: EventHandle,
    ) -> Result<Self, CellTransportError> {
        let (sender, receiver) = mpsc::channel(1024);
        let all_registered = Arc::new(Notify::new());
        let preflight = Arc::new(crate::graph::supplement::GraphCellPreflightBarrier::new(
            cell_count,
        ));
        let registered = Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
        let partition_authority = registration_authority.clone();
        let store_partition_authority = registration_authority.clone();

        // register (unary): learn the cell, count it toward the start barrier, and
        // return its spec + the START handle it must await before dispatching.
        let reg_notify = all_registered.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_REGISTER, move |ctx: Context| {
                let spec_for = spec_for.clone();
                let registration_authority = registration_authority.clone();
                let registered = registered.clone();
                let reg_notify = reg_notify.clone();
                async move {
                    let register: CellRegister = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode CellRegister: {error}"))?;
                    let verified = registration_authority.verify(&register)?;
                    ensure!(
                        verified.cell_id() == register.cell_id,
                        "cell registration proof identity does not match its request"
                    );
                    let peer: PeerInfo = rmp_serde::from_slice(&register.cell_peer)
                        .map_err(|error| anyhow::anyhow!("decode cell PeerInfo: {error}"))?;
                    let registration_fingerprint = blake3::hash(&ctx.payload);
                    if let Some(existing) = registered.lock().get(&register.cell_id) {
                        ensure!(
                            *existing == registration_fingerprint,
                            "cell registration retry changed its authenticated identity"
                        );
                    }
                    let Some(spec) = spec_for(&register)? else {
                        return Err(anyhow::anyhow!(
                            "no launch spec for cell {}",
                            register.cell_id
                        ));
                    };
                    let is_new_registration = {
                        let mut registered = registered.lock();
                        match registered.get(&register.cell_id) {
                            Some(existing) => {
                                ensure!(
                                    *existing == registration_fingerprint,
                                    "cell registration retry changed its authenticated identity"
                                );
                                false
                            }
                            None => {
                                registered.insert(register.cell_id, registration_fingerprint);
                                true
                            }
                        }
                    };
                    ctx.msg
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer cell: {error}"))?;
                    // A successful exact retry returns the same material without
                    // counting one cell twice toward the synchronized start barrier.
                    let is_last_registration =
                        is_new_registration && registered.lock().len() == cell_count as usize;
                    if is_last_registration {
                        reg_notify.notify_one();
                    }
                    let reply = RegisterReply {
                        envelope: spec.envelope,
                        start_event,
                        artifact_channel: spec.artifact_channel,
                    };
                    let bytes = rmp_serde::to_vec(&reply)
                        .map_err(|error| anyhow::anyhow!("encode RegisterReply: {error}"))?;
                    Ok(Some(Bytes::from(bytes)))
                }
            })
            .build(),
        )
        .map_err(io)?;

        let preflight_barrier = preflight.clone();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_PREFLIGHT, move |ctx: Context| {
                let preflight_barrier = preflight_barrier.clone();
                async move {
                    match rmp_serde::from_slice::<CellMessage>(&ctx.payload) {
                        Ok(CellMessage::Preflight { cell_id, result }) => {
                            preflight_barrier.report(cell_id, result);
                        }
                        Ok(_) => {}
                        Err(error) => {
                            tracing::warn!(error = %error, "invalid cellular preflight message")
                        }
                    }
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(io)?;

        // heartbeat (fire-and-forget): push the decoded CellMessage::Heartbeat.
        let heartbeat_sender = sender.clone();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_HEARTBEAT, move |ctx: Context| {
                let sender = heartbeat_sender.clone();
                async move {
                    match rmp_serde::from_slice::<CellMessage>(&ctx.payload) {
                        Ok(message) => {
                            let _ = sender.send(Ok(message)).await;
                        }
                        Err(error) => {
                            let _ = sender
                                .send(Err(CellTransportError::Decode(format!(
                                    "heartbeat: {error}"
                                ))))
                                .await;
                        }
                    }
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(io)?;

        // phase_signal (fire-and-forget): push the decoded CellMessage::PhaseSignal.
        let phase_signal_sender = sender.clone();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_PHASE_SIGNAL, move |ctx: Context| {
                let sender = phase_signal_sender.clone();
                async move {
                    match rmp_serde::from_slice::<CellMessage>(&ctx.payload) {
                        Ok(message) => {
                            let _ = sender.send(Ok(message)).await;
                        }
                        Err(error) => {
                            let _ = sender
                                .send(Err(CellTransportError::Decode(format!(
                                    "phase signal: {error}"
                                ))))
                                .await;
                        }
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
                    let ship: CellPartitionShip = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode partition ship: {error}"))?;
                    ensure!(
                        ship.partition.cell_id() == ship.cell_id,
                        "partition ship identity does not match its partition"
                    );
                    registration_authority.verify_peer_admission(
                        ship.cell_id,
                        CellPeerAdmissionPurpose::Partition,
                        &ship.cell_peer,
                        &ship.admission_proof,
                    )?;
                    // The cell ships from a fresh velo instance the controller has
                    // not seen. Its exact peer proof is checked before admission.
                    let peer: PeerInfo = rmp_serde::from_slice(&ship.cell_peer)
                        .map_err(|error| anyhow::anyhow!("decode ship peer: {error}"))?;
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
                    let ship: CellStorePartitionShip = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode store partition ship: {error}"))?;
                    ensure!(
                        ship.partition.cell_id() == ship.cell_id,
                        "store partition ship identity does not match its partition"
                    );
                    registration_authority.verify_peer_admission(
                        ship.cell_id,
                        CellPeerAdmissionPurpose::StorePartition,
                        &ship.cell_peer,
                        &ship.admission_proof,
                    )?;
                    // The cell ships from a fresh velo instance the controller has
                    // not seen. Its exact peer proof is checked before admission.
                    let peer: PeerInfo = rmp_serde::from_slice(&ship.cell_peer)
                        .map_err(|error| anyhow::anyhow!("decode store ship peer: {error}"))?;
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
/// fetches the cell's launch spec, and [`CellClient::send`] ships control messages and
/// the final partition.
pub struct VeloCellClient {
    velo: Arc<Velo>,
    controller: PeerInfo,
    registration_peer: Vec<u8>,
    credential: Option<Arc<CellRegistrationCredential>>,
}

impl VeloCellClient {
    /// Register the controller peer so the cell can address it, and return the client.
    pub fn connect(velo: Arc<Velo>, controller: PeerInfo) -> Result<Self, CellTransportError> {
        velo.register_peer(controller.clone()).map_err(io)?;
        let registration_peer = rmp_serde::to_vec(&velo.peer_info()).map_err(encode)?;
        Ok(Self {
            velo,
            controller,
            registration_peer,
            credential: None,
        })
    }

    /// Build a client with the per-cell credential needed for fresh peer tickets.
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
            self.registration_peer.clone(),
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
        let body = rmp_serde::to_vec(&CellRegister {
            cell_id,
            cell_peer,
            artifact_capability_digest,
            registration_proof,
        })
        .map_err(encode)?;
        let reply: Bytes = self
            .velo
            .unary(HANDLER_REGISTER)
            .map_err(io)?
            .raw_payload(Bytes::from(body))
            .instance(self.controller.instance_id())
            .send()
            .await
            .map_err(io)?;
        rmp_serde::from_slice(&reply).map_err(decode)
    }

    /// Sign the exact peer bytes and capability digest before registration.
    pub(crate) async fn register_with_credential(
        &self,
        cell_id: u32,
        artifact_capability_digest: Option<[u8; 32]>,
        credential: &CellRegistrationCredential,
    ) -> Result<RegisterReply, CellTransportError> {
        let peer = self.registration_peer.clone();
        let proof = credential
            .sign_register(&peer, artifact_capability_digest)
            .map_err(|error| CellTransportError::Encode(error.to_string()))?;
        self.send_registration(cell_id, peer, artifact_capability_digest, Some(proof))
            .await
    }

    /// Block until the controller triggers the run-wide START event (a synchronized
    /// start: every cell resumes together once all cells have registered). A
    /// poisoned event (the controller aborted before starting) surfaces as an error.
    pub async fn await_start(&self, start_event: EventHandle) -> Result<(), CellTransportError> {
        self.velo
            .event_manager()
            .awaiter(start_event)
            .map_err(io)?
            .await
            .map_err(io)
    }
}

#[async_trait::async_trait]
impl CellClient for VeloCellClient {
    async fn send(&mut self, message: &CellMessage) -> Result<(), CellTransportError> {
        match message {
            CellMessage::Preflight { .. } => {
                let body = rmp_serde::to_vec(message).map_err(encode)?;
                self.velo
                    .am_send(HANDLER_PREFLIGHT)
                    .map_err(io)?
                    .raw_payload(Bytes::from(body))
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
            }
            CellMessage::Heartbeat { .. } => {
                // Fire-and-forget: no ack, so the controller needs no return route.
                let body = rmp_serde::to_vec(message).map_err(encode)?;
                self.velo
                    .am_send(HANDLER_HEARTBEAT)
                    .map_err(io)?
                    .raw_payload(Bytes::from(body))
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
            }
            CellMessage::PhaseSignal { .. } => {
                let body = rmp_serde::to_vec(message).map_err(encode)?;
                self.velo
                    .am_send(HANDLER_PHASE_SIGNAL)
                    .map_err(io)?
                    .raw_payload(Bytes::from(body))
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
                let cell_peer = self.registration_peer.clone();
                let ship = CellPartitionShip {
                    cell_id: partition.cell_id(),
                    admission_proof: credential
                        .sign_peer_admission(CellPeerAdmissionPurpose::Partition, &cell_peer)
                        .map_err(encode)?,
                    cell_peer,
                    partition: partition.clone(),
                };
                let body = rmp_serde::to_vec(&ship).map_err(encode)?;
                let reply: Bytes = self
                    .velo
                    .unary(HANDLER_PARTITION)
                    .map_err(io)?
                    .raw_payload(Bytes::from(body))
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
                let cell_peer = self.registration_peer.clone();
                let ship = CellStorePartitionShip {
                    cell_id: partition.cell_id(),
                    admission_proof: credential
                        .sign_peer_admission(CellPeerAdmissionPurpose::StorePartition, &cell_peer)
                        .map_err(encode)?,
                    cell_peer,
                    partition: (**partition).clone(),
                };
                let body = rmp_serde::to_vec(&ship).map_err(encode)?;
                let reply: Bytes = self
                    .velo
                    .unary(HANDLER_STORE_PARTITION)
                    .map_err(io)?
                    .raw_payload(Bytes::from(body))
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::engine::cellular_registration::CellRegistrationAuthority;

    #[test]
    fn artifact_channel_registration_keeps_envelope_separate() {
        let public_config = ArtifactChannelServerConfig::new(vec![1, 2, 3]);
        let spec = CellRegistrationSpec {
            envelope: b"{\"run\":{}}".to_vec(),
            artifact_channel: Some(public_config),
        };
        assert_eq!(spec.envelope, b"{\"run\":{}}".to_vec());
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
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(4).expect("authority");
        let spec_for: SpecFor = Arc::new(|register| {
            Ok(Some(CellRegistrationSpec {
                envelope: vec![register.cell_id as u8, 0xAB],
                artifact_channel: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            spec_for,
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
    async fn artifact_registration_requires_controller_minted_proof_before_spec() {
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let spec_calls = Arc::new(AtomicUsize::new(0));
        let seen = spec_calls.clone();
        let spec_for: SpecFor = Arc::new(move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Ok(Some(CellRegistrationSpec {
                envelope: vec![0xA5],
                artifact_channel: None,
            }))
        });
        let _controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            spec_for,
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
        assert_eq!(spec_calls.load(Ordering::Relaxed), 0);

        let reply = cell
            .register_with_credential(0, Some([0x11; 32]), &credentials[0])
            .await
            .expect("controller-minted proof registers");
        assert_eq!(reply.envelope, vec![0xA5]);
        assert_eq!(spec_calls.load(Ordering::Relaxed), 1);
        cell.register_with_credential(0, Some([0x11; 32]), &credentials[0])
            .await
            .expect("exact authenticated retry registers");
        assert_eq!(spec_calls.load(Ordering::Relaxed), 2);
        assert!(
            cell.register_with_credential(0, Some([0x22; 32]), &credentials[0])
                .await
                .is_err()
        );
        assert_eq!(spec_calls.load(Ordering::Relaxed), 2);
    }

    // A cell ships its partition from a DIFFERENT velo instance than the one it
    // registered with (as in a real cell, the spec-fetch instance is gone by
    // ship time). The partition ship carries its peer, so the controller can ack
    // the fresh instance even though it only saw the register instance.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ship_from_a_fresh_instance_is_acked() {
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let spec_for: SpecFor = Arc::new(|_| {
            Ok(Some(CellRegistrationSpec {
                envelope: vec![1_u8],
                artifact_channel: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            spec_for,
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
        let spec_for: SpecFor = Arc::new(|_| Ok(None));
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            spec_for,
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
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let spec_for: SpecFor = Arc::new(|_| {
            Ok(Some(CellRegistrationSpec {
                envelope: vec![7_u8],
                artifact_channel: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            spec_for,
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
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) = CellRegistrationAuthority::mint(2).expect("authority");
        let spec_for: SpecFor = Arc::new(|_| {
            Ok(Some(CellRegistrationSpec {
                envelope: vec![9_u8],
                artifact_channel: None,
            }))
        });
        let controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            spec_for,
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
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let (authority, _) = CellRegistrationAuthority::mint(1).expect("authority");
        let spec_for: SpecFor = Arc::new(|_| {
            Ok(Some(CellRegistrationSpec {
                envelope: vec![0xCD],
                artifact_channel: None,
            }))
        });
        let mut controller = VeloControllerTransport::bind_controller(
            controller_velo,
            Arc::new(authority),
            spec_for,
            1,
            start_handle,
        )
        .expect("bind");

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect(cell_velo, controller_peer).expect("connect");

        cell.send(&CellMessage::PhaseSignal {
            cell_id: 7,
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
                assert_eq!(cell_id, 7);
                assert_eq!(phase, "profiling");
                assert_eq!(signal, CellPhaseSignal::Complete);
            }
            other => panic!("expected phase signal, got {other:?}"),
        }
    }
}
