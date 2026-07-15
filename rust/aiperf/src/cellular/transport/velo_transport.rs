// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The velo-backed cell↔controller transport.
//!
//! Realizes the [`CellClient`] / [`ControllerTransport`] seam over the official
//! `ai-dynamo/velo` messaging framework (v0.5.0). Three named handlers on the
//! controller carry the whole protocol:
//!
//! - [`HANDLER_REGISTER`] (unary): a cell sends its [`CellRegister`] (its own
//!   `PeerInfo` + `cell_id`); the controller `register_peer`s it (so replies and
//!   later messages route back) and returns that cell's serialized `CellLaunchSpec`.
//!   This replaces the stdin spec pipe of the process launcher.
//! - [`HANDLER_HEARTBEAT`] (fire-and-forget): a cell's periodic
//!   [`CellMessage::Heartbeat`].
//! - [`HANDLER_PARTITION`] (unary): a cell's final [`RecordsShardPartition`]; the
//!   reply is a [`CellAck`].
//!
//! All bodies are `rmp-serde` carried as velo **raw** payloads (see the module
//! docs on `super`) so t-digest `+inf` and NaN metric values survive the wire.

use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::mpsc;
use velo::{Context, Handler, PeerInfo, Velo};

use super::{
    CellAck, CellClient, CellMessage, CellRegister, CellTransportError, ControllerTransport,
    HANDLER_HEARTBEAT, HANDLER_PARTITION, HANDLER_REGISTER,
};

/// Supplies each cell's serialized (`rmp`) `CellLaunchSpec` by `cell_id`, or
/// `None` if the `cell_id` is out of range. The controller precomputes every
/// cell's spec before binding, so the register handler is a pure lookup.
pub type SpecFor = Arc<dyn Fn(u32) -> Option<Vec<u8>> + Send + Sync>;

fn encode(error: impl std::fmt::Display) -> CellTransportError {
    CellTransportError::Encode(error.to_string())
}
fn decode(error: impl std::fmt::Display) -> CellTransportError {
    CellTransportError::Decode(error.to_string())
}
fn io(error: impl std::fmt::Display) -> CellTransportError {
    CellTransportError::Io(error.to_string())
}

/// The controller's velo endpoint: registers the three handlers and exposes a
/// merged [`ControllerTransport::recv`] stream of every cell's decoded messages.
pub struct VeloControllerTransport {
    /// Held so the velo instance (and thus its registered handlers) outlives the
    /// transport; dropping it tears the messaging plane down.
    _velo: Arc<Velo>,
    receiver: mpsc::Receiver<Result<CellMessage, CellTransportError>>,
}

impl VeloControllerTransport {
    /// Register the register/heartbeat/partition handlers on `velo` and return the
    /// controller transport. `spec_for` supplies each registering cell's
    /// `CellLaunchSpec` bytes.
    pub fn bind_controller(velo: Arc<Velo>, spec_for: SpecFor) -> Result<Self, CellTransportError> {
        let (sender, receiver) = mpsc::channel(1024);

        // register (unary): learn the cell, return its spec bytes.
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_REGISTER, move |ctx: Context| {
                let spec_for = spec_for.clone();
                async move {
                    let register: CellRegister = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode CellRegister: {error}"))?;
                    let peer: PeerInfo = rmp_serde::from_slice(&register.cell_peer)
                        .map_err(|error| anyhow::anyhow!("decode cell PeerInfo: {error}"))?;
                    ctx.msg
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer cell: {error}"))?;
                    match spec_for(register.cell_id) {
                        Some(spec) => Ok(Some(Bytes::from(spec))),
                        None => Err(anyhow::anyhow!("no launch spec for cell {}", register.cell_id)),
                    }
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

        // partition (unary): push the decoded partition, reply with an ack.
        let partition_sender = sender;
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_PARTITION, move |ctx: Context| {
                let sender = partition_sender.clone();
                async move {
                    // The cell serializes the whole `CellMessage` (Partition variant);
                    // decode uniformly with the heartbeat path.
                    let message: CellMessage = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode partition: {error}"))?;
                    let _ = sender.send(Ok(message)).await;
                    let ack = rmp_serde::to_vec(&CellAck { ok: true })
                        .map_err(|error| anyhow::anyhow!("encode ack: {error}"))?;
                    Ok(Some(Bytes::from(ack)))
                }
            })
            .build(),
        )
        .map_err(io)?;

        Ok(Self {
            _velo: velo,
            receiver,
        })
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
/// [`super::connect`]); [`register`](Self::register) fetches the cell's launch
/// spec, and [`CellClient::send`] ships heartbeats and the final partition.
pub struct VeloCellClient {
    velo: Arc<Velo>,
    controller: PeerInfo,
}

impl VeloCellClient {
    /// Register the controller peer so the cell can address it, and return the client.
    pub fn connect(velo: Arc<Velo>, controller: PeerInfo) -> Result<Self, CellTransportError> {
        velo.register_peer(controller.clone()).map_err(io)?;
        Ok(Self { velo, controller })
    }

    /// Send the registration request and return the controller's reply — this
    /// cell's serialized (`rmp`) `CellLaunchSpec` bytes.
    pub async fn register(&self, cell_id: u32) -> Result<Vec<u8>, CellTransportError> {
        let cell_peer = rmp_serde::to_vec(&self.velo.peer_info()).map_err(encode)?;
        let body = rmp_serde::to_vec(&CellRegister { cell_id, cell_peer }).map_err(encode)?;
        let reply: Bytes = self
            .velo
            .unary(HANDLER_REGISTER)
            .map_err(io)?
            .raw_payload(Bytes::from(body))
            .instance(self.controller.instance_id())
            .send()
            .await
            .map_err(io)?;
        Ok(reply.to_vec())
    }
}

#[async_trait::async_trait]
impl CellClient for VeloCellClient {
    async fn send(&mut self, message: &CellMessage) -> Result<(), CellTransportError> {
        let body = rmp_serde::to_vec(message).map_err(encode)?;
        match message {
            CellMessage::Heartbeat { .. } => {
                self.velo
                    .am_send(HANDLER_HEARTBEAT)
                    .map_err(io)?
                    .raw_payload(Bytes::from(body))
                    .instance(self.controller.instance_id())
                    .send()
                    .await
                    .map_err(io)?;
            }
            CellMessage::Partition(_) => {
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
                    return Err(CellTransportError::Io("controller nacked partition".to_owned()));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        // Controller: bind velo, expose a spec_for that returns a known spec byte
        // per cell_id.
        let controller_velo = build_velo(BindSpec::TcpLoopback).await.expect("controller velo");
        let controller_peer = controller_velo.peer_info();
        let spec_for: SpecFor = Arc::new(|cell_id: u32| Some(vec![cell_id as u8, 0xAB]));
        let mut controller =
            VeloControllerTransport::bind_controller(controller_velo, spec_for).expect("bind");

        // Cell: bind velo, register (controller peer handed directly, as the
        // bootstrap would), verify the spec reply, then ship a heartbeat + partition.
        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect(cell_velo, controller_peer).expect("connect");

        let spec = cell.register(3).await.expect("register");
        assert_eq!(spec, vec![3_u8, 0xAB]);

        cell.send(&CellMessage::Heartbeat {
            cell_id: 3,
            heartbeat: Box::new(sample_heartbeat()),
        })
        .await
        .expect("ship heartbeat");
        cell.send(&CellMessage::Partition(sample_partition(3)))
            .await
            .expect("ship partition");

        // Controller receives both, in ship order.
        let mut heartbeats = 0;
        let mut partitions = 0;
        for _ in 0..2 {
            match controller.recv().await.expect("recv").expect("some") {
                CellMessage::Heartbeat { cell_id, .. } => {
                    assert_eq!(cell_id, 3);
                    heartbeats += 1;
                }
                CellMessage::Partition(partition) => {
                    assert_eq!(partition.len(), 1);
                    partitions += 1;
                }
            }
        }
        assert_eq!((heartbeats, partitions), (1, 1));
    }
}
