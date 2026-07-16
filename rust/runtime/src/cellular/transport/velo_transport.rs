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
use std::sync::atomic::{AtomicU32, Ordering};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::sync::mpsc;
use velo::{Context, EventHandle, Handler, PeerInfo, Velo};

use super::{
    CellAck, CellClient, CellMessage, CellPartitionShip, CellRegister, CellStorePartitionShip,
    CellTransportError, ControllerTransport, HANDLER_HEARTBEAT, HANDLER_PARTITION,
    HANDLER_REGISTER, HANDLER_STORE_PARTITION,
};

/// Supplies each cell's serialized (`rmp`) `CellLaunchSpec` by `cell_id`, or
/// `None` if the `cell_id` is out of range. The controller precomputes every
/// cell's spec before binding, so the register handler is a pure lookup.
pub type SpecFor = Arc<dyn Fn(u32) -> Option<Vec<u8>> + Send + Sync>;

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

/// The controller's velo endpoint: registers the three handlers and exposes a
/// merged [`ControllerTransport::recv`] stream of every cell's decoded messages.
pub struct VeloControllerTransport {
    /// Held so the velo instance (and thus its registered handlers) outlives the
    /// transport; dropping it tears the messaging plane down.
    _velo: Arc<Velo>,
    receiver: mpsc::Receiver<Result<CellMessage, CellTransportError>>,
    /// Notified once every `cell_count` cell has registered — the controller then
    /// triggers the START event (synchronized start).
    all_registered: Arc<Notify>,
}

impl VeloControllerTransport {
    /// Register the register/heartbeat/partition handlers on `velo` and return the
    /// controller transport. `spec_for` supplies each registering cell's
    /// `CellLaunchSpec` bytes; `start_event` is the run-wide START handle returned
    /// to each cell; the barrier fires once all `cell_count` cells have registered.
    pub fn bind_controller(
        velo: Arc<Velo>,
        spec_for: SpecFor,
        cell_count: u32,
        start_event: EventHandle,
    ) -> Result<Self, CellTransportError> {
        let (sender, receiver) = mpsc::channel(1024);
        let all_registered = Arc::new(Notify::new());
        let registered = Arc::new(AtomicU32::new(0));

        // register (unary): learn the cell, count it toward the start barrier, and
        // return its spec + the START handle it must await before dispatching.
        let reg_notify = all_registered.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_REGISTER, move |ctx: Context| {
                let spec_for = spec_for.clone();
                let registered = registered.clone();
                let reg_notify = reg_notify.clone();
                async move {
                    let register: CellRegister = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode CellRegister: {error}"))?;
                    let peer: PeerInfo = rmp_serde::from_slice(&register.cell_peer)
                        .map_err(|error| anyhow::anyhow!("decode cell PeerInfo: {error}"))?;
                    ctx.msg
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer cell: {error}"))?;
                    let Some(envelope) = spec_for(register.cell_id) else {
                        return Err(anyhow::anyhow!(
                            "no launch spec for cell {}",
                            register.cell_id
                        ));
                    };
                    // Each cell registers exactly once; the Nth registration releases
                    // the start barrier so the controller triggers START.
                    if registered.fetch_add(1, Ordering::SeqCst) + 1 == cell_count {
                        reg_notify.notify_one();
                    }
                    let reply = RegisterReply {
                        envelope,
                        start_event,
                    };
                    let bytes = rmp_serde::to_vec(&reply)
                        .map_err(|error| anyhow::anyhow!("encode RegisterReply: {error}"))?;
                    Ok(Some(Bytes::from(bytes)))
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
        let partition_sender = sender.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_PARTITION, move |ctx: Context| {
                let sender = partition_sender.clone();
                async move {
                    let ship: CellPartitionShip = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode partition ship: {error}"))?;
                    // The cell ships from a fresh velo instance the controller has
                    // not seen; register it so the ack routes back.
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

        // store_partition (unary): the Stage-C exact-fold sibling of `partition` — push
        // the decoded folded column-store partition, reply with an ack.
        let store_partition_sender = sender;
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_STORE_PARTITION, move |ctx: Context| {
                let sender = store_partition_sender.clone();
                async move {
                    let ship: CellStorePartitionShip = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode store partition ship: {error}"))?;
                    // The cell ships from a fresh velo instance the controller has
                    // not seen; register it so the ack routes back.
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
        })
    }

    /// Resolves once every `cell_count` cell has registered. The controller awaits
    /// this (with a deadline) before triggering the START event.
    pub async fn await_all_registered(&self) {
        self.all_registered.notified().await;
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

    /// Send the registration request and return the controller's [`RegisterReply`]
    /// (the cell's sliced execute envelope + the START event handle to await).
    pub async fn register(&self, cell_id: u32) -> Result<RegisterReply, CellTransportError> {
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
        rmp_serde::from_slice(&reply).map_err(decode)
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
            CellMessage::Partition(partition) => {
                // Ship carries this instance's PeerInfo so the controller can ack back.
                let ship = CellPartitionShip {
                    cell_peer: rmp_serde::to_vec(&self.velo.peer_info()).map_err(encode)?,
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
                let ship = CellStorePartitionShip {
                    cell_peer: rmp_serde::to_vec(&self.velo.peer_info()).map_err(encode)?,
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
        let controller_velo = build_velo(BindSpec::TcpLoopback)
            .await
            .expect("controller velo");
        let controller_peer = controller_velo.peer_info();
        let start = controller_velo
            .event_manager()
            .new_event()
            .expect("start event");
        let start_handle = start.handle();
        let spec_for: SpecFor = Arc::new(|cell_id: u32| Some(vec![cell_id as u8, 0xAB]));
        let mut controller =
            VeloControllerTransport::bind_controller(controller_velo, spec_for, 1, start_handle)
                .expect("bind");

        // Cell: bind velo, register (controller peer handed directly, as the
        // bootstrap would), verify the spec reply, then ship a heartbeat + partition.
        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect(cell_velo, controller_peer).expect("connect");

        let reply = cell.register(3).await.expect("register");
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
                CellMessage::StorePartition(partition) => {
                    panic!("this test ships a records partition, not a store: {partition:?}");
                }
            }
        }
        assert_eq!((heartbeats, partitions), (1, 1));
    }

    /// A cell ships its partition from a DIFFERENT velo instance than the one it
    /// registered with (mirroring the real cell: spec-fetch instance is gone by
    /// ship time). The partition ship carries its peer, so the controller can ack
    /// the fresh instance even though it only saw the register instance.
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
        let spec_for: SpecFor = Arc::new(|_| Some(vec![1_u8]));
        let mut controller =
            VeloControllerTransport::bind_controller(controller_velo, spec_for, 1, start_handle)
                .expect("bind");

        // Register with instance A.
        let cell_a = build_velo(BindSpec::TcpLoopback).await.expect("cell A");
        let client_a = VeloCellClient::connect(cell_a, controller_peer.clone()).expect("connect A");
        client_a.register(0).await.expect("register");

        // Ship from a fresh instance B — the controller never saw B via register.
        let cell_b = build_velo(BindSpec::TcpLoopback).await.expect("cell B");
        let mut client_b = VeloCellClient::connect(cell_b, controller_peer).expect("connect B");
        client_b
            .send(&CellMessage::Partition(sample_partition(0)))
            .await
            .expect("ship from fresh instance");

        match controller.recv().await.expect("recv").expect("some") {
            CellMessage::Partition(partition) => assert_eq!(partition.len(), 1),
            other => panic!("expected partition, got {other:?}"),
        }
    }

    /// Stage C: a metrics-only cell ships a folded `StorePartition` over the new
    /// store handler; the controller decodes it, acks, and surfaces it on the merged
    /// stream. Proves the wire path (`CellMessage::StorePartition` → rmp raw payload →
    /// `HANDLER_STORE_PARTITION` → ack) works over real velo, preserving the store's
    /// record count for the append-merge.
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
        let spec_for: SpecFor = Arc::new(|_| Some(vec![7_u8]));
        let mut controller =
            VeloControllerTransport::bind_controller(controller_velo, spec_for, 1, start_handle)
                .expect("bind");

        // A folded store: a handful of completed records processed into an accumulator.
        let mut accumulator = MetricsAccumulator::new();
        for idx in 0..5u64 {
            let mut record =
                RecordIngest::minimal(1_000 + idx as i64 * 10, 5_000, Phase::Profiling);
            record.request_index = None;
            accumulator.process_record(&record);
        }
        let partition = ColumnStorePartition::from_accumulator(4, &accumulator);

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect(cell_velo, controller_peer).expect("connect");
        cell.register(0).await.expect("register");
        cell.send(&CellMessage::StorePartition(Box::new(partition)))
            .await
            .expect("ship folded store");

        match controller.recv().await.expect("recv").expect("some") {
            CellMessage::StorePartition(partition) => {
                assert_eq!(partition.cell_id(), 4);
                assert_eq!(partition.record_count(), 5);
            }
            other => panic!("expected store partition, got {other:?}"),
        }
    }

    /// The synchronized-start barrier: two cells register, the controller's
    /// `await_all_registered` releases once both have, and both cells' `await_start`
    /// resolve after the controller triggers the START event.
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
        let spec_for: SpecFor = Arc::new(|_| Some(vec![9_u8]));
        let controller =
            VeloControllerTransport::bind_controller(controller_velo, spec_for, 2, start_handle)
                .expect("bind");

        let cell_a_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell A");
        let cell_a = VeloCellClient::connect(cell_a_velo, controller_peer.clone()).expect("A");
        let cell_b_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell B");
        let cell_b = VeloCellClient::connect(cell_b_velo, controller_peer).expect("B");

        let reply_a = cell_a.register(0).await.expect("register A");
        let reply_b = cell_b.register(1).await.expect("register B");
        assert_eq!(reply_a.start_event, start_handle);

        // Both cells registered, so the barrier is released immediately; trigger START.
        controller.await_all_registered().await;
        start.trigger().expect("trigger start");

        // Both cells' awaits resolve now that START fired.
        cell_a
            .await_start(reply_a.start_event)
            .await
            .expect("A start");
        cell_b
            .await_start(reply_b.start_event)
            .await
            .expect("B start");
    }
}
