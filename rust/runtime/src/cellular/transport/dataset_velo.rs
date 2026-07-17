// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo distribution for the dataset data plane.
//!
//! Makes the in-process [`DatasetPublisher`]/[`DatasetIndex`] a distributed fan-out: the
//! controller generates the dataset once and `add`s chunks; each cell subscribes over
//! velo and receives **replay-on-attach** (every chunk so far in the subscribe reply)
//! then the **live tail** (pushed as more chunks land), and builds its owned index. Two
//! handlers, identical in shape to [`phaser_velo`](super::phaser_velo):
//!
//! - `aiperf.dataset.subscribe` (unary, cell → controller): the cell sends its
//!   `PeerInfo`; the controller atomically attaches a broadcast consumer, returns
//!   the snapshot, and spawns a pump forwarding the tail.
//! - `aiperf.dataset.chunk` (fire-and-forget, controller → cell): one pushed chunk event.
//!
//! The payload is opaque `Vec<u8>` (the cell decodes its own request shape).

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use velo::{Context, Handler, PeerInfo, Velo};

use crate::cellular::broadcast::{BroadcastEvent, Subscription};
use crate::cellular::dataset_session::{DatasetChunk, DatasetIndex, DatasetPublisher};

/// The opaque dataset payload on the wire (the cell decodes its own request shape).
pub type WirePayload = Vec<u8>;

/// zstd level for the fan-out wire (matches the artifact-shipping plane's `ZSTD_LEVEL`).
const ZSTD_LEVEL: i32 = 3;

/// Serialize `value` as MessagePack then zstd-compress it — the fan-out wire form. The
/// dataset broadcast replays whole chunks to every cell, so compressing the redundant
/// request bodies (same model/structure, only content differs) is a real win over the
/// uncompressed rmp the phaser control plane can use (its events are tiny).
fn zpack<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    let packed = rmp_serde::to_vec(value)
        .map_err(|error| anyhow::anyhow!("encode dataset wire value: {error}"))?;
    zstd::encode_all(packed.as_slice(), ZSTD_LEVEL)
        .map_err(|error| anyhow::anyhow!("zstd-compress dataset wire value: {error}"))
}

/// Inverse of [`zpack`]: zstd-decompress then MessagePack-decode.
fn zunpack<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    let packed = zstd::decode_all(bytes)
        .map_err(|error| anyhow::anyhow!("zstd-decompress dataset wire value: {error}"))?;
    rmp_serde::from_slice(&packed)
        .map_err(|error| anyhow::anyhow!("decode dataset wire value: {error}"))
}

/// Handler: a cell subscribes to the dataset broadcast and gets the replay snapshot.
pub const HANDLER_DATASET_SUBSCRIBE: &str = "aiperf.dataset.subscribe";
/// Handler: the controller pushes one live dataset chunk to a subscribed cell.
pub const HANDLER_DATASET_CHUNK: &str = "aiperf.dataset.chunk";

#[derive(Serialize, Deserialize)]
struct DatasetSubscribeRequest {
    cell_peer: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct DatasetSubscribeReply {
    replay: Vec<BroadcastEvent<DatasetChunk<WirePayload>>>,
}

/// Controller-side dataset service. Holds the velo instance so the handler + per-cell
/// pumps outlive it.
pub struct DatasetServer {
    _velo: std::sync::Arc<Velo>,
}

impl DatasetServer {
    /// Register the `subscribe` handler on `velo`, serving the given `publisher`. Each
    /// subscribing cell gets the current replay and a pump forwarding the live tail.
    pub fn bind(
        velo: std::sync::Arc<Velo>,
        publisher: DatasetPublisher<WirePayload>,
    ) -> anyhow::Result<Self> {
        let push_velo = velo.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_DATASET_SUBSCRIBE, move |ctx: Context| {
                let publisher = publisher.clone();
                let push_velo = push_velo.clone();
                async move {
                    let request: DatasetSubscribeRequest = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| {
                            anyhow::anyhow!("decode DatasetSubscribeRequest: {error}")
                        })?;
                    let peer: PeerInfo = rmp_serde::from_slice(&request.cell_peer)
                        .map_err(|error| anyhow::anyhow!("decode cell PeerInfo: {error}"))?;
                    let cell = peer.instance_id();
                    push_velo
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer cell: {error}"))?;

                    let Subscription { replay, mut live } = publisher.attach_raw();

                    let pump_velo = push_velo.clone();
                    tokio::spawn(async move {
                        while let Some(event) = live.recv().await {
                            let terminal = matches!(event, BroadcastEvent::Finalized);
                            let Ok(body) = zpack(&event) else {
                                break;
                            };
                            let sent = match pump_velo.am_send(HANDLER_DATASET_CHUNK) {
                                Ok(builder) => {
                                    builder
                                        .raw_payload(Bytes::from(body))
                                        .instance(cell)
                                        .send()
                                        .await
                                }
                                Err(_) => break,
                            };
                            if sent.is_err() || terminal {
                                break;
                            }
                        }
                    });

                    let reply = DatasetSubscribeReply { replay };
                    let bytes = zpack(&reply)?;
                    Ok(Some(Bytes::from(bytes)))
                }
            })
            .build(),
        )
        .map_err(|error| anyhow::anyhow!("registering dataset subscribe handler: {error}"))?;
        Ok(Self { _velo: velo })
    }
}

/// Cell-side dataset client.
pub struct DatasetClient;

impl DatasetClient {
    /// Subscribe to the controller's dataset broadcast and build this cell's owned index
    /// (the requests where `owns(request_id)`), draining to `finalize`. RAM is O(owned)
    /// even though every chunk is observed.
    pub async fn build_owned_index(
        velo: std::sync::Arc<Velo>,
        controller: &PeerInfo,
        owns: impl Fn(u64) -> bool,
    ) -> anyhow::Result<DatasetIndex<WirePayload>> {
        let (tx, live) = mpsc::unbounded_channel::<BroadcastEvent<DatasetChunk<WirePayload>>>();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_DATASET_CHUNK, move |ctx: Context| {
                let tx = tx.clone();
                async move {
                    let event: BroadcastEvent<DatasetChunk<WirePayload>> = zunpack(&ctx.payload)?;
                    let _ = tx.send(event);
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(|error| anyhow::anyhow!("registering dataset chunk handler: {error}"))?;

        velo.register_peer(controller.clone())
            .map_err(|error| anyhow::anyhow!("register_peer controller: {error}"))?;

        let request = DatasetSubscribeRequest {
            cell_peer: rmp_serde::to_vec(&velo.peer_info())
                .map_err(|error| anyhow::anyhow!("encode cell PeerInfo: {error}"))?,
        };
        let body = rmp_serde::to_vec(&request)
            .map_err(|error| anyhow::anyhow!("encode DatasetSubscribeRequest: {error}"))?;
        let reply_bytes: Bytes = velo
            .unary(HANDLER_DATASET_SUBSCRIBE)
            .map_err(|error| anyhow::anyhow!("dataset subscribe builder: {error}"))?
            .raw_payload(Bytes::from(body))
            .instance(controller.instance_id())
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("dataset subscribe send: {error}"))?;
        let reply: DatasetSubscribeReply = zunpack(&reply_bytes)?;

        let sub = Subscription {
            replay: reply.replay,
            live,
        };
        Ok(DatasetIndex::build_owned(sub, owns).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::dataset_session::DatasetRequest;
    use crate::cellular::transport::connect::{BindSpec, build_velo, connect_controller};

    #[tokio::test]
    async fn cells_build_disjoint_owned_indexes_over_velo() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let coordinate = format!("tcp://{}", listener.local_addr().expect("addr"));
        let controller_velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");
        let publisher = DatasetPublisher::<WirePayload>::new();
        let _server =
            DatasetServer::bind(controller_velo.clone(), publisher.clone()).expect("bind");

        // Publish 12 requests in two chunks, then finalize.
        let chunk = |ids: std::ops::Range<u64>| {
            ids.map(|request_id| DatasetRequest {
                request_id,
                payload: format!("req-{request_id}").into_bytes(),
            })
            .collect::<Vec<_>>()
        };
        publisher.add(chunk(0..6));
        publisher.add(chunk(6..12));
        publisher.finalize();

        // Two cells (of 3) build their owned indexes over velo.
        let mut owned_all = Vec::new();
        for cell_id in [0u64, 1] {
            let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
            let controller = connect_controller(&cell_velo, &coordinate)
                .await
                .expect("connect");
            let index = DatasetClient::build_owned_index(cell_velo, &controller, move |id| {
                id % 3 == cell_id
            })
            .await
            .expect("build index");
            let expected: Vec<u64> = (0..12).filter(|id| id % 3 == cell_id).collect();
            assert_eq!(
                index.owned_ids(),
                expected,
                "cell {cell_id} owned set over velo"
            );
            assert_eq!(
                index
                    .get(cell_id)
                    .map(|p| String::from_utf8_lossy(p).into_owned()),
                Some(format!("req-{cell_id}"))
            );
            owned_all.extend(index.owned_ids());
        }
        assert_eq!(
            owned_all.len(),
            8,
            "cells 0 and 1 of 3 own 4 each, disjoint"
        );
    }
}
