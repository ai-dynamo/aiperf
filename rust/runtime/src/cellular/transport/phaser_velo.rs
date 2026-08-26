// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo distribution for the monotonic [`Phaser`](crate::cellular::phaser::Phaser).
//!
//! Makes the in-process phaser a distributed control plane: the controller owns the
//! phaser and `advance`s it; cells subscribe over velo and each receives
//! **replay-on-attach** (the current generation history in the subscribe response) then
//! the **live tail** (pushed as the controller advances). Two handlers:
//!
//! - `aiperf.phaser.subscribe` (unary, cell → controller): the cell sends its
//!   `cell_id`; the surrounding `AuthenticatedFrame` carries its `PeerInfo`. The
//!   controller `register_peer`s it, atomically attaches a broadcast consumer
//!   (snapshot + live receiver split at the seam), returns the snapshot as the reply,
//!   and spawns a **pump** task that forwards the live tail to the cell.
//! - `aiperf.phaser.event` (fire-and-forget, controller → cell): each live
//!   [`BroadcastEvent`](crate::cellular::broadcast::BroadcastEvent)`<PhaseEvent>` the pump
//!   pushes, delivered into the cell's live
//!   channel.
//!
//! The replay/live split happens under the broadcast's one lock (see
//! [`Broadcast::attach`](crate::cellular::broadcast::Broadcast::attach)), so a
//! generation advanced concurrently with a subscribe lands in exactly one of {reply
//! snapshot, pushed live}, with no missed transition or double count.
//!
//! # Trust boundary
//!
//! The local benchmark deployment trusts the controller/cell routing plane: its trusted controller
//! sends raw live Velo pushes with no per-push authenticity or replay
//! rejection. The replay/live seam instead preserves generation ordering and no missed
//! legitimate transitions.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use velo::{Context, Handler, PeerInfo, Velo};

use crate::cellular::broadcast::{BroadcastEvent, Subscription};
use crate::cellular::phaser::{PhaseEvent, Phaser, PhaserSubscription};
use crate::engine::cellular_registration::{
    AdmissionPurpose, CellRegistrationAuthority, CellRegistrationCredential,
};

/// Handler: a cell subscribes to the phaser and gets the replay snapshot.
pub const HANDLER_PHASER_SUBSCRIBE: &str = "aiperf.phaser.subscribe";
/// Handler: the controller pushes one live phaser event to a subscribed cell.
pub const HANDLER_PHASER_EVENT: &str = "aiperf.phaser.event";

/// A cell's subscribe request. Its surrounding `AuthenticatedFrame` carries the
/// `PeerInfo` the controller uses to push live events back to it.
#[derive(Serialize, Deserialize)]
struct PhaserSubscribeRequest {
    cell_id: u32,
}

/// The controller's reply: the replay snapshot (everything advanced before this
/// subscribe, in generation order, plus a trailing `Finalized` if already sealed).
#[derive(Serialize, Deserialize)]
struct PhaserSubscribeReply {
    replay: Vec<BroadcastEvent<PhaseEvent>>,
}

/// Controller-side phaser service. Holds the velo instance so the registered handler
/// (and the per-cell pump tasks) outlive it; drop to tear the control plane down.
pub struct PhaserServer {
    _velo: std::sync::Arc<Velo>,
}

impl PhaserServer {
    /// Register the `subscribe` handler on `velo`, serving the given `phaser`. Each
    /// subscribing cell gets the current replay and a pump that forwards the live tail.
    pub(crate) fn bind(
        velo: std::sync::Arc<Velo>,
        phaser: Phaser,
        registration_authority: std::sync::Arc<CellRegistrationAuthority>,
    ) -> anyhow::Result<Self> {
        let push_velo = velo.clone();
        velo.register_handler(
            Handler::unary_handler_async(HANDLER_PHASER_SUBSCRIBE, move |ctx: Context| {
                let phaser = phaser.clone();
                let push_velo = push_velo.clone();
                let registration_authority = registration_authority.clone();
                async move {
                    let opened = registration_authority
                        .open_payload::<PhaserSubscribeRequest>(
                            AdmissionPurpose::PhaserSubscribe,
                            &ctx.payload,
                        )
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    let (role, _, authenticated_peer, request) = opened.into_parts();
                    anyhow::ensure!(
                        role == crate::engine::cellular_bootstrap::CellularRole::Cell(
                            request.cell_id,
                        ),
                        "AdmissionRejected"
                    );
                    let peer: PeerInfo = rmp_serde::from_slice(&authenticated_peer)
                        .map_err(|_| anyhow::anyhow!("AdmissionRejected"))?;
                    let cell = peer.instance_id();
                    // Register the cell so the pump's `am_send` can route to it.
                    push_velo
                        .register_peer(peer)
                        .map_err(|error| anyhow::anyhow!("register_peer cell: {error}"))?;

                    // Attach atomically across the snapshot/live boundary.
                    let Subscription { replay, mut live } = phaser.attach_raw();

                    // Pump the live tail to the cell. Forwarding stops when the broadcast
                    // is finalized (live yields `Finalized` then closes) or the cell goes
                    // away (`am_send` errors) — either way the task ends, not the run.
                    let pump_velo = push_velo.clone();
                    tokio::spawn(async move {
                        while let Some(event) = live.recv().await {
                            let terminal = matches!(event, BroadcastEvent::Finalized);
                            let Ok(body) = rmp_serde::to_vec(&event) else {
                                break;
                            };
                            let sent = match pump_velo.am_send(HANDLER_PHASER_EVENT) {
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

                    let reply = PhaserSubscribeReply { replay };
                    let bytes = rmp_serde::to_vec(&reply)
                        .map_err(|error| anyhow::anyhow!("encode PhaserSubscribeReply: {error}"))?;
                    Ok(Some(Bytes::from(bytes)))
                }
            })
            .build(),
        )
        .map_err(|error| anyhow::anyhow!("registering phaser subscribe handler: {error}"))?;
        Ok(Self { _velo: velo })
    }
}

/// Cell-side phaser client: subscribes to the controller's phaser and exposes a
/// [`PhaserSubscription`] over the replay it received plus the live tail the controller
/// pushes into a local channel.
pub struct PhaserClient;

impl PhaserClient {
    /// Subscribe `velo` to the controller's phaser at `controller`. Registers the
    /// `event` handler (feeds a local channel), sends the subscribe request, and returns
    /// a [`PhaserSubscription`] whose replay is the controller's snapshot and whose live
    /// tail is the pushed events — the same shape an in-process `Phaser::subscribe`
    /// yields.
    pub(crate) async fn subscribe(
        velo: std::sync::Arc<Velo>,
        controller: &PeerInfo,
        cell_id: u32,
        credential: &CellRegistrationCredential,
    ) -> anyhow::Result<PhaserSubscription> {
        // Register the push handler BEFORE subscribing, so no live event pushed between
        // the subscribe reply and handler registration is lost (the controller only
        // pushes after it has processed the subscribe, but registering first is safe).
        let (tx, live) = mpsc::unbounded_channel::<BroadcastEvent<PhaseEvent>>();
        velo.register_handler(
            Handler::am_handler_async(HANDLER_PHASER_EVENT, move |ctx: Context| {
                let tx = tx.clone();
                async move {
                    let event: BroadcastEvent<PhaseEvent> = rmp_serde::from_slice(&ctx.payload)
                        .map_err(|error| anyhow::anyhow!("decode pushed PhaseEvent: {error}"))?;
                    let _ = tx.send(event);
                    Ok(())
                }
            })
            .build(),
        )
        .map_err(|error| anyhow::anyhow!("registering phaser event handler: {error}"))?;

        velo.register_peer(controller.clone())
            .map_err(|error| anyhow::anyhow!("register_peer controller: {error}"))?;

        if credential.cell_id() != cell_id {
            anyhow::bail!("phaser credential does not match the cell identity");
        }
        let request = PhaserSubscribeRequest { cell_id };
        let body = credential.seal_payload(
            AdmissionPurpose::PhaserSubscribe,
            &velo.peer_info(),
            &request,
        )?;
        let reply_bytes: Bytes = velo
            .unary(HANDLER_PHASER_SUBSCRIBE)
            .map_err(|error| anyhow::anyhow!("phaser subscribe builder: {error}"))?
            .raw_payload(Bytes::from(body))
            .instance(controller.instance_id())
            .send()
            .await
            .map_err(|error| anyhow::anyhow!("phaser subscribe send: {error}"))?;
        let reply: PhaserSubscribeReply = rmp_serde::from_slice(&reply_bytes)
            .map_err(|error| anyhow::anyhow!("decode PhaserSubscribeReply: {error}"))?;

        // Reconstruct the broadcast subscription: replay from the reply, live from the
        // push channel — the same split the in-process broadcast produces.
        Ok(PhaserSubscription::from_subscription(Subscription {
            replay: reply.replay,
            live,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::phaser::PhaseTransition;
    use crate::cellular::transport::connect::{BindSpec, build_velo};
    use crate::engine::cellular_registration::CellRegistrationAuthority;

    #[test]
    fn raw_push_trust_boundary_is_disclosed() {
        let production_module = include_str!("phaser_velo.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("module source has a test boundary");

        assert!(production_module.contains("trusted controller"));
        assert!(production_module.contains("no per-push authenticity"));
        assert!(production_module.contains("generation ordering"));
    }

    // End-to-end over two in-process velo instances: the controller advances the phaser
    // and a cell subscribing over velo observes the full generation sequence — replay
    // for what preceded its subscribe, live for what follows.
    #[tokio::test]
    async fn cell_observes_replay_then_live_generations_over_velo() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let controller_velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");
        let phaser = Phaser::new();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let _server = PhaserServer::bind(
            controller_velo.clone(),
            phaser.clone(),
            std::sync::Arc::new(authority),
        )
        .expect("bind");

        // Advance twice BEFORE the cell subscribes — these must arrive as replay.
        phaser.advance(PhaseTransition::Started);
        phaser.advance(PhaseTransition::ShardsAvailable(4));

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let controller_peer = controller_velo.peer_info();
        let mut sub = PhaserClient::subscribe(cell_velo, &controller_peer, 0, &credentials[0])
            .await
            .expect("subscribe");

        sub.await_generation(2).await.expect("replayed to gen 2");
        assert!(sub.seen_generation() >= 2);

        phaser.advance(PhaseTransition::PhaseAdvance("profiling".into())); // gen 3
        phaser.advance(PhaseTransition::PhaseAdvance("drain".into())); // gen 4
        sub.await_generation(4).await.expect("live to gen 4");
        assert!(sub.seen_generation() >= 4);
    }

    #[tokio::test]
    async fn cell_awaits_named_phase_advances_from_replay_and_live_over_velo() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let controller_velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("controller velo");
        let phaser = Phaser::new();
        let (authority, credentials) = CellRegistrationAuthority::mint(1).expect("authority");
        let _server = PhaserServer::bind(
            controller_velo.clone(),
            phaser.clone(),
            std::sync::Arc::new(authority),
        )
        .expect("bind");

        phaser.advance(PhaseTransition::Started);
        phaser.advance(PhaseTransition::PhaseAdvance("warmup".into()));

        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let controller_peer = controller_velo.peer_info();
        let mut sub = PhaserClient::subscribe(cell_velo, &controller_peer, 0, &credentials[0])
            .await
            .expect("subscribe");

        assert_eq!(
            sub.await_phase_advance("warmup")
                .await
                .expect("replayed warmup"),
            2
        );

        phaser.advance(PhaseTransition::PhaseAdvance("profiling".into()));
        assert_eq!(
            sub.await_phase_advance("profiling")
                .await
                .expect("live profiling"),
            3
        );
    }
}
