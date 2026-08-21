// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cell↔controller hub plugin: folds the register / heartbeat / partition /
//! store-partition velo handlers ([`VeloControllerTransport`]) onto the shared hub
//! velo instance, so a [`Hub`](super::Hub) instance BECOMES the connect anchor the
//! standalone controller is today (the `:9500` role).
//!
//! Unlike the discovery plugin, this plugin's velo surface is inherently
//! peer-registration + streaming coordination (it `register_peer`s each shipping
//! instance and returns per-cell launch envelopes), which has no faithful plain-HTTP
//! mirror. The plugin therefore exposes a small **diagnostic** HTTP surface
//! (`GET {prefix}/status`) reporting the anchor's fixed facts (cell count, prefix),
//! and keeps the full protocol on velo. The velo handlers themselves are the exact
//! [`VeloControllerTransport::bind_controller`] handlers, reused verbatim — only the
//! mount point moves onto the hub.
//!
//! Because [`VeloControllerTransport`] carries a live [`ControllerTransport::recv`]
//! stream the controller must own after registration, the plugin captures the bound
//! transport into a shared slot at [`register_velo_handlers`](HubPlugin::register_velo_handlers)
//! time; the bootstrap holds a clone of that slot (via [`transport_slot`](CellControllerHubPlugin::transport_slot))
//! and [`take`](std::sync::Mutex)s the transport back out once `Hub::register` returns.

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::routing::get;
use serde_json::json;
use velo::{EventHandle, Velo};

use super::plugin::{HubAbiRequirement, HubError, HubPlugin};
use crate::cellular::transport::velo_transport::{PlanRegistration, VeloControllerTransport};
use crate::engine::cellular_registration::CellRegistrationAuthority;

/// The default HTTP mount point / diagnostic identity for the cell↔controller plugin.
pub const CELL_CONTROLLER_PREFIX: &str = "/cell";

/// A shared, take-once slot for the [`VeloControllerTransport`] the plugin binds at
/// registration. The bootstrap holds a clone and takes the transport out after the
/// hub has installed the plugin's velo handlers.
pub type TransportSlot = Arc<Mutex<Option<VeloControllerTransport>>>;

/// Immutable facts the diagnostic HTTP surface reports.
#[derive(Clone, Debug)]
struct CellStatus {
    prefix: String,
    cell_count: u32,
}

/// The cell↔controller hub plugin. Registers the four control-plane velo handlers on
/// the shared hub velo instance and captures the resulting transport for the
/// bootstrap to own.
pub struct CellControllerHubPlugin {
    prefix: String,
    plan_registration: PlanRegistration,
    cell_count: u32,
    start_event: EventHandle,
    registration_authority: Arc<CellRegistrationAuthority>,
    transport: TransportSlot,
}

impl CellControllerHubPlugin {
    /// Build the plugin over the controller's per-cell registration planner, the run's
    /// `cell_count`, and the run-wide START event handle each cell awaits. Uses the
    /// default [`CELL_CONTROLLER_PREFIX`].
    pub(crate) fn new(
        plan_registration: PlanRegistration,
        cell_count: u32,
        start_event: EventHandle,
        registration_authority: Arc<CellRegistrationAuthority>,
    ) -> Self {
        Self {
            prefix: CELL_CONTROLLER_PREFIX.to_owned(),
            plan_registration,
            cell_count,
            start_event,
            registration_authority,
            transport: Arc::new(Mutex::new(None)),
        }
    }

    /// A clone of the take-once slot the bound [`VeloControllerTransport`] lands in.
    /// The bootstrap clones this BEFORE boxing the plugin into `Hub::register`, then
    /// [`Option::take`]s the transport out once registration has installed the velo
    /// handlers.
    pub fn transport_slot(&self) -> TransportSlot {
        self.transport.clone()
    }
}

/// `GET {prefix}/status` — the diagnostic HTTP surface: the anchor's fixed facts.
async fn http_status(State(status): State<Arc<CellStatus>>) -> Json<serde_json::Value> {
    Json(json!({
        "plugin": status.prefix,
        "cell_count": status.cell_count,
        "role": "cell-controller-anchor",
    }))
}

impl HubPlugin for CellControllerHubPlugin {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn required_abi(&self) -> HubAbiRequirement {
        HubAbiRequirement::current()
    }

    fn router(&self) -> axum::Router {
        let status = Arc::new(CellStatus {
            prefix: self.prefix.clone(),
            cell_count: self.cell_count,
        });
        axum::Router::new()
            .route("/status", get(http_status))
            .with_state(status)
    }

    fn register_velo_handlers(&self, velo: &Arc<Velo>) -> Result<(), HubError> {
        // Bind the exact control-plane handlers onto the shared hub velo instance and
        // capture the transport so the bootstrap can own its `recv` stream.
        let transport = VeloControllerTransport::bind_controller(
            velo.clone(),
            self.registration_authority.clone(),
            self.plan_registration.clone(),
            self.cell_count,
            self.start_event,
        )
        .map_err(|error| HubError::VeloHandler {
            prefix: self.prefix.clone(),
            message: error.to_string(),
        })?;
        *self
            .transport
            .lock()
            .expect("cell-controller transport slot poisoned") = Some(transport);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::transport::CellClient;
    use crate::cellular::transport::connect::{BindSpec, build_velo};
    use crate::cellular::transport::velo_transport::CellRegistrationPlan;
    use crate::cellular::transport::velo_transport::VeloCellClient;
    use crate::cellular::{CellMessage, ControllerTransport};
    use crate::engine::cellular_registration::CellRegistrationAuthority;
    use crate::hub::Hub;

    fn plan_registration() -> PlanRegistration {
        Arc::new(|verified| {
            let register: crate::cellular::transport::CellRegister = verified.decode_payload()?;
            Ok(Some(CellRegistrationPlan {
                envelope: vec![register.cell_id as u8, 0xCC],
                artifact: None,
            }))
        })
    }

    /// The plugin mounts on a hub, registers the control-plane velo handlers, and the
    /// captured transport serves a real cell register + partition round-trip — proving
    /// the hub is the connect anchor the standalone controller is today.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plugin_mounts_and_serves_register_over_the_hub() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let velo = build_velo(BindSpec::TcpListener(listener))
            .await
            .expect("hub velo");
        let controller_peer = velo.messenger().peer_info();
        let start = velo.event_manager().new_event().expect("start event");
        let start_handle = start.handle();
        let (authority, credentials) =
            CellRegistrationAuthority::mint(1).expect("registration authority");

        let plugin =
            CellControllerHubPlugin::new(plan_registration(), 1, start_handle, Arc::new(authority));
        let slot = plugin.transport_slot();

        let mut hub = Hub::new(velo);
        hub.register(Box::new(plugin))
            .expect("register cell plugin");
        // The plugin's HTTP prefix is mounted.
        assert_eq!(hub.prefixes().collect::<Vec<_>>(), vec!["/cell"]);
        // The transport was captured at registration.
        let mut transport = slot
            .lock()
            .unwrap()
            .take()
            .expect("transport captured by registration");

        // A real cell connects to the hub's velo by PeerInfo, registers, and ships a
        // heartbeat that the hub-mounted handler surfaces on the captured transport.
        let cell_velo = build_velo(BindSpec::TcpLoopback).await.expect("cell velo");
        let mut cell = VeloCellClient::connect_authenticated(
            cell_velo,
            controller_peer,
            Arc::new(credentials[0].clone()),
        )
        .expect("connect");
        let reply = cell
            .register_with_credential(0, None, &credentials[0])
            .await
            .expect("register");
        assert_eq!(reply.envelope, vec![0_u8, 0xCC]);

        use crate::cellular::heartbeat::HeartbeatAccumulator;
        let mut acc = HeartbeatAccumulator::new();
        acc.observe(Some(20.0), Some(5.0), Some(50.0));
        let hb = acc.snapshot(1, Default::default(), Default::default());
        cell.send(&CellMessage::Heartbeat {
            cell_id: 0,
            heartbeat: Box::new(hb),
        })
        .await
        .expect("ship heartbeat");

        match transport.recv().await.expect("recv").expect("some") {
            CellMessage::Heartbeat { cell_id, .. } => assert_eq!(cell_id, 0),
            other => panic!("expected heartbeat, got {other:?}"),
        }
        let _ = addr;
    }
}
