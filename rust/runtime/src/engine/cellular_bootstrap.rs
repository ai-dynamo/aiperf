// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trusted cellular control-plane bootstrap material.

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use parking_lot::RwLock;
use velo::PeerInfo;

use crate::engine::cellular_registration::{CellRegistrationAuthority, CellRegistrationCredential};

/// Process-private environment variable carrying one opaque cell bootstrap bundle.
pub(crate) const CELL_BOOTSTRAP_ENV: &str = "AIPERF_CELL_BOOTSTRAP";

/// A controller-owned source for authenticated controller peer publication and
/// per-cell private bootstrap material.
pub(crate) trait CellBootstrapProvider: Send + Sync {
    /// The run's verifier used before admitting every cellular peer.
    fn authority(&self) -> Arc<CellRegistrationAuthority>;
    /// Publish the controller identity only after its Velo instance is bound.
    fn publish_controller_peer(&self, peer: &PeerInfo) -> Result<()>;
    /// Return exactly one cell's process-private bootstrap bundle.
    fn bundle_for_cell(&self, cell_id: u32) -> Result<CellBootstrapBundle>;
}

/// Opaque per-cell controller bootstrap material. It intentionally has no
/// `Debug`, serde, or display implementation because it contains a private key.
pub(crate) struct CellBootstrapBundle {
    controller_peer: Vec<u8>,
    credential: CellRegistrationCredential,
}

impl CellBootstrapBundle {
    /// Encode for one local child process only.
    pub(crate) fn encode_launch_value(&self) -> String {
        let credential = self.credential.encode_launch_value();
        let mut bytes = Vec::with_capacity(8 + self.controller_peer.len() + credential.len());
        bytes.extend_from_slice(&(self.controller_peer.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.controller_peer);
        bytes.extend_from_slice(&(credential.len() as u32).to_le_bytes());
        bytes.extend_from_slice(credential.as_bytes());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    }

    fn decode_launch_value(value: &str) -> Result<Self> {
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(value)
            .map_err(|_| anyhow::anyhow!("cell bootstrap bundle is malformed"))?;
        ensure!(bytes.len() >= 8, "cell bootstrap bundle is malformed");
        let peer_len = u32::from_le_bytes(
            bytes[..4]
                .try_into()
                .map_err(|_| anyhow::anyhow!("cell bootstrap bundle is malformed"))?,
        ) as usize;
        ensure!(
            bytes.len() >= 8 + peer_len,
            "cell bootstrap bundle is malformed"
        );
        let credential_len_start = 4 + peer_len;
        let credential_len = u32::from_le_bytes(
            bytes[credential_len_start..credential_len_start + 4]
                .try_into()
                .map_err(|_| anyhow::anyhow!("cell bootstrap bundle is malformed"))?,
        ) as usize;
        ensure!(
            bytes.len() == credential_len_start + 4 + credential_len,
            "cell bootstrap bundle is malformed"
        );
        let credential = std::str::from_utf8(&bytes[credential_len_start + 4..])
            .map_err(|_| anyhow::anyhow!("cell bootstrap bundle is malformed"))?;
        Ok(Self {
            controller_peer: bytes[4..credential_len_start].to_vec(),
            credential: CellRegistrationCredential::from_launch_value(credential)?,
        })
    }

    fn controller_peer(&self) -> Result<PeerInfo> {
        rmp_serde::from_slice(&self.controller_peer)
            .map_err(|_| anyhow::anyhow!("cell bootstrap controller peer is malformed"))
    }
}

struct LocalCellBootstrapProvider {
    authority: Arc<CellRegistrationAuthority>,
    credentials: Vec<CellRegistrationCredential>,
    controller_peer: RwLock<Option<Vec<u8>>>,
}

impl LocalCellBootstrapProvider {
    fn prepare(cell_count: u32) -> Result<Self> {
        let (authority, credentials) = CellRegistrationAuthority::mint(cell_count)?;
        Ok(Self {
            authority: Arc::new(authority),
            credentials,
            controller_peer: RwLock::new(None),
        })
    }
}

impl CellBootstrapProvider for LocalCellBootstrapProvider {
    fn authority(&self) -> Arc<CellRegistrationAuthority> {
        self.authority.clone()
    }

    fn publish_controller_peer(&self, peer: &PeerInfo) -> Result<()> {
        let encoded = rmp_serde::to_vec(peer).context("encoding controller bootstrap peer")?;
        let mut published = self.controller_peer.write();
        ensure!(
            published.is_none(),
            "controller bootstrap peer is already published"
        );
        *published = Some(encoded);
        Ok(())
    }

    fn bundle_for_cell(&self, cell_id: u32) -> Result<CellBootstrapBundle> {
        let credential = self
            .credentials
            .get(cell_id as usize)
            .ok_or_else(|| anyhow::anyhow!("cell bootstrap id is out of range"))?;
        ensure!(
            credential.cell_id() == cell_id,
            "cell bootstrap identity mismatch"
        );
        let controller_peer = self
            .controller_peer
            .read()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("controller bootstrap peer is not published"))?;
        Ok(CellBootstrapBundle {
            controller_peer,
            credential: CellRegistrationCredential::from_launch_value(
                &credential.encode_launch_value(),
            )?,
        })
    }
}

/// Prepare bootstrap material before any controller control listener is bound.
pub(crate) fn prepare_cell_bootstrap(
    is_cross_host: bool,
    cell_count: u32,
) -> Result<Arc<dyn CellBootstrapProvider>> {
    if is_cross_host {
        bail!(
            "cross-host cellular execution requires a deployment-provisioned authenticated bootstrap provider"
        );
    }
    Ok(Arc::new(LocalCellBootstrapProvider::prepare(cell_count)?))
}

/// Load one cell's trusted controller identity and private signing credential.
pub(crate) fn load_authenticated_controller_peer(
    cell_id: u32,
) -> Result<(PeerInfo, CellRegistrationCredential)> {
    let value = std::env::var(CELL_BOOTSTRAP_ENV)
        .context("cell has no controller-provisioned authenticated bootstrap")?;
    let bundle = CellBootstrapBundle::decode_launch_value(&value)?;
    ensure!(
        bundle.credential.cell_id() == cell_id,
        "cell bootstrap credential belongs to another cell"
    );
    Ok((bundle.controller_peer()?, bundle.credential))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_bootstrap_gives_each_cell_a_distinct_private_credential() {
        let provider = LocalCellBootstrapProvider::prepare(2).expect("prepare");
        let velo = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let peer = velo.block_on(async {
            crate::cellular::transport::connect::build_velo(
                crate::cellular::transport::connect::BindSpec::TcpLoopback,
            )
            .await
            .expect("velo")
            .peer_info()
        });
        provider.publish_controller_peer(&peer).expect("publish");
        let first = provider
            .bundle_for_cell(0)
            .expect("first")
            .encode_launch_value();
        let second = provider
            .bundle_for_cell(1)
            .expect("second")
            .encode_launch_value();
        assert_ne!(first, second);
        let first = CellBootstrapBundle::decode_launch_value(&first).expect("decode first");
        assert_eq!(first.credential.cell_id(), 0);
        assert!(CellBootstrapBundle::decode_launch_value("not-a-bundle").is_err());
    }

    #[test]
    fn cross_host_bootstrap_fails_before_controller_binding() {
        assert!(prepare_cell_bootstrap(true, 1).is_err());
    }
}
