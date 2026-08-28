// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated plugin inventory.
//!
//! [`PluginInventoryV1`] is the document a distribution publishes to describe
//! which packages a generation contains.  Its `inventory_digest` is a BLAKE3
//! hash over the canonical JSON serialization of the payload — schema version,
//! generation, and package list — so any post-publication mutation is detected
//! by [`PluginInventoryV1::verify_digest`] before the host acts on it.
//!
//! Publication is atomic: bytes land in a private temporary file that is
//! `rename`d over the final path, so a concurrent reader observes either the
//! previous complete document or the new one and never a partial write.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::InventoryError;
use crate::platform::fs::{is_symlink, open_no_follow};

/// The only inventory schema version this host implements.
pub const INVENTORY_SCHEMA_VERSION: u32 = 1;

/// Disambiguates temporary files produced by concurrent publishers in one
/// process; the pid disambiguates across processes.
static PUBLISH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// One package recorded in a [`PluginInventoryV1`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryPackageV1 {
    /// Normalized package identifier.
    pub id: String,
    /// Semantic version string (`major.minor.patch`).
    pub version: String,
    /// `"blake3:<hex>"` digest of the platform artifact.
    pub artifact_digest: String,
    /// `"blake3:<hex>"` digest of the package manifest.
    pub manifest_digest: String,
    /// Opaque build identifier assigned by the producing distribution.
    pub build_id: Option<String>,
}

/// The authenticated payload the inventory digest is computed over.
///
/// Serializing a dedicated struct rather than the full document keeps the
/// digest independent of the digest field itself, which must not be an input to
/// its own computation.
#[derive(Serialize)]
struct InventoryPayload<'a> {
    schema_version: u32,
    generation: u64,
    packages: &'a [InventoryPackageV1],
}

/// A published plugin inventory (`schema_version: 1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginInventoryV1 {
    /// Always [`INVENTORY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Monotonically increasing generation number.
    pub generation: u64,
    /// Packages this inventory publishes.
    pub packages: Vec<InventoryPackageV1>,
    /// BLAKE3 hex digest of the canonical JSON payload.
    pub inventory_digest: String,
}

impl PluginInventoryV1 {
    /// Build an inventory and immediately compute its digest.
    pub fn new(generation: u64, packages: Vec<InventoryPackageV1>) -> Self {
        let inventory_digest = compute_digest(INVENTORY_SCHEMA_VERSION, generation, &packages);
        Self {
            schema_version: INVENTORY_SCHEMA_VERSION,
            generation,
            packages,
            inventory_digest,
        }
    }

    /// Verify that the stored digest authenticates the stored payload.
    ///
    /// The digests are compared as parsed [`blake3::Hash`] values, whose
    /// `PartialEq` is constant-time.  Comparing hex strings would short-circuit
    /// on the first differing character and leak the length of a matching
    /// prefix, which is a forgery oracle for an attacker who can rewrite the
    /// published file and observe rejection latency.
    pub fn verify_digest(&self) -> Result<(), InventoryError> {
        if self.schema_version != INVENTORY_SCHEMA_VERSION {
            return Err(InventoryError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        let expected = compute_digest(self.schema_version, self.generation, &self.packages);
        let stored = blake3::Hash::from_hex(&self.inventory_digest)
            .map_err(|_| InventoryError::MalformedDigest)?;
        let computed =
            blake3::Hash::from_hex(&expected).map_err(|_| InventoryError::MalformedDigest)?;
        if stored == computed {
            Ok(())
        } else {
            Err(InventoryError::DigestMismatch)
        }
    }

    /// Serialize the inventory to canonical JSON bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_else(|_| {
            // Every field is a plain owned scalar or Vec of scalars, so
            // serialization has no failure mode; an empty document still fails
            // digest verification on the reader side.
            Vec::new()
        })
    }

    /// Publish the inventory atomically at `path`.
    ///
    /// The bytes are written to `<path>.tmp.<pid>.<seq>`, flushed to stable
    /// storage, and then renamed over `path`.  The final path is never opened
    /// for writing, so a reader either sees the previous document or the new
    /// one.
    pub fn publish(&self, path: &Path) -> std::io::Result<()> {
        let seq = PUBLISH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut name = path.as_os_str().to_os_string();
        name.push(format!(".tmp.{}.{seq}", std::process::id()));
        let tmp = std::path::PathBuf::from(name);

        let bytes = self.to_bytes();
        {
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        match std::fs::rename(&tmp, path) {
            Ok(()) => Ok(()),
            Err(e) => {
                // Never leave debris beside the published path.
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    /// Read the inventory at `path` and verify its digest.
    ///
    /// The path is opened without following a symlink at its final component,
    /// so an inventory reached through a link an unprivileged user controls is
    /// refused before any bytes are parsed.
    pub fn load_and_verify(path: &Path) -> Result<Self, InventoryError> {
        if is_symlink(path)? {
            return Err(InventoryError::Symlink(path.to_path_buf()));
        }
        let mut file = open_no_follow(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidInput {
                InventoryError::Symlink(path.to_path_buf())
            } else {
                InventoryError::Io(e)
            }
        })?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let inventory: Self =
            serde_json::from_slice(&bytes).map_err(|e| InventoryError::Parse(e.to_string()))?;
        inventory.verify_digest()?;
        Ok(inventory)
    }
}

/// An inventory whose digest has been verified.
///
/// Holding this type is the proof that the wrapped document authenticated; the
/// install path takes it by reference so a generation can never be materialized
/// from an unverified inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedInventory {
    inventory: PluginInventoryV1,
}

impl AuthenticatedInventory {
    /// Verify `inventory` and wrap it.
    pub fn verify(inventory: PluginInventoryV1) -> Result<Self, InventoryError> {
        inventory.verify_digest()?;
        Ok(Self { inventory })
    }

    /// Read and verify the inventory published at `path`.
    pub fn load(path: &Path) -> Result<Self, InventoryError> {
        Ok(Self {
            inventory: PluginInventoryV1::load_and_verify(path)?,
        })
    }

    /// The verified document.
    pub fn inventory(&self) -> &PluginInventoryV1 {
        &self.inventory
    }

    /// The verified BLAKE3 hex digest.
    pub fn digest(&self) -> &str {
        &self.inventory.inventory_digest
    }

    /// Build a deterministic single-package inventory identified by `build_id`.
    ///
    /// Two distinct build identifiers always produce distinct digests, which
    /// makes this the canonical stand-in for a real distribution inventory in
    /// packaging tests and local development installs.
    pub fn synthetic(build_id: &str) -> Self {
        let artifact = blake3::hash(format!("artifact:{build_id}").as_bytes());
        let manifest = blake3::hash(format!("manifest:{build_id}").as_bytes());
        let inventory = PluginInventoryV1::new(
            1,
            vec![InventoryPackageV1 {
                id: "aiperf.synthetic".to_string(),
                version: "0.0.0".to_string(),
                artifact_digest: format!("blake3:{}", artifact.to_hex()),
                manifest_digest: format!("blake3:{}", manifest.to_hex()),
                build_id: Some(build_id.to_string()),
            }],
        );
        Self { inventory }
    }
}

/// Compute the BLAKE3 hex digest over the canonical JSON payload.
fn compute_digest(schema_version: u32, generation: u64, packages: &[InventoryPackageV1]) -> String {
    let payload = InventoryPayload {
        schema_version,
        generation,
        packages,
    };
    // Serialization of owned scalars is infallible; an empty buffer would still
    // produce a digest no valid payload can match.
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}
