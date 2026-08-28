// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated first-party distribution inventory.
//!
//! The inventory is the host's answer to "which packages is this build of
//! AIPerf entitled to install, and what must their bytes hash to?".  It names
//! content and never locations: an inventory carries canonical manifest and
//! artifact digests, a complete dependency closure, universe and build
//! identity, the required package and key sets, and the authentication root
//! that binds them.  It carries no absolute path, so a publisher cannot use it
//! to steer the host outside its own install root.
//!
//! Two shapes are exposed for the two jobs the inventory does:
//!
//! * [`AuthenticatedInventory`] is the trust decision — [`validate_inventory`]
//!   proves an inventory is internally complete and key-bound before anything
//!   is installed from it.
//! * [`PluginInventoryV1`] is the published document — a digest-authenticated
//!   file that is replaced atomically and read back with no-follow semantics.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::InventoryError;
use crate::platform::fs::{is_symlink, open_no_follow};

/// The inventory document schema version this host implements.
pub const INVENTORY_SCHEMA_VERSION: u32 = 1;

/// Length of a canonical BLAKE3 hex digest.
const DIGEST_HEX_LEN: usize = 64;

/// One package the distribution is entitled to install.
///
/// The entry is purely nominal: an id, a version, the digests its bytes must
/// hash to, the packages it needs, and the key that vouches for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionEntry {
    /// Canonical package id, e.g. `aiperf.exporter.parquet`.
    pub package_id: String,
    /// Canonical SemVer version of the package.
    pub version: String,
    /// BLAKE3 hex digest of the package's canonical manifest bytes.
    pub manifest_digest: String,
    /// BLAKE3 hex digest per target triple.
    pub artifact_digests: BTreeMap<String, String>,
    /// Package ids this entry requires; every one must also have an entry.
    pub depends_on: Vec<String>,
    /// Identity of the key that vouches for this entry.
    pub signing_key_id: String,
}

/// A complete, key-bound inventory of the first-party distribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedInventory {
    /// Identity of the package universe this inventory describes.
    pub universe_id: String,
    /// Identity of the build that produced this inventory.
    pub build_id: String,
    /// Digest of the trust root that authenticates the declared keys.
    pub authentication_root: String,
    /// Package ids that must be present for the distribution to be usable.
    pub required_packages: Vec<String>,
    /// Key identities permitted to vouch for entries in this inventory.
    pub required_keys: Vec<String>,
    /// Every package the distribution carries.
    pub entries: Vec<DistributionEntry>,
}

impl AuthenticatedInventory {
    /// Build a deterministic minimal inventory for tests and fixtures.
    ///
    /// The result validates, so callers can vary one field at a time to
    /// exercise a single refusal.
    pub fn synthetic(build_id: &str) -> Self {
        let key = "aiperf-synthetic-key";
        let digest = "a".repeat(DIGEST_HEX_LEN);
        let mut artifact_digests = BTreeMap::new();
        artifact_digests.insert("x86_64-unknown-linux-gnu".to_string(), digest.clone());
        Self {
            universe_id: "aiperf-synthetic".to_string(),
            build_id: build_id.to_string(),
            authentication_root: digest.clone(),
            required_packages: vec!["aiperf.synthetic".to_string()],
            required_keys: vec![key.to_string()],
            entries: vec![DistributionEntry {
                package_id: "aiperf.synthetic".to_string(),
                version: "1.0.0".to_string(),
                manifest_digest: digest,
                artifact_digests,
                depends_on: Vec::new(),
                signing_key_id: key.to_string(),
            }],
        }
    }

    /// Look up one entry by package id.
    pub fn entry(&self, package_id: &str) -> Option<&DistributionEntry> {
        self.entries.iter().find(|e| e.package_id == package_id)
    }

    /// Report whether any field carries an absolute or platform-rooted path.
    pub fn contains_absolute_path(&self) -> bool {
        self.absolute_path_field().is_some()
    }

    /// BLAKE3 hex digest over a canonical encoding of every field.
    ///
    /// The encoding is built field by field with explicit separators rather
    /// than by serializing, so the digest cannot shift when a serializer
    /// changes its formatting.
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        let mut push = |part: &str| {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part.as_bytes());
        };
        push("aiperf.inventory.v1");
        push(&self.universe_id);
        push(&self.build_id);
        push(&self.authentication_root);
        for pkg in &self.required_packages {
            push(pkg);
        }
        push("|keys|");
        for key in &self.required_keys {
            push(key);
        }
        push("|entries|");
        for entry in &self.entries {
            push(&entry.package_id);
            push(&entry.version);
            push(&entry.manifest_digest);
            for (target, digest) in &entry.artifact_digests {
                push(target);
                push(digest);
            }
            push("|deps|");
            for dep in &entry.depends_on {
                push(dep);
            }
            push("|key|");
            push(&entry.signing_key_id);
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Return the first field carrying an absolute path, if any.
    fn absolute_path_field(&self) -> Option<String> {
        let mut fields: Vec<&str> =
            vec![&self.universe_id, &self.build_id, &self.authentication_root];
        fields.extend(self.required_packages.iter().map(String::as_str));
        fields.extend(self.required_keys.iter().map(String::as_str));
        for entry in &self.entries {
            fields.push(&entry.package_id);
            fields.push(&entry.version);
            fields.push(&entry.signing_key_id);
            fields.extend(entry.depends_on.iter().map(String::as_str));
            fields.extend(entry.artifact_digests.keys().map(String::as_str));
        }
        fields
            .into_iter()
            .find(|f| is_absolute_like(f))
            .map(str::to_string)
    }
}

/// Return `true` if `value` looks like an absolute or platform-rooted path.
///
/// Windows drive prefixes and UNC paths are rejected on every platform: the
/// inventory is portable, so a value that is absolute anywhere is refused
/// everywhere.
fn is_absolute_like(value: &str) -> bool {
    if value.starts_with('/') || value.starts_with('\\') {
        return true;
    }
    let bytes = value.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// Return `true` if `value` is a canonical 64-character BLAKE3 hex digest.
fn is_canonical_digest(value: &str) -> bool {
    value.len() == DIGEST_HEX_LEN && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Prove that an inventory is internally complete and key-bound.
///
/// Every refusal here is a refusal to install: the host will not materialize
/// bytes it cannot name, cannot attribute to a declared key, or whose closure
/// it cannot close.
pub fn validate_inventory(inventory: &AuthenticatedInventory) -> Result<(), InventoryError> {
    for (name, value) in [
        ("universe_id", &inventory.universe_id),
        ("build_id", &inventory.build_id),
        ("authentication_root", &inventory.authentication_root),
    ] {
        if value.is_empty() {
            return Err(InventoryError::EmptyField(name.to_string()));
        }
    }
    if let Some(field) = inventory.absolute_path_field() {
        return Err(InventoryError::AbsolutePath(field));
    }
    if !is_canonical_digest(&inventory.authentication_root) {
        return Err(InventoryError::MalformedDigest {
            context: "authentication_root".to_string(),
            value: inventory.authentication_root.clone(),
        });
    }
    if inventory.entries.is_empty() {
        return Err(InventoryError::EmptyField("entries".to_string()));
    }
    if inventory.required_keys.is_empty() {
        return Err(InventoryError::EmptyField("required_keys".to_string()));
    }

    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for entry in &inventory.entries {
        if !seen.insert(entry.package_id.as_str()) {
            return Err(InventoryError::DuplicatePackage(entry.package_id.clone()));
        }
    }

    for entry in &inventory.entries {
        if !is_canonical_digest(&entry.manifest_digest) {
            return Err(InventoryError::MalformedDigest {
                context: entry.package_id.clone(),
                value: entry.manifest_digest.clone(),
            });
        }
        for digest in entry.artifact_digests.values() {
            if !is_canonical_digest(digest) {
                return Err(InventoryError::MalformedDigest {
                    context: entry.package_id.clone(),
                    value: digest.clone(),
                });
            }
        }
        if !inventory.required_keys.contains(&entry.signing_key_id) {
            return Err(InventoryError::UntrustedSigningKey {
                package_id: entry.package_id.clone(),
                key_id: entry.signing_key_id.clone(),
            });
        }
        for dep in &entry.depends_on {
            if !seen.contains(dep.as_str()) {
                return Err(InventoryError::IncompleteClosure {
                    package_id: entry.package_id.clone(),
                    missing: dep.clone(),
                });
            }
        }
    }

    for required in &inventory.required_packages {
        if !seen.contains(required.as_str()) {
            return Err(InventoryError::MissingRequiredPackage(required.clone()));
        }
    }
    Ok(())
}

/// One package record in a published inventory document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryPackageV1 {
    /// Canonical package id.
    pub id: String,
    /// Canonical SemVer version.
    pub version: String,
    /// Digest of the package artifact, e.g. `blake3:<hex>`.
    pub artifact_digest: String,
    /// Digest of the package manifest.
    pub manifest_digest: String,
    /// Build identity that produced the artifact, when recorded.
    #[serde(default)]
    pub build_id: Option<String>,
}

/// A published, digest-authenticated inventory document.
///
/// `inventory_digest` authenticates `generation` and `packages`.  A reader that
/// loads the document recomputes the digest before trusting any field, so a
/// truncated, forged, or edited document is refused rather than acted on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginInventoryV1 {
    /// Document schema version.
    pub schema_version: u32,
    /// Monotonic generation of this published inventory.
    pub generation: u64,
    /// Packages this generation publishes.
    pub packages: Vec<InventoryPackageV1>,
    /// BLAKE3 hex digest over `generation` and `packages`.
    pub inventory_digest: String,
}

impl PluginInventoryV1 {
    /// Mint a document for `generation` and compute its authenticating digest.
    pub fn new(generation: u64, packages: Vec<InventoryPackageV1>) -> Self {
        let inventory_digest = Self::payload_digest(generation, &packages);
        Self {
            schema_version: INVENTORY_SCHEMA_VERSION,
            generation,
            packages,
            inventory_digest,
        }
    }

    /// Verify that `inventory_digest` authenticates the current payload.
    pub fn verify_digest(&self) -> Result<(), InventoryError> {
        if self.schema_version != INVENTORY_SCHEMA_VERSION {
            return Err(InventoryError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if !is_canonical_digest(&self.inventory_digest) {
            return Err(InventoryError::MalformedDigest {
                context: "inventory_digest".to_string(),
                value: self.inventory_digest.clone(),
            });
        }
        let expected = Self::payload_digest(self.generation, &self.packages);
        // Both sides are canonical hex of the same width, so parsing to
        // `blake3::Hash` gives a fixed-time comparison.
        match (
            expected.parse::<blake3::Hash>(),
            self.inventory_digest.parse::<blake3::Hash>(),
        ) {
            (Ok(a), Ok(b)) if a == b => Ok(()),
            (Ok(_), Ok(_)) => Err(InventoryError::DigestMismatch),
            _ => Err(InventoryError::MalformedDigest {
                context: "inventory_digest".to_string(),
                value: self.inventory_digest.clone(),
            }),
        }
    }

    /// Publish this document at `path`, replacing any previous document.
    ///
    /// The bytes are written to a sibling temporary file, flushed, and renamed
    /// onto `path`.  Rename is atomic, so a concurrent reader observes either
    /// the previous complete document or this one, never a partial write.
    pub fn publish(&self, path: &Path) -> Result<(), InventoryError> {
        let bytes =
            serde_json::to_vec_pretty(self).map_err(|e| InventoryError::Parse(e.to_string()))?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let temp = parent.join(format!(
            ".{}.{}.tmp",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("inventory"),
            std::process::id()
        ));
        {
            let mut file = std::fs::File::create(&temp)?;
            file.write_all(&bytes)?;
            // Durability before the rename: a crash must not publish a name
            // that points at unwritten data.
            file.sync_all()?;
        }
        std::fs::rename(&temp, path).inspect_err(|_| {
            // The temporary file is this process's litter; drop it on failure
            // so a retry is not blocked by debris.
            let _ = std::fs::remove_file(&temp);
        })?;
        Ok(())
    }

    /// Read the document at `path` without following a symlink and verify it.
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
        let document: Self =
            serde_json::from_slice(&bytes).map_err(|e| InventoryError::Parse(e.to_string()))?;
        document.verify_digest()?;
        Ok(document)
    }

    /// BLAKE3 hex digest over the authenticated payload.
    fn payload_digest(generation: u64, packages: &[InventoryPackageV1]) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.plugin.inventory.v1");
        hasher.update(&generation.to_le_bytes());
        for pkg in packages {
            for part in [
                &pkg.id,
                &pkg.version,
                &pkg.artifact_digest,
                &pkg.manifest_digest,
            ] {
                hasher.update(&(part.len() as u64).to_le_bytes());
                hasher.update(part.as_bytes());
            }
            match &pkg.build_id {
                Some(build_id) => {
                    hasher.update(&[1u8]);
                    hasher.update(&(build_id.len() as u64).to_le_bytes());
                    hasher.update(build_id.as_bytes());
                }
                None => {
                    hasher.update(&[0u8]);
                }
            };
        }
        hasher.finalize().to_hex().to_string()
    }
}
