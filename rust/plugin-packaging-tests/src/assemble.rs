// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assembly of a candidate generation into an authenticated inventory.
//!
//! A candidate generation is declared as a fixture: an ordered list of the
//! packages a distribution ships, each naming the artifact and manifest file a
//! build produced for it. Assembly reads those files, hashes their bytes, and
//! mints a [`PluginInventoryV1`] document that is published atomically.
//!
//! The digests are always the assembler's own hash of the bytes on disk. A
//! fixture never declares a digest, so a candidate cannot claim an identity its
//! artifacts do not have; the only thing the fixture controls is *which* files
//! are hashed, and those names are constrained to safe relative paths under the
//! caller-supplied artifacts directory.
//!
//! The `assemble-plugin-distribution` binary (source at
//! `rust/scripts/assemble-plugin-distribution.rs`) is a thin argument parser
//! over this module, so the release path and the test path are the same code.

use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use crate::inventory::{InventoryPackageV1, PluginInventoryV1};

/// File name the assembler publishes inside the output directory.
pub const INVENTORY_FILE_NAME: &str = "plugin-inventory.json";

/// Fixture schema version this assembler understands.
pub const CANDIDATE_FIXTURE_SCHEMA_VERSION: u32 = 1;

/// Suffix appended to a package's synthetic artifact bytes to derive the
/// synthetic manifest bytes, so the two never hash alike.
const SYNTHETIC_MANIFEST_SUFFIX: &str = ":manifest";

/// One package a candidate generation ships.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidatePackage {
    /// Canonical vendor-prefixed package id, e.g. `nvidia/export-basic`.
    pub id: String,
    /// Canonical SemVer version of the package.
    pub version: String,
    /// Artifact file name, relative to the artifacts directory.
    pub artifact: String,
    /// Manifest file name, relative to the artifacts directory.
    pub manifest: String,
    /// Bytes a test may materialize in place of a real build product.
    #[serde(default)]
    pub synthetic_bytes: Option<String>,
}

/// A declared candidate generation.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateFixture {
    /// Fixture schema version.
    pub schema_version: u32,
    /// Generation this candidate publishes.
    pub generation: u64,
    /// Build identity recorded on every published package.
    pub build_id: String,
    /// Packages the generation ships, in publication order.
    #[serde(rename = "package")]
    pub packages: Vec<CandidatePackage>,
}

/// Why a candidate generation could not be assembled.
#[derive(Debug, thiserror::Error)]
pub enum AssembleError {
    /// The fixture text could not be parsed.
    #[error("candidate fixture is malformed: {0}")]
    Fixture(String),
    /// The fixture declares a schema this assembler does not implement.
    #[error("unsupported candidate fixture schema version {0}")]
    UnsupportedSchemaVersion(u32),
    /// The fixture is structurally unusable (empty, or a duplicate id).
    #[error("candidate fixture is unusable: {0}")]
    Unusable(String),
    /// A declared file name is not a safe relative path.
    #[error("`{name}` for package `{package_id}` is not a plain relative file name")]
    UnsafeRelativePath {
        /// Package that declared the name.
        package_id: String,
        /// The refused name.
        name: String,
    },
    /// A declared artifact or manifest file is missing or unreadable.
    #[error("cannot read `{path}` for package `{package_id}`: {source}")]
    MissingArtifact {
        /// Package whose file could not be read.
        package_id: String,
        /// Path the assembler tried to read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The output directory could not be prepared or written.
    #[error("distribution output failed: {0}")]
    Io(#[from] std::io::Error),
    /// The minted inventory could not be published.
    #[error("inventory publication failed: {0}")]
    Inventory(#[from] aiperf_plugin_host::error::InventoryError),
}

impl CandidateFixture {
    /// Parse a candidate generation from its TOML text.
    pub fn parse(text: &str) -> Result<Self, AssembleError> {
        let fixture: Self = toml::from_str(text).map_err(|e| AssembleError::Fixture(e.to_string()))?;
        if fixture.schema_version != CANDIDATE_FIXTURE_SCHEMA_VERSION {
            return Err(AssembleError::UnsupportedSchemaVersion(
                fixture.schema_version,
            ));
        }
        if fixture.build_id.is_empty() {
            return Err(AssembleError::Unusable("build_id is empty".to_string()));
        }
        if fixture.packages.is_empty() {
            return Err(AssembleError::Unusable(
                "no packages are declared".to_string(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for package in &fixture.packages {
            if !seen.insert(package.id.as_str()) {
                return Err(AssembleError::Unusable(format!(
                    "package `{}` is declared twice",
                    package.id
                )));
            }
            for (name, value) in [("artifact", &package.artifact), ("manifest", &package.manifest)]
            {
                if !is_plain_relative_name(value) {
                    return Err(AssembleError::UnsafeRelativePath {
                        package_id: package.id.clone(),
                        name: format!("{name} = {value:?}"),
                    });
                }
            }
        }
        Ok(fixture)
    }

    /// Read and parse the candidate generation stored at `path`.
    pub fn load(path: &Path) -> Result<Self, AssembleError> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// Return this fixture with `generation` substituted.
    #[must_use]
    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    /// Write each package's declared synthetic bytes into `dir`.
    ///
    /// This exists so the assembly pipeline can be exercised without a build:
    /// a package that declares no synthetic bytes is skipped, leaving its real
    /// build product to be staged by whoever produced it.
    pub fn materialize_synthetic_artifacts(&self, dir: &Path) -> Result<(), AssembleError> {
        std::fs::create_dir_all(dir)?;
        for package in &self.packages {
            let Some(bytes) = package.synthetic_bytes.as_ref() else {
                continue;
            };
            std::fs::write(dir.join(&package.artifact), bytes.as_bytes())?;
            std::fs::write(
                dir.join(&package.manifest),
                format!("{bytes}{SYNTHETIC_MANIFEST_SUFFIX}").as_bytes(),
            )?;
        }
        Ok(())
    }
}

/// Return `true` when `value` is a single non-empty path component.
///
/// Anything rooted, empty, or carrying `.`/`..` is refused: the fixture selects
/// files inside the artifacts directory and must not be able to steer a read
/// outside it.
fn is_plain_relative_name(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let mut components = Path::new(value).components();
    let first = components.next();
    components.next().is_none() && matches!(first, Some(Component::Normal(_)))
}

/// Hash one file's bytes into a canonical `blake3:<hex>` digest.
fn digest_file(package_id: &str, path: &Path) -> Result<String, AssembleError> {
    let bytes = std::fs::read(path).map_err(|source| AssembleError::MissingArtifact {
        package_id: package_id.to_string(),
        path: path.to_path_buf(),
        source,
    })?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

/// Report the generation a fresh assembly into `output_dir` should publish.
///
/// A directory with no verifiable prior document has no history to advance, so
/// the caller's declared generation stands. Otherwise the published generation
/// is one past the prior one, which keeps the sequence monotonic even when a
/// fixture is reused verbatim.
pub fn next_generation(output_dir: &Path, declared: u64) -> u64 {
    let path = output_dir.join(INVENTORY_FILE_NAME);
    match PluginInventoryV1::load_and_verify(&path) {
        Ok(prior) => prior.generation.saturating_add(1),
        Err(_) => declared,
    }
}

/// Assemble `fixture` from the artifacts in `artifacts_dir` and publish the
/// inventory document into `output_dir`.
///
/// Returns the path of the published document.
pub fn assemble_distribution(
    fixture: &CandidateFixture,
    artifacts_dir: &Path,
    output_dir: &Path,
) -> Result<PathBuf, AssembleError> {
    let mut packages = Vec::with_capacity(fixture.packages.len());
    for declared in &fixture.packages {
        packages.push(InventoryPackageV1 {
            id: declared.id.clone(),
            version: declared.version.clone(),
            artifact_digest: digest_file(&declared.id, &artifacts_dir.join(&declared.artifact))?,
            manifest_digest: digest_file(&declared.id, &artifacts_dir.join(&declared.manifest))?,
            build_id: Some(fixture.build_id.clone()),
        });
    }

    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join(INVENTORY_FILE_NAME);
    PluginInventoryV1::new(fixture.generation, packages).publish(&path)?;
    Ok(path)
}
