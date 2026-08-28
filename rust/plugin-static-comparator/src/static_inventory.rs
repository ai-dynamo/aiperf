// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The census a statically-linked comparison build must carry.
//!
//! The comparator is the baseline half of the plugin parity experiment: the
//! same components, measured without the dynamic plugin boundary. That number
//! only means something if the two builds ship the *same* components, so this
//! module owns the census as data and refuses any drift from it.
//!
//! [`StaticComparatorRegistry`] is deliberately not an `AIPerfRegistry`: it
//! holds no factories, resolves nothing, and reaches no runtime. It is a local
//! set of `(id, version)` pairs whose only job is to be compared. The plugin
//! candidate crates are `cdylib`-only targets and cannot be linked as Rust
//! dependencies, so the static side declares the census it links rather than
//! importing it, and this module is what pins that declaration to the shipped
//! ids.

use std::collections::BTreeMap;

/// Version every first-party component carries.
///
/// All candidate plugin crates inherit `version.workspace`, and this crate is
/// in the same workspace, so its own package version is that version.
pub const DISTRIBUTION_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Components in the default distribution, in canonical id order.
///
/// `nvidia/transport-dynosim` is behind the `dynosim` Cargo feature, so it is
/// absent from a default build and never appears here.
pub const DEFAULT_DISTRIBUTION_CENSUS: &[(&str, &str)] = &[
    ("nvidia/endpoints", DISTRIBUTION_VERSION),
    ("nvidia/export-basic", DISTRIBUTION_VERSION),
    ("nvidia/export-mlflow", DISTRIBUTION_VERSION),
    ("nvidia/export-otel", DISTRIBUTION_VERSION),
    ("nvidia/export-parquet", DISTRIBUTION_VERSION),
    ("nvidia/export-wandb", DISTRIBUTION_VERSION),
    ("nvidia/transport-dry-run", DISTRIBUTION_VERSION),
    ("nvidia/transport-grpc", DISTRIBUTION_VERSION),
    ("nvidia/transport-http", DISTRIBUTION_VERSION),
    ("nvidia/transport-websocket", DISTRIBUTION_VERSION),
];

/// Components the full distribution adds on top of the default one.
pub const FULL_DISTRIBUTION_EXTRA_CENSUS: &[(&str, &str)] =
    &[("nvidia/transport-dynosim", DISTRIBUTION_VERSION)];

/// Why a census could not be built or did not match.
#[derive(Debug, thiserror::Error)]
pub enum CensusError {
    /// The same component id was registered twice.
    #[error("duplicate component id: {0}")]
    Duplicate(String),
    /// A component id or version is empty.
    #[error("component identity is incomplete: {0}")]
    Incomplete(String),
    /// The registered census is not the expected one.
    #[error("census mismatch: {detail}")]
    Mismatch {
        /// Every missing, unexpected, and version-drifted component.
        detail: String,
    },
}

/// A local, order-independent census of statically linked components.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StaticComparatorRegistry {
    /// Registered component id to version. `BTreeMap` so the census is
    /// reported in canonical id order regardless of registration order.
    components: BTreeMap<String, String>,
}

impl StaticComparatorRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one statically linked component.
    ///
    /// Registration is rejected rather than overwritten on a duplicate id: two
    /// components claiming one id means the census cannot be trusted to say
    /// which one was linked.
    pub fn register(&mut self, id: &str, version: &str) -> Result<(), CensusError> {
        if id.is_empty() || version.is_empty() {
            return Err(CensusError::Incomplete(format!(
                "id={id:?} version={version:?}"
            )));
        }
        if self.components.contains_key(id) {
            return Err(CensusError::Duplicate(id.to_string()));
        }
        self.components.insert(id.to_string(), version.to_string());
        Ok(())
    }

    /// The registered census in canonical id order.
    #[must_use]
    pub fn census(&self) -> Vec<(&str, &str)> {
        self.components
            .iter()
            .map(|(id, version)| (id.as_str(), version.as_str()))
            .collect()
    }

    /// Number of registered components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether no component is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Prove the registered census is exactly `expected`.
    ///
    /// The error names every missing, unexpected, and version-drifted
    /// component, so a failing comparator build says what changed rather than
    /// only that something did.
    pub fn assert_census(&self, expected: &[(&str, &str)]) -> Result<(), CensusError> {
        let expected_map: BTreeMap<&str, &str> = expected.iter().copied().collect();
        let mut differences = Vec::new();

        for (id, version) in &expected_map {
            match self.components.get(*id) {
                None => differences.push(format!("missing `{id}`")),
                Some(found) if found != version => {
                    differences.push(format!("`{id}` is {found}, expected {version}"));
                }
                Some(_) => {}
            }
        }
        for id in self.components.keys() {
            if !expected_map.contains_key(id.as_str()) {
                differences.push(format!("unexpected `{id}`"));
            }
        }

        if differences.is_empty() {
            Ok(())
        } else {
            Err(CensusError::Mismatch {
                detail: differences.join("; "),
            })
        }
    }
}

/// Build the registry for the default distribution.
pub fn default_distribution_registry() -> Result<StaticComparatorRegistry, CensusError> {
    registry_from(DEFAULT_DISTRIBUTION_CENSUS.iter().copied())
}

/// Build the registry for the full distribution.
pub fn full_distribution_registry() -> Result<StaticComparatorRegistry, CensusError> {
    registry_from(
        DEFAULT_DISTRIBUTION_CENSUS
            .iter()
            .chain(FULL_DISTRIBUTION_EXTRA_CENSUS.iter())
            .copied(),
    )
}

/// The full distribution census, in canonical id order.
#[must_use]
pub fn full_distribution_census() -> Vec<(&'static str, &'static str)> {
    let mut census: Vec<(&str, &str)> = DEFAULT_DISTRIBUTION_CENSUS
        .iter()
        .chain(FULL_DISTRIBUTION_EXTRA_CENSUS.iter())
        .copied()
        .collect();
    census.sort_unstable();
    census
}

/// Register every `(id, version)` pair into a fresh registry.
fn registry_from<'a>(
    components: impl Iterator<Item = (&'a str, &'a str)>,
) -> Result<StaticComparatorRegistry, CensusError> {
    let mut registry = StaticComparatorRegistry::new();
    for (id, version) in components {
        registry.register(id, version)?;
    }
    Ok(registry)
}
