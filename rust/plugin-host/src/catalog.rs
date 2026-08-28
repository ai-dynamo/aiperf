// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog resolution: deduplication, shadowing, and quarantine (Task 13).
//!
//! Consumes a flat list of `DiscoveredPackage` records (from `discovery`) plus
//! their manifest bytes and produces an `IntendedCatalog` that separates
//! winners (one per package ID), shadowed losers, ambiguous ties, and
//! quarantined packages that failed parsing.

use std::{collections::HashMap, path::PathBuf};

use crate::{
    discovery::DiscoveredPackage, manifest::PluginManifestV2, normalize::normalize_manifest,
    priority::effective_priority,
};

/// A package that won its priority contest and will be loaded.
#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    pub package_id: String,
    pub version: String,
    pub manifest_path: PathBuf,
    pub effective_priority: i32,
    pub manifest: PluginManifestV2,
}

/// A package that was outranked by a higher-priority package with the same ID.
#[derive(Debug, Clone)]
pub struct ShadowedPackage {
    pub package_id: String,
    pub manifest_path: PathBuf,
    pub effective_priority: i32,
    pub shadowed_by: PathBuf,
}

/// Two packages with identical priority that cannot be deterministically ordered.
#[derive(Debug, Clone)]
pub struct AmbiguousPackage {
    pub package_id: String,
    pub paths: Vec<PathBuf>,
    pub effective_priority: i32,
}

/// A package whose manifest failed to parse or normalize.
#[derive(Debug, Clone)]
pub struct QuarantinedPackage {
    pub manifest_path: PathBuf,
    pub reason: String,
}

/// The result of resolving all discovered packages into a definite load list.
#[derive(Debug, Default)]
pub struct IntendedCatalog {
    pub winners: Vec<ResolvedPackage>,
    pub shadows: Vec<ShadowedPackage>,
    pub ambiguous: Vec<AmbiguousPackage>,
    pub quarantined: Vec<QuarantinedPackage>,
}

struct Candidate {
    manifest_path: PathBuf,
    source_id_kind_ordinal: u8,
    manifest: PluginManifestV2,
}

/// Resolve `discovered` into an `IntendedCatalog`.
///
/// For each distinct package ID:
/// - The highest effective priority wins.
/// - Two packages with the same effective priority are `AmbiguousPackage`.
/// - Packages whose manifests fail parsing/normalization are quarantined.
pub fn resolve_catalog(
    discovered: Vec<DiscoveredPackage>,
    manifest_bytes: HashMap<PathBuf, Vec<u8>>,
) -> IntendedCatalog {
    let mut catalog = IntendedCatalog::default();
    let mut candidates: Vec<Candidate> = vec![];

    for pkg in discovered {
        let bytes = match manifest_bytes.get(&pkg.manifest_path) {
            Some(b) => b,
            None => {
                catalog.quarantined.push(QuarantinedPackage {
                    manifest_path: pkg.manifest_path.clone(),
                    reason: "manifest bytes not provided".to_owned(),
                });
                continue;
            }
        };

        let raw: PluginManifestV2 = match serde_yaml::from_slice(bytes) {
            Ok(r) => r,
            Err(e) => {
                catalog.quarantined.push(QuarantinedPackage {
                    manifest_path: pkg.manifest_path.clone(),
                    reason: format!("parse error: {e}"),
                });
                continue;
            }
        };

        let normalized = match normalize_manifest(raw) {
            Ok(m) => m,
            Err(e) => {
                catalog.quarantined.push(QuarantinedPackage {
                    manifest_path: pkg.manifest_path.clone(),
                    reason: format!("normalize error: {e}"),
                });
                continue;
            }
        };

        candidates.push(Candidate {
            manifest_path: pkg.manifest_path,
            source_id_kind_ordinal: pkg.source_id.kind_ordinal,
            manifest: normalized,
        });
    }

    // Group by package ID; each manifest may declare multiple packages.
    // (package_id) → Vec<(eff_priority, manifest_path, manifest)>
    let mut by_id: HashMap<String, Vec<(i32, PathBuf, PluginManifestV2)>> = HashMap::new();

    for cand in candidates {
        for pkg_entry in &cand.manifest.packages {
            let eff = effective_priority(cand.source_id_kind_ordinal, pkg_entry.priority);
            by_id.entry(pkg_entry.id.clone()).or_default().push((
                eff,
                cand.manifest_path.clone(),
                cand.manifest.clone(),
            ));
        }
    }

    for (pkg_id, mut entries) in by_id {
        // Sort descending by priority.
        entries.sort_by_key(|a| std::cmp::Reverse(a.0));
        let top_priority = entries[0].0;

        let (top_entries, losers): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .partition(|(eff, _, _)| *eff == top_priority);

        // Push shadows for losers.
        let winner_path = top_entries[0].1.clone();
        for (eff, path, _) in losers {
            catalog.shadows.push(ShadowedPackage {
                package_id: pkg_id.clone(),
                manifest_path: path,
                effective_priority: eff,
                shadowed_by: winner_path.clone(),
            });
        }

        if top_entries.len() > 1 {
            catalog.ambiguous.push(AmbiguousPackage {
                package_id: pkg_id,
                paths: top_entries.into_iter().map(|(_, p, _)| p).collect(),
                effective_priority: top_priority,
            });
        } else {
            // Safe: only reachable when top_entries.len() == 1 (len > 1 case handled above).
            let (_, path, manifest) = top_entries.into_iter().next().unwrap();
            let version = manifest
                .packages
                .iter()
                .find(|p| p.id == pkg_id)
                .map(|p| p.version.clone())
                .unwrap_or_default();
            catalog.winners.push(ResolvedPackage {
                package_id: pkg_id,
                version,
                manifest_path: path,
                effective_priority: top_priority,
                manifest,
            });
        }
    }

    catalog
}

/// Convenience: resolve catalog by reading manifest files from disk.
///
/// Manifests are opened with `open_no_follow`, so a manifest path that is a
/// symlink at its final component is refused at the read itself rather than
/// relying on the earlier `symlink_metadata` check in discovery (which a TOCTOU
/// swap could invalidate). A manifest that cannot be read is quarantined; it
/// never aborts resolution of the remaining catalog.
pub fn resolve_catalog_from_disk(
    discovered: Vec<DiscoveredPackage>,
) -> Result<IntendedCatalog, std::io::Error> {
    let mut bytes_map: HashMap<PathBuf, Vec<u8>> = HashMap::new();
    let mut unreadable: HashMap<PathBuf, String> = HashMap::new();

    for pkg in &discovered {
        if bytes_map.contains_key(&pkg.manifest_path) || unreadable.contains_key(&pkg.manifest_path)
        {
            continue;
        }
        match read_manifest_no_follow(&pkg.manifest_path) {
            Ok(bytes) => {
                bytes_map.insert(pkg.manifest_path.clone(), bytes);
            }
            Err(e) => {
                unreadable.insert(pkg.manifest_path.clone(), format!("read error: {e}"));
            }
        }
    }

    let (readable, refused): (Vec<_>, Vec<_>) = discovered
        .into_iter()
        .partition(|pkg| !unreadable.contains_key(&pkg.manifest_path));

    let mut catalog = resolve_catalog(readable, bytes_map);
    for pkg in refused {
        let reason = unreadable
            .get(&pkg.manifest_path)
            .cloned()
            .unwrap_or_else(|| "manifest unreadable".to_owned());
        catalog.quarantined.push(QuarantinedPackage {
            manifest_path: pkg.manifest_path,
            reason,
        });
    }
    Ok(catalog)
}

/// Read a manifest's bytes without following a symlink at the final component.
fn read_manifest_no_follow(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;

    let mut file = crate::platform::fs::open_no_follow(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}
