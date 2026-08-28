// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin discovery: locate manifest files from configured sources (Task 13).
//!
//! Scans directories and environment-variable lists for `plugin.manifest.yaml`
//! files, assigns each a `DiscoverySourceId` that encodes its source kind and
//! authored order, and returns an unsorted list of `DiscoveredPackage` records.
//! Catalog resolution (`catalog::resolve_catalog`) consumes this list.

use std::path::{Path, PathBuf};

use crate::error::DiscoveryError;
use crate::priority::source_kind_ordinal;

/// The name every plugin manifest file must use.
pub const MANIFEST_FILENAME: &str = "plugin.manifest.yaml";

/// Where a discovered plugin came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoverySource {
    /// The standard distribution plugin directory (lowest priority auto source).
    Distribution,
    /// System-wide plugin directory (e.g., `/usr/lib/aiperf/plugins`).
    PlatformSystem,
    /// Per-user plugin directory (e.g., `~/.aiperf/plugins`).
    PlatformUser,
    /// A colon-separated list of directories from an environment variable.
    Environment(String),
    /// A single explicit plugin directory path.
    ExplicitDirectory(PathBuf),
    /// A direct path to a plugin manifest file.
    ExplicitManifest(PathBuf),
    /// A hermetic plugin bundle directory.
    HermeticBundle(PathBuf),
}

impl DiscoverySource {
    /// Return the stable kind ordinal used in priority calculation.
    pub fn kind_ordinal(&self) -> u8 {
        source_kind_ordinal(self)
    }
}

/// Stable identity of a discovery source used in priority ordering.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DiscoverySourceId {
    /// Source kind ordinal; higher = higher priority tier.
    pub kind_ordinal: u8,
    /// Index of this source in the authored `sources` slice.
    pub authored_index: u32,
}

/// A plugin manifest discovered from one source.
#[derive(Debug, Clone)]
pub struct DiscoveredPackage {
    /// Absolute path to the `plugin.manifest.yaml` file.
    pub manifest_path: PathBuf,
    /// Which source this came from.
    pub source_id: DiscoverySourceId,
    /// Effective priority for catalog resolution.
    pub priority: i32,
}

/// Discover plugin manifests from all `sources`.
///
/// When `no_auto_plugins` is `true`, `Distribution`, `PlatformSystem`, and
/// `PlatformUser` sources are silently skipped; only `Environment`,
/// `ExplicitDirectory`, `ExplicitManifest`, and `HermeticBundle` are scanned.
pub fn discover_plugins(
    sources: &[DiscoverySource],
    no_auto_plugins: bool,
) -> Result<Vec<DiscoveredPackage>, DiscoveryError> {
    let mut results: Vec<DiscoveredPackage> = vec![];

    for (idx, source) in sources.iter().enumerate() {
        let authored_index = idx as u32;
        let kind_ordinal = source.kind_ordinal();

        // Skip auto sources when suppressed.
        if no_auto_plugins {
            match source {
                DiscoverySource::Distribution
                | DiscoverySource::PlatformSystem
                | DiscoverySource::PlatformUser => continue,
                _ => {}
            }
        }

        let source_id = DiscoverySourceId {
            kind_ordinal,
            authored_index,
        };

        match source {
            DiscoverySource::ExplicitManifest(path) => {
                // `symlink_metadata` does not follow the final component, so a
                // symlinked manifest is rejected rather than silently resolved.
                match path.symlink_metadata() {
                    Ok(m) if m.file_type().is_file() => {
                        results.push(DiscoveredPackage {
                            // Paths from `read_dir` are already absolute; `open_no_follow`
                            // downstream enforces no-follow.  `canonicalize` would re-open
                            // the path following any symlink, defeating the symlink check
                            // above.
                            manifest_path: path.clone(),
                            source_id,
                            priority: priority_for_source(source),
                        });
                    }
                    Ok(m) if m.file_type().is_symlink() => {
                        tracing::debug!(
                            path = %path.display(),
                            "skipping symlinked explicit plugin manifest"
                        );
                    }
                    _ => {}
                }
            }
            DiscoverySource::ExplicitDirectory(dir) | DiscoverySource::HermeticBundle(dir) => {
                scan_dir(dir, &source_id, priority_for_source(source), &mut results)?;
            }
            DiscoverySource::Environment(var_name) => {
                let val = std::env::var(var_name).unwrap_or_default();
                for dir_str in val.split(':').filter(|s| !s.is_empty()) {
                    let dir = PathBuf::from(dir_str);
                    scan_dir(&dir, &source_id, priority_for_source(source), &mut results)
                        .unwrap_or_default();
                }
            }
            DiscoverySource::Distribution => {
                if let Some(dir) = distribution_plugin_dir() {
                    scan_dir(&dir, &source_id, priority_for_source(source), &mut results)
                        .unwrap_or_default();
                }
            }
            DiscoverySource::PlatformSystem => {
                if let Some(dir) = system_plugin_dir() {
                    scan_dir(&dir, &source_id, priority_for_source(source), &mut results)
                        .unwrap_or_default();
                }
            }
            DiscoverySource::PlatformUser => {
                if let Some(dir) = user_plugin_dir() {
                    scan_dir(&dir, &source_id, priority_for_source(source), &mut results)
                        .unwrap_or_default();
                }
            }
        }
    }

    Ok(results)
}

fn scan_dir(
    dir: &Path,
    source_id: &DiscoverySourceId,
    priority: i32,
    out: &mut Vec<DiscoveredPackage>,
) -> Result<(), DiscoveryError> {
    if !dir.exists() {
        return Ok(());
    }
    let read = std::fs::read_dir(dir).map_err(|e| DiscoveryError::Io {
        path: dir.to_owned(),
        source: e,
    })?;
    for entry in read {
        let entry = entry.map_err(|e| DiscoveryError::Io {
            path: dir.to_owned(),
            source: e,
        })?;
        let path = entry.path();
        // `DirEntry::file_type` reports the entry's own type without following
        // it, so a symlink never redirects discovery outside the scanned tree.
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::debug!(
                    path = %path.display(),
                    error = %e,
                    "skipping plugin directory entry with unreadable file type"
                );
                continue;
            }
        };
        if file_type.is_symlink() {
            tracing::debug!(
                path = %path.display(),
                "skipping symlinked plugin directory entry"
            );
            continue;
        }
        if file_type.is_dir() {
            let candidate = path.join(MANIFEST_FILENAME);
            match candidate.symlink_metadata() {
                Ok(m) if m.file_type().is_file() => {
                    out.push(DiscoveredPackage {
                        // Raw path from `read_dir`; `canonicalize` would follow
                        // symlinks and defeat the file_type check above.
                        manifest_path: candidate,
                        source_id: source_id.clone(),
                        priority,
                    });
                }
                Ok(m) if m.file_type().is_symlink() => {
                    tracing::debug!(
                        path = %candidate.display(),
                        "skipping symlinked plugin manifest"
                    );
                }
                _ => {}
            }
        } else if file_type.is_file()
            && path
                .file_name()
                .map(|n| n == MANIFEST_FILENAME)
                .unwrap_or(false)
        {
            out.push(DiscoveredPackage {
                // Raw path from `read_dir`; `canonicalize` would follow
                // symlinks and defeat the file_type check above.
                manifest_path: path,
                source_id: source_id.clone(),
                priority,
            });
        }
    }
    Ok(())
}

fn priority_for_source(source: &DiscoverySource) -> i32 {
    // Higher ordinal ⟹ higher base priority tier.
    (source.kind_ordinal() as i32) * 100
}

fn distribution_plugin_dir() -> Option<PathBuf> {
    // Relative to the aiperf binary's location: ../share/aiperf/plugins
    std::env::current_exe().ok().and_then(|exe| {
        exe.parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("share/aiperf/plugins"))
    })
}

fn system_plugin_dir() -> Option<PathBuf> {
    Some(PathBuf::from("/usr/lib/aiperf/plugins"))
}

fn user_plugin_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|home| PathBuf::from(home).join(".aiperf/plugins"))
}
