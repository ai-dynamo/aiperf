// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Multi-phase workflow manifest emission.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::engine::protocol::{PhaseRoleSpec, PhaseSpec};

#[derive(Serialize)]
struct PhaseManifest {
    schema_version: u32,
    phases: Vec<PhaseManifestEntry>,
}

#[derive(Serialize)]
struct PhaseManifestEntry {
    phase_name: String,
    phase_kind: String,
    phase_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    profiling_index: Option<usize>,
    exclude_from_results: bool,
}

/// Write `phase_manifest.json` when the workflow is not the canonical two-phase shape.
pub(crate) fn write_phase_manifest(artifact_dir: &Path, phases: &[PhaseSpec]) -> Result<()> {
    let non_canonical = phases.len() > 2
        || phases
            .iter()
            .any(|phase| !matches!(phase.common().name.as_str(), "warmup" | "profiling"));
    if !non_canonical {
        return Ok(());
    }
    let mut profiling_index = 0usize;
    let entries = phases
        .iter()
        .enumerate()
        .map(|(phase_index, phase)| {
            let common = phase.common();
            let role = common.semantic_role();
            let profiling_idx = if role == PhaseRoleSpec::Profiling {
                let idx = profiling_index;
                profiling_index += 1;
                Some(idx)
            } else {
                None
            };
            PhaseManifestEntry {
                phase_name: common.name.clone(),
                phase_kind: match role {
                    PhaseRoleSpec::Warmup => "warmup".into(),
                    PhaseRoleSpec::Profiling => "profiling".into(),
                },
                phase_index,
                profiling_index: profiling_idx,
                exclude_from_results: common.exclude_from_results,
            }
        })
        .collect();
    let payload = PhaseManifest {
        schema_version: 1,
        phases: entries,
    };
    let path = artifact_dir.join("phase_manifest.json");
    write_json(&path, &payload)
}

fn write_json(path: &Path, payload: &PhaseManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating manifest directory {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(payload).context("serializing phase manifest")?;
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Resolve a per-phase export directory under `phases/<name>/`.
#[allow(dead_code)]
pub(crate) fn phase_export_dir(artifact_dir: &Path, phase_name: &str) -> Result<PathBuf> {
    ensure!(
        !phase_name.is_empty() && !phase_name.contains('/') && !phase_name.contains('\\'),
        "invalid phase export name {phase_name:?}"
    );
    Ok(artifact_dir.join("phases").join(phase_name))
}
