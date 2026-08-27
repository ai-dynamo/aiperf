// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary-owned commit of a finalized report projection.
//!
//! The run's authoritative report is written exactly once, atomically, and
//! never over an existing authority. The serialization is
//! `serde_json::to_string_pretty` over whatever projection the caller
//! finalized, so the committed bytes are identical whether the caller is the
//! runtime's native report or a plugin-authored projection.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

/// Atomically commit a finalized report projection as pretty JSON to `path`.
///
/// Writes a sibling temporary file, syncs it, and renames it into place. The
/// rename is refused when `path` already exists: a run's report is written
/// once and never replaces an earlier authority.
pub fn write_finalized_report_json(value: &impl Serialize, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(value).context("serializing summary report")?;
    let (temporary_path, mut temporary) = create_temporary_report(path)?;
    let result = (|| {
        temporary
            .write_all(json.as_bytes())
            .with_context(|| format!("writing temporary report {}", temporary_path.display()))?;
        temporary
            .sync_all()
            .with_context(|| format!("syncing temporary report {}", temporary_path.display()))?;
        drop(temporary);
        if path.exists() {
            anyhow::bail!(
                "authoritative native report already exists: {}",
                path.display()
            );
        }
        std::fs::rename(&temporary_path, path).with_context(|| {
            format!(
                "committing temporary report {} to {}",
                temporary_path.display(),
                path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

fn create_temporary_report(path: &Path) -> Result<(PathBuf, std::fs::File)> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        anyhow::anyhow!("native report path has no file name: {}", path.display())
    })?;
    for sequence in 0..1_024_u16 {
        let temporary_path = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            std::process::id(),
            sequence
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("creating temporary report {}", temporary_path.display())
                });
            }
        }
    }
    anyhow::bail!(
        "could not reserve a temporary report beside {}",
        path.display()
    )
}
