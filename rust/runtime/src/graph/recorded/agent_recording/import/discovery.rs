// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, root-contained discovery for imported session JSONL files.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::config::model::dataset::RecordedAgentSourceFormat;

use super::{
    ImportedAgentError, ImportedAgentReadSet, ImportedAgentSource, ImportedAgentSourceFile,
    ImportedSessionFamily,
};

const SCAN_RECORD_LIMIT: usize = 20;

/// Detect the provider-native source format of one JSONL session file.
///
/// Detection inspects at most twenty non-empty JSON-object records.
pub fn detect_imported_agent_source(
    path: &Path,
) -> Result<ImportedAgentSource, ImportedAgentError> {
    let path = canonical_selected_file(path)?;
    scan_source(&path, None)?.ok_or_else(|| {
        error(
            &path,
            0,
            "unknown",
            "unknown",
            "no recognized source marker in scan",
        )
    })
}

/// Discover the exact source set for an explicit or single-file auto import.
pub fn discover_imported_agent_read_set(
    path: &Path,
    replay_root: Option<&Path>,
    source: RecordedAgentSourceFormat,
    include_subagents: Option<bool>,
) -> Result<ImportedAgentReadSet, ImportedAgentError> {
    let selected_path = canonical_selected_path(path, replay_root)?;
    let metadata = fs::symlink_metadata(&selected_path).map_err(|_| {
        error(
            &selected_path,
            0,
            "unknown",
            "unknown",
            "cannot inspect selected path",
        )
    })?;
    let is_directory = metadata.is_dir();
    let is_file = metadata.is_file();
    if !is_directory && !is_file {
        return Err(error(
            &selected_path,
            0,
            "unknown",
            "unknown",
            "selected path must be a regular file or directory",
        ));
    }
    if source == RecordedAgentSourceFormat::Auto && is_directory {
        return Err(error(
            &selected_path,
            0,
            "unknown",
            "unknown",
            "directory imports require an explicit source_format",
        ));
    }
    if source == RecordedAgentSourceFormat::MiniSweAgent {
        return Err(error(
            &selected_path,
            0,
            "unknown",
            "unknown",
            "Mini-SWE-Agent is not an imported session source",
        ));
    }
    if is_file && !is_jsonl(&selected_path) {
        return Err(error(
            &selected_path,
            0,
            "unknown",
            "unknown",
            "selected session source must be a .jsonl file",
        ));
    }

    let root = resolve_root(&selected_path, is_directory, replay_root)?;
    let resolved_source = match source {
        RecordedAgentSourceFormat::Auto => detect_imported_agent_source(&selected_path)?,
        RecordedAgentSourceFormat::Codex => ImportedAgentSource::Codex,
        RecordedAgentSourceFormat::ClaudeCode => ImportedAgentSource::ClaudeCode,
        RecordedAgentSourceFormat::MiniSweAgent => {
            return Err(error(
                &selected_path,
                0,
                "unknown",
                "unknown",
                "Mini-SWE-Agent is not an imported session source",
            ));
        }
    };
    if include_subagents.is_some() && resolved_source != ImportedAgentSource::ClaudeCode {
        return Err(error(
            &selected_path,
            0,
            resolved_source_name(resolved_source),
            "unknown",
            "include_subagents applies only to Claude Code sources",
        ));
    }
    let candidates = match (resolved_source, is_directory) {
        (ImportedAgentSource::Codex, true) => enumerate_codex(&selected_path)?,
        (ImportedAgentSource::Codex, false) => {
            vec![(selected_path.clone(), ImportedSessionFamily::Session)]
        }
        (ImportedAgentSource::ClaudeCode, true) => {
            enumerate_claude(&selected_path, include_subagents.unwrap_or(true))?
        }
        (ImportedAgentSource::ClaudeCode, false) => {
            vec![(selected_path.clone(), ImportedSessionFamily::Session)]
        }
    };

    let mut canonical_paths = HashSet::new();
    let mut files = Vec::with_capacity(candidates.len());
    for (candidate, family) in candidates {
        let canonical = canonical_regular_file(&candidate, &root)?;
        if !canonical_paths.insert(canonical.clone()) {
            return Err(error(
                &candidate,
                0,
                resolved_source_name(resolved_source),
                "unknown",
                "duplicate canonical source path",
            ));
        }
        if scan_source(&canonical, Some(resolved_source))? != Some(resolved_source) {
            return Err(error(
                &canonical,
                0,
                resolved_source_name(resolved_source),
                "unknown",
                "source marker does not match selected source",
            ));
        }
        let relative_path = canonical.strip_prefix(&root).map_err(|_| {
            error(
                &canonical,
                0,
                resolved_source_name(resolved_source),
                "unknown",
                "source escapes discovery root",
            )
        })?;
        let relative_path = relative_path.to_path_buf();
        if relative_path.as_os_str().is_empty() {
            return Err(error(
                &canonical,
                0,
                resolved_source_name(resolved_source),
                "unknown",
                "invalid root-relative source path",
            ));
        }
        files.push(ImportedAgentSourceFile {
            path: canonical,
            relative_path,
            family,
        });
    }
    if resolved_source == ImportedAgentSource::Codex {
        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    }
    Ok(ImportedAgentReadSet {
        root,
        selected_path,
        source: resolved_source,
        files,
    })
}

fn resolve_root(
    selected_path: &Path,
    is_directory: bool,
    replay_root: Option<&Path>,
) -> Result<PathBuf, ImportedAgentError> {
    match replay_root {
        Some(root) => canonical_root(root),
        None if is_directory => canonical_root(selected_path),
        None => selected_path
            .parent()
            .map(canonical_root)
            .transpose()?
            .ok_or_else(|| {
                error(
                    selected_path,
                    0,
                    "unknown",
                    "unknown",
                    "selected file has no parent directory",
                )
            }),
    }
}

fn canonical_selected_path(
    path: &Path,
    replay_root: Option<&Path>,
) -> Result<PathBuf, ImportedAgentError> {
    let requested = if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(root) = replay_root {
        root.join(path)
    } else {
        path.to_path_buf()
    };
    reject_symlink_components(&requested)?;
    let selected = fs::canonicalize(&requested).map_err(|_| {
        error(
            &requested,
            0,
            "unknown",
            "unknown",
            "cannot resolve selected path",
        )
    })?;
    if let Some(root) = replay_root {
        let root = canonical_root(root)?;
        if !selected.starts_with(root) {
            return Err(error(
                &requested,
                0,
                "unknown",
                "unknown",
                "selected path escapes replay root",
            ));
        }
    }
    Ok(selected)
}

fn canonical_selected_file(path: &Path) -> Result<PathBuf, ImportedAgentError> {
    reject_symlink_components(path)?;
    let selected = fs::canonicalize(path).map_err(|_| {
        error(
            path,
            0,
            "unknown",
            "unknown",
            "cannot resolve selected path",
        )
    })?;
    let metadata = fs::symlink_metadata(&selected).map_err(|_| {
        error(
            &selected,
            0,
            "unknown",
            "unknown",
            "cannot inspect selected path",
        )
    })?;
    if !metadata.is_file() || !is_jsonl(&selected) {
        return Err(error(
            &selected,
            0,
            "unknown",
            "unknown",
            "selected session source must be a regular .jsonl file",
        ));
    }
    Ok(selected)
}

fn canonical_root(root: &Path) -> Result<PathBuf, ImportedAgentError> {
    reject_symlink_components(root)?;
    let canonical = fs::canonicalize(root)
        .map_err(|_| error(root, 0, "unknown", "unknown", "cannot resolve replay root"))?;
    let metadata = fs::symlink_metadata(&canonical).map_err(|_| {
        error(
            &canonical,
            0,
            "unknown",
            "unknown",
            "cannot inspect replay root",
        )
    })?;
    if !metadata.is_dir() {
        return Err(error(
            &canonical,
            0,
            "unknown",
            "unknown",
            "replay root must be a directory",
        ));
    }
    Ok(canonical)
}

fn canonical_regular_file(path: &Path, root: &Path) -> Result<PathBuf, ImportedAgentError> {
    reject_symlink_components(path)?;
    let canonical = fs::canonicalize(path)
        .map_err(|_| error(path, 0, "unknown", "unknown", "cannot resolve source file"))?;
    if !canonical.starts_with(root) {
        return Err(error(
            path,
            0,
            "unknown",
            "unknown",
            "source file escapes discovery root",
        ));
    }
    let metadata = fs::symlink_metadata(&canonical).map_err(|_| {
        error(
            &canonical,
            0,
            "unknown",
            "unknown",
            "cannot inspect source file",
        )
    })?;
    if !metadata.is_file() {
        return Err(error(
            &canonical,
            0,
            "unknown",
            "unknown",
            "source entry must be a regular file",
        ));
    }
    Ok(canonical)
}

fn enumerate_codex(
    root: &Path,
) -> Result<Vec<(PathBuf, ImportedSessionFamily)>, ImportedAgentError> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = stack.pop() {
        let mut entries = fs::read_dir(&directory)
            .map_err(|_| {
                error(
                    &directory,
                    0,
                    "codex",
                    "unknown",
                    "cannot read source directory",
                )
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                error(
                    &directory,
                    0,
                    "codex",
                    "unknown",
                    "cannot read source directory",
                )
            })?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries.into_iter().rev() {
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|_| error(&path, 0, "codex", "unknown", "cannot inspect source entry"))?;
            if metadata.file_type().is_symlink() {
                return Err(error(
                    &path,
                    0,
                    "codex",
                    "unknown",
                    "symlink source entries are forbidden",
                ));
            }
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() && is_jsonl(&path) {
                files.push((path, ImportedSessionFamily::Session));
            }
        }
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(files)
}

fn enumerate_claude(
    root: &Path,
    include_subagents: bool,
) -> Result<Vec<(PathBuf, ImportedSessionFamily)>, ImportedAgentError> {
    let mut entries = fs::read_dir(root)
        .map_err(|_| {
            error(
                root,
                0,
                "claude_code",
                "unknown",
                "cannot read source directory",
            )
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            error(
                root,
                0,
                "claude_code",
                "unknown",
                "cannot read source directory",
            )
        })?;
    entries.sort_by_key(|entry| entry.file_name());
    let mut mains = Vec::new();
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|_| {
            error(
                &path,
                0,
                "claude_code",
                "unknown",
                "cannot inspect source entry",
            )
        })?;
        if metadata.file_type().is_symlink() {
            return Err(error(
                &path,
                0,
                "claude_code",
                "unknown",
                "symlink source entries are forbidden",
            ));
        }
        if metadata.is_file() && is_jsonl(&path) {
            mains.push(path);
        }
    }
    let mut files = mains
        .iter()
        .cloned()
        .map(|path| (path, ImportedSessionFamily::Session))
        .collect::<Vec<_>>();
    if include_subagents {
        for main in mains {
            let Some(stem) = main.file_stem() else {
                continue;
            };
            let subagents = root.join(stem).join("subagents");
            if !subagents.exists() {
                continue;
            }
            reject_symlink_components(&subagents)?;
            let metadata = fs::symlink_metadata(&subagents).map_err(|_| {
                error(
                    &subagents,
                    0,
                    "claude_code",
                    "unknown",
                    "cannot inspect subagent directory",
                )
            })?;
            if !metadata.is_dir() {
                return Err(error(
                    &subagents,
                    0,
                    "claude_code",
                    "unknown",
                    "subagent path must be a directory",
                ));
            }
            let mut subagent_entries = fs::read_dir(&subagents)
                .map_err(|_| {
                    error(
                        &subagents,
                        0,
                        "claude_code",
                        "unknown",
                        "cannot read subagent directory",
                    )
                })?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    error(
                        &subagents,
                        0,
                        "claude_code",
                        "unknown",
                        "cannot read subagent directory",
                    )
                })?;
            subagent_entries.sort_by_key(|entry| entry.file_name());
            for entry in subagent_entries {
                let path = entry.path();
                let metadata = fs::symlink_metadata(&path).map_err(|_| {
                    error(
                        &path,
                        0,
                        "claude_code",
                        "unknown",
                        "cannot inspect subagent entry",
                    )
                })?;
                if metadata.file_type().is_symlink() {
                    return Err(error(
                        &path,
                        0,
                        "claude_code",
                        "unknown",
                        "symlink source entries are forbidden",
                    ));
                }
                if metadata.is_file() && is_agent_subagent_name(&path) {
                    files.push((path, ImportedSessionFamily::Subagent));
                }
            }
        }
    }
    Ok(files)
}

fn scan_source(
    path: &Path,
    expected: Option<ImportedAgentSource>,
) -> Result<Option<ImportedAgentSource>, ImportedAgentError> {
    let source = expected.map_or("unknown", resolved_source_name);
    let file = File::open(path)
        .map_err(|_| error(path, 0, source, "unknown", "cannot read source file"))?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    let mut line = 0;
    let mut records = 0;
    let mut detected = None;
    loop {
        if records == SCAN_RECORD_LIMIT {
            break;
        }
        bytes.clear();
        let read = reader.read_until(b'\n', &mut bytes).map_err(|_| {
            error(
                path,
                line.max(1),
                source,
                "unknown",
                "cannot read source file",
            )
        })?;
        if read == 0 {
            break;
        }
        line += 1;
        if bytes.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        records += 1;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| error(path, line, source, "unknown", "invalid JSON"))?;
        let object = value.as_object().ok_or_else(|| {
            error(
                path,
                line,
                source,
                "unknown",
                "record must be a JSON object",
            )
        })?;
        let record_type = object.get("type").and_then(Value::as_str);
        let has_object_payload = object.get("payload").is_some_and(Value::is_object);
        let is_codex = matches!(
            record_type,
            Some("session_meta" | "event_msg" | "response_item" | "turn_context")
        ) && has_object_payload;
        let is_claude = object.contains_key("sessionId")
            && (object.contains_key("parentUuid")
                || matches!(
                    record_type,
                    Some("permission-mode" | "file-history-snapshot" | "summary")
                ));
        if is_codex && is_claude {
            return Err(error(
                path,
                line,
                "unknown",
                "unknown",
                "ambiguous source markers",
            ));
        }
        let marker = if is_codex {
            Some(ImportedAgentSource::Codex)
        } else if is_claude {
            Some(ImportedAgentSource::ClaudeCode)
        } else {
            None
        };
        if let Some(marker) = marker {
            if let Some(expected) = expected
                && marker != expected
            {
                return Err(error(
                    path,
                    line,
                    resolved_source_name(expected),
                    "unknown",
                    "source marker does not match selected source",
                ));
            }
            if let Some(previous) = detected
                && previous != marker
            {
                return Err(error(
                    path,
                    line,
                    "unknown",
                    "unknown",
                    "ambiguous source markers",
                ));
            }
            detected = Some(marker);
        }
    }
    Ok(detected)
}

fn reject_symlink_components(path: &Path) -> Result<(), ImportedAgentError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = fs::symlink_metadata(&current)
            && metadata.file_type().is_symlink()
        {
            return Err(error(
                &current,
                0,
                "unknown",
                "unknown",
                "symlink source entries are forbidden",
            ));
        }
    }
    Ok(())
}

fn is_jsonl(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".jsonl"))
}

fn is_agent_subagent_name(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(stem) = name.strip_suffix(".jsonl") else {
        return false;
    };
    stem.strip_prefix("agent-")
        .is_some_and(|identifier| !identifier.is_empty())
}

fn resolved_source_name(source: ImportedAgentSource) -> &'static str {
    match source {
        ImportedAgentSource::Codex => "codex",
        ImportedAgentSource::ClaudeCode => "claude_code",
    }
}

fn error(
    path: &Path,
    line: usize,
    source: &'static str,
    record_label: &'static str,
    detail: &'static str,
) -> ImportedAgentError {
    ImportedAgentError::new(path, line, source, record_label, detail)
}
