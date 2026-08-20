// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, root-contained discovery for imported session JSONL files.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use serde_json::Value;

use crate::config::model::dataset::RecordedAgentSourceFormat;

use super::{
    AcquiredImportedAgentSelection, ImportedAgentError, ImportedAgentReadSet, ImportedAgentSource,
    ImportedAgentSourceFile, ImportedSessionFamily,
};

/// Acquire one private immutable imported-session selection.
pub fn acquire_imported_agent_selection(
    path: &Path,
    replay_root: Option<&Path>,
    source: RecordedAgentSourceFormat,
    include_subagents: Option<bool>,
) -> Result<AcquiredImportedAgentSelection, ImportedAgentError> {
    let scratch = tempfile::Builder::new()
        .prefix("aiperf-imported-session-")
        .tempdir()
        .map_err(|_| {
            error(
                path,
                0,
                "unknown",
                "unknown",
                "cannot create import snapshot",
            )
        })?;
    #[cfg(unix)]
    std::fs::set_permissions(scratch.path(), std::fs::Permissions::from_mode(0o700)).map_err(
        |_| {
            error(
                path,
                0,
                "unknown",
                "unknown",
                "cannot secure import snapshot",
            )
        },
    )?;
    let read_set = snapshot_imported_agent_read_set(
        path,
        replay_root,
        source,
        include_subagents,
        scratch.path(),
    )?;
    #[cfg(unix)]
    for file in &read_set.files {
        std::fs::set_permissions(&file.path, std::fs::Permissions::from_mode(0o400)).map_err(
            |_| {
                error(
                    &file.path,
                    0,
                    "unknown",
                    "unknown",
                    "cannot secure snapshot source",
                )
            },
        )?;
    }
    Ok(AcquiredImportedAgentSelection { scratch, read_set })
}

const SCAN_RECORD_LIMIT: usize = 20;

/// Detect the provider-native source format of one JSONL session file.
///
/// Detection inspects at most twenty non-empty JSON-object records.
pub fn detect_imported_agent_source(
    path: &Path,
) -> Result<ImportedAgentSource, ImportedAgentError> {
    let path = canonical_selected_file(path)?;
    scan_source(&path, open_source_file(&path, "unknown")?, None)?.ok_or_else(|| {
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
        RecordedAgentSourceFormat::Auto => scan_source(
            &selected_path,
            open_source_file(&selected_path, "unknown")?,
            None,
        )?
        .ok_or_else(|| {
            error(
                &selected_path,
                0,
                "unknown",
                "unknown",
                "no recognized source marker in scan",
            )
        })?,
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
        if scan_source(
            &canonical,
            open_source_file(&canonical, resolved_source_name(resolved_source))?,
            Some(resolved_source),
        )? != Some(resolved_source)
        {
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

/// Copy the exact discovered source set into a private immutable scratch root.
///
/// The read set stores paths, not source bytes: session parsers and cellular HTTP
/// serving therefore stream files from this controller-owned snapshot.
pub fn snapshot_imported_agent_read_set(
    path: &Path,
    replay_root: Option<&Path>,
    source: RecordedAgentSourceFormat,
    include_subagents: Option<bool>,
    snapshot_root: &Path,
) -> Result<ImportedAgentReadSet, ImportedAgentError> {
    let mut read_set =
        discover_imported_agent_read_set(path, replay_root, source, include_subagents)?;
    let selected_relative = read_set
        .selected_path
        .strip_prefix(&read_set.root)
        .map_err(|_| {
            error(
                &read_set.selected_path,
                0,
                "unknown",
                "unknown",
                "selected source escapes discovery root",
            )
        })?
        .to_path_buf();
    for file in &mut read_set.files {
        let mut source = open_source_file(&file.path, resolved_source_name(read_set.source))?;
        if scan_source(&file.path, &mut source, Some(read_set.source))? != Some(read_set.source) {
            return Err(error(
                &file.path,
                0,
                resolved_source_name(read_set.source),
                "unknown",
                "source marker does not match selected source",
            ));
        }
        let target = snapshot_root.join(&file.relative_path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|_| {
                error(
                    &target,
                    0,
                    "unknown",
                    "unknown",
                    "cannot create snapshot directory",
                )
            })?;
        }
        source.seek(SeekFrom::Start(0)).map_err(|_| {
            error(
                &file.path,
                0,
                "unknown",
                "unknown",
                "cannot read source file",
            )
        })?;
        let mut target_file = fs::File::create(&target).map_err(|_| {
            error(
                &target,
                0,
                "unknown",
                "unknown",
                "cannot create snapshot source",
            )
        })?;
        std::io::copy(&mut source, &mut target_file).map_err(|_| {
            error(
                &file.path,
                0,
                "unknown",
                "unknown",
                "cannot snapshot source file",
            )
        })?;
        file.path = target;
    }
    read_set.root = snapshot_root.to_path_buf();
    read_set.selected_path = snapshot_root.join(selected_relative);
    Ok(read_set)
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

fn open_source_file(path: &Path, source: &'static str) -> Result<fs::File, ImportedAgentError> {
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
    }
    .map_err(|_| error(path, 0, source, "unknown", "cannot read source file"))?;
    #[cfg(not(unix))]
    let file = std::fs::File::open(path)
        .map_err(|_| error(path, 0, source, "unknown", "cannot read source file"))?;
    let metadata = file
        .metadata()
        .map_err(|_| error(path, 0, source, "unknown", "cannot inspect source file"))?;
    if !metadata.is_file() {
        return Err(error(
            path,
            0,
            source,
            "unknown",
            "source must be a regular file",
        ));
    }
    Ok(file)
}

fn scan_source<R: std::io::Read>(
    path: &Path,
    source_bytes: R,
    expected: Option<ImportedAgentSource>,
) -> Result<Option<ImportedAgentSource>, ImportedAgentError> {
    let source = expected.map_or("unknown", resolved_source_name);
    let mut reader = BufReader::new(source_bytes);
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[cfg(unix)]
    #[test]
    fn acquired_selection_uses_private_immutable_snapshot_after_source_swap() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("session.jsonl");
        std::fs::write(
            &source,
            b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"original\"}}\n",
        )
        .unwrap();
        let selection =
            acquire_imported_agent_selection(&source, None, RecordedAgentSourceFormat::Codex, None)
                .unwrap();
        std::fs::write(
            &source,
            b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"replacement\"}}\n",
        )
        .unwrap();

        let read_set = selection.read_set();
        assert_eq!(
            std::fs::metadata(&read_set.root)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&read_set.files[0].path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o400
        );
        assert!(
            std::fs::read(&read_set.files[0].path)
                .unwrap()
                .windows(b"original".len())
                .any(|part| part == b"original")
        );
    }

    #[test]
    fn snapshot_discovery_keeps_the_opened_source_after_a_caller_swap() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("session.jsonl");
        let snapshot = temporary.path().join("snapshot");
        let original = b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"original\"}}\n";
        std::fs::write(&source, original).unwrap();

        let read_set = snapshot_imported_agent_read_set(
            &source,
            None,
            RecordedAgentSourceFormat::Codex,
            None,
            &snapshot,
        )
        .unwrap();
        std::fs::write(
            &source,
            b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"swapped\"}}\n",
        )
        .unwrap();

        assert_eq!(read_set.root, snapshot);
        assert_eq!(std::fs::read(&read_set.files[0].path).unwrap(), original);
    }

    #[test]
    fn snapshot_read_rejects_symlink_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("session.jsonl");
        let outside = temporary.path().join("outside.jsonl");
        std::fs::write(&outside, b"outside").unwrap();
        symlink(&outside, &source).unwrap();

        let error = open_source_file(&source, "codex").unwrap_err().to_string();
        assert!(error.contains("cannot read source file"), "{error}");
    }
}
