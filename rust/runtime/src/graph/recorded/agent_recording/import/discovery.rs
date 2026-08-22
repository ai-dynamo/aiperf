// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, root-contained discovery for imported session JSONL files.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::{
    ffi::CString,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::{ffi::OsStrExt, fs::PermissionsExt},
    },
    path::Component,
};

use serde_json::Value;

use crate::config::model::dataset::RecordedAgentSourceFormat;

use super::{
    AcquiredImportedAgentSelection, ImportedAgentError, ImportedAgentReadSet,
    ImportedAgentSelectionRequest, ImportedAgentSource, ImportedAgentSourceFile,
    ImportedSessionFamily,
};

struct DiscoveredImportedSelection {
    root: PathBuf,
    #[cfg(unix)]
    root_directory: fs::File,
    selected_path: PathBuf,
    requested_source: RecordedAgentSourceFormat,
    candidates: Vec<(PathBuf, PathBuf, ImportedSessionFamily)>,
}

struct OpenedImportedAgentSource {
    source_path: PathBuf,
    relative_path: PathBuf,
    family: ImportedSessionFamily,
    file: fs::File,
}

impl OpenedImportedAgentSource {
    fn open(
        discovered: &DiscoveredImportedSelection,
        source_path: PathBuf,
        relative_path: PathBuf,
        family: ImportedSessionFamily,
    ) -> Result<Self, ImportedAgentError> {
        #[cfg(unix)]
        {
            let file = open_source_file_beneath(
                &discovered.root_directory,
                &relative_path,
                &source_path,
                "unknown",
            )?;
            Ok(Self {
                source_path,
                relative_path,
                family,
                file,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = discovered;
            let _ = relative_path;
            let _ = family;
            Err(error(
                &source_path,
                0,
                "unknown",
                "unknown",
                "secure source acquisition is unavailable on this platform",
            ))
        }
    }

    fn materialize(
        mut self,
        snapshot_root: &Path,
        expected_source: ImportedAgentSource,
    ) -> Result<ImportedAgentSourceFile, ImportedAgentError> {
        self.file.seek(SeekFrom::Start(0)).map_err(|_| {
            error(
                &self.source_path,
                0,
                resolved_source_name(expected_source),
                "unknown",
                "cannot read source file",
            )
        })?;
        if scan_source(
            &self.source_path,
            &mut self.file,
            SourceScanMode::Validate(expected_source),
        )? != Some(expected_source)
        {
            return Err(error(
                &self.source_path,
                0,
                resolved_source_name(expected_source),
                "unknown",
                "source marker does not match selected source",
            ));
        }
        self.file.seek(SeekFrom::Start(0)).map_err(|_| {
            error(
                &self.source_path,
                0,
                resolved_source_name(expected_source),
                "unknown",
                "cannot read source file",
            )
        })?;
        let target = snapshot_root.join(&self.relative_path);
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
        let mut target_file = fs::File::create(&target).map_err(|_| {
            error(
                &target,
                0,
                "unknown",
                "unknown",
                "cannot create snapshot source",
            )
        })?;
        std::io::copy(&mut self.file, &mut target_file).map_err(|_| {
            error(
                &self.source_path,
                0,
                resolved_source_name(expected_source),
                "unknown",
                "cannot snapshot source file",
            )
        })?;
        #[cfg(unix)]
        fs::set_permissions(&target, fs::Permissions::from_mode(0o400)).map_err(|_| {
            error(
                &target,
                0,
                "unknown",
                "unknown",
                "cannot secure snapshot source",
            )
        })?;
        Ok(ImportedAgentSourceFile {
            path: target,
            relative_path: self.relative_path,
            family: self.family,
        })
    }
}

pub(super) fn acquire_selection(
    request: ImportedAgentSelectionRequest,
    parent: Option<&Path>,
) -> Result<AcquiredImportedAgentSelection, ImportedAgentError> {
    let scratch = match parent {
        Some(parent) => tempfile::Builder::new()
            .prefix("aiperf-imported-session-")
            .tempdir_in(parent),
        None => tempfile::Builder::new()
            .prefix("aiperf-imported-session-")
            .tempdir(),
    }
    .map_err(|_| {
        error(
            &request.path,
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
                &request.path,
                0,
                "unknown",
                "unknown",
                "cannot secure import snapshot",
            )
        },
    )?;
    let mut discovered = discover_selection(&request)?;
    let candidates = std::mem::take(&mut discovered.candidates);
    let source = match discovered.requested_source {
        RecordedAgentSourceFormat::Auto => {
            let mut candidates = candidates.into_iter();
            let (source_path, relative_path, family) = candidates.next().ok_or_else(|| {
                error(
                    &discovered.selected_path,
                    0,
                    "unknown",
                    "unknown",
                    "no recognized source marker in scan",
                )
            })?;
            let mut opened =
                OpenedImportedAgentSource::open(&discovered, source_path, relative_path, family)?;
            let source = scan_source(
                &opened.source_path,
                &mut opened.file,
                SourceScanMode::Detect,
            )?
            .ok_or_else(|| {
                error(
                    &opened.source_path,
                    0,
                    "unknown",
                    "unknown",
                    "no recognized source marker in scan",
                )
            })?;
            if request.include_subagents.is_some() && source != ImportedAgentSource::ClaudeCode {
                return Err(error(
                    &discovered.selected_path,
                    0,
                    resolved_source_name(source),
                    "unknown",
                    "include_subagents applies only to Claude Code sources",
                ));
            }
            let files = vec![opened.materialize(scratch.path(), source)?];
            return acquired_selection(discovered, scratch, source, files);
        }
        RecordedAgentSourceFormat::Codex => ImportedAgentSource::Codex,
        RecordedAgentSourceFormat::ClaudeCode => ImportedAgentSource::ClaudeCode,
        RecordedAgentSourceFormat::MiniSweAgent => unreachable!("validated by request"),
    };
    if request.include_subagents.is_some() && source != ImportedAgentSource::ClaudeCode {
        return Err(error(
            &discovered.selected_path,
            0,
            resolved_source_name(source),
            "unknown",
            "include_subagents applies only to Claude Code sources",
        ));
    }
    let mut files = Vec::with_capacity(candidates.len());
    for (source_path, relative_path, family) in candidates {
        let opened =
            OpenedImportedAgentSource::open(&discovered, source_path, relative_path, family)?;
        files.push(opened.materialize(scratch.path(), source)?);
    }
    if source == ImportedAgentSource::Codex {
        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    }
    acquired_selection(discovered, scratch, source, files)
}

fn acquired_selection(
    discovered: DiscoveredImportedSelection,
    scratch: tempfile::TempDir,
    source: ImportedAgentSource,
    files: Vec<ImportedAgentSourceFile>,
) -> Result<AcquiredImportedAgentSelection, ImportedAgentError> {
    let selected_relative = discovered
        .selected_path
        .strip_prefix(&discovered.root)
        .map_err(|_| {
            error(
                &discovered.selected_path,
                0,
                "unknown",
                "unknown",
                "selected source escapes discovery root",
            )
        })?;
    let read_set = ImportedAgentReadSet {
        root: scratch.path().to_path_buf(),
        selected_path: scratch.path().join(selected_relative),
        source,
        files,
    };
    Ok(AcquiredImportedAgentSelection { scratch, read_set })
}

const SCAN_RECORD_LIMIT: usize = 20;

#[derive(Clone, Copy)]
enum SourceScanMode {
    Detect,
    Validate(ImportedAgentSource),
}

/// Detect the provider-native source format of one JSONL session file.
///
/// Detection inspects at most twenty non-empty JSON-object records.
pub fn detect_imported_agent_source(
    path: &Path,
) -> Result<ImportedAgentSource, ImportedAgentError> {
    let path = canonical_selected_file(path)?;
    scan_source(
        &path,
        open_source_file(&path, "unknown")?,
        SourceScanMode::Detect,
    )?
    .ok_or_else(|| {
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
    let request = ImportedAgentSelectionRequest::new(
        path.to_path_buf(),
        replay_root.map(Path::to_path_buf),
        source,
        include_subagents,
    )?;
    let discovered = discover_selection(&request)?;
    let source = match discovered.requested_source {
        RecordedAgentSourceFormat::Auto => detect_imported_agent_source(&discovered.selected_path)?,
        RecordedAgentSourceFormat::Codex => ImportedAgentSource::Codex,
        RecordedAgentSourceFormat::ClaudeCode => ImportedAgentSource::ClaudeCode,
        RecordedAgentSourceFormat::MiniSweAgent => unreachable!("validated by request"),
    };
    let mut files = Vec::with_capacity(discovered.candidates.len());
    for (path, relative_path, family) in discovered.candidates {
        if scan_source(
            &path,
            open_source_file(&path, resolved_source_name(source))?,
            SourceScanMode::Validate(source),
        )? != Some(source)
        {
            return Err(error(
                &path,
                0,
                resolved_source_name(source),
                "unknown",
                "source marker does not match selected source",
            ));
        }
        files.push(ImportedAgentSourceFile {
            path,
            relative_path,
            family,
        });
    }
    if source == ImportedAgentSource::Codex {
        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    }
    Ok(ImportedAgentReadSet {
        root: discovered.root,
        selected_path: discovered.selected_path,
        source,
        files,
    })
}

fn discover_selection(
    request: &ImportedAgentSelectionRequest,
) -> Result<DiscoveredImportedSelection, ImportedAgentError> {
    let selected_path = canonical_selected_path(&request.path, request.replay_root.as_deref())?;
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
    if request.source_format == RecordedAgentSourceFormat::Auto && is_directory {
        return Err(error(
            &selected_path,
            0,
            "unknown",
            "unknown",
            "directory imports require an explicit source_format",
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

    let root = resolve_root(&selected_path, is_directory, request.replay_root.as_deref())?;
    #[cfg(unix)]
    let root_directory = open_directory_nofollow(&root)?;
    let candidates = match (request.source_format, is_directory) {
        (RecordedAgentSourceFormat::Codex, true) => enumerate_codex(&selected_path)?,
        (RecordedAgentSourceFormat::Codex | RecordedAgentSourceFormat::Auto, false) => {
            vec![(selected_path.clone(), ImportedSessionFamily::Session)]
        }
        (RecordedAgentSourceFormat::ClaudeCode, true) => {
            enumerate_claude(&selected_path, request.include_subagents.unwrap_or(true))?
        }
        (RecordedAgentSourceFormat::ClaudeCode, false) => {
            vec![(selected_path.clone(), ImportedSessionFamily::Session)]
        }
        (RecordedAgentSourceFormat::Auto, true) => unreachable!("directory Auto is rejected"),
        (RecordedAgentSourceFormat::MiniSweAgent, _) => unreachable!("validated by request"),
    };

    let mut canonical_paths = HashSet::new();
    let mut files = Vec::with_capacity(candidates.len());
    for (candidate, family) in candidates {
        let canonical = canonical_regular_file(&candidate, &root)?;
        if !canonical_paths.insert(canonical.clone()) {
            return Err(error(
                &candidate,
                0,
                "unknown",
                "unknown",
                "duplicate canonical source path",
            ));
        }
        let relative_path = canonical.strip_prefix(&root).map_err(|_| {
            error(
                &canonical,
                0,
                "unknown",
                "unknown",
                "source escapes discovery root",
            )
        })?;
        let relative_path = relative_path.to_path_buf();
        if relative_path.as_os_str().is_empty() {
            return Err(error(
                &canonical,
                0,
                "unknown",
                "unknown",
                "invalid root-relative source path",
            ));
        }
        files.push((canonical, relative_path, family));
    }
    Ok(DiscoveredImportedSelection {
        root,
        #[cfg(unix)]
        root_directory,
        selected_path,
        requested_source: request.source_format,
        candidates: files,
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

#[cfg(unix)]
fn open_directory_nofollow(path: &Path) -> Result<fs::File, ImportedAgentError> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | directory_open_flags())
        .open(Path::new("/"))
        .map_err(|_| error(path, 0, "unknown", "unknown", "cannot read replay root"))?;
    let relative = path.strip_prefix(Path::new("/")).map_err(|_| {
        error(
            path,
            0,
            "unknown",
            "unknown",
            "replay root must be absolute",
        )
    })?;
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(error(
                path,
                0,
                "unknown",
                "unknown",
                "invalid replay root component",
            ));
        };
        directory = openat_nofollow(
            directory.as_raw_fd(),
            name,
            directory_open_flags(),
            path,
            "unknown",
        )?;
    }
    Ok(directory)
}

#[cfg(unix)]
fn open_source_file_beneath(
    root: &fs::File,
    relative_path: &Path,
    source_path: &Path,
    source: &'static str,
) -> Result<fs::File, ImportedAgentError> {
    let mut components = relative_path.components().peekable();
    let mut directory = None;
    while let Some(component) = components.next() {
        let directory_fd = directory
            .as_ref()
            .map_or_else(|| root.as_raw_fd(), AsRawFd::as_raw_fd);
        let Component::Normal(name) = component else {
            return Err(error(
                source_path,
                0,
                source,
                "unknown",
                "invalid root-relative source path",
            ));
        };
        if components.peek().is_none() {
            let file = openat_nofollow(
                directory_fd,
                name,
                libc::O_RDONLY | libc::O_NONBLOCK,
                source_path,
                source,
            )?;
            let metadata = file.metadata().map_err(|_| {
                error(
                    source_path,
                    0,
                    source,
                    "unknown",
                    "cannot inspect source file",
                )
            })?;
            if !metadata.is_file() {
                return Err(error(
                    source_path,
                    0,
                    source,
                    "unknown",
                    "source must be a regular file",
                ));
            }
            return Ok(file);
        }
        let next = openat_nofollow(
            directory_fd,
            name,
            directory_open_flags(),
            source_path,
            source,
        )?;
        directory = Some(next);
    }
    Err(error(
        source_path,
        0,
        source,
        "unknown",
        "invalid root-relative source path",
    ))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
const fn directory_open_flags() -> libc::c_int {
    libc::O_PATH | libc::O_DIRECTORY
}

#[cfg(target_vendor = "apple")]
const fn directory_open_flags() -> libc::c_int {
    libc::O_SEARCH | libc::O_DIRECTORY
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
const fn directory_open_flags() -> libc::c_int {
    libc::O_RDONLY | libc::O_DIRECTORY
}

#[cfg(unix)]
fn openat_nofollow(
    directory_fd: RawFd,
    name: &std::ffi::OsStr,
    flags: libc::c_int,
    source_path: &Path,
    source: &'static str,
) -> Result<fs::File, ImportedAgentError> {
    let name = CString::new(name.as_bytes()).map_err(|_| {
        error(
            source_path,
            0,
            source,
            "unknown",
            "invalid source path component",
        )
    })?;
    // SAFETY: `directory_fd` is borrowed from a live `File`, `name` is NUL-terminated, and
    // `openat` does not retain either argument after returning.
    let descriptor = unsafe {
        libc::openat(
            directory_fd,
            name.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if descriptor < 0 {
        return Err(error(
            source_path,
            0,
            source,
            "unknown",
            "cannot read source file",
        ));
    }
    // SAFETY: a nonnegative `openat` result is a newly owned descriptor transferred exactly once.
    let descriptor = unsafe { OwnedFd::from_raw_fd(descriptor) };
    Ok(fs::File::from(descriptor))
}

fn scan_source<R: std::io::Read>(
    path: &Path,
    source_bytes: R,
    mode: SourceScanMode,
) -> Result<Option<ImportedAgentSource>, ImportedAgentError> {
    let source = match mode {
        SourceScanMode::Detect => "unknown",
        SourceScanMode::Validate(expected) => resolved_source_name(expected),
    };
    let mut reader = BufReader::new(source_bytes);
    let mut bytes = Vec::new();
    let mut line = 0;
    let mut records = 0;
    let mut detected = None;
    loop {
        if matches!(mode, SourceScanMode::Detect) && records == SCAN_RECORD_LIMIT {
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
            if let SourceScanMode::Validate(expected) = mode {
                if marker != expected {
                    return Err(error(
                        path,
                        line,
                        resolved_source_name(expected),
                        "unknown",
                        "source marker does not match selected source",
                    ));
                }
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

    const DESCRIPTOR_LIMIT_CHILD_ENV: &str = "AIPERF_IMPORTED_ACQUISITION_DESCRIPTOR_LIMIT";

    #[test]
    fn imported_acquisition_materializes_explicit_sources_with_bounded_descriptors() {
        if std::env::var_os(DESCRIPTOR_LIMIT_CHILD_ENV).is_some() {
            let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
            unsafe {
                assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, limit.as_mut_ptr()), 0);
            }
            let mut limit = unsafe { limit.assume_init() };
            limit.rlim_cur = limit.rlim_cur.min(32);
            unsafe {
                assert_eq!(libc::setrlimit(libc::RLIMIT_NOFILE, &limit), 0);
            }
            let temporary = tempfile::tempdir().unwrap();
            for index in 0..64 {
                std::fs::write(
                    temporary.path().join(format!("session-{index:03}.jsonl")),
                    b"{\"type\":\"session_meta\",\"payload\":{}}\n",
                )
                .unwrap();
            }
            let request = ImportedAgentSelectionRequest::new(
                temporary.path().to_path_buf(),
                None,
                RecordedAgentSourceFormat::Codex,
                None,
            )
            .unwrap();
            let selection = request.acquire().expect("acquire all explicit sources");
            assert_eq!(selection.read_set().files.len(), 64);
            return;
        }

        let executable = std::env::current_exe().unwrap();
        let status = std::process::Command::new(executable)
            .arg("--exact")
            .arg("graph::recorded::agent_recording::import::discovery::tests::imported_acquisition_materializes_explicit_sources_with_bounded_descriptors")
            .arg("--nocapture")
            .env(DESCRIPTOR_LIMIT_CHILD_ENV, "1")
            .status()
            .expect("run descriptor-limited acquisition child");
        assert!(
            status.success(),
            "descriptor-limited child failed: {status}"
        );
    }

    #[test]
    fn imported_acquisition_opened_source_copy_is_bound_to_open_inode() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("session.jsonl");
        let replacement = temporary.path().join("replacement.jsonl");
        let original = b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"original\"}}\n";
        std::fs::write(&source, original).unwrap();

        let mut discovered = discover_selection(
            &ImportedAgentSelectionRequest::new(
                source.clone(),
                None,
                RecordedAgentSourceFormat::Codex,
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let (source_path, relative_path, family) = discovered.candidates.remove(0);
        let opened =
            OpenedImportedAgentSource::open(&discovered, source_path, relative_path, family)
                .expect("open original source once");
        std::fs::write(
            &replacement,
            b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"replacement\"}}\n",
        )
        .unwrap();
        std::fs::rename(&replacement, &source).unwrap();

        let scratch = tempfile::tempdir().unwrap();
        let file = opened
            .materialize(scratch.path(), ImportedAgentSource::Codex)
            .expect("materialize opened source");
        assert_eq!(std::fs::read(file.path).unwrap(), original);
    }

    #[test]
    fn imported_acquisition_refuses_ancestor_symlink_swap_after_discovery() {
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        std::fs::create_dir(&sessions).unwrap();
        let source = sessions.join("session.jsonl");
        std::fs::write(
            &source,
            b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"original\"}}\n",
        )
        .unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_bytes = b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"outside\"}}\n";
        std::fs::write(outside.path().join("session.jsonl"), outside_bytes).unwrap();
        let request = ImportedAgentSelectionRequest::new(
            root.path().to_path_buf(),
            None,
            RecordedAgentSourceFormat::Codex,
            None,
        )
        .unwrap();
        let mut discovered = discover_selection(&request).unwrap();
        let (source_path, relative_path, family) = discovered.candidates.remove(0);

        std::fs::rename(&sessions, root.path().join("validated-sessions")).unwrap();
        symlink(outside.path(), &sessions).unwrap();

        let snapshot = tempfile::tempdir().unwrap();
        let result = OpenedImportedAgentSource::open(
            &discovered,
            source_path,
            relative_path.clone(),
            family,
        )
        .and_then(|opened| opened.materialize(snapshot.path(), ImportedAgentSource::Codex));

        assert!(
            result.is_err(),
            "acquisition must refuse a source whose validated ancestor became a symlink"
        );
        assert_ne!(
            std::fs::read(snapshot.path().join(relative_path))
                .ok()
                .as_deref(),
            Some(outside_bytes.as_slice()),
            "outside bytes must not enter the acquired snapshot"
        );
    }

    #[test]
    fn imported_acquisition_refuses_fifo_replacement_without_blocking() {
        use std::sync::mpsc::RecvTimeoutError;
        use std::time::Duration;

        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("session.jsonl");
        std::fs::write(&source, b"{\"type\":\"session_meta\",\"payload\":{}}\n").unwrap();
        let mut discovered = discover_selection(
            &ImportedAgentSelectionRequest::new(
                source.clone(),
                None,
                RecordedAgentSourceFormat::Codex,
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let (source_path, relative_path, family) = discovered.candidates.remove(0);
        std::fs::remove_file(&source).unwrap();
        let fifo = CString::new(source.as_os_str().as_bytes()).unwrap();
        // SAFETY: `fifo` is a live, NUL-free temporary path and the mode is valid.
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);

        let (sender, receiver) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let result =
                OpenedImportedAgentSource::open(&discovered, source_path, relative_path, family);
            let _ = sender.send(result);
        });
        let result = match receiver.recv_timeout(Duration::from_secs(1)) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                let _writer = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&source)
                    .expect("unblock FIFO reader");
                let _ = receiver.recv_timeout(Duration::from_secs(1));
                reader.join().expect("reader thread");
                panic!("recording FIFO open blocked before rejecting its file type");
            }
            Err(RecvTimeoutError::Disconnected) => panic!("reader disconnected"),
        };
        reader.join().expect("reader thread");
        assert!(result.is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn imported_acquisition_supports_source_beneath_search_only_ancestor() {
        let temporary = tempfile::tempdir().unwrap();
        let search_only = temporary.path().join("search-only");
        std::fs::create_dir(&search_only).unwrap();
        let source = search_only.join("session.jsonl");
        std::fs::write(&source, b"{\"type\":\"session_meta\",\"payload\":{}}\n").unwrap();
        std::fs::set_permissions(&search_only, std::fs::Permissions::from_mode(0o111)).unwrap();

        let result = ImportedAgentSelectionRequest::new(
            source,
            None,
            RecordedAgentSourceFormat::Codex,
            None,
        )
        .and_then(ImportedAgentSelectionRequest::acquire);

        std::fs::set_permissions(&search_only, std::fs::Permissions::from_mode(0o700)).unwrap();
        let error = result.err();
        assert!(error.is_none(), "search-only ancestors failed: {error:?}");
    }

    #[test]
    fn imported_acquisition_auto_jsonl_owns_original_after_path_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("session.jsonl");
        std::fs::write(
            &source,
            concat!(
                "{\"type\":\"session_meta\",\"payload\":{\"id\":\"original\"}}\n",
                "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"prompt\"}]}}\n",
                "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}}\n",
            ),
        )
        .unwrap();
        let request = ImportedAgentSelectionRequest::new(
            source.clone(),
            None,
            RecordedAgentSourceFormat::Auto,
            None,
        )
        .unwrap();
        let selection = request.acquire().unwrap();
        std::fs::write(
            &source,
            b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"replacement\"}}\n",
        )
        .unwrap();

        let sessions = super::super::parse_imported_agent_sessions(selection.read_set()).unwrap();
        assert_eq!(selection.read_set().source, ImportedAgentSource::Codex);
        assert_eq!(sessions[0].session_id, "original");
    }

    #[test]
    fn imported_acquisition_validates_late_explicit_records() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("session.jsonl");
        std::fs::write(
            &source,
            format!(
                "{}{{not-json}}\n",
                "{\"type\":\"session_meta\",\"payload\":{}}\n".repeat(SCAN_RECORD_LIMIT)
            ),
        )
        .unwrap();

        let request = ImportedAgentSelectionRequest::new(
            source,
            None,
            RecordedAgentSourceFormat::Codex,
            None,
        )
        .unwrap();
        assert!(request.acquire().is_err());
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
