// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Safe collection and transfer of explicitly declared benchmark artifacts.

use std::{
    collections::BTreeSet,
    fs,
    io::{Cursor, Read},
    path::{Component, Path, PathBuf},
};

use tar::{Archive, EntryType};

use crate::eval::ArtifactDigest;

use super::{ArtifactSpec, DockerRuntime, EvalExecutionError};

/// Collects exactly the declared container files into a private host directory.
pub fn collect_artifacts(
    runtime: &dyn DockerRuntime,
    container: &str,
    artifacts: &[ArtifactSpec],
    destination: &Path,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    fs::create_dir_all(destination).map_err(artifact_error)?;
    let mut collected = Vec::new();
    let mut destinations = BTreeSet::new();
    for artifact in artifacts {
        let archive = runtime.copy_archive(container, artifact.source())?;
        collect_archive(artifact, archive, &mut destinations, &mut collected)?;
    }
    for (relative, bytes) in &collected {
        write_artifact(destination, relative, bytes)?;
    }
    let mut digests = collected
        .into_iter()
        .map(|(relative, bytes)| (relative, ArtifactDigest::from_bytes(&bytes)))
        .collect::<Vec<_>>();
    digests.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(digests)
}

/// Copies a verified collection directory into an isolated verifier directory.
pub fn transfer_artifacts(
    source: &Path,
    destination: &Path,
    collected: &[(String, ArtifactDigest)],
) -> Result<(), EvalExecutionError> {
    fs::create_dir_all(destination).map_err(artifact_error)?;
    for (relative, digest) in collected {
        let relative = relative_path(relative)?;
        let source_path = safe_child(source, &relative)?;
        let metadata = fs::symlink_metadata(&source_path).map_err(artifact_error)?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "collected artifact is not a regular file: {}",
                relative.display()
            )));
        }
        let bytes = fs::read(&source_path).map_err(artifact_error)?;
        if ArtifactDigest::from_bytes(&bytes) != *digest {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "collected artifact digest changed: {}",
                relative.display()
            )));
        }
        write_artifact(destination, relative.to_string_lossy().as_ref(), &bytes)?;
    }
    Ok(())
}

fn collect_archive(
    artifact: &ArtifactSpec,
    archive: Vec<u8>,
    destinations: &mut BTreeSet<String>,
    collected: &mut Vec<(String, Vec<u8>)>,
) -> Result<(), EvalExecutionError> {
    let source_name = Path::new(artifact.source())
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            EvalExecutionError::ArtifactCollection("invalid artifact source".to_owned())
        })?;
    let destination_root = artifact.destination().unwrap_or(source_name);
    let destination_root = relative_path(destination_root)?;
    let mut archive = Archive::new(Cursor::new(archive));
    for entry in archive.entries().map_err(artifact_error)? {
        let mut entry = entry.map_err(artifact_error)?;
        let entry_type = entry.header().entry_type();
        if is_rejected_entry_type(entry_type) {
            return Err(EvalExecutionError::ArtifactCollection(
                "archive contains a link or special file".to_owned(),
            ));
        }
        if !entry_type.is_file() && !entry_type.is_dir() {
            return Err(EvalExecutionError::ArtifactCollection(
                "archive contains an unsupported entry".to_owned(),
            ));
        }
        let path = entry.path().map_err(artifact_error)?;
        let path = archive_relative(&path, source_name, artifact.is_exact_file())?;
        if path.as_os_str().is_empty() {
            if entry_type.is_dir() {
                continue;
            }
            return Err(EvalExecutionError::ArtifactCollection(
                "artifact archive has an empty file path".to_owned(),
            ));
        }
        if entry_type.is_dir() {
            continue;
        }
        let relative_source = path.to_string_lossy();
        if artifact
            .exclude()
            .iter()
            .any(|pattern| glob_matches(pattern, &relative_source))
        {
            continue;
        }
        let relative = if artifact.is_exact_file() {
            destination_root.clone()
        } else {
            destination_root.join(path)
        };
        let relative = relative_path(relative.to_string_lossy().as_ref())?;
        let key = relative.to_string_lossy().into_owned();
        if !destinations.insert(key.clone()) {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "duplicate artifact destination: {key}"
            )));
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(artifact_error)?;
        collected.push((key, bytes));
    }
    Ok(())
}

fn is_rejected_entry_type(entry_type: EntryType) -> bool {
    entry_type.is_symlink()
        || entry_type.is_hard_link()
        || entry_type.is_block_special()
        || entry_type.is_character_special()
        || entry_type.is_fifo()
}

fn archive_relative(
    path: &Path,
    source_name: &str,
    is_exact_file: bool,
) -> Result<PathBuf, EvalExecutionError> {
    let path = relative_path(path.to_string_lossy().as_ref())?;
    let mut components = path.components();
    let first = components.next();
    let remaining = components.as_path();
    let has_source_root =
        first.and_then(|component| component.as_os_str().to_str()) == Some(source_name);
    if is_exact_file {
        if !has_source_root || !remaining.as_os_str().is_empty() {
            return Err(EvalExecutionError::ArtifactCollection(
                "exact artifact archive must contain its requested regular file".to_owned(),
            ));
        }
        return Ok(PathBuf::from(source_name));
    }
    Ok(if has_source_root {
        remaining.to_path_buf()
    } else {
        path
    })
}

fn relative_path(path: &str) -> Result<PathBuf, EvalExecutionError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(EvalExecutionError::ArtifactCollection(
            "artifact destination must be a nonempty relative path".to_owned(),
        ));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(EvalExecutionError::ArtifactCollection(
            "artifact archive path escapes its collection root".to_owned(),
        ));
    }
    Ok(path.to_path_buf())
}

fn safe_child(root: &Path, relative: &Path) -> Result<PathBuf, EvalExecutionError> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        current.push(component);
        if let Ok(metadata) = fs::symlink_metadata(&current) {
            if metadata.file_type().is_symlink() {
                return Err(EvalExecutionError::ArtifactCollection(format!(
                    "artifact path contains a symlink: {}",
                    current.display()
                )));
            }
        }
    }
    Ok(current)
}

fn write_artifact(root: &Path, relative: &str, bytes: &[u8]) -> Result<(), EvalExecutionError> {
    let relative = relative_path(relative)?;
    let path = safe_child(root, &relative)?;
    let parent = path.parent().ok_or_else(|| {
        EvalExecutionError::ArtifactCollection("artifact destination lacks a parent".to_owned())
    })?;
    fs::create_dir_all(parent).map_err(artifact_error)?;
    let path = safe_child(root, &relative)?;
    if path.exists() {
        return Err(EvalExecutionError::ArtifactCollection(format!(
            "artifact destination already exists: {}",
            relative.display()
        )));
    }
    fs::write(path, bytes).map_err(artifact_error)
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    glob_matches_bytes(pattern.as_bytes(), path.as_bytes())
}

fn glob_matches_bytes(pattern: &[u8], path: &[u8]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&b'*', remainder)) => {
            let recursive = remainder.first() == Some(&b'*');
            let remainder = if recursive {
                &remainder[1..]
            } else {
                remainder
            };
            (0..=path.len()).any(|index| {
                (recursive || !path[..index].contains(&b'/'))
                    && glob_matches_bytes(remainder, &path[index..])
            })
        }
        Some((&byte, remainder)) => {
            path.first() == Some(&byte) && glob_matches_bytes(remainder, &path[1..])
        }
    }
}

fn artifact_error(error: std::io::Error) -> EvalExecutionError {
    EvalExecutionError::ArtifactCollection(error.to_string())
}
