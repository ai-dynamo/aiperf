// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Safe collection and transfer of explicitly declared benchmark artifacts.

use std::{
    collections::BTreeSet,
    fs,
    io::{self, Read, Write},
    os::unix::fs::PermissionsExt,
    path::{Component, Path, PathBuf},
};

use tar::{Archive, EntryType};
use tempfile::NamedTempFile;

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
        collect_archive(
            artifact,
            archive,
            destination,
            &mut destinations,
            &mut collected,
        )?;
    }
    collected.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(collected)
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
        let mut source_file = fs::File::open(source_path).map_err(artifact_error)?;
        write_artifact_stream(
            destination,
            relative.to_string_lossy().as_ref(),
            &mut source_file,
            Some(digest),
        )?;
    }
    Ok(())
}

fn collect_archive(
    artifact: &ArtifactSpec,
    archive: Box<dyn Read>,
    destination: &Path,
    destinations: &mut BTreeSet<String>,
    collected: &mut Vec<(String, ArtifactDigest)>,
) -> Result<(), EvalExecutionError> {
    let source_name = Path::new(artifact.source())
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            EvalExecutionError::ArtifactCollection("invalid artifact source".to_owned())
        })?;
    let destination_root = artifact.destination().unwrap_or(source_name);
    let destination_root = relative_path(destination_root)?;
    let mut archive = Archive::new(archive);
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
        let digest = write_artifact_stream(destination, &key, &mut entry, None)?;
        collected.push((key, digest));
    }
    let mut archive = archive.into_inner();
    io::copy(&mut archive, &mut io::sink()).map_err(artifact_error)?;
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
    if !has_source_root {
        return Err(EvalExecutionError::ArtifactCollection(
            "directory artifact archive contains a member outside its declared source".to_owned(),
        ));
    }
    Ok(remaining.to_path_buf())
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

fn write_artifact_stream(
    root: &Path,
    relative: &str,
    source: &mut dyn Read,
    expected_digest: Option<&ArtifactDigest>,
) -> Result<ArtifactDigest, EvalExecutionError> {
    let relative = relative_path(relative)?;
    let parent = ensure_parent_directories(root, &relative)?;
    let path = safe_child(root, &relative)?;
    let mut temporary = NamedTempFile::new_in(parent).map_err(artifact_error)?;
    let digest = {
        let mut writer = HashingWriter::new(temporary.as_file_mut());
        io::copy(source, &mut writer).map_err(artifact_error)?;
        writer.flush().map_err(artifact_error)?;
        writer.finish()?
    };
    if expected_digest.is_some_and(|expected| expected != &digest) {
        return Err(EvalExecutionError::ArtifactCollection(format!(
            "collected artifact digest changed: {}",
            relative.display()
        )));
    }
    let file = temporary.persist_noclobber(path).map_err(|error| {
        EvalExecutionError::ArtifactCollection(format!(
            "artifact destination already exists: {} ({})",
            relative.display(),
            error.error
        ))
    })?;
    file.set_permissions(fs::Permissions::from_mode(0o644))
        .map_err(artifact_error)?;
    Ok(digest)
}

fn ensure_parent_directories(root: &Path, relative: &Path) -> Result<PathBuf, EvalExecutionError> {
    let mut parent = root.to_path_buf();
    let components = relative.components().collect::<Vec<_>>();
    for component in &components[..components.len().saturating_sub(1)] {
        parent.push(component);
        match fs::create_dir(&parent) {
            Ok(()) => fs::set_permissions(&parent, fs::Permissions::from_mode(0o755))
                .map_err(artifact_error)?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(artifact_error(error)),
        }
        let metadata = fs::symlink_metadata(&parent).map_err(artifact_error)?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "artifact parent is not a directory: {}",
                parent.display()
            )));
        }
    }
    Ok(parent)
}

struct HashingWriter<'a> {
    file: &'a mut fs::File,
    hasher: blake3::Hasher,
}

impl<'a> HashingWriter<'a> {
    fn new(file: &'a mut fs::File) -> Self {
        Self {
            file,
            hasher: blake3::Hasher::new(),
        }
    }

    fn finish(self) -> Result<ArtifactDigest, EvalExecutionError> {
        ArtifactDigest::parse(format!("blake3:{}", self.hasher.finalize().to_hex()))
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))
    }
}

impl Write for HashingWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let count = self.file.write(buffer)?;
        self.hasher.update(&buffer[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
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
