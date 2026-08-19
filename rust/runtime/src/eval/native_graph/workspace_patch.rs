// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned validation and atomic application of sealed rollout workspace patches.

use std::{
    collections::BTreeSet,
    fmt::{self, Display, Formatter},
    fs,
    io::{Cursor, Read},
    path::{Component, Path, PathBuf},
};

use tar::Archive;

/// Rejection while validating a workspace patch before it reaches a task workspace.
#[derive(Debug)]
pub(crate) enum NativeGraphWorkspacePatchError {
    /// The supplied archive is larger than the sealed per-patch byte cap.
    PatchTooLarge,
    /// The archive cannot be parsed as a strict tar stream.
    Archive,
    /// An archive entry is not a regular file.
    NonRegularEntry,
    /// An archive path is absolute, empty, or contains a non-normal component.
    InvalidPath,
    /// An archive contains the same destination more than once.
    DuplicatePath,
    /// An archive attempts to change a path outside the sealed contract.
    UndeclaredPath,
    /// A destination parent or existing destination is not a regular safe filesystem object.
    UnsafeDestination,
    /// A staging or replacement operation failed.
    Io(&'static str),
}

impl Display for NativeGraphWorkspacePatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PatchTooLarge => "workspace patch exceeds its sealed byte limit",
            Self::Archive => "workspace patch is not a readable tar archive",
            Self::NonRegularEntry => "workspace patch contains a non-regular entry",
            Self::InvalidPath => "workspace patch contains an invalid path",
            Self::DuplicatePath => "workspace patch contains a duplicate path",
            Self::UndeclaredPath => "workspace patch changes an undeclared path",
            Self::UnsafeDestination => "workspace patch destination is unsafe",
            Self::Io(operation) => {
                return write!(
                    formatter,
                    "workspace patch filesystem operation failed: {operation}"
                );
            }
        })
    }
}

impl std::error::Error for NativeGraphWorkspacePatchError {}

/// Validates one bounded tar patch and atomically replaces only sealed task-relative files.
pub(crate) fn apply_workspace_patch(
    root: &Path,
    archive_bytes: &[u8],
    mutable_paths: &[&str],
    max_patch_bytes: u64,
) -> Result<(), NativeGraphWorkspacePatchError> {
    let max_patch_bytes = usize::try_from(max_patch_bytes)
        .map_err(|_| NativeGraphWorkspacePatchError::PatchTooLarge)?;
    if archive_bytes.len() > max_patch_bytes {
        return Err(NativeGraphWorkspacePatchError::PatchTooLarge);
    }
    let declared = mutable_paths
        .iter()
        .map(|path| normalized_path(Path::new(path)))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if declared.len() != mutable_paths.len() {
        return Err(NativeGraphWorkspacePatchError::DuplicatePath);
    }
    let metadata = fs::symlink_metadata(root)
        .map_err(|_| NativeGraphWorkspacePatchError::Io("workspace metadata"))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(NativeGraphWorkspacePatchError::UnsafeDestination);
    }
    let staging = tempfile::Builder::new()
        .prefix("native-graph-workspace-patch-")
        .tempdir_in(root)
        .map_err(|_| NativeGraphWorkspacePatchError::Io("create staging directory"))?;
    let mut archive = Archive::new(Cursor::new(archive_bytes));
    let mut seen = BTreeSet::new();
    let mut extracted_bytes = 0_u64;
    for entry in archive
        .entries()
        .map_err(|_| NativeGraphWorkspacePatchError::Archive)?
    {
        let entry = entry.map_err(|_| NativeGraphWorkspacePatchError::Archive)?;
        if !entry.header().entry_type().is_file() || entry.header().entry_type().is_gnu_sparse() {
            return Err(NativeGraphWorkspacePatchError::NonRegularEntry);
        }
        let entry_bytes = entry
            .header()
            .size()
            .map_err(|_| NativeGraphWorkspacePatchError::Archive)?;
        extracted_bytes = extracted_bytes
            .checked_add(entry_bytes)
            .ok_or(NativeGraphWorkspacePatchError::PatchTooLarge)?;
        if entry_bytes > max_patch_bytes as u64 || extracted_bytes > max_patch_bytes as u64 {
            return Err(NativeGraphWorkspacePatchError::PatchTooLarge);
        }
        let path = entry
            .path()
            .map_err(|_| NativeGraphWorkspacePatchError::InvalidPath)?;
        let path = normalized_path(&path)?;
        if !declared.contains(&path) {
            return Err(NativeGraphWorkspacePatchError::UndeclaredPath);
        }
        if !seen.insert(path.clone()) {
            return Err(NativeGraphWorkspacePatchError::DuplicatePath);
        }
        let destination = staging.path().join(&path);
        let parent = destination
            .parent()
            .ok_or(NativeGraphWorkspacePatchError::InvalidPath)?;
        fs::create_dir_all(parent)
            .map_err(|_| NativeGraphWorkspacePatchError::Io("create staging parent"))?;
        let mut file = fs::File::create(&destination)
            .map_err(|_| NativeGraphWorkspacePatchError::Io("create staged file"))?;
        let copied = std::io::copy(&mut entry.take(entry_bytes), &mut file)
            .map_err(|_| NativeGraphWorkspacePatchError::Archive)?;
        if copied != entry_bytes {
            return Err(NativeGraphWorkspacePatchError::Archive);
        }
        file.sync_all()
            .map_err(|_| NativeGraphWorkspacePatchError::Io("sync staged file"))?;
    }
    if seen.is_empty() {
        return Err(NativeGraphWorkspacePatchError::Archive);
    }
    for path in &seen {
        validate_destination(root, &path)?;
    }
    commit_staged_patch(root, staging.path(), seen)?;
    Ok(())
}

fn commit_staged_patch(
    root: &Path,
    staging: &Path,
    paths: BTreeSet<PathBuf>,
) -> Result<(), NativeGraphWorkspacePatchError> {
    let rollback = tempfile::Builder::new()
        .prefix("native-graph-workspace-rollback-")
        .tempdir_in(root)
        .map_err(|_| NativeGraphWorkspacePatchError::Io("create rollback directory"))?;
    let mut committed = Vec::with_capacity(paths.len());
    for path in paths {
        let destination = root.join(&path);
        let backup = rollback.path().join(&path);
        let existed = destination.exists();
        if existed {
            let parent = backup
                .parent()
                .ok_or(NativeGraphWorkspacePatchError::InvalidPath)?;
            fs::create_dir_all(parent)
                .map_err(|_| NativeGraphWorkspacePatchError::Io("create rollback parent"))?;
            if fs::rename(&destination, &backup).is_err() {
                rollback_staged_patch(root, rollback.path(), &committed);
                return Err(NativeGraphWorkspacePatchError::Io("back up destination"));
            }
        }
        if fs::rename(staging.join(&path), &destination).is_err() {
            if existed {
                let _ = fs::rename(&backup, &destination);
            }
            rollback_staged_patch(root, rollback.path(), &committed);
            return Err(NativeGraphWorkspacePatchError::Io("replace destination"));
        }
        committed.push((path, existed));
    }
    Ok(())
}

fn rollback_staged_patch(root: &Path, rollback: &Path, committed: &[(PathBuf, bool)]) {
    for (path, existed) in committed.iter().rev() {
        let destination = root.join(path);
        let _ = fs::remove_file(&destination);
        if *existed {
            let _ = fs::rename(rollback.join(path), destination);
        }
    }
}

fn normalized_path(path: &Path) -> Result<PathBuf, NativeGraphWorkspacePatchError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(NativeGraphWorkspacePatchError::InvalidPath);
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(component) => normalized.push(component),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(NativeGraphWorkspacePatchError::InvalidPath);
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(NativeGraphWorkspacePatchError::InvalidPath);
    }
    Ok(normalized)
}

fn validate_destination(root: &Path, path: &Path) -> Result<(), NativeGraphWorkspacePatchError> {
    let mut current = root.to_path_buf();
    let components = path.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(component) = component else {
            return Err(NativeGraphWorkspacePatchError::InvalidPath);
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink()
                    || (index + 1 < components.len() && !metadata.is_dir())
                    || (index + 1 == components.len() && !metadata.is_file())
                {
                    return Err(NativeGraphWorkspacePatchError::UnsafeDestination);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(NativeGraphWorkspacePatchError::Io("inspect destination")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tar::{Builder, EntryType, Header};

    use super::apply_workspace_patch;

    #[test]
    fn applies_only_declared_regular_files() {
        let root = tempfile::tempdir().expect("workspace root is created");
        let patch = archive([("result.txt", b"south\n".as_slice())]);

        apply_workspace_patch(root.path(), &patch, &["result.txt"], 4_096)
            .expect("declared regular patch applies");

        assert_eq!(
            fs::read(root.path().join("result.txt")).expect("result exists"),
            b"south\n"
        );
    }

    #[test]
    fn rejects_unsafe_or_undeclared_entries_without_writing() {
        let root = tempfile::tempdir().expect("workspace root is created");
        fs::write(root.path().join("result.txt"), b"original\n").expect("seed result");
        let patch = parent_path_archive("../result.txt", b"north\n");

        assert!(apply_workspace_patch(root.path(), &patch, &["result.txt"], 4_096).is_err());
        assert_eq!(
            fs::read(root.path().join("result.txt")).expect("seed remains"),
            b"original\n"
        );
    }

    #[test]
    fn rejects_oversized_duplicate_or_undeclared_patches_without_writing() {
        let root = tempfile::tempdir().expect("workspace root is created");
        fs::write(root.path().join("result.txt"), b"original\n").expect("seed result");

        let oversized = archive([("result.txt", b"south\n".as_slice())]);
        assert!(
            apply_workspace_patch(
                root.path(),
                &oversized,
                &["result.txt"],
                u64::try_from(oversized.len() - 1).expect("archive length fits"),
            )
            .is_err()
        );
        assert!(
            apply_workspace_patch(
                root.path(),
                &archive([
                    ("result.txt", b"north\n".as_slice()),
                    ("result.txt", b"south\n".as_slice()),
                ]),
                &["result.txt"],
                4_096,
            )
            .is_err()
        );
        assert!(
            apply_workspace_patch(
                root.path(),
                &archive([("other.txt", b"north\n".as_slice())]),
                &["result.txt"],
                4_096,
            )
            .is_err()
        );
        assert_eq!(
            fs::read(root.path().join("result.txt")).expect("seed remains"),
            b"original\n"
        );
    }

    #[test]
    fn rejects_a_declared_oversized_entry_before_staging_its_contents() {
        let root = tempfile::tempdir().expect("workspace root is created");
        fs::write(root.path().join("result.txt"), b"original\n").expect("seed result");

        let patch = declared_size_archive("result.txt", 4_097);
        assert!(matches!(
            apply_workspace_patch(root.path(), &patch, &["result.txt"], 4_096),
            Err(super::NativeGraphWorkspacePatchError::PatchTooLarge)
        ));
        assert_eq!(
            fs::read(root.path().join("result.txt")).expect("seed remains"),
            b"original\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_multi_file_patch_without_partially_replacing_prior_files() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("workspace root is created");
        fs::write(root.path().join("first.txt"), b"first-original\n").expect("seed first result");
        fs::write(root.path().join("outside.txt"), b"outside\n").expect("seed outside");
        symlink("outside.txt", root.path().join("second.txt"))
            .expect("create unsafe second destination");

        let patch = archive([
            ("first.txt", b"first-patched\n".as_slice()),
            ("second.txt", b"second-patched\n".as_slice()),
        ]);
        assert!(
            apply_workspace_patch(root.path(), &patch, &["first.txt", "second.txt"], 4_096,)
                .is_err()
        );
        assert_eq!(
            fs::read(root.path().join("first.txt")).expect("first remains"),
            b"first-original\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_archive_and_destination_symlinks_without_writing() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("workspace root is created");
        fs::write(root.path().join("outside.txt"), b"outside\n").expect("seed outside");
        symlink("outside.txt", root.path().join("result.txt")).expect("create destination link");
        assert!(
            apply_workspace_patch(
                root.path(),
                &archive([("result.txt", b"north\n".as_slice())]),
                &["result.txt"],
                4_096,
            )
            .is_err()
        );
        assert!(
            apply_workspace_patch(
                root.path(),
                &symlink_archive("result.txt", "outside.txt"),
                &["result.txt"],
                4_096,
            )
            .is_err()
        );
        assert_eq!(
            fs::read(root.path().join("outside.txt")).expect("outside remains"),
            b"outside\n"
        );
    }

    fn archive<const N: usize>(entries: [(&str, &[u8]); N]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut builder = Builder::new(&mut bytes);
            for (path, contents) in entries {
                let mut header = Header::new_gnu();
                header.set_size(u64::try_from(contents.len()).expect("test length fits"));
                header.set_mode(0o600);
                header.set_cksum();
                builder
                    .append_data(&mut header, path, contents)
                    .expect("test archive entry writes");
            }
            builder.finish().expect("test archive finishes");
        }
        bytes
    }

    #[cfg(unix)]
    fn symlink_archive(path: &str, target: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut builder = Builder::new(&mut bytes);
            let mut header = Header::new_gnu();
            header.set_entry_type(EntryType::Symlink);
            header.set_size(0);
            header.set_mode(0o600);
            header
                .set_link_name(target)
                .expect("test link target is valid");
            header.set_cksum();
            builder
                .append_data(&mut header, path, std::io::empty())
                .expect("test symlink entry writes");
            builder.finish().expect("test archive finishes");
        }
        bytes
    }

    fn parent_path_archive(path: &str, contents: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut builder = Builder::new(&mut bytes);
            let mut header = Header::new_gnu();
            header.set_size(u64::try_from(contents.len()).expect("test length fits"));
            header.set_mode(0o600);
            let path_bytes = path.as_bytes();
            header.as_mut_bytes()[..path_bytes.len()].copy_from_slice(path_bytes);
            header.set_cksum();
            builder
                .append(&header, contents)
                .expect("test archive entry writes");
            builder.finish().expect("test archive finishes");
        }
        bytes
    }

    fn declared_size_archive(path: &str, size: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut header = Header::new_gnu();
        header.set_size(size);
        header.set_mode(0o600);
        header.set_path(path).expect("test path is valid");
        header.set_cksum();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0; 1_024]);
        bytes
    }
}
