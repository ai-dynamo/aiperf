// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-time acquisition of a dataset-wide verbatim system prompt.

use std::fmt::{self, Display, Formatter};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Failure to select or acquire one verbatim system-prompt source.
#[derive(Debug)]
pub(crate) struct SystemPromptError {
    message: String,
}

impl Display for SystemPromptError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SystemPromptError {}

/// Resolve one inline or file-backed prompt into owned exact text.
pub(crate) fn resolve_system_prompt(
    inline: Option<&str>,
    file: Option<&Path>,
) -> Result<Option<String>, SystemPromptError> {
    if inline.is_some() && file.is_some() {
        return Err(SystemPromptError {
            message:
                "--system-prompt and --system-prompt-file are mutually exclusive; set exactly one"
                    .to_string(),
        });
    }
    if let Some(prompt) = inline {
        if prompt.trim().is_empty() {
            return Err(SystemPromptError {
                message: "--system-prompt is empty or whitespace-only; omit it to run without a system prompt"
                    .to_string(),
            });
        }
        return Ok(Some(prompt.to_string()));
    }
    file.map(read_prompt_file).transpose()
}

fn read_prompt_file(path: &Path) -> Result<String, SystemPromptError> {
    let mut source = open_prompt_file(path)?;
    let capacity = source
        .metadata()
        .ok()
        .and_then(|metadata| usize::try_from(metadata.len()).ok())
        .unwrap_or(0);
    let mut prompt = String::with_capacity(capacity);
    source
        .read_to_string(&mut prompt)
        .map_err(|error| file_error(path, error))?;
    if prompt.trim().is_empty() {
        return Err(SystemPromptError {
            message: format!(
                "--system-prompt-file is empty or whitespace-only: {}; omit it to run without a system prompt",
                path.display()
            ),
        });
    }
    Ok(prompt)
}

fn absolute_prompt_path(path: &Path) -> Result<PathBuf, SystemPromptError> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|current| current.join(path))
        .map_err(|error| file_error(path, error))
}

#[cfg(unix)]
fn open_prompt_file(path: &Path) -> Result<std::fs::File, SystemPromptError> {
    use std::ffi::{CString, OsStr};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Component;

    fn directory_flags() -> libc::c_int {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            libc::O_PATH | libc::O_DIRECTORY
        }
        #[cfg(target_vendor = "apple")]
        {
            libc::O_SEARCH | libc::O_DIRECTORY
        }
        #[cfg(all(
            unix,
            not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
        ))]
        {
            libc::O_RDONLY | libc::O_DIRECTORY
        }
    }

    fn openat(
        directory: &std::fs::File,
        name: &OsStr,
        flags: libc::c_int,
        authored_path: &Path,
    ) -> Result<std::fs::File, SystemPromptError> {
        let name = CString::new(name.as_bytes()).map_err(|_| {
            file_error(
                authored_path,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "path component contains a NUL byte",
                ),
            )
        })?;
        // SAFETY: the directory descriptor is borrowed from a live `File`, the
        // name is NUL-terminated, and `openat` does not retain either argument.
        let descriptor = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if descriptor < 0 {
            return Err(file_error(authored_path, std::io::Error::last_os_error()));
        }
        // SAFETY: successful `openat` returned one new descriptor, transferred
        // exactly once to `OwnedFd` and then `File` for RAII cleanup.
        let descriptor = unsafe { OwnedFd::from_raw_fd(descriptor) };
        Ok(std::fs::File::from(descriptor))
    }

    let absolute = absolute_prompt_path(path)?;
    let mut directory = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | directory_flags())
        .open(Path::new("/"))
        .map_err(|error| file_error(path, error))?;
    let mut components = absolute.components().peekable();
    while let Some(component) = components.next() {
        let name = match component {
            Component::RootDir => continue,
            Component::Normal(name) => name,
            Component::ParentDir => OsStr::new(".."),
            Component::CurDir => OsStr::new("."),
            Component::Prefix(_) => return Err(file_error(path, "invalid Unix path prefix")),
        };
        let is_leaf = components.peek().is_none();
        let flags = if is_leaf {
            libc::O_RDONLY | libc::O_NONBLOCK
        } else {
            directory_flags()
        };
        directory = openat(&directory, name, flags, path)?;
        if is_leaf {
            let metadata = directory
                .metadata()
                .map_err(|error| file_error(path, error))?;
            if !metadata.is_file() {
                return Err(file_error(path, "path is not a regular file"));
            }
            return Ok(directory);
        }
    }
    Err(file_error(path, "path has no file component"))
}

#[cfg(not(unix))]
fn open_prompt_file(path: &Path) -> Result<std::fs::File, SystemPromptError> {
    let absolute = absolute_prompt_path(path)?;
    for component in absolute.ancestors() {
        let metadata =
            std::fs::symlink_metadata(component).map_err(|error| file_error(path, error))?;
        if metadata.file_type().is_symlink() {
            return Err(file_error(path, "path contains a symlink component"));
        }
    }
    let canonical = absolute
        .canonicalize()
        .map_err(|error| file_error(path, error))?;
    let source = std::fs::File::open(canonical).map_err(|error| file_error(path, error))?;
    if !source
        .metadata()
        .map_err(|error| file_error(path, error))?
        .is_file()
    {
        return Err(file_error(path, "path is not a regular file"));
    }
    Ok(source)
}

fn file_error(path: &Path, cause: impl Display) -> SystemPromptError {
    SystemPromptError {
        message: format!(
            "--system-prompt-file could not be read: {}. Expected a readable regular UTF-8 text file with no symlinked path component: {cause}",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::resolve_system_prompt;

    #[test]
    fn inline_prompt_preserves_exact_text() {
        let prompt = "  first line\nsecond line\n  ";

        let resolved =
            resolve_system_prompt(Some(prompt), None).expect("valid inline prompt should resolve");

        assert_eq!(resolved.as_deref(), Some(prompt));
    }

    #[test]
    fn file_prompt_is_owned_after_one_resolution() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("system.txt");
        std::fs::write(&path, "  from file\n").expect("write initial system prompt fixture");

        let resolved =
            resolve_system_prompt(None, Some(&path)).expect("valid file prompt should resolve");
        std::fs::write(&path, "replacement").expect("replace source after resolution");

        assert_eq!(resolved.as_deref(), Some("  from file\n"));
    }

    #[test]
    fn absent_prompt_resolves_to_none() {
        assert_eq!(
            resolve_system_prompt(None, None).expect("absent prompt should be valid"),
            None
        );
    }

    #[test]
    fn source_selection_rejects_both_and_blank_values() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("system.txt");
        std::fs::write(&path, "from file").expect("write system prompt fixture");

        let both = resolve_system_prompt(Some("inline"), Some(&path))
            .expect_err("two prompt sources must conflict")
            .to_string();
        assert!(both.contains("--system-prompt"), "{both}");
        assert!(both.contains("--system-prompt-file"), "{both}");

        let blank = resolve_system_prompt(Some(" \n\t "), None)
            .expect_err("blank inline prompt must fail")
            .to_string();
        assert!(blank.contains("empty or whitespace-only"), "{blank}");
    }

    #[test]
    fn file_prompt_rejects_invalid_source_shapes() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let blank = directory.path().join("blank.txt");
        let invalid_utf8 = directory.path().join("invalid.txt");
        let missing = directory.path().join("missing.txt");
        std::fs::write(&blank, " \n\t").expect("write blank prompt fixture");
        std::fs::write(&invalid_utf8, [0xff, 0xfe]).expect("write invalid UTF-8 fixture");

        for (path, expected) in [
            (blank.as_path(), "empty or whitespace-only"),
            (invalid_utf8.as_path(), "UTF-8"),
            (missing.as_path(), "regular UTF-8"),
            (directory.path(), "regular UTF-8"),
        ] {
            let error = resolve_system_prompt(None, Some(path))
                .expect_err("invalid file prompt must fail")
                .to_string();
            assert!(error.contains(expected), "{path:?}: {error}");
            assert!(
                error.contains(&path.display().to_string()),
                "{path:?}: {error}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn file_prompt_rejects_symlink_leaf_and_parent() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary directory");
        let real_parent = directory.path().join("real");
        std::fs::create_dir(&real_parent).expect("create real parent");
        let real_file = real_parent.join("system.txt");
        std::fs::write(&real_file, "system").expect("write prompt fixture");

        let leaf_link = directory.path().join("leaf-link.txt");
        symlink(&real_file, &leaf_link).expect("create leaf symlink");
        let parent_link = directory.path().join("parent-link");
        symlink(&real_parent, &parent_link).expect("create parent symlink");

        for path in [leaf_link, parent_link.join(Path::new("system.txt"))] {
            let error = resolve_system_prompt(None, Some(&path))
                .expect_err("symlinked prompt path must fail")
                .to_string();
            assert!(error.contains("symlink"), "{path:?}: {error}");
            assert!(
                error.contains(&path.display().to_string()),
                "{path:?}: {error}"
            );
        }
    }
}
