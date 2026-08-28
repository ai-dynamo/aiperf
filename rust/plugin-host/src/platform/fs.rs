// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No-follow filesystem primitives.
//!
//! `open_no_follow` opens a file using `O_NOFOLLOW` on Unix, refusing to open
//! a symlink at the final path component.  On Windows it opens with
//! `FILE_FLAG_OPEN_REPARSE_POINT` so the reparse point itself is opened rather
//! than its target.  `create_no_follow` applies the same refusal to the write
//! side, so a planted link cannot redirect a create or a truncate.

use std::fs::File;
use std::path::Path;

/// Open a regular file without following a symlink at its final path component.
///
/// Returns `Err` with `ErrorKind::InvalidInput` if the path is a symlink.
pub fn open_no_follow(path: &Path) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW causes open() to fail with ELOOP if the final component is
        // a symbolic link.
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        // Open the reparse point itself, not its target.
        let f = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)?;
        // Reject if this handle actually is a reparse point (symlink/junction).
        let meta = f.metadata()?;
        if meta.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path is a reparse point (symlink or junction)",
            ));
        }
        Ok(f)
    }
    #[cfg(not(any(unix, windows)))]
    {
        // Fallback: check for symlink manually before opening.
        if is_symlink(path)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path is a symlink",
            ));
        }
        File::open(path)
    }
}

/// Create or truncate a file for writing without following a symlink at its
/// final path component.
///
/// Returns `Err` when the final component already exists as a symlink, so a
/// planted link can never redirect a write outside the intended directory.
pub fn create_no_follow(path: &Path) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW makes open() fail with ELOOP when the final component is a
        // symbolic link, including a dangling one that O_CREAT would otherwise
        // resolve and create through.
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
    }
    #[cfg(not(unix))]
    {
        // Without O_NOFOLLOW the check and the open cannot be one operation, so
        // refuse an existing reparse point before creating or truncating.
        match path.symlink_metadata() {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "path is a symlink",
                ));
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
    }
}

/// Return `true` if `path` itself is a symbolic link (without following it).
pub fn is_symlink(path: &Path) -> std::io::Result<bool> {
    Ok(path.symlink_metadata()?.file_type().is_symlink())
}
