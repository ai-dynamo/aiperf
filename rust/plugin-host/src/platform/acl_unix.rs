// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unix discovery-authority checks.
//!
//! A discovery path is trusted only when the host can prove that no user
//! outside a small trusted set can replace its bytes between inspection and
//! load.  That proof is three facts about the path itself — it is not a
//! symlink, it carries no group or world write bit, and its owner is in the
//! trusted uid set — plus, under a strict policy, a conclusive answer about
//! whether an extended ACL grants access the mode bits do not describe.
//!
//! Every inconclusive answer is a refusal.  A path whose access-control state
//! this host cannot read is reported as [`AuthorityError::UnknownAclSemantics`]
//! rather than accepted on its mode bits alone.

use std::os::unix::fs::MetadataExt;
use std::path::Path;

use crate::error::AuthorityError;

/// Bits that must be clear for a path to be considered immutable by others.
const WORLD_WRITE: u32 = 0o002;
const GROUP_WRITE: u32 = 0o020;

/// The extended-attribute name Linux uses for a POSIX access ACL.
#[cfg(target_os = "linux")]
const POSIX_ACL_ACCESS: &[u8] = b"system.posix_acl_access\0";

/// Policy governing which owners and access-control states the host accepts.
#[derive(Debug, Clone)]
pub struct AuthorityPolicy {
    /// Exhaustive set of uids permitted to own a trusted path.
    ///
    /// This is exhaustive on purpose: a privileged host that also trusted its
    /// own euid unconditionally could be steered by a path it created while
    /// dropping privileges.
    pub trusted_uids: Vec<u32>,
    /// Reject a path whose extended-ACL state cannot be determined.
    pub require_acl_probe: bool,
    /// Accept a group-writable path.  Off by default.
    pub allow_group_writable: bool,
}

impl Default for AuthorityPolicy {
    fn default() -> Self {
        Self {
            trusted_uids: vec![0, current_euid()],
            require_acl_probe: false,
            allow_group_writable: false,
        }
    }
}

/// Return the effective uid of the calling process.
pub fn current_euid() -> u32 {
    // SAFETY: `geteuid` takes no arguments, touches no caller memory, and is
    // documented as always succeeding.
    unsafe { libc::geteuid() }
}

/// Prove that `path` is under trusted authority using the default policy.
pub fn check_path_authority(path: &Path) -> Result<(), AuthorityError> {
    check_path_authority_with(path, &AuthorityPolicy::default())
}

/// Prove that `path` is under trusted authority using an explicit `policy`.
///
/// The path is inspected with `symlink_metadata`, so a symlink is reported as
/// such instead of being silently resolved to its target's ownership.
pub fn check_path_authority_with(
    path: &Path,
    policy: &AuthorityPolicy,
) -> Result<(), AuthorityError> {
    let meta = std::fs::symlink_metadata(path).map_err(|source| AuthorityError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if meta.file_type().is_symlink() {
        return Err(AuthorityError::Symlink(path.to_path_buf()));
    }

    let mode = meta.mode() & 0o7777;
    if mode & WORLD_WRITE != 0 {
        return Err(AuthorityError::WorldWritable {
            path: path.to_path_buf(),
            mode,
        });
    }
    if mode & GROUP_WRITE != 0 && !policy.allow_group_writable {
        return Err(AuthorityError::GroupWritable {
            path: path.to_path_buf(),
            mode,
        });
    }

    let uid = meta.uid();
    if !policy.trusted_uids.contains(&uid) {
        return Err(AuthorityError::UntrustedOwner {
            path: path.to_path_buf(),
            uid,
        });
    }

    if policy.require_acl_probe && has_extended_acl(path)? {
        return Err(AuthorityError::UnknownAclSemantics {
            path: path.to_path_buf(),
            detail: "an extended ACL grants access the mode bits do not describe".to_string(),
        });
    }
    Ok(())
}

/// Prove that `path` and every ancestor directory up to and including `root`
/// are under trusted authority.
///
/// A trusted file inside a world-writable directory is not trusted: the
/// directory's write bit lets any user unlink the file and put another in its
/// place, which is exactly the substitution the per-path check cannot see.
pub fn check_tree_authority(path: &Path, root: &Path) -> Result<(), AuthorityError> {
    check_tree_authority_with(path, root, &AuthorityPolicy::default())
}

/// [`check_tree_authority`] with an explicit policy.
pub fn check_tree_authority_with(
    path: &Path,
    root: &Path,
    policy: &AuthorityPolicy,
) -> Result<(), AuthorityError> {
    check_path_authority_with(path, policy)?;
    let mut cursor = path;
    while let Some(parent) = cursor.parent() {
        if parent.as_os_str().is_empty() {
            break;
        }
        check_path_authority_with(parent, policy)?;
        if parent == root {
            break;
        }
        cursor = parent;
    }
    Ok(())
}

/// Report whether `path` carries an extended POSIX access ACL.
///
/// Returns [`AuthorityError::UnknownAclSemantics`] when the platform cannot
/// answer the question, so callers under a strict policy fail closed instead of
/// trusting the mode bits alone.
#[cfg(target_os = "linux")]
pub fn has_extended_acl(path: &Path) -> Result<bool, AuthorityError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        AuthorityError::UnknownAclSemantics {
            path: path.to_path_buf(),
            detail: "path contains an interior NUL byte".to_string(),
        }
    })?;
    // `lgetxattr` does not follow a symlink at the final component, matching the
    // no-follow discipline the rest of the authority check uses.
    // SAFETY: both pointers are valid NUL-terminated C strings owned by this
    // frame, and a null value buffer with zero size is the documented way to
    // query only for presence.
    let rc = unsafe {
        libc::lgetxattr(
            c_path.as_ptr(),
            POSIX_ACL_ACCESS.as_ptr().cast::<libc::c_char>(),
            std::ptr::null_mut(),
            0,
        )
    };
    if rc >= 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        // No ACL attribute is set, or the filesystem cannot carry one at all:
        // both are conclusive negatives.
        Some(libc::ENODATA) | Some(libc::ENOTSUP) => Ok(false),
        Some(libc::ENOENT) | Some(libc::EACCES) => Err(AuthorityError::Io {
            path: path.to_path_buf(),
            source: err,
        }),
        _ => Err(AuthorityError::UnknownAclSemantics {
            path: path.to_path_buf(),
            detail: err.to_string(),
        }),
    }
}

/// Report whether `path` carries an extended access ACL.
///
/// On a Unix platform whose ACL representation this host does not implement the
/// answer is always inconclusive, which is a refusal under a strict policy.
#[cfg(not(target_os = "linux"))]
pub fn has_extended_acl(path: &Path) -> Result<bool, AuthorityError> {
    Err(AuthorityError::UnknownAclSemantics {
        path: path.to_path_buf(),
        detail: "extended ACL probing is not implemented for this platform".to_string(),
    })
}

/// Prove that a discovery directory is under trusted authority.
pub fn check_directory_trust(path: &Path) -> Result<(), AuthorityError> {
    check_path_authority(path)
}
