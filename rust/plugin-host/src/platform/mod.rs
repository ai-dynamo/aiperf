// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Platform-specific filesystem primitives for no-follow file access and
//! discovery-authority checks.
//!
//! [`check_directory_trust`] is the platform-neutral entry point: it dispatches
//! to the Unix owner/mode/ACL check or to the Windows module, which fails
//! closed because its security-descriptor evaluation is not implemented.

use std::path::Path;

use crate::error::AuthorityError;

#[cfg(unix)]
pub mod acl_unix;
#[cfg(windows)]
pub mod acl_windows;
pub mod fs;

/// Prove that a plugin discovery directory is under trusted authority.
#[cfg(unix)]
pub fn check_directory_trust(path: &Path) -> Result<(), AuthorityError> {
    acl_unix::check_directory_trust(path)
}

/// Prove that a plugin discovery directory is under trusted authority.
#[cfg(windows)]
pub fn check_directory_trust(path: &Path) -> Result<(), AuthorityError> {
    acl_windows::check_directory_trust(path)
}

/// Prove that a plugin discovery directory is under trusted authority.
///
/// A platform with no implemented access-control model cannot supply that
/// proof, so the check fails closed.
#[cfg(not(any(unix, windows)))]
pub fn check_directory_trust(path: &Path) -> Result<(), AuthorityError> {
    Err(AuthorityError::UnknownAclSemantics {
        path: path.to_path_buf(),
        detail: "no access-control model is implemented for this platform".to_string(),
    })
}
