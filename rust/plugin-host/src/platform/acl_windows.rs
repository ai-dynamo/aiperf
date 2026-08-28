// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Windows discovery-authority checks.
//!
//! Proving that a Windows path is under trusted authority requires reading its
//! security descriptor and evaluating the DACL against the process token, which
//! this host does not yet implement.  Rather than approximate that with a
//! partial check, every query fails closed with
//! [`AuthorityError::UnknownAclSemantics`]: a path whose access-control state
//! the host cannot evaluate is not a trusted path.

use std::path::Path;

use crate::error::AuthorityError;

/// Refuse a path because Windows ACL evaluation is not implemented.
fn unimplemented(path: &Path) -> AuthorityError {
    AuthorityError::UnknownAclSemantics {
        path: path.to_path_buf(),
        detail: "Windows security-descriptor evaluation is not implemented".to_string(),
    }
}

/// Prove that `path` is under trusted authority.  Always refuses.
pub fn check_path_authority(path: &Path) -> Result<(), AuthorityError> {
    Err(unimplemented(path))
}

/// Prove that `path` and its ancestors up to `root` are under trusted
/// authority.  Always refuses.
pub fn check_tree_authority(path: &Path, _root: &Path) -> Result<(), AuthorityError> {
    Err(unimplemented(path))
}

/// Report whether `path` carries an extended ACL.  Always inconclusive.
pub fn has_extended_acl(path: &Path) -> Result<bool, AuthorityError> {
    Err(unimplemented(path))
}

/// Prove that a discovery directory is under trusted authority.  Always
/// refuses.
pub fn check_directory_trust(path: &Path) -> Result<(), AuthorityError> {
    Err(unimplemented(path))
}
