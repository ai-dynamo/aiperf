// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Private bootstrap-bundle creation for native Kubernetes roles.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::contract::{BootstrapReference, NativeK8sRole};
use super::error::KubeError;

/// Create one no-follow, create-new `0600` bootstrap bundle for a role.
pub fn create_bundle(
    directory: &Path,
    secret_name: String,
    role: NativeK8sRole,
    mount_path: String,
    contents: &[u8],
) -> Result<(PathBuf, BootstrapReference), KubeError> {
    std::fs::create_dir_all(directory).map_err(KubeError::Io)?;
    let filename = format!("{secret_name}.bootstrap");
    let path = directory.join(filename);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .map_err(KubeError::Io)?;
    file.write_all(contents).map_err(KubeError::Io)?;
    file.sync_all().map_err(KubeError::Io)?;
    let reference = BootstrapReference {
        secret_name,
        role,
        mount_path,
        sha256: format!("{:x}", Sha256::digest(contents)),
    };
    Ok((path, reference))
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    #[test]
    fn bundle_is_create_new_and_private() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let (path, reference) = create_bundle(
            directory.path(),
            "controller-bootstrap".to_string(),
            NativeK8sRole::Controller,
            "/run/aiperf/bootstrap".to_string(),
            b"private material",
        )
        .expect("bundle");
        assert_eq!(std::fs::metadata(path).expect("metadata").permissions().mode() & 0o777, 0o600);
        assert_eq!(reference.sha256, format!("{:x}", Sha256::digest(b"private material")));
        assert!(create_bundle(
            directory.path(),
            "controller-bootstrap".to_string(),
            NativeK8sRole::Controller,
            "/run/aiperf/bootstrap".to_string(),
            b"private material",
        )
        .is_err());
    }
}
