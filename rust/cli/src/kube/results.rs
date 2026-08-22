// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ready-manifest-only result downloads with bounded SHA-256 verification.
//!
//! Downloads consume the authenticated durable manifest served by the operator.
//! An artifact is written only after its bytes match the manifest length and
//! digest, so a truncated or substituted transfer never lands on disk.

use std::collections::HashSet;
#[cfg(unix)]
use std::ffi::{CString, OsStr};
#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::contract::CONTRACT_VERSION;
use super::error::KubeError;

/// Largest accepted single artifact transfer.
pub const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;

/// Producer manifest durably published through the operator API.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResultsManifest {
    /// Contract version of the producing controller.
    pub contract_version: String,
    /// Identifier of the run that produced these artifacts.
    pub run_id: String,
    /// Whether the producer committed every artifact it intends to publish.
    pub ready: bool,
    /// Whether the run terminated through cancellation.
    pub was_cancelled: bool,
    /// Producer-side root the artifact paths are relative to.
    pub artifact_root: String,
    /// Committed artifacts in producer order.
    pub artifacts: Vec<ManifestArtifact>,
}

/// One committed artifact with the exact bytes and digest the producer wrote.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ManifestArtifact {
    /// Relative artifact path.
    pub path: String,
    /// Lowercase hexadecimal SHA-256 of the committed bytes.
    pub sha256: String,
    /// Exact committed byte length.
    pub bytes: u64,
    /// Producer-declared content type.
    pub content_type: String,
}

/// Bounded artifact transfer seam. Implementations own their own deadlines.
pub trait ArtifactFetcher {
    /// Fetch the complete bytes of one manifest-declared artifact path.
    fn fetch(&self, path: &str) -> Result<Vec<u8>, KubeError>;
}

/// Decode a producer manifest and refuse anything that is not a ready v1 document.
pub fn parse_manifest(body: &[u8]) -> Result<ResultsManifest, KubeError> {
    let manifest: ResultsManifest = serde_json::from_slice(body)
        .map_err(|error| KubeError::Decode(format!("results manifest is not valid: {error}")))?;
    if manifest.contract_version != CONTRACT_VERSION {
        return Err(KubeError::UnsupportedContractVersion(
            manifest.contract_version,
        ));
    }
    if !manifest.ready {
        return Err(KubeError::ContractValidation(
            "results manifest is not ready".to_string(),
        ));
    }
    if manifest.run_id.is_empty() || manifest.artifact_root.is_empty() {
        return Err(KubeError::ContractValidation(
            "results manifest omits its run identity".to_string(),
        ));
    }
    let mut seen = HashSet::new();
    for artifact in &manifest.artifacts {
        if safe_relative(&artifact.path).is_none() {
            return Err(KubeError::ContractValidation(format!(
                "results manifest declares unsafe artifact path {}",
                artifact.path
            )));
        }
        if !seen.insert(artifact.path.as_str()) {
            return Err(KubeError::ContractValidation(format!(
                "results manifest repeats artifact path {}",
                artifact.path
            )));
        }
        if artifact.sha256.len() != 64
            || !artifact
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(KubeError::ContractValidation(format!(
                "results manifest declares a non-canonical digest for {}",
                artifact.path
            )));
        }
        if artifact.bytes > MAX_ARTIFACT_BYTES {
            return Err(KubeError::ContractValidation(format!(
                "artifact {} exceeds {MAX_ARTIFACT_BYTES} bytes",
                artifact.path
            )));
        }
    }
    Ok(manifest)
}

/// Download every manifest artifact into `destination`, verifying each transfer.
pub fn download(
    manifest: &ResultsManifest,
    fetcher: &dyn ArtifactFetcher,
    destination: &Path,
) -> Result<Vec<PathBuf>, KubeError> {
    #[cfg(not(unix))]
    {
        let _ = (manifest, fetcher, destination);
        return Err(KubeError::ContractValidation(
            "secure result downloads require POSIX no-follow descriptors".to_string(),
        ));
    }
    #[cfg(unix)]
    let root = open_or_create_directory_path(destination)?;
    let mut written = Vec::with_capacity(manifest.artifacts.len());
    for artifact in &manifest.artifacts {
        let bytes = fetcher.fetch(&artifact.path)?;
        verify(artifact, &bytes)?;
        let relative = safe_relative(&artifact.path).ok_or_else(|| {
            KubeError::ContractValidation(format!("unsafe artifact path {}", artifact.path))
        })?;
        #[cfg(unix)]
        write_beneath(&root, &relative, &bytes)?;
        let target = destination.join(&relative);
        written.push(target);
    }
    Ok(written)
}

#[cfg(unix)]
fn component_name(name: &OsStr) -> std::io::Result<CString> {
    CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "result path component contains NUL",
        )
    })
}

#[cfg(unix)]
fn open_directory_at(parent: &File, name: &OsStr, create: bool) -> std::io::Result<File> {
    let name = component_name(name)?;
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW;
    let mut descriptor = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags) };
    if descriptor < 0
        && create
        && std::io::Error::last_os_error().kind() == std::io::ErrorKind::NotFound
    {
        if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(error);
            }
        } else {
            parent.sync_all()?;
        }
        descriptor = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags) };
    }
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let directory = unsafe { File::from_raw_fd(descriptor) };
    if !directory.metadata()?.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "result path component is not a directory",
        ));
    }
    directory.sync_all()?;
    Ok(directory)
}

#[cfg(unix)]
fn open_or_create_directory_path(path: &Path) -> std::io::Result<File> {
    let mut directory = File::open(if path.is_absolute() {
        Path::new("/")
    } else {
        Path::new(".")
    })?;
    for component in path.components() {
        match component {
            Component::RootDir | Component::CurDir => continue,
            Component::Normal(name) => directory = open_directory_at(&directory, name, true)?,
            Component::ParentDir | Component::Prefix(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "result destination is not canonical",
                ));
            }
        }
    }
    Ok(directory)
}

#[cfg(unix)]
fn open_regular_at(parent: &File, name: &OsStr) -> std::io::Result<File> {
    let name = component_name(name)?;
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let file = unsafe { File::from_raw_fd(descriptor) };
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "result destination is not a regular file",
        ));
    }
    Ok(file)
}

#[cfg(unix)]
fn write_beneath(root: &File, relative: &Path, body: &[u8]) -> std::io::Result<()> {
    let mut components = relative.components().peekable();
    let mut directory = root.try_clone()?;
    let leaf = loop {
        let Some(component) = components.next() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "result artifact path is empty",
            ));
        };
        let Component::Normal(name) = component else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "result artifact path is not canonical",
            ));
        };
        if components.peek().is_none() {
            break name;
        }
        directory = open_directory_at(&directory, name, true)?;
    };
    let temporary = format!(".aiperf-download-{}", uuid::Uuid::new_v4());
    let temporary_name = component_name(OsStr::new(&temporary))?;
    let descriptor = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            temporary_name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut temporary_file = unsafe { File::from_raw_fd(descriptor) };
    let result = (|| {
        temporary_file.write_all(body)?;
        temporary_file.sync_all()?;
        let leaf_name = component_name(leaf)?;
        if unsafe {
            libc::linkat(
                directory.as_raw_fd(),
                temporary_name.as_ptr(),
                directory.as_raw_fd(),
                leaf_name.as_ptr(),
                0,
            )
        } != 0
        {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(error);
            }
            let mut existing = open_regular_at(&directory, leaf)?;
            let mut existing_bytes = Vec::new();
            std::io::Read::by_ref(&mut existing)
                .take(MAX_ARTIFACT_BYTES + 1)
                .read_to_end(&mut existing_bytes)?;
            if existing_bytes != body {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "result destination already contains different bytes",
                ));
            }
        }
        directory.sync_all()
    })();
    let unlink_result =
        unsafe { libc::unlinkat(directory.as_raw_fd(), temporary_name.as_ptr(), 0) };
    if unlink_result == 0 {
        directory.sync_all()?;
    }
    result
}

/// Confirm transferred bytes match the producer's declared length and digest.
pub fn verify(artifact: &ManifestArtifact, bytes: &[u8]) -> Result<(), KubeError> {
    if bytes.len() as u64 != artifact.bytes {
        return Err(KubeError::ContractValidation(format!(
            "artifact {} transferred {} bytes but the manifest declares {}",
            artifact.path,
            bytes.len(),
            artifact.bytes
        )));
    }
    let digest = Sha256::digest(bytes);
    let digest = digest
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            use std::fmt::Write;
            // Writing into a String cannot fail; the result is discarded deliberately.
            let _ = write!(hex, "{byte:02x}");
            hex
        });
    if digest != artifact.sha256 {
        return Err(KubeError::ContractValidation(format!(
            "artifact {} digest {digest} does not match the manifest",
            artifact.path
        )));
    }
    Ok(())
}

fn safe_relative(path: &str) -> Option<PathBuf> {
    let candidate = Path::new(path);
    if path.is_empty() || candidate.is_absolute() {
        return None;
    }
    let mut relative = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            _ => return None,
        }
    }
    (!relative.as_os_str().is_empty()).then_some(relative)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticFetcher(Vec<u8>);
    impl ArtifactFetcher for StaticFetcher {
        fn fetch(&self, _path: &str) -> Result<Vec<u8>, KubeError> {
            Ok(self.0.clone())
        }
    }

    fn manifest_json(digest: &str, bytes: u64) -> Vec<u8> {
        format!(
            r#"{{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{{"path":"profile.json","sha256":"{digest}","bytes":{bytes},"contentType":"application/json"}}]}}"#
        )
        .into_bytes()
    }

    #[test]
    fn unready_and_traversing_manifests_are_refused() {
        let unready = br#"{"contractVersion":"native-k8s/v1","runId":"run-1","ready":false,"wasCancelled":false,"artifactRoot":"/results","artifacts":[]}"#;
        assert!(parse_manifest(unready).is_err());
        let traversal = br#"{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{"path":"../escape.json","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":1,"contentType":"application/json"}]}"#;
        assert!(parse_manifest(traversal).is_err());
    }

    #[test]
    fn download_writes_only_digest_matched_artifacts() {
        let payload = b"{\"ok\":true}".to_vec();
        let digest = format!("{:x}", Sha256::digest(&payload));
        let manifest =
            parse_manifest(&manifest_json(&digest, payload.len() as u64)).expect("manifest");
        let destination = tempfile::tempdir().expect("tempdir");
        let written = download(
            &manifest,
            &StaticFetcher(payload.clone()),
            destination.path(),
        )
        .expect("download");
        assert_eq!(written.len(), 1);
        assert_eq!(std::fs::read(&written[0]).expect("read"), payload);
    }

    #[test]
    fn a_substituted_transfer_never_lands_on_disk() {
        let payload = b"{\"ok\":true}".to_vec();
        let digest = format!("{:x}", Sha256::digest(&payload));
        let manifest =
            parse_manifest(&manifest_json(&digest, payload.len() as u64)).expect("manifest");
        let destination = tempfile::tempdir().expect("tempdir");
        let error = download(
            &manifest,
            &StaticFetcher(b"{\"ok\":fals}".to_vec()),
            destination.path(),
        )
        .expect_err("digest mismatch must fail");
        assert!(matches!(error, KubeError::ContractValidation(_)));
        assert!(!destination.path().join("profile.json").exists());
    }

    #[cfg(unix)]
    #[test]
    fn download_refuses_an_ancestor_symlink_without_writing_outside_root() {
        use std::os::unix::fs::symlink;

        let payload = b"private service-account token".to_vec();
        let digest = format!("{:x}", Sha256::digest(&payload));
        let manifest = parse_manifest(
            format!(
                r#"{{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{{"path":"nested/token","sha256":"{digest}","bytes":{},"contentType":"application/octet-stream"}}]}}"#,
                payload.len()
            )
            .as_bytes(),
        )
        .expect("manifest");
        let destination = tempfile::tempdir().expect("destination");
        let outside = tempfile::tempdir().expect("outside");
        symlink(outside.path(), destination.path().join("nested")).expect("symlink");

        download(&manifest, &StaticFetcher(payload), destination.path())
            .expect_err("an ancestor symlink must fail closed");
        assert!(!outside.path().join("token").exists());
    }

    #[cfg(unix)]
    #[test]
    fn download_refuses_a_leaf_symlink_without_overwriting_its_target() {
        use std::os::unix::fs::symlink;

        let payload = b"new bytes".to_vec();
        let digest = format!("{:x}", Sha256::digest(&payload));
        let manifest =
            parse_manifest(&manifest_json(&digest, payload.len() as u64)).expect("manifest");
        let destination = tempfile::tempdir().expect("destination");
        let outside = tempfile::NamedTempFile::new().expect("outside");
        std::fs::write(outside.path(), b"keep me").expect("seed outside");
        symlink(outside.path(), destination.path().join("profile.json")).expect("symlink");

        download(&manifest, &StaticFetcher(payload), destination.path())
            .expect_err("a leaf symlink must fail closed");
        assert_eq!(std::fs::read(outside.path()).expect("outside"), b"keep me");
    }
}
