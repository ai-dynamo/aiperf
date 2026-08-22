// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ready-manifest-only result downloads with bounded SHA-256 verification.
//!
//! Downloads consume the producer manifest served by `aiperf results-sidecar`.
//! An artifact is written only after its bytes match the manifest length and
//! digest, so a truncated or substituted transfer never lands on disk.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::contract::CONTRACT_VERSION;
use super::error::KubeError;

/// Largest accepted single artifact transfer.
pub const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;

/// Producer manifest published by the controller pod's results sidecar.
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
    let mut written = Vec::with_capacity(manifest.artifacts.len());
    for artifact in &manifest.artifacts {
        let bytes = fetcher.fetch(&artifact.path)?;
        verify(artifact, &bytes)?;
        let relative = safe_relative(&artifact.path).ok_or_else(|| {
            KubeError::ContractValidation(format!("unsafe artifact path {}", artifact.path))
        })?;
        let target = destination.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &bytes)?;
        written.push(target);
    }
    Ok(written)
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
}
