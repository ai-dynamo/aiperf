// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native envelope loading and Kubernetes submission projections.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::client::{AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION, KubeClient};
use super::contract::{ControllerEnvelope, NativeK8sRole, validate_envelope};
use super::error::KubeError;
use super::manifest;

/// Read a strict native Kubernetes envelope from a user-owned file.
pub fn load_envelope(path: &Path) -> anyhow::Result<ControllerEnvelope> {
    let source = std::fs::read(path).map_err(|error| {
        anyhow::anyhow!(
            "failed to read native Kubernetes envelope {}: {error}",
            path.display()
        )
    })?;
    let value: Value = serde_json::from_slice(&source).map_err(|error| {
        anyhow::anyhow!(
            "failed to decode native Kubernetes envelope {}: {error}",
            path.display()
        )
    })?;
    validate_envelope(value).map_err(anyhow::Error::from)
}

/// Return the API path for a namespace-owned AIPerfJob collection.
pub fn jobs_path(namespace: &str) -> String {
    format!("/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/{AIPERF_PLURAL}")
}

/// Bootstrap material path selected on the command line for one role.
pub fn material_paths(args: &[String]) -> Result<BTreeMap<NativeK8sRole, PathBuf>, KubeError> {
    let mut selected = BTreeMap::new();
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        let value = if let Some(value) = argument.strip_prefix("--bootstrap-material=") {
            Some(value.to_string())
        } else if argument == "--bootstrap-material" {
            Some(
                arguments
                    .next()
                    .ok_or_else(|| {
                        KubeError::Decode(
                            "--bootstrap-material requires <role>=<path>".to_string(),
                        )
                    })?
                    .clone(),
            )
        } else {
            None
        };
        let Some(value) = value else { continue };
        let (role, path) = value.split_once('=').ok_or_else(|| {
            KubeError::Decode(format!("--bootstrap-material {value} is not <role>=<path>"))
        })?;
        let role = match role {
            "controller" => NativeK8sRole::Controller,
            "cell" => NativeK8sRole::Cell,
            "results-sidecar" => NativeK8sRole::ResultsSidecar,
            other => {
                return Err(KubeError::Decode(format!(
                    "--bootstrap-material names unknown role {other}"
                )));
            }
        };
        if selected.insert(role, PathBuf::from(path)).is_some() {
            return Err(KubeError::Decode(format!(
                "--bootstrap-material repeats role {role:?}"
            )));
        }
    }
    Ok(selected)
}

/// Create one immutable Secret per role after proving the envelope digest.
///
/// Material never leaves this call as plaintext in a CR, JobSet, or log; the
/// envelope keeps only the reference metadata the operator is allowed to see.
pub fn create_bootstrap_secrets(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    material: &BTreeMap<NativeK8sRole, PathBuf>,
) -> anyhow::Result<usize> {
    let mut created = 0;
    for role in &envelope.roles {
        let Some(path) = material.get(&role.name) else {
            continue;
        };
        let bytes = std::fs::read(path).map_err(|error| {
            anyhow::anyhow!("failed to read bootstrap material {}: {error}", path.display())
        })?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if digest != role.bootstrap.sha256 {
            anyhow::bail!(
                "bootstrap material {} does not match the envelope digest for {:?}",
                path.display(),
                role.name
            );
        }
        let body = json!({
            "apiVersion": "v1",
            "kind": "Secret",
            "type": "Opaque",
            "immutable": true,
            "metadata": {
                "name": role.bootstrap.secret_name,
                "namespace": envelope.namespace,
                "labels": {"aiperf.nvidia.com/run-id": envelope.run_id},
            },
            "data": {"bootstrap": BASE64.encode(&bytes)},
        });
        let body = serde_json::to_vec(&body)
            .map_err(|error| anyhow::anyhow!("failed to encode bootstrap Secret: {error}"))?;
        let status = client.request(
            "POST",
            &format!("/api/v1/namespaces/{}/secrets", envelope.namespace),
            "application/json",
            body,
        )?;
        if !(200..300).contains(&status) && status != 409 {
            anyhow::bail!(
                "bootstrap Secret {} creation returned HTTP {status}",
                role.bootstrap.secret_name
            );
        }
        created += 1;
    }
    Ok(created)
}

/// Submit one immutable envelope projection as an AIPerfJob.
pub fn submit_profile(client: &KubeClient, envelope: &ControllerEnvelope) -> anyhow::Result<u16> {
    let body = manifest::project(envelope).map_err(anyhow::Error::from)?;
    let body = serde_json::to_vec(&body).map_err(|error| {
        anyhow::anyhow!("failed to serialize native Kubernetes profile: {error}")
    })?;
    client
        .request(
            "POST",
            &jobs_path(&envelope.namespace),
            "application/json",
            body,
        )
        .map_err(anyhow::Error::from)
}

/// Submit a native sweep that references only independently validated envelopes.
pub fn submit_sweep(client: &KubeClient, envelopes: &[ControllerEnvelope]) -> anyhow::Result<u16> {
    let first = envelopes.first().ok_or_else(|| {
        anyhow::anyhow!("native Kubernetes sweep requires at least one --envelope")
    })?;
    if envelopes
        .iter()
        .any(|envelope| envelope.namespace != first.namespace)
    {
        anyhow::bail!("native Kubernetes sweep envelopes must use one namespace");
    }
    let body = json!({
        "apiVersion": format!("{AIPERF_GROUP}/{AIPERF_VERSION}"),
        "kind": "AIPerfSweep",
        "metadata": {"name": format!("{}-sweep", first.job_id), "namespace": first.namespace},
        "spec": {"contractVersion": super::contract::CONTRACT_VERSION, "envelopes": envelopes},
    });
    let body = serde_json::to_vec(&body)
        .map_err(|error| anyhow::anyhow!("failed to serialize native Kubernetes sweep: {error}"))?;
    client
        .request(
            "POST",
            &format!(
                "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{}/aiperfsweeps",
                first.namespace
            ),
            "application/json",
            body,
        )
        .map_err(anyhow::Error::from)
}

/// Extract every repeatable `--envelope <path>` or `--envelope=<path>` argument.
pub fn envelope_paths(args: &[String]) -> Result<Vec<&Path>, KubeError> {
    let mut paths = Vec::new();
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        if let Some(path) = argument.strip_prefix("--envelope=") {
            paths.push(Path::new(path));
        } else if argument == "--envelope" {
            let path = arguments.next().ok_or_else(|| {
                KubeError::ContractValidation("--envelope requires a path".to_string())
            })?;
            paths.push(Path::new(path));
        }
    }
    if paths.is_empty() {
        return Err(KubeError::ContractValidation(
            "native Kubernetes profile and sweep require --envelope <native-k8s/v1.json>"
                .to_string(),
        ));
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_arguments_preserve_repeatable_order() {
        let arguments = [
            "--envelope".to_string(),
            "one.json".to_string(),
            "--envelope=two.json".to_string(),
        ];
        let paths = envelope_paths(&arguments).expect("paths");
        assert_eq!(paths, vec![Path::new("one.json"), Path::new("two.json")]);
    }

    #[test]
    fn bootstrap_material_selects_roles_and_rejects_unknown_names() {
        let arguments = [
            "--bootstrap-material".to_string(),
            "controller=/run/controller.bin".to_string(),
            "--bootstrap-material=cell=/run/cell.bin".to_string(),
        ];
        let selected = material_paths(&arguments).expect("material");
        assert_eq!(selected.len(), 2);
        assert_eq!(
            selected.get(&NativeK8sRole::Controller).map(PathBuf::as_path),
            Some(Path::new("/run/controller.bin"))
        );
        assert!(material_paths(&["--bootstrap-material=aggregator=/x".to_string()]).is_err());
    }

    #[test]
    fn job_collection_is_namespace_scoped() {
        assert_eq!(
            jobs_path("bench"),
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs"
        );
    }
}
