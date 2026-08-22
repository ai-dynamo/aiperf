// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native envelope loading and Kubernetes submission projections.

use std::path::Path;

use serde_json::{Value, json};

use super::client::{AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION, KubeClient};
use super::contract::{ControllerEnvelope, validate_envelope};
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
    fn job_collection_is_namespace_scoped() {
        assert_eq!(
            jobs_path("bench"),
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs"
        );
    }
}
