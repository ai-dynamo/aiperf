// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict typed representation and schema validation for `native-k8s/v1`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::KubeError;

/// The only Kubernetes contract version accepted by this binary.
pub const CONTRACT_VERSION: &str = "native-k8s/v1";

/// A submitted controller envelope.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ControllerEnvelope {
    /// Versioned boundary identifier.
    pub contract_version: String,
    /// Stable run identity.
    pub run_id: String,
    /// Target Kubernetes namespace.
    pub namespace: String,
    /// AIPerfJob name.
    pub job_id: String,
    /// Immutable benchmark-image digest.
    pub image_digest: String,
    /// Number of cellular workers.
    pub cells: u32,
    /// Controller-visible artifact root.
    pub artifact_root: String,
    /// Immutable native configuration reference.
    pub config_ref: NamedReference,
    /// Reachable controller coordinate for cells.
    pub controller_address: String,
    /// Fixed v1 workload roles.
    pub roles: Vec<RoleEnvelope>,
}

/// A named Kubernetes object reference.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NamedReference {
    /// Object name in the submitted namespace.
    pub name: String,
}

/// A fixed workload role and its process material.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RoleEnvelope {
    /// The v1 workload role.
    pub name: NativeK8sRole,
    /// OCI command vector.
    pub command: Vec<String>,
    /// OCI argument vector.
    pub argv: Vec<String>,
    /// Fixed process environment.
    pub environment: std::collections::BTreeMap<String, String>,
    /// Reference-only bootstrap mount.
    pub bootstrap: BootstrapReference,
}

/// Roles representable by `native-k8s/v1`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeK8sRole {
    /// Cellular controller process.
    Controller,
    /// Cellular worker process.
    Cell,
    /// Read-only final-artifact server.
    ResultsSidecar,
}

/// Reference to Rust-minted immutable bootstrap material.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BootstrapReference {
    /// Immutable Secret name.
    pub secret_name: String,
    /// Role for which the bootstrap is valid.
    pub role: NativeK8sRole,
    /// Absolute container mount path.
    pub mount_path: String,
    /// SHA-256 digest of the private bootstrap bytes.
    pub sha256: String,
}

/// Image features required by the projected envelope.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageCapabilities {
    /// Versioned boundary identifier.
    pub contract_version: String,
    /// Digest of the inspected image.
    pub image_digest: String,
    /// Image supports native cellular execution.
    pub cellular: bool,
    /// Image contains the native results sidecar.
    pub results_sidecar: bool,
    /// v1 must refuse hierarchical aggregation.
    pub hierarchical_aggregation: bool,
}

/// Decode and validate an envelope before it can cross the CLI/operator boundary.
pub fn validate_envelope(value: Value) -> Result<ControllerEnvelope, KubeError> {
    require_supported_version(&value)?;
    validate_schema(
        include_str!("../../../../contracts/native-k8s/v1/controller-envelope.schema.json"),
        &value,
    )?;
    let envelope = serde_json::from_value::<ControllerEnvelope>(value)
        .map_err(|error| KubeError::Decode(error.to_string()))?;
    for role in &envelope.roles {
        if role.name != role.bootstrap.role {
            return Err(KubeError::ContractValidation(format!(
                "bootstrap role for {:?} does not match workload role {:?}",
                role.bootstrap.role, role.name
            )));
        }
    }
    Ok(envelope)
}

/// Decode and validate image capabilities against an envelope's immutable digest.
pub fn validate_image_capabilities(
    value: Value,
    image_digest: &str,
) -> Result<ImageCapabilities, KubeError> {
    require_supported_version(&value)?;
    validate_schema(
        include_str!("../../../../contracts/native-k8s/v1/image-capabilities.schema.json"),
        &value,
    )?;
    let capabilities = serde_json::from_value::<ImageCapabilities>(value)
        .map_err(|error| KubeError::Decode(error.to_string()))?;
    if capabilities.image_digest != image_digest {
        return Err(KubeError::ContractValidation(format!(
            "image capability digest {} does not match envelope digest {image_digest}",
            capabilities.image_digest
        )));
    }
    Ok(capabilities)
}

fn require_supported_version(value: &Value) -> Result<(), KubeError> {
    let version = value
        .get("contractVersion")
        .and_then(Value::as_str)
        .ok_or_else(|| KubeError::UnsupportedContractVersion("<missing>".to_string()))?;
    if version != CONTRACT_VERSION {
        return Err(KubeError::UnsupportedContractVersion(version.to_string()));
    }
    Ok(())
}

fn validate_schema(schema_source: &str, value: &Value) -> Result<(), KubeError> {
    let schema: Value = serde_json::from_str(schema_source)
        .map_err(|error| KubeError::Decode(format!("embedded schema is invalid: {error}")))?;
    let validator = jsonschema::validator_for(&schema)
        .map_err(|error| KubeError::ContractValidation(error.to_string()))?;
    if let Err(error) = validator.validate(value) {
        return Err(KubeError::ContractValidation(error.to_string()));
    }
    Ok(())
}
