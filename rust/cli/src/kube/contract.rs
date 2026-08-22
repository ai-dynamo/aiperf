// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict typed representation and schema validation for `native-k8s/v1`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::net::Ipv6Addr;

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
    /// One immutable bootstrap reference for each cellular worker identity.
    pub cell_bootstraps: Vec<CellBootstrapReference>,
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
    /// Reference-only bootstrap mount for non-cell roles.
    pub bootstrap: Option<BootstrapReference>,
}

/// Roles representable by `native-k8s/v1`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
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

/// Reference to Rust-minted immutable bootstrap material for one cell identity.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CellBootstrapReference {
    /// Zero-based cellular worker identity.
    pub cell_id: u32,
    /// Immutable Secret name.
    pub secret_name: String,
    /// This reference always authorizes a cellular worker.
    #[serde(default = "cell_bootstrap_role")]
    pub role: NativeK8sRole,
    /// Absolute container mount path.
    pub mount_path: String,
    /// SHA-256 digest of the private bootstrap bytes.
    pub sha256: String,
}

fn cell_bootstrap_role() -> NativeK8sRole {
    NativeK8sRole::Cell
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
    if !is_valid_controller_coordinate(&envelope.controller_address) {
        return Err(KubeError::ContractValidation(
            "controllerAddress must be tcp://HOST:PORT or tcp://[IPv6]:PORT".to_string(),
        ));
    }
    for role in &envelope.roles {
        if role.name == NativeK8sRole::Cell && role.bootstrap.is_some() {
            return Err(KubeError::ContractValidation(
                "cell bootstrap must be specified by cellBootstraps".to_string(),
            ));
        }
        let Some(bootstrap) = &role.bootstrap else {
            if role.name == NativeK8sRole::Cell {
                continue;
            }
            return Err(KubeError::ContractValidation(format!(
                "non-cell role {:?} has no bootstrap reference",
                role.name
            )));
        };
        if role.name != bootstrap.role {
            return Err(KubeError::ContractValidation(format!(
                "bootstrap role for {:?} does not match workload role {:?}",
                bootstrap.role, role.name
            )));
        }
    }
    if envelope.cell_bootstraps.len() != envelope.cells as usize
        || envelope
            .cell_bootstraps
            .iter()
            .enumerate()
            .any(|(index, bootstrap)| {
                bootstrap.cell_id != index as u32 || bootstrap.role != NativeK8sRole::Cell
            })
    {
        return Err(KubeError::ContractValidation(
            "cellBootstraps must contain each cell id exactly once".to_string(),
        ));
    }
    let mut secret_names = std::collections::BTreeSet::new();
    for role in &envelope.roles {
        if let Some(bootstrap) = &role.bootstrap {
            if !secret_names.insert(&bootstrap.secret_name) {
                return Err(KubeError::ContractValidation(
                    "bootstrap Secret names must be unique".to_string(),
                ));
            }
        }
    }
    for bootstrap in &envelope.cell_bootstraps {
        if !secret_names.insert(&bootstrap.secret_name) {
            return Err(KubeError::ContractValidation(
                "bootstrap Secret names must be unique".to_string(),
            ));
        }
    }
    Ok(envelope)
}

fn is_valid_controller_coordinate(address: &str) -> bool {
    let coordinate = match address.strip_prefix("tcp://") {
        Some(coordinate) => coordinate,
        None if address.contains("://") => return false,
        None => address,
    };
    let (host, port, is_ipv6) = if let Some(remainder) = coordinate.strip_prefix('[') {
        let Some((host, port)) = remainder.split_once("]:") else {
            return false;
        };
        (host, port, true)
    } else {
        let Some((host, port)) = coordinate.rsplit_once(':') else {
            return false;
        };
        (host, port, false)
    };
    let valid_host = if is_ipv6 {
        host.parse::<Ipv6Addr>().is_ok()
    } else {
        !host.is_empty()
            && !host.contains(['/', ':', '[', ']'])
            && host.bytes().all(|byte| byte.is_ascii_graphic())
    };
    valid_host && port.parse::<u16>().ok().filter(|port| *port > 0).is_some()
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
