// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Exact `native-k8s/v1` resource projection without operator-side derivation.

use serde_json::{Value, json};

use super::contract::{ControllerEnvelope, NativeK8sRole, RoleEnvelope};
use super::error::KubeError;

/// Project an accepted envelope into its AIPerfJob and exact controller/cell topology.
pub fn project(envelope: &ControllerEnvelope) -> Result<Value, KubeError> {
    let controller = role(envelope, NativeK8sRole::Controller)?;
    let cell = role(envelope, NativeK8sRole::Cell)?;
    let sidecar = role(envelope, NativeK8sRole::ResultsSidecar)?;
    Ok(json!({
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": envelope.job_id, "namespace": envelope.namespace},
        "spec": {
            "envelope": envelope,
            "jobSet": {
                "apiVersion": "jobset.x-k8s.io/v1alpha2",
                "kind": "JobSet",
                "spec": {"replicatedJobs": [
                    job("controller", 1, envelope, &[controller, sidecar]),
                    job("cell", envelope.cells, envelope, &[cell]),
                ]}
            }
        }
    }))
}

fn role(envelope: &ControllerEnvelope, needle: NativeK8sRole) -> Result<&RoleEnvelope, KubeError> {
    envelope.roles.iter().find(|role| role.name == needle).ok_or_else(|| {
        KubeError::ContractValidation(format!("native-k8s/v1 envelope omits {needle:?} role"))
    })
}

fn job(name: &str, replicas: u32, envelope: &ControllerEnvelope, roles: &[&RoleEnvelope]) -> Value {
    json!({
        "name": name,
        "replicas": replicas,
        "template": {"spec": {
            "restartPolicy": "Never",
            "serviceAccountName": "aiperf-workload",
            "containers": roles.iter().map(|role| container(role, envelope)).collect::<Vec<_>>(),
            "volumes": roles.iter().map(volume).collect::<Vec<_>>(),
        }}
    })
}

fn container(role: &RoleEnvelope, envelope: &ControllerEnvelope) -> Value {
    let mut environment = role.environment.clone();
    environment.insert("AIPERF_JOB_ID".to_string(), envelope.job_id.clone());
    environment.insert("AIPERF_NAMESPACE".to_string(), envelope.namespace.clone());
    environment.insert("AIPERF_CELL_LAUNCHER".to_string(), "k8s".to_string());
    environment.insert("AIPERF_CONTROLLER_ADDRESS".to_string(), envelope.controller_address.clone());
    environment.insert("AIPERF_ROLE_BOOTSTRAP_PATH".to_string(), role.bootstrap.mount_path.clone());
    json!({
        "name": role.name,
        "image": envelope.image_digest,
        "command": role.command,
        "args": role.argv,
        "env": environment.into_iter().map(|(name, value)| json!({"name": name, "value": value})).collect::<Vec<_>>(),
        "volumeMounts": [{"name": format!("bootstrap-{}", role.name), "mountPath": role.bootstrap.mount_path, "readOnly": true}],
    })
}

fn volume(role: &&RoleEnvelope) -> Value {
    json!({"name": format!("bootstrap-{}", role.name), "secret": {"secretName": role.bootstrap.secret_name}})
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::kube::contract::validate_envelope;

    #[test]
    fn projects_exact_three_native_roles_without_secret_bytes() {
        let input: Value = serde_json::from_str(include_str!("../../../../contracts/native-k8s/v1/fixtures/valid-multi-cell-envelope.json")).expect("fixture");
        let projected = project(&validate_envelope(input).expect("envelope")).expect("projection");
        let jobs = projected["spec"]["jobSet"]["spec"]["replicatedJobs"].as_array().expect("jobs");
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0]["template"]["spec"]["containers"].as_array().expect("containers").len(), 2);
        assert_eq!(jobs[1]["replicas"], 4);
        assert!(serde_json::to_string(&projected).expect("JSON").contains("secretName"));
        assert!(!serde_json::to_string(&projected).expect("JSON").contains("private bootstrap"));
    }
}
