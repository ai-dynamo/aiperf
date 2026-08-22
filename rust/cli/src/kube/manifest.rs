// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Exact `native-k8s/v1` AIPerfJob envelope projection.

use serde_json::{Value, json};

use super::contract::ControllerEnvelope;
use super::error::KubeError;

/// Project an accepted envelope into the installed AIPerfJob resource.
pub fn project(envelope: &ControllerEnvelope) -> Result<Value, KubeError> {
    Ok(json!({
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": envelope.job_id, "namespace": envelope.namespace},
        "spec": {"envelope": envelope}
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::kube::contract::validate_envelope;

    #[test]
    fn projects_envelope_without_secret_bytes() {
        let input: Value = serde_json::from_str(include_str!(
            "../../../../contracts/native-k8s/v1/fixtures/valid-multi-cell-envelope.json"
        ))
        .expect("fixture");
        let projected = project(&validate_envelope(input).expect("envelope")).expect("projection");
        assert_eq!(projected["spec"]["envelope"]["cells"], 4);
        assert!(
            !serde_json::to_string(&projected)
                .expect("JSON")
                .contains("private bootstrap")
        );
    }

    #[test]
    fn generated_aiperf_job_conforms_to_checked_in_crd() {
        let input: Value = serde_json::from_str(include_str!(
            "../../../../contracts/native-k8s/v1/fixtures/valid-one-cell-envelope.json"
        ))
        .expect("fixture");
        let projected = project(&validate_envelope(input).expect("envelope")).expect("projection");
        let crd: Value = serde_yaml::from_str(include_str!(
            "../../../../deploy/aiperf-k8s-operator/crds/aiperfjobs.aiperf.nvidia.com.yaml"
        ))
        .expect("CRD YAML");
        let schema = &crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"];
        let validator = jsonschema::validator_for(schema).expect("CRD schema");

        assert!(
            validator.validate(&projected).is_ok(),
            "generated AIPerfJob must conform to the installed CRD"
        );
        assert_eq!(
            projected["spec"]
                .as_object()
                .expect("AIPerfJob spec")
                .keys()
                .collect::<Vec<_>>(),
            ["envelope"]
        );
    }
}
