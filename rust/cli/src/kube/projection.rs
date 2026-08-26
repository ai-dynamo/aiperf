// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared `native-k8s/v1` controller-envelope projection over resolved bootstrap material.

use std::collections::{BTreeMap, BTreeSet};

use super::contract::{ControllerEnvelope, NativeK8sRole};
use super::error::KubeError;
use super::submission::BootstrapMaterialTarget;

/// SHA-256 digest of the bootstrap bundle selected for each workload identity.
pub type BootstrapDigests = BTreeMap<BootstrapMaterialTarget, String>;

/// Every workload identity for which an envelope declares bootstrap material.
pub fn bootstrap_targets(envelope: &ControllerEnvelope) -> BTreeSet<BootstrapMaterialTarget> {
    envelope
        .roles
        .iter()
        .filter(|role| role.bootstrap.is_some())
        .map(|role| BootstrapMaterialTarget::Role(role.name))
        .chain(
            envelope
                .cell_bootstraps
                .iter()
                .map(|bootstrap| BootstrapMaterialTarget::Cell(bootstrap.cell_id)),
        )
        .collect()
}

/// Locate the Secret name, authorized role, and mount path an envelope declares for one identity.
///
/// Callers that mint material need these three reference-only fields before any bytes exist;
/// the digest is the only field the minted bundle contributes back.
pub fn declared_reference(
    envelope: &ControllerEnvelope,
    target: &BootstrapMaterialTarget,
) -> Option<(String, NativeK8sRole, String)> {
    match target {
        BootstrapMaterialTarget::Role(name) => envelope
            .roles
            .iter()
            .find(|role| role.name == *name)
            .and_then(|role| role.bootstrap.as_ref())
            .map(|bootstrap| {
                (
                    bootstrap.secret_name.clone(),
                    bootstrap.role,
                    bootstrap.mount_path.clone(),
                )
            }),
        BootstrapMaterialTarget::Cell(cell_id) => envelope
            .cell_bootstraps
            .iter()
            .find(|bootstrap| bootstrap.cell_id == *cell_id)
            .map(|bootstrap| {
                (
                    bootstrap.secret_name.clone(),
                    bootstrap.role,
                    bootstrap.mount_path.clone(),
                )
            }),
    }
}

/// Rebuild an envelope whose bootstrap references carry the resolved bundle digests.
///
/// Minted material cannot be predicted when the envelope is authored, so the submitted
/// digest is always the digest of the bytes this process actually placed in the Secret.
/// Every declared identity must be resolved; a missing one is a caller invariant failure.
pub fn build_controller_envelope(
    base: &ControllerEnvelope,
    digests: &BootstrapDigests,
) -> Result<ControllerEnvelope, KubeError> {
    let mut envelope = base.clone();
    for role in &mut envelope.roles {
        let name = role.name;
        if let Some(bootstrap) = &mut role.bootstrap {
            bootstrap
                .sha256
                .clone_from(resolve(digests, &BootstrapMaterialTarget::Role(name))?);
        }
    }
    for bootstrap in &mut envelope.cell_bootstraps {
        let target = BootstrapMaterialTarget::Cell(bootstrap.cell_id);
        bootstrap.sha256.clone_from(resolve(digests, &target)?);
    }
    Ok(envelope)
}

fn resolve<'a>(
    digests: &'a BootstrapDigests,
    target: &BootstrapMaterialTarget,
) -> Result<&'a String, KubeError> {
    digests.get(target).ok_or_else(|| {
        KubeError::ContractValidation(format!(
            "no bootstrap material was resolved for {target:?}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kube::contract::validate_envelope;

    fn fixture() -> ControllerEnvelope {
        let input = serde_json::from_str(include_str!(
            "../../../../contracts/native-k8s/v1/fixtures/valid-one-cell-envelope.json"
        ))
        .expect("fixture JSON");
        validate_envelope(input).expect("fixture valid")
    }

    #[test]
    fn envelope_carries_resolved_digests_for_every_declared_identity() {
        let base = fixture();
        let digests = BootstrapDigests::from([
            (
                BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
                "a".repeat(64),
            ),
            (BootstrapMaterialTarget::Cell(0), "b".repeat(64)),
        ]);

        let projected = build_controller_envelope(&base, &digests).expect("projection");

        assert_eq!(
            projected.roles[0]
                .bootstrap
                .as_ref()
                .map(|bootstrap| bootstrap.sha256.as_str()),
            Some("a".repeat(64).as_str())
        );
        assert_eq!(projected.cell_bootstraps[0].sha256, "b".repeat(64));
        assert_eq!(
            bootstrap_targets(&base),
            digests.keys().cloned().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn unresolved_identity_refuses_projection() {
        let base = fixture();
        let digests = BootstrapDigests::from([(
            BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
            "a".repeat(64),
        )]);

        assert!(build_controller_envelope(&base, &digests).is_err());
    }
}
