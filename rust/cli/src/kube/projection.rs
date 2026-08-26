// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared `native-k8s/v1` controller-envelope projection over resolved bootstrap material.

use std::collections::{BTreeMap, BTreeSet};

use super::contract::{ControllerEnvelope, NativeK8sRole};
use super::error::KubeError;
use super::submission::BootstrapMaterialTarget;

/// SHA-256 digest of the bootstrap bundle this process minted for each workload identity.
pub type BootstrapDigests = BTreeMap<BootstrapMaterialTarget, String>;

/// The reference-only bootstrap fields an envelope declares for one workload identity.
#[derive(Clone, Debug)]
pub struct DeclaredBootstrap {
    /// Immutable Secret name.
    pub secret_name: String,
    /// Role the bundle authorizes.
    pub role: NativeK8sRole,
    /// Absolute container mount path.
    pub mount_path: String,
    /// Authored digest, which is the integrity check against operator-supplied material.
    pub sha256: String,
}

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

/// Locate the bootstrap reference an envelope declares for one workload identity.
///
/// Callers that mint material need the reference-only fields before any bytes exist;
/// callers that accept operator-supplied material need the authored digest to check it.
pub fn declared_bootstrap(
    envelope: &ControllerEnvelope,
    target: &BootstrapMaterialTarget,
) -> Option<DeclaredBootstrap> {
    match target {
        BootstrapMaterialTarget::Role(name) => envelope
            .roles
            .iter()
            .find(|role| role.name == *name)
            .and_then(|role| role.bootstrap.as_ref())
            .map(|bootstrap| DeclaredBootstrap {
                secret_name: bootstrap.secret_name.clone(),
                role: bootstrap.role,
                mount_path: bootstrap.mount_path.clone(),
                sha256: bootstrap.sha256.clone(),
            }),
        BootstrapMaterialTarget::Cell(cell_id) => envelope
            .cell_bootstraps
            .iter()
            .find(|bootstrap| bootstrap.cell_id == *cell_id)
            .map(|bootstrap| DeclaredBootstrap {
                secret_name: bootstrap.secret_name.clone(),
                role: bootstrap.role,
                mount_path: bootstrap.mount_path.clone(),
                sha256: bootstrap.sha256.clone(),
            }),
    }
}

/// Rebuild an envelope whose minted bootstrap references carry the minted bundle digests.
///
/// Minted material cannot be predicted when the envelope is authored, so a minted identity's
/// submitted digest is the digest of the bytes this process placed in its Secret. An identity
/// whose material the operator supplied keeps its authored digest, which stays the integrity
/// check against the named file.
///
/// Every bundle in one run shares a single nonce and roster, so a run mints either every
/// identity or none. A partial minted set is refused here as well as at the submission
/// boundary, because such a run could only fail later at cellular registration.
pub fn build_controller_envelope(
    base: &ControllerEnvelope,
    minted: &BootstrapDigests,
) -> Result<ControllerEnvelope, KubeError> {
    if !minted.is_empty()
        && minted.keys().cloned().collect::<BTreeSet<_>>() != bootstrap_targets(base)
    {
        return Err(KubeError::ContractValidation(
            "minted bootstrap material must cover every declared workload identity or none"
                .to_string(),
        ));
    }
    let mut envelope = base.clone();
    for role in &mut envelope.roles {
        let target = BootstrapMaterialTarget::Role(role.name);
        if let Some(bootstrap) = &mut role.bootstrap
            && let Some(digest) = minted.get(&target)
        {
            bootstrap.sha256.clone_from(digest);
        }
    }
    for bootstrap in &mut envelope.cell_bootstraps {
        let target = BootstrapMaterialTarget::Cell(bootstrap.cell_id);
        if let Some(digest) = minted.get(&target) {
            bootstrap.sha256.clone_from(digest);
        }
    }
    Ok(envelope)
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
    fn minted_digests_replace_every_authored_digest() {
        let base = fixture();
        let minted = BootstrapDigests::from([
            (
                BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
                "a".repeat(64),
            ),
            (BootstrapMaterialTarget::Cell(0), "b".repeat(64)),
        ]);

        let projected = build_controller_envelope(&base, &minted).expect("projection");

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
            minted.keys().cloned().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn operator_supplied_material_keeps_its_authored_digest() {
        let base = fixture();

        let projected =
            build_controller_envelope(&base, &BootstrapDigests::new()).expect("projection");

        assert_eq!(
            projected.cell_bootstraps[0].sha256,
            base.cell_bootstraps[0].sha256
        );
        assert_eq!(
            projected.roles[0]
                .bootstrap
                .as_ref()
                .map(|bootstrap| bootstrap.sha256.clone()),
            base.roles[0]
                .bootstrap
                .as_ref()
                .map(|bootstrap| bootstrap.sha256.clone())
        );
    }

    #[test]
    fn partially_minted_material_refuses_projection() {
        let base = fixture();
        let minted = BootstrapDigests::from([(
            BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
            "a".repeat(64),
        )]);

        assert!(build_controller_envelope(&base, &minted).is_err());
    }
}
