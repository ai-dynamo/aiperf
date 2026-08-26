// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native envelope loading and Kubernetes submission projections.

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

use aiperf_runtime::engine::cellular_bootstrap::{CellularRole, mint_deployment_material};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing::warn;

use super::bootstrap::create_bundle;
use super::client::{AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION, KubeClient};
use super::contract::{
    ControllerEnvelope, NativeK8sRole, SWEEP_CONTROLLER_ROLE_NAME, SweepBootstrapReference,
    SweepEnvelope, validate_envelope, validate_image_capabilities,
};
use super::error::KubeError;
use super::manifest;
use super::projection::{
    BootstrapDigests, bootstrap_targets, build_controller_envelope, declared_bootstrap,
};

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

/// Read and validate a capability document against the exact submitted image digest.
pub fn validate_image_capability_document(path: &Path, image_digest: &str) -> anyhow::Result<()> {
    let source = std::fs::read(path).map_err(|error| {
        anyhow::anyhow!(
            "failed to read image capability document {}: {error}",
            path.display()
        )
    })?;
    let value: Value = serde_json::from_slice(&source).map_err(|error| {
        anyhow::anyhow!(
            "failed to decode image capability document {}: {error}",
            path.display()
        )
    })?;
    validate_image_capabilities(value, image_digest)
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("image capability document is invalid: {error}"))
}

/// Return the API path for a namespace-owned AIPerfJob collection.
pub fn jobs_path(namespace: &str) -> String {
    format!("/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/{AIPERF_PLURAL}")
}

/// Bootstrap material path selected on the command line for one workload identity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BootstrapMaterialTarget {
    /// The controller workload role.
    Role(NativeK8sRole),
    /// One numbered cellular worker.
    Cell(u32),
}

/// Parse bootstrap material paths selected on the command line.
pub fn material_paths(
    args: &[String],
) -> Result<BTreeMap<BootstrapMaterialTarget, PathBuf>, KubeError> {
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
                        KubeError::Decode("--bootstrap-material requires <role>=<path>".to_string())
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
        let target = match role {
            "controller" => BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
            cell => {
                let Some(cell_id) = cell.strip_prefix("cell-") else {
                    return Err(KubeError::Decode(format!(
                        "--bootstrap-material names unknown role or cell identity {cell}"
                    )));
                };
                let cell_id = cell_id.parse().map_err(|_| {
                    KubeError::Decode(format!("--bootstrap-material has invalid cell id {cell}"))
                })?;
                BootstrapMaterialTarget::Cell(cell_id)
            }
        };
        if selected
            .insert(target.clone(), PathBuf::from(path))
            .is_some()
        {
            return Err(KubeError::Decode(format!(
                "--bootstrap-material repeats target {target:?}"
            )));
        }
    }
    Ok(selected)
}

struct PreparedBootstrapSecret {
    name: String,
    was_created: bool,
}

fn prepare_bootstrap_secrets(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    material: &BTreeMap<BootstrapMaterialTarget, PathBuf>,
) -> anyhow::Result<Vec<PreparedBootstrapSecret>> {
    let expected = bootstrap_targets(envelope);
    if material.len() != expected.len() || material.keys().any(|target| !expected.contains(target))
    {
        anyhow::bail!("every workload identity must resolve to exactly one bootstrap bundle");
    }
    let mut prepared = Vec::with_capacity(expected.len());
    for role in &envelope.roles {
        let Some(bootstrap) = &role.bootstrap else {
            continue;
        };
        let path = material
            .get(&BootstrapMaterialTarget::Role(role.name))
            .ok_or_else(|| anyhow::anyhow!("missing bootstrap material for {:?}", role.name))?;
        if role.name != NativeK8sRole::Controller {
            anyhow::bail!("only the controller role may carry a role bootstrap")
        }
        let result = create_bootstrap_secret(
            client,
            envelope,
            path,
            &bootstrap.secret_name,
            &bootstrap.sha256,
            "controller",
        );
        match result {
            Ok(was_created) => prepared.push(PreparedBootstrapSecret {
                name: bootstrap.secret_name.clone(),
                was_created,
            }),
            Err(error) => {
                return Err(with_bootstrap_rollback(client, envelope, &prepared, error));
            }
        }
    }
    for bootstrap in &envelope.cell_bootstraps {
        let path = material
            .get(&BootstrapMaterialTarget::Cell(bootstrap.cell_id))
            .ok_or_else(|| {
                anyhow::anyhow!("missing bootstrap material for cell {}", bootstrap.cell_id)
            })?;
        let result = create_bootstrap_secret(
            client,
            envelope,
            path,
            &bootstrap.secret_name,
            &bootstrap.sha256,
            "cell",
        );
        match result {
            Ok(was_created) => prepared.push(PreparedBootstrapSecret {
                name: bootstrap.secret_name.clone(),
                was_created,
            }),
            Err(error) => {
                return Err(with_bootstrap_rollback(client, envelope, &prepared, error));
            }
        }
    }
    Ok(prepared)
}

fn create_bootstrap_secret(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    path: &Path,
    secret_name: &str,
    expected_digest: &str,
    role: &str,
) -> anyhow::Result<bool> {
    let bytes = std::fs::read(path).map_err(|error| {
        anyhow::anyhow!(
            "failed to read bootstrap material {}: {error}",
            path.display()
        )
    })?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != expected_digest {
        anyhow::bail!(
            "bootstrap material {} does not match the envelope digest for {role}",
            path.display()
        );
    }
    let body = json!({
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "immutable": true,
        "metadata": {
            "name": secret_name,
            "namespace": envelope.namespace,
            "labels": {"aiperf.nvidia.com/run-id": envelope.run_id, "aiperf.nvidia.com/role": role},
            "annotations": {"aiperf.nvidia.com/sha256": expected_digest},
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
    if status == 409 {
        validate_existing_bootstrap_secret(client, envelope, secret_name, expected_digest, role)?;
        Ok(false)
    } else if !(200..300).contains(&status) {
        anyhow::bail!(
            "bootstrap Secret {} creation returned HTTP {status}",
            secret_name
        );
    } else {
        Ok(true)
    }
}

fn validate_existing_bootstrap_secret(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    secret_name: &str,
    expected_digest: &str,
    role: &str,
) -> anyhow::Result<()> {
    let response = client.execute(
        "GET",
        &format!(
            "/api/v1/namespaces/{}/secrets/{secret_name}",
            envelope.namespace
        ),
        "application/json",
        Vec::new(),
    )?;
    if !response.is_success() {
        anyhow::bail!(
            "bootstrap Secret {secret_name} conflict lookup returned HTTP {}",
            response.status
        );
    }
    let existing: Value = serde_json::from_slice(&response.body).map_err(|error| {
        anyhow::anyhow!("failed to decode existing bootstrap Secret {secret_name}: {error}")
    })?;
    let metadata = &existing["metadata"];
    let has_owner = metadata["ownerReferences"]
        .as_array()
        .is_some_and(|owners| !owners.is_empty());
    if existing["immutable"] != true
        || metadata["name"] != secret_name
        || metadata["namespace"] != envelope.namespace
        || metadata["labels"]["aiperf.nvidia.com/run-id"] != envelope.run_id
        || metadata["labels"]["aiperf.nvidia.com/role"] != role
        || metadata["annotations"]["aiperf.nvidia.com/sha256"] != expected_digest
        || has_owner
    {
        anyhow::bail!(
            "existing bootstrap Secret {secret_name} identity does not match the submitted run"
        );
    }
    let encoded = existing["data"]["bootstrap"].as_str().ok_or_else(|| {
        anyhow::anyhow!("existing bootstrap Secret {secret_name} has no bootstrap bytes")
    })?;
    let bytes = BASE64.decode(encoded).map_err(|error| {
        anyhow::anyhow!(
            "existing bootstrap Secret {secret_name} has invalid bootstrap encoding: {error}"
        )
    })?;
    if format!("{:x}", Sha256::digest(bytes)) != expected_digest {
        anyhow::bail!(
            "existing bootstrap Secret {secret_name} bootstrap bytes do not match the submitted digest"
        );
    }
    Ok(())
}

/// Bootstrap bundles resolved for one submission, either all minted or all operator-supplied.
struct PreparedMaterial {
    /// Private per-run directory, present only when this submission minted material.
    directory: Option<PathBuf>,
    /// Bundle files this submission minted and therefore owns.
    minted_files: Vec<PathBuf>,
    /// Bundle file selected for every declared workload identity.
    paths: BTreeMap<BootstrapMaterialTarget, PathBuf>,
    /// Digest of every bundle this submission minted; empty when material was supplied.
    minted_digests: BootstrapDigests,
}

impl PreparedMaterial {
    /// Unlink every minted bundle and remove the private directory, reporting what survived.
    ///
    /// The immutable Secrets are the durable copy, so the local files are removed on both
    /// the success and the rollback path rather than being retained for inspection.
    fn cleanup(&self) -> Vec<String> {
        let mut failures = Vec::new();
        for path in &self.minted_files {
            if let Err(error) = std::fs::remove_file(path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                failures.push(format!(
                    "minted bootstrap bundle {} cleanup failed: {error}",
                    path.display()
                ));
            }
        }
        if let Some(directory) = &self.directory
            && let Err(error) = std::fs::remove_dir(directory)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            failures.push(format!(
                "minted bootstrap directory {} cleanup failed: {error}",
                directory.display()
            ));
        }
        failures
    }
}

/// Private per-run directory holding this submission's minted bundles.
///
/// The name is derived from the run identity so a second submission of the same run
/// fails on the exclusive directory creation instead of overwriting live material.
fn material_directory(envelope: &ControllerEnvelope) -> PathBuf {
    std::env::temp_dir().join(format!(
        "aiperf-bootstrap-{}-{}",
        envelope.namespace, envelope.run_id
    ))
}

/// Resolve one bundle for every declared workload identity, before any cluster effect.
///
/// A run's bundles all share one nonce and one roster, so material is minted for every
/// identity at once or supplied for every identity at once. A partial `--bootstrap-material`
/// selection is refused here rather than deferred to a cellular registration that could
/// only fail: the controller bundle would not name a separately minted cell's key.
fn prepare_material(
    envelope: &ControllerEnvelope,
    supplied: &BTreeMap<BootstrapMaterialTarget, PathBuf>,
) -> anyhow::Result<PreparedMaterial> {
    let expected = bootstrap_targets(envelope);
    if let Some(unknown) = supplied.keys().find(|target| !expected.contains(target)) {
        anyhow::bail!(
            "--bootstrap-material names {unknown:?}, which the envelope does not declare"
        );
    }
    let mut prepared = PreparedMaterial {
        directory: None,
        minted_files: Vec::new(),
        paths: BTreeMap::new(),
        minted_digests: BootstrapDigests::new(),
    };
    if supplied.is_empty() {
        if let Err(error) = mint_material(envelope, &expected, &mut prepared) {
            let failures = prepared.cleanup();
            if failures.is_empty() {
                return Err(error);
            }
            return Err(anyhow::anyhow!(
                "{error}; bootstrap material cleanup failed: {}",
                failures.join("; ")
            ));
        }
        return Ok(prepared);
    }
    if supplied.len() != expected.len() {
        anyhow::bail!(
            "--bootstrap-material must name every workload identity or none: one run's bundles share a single nonce and roster"
        );
    }
    for (target, path) in supplied {
        let declared = declared_bootstrap(envelope, target)
            .ok_or_else(|| anyhow::anyhow!("envelope declares no bootstrap for {target:?}"))?;
        let bytes = std::fs::read(path).map_err(|error| {
            anyhow::anyhow!(
                "failed to read bootstrap material {}: {error}",
                path.display()
            )
        })?;
        if format!("{:x}", Sha256::digest(&bytes)) != declared.sha256 {
            anyhow::bail!(
                "bootstrap material {} does not match the envelope digest for {target:?}",
                path.display()
            );
        }
        prepared.paths.insert(target.clone(), path.clone());
    }
    Ok(prepared)
}

/// Mint one run's complete bundle set and write each into the private per-run directory.
fn mint_material(
    envelope: &ControllerEnvelope,
    targets: &BTreeSet<BootstrapMaterialTarget>,
    prepared: &mut PreparedMaterial,
) -> anyhow::Result<()> {
    let roles = (0..envelope.cells)
        .map(CellularRole::Cell)
        .collect::<Vec<_>>();
    // The bundles are opaque bytes; only their digests leave this function. Minting first
    // keeps a rejected roster from leaving an empty private directory behind.
    let material = mint_deployment_material(&roles)
        .map_err(|error| anyhow::anyhow!("failed to mint cellular bootstrap material: {error}"))?;
    let directory = material_directory(envelope);
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&directory)
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to create private bootstrap directory {}: {error}",
                directory.display()
            )
        })?;
    prepared.directory = Some(directory.clone());
    for target in targets {
        let declared = declared_bootstrap(envelope, target)
            .ok_or_else(|| anyhow::anyhow!("envelope declares no bootstrap for {target:?}"))?;
        let bytes = match target {
            BootstrapMaterialTarget::Role(NativeK8sRole::Controller) => &material.controller,
            BootstrapMaterialTarget::Role(other) => {
                anyhow::bail!("workload role {other:?} must not carry a role bootstrap")
            }
            BootstrapMaterialTarget::Cell(cell_id) => material
                .roles
                .get(&CellularRole::Cell(*cell_id))
                .ok_or_else(|| anyhow::anyhow!("minted material omits cell {cell_id}"))?,
        };
        let (path, reference) = create_bundle(
            &directory,
            declared.secret_name,
            declared.role,
            declared.mount_path,
            bytes.as_slice(),
        )
        .map_err(|error| {
            anyhow::anyhow!("failed to write bootstrap bundle for {target:?}: {error}")
        })?;
        prepared.minted_files.push(path.clone());
        prepared.paths.insert(target.clone(), path);
        prepared
            .minted_digests
            .insert(target.clone(), reference.sha256);
    }
    Ok(())
}

/// Mint or accept bootstrap material, then submit one workload as a compensating transaction.
pub fn submit_profile_transactionally(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    material: &BTreeMap<BootstrapMaterialTarget, PathBuf>,
) -> anyhow::Result<u16> {
    let prepared = prepare_material(envelope, material)?;
    let outcome = build_controller_envelope(envelope, &prepared.minted_digests)
        .map_err(anyhow::Error::from)
        .and_then(|submitted| submit_prepared_profile(client, &submitted, &prepared.paths));
    let failures = prepared.cleanup();
    match outcome {
        Ok(status) => {
            for failure in failures {
                warn!(detail = %failure, "minted bootstrap material was left on disk");
            }
            Ok(status)
        }
        Err(error) if failures.is_empty() => Err(error),
        Err(error) => Err(anyhow::anyhow!(
            "{error}; bootstrap material cleanup failed: {}",
            failures.join("; ")
        )),
    }
}

fn submit_prepared_profile(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    material: &BTreeMap<BootstrapMaterialTarget, PathBuf>,
) -> anyhow::Result<u16> {
    let prepared = prepare_bootstrap_secrets(client, envelope, material)?;
    let body = match manifest::project(envelope)
        .map_err(anyhow::Error::from)
        .and_then(|body| {
            serde_json::to_vec(&body).map_err(|error| {
                anyhow::anyhow!("failed to serialize native Kubernetes profile: {error}")
            })
        }) {
        Ok(body) => body,
        Err(error) => {
            return Err(with_bootstrap_rollback(client, envelope, &prepared, error));
        }
    };
    let response = match client.execute(
        "POST",
        &jobs_path(&envelope.namespace),
        "application/json",
        body,
    ) {
        Ok(response) => response,
        Err(error) => {
            return Err(rollback_submission(
                client,
                envelope,
                &prepared,
                0,
                anyhow::Error::from(error),
            ));
        }
    };
    if !response.is_success() {
        return Err(with_bootstrap_rollback(
            client,
            envelope,
            &prepared,
            anyhow::anyhow!("AIPerfJob creation returned HTTP {}", response.status),
        ));
    }
    let resource: Value = serde_json::from_slice(&response.body).map_err(|error| {
        rollback_submission(
            client,
            envelope,
            &prepared,
            0,
            anyhow::anyhow!("created AIPerfJob response is invalid: {error}"),
        )
    })?;
    let metadata = resource.get("metadata").and_then(Value::as_object);
    let object_uid = metadata
        .and_then(|metadata| metadata.get("uid"))
        .and_then(Value::as_str)
        .filter(|uid| !uid.is_empty());
    let Some(object_uid) = object_uid else {
        return Err(rollback_submission(
            client,
            envelope,
            &prepared,
            0,
            anyhow::anyhow!("created AIPerfJob response omits its UID"),
        ));
    };
    if metadata
        .and_then(|metadata| metadata.get("name"))
        .and_then(Value::as_str)
        != Some(envelope.job_id.as_str())
        || metadata
            .and_then(|metadata| metadata.get("namespace"))
            .and_then(Value::as_str)
            != Some(envelope.namespace.as_str())
    {
        return Err(rollback_submission(
            client,
            envelope,
            &prepared,
            0,
            anyhow::anyhow!("created AIPerfJob response identity does not match the submission"),
        ));
    }
    let owner = json!([{
        "apiVersion": format!("{AIPERF_GROUP}/{AIPERF_VERSION}"),
        "kind": "AIPerfJob",
        "name": envelope.job_id,
        "uid": object_uid,
        "controller": true,
    }]);
    let patch = match serde_json::to_vec(&json!({"metadata": {"ownerReferences": owner}})) {
        Ok(patch) => patch,
        Err(error) => {
            return Err(rollback_submission(
                client,
                envelope,
                &prepared,
                0,
                anyhow::anyhow!("failed to encode bootstrap owner reference: {error}"),
            ));
        }
    };
    for (index, secret) in prepared.iter().enumerate() {
        let response = match client.execute(
            "PATCH",
            &format!(
                "/api/v1/namespaces/{}/secrets/{}",
                envelope.namespace, secret.name
            ),
            "application/merge-patch+json",
            patch.clone(),
        ) {
            Ok(response) => response,
            Err(error) => {
                return Err(rollback_submission(
                    client,
                    envelope,
                    &prepared,
                    index,
                    anyhow::Error::from(error),
                ));
            }
        };
        if !response.is_success() {
            return Err(rollback_submission(
                client,
                envelope,
                &prepared,
                index,
                anyhow::anyhow!(
                    "bootstrap Secret {} owner reference returned HTTP {}",
                    secret.name,
                    response.status
                ),
            ));
        }
    }
    Ok(response.status)
}

fn rollback_submission(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    prepared: &[PreparedBootstrapSecret],
    bound_count: usize,
    primary: anyhow::Error,
) -> anyhow::Error {
    let unbind_error = rollback_existing_owner_bindings(
        client,
        envelope,
        &prepared[..bound_count.min(prepared.len())],
    )
    .err();
    let cr_response = client.execute(
        "DELETE",
        &format!("{}/{}", jobs_path(&envelope.namespace), envelope.job_id),
        "application/json",
        Vec::new(),
    );
    let cr_error = match cr_response {
        Ok(response) if response.is_success() || response.status == 404 => None,
        Ok(response) => Some(format!(
            "AIPerfJob cleanup returned HTTP {}",
            response.status
        )),
        Err(error) => Some(format!("AIPerfJob cleanup failed: {error}")),
    };
    let bootstrap_error = rollback_bootstrap_secrets(client, envelope, prepared).err();
    let mut failures = Vec::new();
    if let Some(error) = unbind_error {
        failures.push(format!("{error:#}"));
    }
    if let Some(error) = cr_error {
        failures.push(error);
    }
    if let Some(error) = bootstrap_error {
        failures.push(format!("{error:#}"));
    }
    if failures.is_empty() {
        primary
    } else {
        anyhow::anyhow!("{primary}; rollback failed: {}", failures.join("; "))
    }
}

fn rollback_existing_owner_bindings(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    bound: &[PreparedBootstrapSecret],
) -> anyhow::Result<()> {
    let patch = serde_json::to_vec(&json!({"metadata": {"ownerReferences": []}}))
        .map_err(|error| anyhow::anyhow!("failed to encode owner rollback: {error}"))?;
    let mut failures = Vec::new();
    for secret in bound.iter().filter(|secret| !secret.was_created) {
        match client.execute(
            "PATCH",
            &format!(
                "/api/v1/namespaces/{}/secrets/{}",
                envelope.namespace, secret.name
            ),
            "application/merge-patch+json",
            patch.clone(),
        ) {
            Ok(response) if response.is_success() => {}
            Ok(response) => failures.push(format!(
                "Secret {} owner rollback returned HTTP {}",
                secret.name, response.status
            )),
            Err(error) => failures.push(format!(
                "Secret {} owner rollback failed: {error}",
                secret.name
            )),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(failures.join("; "))
    }
}

fn with_bootstrap_rollback(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    prepared: &[PreparedBootstrapSecret],
    primary: anyhow::Error,
) -> anyhow::Error {
    match rollback_bootstrap_secrets(client, envelope, prepared) {
        Ok(()) => primary,
        Err(error) => anyhow::anyhow!("{primary}; bootstrap rollback failed: {error:#}"),
    }
}

fn rollback_bootstrap_secrets(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    prepared: &[PreparedBootstrapSecret],
) -> anyhow::Result<()> {
    let mut failures = Vec::new();
    for secret in prepared.iter().filter(|secret| secret.was_created) {
        match client.execute(
            "DELETE",
            &format!(
                "/api/v1/namespaces/{}/secrets/{}",
                envelope.namespace, secret.name
            ),
            "application/json",
            Vec::new(),
        ) {
            Ok(response) if response.is_success() || response.status == 404 => {}
            Ok(response) => failures.push(format!(
                "Secret {} cleanup returned HTTP {}",
                secret.name, response.status
            )),
            Err(error) => failures.push(format!("Secret {} cleanup failed: {error}", secret.name)),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(failures.join("; "))
    }
}

/// Mint bootstrap material for the sweep-controller, POST a bootstrap Secret, then POST
/// the `AIPerfSweep` CR. On CR failure the Secret is deleted (compensating transaction).
///
/// `capabilities` is the raw image-capabilities JSON value; the function validates it and
/// refuses if the image does not declare `cellular: true`.
pub fn submit_sweep_transactionally(
    client: &KubeClient,
    envelope: &SweepEnvelope,
    capabilities: serde_json::Value,
) -> anyhow::Result<u16> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt as _;

    use super::sweep_controller::AIPERFSWEEPS_PLURAL;

    // 1. Validate image capabilities and require cellular support.
    let image_digest = envelope
        .image_reference
        .rsplit_once('@')
        .map(|(_, d)| d.to_string())
        .ok_or_else(|| anyhow::anyhow!("sweep imageReference is not digest-qualified"))?;
    let caps = validate_image_capabilities(capabilities, &image_digest)
        .map_err(|e| anyhow::anyhow!("image capability document is invalid: {e}"))?;
    if !caps.cellular {
        anyhow::bail!("sweep requires cellular: true in the image capability document");
    }

    // 2. Mint sweep-controller bootstrap material.
    // Cell(0) satisfies the non-empty roster requirement; only material.controller is used.
    let material = mint_deployment_material(&[CellularRole::Cell(0)]).map_err(|e| {
        anyhow::anyhow!("failed to mint sweep-controller bootstrap material: {e}")
    })?;
    let controller_bytes = &material.controller;

    // 3. Write to a private per-run 0600 temp file (mirrors prepare_material pattern).
    let temp_dir = std::env::temp_dir().join(format!(
        "aiperf-sweep-bootstrap-{}-{}",
        envelope.namespace, envelope.run_id
    ));
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&temp_dir)
        .map_err(|e| {
            anyhow::anyhow!(
                "failed to create sweep bootstrap directory {}: {e}",
                temp_dir.display()
            )
        })?;
    let temp_path = temp_dir.join("sweep-controller");
    if let Err(e) = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp_path)
        .and_then(|mut f| f.write_all(controller_bytes))
    {
        let _ = std::fs::remove_dir(&temp_dir);
        anyhow::bail!(
            "failed to write sweep-controller bootstrap material {}: {e}",
            temp_path.display()
        );
    }

    // 4. Compute SHA-256 digest of the bootstrap bytes.
    let sha256_hex = format!("{:x}", Sha256::digest(controller_bytes));

    // 5. Build the updated sweep envelope carrying the bootstrap reference.
    let secret_name = format!("bootstrap-sweep-{}", envelope.run_id);
    let mut updated = envelope.clone();
    updated.sweep_controller.bootstrap = Some(SweepBootstrapReference {
        secret_name: secret_name.clone(),
        role: SWEEP_CONTROLLER_ROLE_NAME.to_string(),
        mount_path: "/etc/aiperf/bootstrap/sweep-controller".to_string(),
        sha256: sha256_hex.clone(),
    });

    // 6. POST the bootstrap Secret.
    let secret_body = json!({
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "immutable": true,
        "metadata": {
            "name": secret_name,
            "namespace": envelope.namespace,
            "labels": {
                "aiperf.nvidia.com/run-id": envelope.run_id,
                "aiperf.nvidia.com/role": SWEEP_CONTROLLER_ROLE_NAME,
            },
            "annotations": {"aiperf.nvidia.com/sha256": sha256_hex},
        },
        "data": {"bootstrap": BASE64.encode(controller_bytes)},
    });
    let secret_bytes = serde_json::to_vec(&secret_body)
        .map_err(|e| anyhow::anyhow!("failed to encode sweep bootstrap Secret: {e}"))?;
    let secret_status = client.request(
        "POST",
        &format!("/api/v1/namespaces/{}/secrets", envelope.namespace),
        "application/json",
        secret_bytes,
    )?;
    if !(200..300).contains(&secret_status) {
        let _ = std::fs::remove_file(&temp_path);
        let _ = std::fs::remove_dir(&temp_dir);
        anyhow::bail!("sweep bootstrap Secret creation returned HTTP {secret_status}");
    }

    // 7. POST the AIPerfSweep CR. On failure, DELETE the bootstrap Secret.
    let sweep_envelope_value = serde_json::to_value(&updated)
        .map_err(|e| anyhow::anyhow!("failed to serialize sweep envelope: {e}"))?;
    let cr_body = json!({
        "apiVersion": format!("{AIPERF_GROUP}/{AIPERF_VERSION}"),
        "kind": "AIPerfSweep",
        "metadata": {
            "name": envelope.run_id,
            "namespace": envelope.namespace,
        },
        "spec": {
            "sweepEnvelope": sweep_envelope_value,
        }
    });
    let cr_bytes = serde_json::to_vec(&cr_body)
        .map_err(|e| anyhow::anyhow!("failed to encode AIPerfSweep CR: {e}"))?;
    let cr_path = format!(
        "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{}/{AIPERFSWEEPS_PLURAL}",
        envelope.namespace
    );
    let cr_status = client.request("POST", &cr_path, "application/json", cr_bytes)?;
    if !(200..300).contains(&cr_status) {
        // Compensating delete: best-effort; ignore errors to surface the primary failure.
        let _ = client.execute(
            "DELETE",
            &format!(
                "/api/v1/namespaces/{}/secrets/{secret_name}",
                envelope.namespace
            ),
            "application/json",
            Vec::new(),
        );
        let _ = std::fs::remove_file(&temp_path);
        let _ = std::fs::remove_dir(&temp_dir);
        anyhow::bail!("AIPerfSweep CR creation returned HTTP {cr_status}");
    }

    // 8. Cleanup the temp file on success.
    if let Err(e) = std::fs::remove_file(&temp_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!(
                detail = %format!("sweep bootstrap temp file cleanup failed: {e}"),
                "minted sweep-controller bootstrap material was left on disk"
            );
        }
    }
    let _ = std::fs::remove_dir(&temp_dir);

    Ok(cr_status)
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
            "native Kubernetes profile and validate require --envelope <native-k8s/v1.json>"
                .to_string(),
        ));
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};
    use std::sync::{Arc, Mutex};

    use super::super::auth::KubeCredentials;
    use super::super::client::{KubeRequest, KubeResponse, KubeTransport, KubeWatch};
    use super::super::contract::CellBootstrapReference;

    use super::*;

    const FIXTURES: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../contracts/native-k8s/v1/fixtures/"
    );

    fn fixture(name: &str) -> ControllerEnvelope {
        let source = std::fs::read_to_string(format!("{FIXTURES}{name}")).expect("fixture read");
        validate_envelope(serde_json::from_str(&source).expect("fixture JSON"))
            .expect("fixture valid")
    }

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
            "--bootstrap-material=cell-0=/run/cell-0.bin".to_string(),
            "--bootstrap-material=cell-1=/run/cell-1.bin".to_string(),
        ];
        let selected = material_paths(&arguments).expect("material");
        assert_eq!(selected.len(), 3);
        assert_eq!(
            selected
                .get(&BootstrapMaterialTarget::Role(NativeK8sRole::Controller))
                .map(PathBuf::as_path),
            Some(Path::new("/run/controller.bin"))
        );
        assert_eq!(
            selected
                .get(&BootstrapMaterialTarget::Cell(1))
                .map(PathBuf::as_path),
            Some(Path::new("/run/cell-1.bin"))
        );
        assert!(material_paths(&["--bootstrap-material=cell=/x".to_string()]).is_err());
        assert!(material_paths(&["--bootstrap-material=aggregator=/x".to_string()]).is_err());
    }

    #[test]
    fn secret_conflict_rejects_wrong_run_role_or_digest_identity() {
        for (field, wrong_value) in [
            ("run", "other-run"),
            ("role", "cell"),
            (
                "digest",
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            ),
        ] {
            let test = secret_conflict_fixture();
            let mut existing = test.identity.clone();
            match field {
                "run" => {
                    existing["metadata"]["labels"]["aiperf.nvidia.com/run-id"] =
                        Value::String(wrong_value.to_string());
                }
                "role" => {
                    existing["metadata"]["labels"]["aiperf.nvidia.com/role"] =
                        Value::String(wrong_value.to_string());
                }
                "digest" => {
                    existing["metadata"]["annotations"]["aiperf.nvidia.com/sha256"] =
                        Value::String(wrong_value.to_string());
                }
                _ => unreachable!(),
            }
            test.transport.push_response(409, Vec::new());
            test.transport.push_response(
                200,
                serde_json::to_vec(&existing).expect("existing Secret JSON"),
            );

            let error = create_bootstrap_secret(
                &test.client,
                &test.envelope,
                &test.material,
                "bootstrap-controller",
                &test.digest,
                "controller",
            )
            .expect_err("mismatched Secret identity must fail");
            assert!(
                error.to_string().contains("identity does not match"),
                "unexpected {field} mismatch error: {error}"
            );
        }
    }

    #[test]
    fn matching_secret_conflict_is_idempotently_accepted_after_get() {
        let test = secret_conflict_fixture();
        test.transport.push_response(409, Vec::new());
        test.transport.push_response(
            200,
            serde_json::to_vec(&test.identity).expect("existing Secret JSON"),
        );

        create_bootstrap_secret(
            &test.client,
            &test.envelope,
            &test.material,
            "bootstrap-controller",
            &test.digest,
            "controller",
        )
        .expect("matching existing Secret");

        let requests = test.transport.requests.lock().expect("requests");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].method, "POST");
        let submitted: Value = serde_json::from_slice(&requests[0].body).expect("submitted Secret");
        assert_eq!(submitted["immutable"], true);
        assert_eq!(
            submitted["metadata"]["labels"],
            test.identity["metadata"]["labels"]
        );
        assert_eq!(
            submitted["metadata"]["annotations"],
            test.identity["metadata"]["annotations"]
        );
        assert_eq!(requests[1].method, "GET");
        assert_eq!(
            requests[1].path,
            "/api/v1/namespaces/bench/secrets/bootstrap-controller"
        );
    }

    #[test]
    fn secret_conflict_rejects_material_owned_by_another_cr() {
        let test = secret_conflict_fixture();
        let mut existing = test.identity.clone();
        existing["metadata"]["ownerReferences"] = json!([{
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "name": "old-job",
            "uid": "old-incarnation",
            "controller": true,
        }]);
        test.transport.push_response(409, Vec::new());
        test.transport.push_response(
            200,
            serde_json::to_vec(&existing).expect("existing Secret JSON"),
        );

        let error = create_bootstrap_secret(
            &test.client,
            &test.envelope,
            &test.material,
            "bootstrap-controller",
            &test.digest,
            "controller",
        )
        .expect_err("another CR's Secret must not be rebound");

        assert!(error.to_string().contains("identity does not match"));
    }

    #[test]
    fn transactional_submission_binds_every_bootstrap_to_the_created_cr() {
        let transaction = transaction_fixture();
        for _ in 0..2 {
            transaction.transport.push_response(201, Vec::new());
        }
        transaction.transport.push_response(
            201,
            br#"{"metadata":{"name":"job-1","namespace":"bench","uid":"4f78fcbe-9aae-4cc9-ae19-204231b21575"}}"#.to_vec(),
        );
        for _ in 0..2 {
            transaction.transport.push_response(200, Vec::new());
        }

        let status = submit_profile_transactionally(
            &transaction.client,
            &transaction.envelope,
            &transaction.material,
        )
        .expect("transactional submission");

        assert_eq!(status, 201);
        let requests = transaction.transport.requests.lock().expect("requests");
        assert_eq!(requests.len(), 5);
        assert_eq!(requests[2].method, "POST");
        assert_eq!(requests[2].path, jobs_path("bench"));
        for request in &requests[3..] {
            assert_eq!(request.method, "PATCH");
            let patch: Value = serde_json::from_slice(&request.body).expect("owner patch");
            assert_eq!(
                patch,
                json!({"metadata": {"ownerReferences": [{
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfJob",
                    "name": "job-1",
                    "uid": "4f78fcbe-9aae-4cc9-ae19-204231b21575",
                    "controller": true,
                }]}})
            );
        }
    }

    #[test]
    fn transactional_submission_rolls_back_new_secrets_when_cr_is_rejected() {
        let transaction = transaction_fixture();
        for _ in 0..2 {
            transaction.transport.push_response(201, Vec::new());
        }
        transaction
            .transport
            .push_response(422, b"invalid CR".to_vec());
        for _ in 0..2 {
            transaction.transport.push_response(200, Vec::new());
        }

        let error = submit_profile_transactionally(
            &transaction.client,
            &transaction.envelope,
            &transaction.material,
        )
        .expect_err("rejected CR must fail the transaction");

        assert!(error.to_string().contains("HTTP 422"));
        let requests = transaction.transport.requests.lock().expect("requests");
        assert_eq!(requests.len(), 5);
        assert!(
            requests[3..]
                .iter()
                .all(|request| request.method == "DELETE")
        );
        assert!(
            requests[3..]
                .iter()
                .all(|request| request.path.contains("/secrets/bootstrap-"))
        );
    }

    #[test]
    fn transactional_submission_removes_cr_and_secrets_when_owner_binding_fails() {
        let transaction = transaction_fixture();
        for _ in 0..2 {
            transaction.transport.push_response(201, Vec::new());
        }
        transaction.transport.push_response(
            201,
            br#"{"metadata":{"name":"job-1","namespace":"bench","uid":"4f78fcbe-9aae-4cc9-ae19-204231b21575"}}"#.to_vec(),
        );
        transaction
            .transport
            .push_response(500, b"patch failed".to_vec());
        transaction.transport.push_response(200, Vec::new());
        for _ in 0..2 {
            transaction.transport.push_response(200, Vec::new());
        }

        let error = submit_profile_transactionally(
            &transaction.client,
            &transaction.envelope,
            &transaction.material,
        )
        .expect_err("failed owner binding must roll back the CR");

        assert!(error.to_string().contains("owner reference"));
        let requests = transaction.transport.requests.lock().expect("requests");
        assert_eq!(requests[4].method, "DELETE");
        assert_eq!(requests[4].path, format!("{}/job-1", jobs_path("bench")));
        assert!(
            requests[5..]
                .iter()
                .all(|request| request.method == "DELETE")
        );
    }

    #[test]
    fn secret_conflict_rejects_forged_digest_metadata_for_wrong_bytes() {
        let test = secret_conflict_fixture();
        let mut forged = test.identity.clone();
        forged["data"]["bootstrap"] = BASE64.encode(b"substituted bootstrap").into();
        test.transport.push_response(409, Vec::new());
        test.transport.push_response(
            200,
            serde_json::to_vec(&forged).expect("forged Secret JSON"),
        );

        let error = create_bootstrap_secret(
            &test.client,
            &test.envelope,
            &test.material,
            "bootstrap-controller",
            &test.digest,
            "controller",
        )
        .expect_err("matching metadata must not hide substituted Secret bytes");
        assert!(error.to_string().contains("bootstrap bytes do not match"));
    }

    #[test]
    fn job_collection_is_namespace_scoped() {
        assert_eq!(
            jobs_path("bench"),
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs"
        );
    }

    #[test]
    fn submission_mints_material_for_every_role() {
        let envelope = envelope_with_cells("run-mint-all", 3);
        let mint = mint_fixture(&envelope, 4);

        let status = submit_profile_transactionally(&mint.client, &envelope, &BTreeMap::new())
            .expect("minted submission");

        assert_eq!(status, 201);
        let requests = mint.transport.requests.lock().expect("requests");
        let bundles = submitted_bundles(&requests);
        assert_eq!(bundles.len(), 4);
        assert_eq!(
            bundles
                .iter()
                .map(|(_, bytes)| bytes.clone())
                .collect::<BTreeSet<_>>()
                .len(),
            4
        );
        assert!(!material_directory(&envelope).exists());
    }

    #[test]
    fn submission_envelope_never_carries_material_bytes() {
        let envelope = envelope_with_cells("run-mint-opaque", 2);
        let mint = mint_fixture(&envelope, 3);

        submit_profile_transactionally(&mint.client, &envelope, &BTreeMap::new())
            .expect("minted submission");

        let requests = mint.transport.requests.lock().expect("requests");
        let bundles = submitted_bundles(&requests);
        let submitted = requests
            .iter()
            .find(|request| request.method == "POST" && request.path == jobs_path("bench"))
            .expect("AIPerfJob submission");
        for (_, bytes) in &bundles {
            assert!(!contains_bytes(&submitted.body, bytes));
            assert!(!contains_bytes(
                &submitted.body,
                BASE64.encode(bytes).as_bytes()
            ));
            for window in bytes.windows(16) {
                assert!(
                    !contains_bytes(&submitted.body, window),
                    "projected envelope leaked minted bootstrap bytes"
                );
            }
        }
    }

    #[test]
    fn submission_rollback_removes_minted_files_and_secrets() {
        let envelope = envelope_with_cells("run-mint-rollback", 2);
        let transport = Arc::new(ConflictTransport::default());
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        for _ in 0..3 {
            transport.push_response(201, Vec::new());
        }
        transport.push_response(422, b"invalid CR".to_vec());
        for _ in 0..3 {
            transport.push_response(200, Vec::new());
        }

        let error = submit_profile_transactionally(&client, &envelope, &BTreeMap::new())
            .expect_err("rejected CR must fail the transaction");

        assert!(error.to_string().contains("HTTP 422"));
        let requests = transport.requests.lock().expect("requests");
        let deleted = requests
            .iter()
            .filter(|request| request.method == "DELETE")
            .map(|request| request.path.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(deleted.len(), 3);
        assert!(
            deleted
                .iter()
                .all(|path| path.contains("/secrets/bootstrap-"))
        );
        assert!(!material_directory(&envelope).exists());
    }

    #[test]
    fn submission_honors_explicit_bootstrap_material() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let mut envelope = envelope_with_cells("run-mint-explicit", 1);
        let provisioned = mint_deployment_material(&[CellularRole::Cell(0)])
            .expect("operator provisioned material");
        let cell_bytes = provisioned
            .roles
            .get(&CellularRole::Cell(0))
            .expect("cell bundle")
            .clone();
        let material = BTreeMap::from([
            (
                BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
                provisioned_file(directory.path(), "controller", &provisioned.controller),
            ),
            (
                BootstrapMaterialTarget::Cell(0),
                provisioned_file(directory.path(), "cell-0", &cell_bytes),
            ),
        ]);
        let controller_digest = format!("{:x}", Sha256::digest(&provisioned.controller));
        let cell_digest = format!("{:x}", Sha256::digest(&cell_bytes));
        envelope.roles[0]
            .bootstrap
            .as_mut()
            .expect("controller bootstrap")
            .sha256
            .clone_from(&controller_digest);
        envelope.cell_bootstraps[0].sha256.clone_from(&cell_digest);
        let mint = mint_fixture(&envelope, 2);

        submit_profile_transactionally(&mint.client, &envelope, &material)
            .expect("operator provisioned submission");

        let requests = mint.transport.requests.lock().expect("requests");
        let bundles = submitted_bundles(&requests);
        assert_eq!(
            bundles,
            vec![
                ("bootstrap-controller".to_string(), provisioned.controller),
                ("bootstrap-cell-0".to_string(), cell_bytes),
            ]
        );
        let submitted = requests
            .iter()
            .find(|request| request.method == "POST" && request.path == jobs_path("bench"))
            .expect("AIPerfJob submission");
        let projected: Value = serde_json::from_slice(&submitted.body).expect("AIPerfJob JSON");
        assert_eq!(
            projected["spec"]["envelope"]["cellBootstraps"][0]["sha256"],
            Value::String(cell_digest)
        );
        assert!(!material_directory(&envelope).exists());
    }

    #[test]
    fn partial_or_mismatched_bootstrap_material_is_refused_before_cluster_effects() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let mut envelope = envelope_with_cells("run-mint-refused", 1);
        let provisioned =
            mint_deployment_material(&[CellularRole::Cell(0)]).expect("provisioned material");
        let cell_path = provisioned_file(
            directory.path(),
            "cell-0",
            provisioned
                .roles
                .get(&CellularRole::Cell(0))
                .expect("cell bundle"),
        );
        let controller_path =
            provisioned_file(directory.path(), "controller", &provisioned.controller);
        envelope.roles[0]
            .bootstrap
            .as_mut()
            .expect("controller bootstrap")
            .sha256 = format!("{:x}", Sha256::digest(&provisioned.controller));
        let transport = Arc::new(ConflictTransport::default());
        let client = KubeClient::with_transport(test_credentials(), transport.clone());

        let partial = submit_profile_transactionally(
            &client,
            &envelope,
            &BTreeMap::from([(BootstrapMaterialTarget::Cell(0), cell_path.clone())]),
        )
        .expect_err("a partial mint would produce an incoherent roster");
        assert!(partial.to_string().contains("every workload identity"));

        let complete = BTreeMap::from([
            (
                BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
                controller_path,
            ),
            (BootstrapMaterialTarget::Cell(0), cell_path),
        ]);
        let mismatched = submit_profile_transactionally(&client, &envelope, &complete)
            .expect_err("the cell file does not match its authored digest");
        assert!(
            mismatched
                .to_string()
                .contains("does not match the envelope digest")
        );
        assert!(transport.requests.lock().expect("requests").is_empty());
        assert!(!material_directory(&envelope).exists());
    }

    /// Write one operator-provisioned bundle and return the path naming it.
    fn provisioned_file(directory: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(&path, bytes).expect("provisioned material");
        path
    }

    struct MintFixture {
        client: KubeClient,
        transport: Arc<ConflictTransport>,
    }

    /// Queue a successful submission for `secret_count` bootstrap identities.
    fn mint_fixture(envelope: &ControllerEnvelope, secret_count: usize) -> MintFixture {
        let transport = Arc::new(ConflictTransport::default());
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        for _ in 0..secret_count {
            transport.push_response(201, Vec::new());
        }
        transport.push_response(
            201,
            serde_json::to_vec(&json!({"metadata": {
                "name": envelope.job_id,
                "namespace": envelope.namespace,
                "uid": "4f78fcbe-9aae-4cc9-ae19-204231b21575",
            }}))
            .expect("created AIPerfJob JSON"),
        );
        for _ in 0..secret_count {
            transport.push_response(200, Vec::new());
        }
        MintFixture { client, transport }
    }

    /// Build a validated envelope with `cells` cell identities under a unique run id.
    fn envelope_with_cells(run_id: &str, cells: u32) -> ControllerEnvelope {
        let mut envelope = fixture("valid-one-cell-envelope.json");
        envelope.run_id = run_id.to_string();
        envelope.cells = cells;
        let template = envelope.cell_bootstraps[0].clone();
        envelope.cell_bootstraps = (0..cells)
            .map(|cell_id| CellBootstrapReference {
                cell_id,
                secret_name: format!("bootstrap-cell-{cell_id}"),
                ..template.clone()
            })
            .collect();
        envelope
    }

    /// Recover the exact bootstrap bytes this submission placed in each Secret.
    fn submitted_bundles(requests: &[KubeRequest]) -> Vec<(String, Vec<u8>)> {
        requests
            .iter()
            .filter(|request| request.method == "POST" && request.path.ends_with("/secrets"))
            .map(|request| {
                let body: Value = serde_json::from_slice(&request.body).expect("Secret JSON");
                let name = body["metadata"]["name"]
                    .as_str()
                    .expect("Secret name")
                    .to_string();
                let bytes = BASE64
                    .decode(body["data"]["bootstrap"].as_str().expect("bootstrap"))
                    .expect("bootstrap encoding");
                (name, bytes)
            })
            .collect()
    }

    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        !needle.is_empty()
            && haystack.len() >= needle.len()
            && haystack
                .windows(needle.len())
                .any(|window| window == needle)
    }

    struct SecretConflictFixture {
        client: KubeClient,
        transport: Arc<ConflictTransport>,
        envelope: ControllerEnvelope,
        material: PathBuf,
        digest: String,
        identity: Value,
        _directory: tempfile::TempDir,
    }

    struct TransactionFixture {
        client: KubeClient,
        transport: Arc<ConflictTransport>,
        envelope: ControllerEnvelope,
        material: BTreeMap<BootstrapMaterialTarget, PathBuf>,
        _directory: tempfile::TempDir,
    }

    fn transaction_fixture() -> TransactionFixture {
        let directory = tempfile::tempdir().expect("temporary directory");
        let bytes = b"transaction bootstrap";
        let digest = format!("{:x}", Sha256::digest(bytes));
        let mut envelope = fixture("valid-one-cell-envelope.json");
        let mut material = BTreeMap::new();
        for role in &mut envelope.roles {
            if let Some(bootstrap) = &mut role.bootstrap {
                bootstrap.sha256.clone_from(&digest);
                let path = directory.path().join(format!("{:?}", role.name));
                std::fs::write(&path, bytes).expect("bootstrap material");
                material.insert(BootstrapMaterialTarget::Role(role.name), path);
            }
        }
        for bootstrap in &mut envelope.cell_bootstraps {
            bootstrap.sha256.clone_from(&digest);
            let path = directory.path().join(format!("cell-{}", bootstrap.cell_id));
            std::fs::write(&path, bytes).expect("cell bootstrap material");
            material.insert(BootstrapMaterialTarget::Cell(bootstrap.cell_id), path);
        }
        let transport = Arc::new(ConflictTransport::default());
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        TransactionFixture {
            client,
            transport,
            envelope,
            material,
            _directory: directory,
        }
    }

    fn secret_conflict_fixture() -> SecretConflictFixture {
        let directory = tempfile::tempdir().expect("temporary directory");
        let material = directory.path().join("bootstrap");
        std::fs::write(&material, b"controller bootstrap").expect("bootstrap material");
        let digest = format!("{:x}", Sha256::digest(b"controller bootstrap"));
        let envelope = fixture("valid-one-cell-envelope.json");
        let identity = json!({
            "apiVersion": "v1",
            "kind": "Secret",
            "immutable": true,
            "metadata": {
                "name": "bootstrap-controller",
                "namespace": "bench",
                "labels": {
                    "aiperf.nvidia.com/run-id": envelope.run_id,
                    "aiperf.nvidia.com/role": "controller",
                },
                "annotations": {"aiperf.nvidia.com/sha256": digest},
            },
            "data": {"bootstrap": BASE64.encode(b"controller bootstrap")},
        });
        let transport = Arc::new(ConflictTransport::default());
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        SecretConflictFixture {
            client,
            transport,
            envelope,
            material,
            digest,
            identity,
            _directory: directory,
        }
    }

    fn test_credentials() -> KubeCredentials {
        KubeCredentials {
            host: "127.0.0.1".to_string(),
            port: 443,
            server_name: "localhost".to_string(),
            token: Some("token".to_string()),
            client_certificate_pem: None,
            client_key_pem: None,
            ca_pem: None,
            insecure_skip_tls_verify: true,
        }
    }

    #[derive(Default)]
    struct ConflictTransport {
        requests: Mutex<Vec<KubeRequest>>,
        responses: Mutex<VecDeque<KubeResponse>>,
    }

    impl ConflictTransport {
        fn push_response(&self, status: u16, body: Vec<u8>) {
            self.responses
                .lock()
                .expect("responses")
                .push_back(KubeResponse { status, body });
        }
    }

    impl KubeTransport for ConflictTransport {
        fn send(
            &self,
            _credentials: &KubeCredentials,
            request: KubeRequest,
        ) -> Result<KubeResponse, KubeError> {
            self.requests.lock().expect("requests").push(request);
            self.responses
                .lock()
                .expect("responses")
                .pop_front()
                .ok_or_else(|| KubeError::Transport("missing test response".to_string()))
        }

        fn watch(
            &self,
            _credentials: &KubeCredentials,
            _request: KubeRequest,
        ) -> Result<KubeWatch, KubeError> {
            Err(KubeError::Transport("watch is not used".to_string()))
        }
    }

    #[test]
    fn projected_envelope_passes_strict_schema_validation() {
        // Regression guard for the skip_serializing_if = "Option::is_none" fix on
        // RoleEnvelope::bootstrap. manifest::project serializes ControllerEnvelope at
        // spec.envelope; without the attribute, None bootstrap serialized as "bootstrap":null,
        // which the strict JSON schema rejects. This test ensures re-adding a null-serializing
        // optional field in ControllerEnvelope shows up immediately as a Rust test failure.
        let envelope = fixture("valid-one-cell-envelope.json");
        let projected = super::super::manifest::project(&envelope).expect("project envelope");
        validate_envelope(projected["spec"]["envelope"].clone())
            .expect("spec.envelope must pass strict schema validation after serialization");
    }
}
