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
use super::contract::{
    ControllerEnvelope, NativeK8sRole, validate_envelope, validate_image_capabilities,
};
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
            cell if let Some(cell_id) = cell.strip_prefix("cell-") => {
                let cell_id = cell_id.parse().map_err(|_| {
                    KubeError::Decode(format!("--bootstrap-material has invalid cell id {cell}"))
                })?;
                BootstrapMaterialTarget::Cell(cell_id)
            }
            other => {
                return Err(KubeError::Decode(format!(
                    "--bootstrap-material names unknown role or cell identity {other}"
                )));
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
    let expected = envelope
        .roles
        .iter()
        .filter_map(|role| {
            role.bootstrap
                .as_ref()
                .map(|_| BootstrapMaterialTarget::Role(role.name))
        })
        .chain(
            envelope
                .cell_bootstraps
                .iter()
                .map(|bootstrap| BootstrapMaterialTarget::Cell(bootstrap.cell_id)),
        )
        .collect::<std::collections::BTreeSet<_>>();
    if material.len() != expected.len() || material.keys().any(|target| !expected.contains(target))
    {
        anyhow::bail!(
            "--bootstrap-material must provide exactly one path for every workload identity"
        );
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

/// Create, submit, and owner-bind one workload as a compensating transaction.
pub fn submit_profile_transactionally(
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
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use super::super::auth::KubeCredentials;
    use super::super::client::{KubeRequest, KubeResponse, KubeTransport, KubeWatch};

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
}
