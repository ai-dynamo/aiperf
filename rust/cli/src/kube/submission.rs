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

/// Bootstrap material path selected on the command line for one workload identity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BootstrapMaterialTarget {
    /// One non-cell workload role.
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
            "results-sidecar" => BootstrapMaterialTarget::Role(NativeK8sRole::ResultsSidecar),
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

/// Create one immutable Secret per workload identity after proving the envelope digest.
///
/// Material never leaves this call as plaintext in a CR, JobSet, or log; the
/// envelope keeps only the reference metadata the operator is allowed to see.
pub fn create_bootstrap_secrets(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    material: &BTreeMap<BootstrapMaterialTarget, PathBuf>,
) -> anyhow::Result<usize> {
    let expected = envelope
        .roles
        .iter()
        .filter(|role| role.name != NativeK8sRole::Cell)
        .map(|role| BootstrapMaterialTarget::Role(role.name))
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
    let mut created = 0;
    for role in &envelope.roles {
        if role.name == NativeK8sRole::Cell {
            continue;
        }
        let Some(bootstrap) = &role.bootstrap else {
            continue;
        };
        let path = material
            .get(&BootstrapMaterialTarget::Role(role.name))
            .ok_or_else(|| anyhow::anyhow!("missing bootstrap material for {:?}", role.name))?;
        let role_name = match role.name {
            NativeK8sRole::Controller => "controller",
            NativeK8sRole::ResultsSidecar => "results-sidecar",
            NativeK8sRole::Cell => {
                anyhow::bail!("cell roles must use per-cell bootstrap references")
            }
        };
        create_bootstrap_secret(
            client,
            envelope,
            path,
            &bootstrap.secret_name,
            &bootstrap.sha256,
            role_name,
        )?;
        created += 1;
    }
    for bootstrap in &envelope.cell_bootstraps {
        let path = material
            .get(&BootstrapMaterialTarget::Cell(bootstrap.cell_id))
            .ok_or_else(|| {
                anyhow::anyhow!("missing bootstrap material for cell {}", bootstrap.cell_id)
            })?;
        create_bootstrap_secret(
            client,
            envelope,
            path,
            &bootstrap.secret_name,
            &bootstrap.sha256,
            "cell",
        )?;
        created += 1;
    }
    Ok(created)
}

/// Refuse a sweep whose envelopes cannot share one CLI material selection.
pub fn validate_sweep_material_compatibility(
    envelopes: &[ControllerEnvelope],
) -> anyhow::Result<()> {
    let Some(first) = envelopes.first() else {
        return Ok(());
    };
    let expected = bootstrap_material_digests(first);
    if envelopes[1..]
        .iter()
        .any(|envelope| bootstrap_material_digests(envelope) != expected)
    {
        anyhow::bail!(
            "native Kubernetes sweep envelopes must require identical bootstrap material targets and digests"
        );
    }
    Ok(())
}

fn bootstrap_material_digests(
    envelope: &ControllerEnvelope,
) -> BTreeMap<BootstrapMaterialTarget, &str> {
    let mut digests = BTreeMap::new();
    for role in &envelope.roles {
        if role.name != NativeK8sRole::Cell {
            if let Some(bootstrap) = &role.bootstrap {
                digests.insert(
                    BootstrapMaterialTarget::Role(role.name),
                    bootstrap.sha256.as_str(),
                );
            }
        }
    }
    for bootstrap in &envelope.cell_bootstraps {
        digests.insert(
            BootstrapMaterialTarget::Cell(bootstrap.cell_id),
            bootstrap.sha256.as_str(),
        );
    }
    digests
}

fn create_bootstrap_secret(
    client: &KubeClient,
    envelope: &ControllerEnvelope,
    path: &Path,
    secret_name: &str,
    expected_digest: &str,
    role: &str,
) -> anyhow::Result<()> {
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
    } else if !(200..300).contains(&status) {
        anyhow::bail!(
            "bootstrap Secret {} creation returned HTTP {status}",
            secret_name
        );
    }
    Ok(())
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
    if existing["immutable"] != true
        || metadata["name"] != secret_name
        || metadata["namespace"] != envelope.namespace
        || metadata["labels"]["aiperf.nvidia.com/run-id"] != envelope.run_id
        || metadata["labels"]["aiperf.nvidia.com/role"] != role
        || metadata["annotations"]["aiperf.nvidia.com/sha256"] != expected_digest
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
    fn sweep_requires_one_compatible_bootstrap_material_set() {
        let one_cell = fixture("valid-one-cell-envelope.json");
        let multi_cell = fixture("valid-multi-cell-envelope.json");
        assert!(
            validate_sweep_material_compatibility(&[one_cell.clone(), one_cell.clone(),]).is_ok()
        );
        assert!(validate_sweep_material_compatibility(&[one_cell, multi_cell]).is_err());
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
