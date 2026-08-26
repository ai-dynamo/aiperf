// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use super::auth::{KubeAuthOptions, KubeCredentials};
use super::client::{
    DEFAULT_REQUEST_DEADLINE, DEFAULT_WATCH_DEADLINE, KubeClient, KubeRequest, KubeTransport,
    KubeWatch,
};
use super::contract::{ControllerEnvelope, validate_envelope, validate_image_capabilities};
use super::error::KubeError;
use super::projection::{BootstrapDigests, build_controller_envelope};
use super::submission::BootstrapMaterialTarget;

const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/native-k8s/v1/fixtures/"
);

fn fixture(name: &str) -> Value {
    let source = std::fs::read_to_string(format!("{FIXTURES}{name}")).expect("fixture read");
    serde_json::from_str(&source).expect("fixture JSON")
}

#[test]
fn accepts_runnable_envelopes() {
    assert_eq!(
        validate_envelope(fixture("valid-one-cell-envelope.json"))
            .expect("one cell")
            .cells,
        1
    );
    assert_eq!(
        validate_envelope(fixture("valid-multi-cell-envelope.json"))
            .expect("multi cell")
            .cells,
        4
    );
}

#[test]
fn requires_a_digest_qualified_remote_image_reference() {
    let mut bare_digest = fixture("valid-one-cell-envelope.json");
    bare_digest["imageReference"] = bare_digest["imageDigest"].clone();
    assert!(matches!(
        validate_envelope(bare_digest),
        Err(KubeError::ContractValidation(_))
    ));

    let mut mismatch = fixture("valid-one-cell-envelope.json");
    mismatch["imageReference"] = Value::String(format!(
        "registry.example.com/aiperf/runner@sha256:{}",
        "1".repeat(64)
    ));
    assert!(matches!(
        validate_envelope(mismatch),
        Err(KubeError::ContractValidation(message))
            if message == "imageReference digest must equal imageDigest"
    ));

    let mut valid = fixture("valid-one-cell-envelope.json");
    valid["imageReference"] = Value::String(format!(
        "registry.example.com/aiperf/runner@{}",
        valid["imageDigest"].as_str().expect("fixture digest")
    ));
    assert_eq!(
        validate_envelope(valid)
            .expect("digest-qualified image reference")
            .image_reference,
        format!(
            "registry.example.com/aiperf/runner@sha256:{}",
            "0".repeat(64)
        )
    );
}

#[test]
fn refuses_noncanonical_kubernetes_identity_and_artifact_roots() {
    for (field, invalid) in [
        ("namespace", "NOT_A_NAMESPACE"),
        (
            "namespace",
            "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn",
        ),
        ("jobId", "../other"),
        ("jobId", "job.with.dot"),
        ("runId", "run/other"),
        (
            "runId",
            "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr",
        ),
        ("artifactRoot", "/results/../secrets"),
        ("artifactRoot", "/etc"),
        ("artifactRoot", "/results//nested"),
    ] {
        let mut envelope = fixture("valid-one-cell-envelope.json");
        envelope[field] = Value::String(invalid.to_string());
        assert!(
            matches!(
                validate_envelope(envelope),
                Err(KubeError::ContractValidation(_))
            ),
            "{field} accepted {invalid:?}"
        );
    }
}

#[test]
fn requires_a_config_content_digest() {
    let mut envelope = fixture("valid-one-cell-envelope.json");
    envelope["configRef"]
        .as_object_mut()
        .expect("configRef object")
        .remove("sha256");
    assert!(matches!(
        validate_envelope(envelope),
        Err(KubeError::ContractValidation(_))
    ));
}

#[test]
fn refuses_role_mismatched_bootstrap() {
    let mut envelope = fixture("valid-one-cell-envelope.json");
    envelope["roles"][0]["bootstrap"]["role"] = Value::String("cell".to_string());
    assert!(matches!(
        validate_envelope(envelope),
        Err(KubeError::ContractValidation(_))
    ));
}

#[test]
fn refuses_duplicate_bootstrap_secret_names() {
    let mut envelope = fixture("valid-one-cell-envelope.json");
    envelope["cellBootstraps"][0]["secretName"] =
        envelope["roles"][0]["bootstrap"]["secretName"].clone();
    assert!(matches!(
        validate_envelope(envelope),
        Err(KubeError::ContractValidation(message)) if message == "bootstrap Secret names must be unique"
    ));
}

#[test]
fn requires_an_unambiguous_controller_coordinate() {
    let mut malformed = fixture("valid-one-cell-envelope.json");
    malformed["controllerAddress"] = Value::String("controller:443:8443".to_string());
    assert!(matches!(
        validate_envelope(malformed),
        Err(KubeError::ContractValidation(message))
            if message == "controllerAddress must be tcp://HOST:PORT or tcp://[IPv6]:PORT"
    ));

    let mut ipv6 = fixture("valid-one-cell-envelope.json");
    ipv6["controllerAddress"] = Value::String("tcp://[2001:db8::1]:443".to_string());
    assert!(validate_envelope(ipv6).is_ok());
}

#[test]
fn envelope_requires_an_explicit_controller_port() {
    let mut portless = fixture("valid-one-cell-envelope.json");
    portless["controllerAddress"] = Value::String("controller".to_string());
    assert!(matches!(
        validate_envelope(portless),
        Err(KubeError::ContractValidation(message))
            if message == "controllerAddress must be tcp://HOST:PORT or tcp://[IPv6]:PORT"
    ));

    let mut with_port = fixture("valid-one-cell-envelope.json");
    with_port["controllerAddress"] = Value::String("controller:9500".to_string());
    assert!(validate_envelope(with_port).is_ok());
}

#[test]
fn refuses_non_v1_and_unknown_role_or_field() {
    assert!(matches!(
        validate_envelope(fixture("invalid-version-envelope.json")),
        Err(KubeError::UnsupportedContractVersion(_))
    ));
    assert!(matches!(
        validate_envelope(fixture("aggregator-envelope.json")),
        Err(KubeError::ContractValidation(_))
    ));
    assert!(matches!(
        validate_envelope(fixture("unknown-field-envelope.json")),
        Err(KubeError::ContractValidation(_))
    ));
}

#[test]
fn validates_required_image_capabilities_and_digest() {
    let envelope = validate_envelope(fixture("valid-one-cell-envelope.json")).expect("envelope");
    assert!(
        validate_image_capabilities(
            fixture("missing-cellular-capability.json"),
            &envelope.image_digest
        )
        .is_err()
    );
    assert!(
        validate_image_capabilities(
            fixture("image-mismatch-capability.json"),
            &envelope.image_digest
        )
        .is_err()
    );
}

#[test]
fn kubeconfig_precedence_is_explicit_then_environment_then_home() {
    let explicit = Path::new("/explicit/config");
    assert_eq!(
        KubeAuthOptions::kubeconfig_path_from(
            Some(explicit),
            Some("/environment/config".into()),
            Some(PathBuf::from("/home/user"))
        )
        .expect("explicit"),
        explicit
    );
    assert_eq!(
        KubeAuthOptions::kubeconfig_path_from(
            None,
            Some("/environment/config".into()),
            Some(PathBuf::from("/home/user"))
        )
        .expect("environment"),
        PathBuf::from("/environment/config")
    );
    assert_eq!(
        KubeAuthOptions::kubeconfig_path_from(None, None, Some(PathBuf::from("/home/user")))
            .expect("home"),
        PathBuf::from("/home/user/.kube/config")
    );
}

#[test]
fn exec_credential_resolves_token() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let script = directory.path().join("credential.sh");
    std::fs::write(
        &script,
        "#!/bin/sh\nprintf '%s' '{\"status\":{\"token\":\"exec-token\"}}'\n",
    )
    .expect("script write");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700))
            .expect("script permissions");
    }
    let config = format!(
        "apiVersion: v1\ncurrent-context: test\nclusters:\n- name: cluster\n  cluster:\n    server: https://example.test:6443\n    certificate-authority-data: Y2E=\ncontexts:\n- name: test\n  context:\n    cluster: cluster\n    user: user\nusers:\n- name: user\n  user:\n    exec:\n      command: {}\n",
        script.display()
    );
    let path = directory.path().join("config");
    std::fs::write(&path, config).expect("config write");
    let credentials = KubeAuthOptions {
        kubeconfig: Some(path),
        ..Default::default()
    }
    .resolve()
    .expect("exec credential");
    assert_eq!(credentials.token.as_deref(), Some("exec-token"));
}

#[test]
fn resolves_ipv6_kubeconfig_authority() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("config");
    std::fs::write(
        &path,
        "apiVersion: v1\ncurrent-context: test\nclusters:\n- name: cluster\n  cluster:\n    server: https://[::1]:6443\n    certificate-authority-data: Y2E=\ncontexts:\n- name: test\n  context:\n    cluster: cluster\n    user: user\nusers:\n- name: user\n  user:\n    token: token\n",
    ).expect("config write");
    let credentials = KubeAuthOptions {
        kubeconfig: Some(path),
        ..Default::default()
    }
    .resolve()
    .expect("IPv6 credentials");
    assert_eq!(credentials.host, "::1");
    assert_eq!(credentials.port, 6443);
}

#[test]
fn rejects_invalid_cluster_ca_before_network_io() {
    let credentials = credentials(Some(b"not PEM".to_vec()));
    let client = KubeClient::from_credentials(credentials).expect("client construction");
    assert!(matches!(
        client.merge_patch("/apis/test", &serde_json::json!({})),
        Err(KubeError::Tls(_))
    ));
}

#[test]
fn defaults_and_overrides_keep_deadlines_finite() {
    let client =
        KubeClient::from_credentials(credentials(Some(b"not PEM".to_vec()))).expect("client");
    assert_eq!(client.request_deadline(), DEFAULT_REQUEST_DEADLINE);
    assert_eq!(client.watch_deadline(), DEFAULT_WATCH_DEADLINE);
    assert!(
        client
            .with_deadlines(Duration::ZERO, Duration::from_secs(1))
            .is_err()
    );
}

#[test]
fn watch_disconnect_is_not_reported_as_idle() {
    let watch = KubeWatch::closed_for_test();
    assert!(matches!(
        watch.poll(Duration::ZERO),
        Ok(super::client::KubeWatchPoll::Closed)
    ));
}

#[test]
fn merge_patch_constructs_reporter_compatible_request() {
    let transport = Arc::new(RecordingTransport::default());
    let client = KubeClient::with_transport(credentials(None), transport.clone());
    let status = client
        .merge_patch(
            "/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs/job/status",
            &serde_json::json!({"status":{"phase":"Done"}}),
        )
        .expect("patch");
    assert_eq!(status, 200);
    let request = transport
        .request
        .lock()
        .expect("recording lock")
        .clone()
        .expect("request");
    assert_eq!(request.method, "PATCH");
    assert_eq!(request.content_type, "application/merge-patch+json");
    assert_eq!(request.deadline, DEFAULT_REQUEST_DEADLINE);
}

fn credentials(ca_pem: Option<Vec<u8>>) -> KubeCredentials {
    KubeCredentials {
        host: "127.0.0.1".to_string(),
        port: 443,
        server_name: "localhost".to_string(),
        token: Some("token".to_string()),
        client_certificate_pem: None,
        client_key_pem: None,
        ca_pem,
        insecure_skip_tls_verify: false,
    }
}

#[derive(Default)]
struct RecordingTransport {
    request: Mutex<Option<KubeRequest>>,
}
impl KubeTransport for RecordingTransport {
    fn send(
        &self,
        _credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<super::client::KubeResponse, KubeError> {
        *self
            .request
            .lock()
            .map_err(|_| KubeError::Transport("recording lock poisoned".to_string()))? =
            Some(request);
        Ok(super::client::KubeResponse {
            status: 200,
            body: br#"{"items":[]}"#.to_vec(),
        })
    }
    fn watch(
        &self,
        _credentials: &KubeCredentials,
        _request: KubeRequest,
    ) -> Result<super::client::KubeWatch, KubeError> {
        Err(KubeError::Transport(
            "watch is not needed by this recording transport".to_string(),
        ))
    }
}

// --- kube init scaffold tests ---

#[test]
fn init_writes_a_config_and_capability_pair() {
    let dir = tempfile::tempdir().expect("tempdir");
    let exit = super::scaffold::run(&[
        "--output-directory".to_string(),
        dir.path().display().to_string(),
    ])
    .expect("scaffold run");
    assert_eq!(exit, 0);
    assert!(
        dir.path().join("benchmark.yaml").exists(),
        "benchmark.yaml must be written"
    );
    let cap_path = dir.path().join("image-capabilities.json");
    assert!(cap_path.exists(), "image-capabilities.json must be written");
    let cap_text = std::fs::read_to_string(&cap_path).expect("read capabilities");
    let cap: serde_json::Value = serde_json::from_str(&cap_text).expect("capabilities JSON");
    assert_eq!(
        cap["imageDigest"].as_str().expect("imageDigest field"),
        super::scaffold::PLACEHOLDER_DIGEST,
        "capability doc must carry the placeholder digest"
    );
}

#[test]
fn init_scaffold_fails_validation_until_edited() {
    let dir = tempfile::tempdir().expect("tempdir");
    super::scaffold::run(&[
        "--output-directory".to_string(),
        dir.path().display().to_string(),
    ])
    .expect("scaffold run");
    let cap_text =
        std::fs::read_to_string(dir.path().join("image-capabilities.json")).expect("read");
    let cap: serde_json::Value = serde_json::from_str(&cap_text).expect("json");
    // The schema rejects the placeholder via the ^sha256:[0-9a-f]{64}$ pattern.
    // Passing the placeholder as the expected digest rules out a plain digest-mismatch
    // error, leaving schema rejection as the only possible cause of Err.
    assert!(
        matches!(
            validate_image_capabilities(cap, super::scaffold::PLACEHOLDER_DIGEST),
            Err(KubeError::ContractValidation(_))
        ),
        "unedited scaffold must fail schema validation"
    );
}

#[test]
fn init_refuses_to_overwrite_without_force() {
    let dir = tempfile::tempdir().expect("tempdir");
    let args = vec![
        "--output-directory".to_string(),
        dir.path().display().to_string(),
    ];
    super::scaffold::run(&args).expect("first scaffold run");
    let error = super::scaffold::run(&args).expect_err("second scaffold run must refuse overwrite");
    assert!(
        error.to_string().contains("already exists"),
        "overwrite refusal must name the existing file: {error}"
    );
}

#[test]
fn init_never_contacts_the_cluster() {
    let dir = tempfile::tempdir().expect("tempdir");
    let kubeconfig_path = dir.path().join("unroutable.yaml");
    std::fs::write(
        &kubeconfig_path,
        concat!(
            "apiVersion: v1\ncurrent-context: test\n",
            "clusters:\n- name: cluster\n  cluster:\n    server: https://192.0.2.1:6443\n",
            "contexts:\n- name: test\n  context:\n    cluster: cluster\n    user: user\n",
            "users:\n- name: user\n  user:\n    token: notoken\n",
        ),
    )
    .expect("kubeconfig write");
    // kube init reads no kubeconfig.
    unsafe { std::env::set_var("KUBECONFIG", kubeconfig_path.as_os_str()) };
    let exit = super::command::run(&[
        "init".to_string(),
        "--template".to_string(),
        "minimal".to_string(),
        "--output-directory".to_string(),
        dir.path().join("output").display().to_string(),
    ])
    .expect("kube init with unroutable KUBECONFIG must succeed");
    assert_eq!(exit, 0, "kube init must not contact the cluster");
}

// --- kube generate tests ---

const TEST_IMAGE: &str = "registry.example.com/aiperf/runner@sha256:0000000000000000000000000000000000000000000000000000000000000000";

/// Helper: run `kube generate` with a temporary config file and the given args.
/// Returns the raw output bytes and the temp directory (keeps temp dir alive).
fn run_generate(extra_args: &[&str]) -> (Vec<u8>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("benchmark.yaml");
    std::fs::write(&config_path, "# Config-v2 placeholder for tests").expect("config write");
    let output_path = dir.path().join("generated.json");
    let mut args = vec![
        "generate".to_string(),
        "--config".to_string(),
        config_path.display().to_string(),
        "--image".to_string(),
        TEST_IMAGE.to_string(),
        "--cells".to_string(),
        "1".to_string(),
        "--output".to_string(),
        output_path.display().to_string(),
    ];
    for arg in extra_args {
        args.push(arg.to_string());
    }
    let exit = super::command::run(&args).expect("generate command must succeed");
    assert_eq!(exit, 0, "generate must exit 0");
    let bytes = std::fs::read(&output_path).expect("output file must exist");
    (bytes, dir)
}

/// Zero out every bootstrap sha256 field so two envelopes can be compared without
/// caring about which placeholder or minted digest they carry.
fn mask_bootstrap_digests(mut envelope: ControllerEnvelope) -> ControllerEnvelope {
    for role in &mut envelope.roles {
        if let Some(bootstrap) = &mut role.bootstrap {
            bootstrap.sha256 = "0".repeat(64);
        }
    }
    for bootstrap in &mut envelope.cell_bootstraps {
        bootstrap.sha256 = "0".repeat(64);
    }
    envelope
}

#[test]
fn generate_output_validates() {
    let (bytes, _dir) = run_generate(&[]);
    let value: Value = serde_json::from_slice(&bytes).expect("output must be valid JSON");
    validate_envelope(value).expect("generated envelope must pass validate_envelope");
}

#[test]
fn generate_matches_profile_projection() {
    // Generate an envelope with known inputs.
    let (bytes, _dir) = run_generate(&[]);
    let gen_envelope: ControllerEnvelope =
        serde_json::from_slice(&bytes).expect("generated output must parse as ControllerEnvelope");

    // Build a fake minted-digests map covering every declared bootstrap target.
    // profile would call build_controller_envelope with these instead of "0"*64 placeholders.
    let fake_digests = BootstrapDigests::from([
        (
            BootstrapMaterialTarget::Role(super::contract::NativeK8sRole::Controller),
            "a".repeat(64),
        ),
        (BootstrapMaterialTarget::Cell(0), "a".repeat(64)),
    ]);
    let submitted =
        build_controller_envelope(&gen_envelope, &fake_digests).expect("minted projection");

    // After masking bootstrap sha256 fields, both envelopes must be identical:
    // build_controller_envelope only replaces digest fields and preserves everything else.
    assert_eq!(
        serde_json::to_string(&mask_bootstrap_digests(gen_envelope)).expect("gen JSON"),
        serde_json::to_string(&mask_bootstrap_digests(submitted)).expect("submitted JSON"),
        "build_controller_envelope must only alter bootstrap sha256 fields"
    );
}

#[test]
fn generate_never_contacts_the_cluster() {
    // run_generate constructs no KubeClient on any code path, so no cluster is consulted.
    // Verify with two cells to confirm cell_bootstraps sizing works independently of any dial.
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("my-bench.yaml");
    std::fs::write(&config_path, "# placeholder").expect("config write");
    let output_path = dir.path().join("out.json");
    let exit = super::command::run(&[
        "generate".to_string(),
        "--config".to_string(),
        config_path.display().to_string(),
        "--image".to_string(),
        TEST_IMAGE.to_string(),
        "--cells".to_string(),
        "2".to_string(),
        "--namespace".to_string(),
        "isolated".to_string(),
        "--output".to_string(),
        output_path.display().to_string(),
    ])
    .expect("generate must succeed without a cluster");
    assert_eq!(exit, 0);
    let bytes = std::fs::read(&output_path).expect("output file");
    let value: Value = serde_json::from_slice(&bytes).expect("valid JSON");
    assert_eq!(value["cells"], 2, "cells must match --cells");
    assert_eq!(
        value["cellBootstraps"].as_array().map(|a| a.len()),
        Some(2),
        "cellBootstraps must be sized by --cells"
    );
}
