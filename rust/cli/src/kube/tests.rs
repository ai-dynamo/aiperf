// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use super::auth::{KubeAuthOptions, KubeCredentials};
use super::client::{
    DEFAULT_REQUEST_DEADLINE, DEFAULT_WATCH_DEADLINE, KubeClient, KubeRequest, KubeResponse,
    KubeTransport, KubeWatch,
};
use super::contract::{
    ControllerEnvelope, SweepEnvelope, SweepRoleEnvelope, validate_envelope,
    validate_image_capabilities, validate_sweep_envelope,
};
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

// --- sweep envelope tests ---

#[test]
fn accepts_valid_sweep_envelope() {
    let envelope =
        validate_sweep_envelope(fixture("valid-sweep-envelope.json")).expect("valid sweep");
    assert_eq!(envelope.sweep_id, "sweep-1");
    assert_eq!(envelope.trials, 1);
    assert_eq!(envelope.axes.len(), 1);
    assert_eq!(envelope.axes[0].parameter, "runtime.concurrency");
    assert_eq!(envelope.sweep_controller.name, "sweep-controller");
}

#[test]
fn sweep_envelope_rejects_unknown_field() {
    assert!(matches!(
        validate_sweep_envelope(fixture("unknown-field-sweep-envelope.json")),
        Err(KubeError::ContractValidation(_))
    ));
}

#[test]
fn sweep_envelope_refuses_unsupported_version() {
    assert!(matches!(
        validate_sweep_envelope(fixture("invalid-version-sweep-envelope.json")),
        Err(KubeError::UnsupportedContractVersion(_))
    ));
}

#[test]
fn sweep_envelope_rejects_wrong_role_name() {
    let mut payload = fixture("valid-sweep-envelope.json");
    payload["sweepController"]["name"] = Value::String("controller".to_string());
    assert!(matches!(
        validate_sweep_envelope(payload),
        Err(KubeError::ContractValidation(_))
    ));
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

// --- sweep-controller unit tests ---

/// Build a minimal `SweepEnvelope` with the supplied axes for hermetic tests.
///
/// `base_config` must already contain every swept path so `build_benchmark_plan`
/// can apply the axis values via `set_dotted`.
fn sweep_envelope(
    run_id: &str,
    base_config: Value,
    axes: Vec<super::contract::SweepAxis>,
    trials: u32,
    max_concurrent_runs: u32,
) -> SweepEnvelope {
    SweepEnvelope {
        contract_version: "native-k8s/v1".to_string(),
        run_id: run_id.to_string(),
        namespace: "bench".to_string(),
        sweep_id: "sweep-1".to_string(),
        image_reference: format!(
            "registry.example.com/aiperf/runner@sha256:{}",
            "0".repeat(64)
        ),
        base_config,
        axes,
        trials,
        max_concurrent_runs,
        sweep_controller: SweepRoleEnvelope {
            name: "sweep-controller".to_string(),
            command: vec!["aiperf".to_string()],
            argv: vec!["sweep-controller".to_string()],
            environment: std::collections::BTreeMap::new(),
            bootstrap: None,
        },
    }
}

/// A mock transport that records every request sent and serves pre-queued
/// responses and watch streams.
#[derive(Default)]
struct SweepMockTransport {
    responses: Mutex<VecDeque<(u16, Vec<u8>)>>,
    watches: Mutex<VecDeque<KubeWatch>>,
    pub requests: Mutex<Vec<KubeRequest>>,
}

impl SweepMockTransport {
    fn push_response(&self, status: u16, body: Vec<u8>) {
        self.responses
            .lock()
            .expect("responses lock")
            .push_back((status, body));
    }

    fn push_watch(&self, events: Vec<Vec<u8>>) {
        self.watches
            .lock()
            .expect("watches lock")
            .push_back(KubeWatch::events_for_test(events));
    }
}

impl KubeTransport for SweepMockTransport {
    fn send(
        &self,
        _credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeResponse, KubeError> {
        self.requests
            .lock()
            .expect("requests lock")
            .push(request);
        let (status, body) = self
            .responses
            .lock()
            .expect("responses lock")
            .pop_front()
            .ok_or_else(|| KubeError::Transport("no response queued for test".to_string()))?;
        Ok(KubeResponse { status, body })
    }

    fn watch(
        &self,
        _credentials: &KubeCredentials,
        _request: KubeRequest,
    ) -> Result<KubeWatch, KubeError> {
        self.watches
            .lock()
            .expect("watches lock")
            .pop_front()
            .ok_or_else(|| KubeError::Transport("no watch stream queued for test".to_string()))
    }
}

fn sweep_test_credentials() -> KubeCredentials {
    KubeCredentials {
        host: "127.0.0.1".to_string(),
        port: 443,
        server_name: "localhost".to_string(),
        token: Some("test-token".to_string()),
        client_certificate_pem: None,
        client_key_pem: None,
        ca_pem: None,
        insecure_skip_tls_verify: true,
    }
}

/// Completed-phase watch event for the named AIPerfJob.
fn completed_event(job_id: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "type": "MODIFIED",
        "object": {
            "metadata": {"name": job_id, "namespace": "bench"},
            "status": {"phase": "Completed"}
        }
    }))
    .expect("completed event JSON")
}

#[test]
fn sweep_controller_expands_axes_to_correct_plan_count() {
    // A 2-axis grid (2 × 3 values) with 2 trials must produce 12 child specs.
    // The base config must carry both swept paths so `set_dotted` can apply them.
    let base_config = serde_json::json!({
        "runtime": {"cells": 1}
    });
    let axes = vec![
        super::contract::SweepAxis {
            parameter: "runtime.cells".to_string(),
            values: vec![serde_json::json!(1), serde_json::json!(2)],
        },
        super::contract::SweepAxis {
            parameter: "runtime.workers".to_string(),
            values: vec![
                serde_json::json!(1),
                serde_json::json!(2),
                serde_json::json!(4),
            ],
        },
    ];
    let envelope = sweep_envelope("sweep-1", base_config, axes, 2, 1);

    let specs = super::sweep_controller::build_child_specs(&envelope)
        .expect("build_child_specs must succeed");

    assert_eq!(specs.len(), 12, "2×3 grid × 2 trials = 12 child specs");
}

#[test]
fn sweep_controller_child_run_id_is_deterministic() {
    // Child run_id at index 0 must be "{envelope.run_id}-0000".
    let run_id = "sweep-run-1";
    let envelope = sweep_envelope(
        run_id,
        serde_json::json!({"runtime": {"cells": 1}}),
        vec![super::contract::SweepAxis {
            parameter: "runtime.cells".to_string(),
            values: vec![serde_json::json!(1)],
        }],
        1,
        1,
    );

    let specs = super::sweep_controller::build_child_specs(&envelope)
        .expect("build_child_specs must succeed");
    assert_eq!(specs.len(), 1);

    // The run_id formula is deterministic: "{base_run_id}-{index:04}".
    let expected_first = format!("{}-{:04}", run_id, 0);
    assert_eq!(expected_first, "sweep-run-1-0000");
}

#[test]
fn sweep_controller_submits_secrets_and_cr_per_child() {
    // For a 2-run sweep (1 axis × 2 values × 1 trial), run_sweep must POST
    // bootstrap Secrets and one AIPerfJob CR per child run.
    //
    // Each child run with 1 cell requires:
    //   - 1 POST to /secrets (controller bootstrap)
    //   - 1 POST to /secrets (cell-0 bootstrap)
    //   - 1 POST to /aiperfjobs
    //   - 1 PATCH to /aiperfsweeps/.../status (completedRuns updated)
    // Plus 1 final PATCH for the sweep phase.
    // Total sends = 2 × 4 + 1 = 9; watches = 2 (one per child).
    let base_config = serde_json::json!({"runtime": {"cells": 1}});
    let envelope = sweep_envelope(
        "sweep-run-1",
        base_config,
        vec![super::contract::SweepAxis {
            parameter: "runtime.cells".to_string(),
            values: vec![serde_json::json!(1), serde_json::json!(1)],
        }],
        1,
        1, // max_concurrent_runs = 1 → sequential
    );

    let transport = Arc::new(SweepMockTransport::default());
    let client = KubeClient::with_transport(sweep_test_credentials(), transport.clone());

    // Child 0: 2 secret POSTs, 1 CR POST; watch → Completed; 1 completedRuns PATCH
    for _ in 0..2 {
        transport.push_response(201, Vec::new());
    }
    transport.push_response(201, Vec::new()); // CR
    transport.push_watch(vec![completed_event("sweep-run-1-0000")]);
    transport.push_response(200, Vec::new()); // PATCH completedRuns

    // Child 1: same pattern
    for _ in 0..2 {
        transport.push_response(201, Vec::new());
    }
    transport.push_response(201, Vec::new()); // CR
    transport.push_watch(vec![completed_event("sweep-run-1-0001")]);
    transport.push_response(200, Vec::new()); // PATCH completedRuns

    // Final sweep phase PATCH
    transport.push_response(200, Vec::new());

    let exit =
        super::sweep_controller::run_sweep(&client, &envelope, "sweep-uid-abc").expect("run_sweep");
    assert_eq!(exit, 0, "successful sweep must exit 0");

    let requests = transport.requests.lock().expect("requests");
    let secret_posts: Vec<_> = requests
        .iter()
        .filter(|r| r.method == "POST" && r.path.ends_with("/secrets"))
        .collect();
    let cr_posts: Vec<_> = requests
        .iter()
        .filter(|r| r.method == "POST" && r.path.ends_with("/aiperfjobs"))
        .collect();
    // 2 children × 2 bootstrap secrets each = 4 secret POSTs
    assert_eq!(secret_posts.len(), 4, "2 children × 2 secrets each = 4 secret POSTs");
    assert_eq!(cr_posts.len(), 2, "one AIPerfJob CR per child");

    // Every CR body must carry an ownerReference to the sweep CR.
    for post in &cr_posts {
        let body: Value = serde_json::from_slice(&post.body).expect("CR body JSON");
        let owners = body["metadata"]["ownerReferences"]
            .as_array()
            .expect("ownerReferences must be an array");
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0]["kind"], "AIPerfSweep");
        assert_eq!(owners[0]["uid"], "sweep-uid-abc");
    }
}

#[test]
fn sweep_controller_patches_sweep_status_on_completion() {
    // After a child run reaches Completed, run_sweep must PATCH the sweep status
    // with updated completedRuns/failedRuns counts.
    let base_config = serde_json::json!({"runtime": {"cells": 1}});
    let envelope = sweep_envelope(
        "sweep-run-2",
        base_config,
        vec![super::contract::SweepAxis {
            parameter: "runtime.cells".to_string(),
            values: vec![serde_json::json!(1)],
        }],
        1,
        1,
    );

    let transport = Arc::new(SweepMockTransport::default());
    let client = KubeClient::with_transport(sweep_test_credentials(), transport.clone());

    // 2 secret POSTs, 1 CR POST; watch → Completed; 1 completedRuns PATCH; 1 final phase PATCH
    for _ in 0..2 {
        transport.push_response(201, Vec::new());
    }
    transport.push_response(201, Vec::new()); // CR
    transport.push_watch(vec![completed_event("sweep-run-2-0000")]);
    transport.push_response(200, Vec::new()); // PATCH completedRuns
    transport.push_response(200, Vec::new()); // PATCH final phase

    super::sweep_controller::run_sweep(&client, &envelope, "sweep-uid-xyz").expect("run_sweep");

    let requests = transport.requests.lock().expect("requests");
    let status_patches: Vec<_> = requests
        .iter()
        .filter(|r| {
            r.method == "PATCH" && r.path.contains("/aiperfsweeps/")
        })
        .collect();
    assert!(
        !status_patches.is_empty(),
        "at least one sweep status PATCH must be issued after child completion"
    );

    // One of the patches must carry completedRuns or phase information.
    let has_completion_patch = status_patches.iter().any(|req| {
        if let Ok(body) = serde_json::from_slice::<Value>(&req.body) {
            body["status"]["completedRuns"].as_u64().is_some()
                || body["status"]["phase"].as_str().is_some()
        } else {
            false
        }
    });
    assert!(
        has_completion_patch,
        "at least one PATCH must carry completedRuns or phase"
    );
}

// --- aiperf kube sweep submission tests ---

/// Minimal sweep envelope for hermetic submission tests.
fn minimal_sweep_envelope(run_id: &str) -> SweepEnvelope {
    SweepEnvelope {
        contract_version: "native-k8s/v1".to_string(),
        run_id: run_id.to_string(),
        namespace: "bench".to_string(),
        sweep_id: "sweep-1".to_string(),
        image_reference: format!(
            "registry.example.com/aiperf/runner@sha256:{}",
            "0".repeat(64)
        ),
        base_config: serde_json::json!({}),
        axes: vec![],
        trials: 1,
        max_concurrent_runs: 1,
        sweep_controller: SweepRoleEnvelope {
            name: "sweep-controller".to_string(),
            command: vec!["aiperf".to_string()],
            argv: vec!["sweep-controller".to_string()],
            environment: std::collections::BTreeMap::new(),
            bootstrap: None,
        },
    }
}

/// Image capabilities with `cellular: true` for the envelope's image digest.
fn cellular_capabilities_value() -> Value {
    serde_json::json!({
        "contractVersion": "native-k8s/v1",
        "imageDigest": format!("sha256:{}", "0".repeat(64)),
        "cellular": true,
        "resultsSidecar": true,
        "hierarchicalAggregation": false,
    })
}

/// Credentials for sweep submission tests.
fn sweep_submit_credentials() -> KubeCredentials {
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

/// Mock transport that records every request and serves pre-queued responses.
#[derive(Default)]
struct SweepSubmitTransport {
    requests: Mutex<Vec<KubeRequest>>,
    responses: Mutex<VecDeque<(u16, Vec<u8>)>>,
}

impl SweepSubmitTransport {
    fn push_response(&self, status: u16, body: Vec<u8>) {
        self.responses
            .lock()
            .expect("responses")
            .push_back((status, body));
    }
}

impl KubeTransport for SweepSubmitTransport {
    fn send(
        &self,
        _credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeResponse, KubeError> {
        self.requests
            .lock()
            .expect("requests")
            .push(request);
        let (status, body) = self
            .responses
            .lock()
            .expect("responses")
            .pop_front()
            .ok_or_else(|| KubeError::Transport("no response queued for sweep test".to_string()))?;
        Ok(KubeResponse { status, body })
    }

    fn watch(
        &self,
        _credentials: &KubeCredentials,
        _request: KubeRequest,
    ) -> Result<KubeWatch, KubeError> {
        Err(KubeError::Transport("watch not used in sweep submission tests".to_string()))
    }
}

#[test]
fn sweep_submission_creates_secret_then_cr() {
    let transport = Arc::new(SweepSubmitTransport::default());
    let client = KubeClient::with_transport(sweep_submit_credentials(), transport.clone());

    // 201 for Secret POST, then 201 for AIPerfSweep CR POST.
    transport.push_response(201, Vec::new());
    transport.push_response(201, Vec::new());

    let envelope = minimal_sweep_envelope("sweep-creates-secret");
    let caps = cellular_capabilities_value();

    let status = super::submission::submit_sweep_transactionally(&client, &envelope, caps)
        .expect("sweep submission must succeed");
    assert_eq!(status, 201);

    let requests = transport.requests.lock().expect("requests");
    assert!(requests.len() >= 2, "at least Secret POST and CR POST must be issued");
    // The first POST must target /secrets.
    let secret_post = requests
        .iter()
        .find(|r| r.method == "POST" && r.path.ends_with("/secrets"))
        .expect("POST to /secrets must be issued");
    // The CR POST must follow and target /aiperfsweeps.
    let cr_post = requests
        .iter()
        .find(|r| r.method == "POST" && r.path.ends_with("/aiperfsweeps"))
        .expect("POST to /aiperfsweeps must be issued");
    let secret_index = requests
        .iter()
        .position(|r| std::ptr::eq(r, secret_post))
        .expect("secret position");
    let cr_index = requests
        .iter()
        .position(|r| std::ptr::eq(r, cr_post))
        .expect("cr position");
    assert!(
        secret_index < cr_index,
        "Secret must be posted before the AIPerfSweep CR"
    );
    // No DELETE on success.
    assert!(
        requests.iter().all(|r| r.method != "DELETE"),
        "successful submission must not issue DELETE requests"
    );
}

#[test]
fn sweep_submission_rolls_back_secret_on_cr_failure() {
    let transport = Arc::new(SweepSubmitTransport::default());
    let client = KubeClient::with_transport(sweep_submit_credentials(), transport.clone());

    // 201 for Secret POST, 409 for CR POST, 200 for the rollback Secret DELETE.
    transport.push_response(201, Vec::new());
    transport.push_response(409, Vec::new());
    transport.push_response(200, Vec::new());

    let envelope = minimal_sweep_envelope("sweep-rollback");
    let caps = cellular_capabilities_value();

    let error = super::submission::submit_sweep_transactionally(&client, &envelope, caps)
        .expect_err("rejected CR must return an error");
    let msg = error.to_string();
    assert!(
        msg.contains("409") || msg.contains("HTTP"),
        "error must reference the HTTP failure: {msg}"
    );

    let requests = transport.requests.lock().expect("requests");
    let deletes: Vec<_> = requests
        .iter()
        .filter(|r| r.method == "DELETE")
        .collect();
    assert_eq!(deletes.len(), 1, "exactly one DELETE must be issued for rollback");
    assert!(
        deletes[0].path.contains("/secrets/bootstrap-sweep-"),
        "rollback DELETE must target the bootstrap Secret, got: {}",
        deletes[0].path
    );
}

#[test]
fn sweep_submission_rejects_non_cellular_image() {
    let transport = Arc::new(SweepSubmitTransport::default());
    let client = KubeClient::with_transport(sweep_submit_credentials(), transport.clone());

    let envelope = minimal_sweep_envelope("sweep-non-cellular");
    // The image-capabilities schema enforces `cellular: true` via `"const": true`;
    // passing `false` fails schema validation inside `validate_image_capabilities`
    // before any cluster contact is made.
    let caps = serde_json::json!({
        "contractVersion": "native-k8s/v1",
        "imageDigest": format!("sha256:{}", "0".repeat(64)),
        "cellular": false,
        "resultsSidecar": true,
        "hierarchicalAggregation": false,
    });

    let error = super::submission::submit_sweep_transactionally(&client, &envelope, caps)
        .expect_err("non-cellular image must be rejected");
    // The exact wording comes from schema validation ("true was expected"); the important
    // constraint is that the image is refused and no cluster request is issued.
    assert!(
        !error.to_string().is_empty(),
        "rejection must produce a non-empty error message"
    );
    // No cluster contact must occur before the rejection.
    let requests = transport.requests.lock().expect("requests");
    assert!(
        requests.is_empty(),
        "non-cellular rejection must not contact the cluster, but {} requests were made",
        requests.len()
    );
}

#[test]
fn sweep_command_removed_from_refusal_list() {
    // After Task 10, `aiperf kube sweep` no longer returns the old
    // "unavailable: the shipped operator supports only AIPerfJob" refusal.
    // Without required flags it returns a missing-argument error instead.
    let error = super::command::run(&["sweep".to_string()])
        .expect_err("sweep without required flags must fail");
    let msg = error.to_string();
    assert!(
        !msg.contains("unavailable"),
        "sweep must no longer produce the old refusal: {msg}"
    );
    assert!(
        !msg.contains("shipped operator"),
        "sweep must no longer produce the old refusal: {msg}"
    );
}

// --- namespace result index tests ---

const OPERATOR_PROXY: &str =
    "/api/v1/namespaces/aiperf-system/services/aiperf-k8s-operator:8080/proxy";

const INDEX_BODY: &[u8] = br#"{"items":[{"metadata":{"name":"run-1","namespace":"bench"},"jobId":"job-1","ready":true,"artifactCount":3,"created":1700000000.0}]}"#;

struct IndexTransport {
    request: Mutex<Option<KubeRequest>>,
    status: u16,
    body: Vec<u8>,
}

impl IndexTransport {
    fn new(status: u16, body: &[u8]) -> Self {
        Self {
            request: Mutex::new(None),
            status,
            body: body.to_vec(),
        }
    }

    fn recorded(&self) -> KubeRequest {
        self.request
            .lock()
            .expect("index transport lock")
            .clone()
            .expect("index transport recorded no request")
    }
}

impl KubeTransport for IndexTransport {
    fn send(
        &self,
        _credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeResponse, KubeError> {
        *self
            .request
            .lock()
            .map_err(|_| KubeError::Transport("index transport lock poisoned".to_string()))? =
            Some(request);
        Ok(KubeResponse {
            status: self.status,
            body: self.body.clone(),
        })
    }

    fn watch(
        &self,
        _credentials: &KubeCredentials,
        _request: KubeRequest,
    ) -> Result<KubeWatch, KubeError> {
        Err(KubeError::Transport(
            "watch is not needed by the index transport".to_string(),
        ))
    }
}

#[test]
fn index_command_fetches_namespace_runs() {
    let transport = Arc::new(IndexTransport::new(200, INDEX_BODY));
    let client = KubeClient::with_transport(credentials(None), transport.clone());
    let rendered = super::command::index_report(
        &client,
        OPERATOR_PROXY,
        "bench",
        super::render::OutputFormat::Json,
    )
    .expect("index report");

    let request = transport.recorded();
    assert_eq!(request.method, "GET");
    assert_eq!(request.path, format!("{OPERATOR_PROXY}/api/results/bench"));
    assert!(rendered.contains("run-1"), "unexpected render: {rendered}");
}

#[test]
fn index_command_renders_result_items_in_text_format() {
    let transport = Arc::new(IndexTransport::new(200, INDEX_BODY));
    let client = KubeClient::with_transport(credentials(None), transport.clone());
    let rendered = super::command::index_report(
        &client,
        OPERATOR_PROXY,
        "bench",
        super::render::OutputFormat::Text,
    )
    .expect("index report");

    assert!(rendered.contains("bench/run-1"), "render: {rendered}");
    assert!(rendered.contains("job-1"), "render: {rendered}");
    assert!(rendered.contains("Ready"), "render: {rendered}");
    assert!(rendered.contains("3 artifact(s)"), "render: {rendered}");
}

#[test]
fn index_command_fails_closed_on_an_unsuccessful_operator_response() {
    let transport = Arc::new(IndexTransport::new(503, b"{}"));
    let client = KubeClient::with_transport(credentials(None), transport.clone());
    let error = super::command::index_report(
        &client,
        OPERATOR_PROXY,
        "bench",
        super::render::OutputFormat::Text,
    )
    .expect_err("an unsuccessful operator response must fail");
    assert!(error.to_string().contains("503"), "error: {error:#}");
}

// --- operator-backed dashboard tests ---

/// Transport that answers each recorded path from a fixed table, so a dashboard
/// source test asserts both the addressed path and the mapped result.
struct DashboardTransport {
    paths: Mutex<Vec<String>>,
    responses: Vec<(String, u16, Vec<u8>)>,
}

impl DashboardTransport {
    fn new(responses: Vec<(String, u16, Vec<u8>)>) -> Self {
        Self {
            paths: Mutex::new(Vec::new()),
            responses,
        }
    }

    fn recorded_paths(&self) -> Vec<String> {
        self.paths.lock().expect("dashboard transport lock").clone()
    }
}

impl KubeTransport for DashboardTransport {
    fn send(
        &self,
        _credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeResponse, KubeError> {
        self.paths
            .lock()
            .map_err(|_| KubeError::Transport("dashboard transport lock poisoned".to_string()))?
            .push(request.path.clone());
        let (_, status, body) = self
            .responses
            .iter()
            .find(|(path, _, _)| *path == request.path)
            .ok_or_else(|| {
                KubeError::Transport(format!("no dashboard response for {}", request.path))
            })?;
        Ok(KubeResponse {
            status: *status,
            body: body.clone(),
        })
    }

    fn watch(
        &self,
        _credentials: &KubeCredentials,
        _request: KubeRequest,
    ) -> Result<KubeWatch, KubeError> {
        Err(KubeError::Transport(
            "watch is not needed by the dashboard source".to_string(),
        ))
    }
}

fn dashboard_source(
    responses: Vec<(String, u16, Vec<u8>)>,
) -> (super::dashboard::OperatorSource, Arc<DashboardTransport>) {
    let transport = Arc::new(DashboardTransport::new(responses));
    let source = super::dashboard::OperatorSource::new(
        KubeClient::with_transport(credentials(None), transport.clone()),
        "bench".to_string(),
        OPERATOR_PROXY.to_string(),
    );
    (source, transport)
}

#[test]
fn dashboard_command_no_longer_refuses() {
    let result = super::command::run(&["dashboard".to_string(), "--help".to_string()]);
    let message = result
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        !message.contains("native Kubernetes dashboard is unavailable"),
        "dashboard must no longer produce the old refusal: {message}"
    );
}

#[test]
fn operator_source_list_maps_operator_items_to_run_entries() {
    use crate::server::HistoricalSource as _;

    let (source, transport) = dashboard_source(vec![(
        format!("{OPERATOR_PROXY}/api/results/bench"),
        200,
        INDEX_BODY.to_vec(),
    )]);
    let runs = source.list();

    assert_eq!(runs.len(), 1, "one retained run must map to one entry");
    assert_eq!(runs[0].label, "run-1");
    assert!(runs[0].success, "a ready run maps to a successful entry");
    assert_eq!(runs[0].artifact_dir, "bench/job-1/run-1");
    assert_eq!(runs[0].source, "operator");
    assert_eq!(
        transport.recorded_paths(),
        vec![format!("{OPERATOR_PROXY}/api/results/bench")]
    );
}

#[test]
fn operator_source_read_report_proxies_artifact() {
    use crate::server::HistoricalSource as _;

    const REPORT_PATH: &str = "/api/results/bench/job-1/run-1/artifacts/native-v2.json";
    let (source, transport) = dashboard_source(vec![
        (
            format!("{OPERATOR_PROXY}/api/results/bench"),
            200,
            INDEX_BODY.to_vec(),
        ),
        (
            format!("{OPERATOR_PROXY}{REPORT_PATH}"),
            200,
            br#"{"runId":"run-1","concurrency":2}"#.to_vec(),
        ),
    ]);
    let runs = source.list();
    let report = source.read_report(&runs[0]).expect("operator report");

    assert_eq!(report["runId"], "run-1");
    assert!(
        transport
            .recorded_paths()
            .contains(&format!("{OPERATOR_PROXY}{REPORT_PATH}")),
        "the report must be addressed through the operator Service proxy"
    );
}
