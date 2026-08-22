// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use super::auth::{KubeAuthOptions, KubeCredentials};
use super::client::{KubeClient, KubeRequest, KubeTransport, DEFAULT_REQUEST_DEADLINE, DEFAULT_WATCH_DEADLINE};
use super::contract::{validate_envelope, validate_image_capabilities};
use super::error::KubeError;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../contracts/native-k8s/v1/fixtures/");

fn fixture(name: &str) -> Value {
    let source = std::fs::read_to_string(format!("{FIXTURES}{name}")).expect("fixture read");
    serde_json::from_str(&source).expect("fixture JSON")
}

#[test]
fn accepts_runnable_envelopes() {
    assert_eq!(validate_envelope(fixture("valid-one-cell-envelope.json")).expect("one cell").cells, 1);
    assert_eq!(validate_envelope(fixture("valid-multi-cell-envelope.json")).expect("multi cell").cells, 4);
}

#[test]
fn refuses_role_mismatched_bootstrap() {
    let mut envelope = fixture("valid-one-cell-envelope.json");
    envelope["roles"][0]["bootstrap"]["role"] = Value::String("cell".to_string());
    assert!(matches!(validate_envelope(envelope), Err(KubeError::ContractValidation(_))));
}

#[test]
fn refuses_non_v1_and_unknown_role_or_field() {
    assert!(matches!(validate_envelope(fixture("invalid-version-envelope.json")), Err(KubeError::UnsupportedContractVersion(_))));
    assert!(matches!(validate_envelope(fixture("aggregator-envelope.json")), Err(KubeError::ContractValidation(_))));
    assert!(matches!(validate_envelope(fixture("unknown-field-envelope.json")), Err(KubeError::ContractValidation(_))));
}

#[test]
fn validates_required_image_capabilities_and_digest() {
    let envelope = validate_envelope(fixture("valid-one-cell-envelope.json")).expect("envelope");
    assert!(validate_image_capabilities(fixture("missing-cellular-capability.json"), &envelope.image_digest).is_err());
    assert!(validate_image_capabilities(fixture("image-mismatch-capability.json"), &envelope.image_digest).is_err());
}

#[test]
fn kubeconfig_precedence_is_explicit_then_environment_then_home() {
    let explicit = Path::new("/explicit/config");
    assert_eq!(KubeAuthOptions::kubeconfig_path_from(Some(explicit), Some("/environment/config".into()), Some(PathBuf::from("/home/user"))).expect("explicit"), explicit);
    assert_eq!(KubeAuthOptions::kubeconfig_path_from(None, Some("/environment/config".into()), Some(PathBuf::from("/home/user"))).expect("environment"), PathBuf::from("/environment/config"));
    assert_eq!(KubeAuthOptions::kubeconfig_path_from(None, None, Some(PathBuf::from("/home/user"))).expect("home"), PathBuf::from("/home/user/.kube/config"));
}

#[test]
fn exec_credential_resolves_token() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let script = directory.path().join("credential.sh");
    std::fs::write(&script, "#!/bin/sh\nprintf '%s' '{\"status\":{\"token\":\"exec-token\"}}'\n").expect("script write");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).expect("script permissions");
    }
    let config = format!("apiVersion: v1\ncurrent-context: test\nclusters:\n- name: cluster\n  cluster:\n    server: https://example.test:6443\n    certificate-authority-data: Y2E=\ncontexts:\n- name: test\n  context:\n    cluster: cluster\n    user: user\nusers:\n- name: user\n  user:\n    exec:\n      command: {}\n", script.display());
    let path = directory.path().join("config");
    std::fs::write(&path, config).expect("config write");
    let credentials = KubeAuthOptions { kubeconfig: Some(path), ..Default::default() }.resolve().expect("exec credential");
    assert_eq!(credentials.token.as_deref(), Some("exec-token"));
}

#[test]
fn rejects_invalid_cluster_ca_before_network_io() {
    let credentials = credentials(Some(b"not PEM".to_vec()));
    let client = KubeClient::from_credentials(credentials).expect("client construction");
    assert!(matches!(client.merge_patch("/apis/test", &serde_json::json!({})), Err(KubeError::Tls(_))));
}

#[test]
fn defaults_and_overrides_keep_deadlines_finite() {
    let client = KubeClient::from_credentials(credentials(Some(b"not PEM".to_vec()))).expect("client");
    assert_eq!(client.request_deadline(), DEFAULT_REQUEST_DEADLINE);
    assert_eq!(client.watch_deadline(), DEFAULT_WATCH_DEADLINE);
    assert!(client.with_deadlines(Duration::ZERO, Duration::from_secs(1)).is_err());
}

#[test]
fn merge_patch_constructs_reporter_compatible_request() {
    let transport = Arc::new(RecordingTransport::default());
    let client = KubeClient::with_transport(credentials(None), transport.clone());
    let status = client.merge_patch("/apis/aiperf.nvidia.com/v1alpha1/namespaces/bench/aiperfjobs/job/status", &serde_json::json!({"status":{"phase":"Done"}})).expect("patch");
    assert_eq!(status, 200);
    let request = transport.request.lock().expect("recording lock").clone().expect("request");
    assert_eq!(request.method, "PATCH");
    assert_eq!(request.content_type, "application/merge-patch+json");
    assert_eq!(request.deadline, DEFAULT_REQUEST_DEADLINE);
}

fn credentials(ca_pem: Option<Vec<u8>>) -> KubeCredentials {
    KubeCredentials { host: "127.0.0.1".to_string(), port: 443, server_name: "localhost".to_string(), token: Some("token".to_string()), client_certificate_pem: None, client_key_pem: None, ca_pem, insecure_skip_tls_verify: false }
}

#[derive(Default)]
struct RecordingTransport { request: Mutex<Option<KubeRequest>> }
impl KubeTransport for RecordingTransport {
    fn send(&self, _credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError> {
        *self.request.lock().map_err(|_| KubeError::Transport("recording lock poisoned".to_string()))? = Some(request);
        Ok(200)
    }
    fn watch(&self, credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError> { self.send(credentials, request) }
}
