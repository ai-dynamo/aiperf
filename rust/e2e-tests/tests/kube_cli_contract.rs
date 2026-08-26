// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public `aiperf kube` contract.
//!
//! The hermetic tests drive the real binary without a cluster. The kind tests
//! are `#[ignore]`d so ordinary `cargo test` stays hermetic; CI provisions a
//! cluster and runs them with
//! `cargo test -p aiperf-e2e-tests --test kube_cli_contract -- --ignored`.

mod common;

use std::io::Write;
use std::process::{Command, Output, Stdio};

use common::exec_binary;

/// Every command the native surface owns. Nothing here reaches Python.
const NATIVE_COMMANDS: &[&str] = &[
    "init",
    "validate",
    "profile",
    "sweep",
    "generate",
    "attach",
    "list",
    "logs",
    "results",
    "show",
    "debug",
    "watch",
    "preflight",
    "dashboard",
    "index",
];

fn kube(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(exec_binary())
        .arg("kube")
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run aiperf kube");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn help_lists_every_native_command() {
    let (code, stdout, stderr) = kube(&["--help"]);
    assert_eq!(code, 0, "kube --help failed: {stderr}");
    for command in NATIVE_COMMANDS {
        assert!(
            stdout.contains(command),
            "kube help omits {command}: {stdout}"
        );
    }
}

#[test]
fn no_command_delegates_to_python() {
    let (_, stdout, stderr) = kube(&["--help"]);
    let combined = format!("{stdout}{stderr}").to_lowercase();
    for marker in ["python -m aiperf", "aiperf.entrypoint", "aiperf.cli"] {
        assert!(
            !combined.contains(marker),
            "native kube help references the Python distribution: {marker}"
        );
    }
}

#[test]
fn unknown_commands_fail_closed() {
    let (code, _, stderr) = kube(&["teleport"]);
    assert_ne!(code, 0, "an unknown kube command must fail");
    assert!(
        stderr.contains("unknown native Kubernetes command"),
        "unexpected failure text: {stderr}"
    );
}

#[test]
fn envelope_commands_require_an_envelope() {
    let (code, _, stderr) = kube(&["validate"]);
    assert_ne!(code, 0, "validate without an envelope must fail");
    assert!(!stderr.is_empty(), "failures must explain themselves");
}

#[test]
fn index_reaches_the_operator_instead_of_refusing() {
    let (code, _, stderr) = kube(&["index", "--kubeconfig=/nonexistent"]);
    assert_ne!(code, 0, "index without a reachable cluster must fail");
    assert!(
        !stderr.contains("shipped operator supports only"),
        "index must no longer refuse before cluster access: {stderr}"
    );
}

#[test]
fn dashboard_reaches_the_operator_instead_of_refusing() {
    let (code, stdout, stderr) = kube(&["dashboard", "--help"]);
    assert_eq!(code, 0, "dashboard help failed: {stderr}");
    assert!(
        stdout.contains("aiperf kube dashboard"),
        "dashboard help omits its own usage: {stdout}"
    );

    let (code, _, stderr) = kube(&["dashboard", "--kubeconfig=/nonexistent"]);
    assert_ne!(code, 0, "dashboard without a reachable cluster must fail");
    assert!(
        !stderr.contains("no dashboard upstream is implemented"),
        "dashboard must no longer refuse before cluster access: {stderr}"
    );
}

// requires: the workflow-provisioned kind target and KUBECONFIG
#[test]
#[ignore]
fn kind_native_cli_reaches_the_workflow_provisioned_cluster() {
    let (code, _, stderr) = kube(&["preflight"]);
    assert_eq!(
        code, 0,
        "preflight failed against the live cluster: {stderr}"
    );
    let (code, _, stderr) = kube(&["list", "--namespace", "aiperf-system"]);
    assert_eq!(code, 0, "list failed against the live cluster: {stderr}");
}

// requires: the workflow-provisioned kind target, KUBECONFIG, and the checkout-built operator image
#[test]
#[ignore]
fn kind_results_survive_producer_deletion_and_operator_restart() {
    const OPERATOR_NAMESPACE: &str = "aiperf-system";
    const OPERATOR_DEPLOYMENT: &str = "aiperf-k8s";
    const RESULTS_SERVICE: &str = "aiperf-k8s-operator";
    const RESULTS_CLAIM: &str = "aiperf-k8s-results";
    const PRODUCER_JOB: &str = "aiperf-results-producer";
    const ARTIFACT: &[u8] = b"durable results survived producer deletion\n";
    const UPLOAD_SCRIPT: &str = r#"
import hashlib
import json
import urllib.error
import urllib.request

base = "http://aiperf-k8s-operator.aiperf-system.svc:8080/api/uploads/bench/job-1/run-1"
results = "http://aiperf-k8s-operator.aiperf-system.svc:8080/api/results/bench/job-1/run-1"
artifact = b"durable results survived producer deletion\n"

def status(method, url, body=None, declared_length=None):
    headers = {}
    if body is not None:
        length = len(body) if declared_length is None else declared_length
        headers = {
            "Content-Length": str(length),
            "Content-Type": "application/json" if url.endswith("/manifest") else "application/octet-stream",
            "X-AIPerf-Content-Length": str(length),
            "X-AIPerf-Content-SHA256": hashlib.sha256(body).hexdigest(),
        }
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code

assert status("GET", results + "/manifest") == 409
assert status("PUT", base + "/artifacts/nested/%2E%2E/escape.txt", b"unsafe") == 422
assert status("POST", base + "/manifest", b"", declared_length=1048577) == 413

assert status("PUT", base + "/artifacts/nested/result.txt", artifact) in (200, 201)
assert status("GET", results + "/manifest") == 409
assert status("GET", results + "/artifacts/nested/result.txt") == 404
manifest = json.dumps(
    {
        "contractVersion": "native-k8s/v1",
        "runId": "run-1",
        "ready": True,
        "wasCancelled": False,
        "artifactRoot": "/results",
        "artifacts": [
            {
                "path": "nested/result.txt",
                "sha256": hashlib.sha256(artifact).hexdigest(),
                "bytes": len(artifact),
                "contentType": "application/octet-stream",
            }
        ],
    },
    separators=(",", ":"),
).encode()
assert status("POST", base + "/manifest", manifest) in (200, 201)
"#;

    let operator_image = std::env::var("AIPERF_E2E_OPERATOR_IMAGE")
        .expect("workflow must identify the checkout-built operator image");
    let service = kubectl_json(&[
        "get",
        "service",
        RESULTS_SERVICE,
        "--namespace",
        OPERATOR_NAMESPACE,
    ]);
    assert_eq!(service["spec"]["type"], "ClusterIP");
    assert!(
        service["spec"]["clusterIP"]
            .as_str()
            .is_some_and(|address| !address.is_empty() && address != "None"),
        "results Service must have a reachable ClusterIP: {service}"
    );

    let claim = kubectl_json(&[
        "get",
        "persistentvolumeclaim",
        RESULTS_CLAIM,
        "--namespace",
        OPERATOR_NAMESPACE,
    ]);
    assert_eq!(claim["status"]["phase"], "Bound");
    assert!(
        claim["spec"]["volumeName"]
            .as_str()
            .is_some_and(|name| !name.is_empty()),
        "results claim must be backed by a persistent volume: {claim}"
    );

    let deployment = kubectl_json(&[
        "get",
        "deployment",
        OPERATOR_DEPLOYMENT,
        "--namespace",
        OPERATOR_NAMESPACE,
    ]);
    assert_eq!(
        deployment["spec"]["template"]["spec"]["containers"][0]["image"],
        operator_image
    );
    assert_eq!(
        deployment["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"],
        RESULTS_CLAIM
    );
    let old_operator_uid = operator_pod_uid(OPERATOR_NAMESPACE);

    let producer = serde_json::json!({
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": PRODUCER_JOB,
            "namespace": OPERATOR_NAMESPACE,
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": {"app": PRODUCER_JOB}},
                "spec": {
                    "automountServiceAccountToken": false,
                    "enableServiceLinks": false,
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "producer",
                        "image": operator_image,
                        "imagePullPolicy": "Never",
                        "command": ["python", "-c", UPLOAD_SCRIPT],
                    }],
                },
            },
        },
    });
    kubectl_apply(&producer);
    kubectl_checked(&[
        "wait",
        "--for=condition=complete",
        &format!("job/{PRODUCER_JOB}"),
        "--namespace",
        OPERATOR_NAMESPACE,
        "--timeout=90s",
    ]);

    let all_pods = kubectl_json(&["get", "pods", "--all-namespaces"]);
    let claim_users = all_pods["items"]
        .as_array()
        .expect("pod list items")
        .iter()
        .filter(|pod| {
            pod["spec"]["volumes"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|volume| volume["persistentVolumeClaim"]["claimName"] == RESULTS_CLAIM)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        claim_users.len(),
        1,
        "only the results deployment owns the PVC"
    );
    assert_eq!(
        claim_users[0]["metadata"]["labels"]["app.kubernetes.io/instance"],
        "aiperf-k8s"
    );

    kubectl_checked(&[
        "delete",
        "job",
        PRODUCER_JOB,
        "--namespace",
        OPERATOR_NAMESPACE,
        "--wait=true",
        "--timeout=60s",
    ]);
    let producer_pods = kubectl_json(&[
        "get",
        "pods",
        "--namespace",
        OPERATOR_NAMESPACE,
        "--selector",
        &format!("job-name={PRODUCER_JOB}"),
    ]);
    assert_eq!(producer_pods["items"], serde_json::json!([]));

    kubectl_checked(&[
        "rollout",
        "restart",
        &format!("deployment/{OPERATOR_DEPLOYMENT}"),
        "--namespace",
        OPERATOR_NAMESPACE,
    ]);
    kubectl_checked(&[
        "rollout",
        "status",
        &format!("deployment/{OPERATOR_DEPLOYMENT}"),
        "--namespace",
        OPERATOR_NAMESPACE,
        "--timeout=120s",
    ]);
    assert_ne!(old_operator_uid, operator_pod_uid(OPERATOR_NAMESPACE));

    let destination = tempfile::tempdir().expect("results destination");
    let destination = destination.path().to_str().expect("UTF-8 destination");
    let (code, stdout, stderr) = kube(&[
        "results",
        "job-1",
        "--namespace",
        "bench",
        "--run-id",
        "run-1",
        "--output-directory",
        destination,
    ]);
    assert_eq!(code, 0, "durable results retrieval failed: {stderr}");
    assert!(
        stdout.contains("verified 1 artifacts"),
        "unexpected output: {stdout}"
    );
    assert_eq!(
        std::fs::read(format!("{destination}/nested/result.txt")).expect("downloaded artifact"),
        ARTIFACT
    );
}

fn kubectl_json(args: &[&str]) -> serde_json::Value {
    let output = kubectl_output(args, None, true);
    serde_json::from_slice(&output.stdout).expect("kubectl JSON document")
}

fn kubectl_checked(args: &[&str]) {
    kubectl_output(args, None, false);
}

fn kubectl_apply(document: &serde_json::Value) {
    kubectl_output(
        &["apply", "--filename", "-"],
        Some(serde_json::to_vec(document).expect("serialize Kubernetes document")),
        false,
    );
}

fn kubectl_output(args: &[&str], stdin: Option<Vec<u8>>, json_output: bool) -> Output {
    let mut command = Command::new("kubectl");
    command.args(args);
    if json_output {
        command.args(["--output", "json"]);
    }
    let mut child = command
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn kubectl");
    if let Some(body) = stdin {
        child
            .stdin
            .take()
            .expect("kubectl stdin")
            .write_all(&body)
            .expect("write kubectl input");
    }
    let output = child.wait_with_output().expect("wait for kubectl");
    assert!(
        output.status.success(),
        "kubectl {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn operator_pod_uid(namespace: &str) -> String {
    let pods = kubectl_json(&[
        "get",
        "pods",
        "--namespace",
        namespace,
        "--selector",
        "app.kubernetes.io/name=aiperf-k8s-operator,app.kubernetes.io/instance=aiperf-k8s",
    ]);
    let items = pods["items"].as_array().expect("operator pod list");
    assert_eq!(
        items.len(),
        1,
        "expected one results deployment pod: {pods}"
    );
    items[0]["metadata"]["uid"]
        .as_str()
        .expect("operator pod UID")
        .to_string()
}
