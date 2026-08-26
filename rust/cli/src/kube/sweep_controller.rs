// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! In-cluster sweep-controller role.
//!
//! Dispatched by the operator's JobSet when it provisions an `AIPerfSweep` CR.
//! Reads the mounted sweep envelope, expands the parameter grid, and manages
//! child `AIPerfJob` lifecycle until all runs reach a terminal state.

use std::path::Path;
use std::time::Duration;

use aiperf_runtime::engine::cellular_bootstrap::{CellularRole, mint_deployment_material};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use super::auth::in_cluster_credentials;
use super::client::{AIPERF_GROUP, AIPERF_VERSION, KubeClient, KubeWatchPoll};
use super::contract::{
    BootstrapReference, CellBootstrapReference, CONTRACT_VERSION, ControllerEnvelope,
    NamedReference, NativeK8sRole, RoleEnvelope, SweepEnvelope, validate_sweep_envelope,
};
use super::manifest;
use super::projection::{BootstrapDigests, build_controller_envelope};
use super::submission::BootstrapMaterialTarget;
use crate::model::BenchmarkConfig;
use crate::sweep::plan::{Sweep, SweepAxis, build_benchmark_plan};

const SWEEP_ENVELOPE_PATH: &str = "/etc/aiperf/config/sweep-envelope.json";
const SA_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const SA_CA_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";
/// API plural for AIPerfSweep custom resources.
pub(crate) const AIPERFSWEEPS_PLURAL: &str = "aiperfsweeps";
const WATCH_POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// Entry point dispatched from `dispatch.rs` for the `sweep-controller` command.
pub fn run() -> anyhow::Result<i32> {
    // Read the sweep envelope from the operator-mounted ConfigMap.
    let envelope_bytes = std::fs::read(SWEEP_ENVELOPE_PATH).map_err(|e| {
        anyhow::anyhow!("failed to read sweep envelope {SWEEP_ENVELOPE_PATH}: {e}")
    })?;
    let envelope_value: Value = serde_json::from_slice(&envelope_bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode sweep envelope: {e}"))?;
    let envelope = validate_sweep_envelope(envelope_value)?;

    // Build an in-cluster Kubernetes client from the mounted service-account material.
    let host = std::env::var("KUBERNETES_SERVICE_HOST")
        .map_err(|_| anyhow::anyhow!("KUBERNETES_SERVICE_HOST is not set"))?;
    let port = std::env::var("KUBERNETES_SERVICE_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(443);
    let credentials = in_cluster_credentials(
        host,
        port,
        Path::new(SA_TOKEN_PATH),
        Path::new(SA_CA_PATH),
    )?;
    let client = KubeClient::from_credentials(credentials)?;

    // Resolve the sweep CR UID for child owner references.
    let sweep_uid = get_sweep_uid(&client, &envelope.namespace, &envelope.run_id)?;

    run_sweep(&client, &envelope, &sweep_uid)
}

/// Resolve the AIPerfSweep CR UID so child CRs can declare it as their owner.
fn get_sweep_uid(
    client: &KubeClient,
    namespace: &str,
    sweep_name: &str,
) -> anyhow::Result<String> {
    let path = format!(
        "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/{AIPERFSWEEPS_PLURAL}/{sweep_name}"
    );
    let response = client.execute("GET", &path, "application/json", Vec::new())?;
    if !response.is_success() {
        anyhow::bail!("GET sweep CR returned HTTP {}", response.status);
    }
    let cr: Value = serde_json::from_slice(&response.body)
        .map_err(|e| anyhow::anyhow!("failed to decode sweep CR response: {e}"))?;
    cr["metadata"]["uid"]
        .as_str()
        .filter(|uid| !uid.is_empty())
        .map(String::from)
        .ok_or_else(|| anyhow::anyhow!("sweep CR response omits metadata.uid"))
}

/// Orchestrate all child runs.
///
/// Exposed with `pub(crate)` visibility so hermetic tests can inject a mock
/// `KubeClient` via `KubeClient::with_transport`.
pub(crate) fn run_sweep(
    client: &KubeClient,
    envelope: &SweepEnvelope,
    sweep_uid: &str,
) -> anyhow::Result<i32> {
    let child_configs = build_child_specs(envelope)?;
    let namespace = &envelope.namespace;
    let sweep_name = &envelope.run_id;
    let max_concurrent = envelope.max_concurrent_runs.max(1) as usize;
    let total = child_configs.len();

    info!(
        sweep_id = %envelope.sweep_id,
        total_runs = total,
        max_concurrent,
        "sweep-controller starting"
    );

    // Sliding-window execution: issue up to `max_concurrent` children, drain one
    // before issuing the next when the window is full.
    let mut running: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut completed_count = 0u32;
    let mut failed_count = 0u32;

    for (index, base_config) in child_configs.into_iter().enumerate() {
        // Drain one slot before issuing when the window is full.
        while running.len() >= max_concurrent {
            // SAFETY: the `len() >= max_concurrent` guard above ensures the
            // queue is non-empty before we pop.
            let waiting = running.pop_front().expect("non-empty running queue");
            let succeeded = wait_for_child_completion(client, namespace, &waiting)?;
            if succeeded {
                completed_count += 1;
            } else {
                failed_count += 1;
            }
            patch_sweep_run_counts(client, namespace, sweep_name, completed_count, failed_count)?;
        }

        let run_id = format!("{}-{:04}", envelope.run_id, index);
        debug!(run_id = %run_id, index, "submitting child run");
        submit_child_run(
            client,
            envelope,
            &run_id,
            sweep_name,
            sweep_uid,
            &base_config,
        )?;
        running.push_back(run_id);
    }

    // Drain all remaining in-flight children.
    while let Some(waiting) = running.pop_front() {
        let succeeded = wait_for_child_completion(client, namespace, &waiting)?;
        if succeeded {
            completed_count += 1;
        } else {
            failed_count += 1;
        }
        patch_sweep_run_counts(client, namespace, sweep_name, completed_count, failed_count)?;
    }

    // Patch the final sweep phase.
    let is_all_failed = total > 0 && failed_count == total as u32;
    let final_phase = if is_all_failed { "Failed" } else { "Completed" };
    patch_sweep_phase(client, namespace, sweep_name, final_phase)?;

    info!(
        sweep_id = %envelope.sweep_id,
        completed = completed_count,
        failed = failed_count,
        phase = final_phase,
        "sweep-controller done"
    );

    Ok(if is_all_failed { 1 } else { 0 })
}

/// Expand the sweep envelope into one `BenchmarkConfig` per child run.
///
/// Grid combinations are produced first, then each combination is repeated
/// `trials` times, yielding `combinations × trials` child configs.
pub(crate) fn build_child_specs(
    envelope: &SweepEnvelope,
) -> anyhow::Result<Vec<BenchmarkConfig>> {
    let base_config: BenchmarkConfig =
        serde_json::from_value(envelope.base_config.clone())
            .map_err(|e| anyhow::anyhow!("base_config is not a valid BenchmarkConfig: {e}"))?;

    // Convert contract axes (parameter + values) to plan axes (path + seg + values).
    let axes: Vec<SweepAxis> = envelope
        .axes
        .iter()
        .map(|axis| {
            let seg = axis
                .parameter
                .rsplit('.')
                .next()
                .unwrap_or(axis.parameter.as_str())
                .to_string();
            SweepAxis {
                path: axis.parameter.clone(),
                seg,
                values: axis.values.clone(),
            }
        })
        .collect();

    let sweep = Sweep::grid(axes);
    let base_runs = build_benchmark_plan(&base_config, &sweep, None)?;

    let trials = envelope.trials.max(1) as usize;
    let mut specs = Vec::with_capacity(base_runs.len() * trials);
    for run in base_runs {
        for _ in 0..trials {
            specs.push(run.cfg.clone());
        }
    }
    Ok(specs)
}

/// Mint bootstrap material, create Secrets, build a child `ControllerEnvelope`,
/// and POST the child `AIPerfJob` CR with an owner reference to the sweep CR.
fn submit_child_run(
    client: &KubeClient,
    sweep_envelope: &SweepEnvelope,
    run_id: &str,
    sweep_name: &str,
    sweep_uid: &str,
    base_config: &BenchmarkConfig,
) -> anyhow::Result<()> {
    let namespace = &sweep_envelope.namespace;
    let cells = base_config
        .runtime
        .as_ref()
        .map(|r| r.cells)
        .unwrap_or(1);

    // Mint fresh bootstrap material for this child run.
    let roles: Vec<CellularRole> = (0..cells).map(CellularRole::Cell).collect();
    let material = mint_deployment_material(&roles).map_err(|e| {
        anyhow::anyhow!("failed to mint bootstrap material for {run_id}: {e}")
    })?;

    // Create the controller bootstrap Secret and record its digest.
    let controller_secret_name = format!("bootstrap-{run_id}-ctrl");
    let controller_digest = post_bootstrap_secret(
        client,
        namespace,
        run_id,
        &controller_secret_name,
        &material.controller,
        "controller",
    )?;

    // Create one cell bootstrap Secret per cell and build the cell-bootstrap list.
    let mut minted = BootstrapDigests::new();
    minted.insert(
        BootstrapMaterialTarget::Role(NativeK8sRole::Controller),
        controller_digest,
    );

    let base_cell_bootstraps: Vec<CellBootstrapReference> = (0..cells)
        .map(|cell_id| CellBootstrapReference {
            cell_id,
            secret_name: format!("bootstrap-{run_id}-cell-{cell_id}"),
            role: NativeK8sRole::Cell,
            mount_path: "/bootstrap".to_string(),
            sha256: "0".repeat(64), // placeholder; replaced by build_controller_envelope
        })
        .collect();

    for cell_bootstrap in &base_cell_bootstraps {
        let cell_id = cell_bootstrap.cell_id;
        let bytes = material
            .roles
            .get(&CellularRole::Cell(cell_id))
            .ok_or_else(|| anyhow::anyhow!("minted material missing cell {cell_id}"))?;
        let digest = post_bootstrap_secret(
            client,
            namespace,
            run_id,
            &cell_bootstrap.secret_name,
            bytes,
            "cell",
        )?;
        minted.insert(BootstrapMaterialTarget::Cell(cell_id), digest);
    }

    // Build the base envelope then project with the actual minted digests.
    let (_, image_digest) = sweep_envelope
        .image_reference
        .rsplit_once('@')
        .ok_or_else(|| anyhow::anyhow!("sweep imageReference is not digest-qualified"))?;

    let base_envelope = ControllerEnvelope {
        contract_version: CONTRACT_VERSION.to_string(),
        run_id: run_id.to_string(),
        namespace: namespace.clone(),
        job_id: run_id.to_string(),
        image_digest: image_digest.to_string(),
        image_reference: sweep_envelope.image_reference.clone(),
        cells,
        artifact_root: "/results".to_string(),
        config_ref: NamedReference {
            name: run_id.to_string(),
            sha256: "0".repeat(64),
        },
        controller_address: "tcp://aiperf-controller-svc:9500".to_string(),
        roles: vec![
            RoleEnvelope {
                name: NativeK8sRole::Controller,
                command: vec!["aiperf".to_string()],
                argv: vec!["controller".to_string()],
                environment: std::collections::BTreeMap::new(),
                bootstrap: Some(BootstrapReference {
                    secret_name: controller_secret_name,
                    role: NativeK8sRole::Controller,
                    mount_path: "/bootstrap".to_string(),
                    sha256: "0".repeat(64), // replaced by build_controller_envelope
                }),
            },
            RoleEnvelope {
                name: NativeK8sRole::Cell,
                command: vec!["aiperf".to_string()],
                argv: vec!["cell".to_string()],
                environment: std::collections::BTreeMap::new(),
                bootstrap: None,
            },
            RoleEnvelope {
                name: NativeK8sRole::ResultsSidecar,
                command: vec!["aiperf".to_string()],
                argv: vec!["results-sidecar".to_string()],
                environment: std::collections::BTreeMap::new(),
                bootstrap: None,
            },
        ],
        cell_bootstraps: base_cell_bootstraps,
    };

    let child_envelope = build_controller_envelope(&base_envelope, &minted)?;

    // Project to an AIPerfJob CR body and inject the sweep owner reference.
    let mut cr_body =
        manifest::project(&child_envelope).map_err(anyhow::Error::from)?;
    cr_body["metadata"]["ownerReferences"] = json!([{
        "apiVersion": format!("{AIPERF_GROUP}/{AIPERF_VERSION}"),
        "kind": "AIPerfSweep",
        "name": sweep_name,
        "uid": sweep_uid,
        "controller": true,
        "blockOwnerDeletion": true,
    }]);

    let cr_bytes = serde_json::to_vec(&cr_body)
        .map_err(|e| anyhow::anyhow!("failed to serialize child AIPerfJob: {e}"))?;
    let cr_path = format!(
        "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/aiperfjobs"
    );
    let response = client.execute("POST", &cr_path, "application/json", cr_bytes)?;
    if !response.is_success() {
        anyhow::bail!(
            "AIPerfJob creation for {run_id} returned HTTP {}",
            response.status
        );
    }
    Ok(())
}

/// POST one immutable bootstrap Secret and return the SHA-256 hex digest of its bytes.
fn post_bootstrap_secret(
    client: &KubeClient,
    namespace: &str,
    run_id: &str,
    secret_name: &str,
    bytes: &[u8],
    role: &str,
) -> anyhow::Result<String> {
    let digest = format!("{:x}", Sha256::digest(bytes));
    let body = json!({
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "immutable": true,
        "metadata": {
            "name": secret_name,
            "namespace": namespace,
            "labels": {
                "aiperf.nvidia.com/run-id": run_id,
                "aiperf.nvidia.com/role": role,
            },
            "annotations": {"aiperf.nvidia.com/sha256": digest},
        },
        "data": {"bootstrap": BASE64.encode(bytes)},
    });
    let body_bytes = serde_json::to_vec(&body)
        .map_err(|e| anyhow::anyhow!("failed to serialize bootstrap Secret {secret_name}: {e}"))?;
    let status = client.request(
        "POST",
        &format!("/api/v1/namespaces/{namespace}/secrets"),
        "application/json",
        body_bytes,
    )?;
    if !(200..300).contains(&status) {
        anyhow::bail!(
            "bootstrap Secret {secret_name} creation returned HTTP {status}"
        );
    }
    Ok(digest)
}

/// Poll the child AIPerfJob watch stream until it reaches `Completed` or `Failed`.
///
/// Returns `true` for `Completed` and `false` for `Failed`. Reconnects the
/// watch on clean EOF so a brief lapse in the API server does not abort the sweep.
fn wait_for_child_completion(
    client: &KubeClient,
    namespace: &str,
    job_id: &str,
) -> anyhow::Result<bool> {
    let watch_path = format!(
        "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/aiperfjobs?watch=true&fieldSelector=metadata.name={job_id}"
    );
    loop {
        let watch = client.watch(&watch_path)?;
        loop {
            match watch.poll(WATCH_POLL_TIMEOUT)? {
                KubeWatchPoll::Record(bytes) => {
                    let event: Value =
                        serde_json::from_slice(&bytes).map_err(|e| {
                            anyhow::anyhow!(
                                "invalid watch event for {job_id}: {e}"
                            )
                        })?;
                    match event["object"]["status"]["phase"].as_str() {
                        Some("Completed") => return Ok(true),
                        Some("Failed") => return Ok(false),
                        _ => continue,
                    }
                }
                KubeWatchPoll::Idle => continue,
                KubeWatchPoll::Closed => break, // reconnect on clean EOF
            }
        }
    }
}

/// Patch the sweep `.status` with updated `completedRuns` / `failedRuns` counts.
fn patch_sweep_run_counts(
    client: &KubeClient,
    namespace: &str,
    sweep_name: &str,
    completed: u32,
    failed: u32,
) -> anyhow::Result<()> {
    let path = sweep_status_path(namespace, sweep_name);
    let status = client.merge_patch(
        &path,
        &json!({
            "status": {
                "completedRuns": completed,
                "failedRuns": failed,
            }
        }),
    )?;
    if !(200..300).contains(&status) {
        warn!(
            completed,
            failed,
            http_status = status,
            "sweep run-count status patch did not succeed"
        );
    }
    Ok(())
}

/// Patch the sweep `.status.phase` to the terminal value.
fn patch_sweep_phase(
    client: &KubeClient,
    namespace: &str,
    sweep_name: &str,
    phase: &str,
) -> anyhow::Result<()> {
    let path = sweep_status_path(namespace, sweep_name);
    let status = client.merge_patch(&path, &json!({"status": {"phase": phase}}))?;
    if !(200..300).contains(&status) {
        warn!(
            %phase,
            http_status = status,
            "sweep phase status patch did not succeed"
        );
    }
    Ok(())
}

fn sweep_status_path(namespace: &str, sweep_name: &str) -> String {
    format!(
        "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/{AIPERFSWEEPS_PLURAL}/{sweep_name}/status"
    )
}
