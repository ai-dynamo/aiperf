// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf kube` command surface.

use sha2::{Digest as _, Sha256};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::Parser;

use super::auth::KubeAuthOptions;
use super::client::{KubeClient, KubeWatch, KubeWatchPoll};
use super::contract::{
    BootstrapReference, CONTRACT_VERSION, CellBootstrapReference, ControllerEnvelope,
    NamedReference, NativeK8sRole, RoleEnvelope, validate_envelope, validate_sweep_envelope,
};
use super::error::KubeError;
use super::render::{OutputFormat, render};
use super::results::{ArtifactFetcher, MAX_ARTIFACT_BYTES, download, parse_manifest};
use super::submission::{
    envelope_paths, jobs_path, load_envelope, material_paths, submit_profile_transactionally,
    submit_sweep_transactionally, validate_image_capability_document,
};

/// Maximum bounded reconnects a streaming command performs before failing.
const MAX_WATCH_RECONNECTS: u32 = 5;

const RESULTS_API_PORT: u16 = 8080;
const DEFAULT_OPERATOR_NAMESPACE: &str = "aiperf-system";
const DEFAULT_OPERATOR_SERVICE: &str = "aiperf-k8s-operator";

const COMMANDS: &[&str] = &[
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

/// Run a Kubernetes command without delegating to the Python distribution.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let Some(command) = args.first().map(String::as_str) else {
        return help();
    };
    if command == "--help" || command == "help" {
        return help();
    }
    if !COMMANDS.contains(&command) {
        anyhow::bail!("unknown native Kubernetes command {command}");
    }
    if command == "index" {
        return run_index(args);
    }
    if command == "generate" {
        return run_generate(&args[1..]);
    }
    if command == "init" {
        return super::scaffold::run(&args[1..]);
    }
    if command == "dashboard" {
        return super::dashboard::run(&args[1..]);
    }
    if command == "sweep" {
        return run_sweep(&args[1..]);
    }
    if matches!(command, "profile" | "validate") {
        return envelope_command(command, &args[1..]);
    }
    let client = KubeClient::from_options(&auth_options(args)?)?;
    let namespace = namespace(args)?;
    let format = OutputFormat::from_args(args)?;
    let collection = jobs_path(namespace);
    match command {
        "preflight" => report_status(command, client.request("GET", "/version", "", Vec::new())?),
        "list" => report_document(command, format, &client, &collection),
        "show" | "debug" => {
            let name = required_name(args)?;
            report_document(command, format, &client, &format!("{collection}/{name}"))
        }
        "results" => download_results(&client, namespace, required_name(args)?, args),
        "logs" => stream_logs(&client, namespace, required_name(args)?),
        "watch" | "attach" => stream_events(&client, &collection),
        _ => unreachable!(),
    }
}

/// Fetch one bounded API document and print it in the selected format.
fn report_document(
    command: &str,
    format: OutputFormat,
    client: &KubeClient,
    path: &str,
) -> anyhow::Result<i32> {
    let response = client.execute("GET", path, "", Vec::new())?;
    if !response.is_success() {
        anyhow::bail!(
            "native Kubernetes {command} API request returned HTTP {}",
            response.status
        );
    }
    println!("{}", render(format, &response.body)?);
    Ok(0)
}

/// List every retained result run the operator holds for one namespace.
fn run_index(args: &[String]) -> anyhow::Result<i32> {
    let client = KubeClient::from_options(&auth_options(args)?)?;
    let namespace = namespace(args)?;
    let format = OutputFormat::from_args(args)?;
    let operator_prefix = operator_service_proxy(args)?;
    println!(
        "{}",
        index_report(&client, &operator_prefix, namespace, format)?
    );
    Ok(0)
}

/// Fetch and render the operator's retained result index for one namespace.
///
/// The request travels the same authenticated Kubernetes Service proxy the
/// `results` command uses, so the durable index is reachable without any
/// additional ingress.
pub(super) fn index_report(
    client: &KubeClient,
    operator_prefix: &str,
    namespace: &str,
    format: OutputFormat,
) -> anyhow::Result<String> {
    let response = client.execute(
        "GET",
        &format!(
            "{operator_prefix}/api/results/{}",
            encode_segment(namespace)
        ),
        "",
        Vec::new(),
    )?;
    if !response.is_success() {
        anyhow::bail!(
            "native Kubernetes index API request returned HTTP {}",
            response.status
        );
    }
    Ok(render(format, &response.body)?)
}

/// Bounded artifact transfer through the operator Service proxy after Job completion.
struct OperatorFetcher<'client> {
    client: &'client KubeClient,
    prefix: String,
}

impl ArtifactFetcher for OperatorFetcher<'_> {
    fn fetch(&self, path: &str) -> Result<Vec<u8>, KubeError> {
        let response = self.client.execute_with_response_limit(
            "GET",
            &format!("{}/artifacts/{}", self.prefix, encode_relative_path(path)),
            "",
            Vec::new(),
            MAX_ARTIFACT_BYTES as usize,
        )?;
        if !response.is_success() {
            return Err(KubeError::Transport(format!(
                "operator results API returned HTTP {} for {path}",
                response.status
            )));
        }
        Ok(response.body)
    }
}

/// Download every committed artifact after verifying the producer manifest.
fn download_results(
    client: &KubeClient,
    namespace: &str,
    name: &str,
    args: &[String],
) -> anyhow::Result<i32> {
    let (operator_prefix, run_id) = operator_results_location(args)?;
    let prefix = format!(
        "{operator_prefix}/api/results/{}/{}/{}",
        encode_segment(namespace),
        encode_segment(name),
        encode_segment(&run_id)
    );
    let response = client.execute_with_response_limit(
        "GET",
        &format!("{prefix}/manifest"),
        "",
        Vec::new(),
        super::client::MAX_RESPONSE_BYTES,
    )?;
    if !response.is_success() {
        anyhow::bail!(
            "results manifest is unavailable from the durable operator API: HTTP {}",
            response.status
        );
    }
    let manifest = parse_manifest(&response.body)?;
    if manifest.run_id != run_id {
        anyhow::bail!("operator results manifest does not match the AIPerfJob run");
    }
    let fetcher = OperatorFetcher { client, prefix };
    let destination = flag_value(args, "--output-directory")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("aiperf-results").join(&manifest.run_id));
    let written = download(&manifest, &fetcher, &destination)?;
    println!(
        "native Kubernetes results: verified {} artifacts into {}",
        written.len(),
        destination.display()
    );
    Ok(0)
}

fn operator_results_location(args: &[String]) -> anyhow::Result<(String, String)> {
    let trusted_run_id = trusted_run_id(args)?;
    let service = operator_service_proxy(args)?;
    Ok((service, trusted_run_id))
}

/// The Kubernetes Service proxy prefix every operator results request travels.
pub(super) fn operator_service_proxy(args: &[String]) -> anyhow::Result<String> {
    let namespace = flag_value(args, "--operator-namespace")
        .unwrap_or_else(|| DEFAULT_OPERATOR_NAMESPACE.to_string());
    let service = flag_value(args, "--operator-service")
        .unwrap_or_else(|| DEFAULT_OPERATOR_SERVICE.to_string());
    if !is_dns_label(&namespace) || !is_dns_label(&service) {
        anyhow::bail!("operator namespace and service must be DNS labels");
    }
    Ok(format!(
        "/api/v1/namespaces/{namespace}/services/{service}:{RESULTS_API_PORT}/proxy"
    ))
}

fn is_dns_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 63
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && value
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
}

/// Encode one URL path segment of an operator results address.
pub(super) fn encode_segment(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

fn encode_relative_path(value: &str) -> String {
    value
        .split('/')
        .map(encode_segment)
        .collect::<Vec<_>>()
        .join("/")
}

/// Stream container logs byte for byte without reframing or re-encoding them.
fn stream_logs(client: &KubeClient, namespace: &str, name: &str) -> anyhow::Result<i32> {
    let mut stdout = std::io::stdout().lock();
    stream_logs_to(client, namespace, name, &mut stdout)
}

fn stream_logs_to(
    client: &KubeClient,
    namespace: &str,
    name: &str,
    output: &mut impl Write,
) -> anyhow::Result<i32> {
    let pod = controller_pod_name(client, namespace, name)?;
    let watch = client.watch(&format!(
        "/api/v1/namespaces/{}/pods/{}/log?container=controller&follow=true",
        encode_segment(namespace),
        encode_segment(&pod),
    ))?;
    loop {
        match watch.poll(client.watch_deadline())? {
            KubeWatchPoll::Record(record) => output.write_all(&record)?,
            KubeWatchPoll::Idle => anyhow::bail!("Kubernetes log stream timed out"),
            KubeWatchPoll::Closed => break,
        }
    }
    output.flush()?;
    Ok(0)
}

fn controller_pod_name(
    client: &KubeClient,
    namespace: &str,
    job_id: &str,
) -> anyhow::Result<String> {
    let jobset = client.execute(
        "GET",
        &format!(
            "/apis/jobset.x-k8s.io/v1alpha2/namespaces/{}/jobsets/{}",
            encode_segment(namespace),
            encode_segment(job_id),
        ),
        "",
        Vec::new(),
    )?;
    if !jobset.is_success() {
        anyhow::bail!(
            "JobSet is unavailable while resolving logs: HTTP {}",
            jobset.status
        );
    }
    let jobset: serde_json::Value = serde_json::from_slice(&jobset.body)
        .map_err(|error| anyhow::anyhow!("JobSet response is invalid: {error}"))?;
    let jobset_uid = jobset
        .pointer("/metadata/uid")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("JobSet response omits its object UID"))?;
    if jobset
        .pointer("/metadata/name")
        .and_then(|value| value.as_str())
        != Some(job_id)
        || jobset
            .pointer("/metadata/namespace")
            .and_then(|value| value.as_str())
            != Some(namespace)
    {
        anyhow::bail!("JobSet response identity does not match the requested object");
    }
    let selector = encode_segment(&format!(
        "jobset.sigs.k8s.io/jobset-name={job_id},jobset.sigs.k8s.io/replicatedjob-name=controller"
    ));
    let pods = client.execute(
        "GET",
        &format!(
            "/api/v1/namespaces/{}/pods?labelSelector={selector}",
            encode_segment(namespace)
        ),
        "",
        Vec::new(),
    )?;
    if !pods.is_success() {
        anyhow::bail!("controller Pod lookup returned HTTP {}", pods.status);
    }
    let pods: serde_json::Value = serde_json::from_slice(&pods.body)
        .map_err(|error| anyhow::anyhow!("controller Pod list is invalid: {error}"))?;
    let items = pods["items"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("controller Pod list omits items"))?;
    items
        .iter()
        .filter_map(|pod| controller_pod_candidate(pod, namespace, job_id, jobset_uid))
        .max()
        .map(|(_, _, _, name)| name)
        .ok_or_else(|| anyhow::anyhow!("no controller Pod belongs to the current JobSet"))
}

fn controller_pod_candidate(
    pod: &serde_json::Value,
    namespace: &str,
    job_id: &str,
    jobset_uid: &str,
) -> Option<(bool, u8, String, String)> {
    let metadata = pod.get("metadata")?;
    let labels = metadata.get("labels")?;
    let name = metadata.get("name")?.as_str()?;
    if metadata.get("namespace")?.as_str()? != namespace
        || labels.get("jobset.sigs.k8s.io/jobset-name")?.as_str()? != job_id
        || labels.get("jobset.sigs.k8s.io/jobset-uid")?.as_str()? != jobset_uid
        || labels
            .get("jobset.sigs.k8s.io/replicatedjob-name")?
            .as_str()?
            != "controller"
    {
        return None;
    }
    let is_live = metadata.get("deletionTimestamp").is_none();
    let phase_rank = match pod
        .pointer("/status/phase")
        .and_then(|value| value.as_str())
    {
        Some("Running") => 3,
        Some("Succeeded" | "Failed") => 2,
        Some("Pending") => 1,
        _ => 0,
    };
    let created = metadata
        .get("creationTimestamp")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    Some((is_live, phase_rank, created.to_string(), name.to_string()))
}

/// Follow a watch with bounded reconnects so one closed stream is not fatal.
fn stream_events(client: &KubeClient, path: &str) -> anyhow::Result<i32> {
    let mut stdout = std::io::stdout().lock();
    stream_events_to(client, path, &mut stdout)
}

fn stream_events_to(
    client: &KubeClient,
    collection: &str,
    output: &mut impl Write,
) -> anyhow::Result<i32> {
    let mut reconnects = 0;
    let mut resource_version = None;
    loop {
        let path = watch_path(collection, resource_version.as_deref());
        let watch = match client.watch(&path) {
            Ok(watch) => watch,
            Err(error) => {
                reconnect_or_fail(&mut reconnects, anyhow::Error::new(error))?;
                continue;
            }
        };
        match watch_once(
            &watch,
            client.watch_deadline(),
            output,
            &mut resource_version,
        )? {
            WatchEnd::Closed => reconnect_or_fail(
                &mut reconnects,
                anyhow::anyhow!("Kubernetes watch response reached EOF"),
            )?,
            WatchEnd::Idle => reconnect_or_fail(
                &mut reconnects,
                anyhow::anyhow!("Kubernetes watch timed out without an event"),
            )?,
            WatchEnd::Transport(error) => {
                reconnect_or_fail(&mut reconnects, anyhow::Error::new(error))?
            }
            WatchEnd::Expired => {
                resource_version = Some(relist_resource_version(client, collection)?);
                reconnect_or_fail(
                    &mut reconnects,
                    anyhow::anyhow!("Kubernetes watch resource version expired"),
                )?;
            }
        }
    }
}

fn reconnect_or_fail(reconnects: &mut u32, error: anyhow::Error) -> anyhow::Result<()> {
    if *reconnects >= MAX_WATCH_RECONNECTS {
        return Err(error);
    }
    *reconnects += 1;
    tracing::debug!(
        error = %error,
        reconnects = *reconnects,
        component = "kube-watch",
        "reopening bounded Kubernetes watch"
    );
    Ok(())
}

fn watch_path(collection: &str, resource_version: Option<&str>) -> String {
    let mut path = format!("{collection}?watch=true&allowWatchBookmarks=true");
    if let Some(resource_version) = resource_version {
        path.push_str("&resourceVersion=");
        path.push_str(&encode_segment(resource_version));
    }
    path
}

fn relist_resource_version(client: &KubeClient, collection: &str) -> anyhow::Result<String> {
    let response = client.execute("GET", collection, "", Vec::new())?;
    if !response.is_success() {
        anyhow::bail!(
            "Kubernetes relist after expired watch returned HTTP {}",
            response.status
        );
    }
    let list: serde_json::Value = serde_json::from_slice(&response.body)
        .map_err(|error| anyhow::anyhow!("Kubernetes relist response is invalid: {error}"))?;
    list.pointer("/metadata/resourceVersion")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("Kubernetes relist omits metadata.resourceVersion"))
}

/// The value of `--flag value` or `--flag=value`, when present.
pub(super) fn flag_value(args: &[String], flag: &str) -> Option<String> {
    let mut arguments = args.iter();
    let equals = format!("{flag}=");
    while let Some(argument) = arguments.next() {
        if let Some(value) = argument.strip_prefix(&equals) {
            return Some(value.to_string());
        }
        if argument == flag {
            return arguments.next().cloned();
        }
    }
    None
}

/// Arguments for `aiperf kube generate`.
#[derive(Debug, Parser)]
#[command(
    name = "kube-generate",
    about = "Render a native-k8s/v1 controller envelope from a Config-v2 file and flags"
)]
struct GenerateArgs {
    /// Config-v2 YAML file whose filename stem names the ConfigMap reference.
    #[arg(long)]
    config: PathBuf,
    /// Digest-qualified image reference, e.g. `registry/img@sha256:<64-hex>`.
    #[arg(long)]
    image: String,
    /// Number of cellular workers.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    cells: u32,
    /// Target Kubernetes namespace (default: `aiperf`).
    #[arg(long, default_value = "aiperf")]
    namespace: String,
    /// AIPerfJob name; defaults to the sanitized config filename stem.
    #[arg(long)]
    job_id: Option<String>,
    /// Write the envelope to this file instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

/// Render a `native-k8s/v1` controller envelope from a Config-v2 file and flags.
///
/// Bootstrap digest fields carry `"0"×64` placeholders that `submit_profile_transactionally`
/// replaces with minted values at submission time; all other envelope fields are preserved
/// verbatim by `build_controller_envelope`.
fn run_generate(args: &[String]) -> anyhow::Result<i32> {
    let full: Vec<String> = std::iter::once("kube-generate".to_string())
        .chain(args.iter().cloned())
        .collect();
    let parsed = match GenerateArgs::try_parse_from(&full) {
        Ok(parsed) => parsed,
        Err(err) => {
            err.print().ok();
            return Ok(err.exit_code());
        }
    };

    let image_digest = parsed
        .image
        .rsplit_once('@')
        .map(|(_, digest)| digest.to_string())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--image must be a digest-qualified reference (registry/img@sha256:<64-hex>)"
            )
        })?;

    let config_bytes = std::fs::read(&parsed.config).map_err(|error| {
        anyhow::anyhow!(
            "failed to read config file {}: {error}",
            parsed.config.display()
        )
    })?;
    let config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));

    let config_stem = parsed
        .config
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("config");
    let config_name = to_dns_label(config_stem);
    let job_id = parsed
        .job_id
        .as_deref()
        .map(to_dns_label)
        .unwrap_or_else(|| config_name.clone());
    let namespace = to_dns_label(&parsed.namespace);

    let run_id = {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("run-{secs}")
    };

    let cells = parsed.cells;

    let envelope = ControllerEnvelope {
        contract_version: CONTRACT_VERSION.to_string(),
        run_id,
        namespace,
        job_id,
        image_digest,
        image_reference: parsed.image,
        cells,
        artifact_root: "/results".to_string(),
        config_ref: NamedReference {
            name: config_name,
            sha256: config_sha256,
        },
        controller_address: "tcp://aiperf-controller-svc:9500".to_string(),
        roles: vec![
            RoleEnvelope {
                name: NativeK8sRole::Controller,
                command: vec!["aiperf".to_string()],
                argv: vec!["controller".to_string()],
                environment: std::collections::BTreeMap::new(),
                bootstrap: Some(BootstrapReference {
                    secret_name: "bootstrap-controller".to_string(),
                    role: NativeK8sRole::Controller,
                    mount_path: "/bootstrap".to_string(),
                    sha256: "0".repeat(64),
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
        cell_bootstraps: (0..cells)
            .map(|cell_id| CellBootstrapReference {
                cell_id,
                secret_name: format!("bootstrap-cell-{cell_id}"),
                role: NativeK8sRole::Cell,
                mount_path: "/bootstrap".to_string(),
                sha256: "0".repeat(64),
            })
            .collect(),
    };

    // Validate before emitting so any construction bug surfaces as a clear error.
    let as_value = serde_json::to_value(&envelope)
        .map_err(|error| anyhow::anyhow!("failed to serialize envelope: {error}"))?;
    validate_envelope(as_value)
        .map_err(|error| anyhow::anyhow!("generated envelope is invalid: {error}"))?;

    let body = serde_json::to_string_pretty(&envelope).map_err(|error| {
        anyhow::anyhow!("failed to serialize native Kubernetes generate output: {error}")
    })?;

    match parsed.output {
        Some(path) => std::fs::write(&path, body.as_bytes()).map_err(|error| {
            anyhow::anyhow!(
                "failed to write generate output to {}: {error}",
                path.display()
            )
        })?,
        None => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(body.as_bytes())?;
            stdout.write_all(b"\n")?;
        }
    }
    Ok(0)
}

/// Sanitize an arbitrary string into a DNS label: lowercase alphanumeric with hyphens,
/// no leading or trailing hyphens, max 63 characters.
fn to_dns_label(input: &str) -> String {
    let normalized: String = input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = normalized.trim_matches('-');
    let label = if trimmed.is_empty() {
        "config"
    } else {
        trimmed
    };
    let truncated: String = label.chars().take(63).collect();
    truncated.trim_end_matches('-').to_string()
}

fn envelope_command(command: &str, args: &[String]) -> anyhow::Result<i32> {
    let paths = envelope_paths(args)?;
    let envelopes = paths
        .into_iter()
        .map(load_envelope)
        .collect::<anyhow::Result<Vec<_>>>()?;
    if envelopes.len() != 1 {
        anyhow::bail!("native Kubernetes {command} accepts exactly one --envelope");
    }
    let capability_path = flag_value(args, "--image-capabilities").ok_or_else(|| {
        anyhow::anyhow!(
            "native Kubernetes {command} requires --image-capabilities <native-k8s/v1.json>"
        )
    })?;
    validate_image_capability_document(
        std::path::Path::new(&capability_path),
        &envelopes[0].image_digest,
    )?;
    if command == "validate" {
        println!("native Kubernetes validate: native-k8s/v1 envelope is valid");
        return Ok(0);
    }
    let material = material_paths(args)?;
    let client = KubeClient::from_options(&auth_options(args)?)?;
    let status = submit_profile_transactionally(&client, &envelopes[0], &material)?;
    report_status(command, status)
}

/// Dispatch `aiperf kube sweep`: validate the sweep envelope and image capabilities,
/// then submit the sweep transactionally via `submit_sweep_transactionally`.
fn run_sweep(args: &[String]) -> anyhow::Result<i32> {
    let envelope_path = flag_value(args, "--envelope").ok_or_else(|| {
        anyhow::anyhow!("native Kubernetes sweep requires --envelope <sweep-envelope.json>")
    })?;
    let capability_path = flag_value(args, "--image-capabilities").ok_or_else(|| {
        anyhow::anyhow!(
            "native Kubernetes sweep requires --image-capabilities <native-k8s/v1.json>"
        )
    })?;

    // Load and validate the sweep envelope.
    let envelope_bytes = std::fs::read(&envelope_path)
        .map_err(|e| anyhow::anyhow!("failed to read sweep envelope {envelope_path}: {e}"))?;
    let envelope_value: serde_json::Value = serde_json::from_slice(&envelope_bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode sweep envelope {envelope_path}: {e}"))?;
    let envelope = validate_sweep_envelope(envelope_value).map_err(anyhow::Error::from)?;

    // Load the image capability document (validation happens inside submit).
    let cap_bytes = std::fs::read(&capability_path).map_err(|e| {
        anyhow::anyhow!("failed to read image capability document {capability_path}: {e}")
    })?;
    let cap_value: serde_json::Value = serde_json::from_slice(&cap_bytes).map_err(|e| {
        anyhow::anyhow!("failed to decode image capability document {capability_path}: {e}")
    })?;

    let client = KubeClient::from_options(&auth_options(args)?)?;
    let status = submit_sweep_transactionally(&client, &envelope, cap_value)?;

    // Optionally poll sweep phase until terminal.
    if args.iter().any(|a| a == "--watch") {
        watch_sweep(&client, &envelope.namespace, &envelope.run_id)?;
    }

    report_status("sweep", status)
}

/// Poll the AIPerfSweep CR until its phase reaches `Completed` or `Failed`.
fn watch_sweep(client: &KubeClient, namespace: &str, run_id: &str) -> anyhow::Result<()> {
    use super::client::{AIPERF_GROUP, AIPERF_VERSION};
    use super::sweep_controller::AIPERFSWEEPS_PLURAL;

    let path = format!(
        "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/{AIPERFSWEEPS_PLURAL}/{run_id}"
    );
    loop {
        let response = client.execute("GET", &path, "", Vec::new())?;
        if !response.is_success() {
            anyhow::bail!("sweep status poll returned HTTP {}", response.status);
        }
        let cr: serde_json::Value = serde_json::from_slice(&response.body)
            .map_err(|e| anyhow::anyhow!("sweep status response is invalid: {e}"))?;
        let phase = cr
            .pointer("/status/phase")
            .and_then(|v| v.as_str())
            .unwrap_or("Pending");
        let completed = cr
            .pointer("/status/completedRuns")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let failed = cr
            .pointer("/status/failedRuns")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        println!(
            "native Kubernetes sweep: phase={phase} completedRuns={completed} failedRuns={failed}"
        );
        if matches!(phase, "Completed" | "Failed") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    Ok(())
}

/// Resolve kubeconfig/context/token selection from the command's arguments.
pub(super) fn auth_options(args: &[String]) -> anyhow::Result<KubeAuthOptions> {
    let mut options = KubeAuthOptions::default();
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        if let Some(path) = argument.strip_prefix("--kubeconfig=") {
            options.kubeconfig = Some(path.into());
        } else if argument == "--kubeconfig" {
            options.kubeconfig = Some(
                arguments
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--kubeconfig requires a path"))?
                    .into(),
            );
        } else if let Some(context) = argument.strip_prefix("--context=") {
            options.context = Some(context.to_string());
        } else if argument == "--context" {
            options.context = Some(
                arguments
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--context requires a value"))?
                    .to_string(),
            );
        } else if argument == "--insecure-skip-tls-verify" {
            options.insecure_skip_tls_verify = true;
        }
    }
    Ok(options)
}

/// The selected `--namespace`, defaulting to `default`. Must be a DNS label.
pub(super) fn namespace(args: &[String]) -> anyhow::Result<&str> {
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        if let Some(namespace) = argument.strip_prefix("--namespace=") {
            if is_dns_label(namespace) {
                return Ok(namespace);
            }
            anyhow::bail!("--namespace must be a DNS label");
        }
        if argument == "--namespace" {
            let namespace = arguments
                .next()
                .map(String::as_str)
                .ok_or_else(|| anyhow::anyhow!("--namespace requires a value"))?;
            if is_dns_label(namespace) {
                return Ok(namespace);
            }
            anyhow::bail!("--namespace must be a DNS label");
        }
    }
    Ok("default")
}

fn required_name(args: &[String]) -> anyhow::Result<&str> {
    let name = args
        .get(1)
        .map(String::as_str)
        .filter(|name| !name.starts_with('-'))
        .ok_or_else(|| anyhow::anyhow!("command requires an AIPerfJob name"))?;
    if !is_dns_label(name) {
        anyhow::bail!("AIPerfJob name must be a DNS label");
    }
    Ok(name)
}

fn trusted_run_id(args: &[String]) -> anyhow::Result<String> {
    let run_id = flag_value(args, "--run-id").ok_or_else(|| {
        anyhow::anyhow!("durable results require the submitted envelope's trusted --run-id")
    })?;
    if !is_dns_label(&run_id) {
        anyhow::bail!("--run-id must be a DNS label");
    }
    Ok(run_id)
}

fn report_status(command: &str, status: u16) -> anyhow::Result<i32> {
    if !(200..300).contains(&status) {
        anyhow::bail!("native Kubernetes {command} API request returned HTTP {status}");
    }
    println!("native Kubernetes {command}: HTTP {status}");
    Ok(0)
}

enum WatchEnd {
    Closed,
    Idle,
    Expired,
    Transport(KubeError),
}

fn watch_once(
    watch: &KubeWatch,
    timeout: Duration,
    output: &mut impl Write,
    resource_version: &mut Option<String>,
) -> anyhow::Result<WatchEnd> {
    loop {
        let record = match watch.poll(timeout) {
            Ok(KubeWatchPoll::Record(record)) => record,
            Ok(KubeWatchPoll::Idle) => return Ok(WatchEnd::Idle),
            Ok(KubeWatchPoll::Closed) => return Ok(WatchEnd::Closed),
            Err(error) => return Ok(WatchEnd::Transport(error)),
        };
        let event: serde_json::Value = serde_json::from_slice(&record)
            .map_err(|error| anyhow::anyhow!("Kubernetes watch event is invalid: {error}"))?;
        let event_type = event["type"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Kubernetes watch event omits its type"))?;
        if event_type == "ERROR" {
            let object = &event["object"];
            let code = object["code"].as_u64();
            if code == Some(410) {
                return Ok(WatchEnd::Expired);
            }
            let reason = object["reason"].as_str().unwrap_or("Unknown");
            let message = object["message"].as_str().unwrap_or("no message");
            anyhow::bail!(
                "Kubernetes watch ERROR {} {reason}: {message}",
                code.map_or_else(|| "without code".to_string(), |value| value.to_string())
            );
        }
        if !matches!(event_type, "ADDED" | "MODIFIED" | "DELETED" | "BOOKMARK") {
            anyhow::bail!("Kubernetes watch event has unsupported type {event_type}");
        }
        let next_version = event
            .pointer("/object/metadata/resourceVersion")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("Kubernetes watch event omits metadata.resourceVersion")
            })?;
        *resource_version = Some(next_version.to_string());
        if event_type != "BOOKMARK" {
            output.write_all(&record)?;
        }
    }
}

fn help() -> anyhow::Result<i32> {
    println!("aiperf kube <{}>", COMMANDS.join("|"));
    println!(
        "aiperf kube results <job> [--run-id <id>] [--operator-service <name>] [--operator-namespace <namespace>]"
    );
    println!(
        "aiperf kube index [--namespace <namespace>] [--operator-service <name>] [--operator-namespace <namespace>]"
    );
    println!(
        "aiperf kube dashboard [--namespace <namespace>] [--port <port>] [--operator-service <name>] [--operator-namespace <namespace>]"
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use sha2::{Digest, Sha256};

    use crate::kube::auth::KubeCredentials;
    use crate::kube::client::{KubeRequest, KubeResponse, KubeTransport, MAX_RESPONSE_BYTES};

    const FIXTURES: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../contracts/native-k8s/v1/fixtures/"
    );

    #[test]
    fn help_lists_the_complete_native_surface() {
        let commands = COMMANDS.join(" ");
        assert_eq!(COMMANDS.len(), 15);
        assert!(commands.contains("profile"));
        assert!(commands.contains("dashboard"));
    }

    #[test]
    fn index_no_longer_refuses_before_cluster_access() {
        // `sweep` left the refusal list in Task 10, `index` in Task 12, and
        // `dashboard` in Task 13; nothing on the surface refuses for want of a
        // shipped backend.
        let error = run(&["index".to_string(), "--kubeconfig=/nonexistent".to_string()])
            .expect_err("index without a reachable cluster must fail");
        assert!(
            !error.to_string().contains("shipped operator supports only"),
            "index must no longer produce the old refusal: {error:#}"
        );
    }

    #[test]
    fn durable_results_location_uses_only_trusted_local_service_identity() {
        assert_eq!(
            operator_service_proxy(&[]).expect("default service identity"),
            "/api/v1/namespaces/aiperf-system/services/aiperf-k8s-operator:8080/proxy"
        );
        assert_eq!(
            operator_service_proxy(&[
                "--operator-service=operator".to_string(),
                "--operator-namespace".to_string(),
                "control-plane".to_string(),
            ])
            .expect("configured service identity"),
            "/api/v1/namespaces/control-plane/services/operator:8080/proxy"
        );
        assert!(
            operator_service_proxy(&["--operator-service=operator.attacker".to_string()]).is_err()
        );
    }

    #[test]
    fn validate_applies_the_selected_image_capability_document() {
        let error = run(&[
            "validate".to_string(),
            "--envelope".to_string(),
            format!("{FIXTURES}valid-one-cell-envelope.json"),
            "--image-capabilities".to_string(),
            format!("{FIXTURES}missing-cellular-capability.json"),
        ])
        .expect_err("missing cellular support must fail validation");
        assert!(
            error.to_string().contains("image capability document"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn profile_requires_image_capabilities_before_cluster_access() {
        let error = run(&[
            "profile".to_string(),
            "--envelope".to_string(),
            format!("{FIXTURES}valid-one-cell-envelope.json"),
        ])
        .expect_err("capability document is mandatory");
        assert_eq!(
            error.to_string(),
            "native Kubernetes profile requires --image-capabilities <native-k8s/v1.json>"
        );
    }

    #[test]
    fn namespace_defaults_and_parses_both_forms() {
        assert_eq!(
            namespace(&["list".to_string()]).expect("default"),
            "default"
        );
        assert_eq!(
            namespace(&["list".to_string(), "--namespace=bench".to_string()]).expect("equals"),
            "bench"
        );
        assert_eq!(
            namespace(&[
                "list".to_string(),
                "--namespace".to_string(),
                "bench".to_string(),
            ])
            .expect("separate"),
            "bench"
        );
    }

    #[test]
    fn auth_options_accept_explicit_kubeconfig_context_and_tls_escape_hatch() {
        let options = auth_options(&[
            "--kubeconfig=/tmp/config".to_string(),
            "--context".to_string(),
            "bench".to_string(),
            "--insecure-skip-tls-verify".to_string(),
        ])
        .expect("options");
        assert_eq!(
            options.kubeconfig.expect("config"),
            std::path::PathBuf::from("/tmp/config")
        );
        assert_eq!(options.context.as_deref(), Some("bench"));
        assert!(options.insecure_skip_tls_verify);
    }

    #[test]
    fn streaming_and_download_flags_are_parsed_natively() {
        let args = [
            "results".to_string(),
            "job-1".to_string(),
            "--output-directory".to_string(),
            "/tmp/out".to_string(),
            "--port=19999".to_string(),
        ];
        assert_eq!(
            flag_value(&args, "--output-directory").as_deref(),
            Some("/tmp/out")
        );
        assert_eq!(flag_value(&args, "--port").as_deref(), Some("19999"));
        assert_eq!(flag_value(&args, "--missing"), None);
    }

    #[test]
    fn job_commands_require_an_explicit_name() {
        assert!(required_name(&["show".to_string()]).is_err());
        assert_eq!(
            required_name(&["show".to_string(), "job-1".to_string()]).expect("name"),
            "job-1"
        );
        assert!(required_name(&["show".to_string(), "../other".to_string()]).is_err());
    }

    #[test]
    fn command_identity_flags_use_the_envelope_dns_syntax() {
        assert!(
            namespace(&[
                "list".to_string(),
                "--namespace=NOT_A_NAMESPACE".to_string()
            ])
            .is_err()
        );
        assert!(
            trusted_run_id(&[
                "results".to_string(),
                "job-1".to_string(),
                "--run-id=run/other".to_string()
            ])
            .is_err()
        );
    }

    #[test]
    fn watch_reconnects_and_emits_events_after_the_first_stream_ends() {
        let first =
            b"{\"type\":\"ADDED\",\"object\":{\"metadata\":{\"resourceVersion\":\"10\"}}}\n"
                .to_vec();
        let second =
            b"{\"type\":\"MODIFIED\",\"object\":{\"metadata\":{\"resourceVersion\":\"11\"}}}\n"
                .to_vec();
        let transport = Arc::new(WatchTransport {
            watches: Mutex::new(vec![
                super::super::client::KubeWatch::events_for_test(vec![second.clone()]),
                super::super::client::KubeWatch::events_for_test(vec![first.clone()]),
            ]),
            watch_paths: Mutex::new(Vec::new()),
            list_response: None,
            requests: Mutex::new(Vec::new()),
        });
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        let mut output = Vec::new();

        let error = stream_events_to(&client, "/watch", &mut output).expect_err("streams end");
        assert!(error.to_string().contains("watch streams exhausted"));
        assert_eq!(output, [first, second].concat());
        let paths = transport.watch_paths.lock().expect("watch paths");
        assert_eq!(paths[0], "/watch?watch=true&allowWatchBookmarks=true");
        assert_eq!(
            paths[1],
            "/watch?watch=true&allowWatchBookmarks=true&resourceVersion=10"
        );
        assert!(
            paths[2..]
                .iter()
                .all(|path| path.ends_with("resourceVersion=11"))
        );
    }

    #[test]
    fn watch_surfaces_kubernetes_error_events_without_emitting_them() {
        let transport = Arc::new(WatchTransport {
            watches: Mutex::new(vec![
                super::super::client::KubeWatch::events_for_test(vec![
                    b"{\"type\":\"ERROR\",\"object\":{\"code\":403,\"reason\":\"Forbidden\",\"message\":\"denied\"}}\n"
                        .to_vec(),
                ]),
            ]),
            watch_paths: Mutex::new(Vec::new()),
            list_response: None,
            requests: Mutex::new(Vec::new()),
        });
        let client = KubeClient::with_transport(test_credentials(), transport);
        let mut output = Vec::new();

        let error = stream_events_to(&client, "/watch", &mut output)
            .expect_err("Kubernetes ERROR event is terminal");
        assert!(error.to_string().contains("Forbidden"), "{error:#}");
        assert!(output.is_empty());
    }

    #[test]
    fn expired_watch_relists_before_reconnecting_from_a_fresh_version() {
        let first =
            b"{\"type\":\"ADDED\",\"object\":{\"metadata\":{\"resourceVersion\":\"42\"}}}\n"
                .to_vec();
        let after_relist =
            b"{\"type\":\"MODIFIED\",\"object\":{\"metadata\":{\"resourceVersion\":\"101\"}}}\n"
                .to_vec();
        let transport = Arc::new(WatchTransport {
            watches: Mutex::new(vec![
                super::super::client::KubeWatch::events_for_test(vec![after_relist.clone()]),
                super::super::client::KubeWatch::events_for_test(vec![
                    b"{\"type\":\"ERROR\",\"object\":{\"code\":410,\"reason\":\"Expired\",\"message\":\"too old\"}}\n"
                        .to_vec(),
                ]),
                super::super::client::KubeWatch::events_for_test(vec![first.clone()]),
            ]),
            watch_paths: Mutex::new(Vec::new()),
            list_response: Some(
                br#"{"metadata":{"resourceVersion":"100"},"items":[]}"#.to_vec(),
            ),
            requests: Mutex::new(Vec::new()),
        });
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        let mut output = Vec::new();

        let error = stream_events_to(&client, "/watch", &mut output).expect_err("streams end");
        assert!(error.to_string().contains("watch streams exhausted"));
        assert_eq!(output, [first, after_relist].concat());
        assert_eq!(
            transport.requests.lock().expect("list requests")[0].path,
            "/watch"
        );
        assert!(
            transport
                .watch_paths
                .lock()
                .expect("watch paths")
                .iter()
                .any(|path| path.ends_with("resourceVersion=100"))
        );
    }

    #[test]
    fn logs_discovers_the_current_controller_pod_before_following() {
        let transport = Arc::new(PodLogTransport {
            requests: Mutex::new(Vec::new()),
            log_paths: Mutex::new(Vec::new()),
        });
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        let mut output = Vec::new();

        stream_logs_to(&client, "bench", "job-1", &mut output).expect("controller logs");

        assert_eq!(output, b"controller output\n");
        let requests = transport.requests.lock().expect("pod lookup requests");
        assert!(requests[0].path.ends_with("/jobsets/job-1"));
        assert!(requests[1].path.contains("/pods?labelSelector="));
        assert!(
            requests[1]
                .path
                .contains("jobset.sigs.k8s.io%2Fjobset-name%3Djob-1")
        );
        assert_eq!(
            *transport.log_paths.lock().expect("log paths"),
            vec![
                "/api/v1/namespaces/bench/pods/job-1-controller-0-0-random/log?container=controller&follow=true"
                    .to_string()
            ]
        );
    }

    #[test]
    fn results_downloads_a_declared_artifact_larger_than_generic_api_responses() {
        let payload = vec![b'x'; MAX_RESPONSE_BYTES + 1];
        let digest = format!("{:x}", Sha256::digest(&payload));
        let manifest = format!(
            r#"{{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{{"path":"large.bin","sha256":"{digest}","bytes":{},"contentType":"application/octet-stream"}}]}}"#,
            payload.len()
        )
        .into_bytes();
        let transport = Arc::new(CompletedResultsTransport {
            manifest,
            artifact: payload.clone(),
            requests: Mutex::new(Vec::new()),
        });
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        let directory = tempfile::tempdir().expect("tempdir");
        let args = vec![
            "results".to_string(),
            "job-1".to_string(),
            "--run-id=run-1".to_string(),
            "--output-directory".to_string(),
            directory.path().display().to_string(),
        ];

        download_results(&client, "bench", "job-1", &args).expect("results download");
        assert_eq!(
            std::fs::read(directory.path().join("large.bin")).expect("artifact"),
            payload
        );
        let requests = transport.requests.lock().expect("requests");
        assert!(
            requests
                .iter()
                .any(|request| request.path.ends_with("/manifest"))
        );
        assert!(requests.iter().any(|request| {
            request.path.ends_with("/artifacts/large.bin")
                && request.response_limit == MAX_ARTIFACT_BYTES as usize
        }));
    }

    #[test]
    fn completed_job_results_use_only_the_persistent_operator_service_proxy() {
        let payload = b"{}".to_vec();
        let digest = format!("{:x}", Sha256::digest(&payload));
        let manifest = format!(
            r#"{{"contractVersion":"native-k8s/v1","runId":"run-1","ready":true,"wasCancelled":false,"artifactRoot":"/results","artifacts":[{{"path":"nested/summary.json","sha256":"{digest}","bytes":{},"contentType":"application/json"}}]}}"#,
            payload.len()
        )
        .into_bytes();
        let transport = Arc::new(CompletedResultsTransport {
            manifest,
            artifact: payload,
            requests: Mutex::new(Vec::new()),
        });
        let client = KubeClient::with_transport(test_credentials(), transport.clone());
        let directory = tempfile::tempdir().expect("tempdir");
        let args = vec![
            "results".to_string(),
            "job-1".to_string(),
            "--run-id=run-1".to_string(),
            "--output-directory".to_string(),
            directory.path().display().to_string(),
        ];

        download_results(&client, "bench", "job-1", &args).expect("persisted results download");

        assert_eq!(
            std::fs::read(directory.path().join("nested/summary.json")).expect("artifact"),
            b"{}"
        );
        let requests = transport.requests.lock().expect("requests");
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].path,
            "/api/v1/namespaces/aiperf-system/services/aiperf-k8s-operator:8080/proxy/api/results/bench/job-1/run-1/manifest"
        );
        assert_eq!(
            requests[1].path,
            "/api/v1/namespaces/aiperf-system/services/aiperf-k8s-operator:8080/proxy/api/results/bench/job-1/run-1/artifacts/nested/summary.json"
        );
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

    struct CompletedResultsTransport {
        manifest: Vec<u8>,
        artifact: Vec<u8>,
        requests: Mutex<Vec<KubeRequest>>,
    }

    struct WatchTransport {
        watches: Mutex<Vec<super::super::client::KubeWatch>>,
        watch_paths: Mutex<Vec<String>>,
        list_response: Option<Vec<u8>>,
        requests: Mutex<Vec<KubeRequest>>,
    }

    struct PodLogTransport {
        requests: Mutex<Vec<KubeRequest>>,
        log_paths: Mutex<Vec<String>>,
    }

    impl KubeTransport for WatchTransport {
        fn send(
            &self,
            _credentials: &KubeCredentials,
            request: KubeRequest,
        ) -> Result<KubeResponse, KubeError> {
            self.requests
                .lock()
                .map_err(|_| KubeError::Transport("list requests lock poisoned".to_string()))?
                .push(request);
            self.list_response
                .clone()
                .map(|body| KubeResponse { status: 200, body })
                .ok_or_else(|| KubeError::Transport("request not used".to_string()))
        }

        fn watch(
            &self,
            _credentials: &KubeCredentials,
            request: KubeRequest,
        ) -> Result<super::super::client::KubeWatch, KubeError> {
            self.watch_paths
                .lock()
                .map_err(|_| KubeError::Transport("watch paths lock poisoned".to_string()))?
                .push(request.path);
            self.watches
                .lock()
                .map_err(|_| KubeError::Transport("watches lock poisoned".to_string()))?
                .pop()
                .ok_or_else(|| KubeError::Transport("watch streams exhausted".to_string()))
        }
    }

    impl KubeTransport for PodLogTransport {
        fn send(
            &self,
            _credentials: &KubeCredentials,
            request: KubeRequest,
        ) -> Result<KubeResponse, KubeError> {
            let body = if request.path.ends_with("/jobsets/job-1") {
                br#"{"metadata":{"name":"job-1","namespace":"bench","uid":"jobset-uid"}}"#.to_vec()
            } else if request.path.contains("/pods?labelSelector=") {
                br#"{"items":[{"metadata":{"name":"old-guessed-name","namespace":"bench","creationTimestamp":"2026-01-01T00:00:00Z","labels":{"jobset.sigs.k8s.io/jobset-name":"job-1","jobset.sigs.k8s.io/jobset-uid":"old-uid","jobset.sigs.k8s.io/replicatedjob-name":"controller"}},"status":{"phase":"Running"}},{"metadata":{"name":"job-1-controller-0-0-random","namespace":"bench","creationTimestamp":"2026-01-02T00:00:00Z","labels":{"jobset.sigs.k8s.io/jobset-name":"job-1","jobset.sigs.k8s.io/jobset-uid":"jobset-uid","jobset.sigs.k8s.io/replicatedjob-name":"controller"}},"status":{"phase":"Running"}}]}"#
                    .to_vec()
            } else {
                return Err(KubeError::Transport(format!(
                    "unexpected pod lookup {}",
                    request.path
                )));
            };
            self.requests
                .lock()
                .map_err(|_| KubeError::Transport("pod requests lock poisoned".to_string()))?
                .push(request);
            Ok(KubeResponse { status: 200, body })
        }

        fn watch(
            &self,
            _credentials: &KubeCredentials,
            request: KubeRequest,
        ) -> Result<super::super::client::KubeWatch, KubeError> {
            self.log_paths
                .lock()
                .map_err(|_| KubeError::Transport("log paths lock poisoned".to_string()))?
                .push(request.path);
            Ok(super::super::client::KubeWatch::events_for_test(vec![
                b"controller output\n".to_vec(),
            ]))
        }
    }

    impl KubeTransport for CompletedResultsTransport {
        fn send(
            &self,
            _credentials: &KubeCredentials,
            request: KubeRequest,
        ) -> Result<KubeResponse, KubeError> {
            self.requests
                .lock()
                .map_err(|_| KubeError::Transport("requests lock poisoned".to_string()))?
                .push(request.clone());
            let (status, body) = if request
                .path
                .ends_with("/api/results/bench/job-1/run-1/manifest")
            {
                (200, self.manifest.clone())
            } else if request
                .path
                .ends_with("/api/results/bench/job-1/run-1/artifacts/nested/summary.json")
                || request
                    .path
                    .ends_with("/api/results/bench/job-1/run-1/artifacts/large.bin")
            {
                (200, self.artifact.clone())
            } else {
                return Err(KubeError::Transport(format!(
                    "unexpected path {}",
                    request.path
                )));
            };
            Ok(KubeResponse { status, body })
        }

        fn watch(
            &self,
            _credentials: &KubeCredentials,
            _request: KubeRequest,
        ) -> Result<super::super::client::KubeWatch, KubeError> {
            Err(KubeError::Transport("watch not used".to_string()))
        }
    }
}
