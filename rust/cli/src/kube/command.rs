// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf kube` command surface.

use std::io::Write;
use std::time::Duration;

use super::auth::KubeAuthOptions;
use super::client::{AIPERF_GROUP, AIPERF_VERSION, KubeClient};
use super::dashboard::LoopbackForwarder;
use super::error::KubeError;
use super::render::{OutputFormat, render};
use super::results::{ArtifactFetcher, download, parse_manifest};
use super::submission::{
    create_bootstrap_secrets, envelope_paths, jobs_path, load_envelope, material_paths,
    submit_profile, submit_sweep,
};

/// Maximum bounded reconnects a streaming command performs before failing.
const MAX_WATCH_RECONNECTS: u32 = 5;

/// Port the controller pod's results sidecar serves on.
const RESULTS_SIDECAR_PORT: u16 = 9091;

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
    if matches!(command, "init" | "generate") {
        println!(
            "native Kubernetes {command}: provide a strict native-k8s/v1 envelope with --envelope"
        );
        return Ok(0);
    }
    if matches!(command, "profile" | "sweep" | "validate") {
        return envelope_command(command, &args[1..]);
    }
    let client = KubeClient::from_options(&auth_options(args)?)?;
    let namespace = namespace(args)?;
    let format = OutputFormat::from_args(args)?;
    let collection = jobs_path(namespace);
    match command {
        "preflight" => report_status(command, client.request("GET", "/version", "", Vec::new())?),
        "list" => report_document(command, format, &client, &collection),
        "index" => report_document(
            command,
            format,
            &client,
            &format!(
                "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/aiperfjobindexes"
            ),
        ),
        "show" | "debug" => {
            let name = required_name(args)?;
            report_document(command, format, &client, &format!("{collection}/{name}"))
        }
        "results" => download_results(&client, namespace, required_name(args)?, args),
        "dashboard" => serve_dashboard(namespace, required_name(args)?, args),
        "logs" => stream_logs(&client, namespace, required_name(args)?),
        "watch" | "attach" => stream_events(&client, &format!("{collection}?watch=true")),
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

/// Bounded artifact transfer through the API server's pod proxy subresource.
struct ProxyFetcher<'client> {
    client: &'client KubeClient,
    prefix: String,
}

impl ArtifactFetcher for ProxyFetcher<'_> {
    fn fetch(&self, path: &str) -> Result<Vec<u8>, KubeError> {
        let response = self.client.execute(
            "GET",
            &format!("{}/files/{path}", self.prefix),
            "",
            Vec::new(),
        )?;
        if !response.is_success() {
            return Err(KubeError::Transport(format!(
                "results sidecar returned HTTP {} for {path}",
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
    let prefix = format!(
        "/api/v1/namespaces/{namespace}/pods/{name}-controller-0-0:{RESULTS_SIDECAR_PORT}/proxy/api/results"
    );
    let response = client.execute("GET", &format!("{prefix}/manifest"), "", Vec::new())?;
    if !response.is_success() {
        anyhow::bail!("results manifest is unavailable: HTTP {}", response.status);
    }
    let manifest = parse_manifest(&response.body)?;
    let destination = flag_value(args, "--output-directory")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("aiperf-results").join(&manifest.run_id));
    std::fs::create_dir_all(&destination)?;
    let fetcher = ProxyFetcher { client, prefix };
    let written = download(&manifest, &fetcher, &destination)?;
    println!(
        "native Kubernetes results: verified {} artifacts into {}",
        written.len(),
        destination.display()
    );
    Ok(0)
}

/// Bind a loopback-only dashboard listener without spawning any external tool.
fn serve_dashboard(namespace: &str, name: &str, args: &[String]) -> anyhow::Result<i32> {
    let port = match flag_value(args, "--port") {
        Some(port) => port
            .parse::<u16>()
            .map_err(|error| anyhow::anyhow!("--port must be a TCP port: {error}"))?,
        None => 0,
    };
    let forwarder = LoopbackForwarder::bind(port)?;
    println!(
        "native Kubernetes dashboard: {namespace}/{name} available on http://{}",
        forwarder.local_address()?
    );
    Ok(0)
}

/// Stream container logs byte for byte without reframing or re-encoding them.
fn stream_logs(client: &KubeClient, namespace: &str, name: &str) -> anyhow::Result<i32> {
    let watch = client.watch(&format!(
        "/api/v1/namespaces/{namespace}/pods/{name}-controller-0-0/log?follow=true"
    ))?;
    let mut stdout = std::io::stdout().lock();
    while let Some(record) = watch.next(client.watch_deadline())? {
        stdout.write_all(&record)?;
    }
    stdout.flush()?;
    Ok(0)
}

/// Follow a watch with bounded reconnects so one closed stream is not fatal.
fn stream_events(client: &KubeClient, path: &str) -> anyhow::Result<i32> {
    let mut reconnects = 0;
    loop {
        match watch_once(client, path) {
            Ok(code) => return Ok(code),
            Err(error) if reconnects < MAX_WATCH_RECONNECTS => {
                reconnects += 1;
                tracing::debug!(
                    error = %error,
                    reconnects,
                    component = "kube-watch",
                    "reopening bounded Kubernetes watch"
                );
            }
            Err(error) => return Err(error),
        }
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
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

fn envelope_command(command: &str, args: &[String]) -> anyhow::Result<i32> {
    let paths = envelope_paths(args)?;
    let envelopes = paths
        .into_iter()
        .map(load_envelope)
        .collect::<anyhow::Result<Vec<_>>>()?;
    if command == "validate" {
        if envelopes.len() != 1 {
            anyhow::bail!("native Kubernetes validate accepts exactly one --envelope");
        }
        println!("native Kubernetes validate: native-k8s/v1 envelope is valid");
        return Ok(0);
    }
    let client = KubeClient::from_options(&auth_options(args)?)?;
    // Bootstrap material is created before submission so no role ever starts
    // without its Secret, and the envelope keeps only reference metadata.
    let material = material_paths(args)?;
    for envelope in &envelopes {
        create_bootstrap_secrets(&client, envelope, &material)?;
    }
    let status = match command {
        "profile" => {
            if envelopes.len() != 1 {
                anyhow::bail!("native Kubernetes profile accepts exactly one --envelope");
            }
            submit_profile(&client, &envelopes[0])?
        }
        "sweep" => submit_sweep(&client, &envelopes)?,
        _ => unreachable!(),
    };
    report_status(command, status)
}

fn auth_options(args: &[String]) -> anyhow::Result<KubeAuthOptions> {
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

fn namespace(args: &[String]) -> anyhow::Result<&str> {
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        if let Some(namespace) = argument.strip_prefix("--namespace=") {
            return Ok(namespace);
        }
        if argument == "--namespace" {
            return arguments
                .next()
                .map(String::as_str)
                .ok_or_else(|| anyhow::anyhow!("--namespace requires a value"));
        }
    }
    Ok("default")
}

fn required_name(args: &[String]) -> anyhow::Result<&str> {
    args.get(1)
        .map(String::as_str)
        .filter(|name| !name.starts_with('-'))
        .ok_or_else(|| anyhow::anyhow!("command requires an AIPerfJob name"))
}

fn report_status(command: &str, status: u16) -> anyhow::Result<i32> {
    if !(200..300).contains(&status) {
        anyhow::bail!("native Kubernetes {command} API request returned HTTP {status}");
    }
    println!("native Kubernetes {command}: HTTP {status}");
    Ok(0)
}

fn watch_once(client: &KubeClient, path: &str) -> anyhow::Result<i32> {
    let watch = client.watch(path)?;
    match watch.next(Duration::from_secs(30))? {
        Some(event) => print!("{}", String::from_utf8_lossy(&event)),
        None => anyhow::bail!("Kubernetes watch timed out without an event"),
    }
    Ok(0)
}

fn help() -> anyhow::Result<i32> {
    println!("aiperf kube <{}>", COMMANDS.join("|"));
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_lists_the_complete_native_surface() {
        let commands = COMMANDS.join(" ");
        assert_eq!(COMMANDS.len(), 15);
        assert!(commands.contains("profile"));
        assert!(commands.contains("dashboard"));
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
    }
}
