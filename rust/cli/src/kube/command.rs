// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf kube` command surface.

use std::time::Duration;

use super::auth::KubeAuthOptions;
use super::client::{AIPERF_GROUP, AIPERF_VERSION, KubeClient};
use super::submission::{envelope_paths, jobs_path, load_envelope, submit_profile, submit_sweep};

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
    let client = KubeClient::from_options(&KubeAuthOptions::default())?;
    let namespace = namespace(args)?;
    let collection = jobs_path(namespace);
    match command {
        "preflight" => report_status(command, client.request("GET", "/version", "", Vec::new())?),
        "list" => report_status(command, client.request("GET", &collection, "", Vec::new())?),
        "index" => report_status(
            command,
            client.request(
                "GET",
                &format!(
                    "/apis/{AIPERF_GROUP}/{AIPERF_VERSION}/namespaces/{namespace}/aiperfjobindexes"
                ),
                "",
                Vec::new(),
            )?,
        ),
        "show" | "debug" | "results" | "dashboard" => {
            let name = required_name(args)?;
            report_status(
                command,
                client.request("GET", &format!("{collection}/{name}"), "", Vec::new())?,
            )
        }
        "watch" | "attach" | "logs" => watch_once(&client, &format!("{collection}?watch=true")),
        _ => unreachable!(),
    }
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
    let client = KubeClient::from_options(&KubeAuthOptions::default())?;
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
    fn job_commands_require_an_explicit_name() {
        assert!(required_name(&["show".to_string()]).is_err());
        assert_eq!(
            required_name(&["show".to_string(), "job-1".to_string()]).expect("name"),
            "job-1"
        );
    }
}
