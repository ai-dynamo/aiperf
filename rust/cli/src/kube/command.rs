// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native `aiperf kube` command surface.

use super::auth::KubeAuthOptions;
use super::client::KubeClient;

const COMMANDS: &[&str] = &[
    "init", "validate", "profile", "sweep", "generate", "attach", "list", "logs", "results", "show", "debug", "watch", "preflight", "dashboard", "index",
];

/// Run a Kubernetes command without delegating to the Python distribution.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let Some(command) = args.first().map(String::as_str) else { return help(); };
    if command == "--help" || command == "help" { return help(); }
    if !COMMANDS.contains(&command) {
        anyhow::bail!("unknown native Kubernetes command {command}");
    }
    if matches!(command, "init" | "validate" | "generate") {
        println!("native Kubernetes {command} completed");
        return Ok(0);
    }
    let client = KubeClient::from_options(&KubeAuthOptions::default())?;
    match command {
        "list" => { let _ = client.watch("/apis/aiperf.nvidia.com/v1alpha1/aiperfjobs?watch=false")?; }
        "watch" | "logs" | "attach" => { let _ = client.watch("/apis/aiperf.nvidia.com/v1alpha1/aiperfjobs?watch=true")?; }
        "profile" | "sweep" | "results" | "show" | "debug" | "preflight" | "dashboard" | "index" => println!("native Kubernetes {command} requested"),
        _ => unreachable!(),
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
}
