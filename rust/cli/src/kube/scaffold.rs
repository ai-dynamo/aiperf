// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf kube init` — scaffold a Config-v2 YAML and image-capabilities pair.
//!
//! No cluster contact occurs on any path. The emitted `image-capabilities.json`
//! carries a literal placeholder digest that fails `validate_image_capabilities`
//! by design, so an unedited scaffold cannot reach the cluster via `kube validate`.

use std::path::PathBuf;

use clap::Parser;

use crate::config::templates_data::TEMPLATES;

/// The literal placeholder written into the scaffolded `image-capabilities.json`.
///
/// `validate_image_capabilities` rejects this string because the schema requires
/// `^sha256:[0-9a-f]{64}$` and the angle-bracketed token does not match it, so an
/// unedited scaffold fails closed at `kube validate` before any cluster is reached.
pub const PLACEHOLDER_DIGEST: &str = "sha256:<64-hex-digest>";

const CONFIG_FILENAME: &str = "benchmark.yaml";
const CAPABILITIES_FILENAME: &str = "image-capabilities.json";
const DEFAULT_TEMPLATE: &str = "minimal";

#[derive(Debug, Parser)]
#[command(
    name = "kube-init",
    about = "Scaffold a Config-v2 YAML and image-capabilities pair for a native Kubernetes run"
)]
struct ScaffoldArgs {
    /// Template id (see `--list`).
    #[arg(long)]
    template: Option<String>,
    /// Write files into this directory instead of the current directory.
    #[arg(long)]
    output_directory: Option<PathBuf>,
    /// Overwrite existing files.
    #[arg(long)]
    force: bool,
    /// List available templates and exit.
    #[arg(long)]
    list: bool,
}

/// Run `aiperf kube init [--output-directory <dir>] [--template <name>] [--force] [--list]`.
///
/// Writes `benchmark.yaml` and `image-capabilities.json` into the output directory.
/// Unknown flags cause a usage error. No cluster contact occurs on any path.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let full: Vec<String> = std::iter::once("kube-init".to_string())
        .chain(args.iter().cloned())
        .collect();
    let parsed = match ScaffoldArgs::try_parse_from(&full) {
        Ok(parsed) => parsed,
        Err(err) => {
            err.print().ok();
            return Ok(err.exit_code());
        }
    };

    if parsed.list {
        list_templates();
        return Ok(0);
    }

    let template_name = parsed
        .template
        .as_deref()
        .unwrap_or(DEFAULT_TEMPLATE)
        .to_string();
    let output_dir = parsed
        .output_directory
        .unwrap_or_else(|| PathBuf::from("."));
    let is_force = parsed.force;

    let template = TEMPLATES
        .iter()
        .find(|t| t.name == template_name)
        .ok_or_else(|| {
            anyhow::anyhow!("unknown template {template_name:?}; run 'aiperf kube init --list'")
        })?;

    let config_content = strip_spdx_header(template.content);
    let capabilities_content = format_capabilities();

    // Check existence before creating directories so a refused overwrite
    // does not leave empty directories behind.
    let config_path = output_dir.join(CONFIG_FILENAME);
    let capabilities_path = output_dir.join(CAPABILITIES_FILENAME);

    if !is_force {
        if config_path.exists() {
            anyhow::bail!(
                "{} already exists; use --force to overwrite",
                config_path.display()
            );
        }
        if capabilities_path.exists() {
            anyhow::bail!(
                "{} already exists; use --force to overwrite",
                capabilities_path.display()
            );
        }
    }

    std::fs::create_dir_all(&output_dir)?;
    std::fs::write(&config_path, config_content)?;
    std::fs::write(&capabilities_path, capabilities_content)?;

    println!("Wrote {}", config_path.display());
    println!("Wrote {}", capabilities_path.display());
    println!("Edit imageDigest in {CAPABILITIES_FILENAME} before running 'aiperf kube validate'.");

    Ok(0)
}

/// Format the scaffolded image-capabilities document.
///
/// The placeholder digest makes the document fail `validate_image_capabilities`
/// until the operator edits it to a real image digest.
fn format_capabilities() -> String {
    // Serialise deterministically so the file is stable for source control.
    let doc = serde_json::json!({
        "contractVersion": "native-k8s/v1",
        "imageDigest": PLACEHOLDER_DIGEST,
        "cellular": true,
        "resultsSidecar": true,
        "hierarchicalAggregation": false,
    });
    // serde_json::to_string_pretty cannot fail on a literal Value constructed above.
    let mut out = serde_json::to_string_pretty(&doc).expect("serialising a static JSON value");
    out.push('\n');
    out
}

/// Print available templates grouped by category, matching `aiperf config init --list`.
fn list_templates() {
    let mut cats: std::collections::BTreeMap<&str, Vec<&crate::config::templates_data::Template>> =
        std::collections::BTreeMap::new();
    for t in TEMPLATES {
        cats.entry(t.category).or_default().push(t);
    }
    for (cat, items) in cats {
        println!("\n{cat}");
        for t in items {
            println!("  {:<28} {}", t.name, t.title);
            println!("  {:<28} {}", "", t.description);
        }
    }
    println!();
}

/// Remove leading SPDX and `yaml-language-server:` comment lines from template content.
fn strip_spdx_header(content: &str) -> String {
    let mut lines = content.lines().peekable();
    while let Some(line) = lines.peek() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("# SPDX-") || trimmed.starts_with("# yaml-language-server:") {
            lines.next();
        } else {
            break;
        }
    }
    let rest: Vec<&str> = lines.collect();
    let mut out = rest.join("\n");
    if content.ends_with('\n') {
        out.push('\n');
    }
    out
}
