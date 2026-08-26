// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf config` subcommands.

pub mod templates_data;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use templates_data::TEMPLATES;

#[derive(Debug, Parser)]
#[command(name = "config", disable_help_subcommand = true)]
struct ConfigCli {
    #[command(subcommand)]
    command: ConfigCommand,
}

#[derive(Debug, Subcommand)]
enum ConfigCommand {
    /// Scaffold a config file from a template.
    Init(InitArgs),
    /// Validate a config file.
    Validate(ValidateArgs),
    /// Preview the runs a sweeping config expands to.
    Expand(ExpandArgs),
}

#[derive(Debug, Args)]
struct InitArgs {
    /// Template id (see `--list`).
    #[arg(long)]
    template: Option<String>,
    /// List available templates and exit.
    #[arg(long)]
    list: bool,
    /// Filter `--list` by category.
    #[arg(long)]
    category: Option<String>,
    /// Override the model name.
    #[arg(long)]
    model: Option<String>,
    /// Override the endpoint URL.
    #[arg(long)]
    url: Option<String>,
    /// Write to this path instead of stdout.
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ValidateArgs {
    /// Path to the config file.
    config_file: PathBuf,
}

#[derive(Debug, Args)]
struct ExpandArgs {
    /// Path to the config file.
    #[arg(long)]
    config: Option<PathBuf>,
}

/// Run `aiperf config <args>` (argv without the leading `config`).
pub fn run(argv: &[String]) -> anyhow::Result<i32> {
    let full: Vec<String> = std::iter::once("config".to_string())
        .chain(argv.iter().cloned())
        .collect();
    let cli = match ConfigCli::try_parse_from(&full) {
        Ok(cli) => cli,
        Err(err) => {
            err.print().ok();
            return Ok(err.exit_code());
        }
    };
    match cli.command {
        ConfigCommand::Init(a) => init(a),
        ConfigCommand::Validate(a) => validate(a),
        ConfigCommand::Expand(a) => expand(a),
    }
}

fn init(args: InitArgs) -> anyhow::Result<i32> {
    if args.list {
        list_templates(args.category.as_deref());
        return Ok(0);
    }
    let Some(name) = args.template.as_deref() else {
        eprintln!(
            "Specify a template with --template, or run \
             'aiperf config init --list' to see what is available."
        );
        return Ok(2);
    };
    let Some(t) = TEMPLATES.iter().find(|t| t.name == name) else {
        eprintln!("Error: unknown template {name:?}; run 'aiperf config init --list'");
        return Ok(1);
    };
    let mut content = t.content.to_string();
    if let Some(model) = &args.model {
        content = override_scalar(&content, &["model", "models"], model);
    }
    if let Some(url) = &args.url {
        content = override_scalar(&content, &["url", "urls"], url);
    }
    content = strip_spdx_header(&content);
    match args.output {
        None => print!("{content}"),
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &content)?;
            println!("Wrote {name} template to {}", path.display());
        }
    }
    Ok(0)
}

pub(crate) fn list_templates(category: Option<&str>) {
    let mut cats: std::collections::BTreeMap<&str, Vec<&templates_data::Template>> =
        std::collections::BTreeMap::new();
    for t in TEMPLATES {
        if category.is_none_or(|c| t.category.eq_ignore_ascii_case(c)) {
            cats.entry(t.category).or_default().push(t);
        }
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

fn validate(args: ValidateArgs) -> anyhow::Result<i32> {
    match crate::yaml::resolve(
        &args.config_file,
        Some(PathBuf::from("/tmp/aiperf-validate")),
    ) {
        Ok(_) => {
            println!("{} is valid", args.config_file.display());
            Ok(0)
        }
        Err(e) => {
            eprintln!("aiperf: config invalid: {e:#}");
            Ok(1)
        }
    }
}

fn expand(args: ExpandArgs) -> anyhow::Result<i32> {
    let Some(path) = args.config else {
        anyhow::bail!("config expand requires --config <file>");
    };
    let mut base = crate::yaml::read_env_substituted(&path)?;
    let Some(sweep) = crate::sweep::yaml_sweep::parse(&base)? else {
        println!("No `sweep:` block — this config resolves to a single run.");
        return Ok(0);
    };
    crate::sweep::yaml_sweep::normalize_benchmark(&mut base);
    let variations = sweep.expand(&base)?;
    println!("Sweep expands to {} run(s):", variations.len());
    for v in &variations {
        println!("  [{}] {}  (dir: {})", v.index, v.label, v.dir_name);
    }
    Ok(0)
}

/// Remove leading SPDX and YAML-language-server comment lines.
pub(crate) fn strip_spdx_header(content: &str) -> String {
    let mut lines = content.lines().peekable();
    while let Some(line) = lines.peek() {
        let t = line.trim_start();
        if t.starts_with("# SPDX-") || t.starts_with("# yaml-language-server:") {
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

/// Replace the whole line of the first matching top-level-ish key, preserving its
/// indentation but not any trailing inline comment. Handles the shorthand
/// `model:`/`url:` and
/// plural `models:`/`urls:` (the latter written as a one-element inline list).
fn override_scalar(content: &str, keys: &[&str], value: &str) -> String {
    let mut out = Vec::new();
    let mut done = false;
    for line in content.lines() {
        if !done {
            let trimmed = line.trim_start();
            let indent = &line[..line.len() - trimmed.len()];
            for key in keys {
                let prefix = format!("{key}:");
                if trimmed.starts_with(&prefix) && !trimmed.starts_with('#') {
                    let plural = key.ends_with('s');
                    let rendered = if plural {
                        format!("{indent}{key}: [{value}]")
                    } else {
                        format!("{indent}{key}: {value}")
                    };
                    out.push(rendered);
                    done = true;
                    break;
                }
            }
            if done {
                continue;
            }
        }
        out.push(line.to_string());
    }
    let mut s = out.join("\n");
    if content.ends_with('\n') {
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_templates_embedded_nonempty() {
        assert!(TEMPLATES.len() >= 20);
        assert!(TEMPLATES.iter().any(|t| t.name == "minimal"));
        assert!(TEMPLATES.iter().all(|t| !t.content.is_empty()));
    }

    #[test]
    fn strip_header_removes_spdx() {
        let src = "# SPDX-FileCopyrightText: x\n# SPDX-License-Identifier: y\n# yaml-language-server: z\nschemaVersion: \"2.0\"\n";
        assert_eq!(strip_spdx_header(src), "schemaVersion: \"2.0\"\n");
    }

    #[test]
    fn override_model_and_url() {
        let src = "benchmark:\n  model: old-model\n  endpoint:\n    url: http://old\n";
        let s = override_scalar(src, &["model", "models"], "new-model");
        assert!(s.contains("model: new-model"), "{s}");
        let s = override_scalar(&s, &["url", "urls"], "http://new");
        assert!(s.contains("url: http://new"), "{s}");
    }
}
