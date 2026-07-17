// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! SPEED-Bench report generation from AIPerf run directories.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

const PROFILE_JSON: &str = "profile_export_aiperf.json";
const SERVER_METRICS_JSON: &str = "server_metrics_export.json";
const CATEGORY_PREFIXES: &[&str] = &["speed_bench_", "spec_al_"];

const QUALITATIVE_CATEGORIES: &[&str] = &[
    "coding",
    "humanities",
    "math",
    "multilingual",
    "qa",
    "rag",
    "reasoning",
    "roleplay",
    "stem",
    "summarization",
    "writing",
];
const THROUGHPUT_TIERS: &[&str] = &["low_entropy", "mixed", "high_entropy"];
const SPEC_AL_BENCHMARKS: &[&str] = &["gsm8k", "math500", "mtbench", "humaneval", "mbpp"];

const ACCEPT_LENGTH_METRICS: &[&str] = &[
    "sglang:spec_accept_length",
    "vllm:spec_decode_mean_accepted_length",
    "trtllm:spec_accept_length",
];
const ACCEPT_RATE_METRICS: &[&str] = &[
    "sglang:spec_accept_rate",
    "vllm:spec_decode_draft_acceptance_rate",
    "trtllm:spec_accept_rate",
];

fn find_run_dirs(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut run_dirs = Vec::new();
    for p in paths {
        if !p.is_dir() {
            eprintln!("Warning: {} is not a directory, skipping", p.display());
            continue;
        }
        if p.join(PROFILE_JSON).exists() {
            run_dirs.push(p.clone());
        } else if let Ok(rd) = std::fs::read_dir(p) {
            let mut children: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
            children.sort();
            for child in children {
                if child.is_dir() && child.join(PROFILE_JSON).exists() {
                    run_dirs.push(child);
                }
            }
        }
    }
    run_dirs
}

fn load_json(path: &Path) -> Option<Value> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn extract_category(profile: &Value) -> Option<String> {
    let datasets = profile.get("input_config")?.get("datasets")?.as_array()?;
    for entry in datasets {
        let name = entry
            .get("format")
            .or_else(|| entry.get("dataset"))
            .and_then(Value::as_str);
        if let Some(name) = name {
            for prefix in CATEGORY_PREFIXES {
                if let Some(cat) = name.strip_prefix(prefix) {
                    return Some(cat.to_string());
                }
            }
        }
    }
    None
}

fn extract_model(profile: &Value) -> String {
    profile
        .get("input_config")
        .and_then(|c| c.get("models"))
        .and_then(|m| m.get("items"))
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|e| {
                e.get("name")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn metric_stat(metrics: &Value, name: &str, stat: &str) -> Option<f64> {
    metrics
        .get(name)?
        .get("series")?
        .as_array()?
        .first()?
        .get("stats")?
        .get(stat)?
        .as_f64()
}

fn extract_accept_length(server: &Value) -> Option<f64> {
    let metrics = server.get("metrics")?;
    for name in ACCEPT_LENGTH_METRICS {
        if let Some(v) = metric_stat(metrics, name, "avg") {
            return Some(v);
        }
    }
    let accepted = metric_stat(metrics, "vllm:spec_decode_num_accepted_tokens", "total");
    let drafts = metric_stat(metrics, "vllm:spec_decode_num_drafts", "total");
    if let (Some(a), Some(d)) = (accepted, drafts) {
        if d > 0.0 {
            return Some(a / d + 1.0);
        }
    }
    if let Some(obj) = metrics.as_object() {
        for (name, data) in obj {
            let lower = name.to_lowercase();
            if lower.contains("spec") && lower.contains("accept") && lower.contains("length") {
                if let Some(v) = data
                    .get("series")
                    .and_then(Value::as_array)
                    .and_then(|s| s.first())
                    .and_then(|s| s.get("stats"))
                    .and_then(|s| s.get("avg"))
                    .and_then(Value::as_f64)
                {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn extract_accept_rate(server: &Value) -> Option<f64> {
    let metrics = server.get("metrics")?;
    for name in ACCEPT_RATE_METRICS {
        if let Some(v) = metric_stat(metrics, name, "avg") {
            return Some(v);
        }
    }
    let accepted = metric_stat(metrics, "vllm:spec_decode_num_accepted_tokens", "total");
    let draft_tokens = metric_stat(metrics, "vllm:spec_decode_num_draft_tokens", "total");
    if let (Some(a), Some(d)) = (accepted, draft_tokens) {
        if d > 0.0 {
            return Some(a / d);
        }
    }
    None
}

fn extract_throughput(profile: &Value) -> Option<f64> {
    profile.get("output_token_throughput")?.get("avg")?.as_f64()
}

/// `{model: {category: value}}` matrix (`build_report`).
fn build_report(
    run_dirs: &[PathBuf],
    metric: &str,
) -> BTreeMap<String, BTreeMap<String, Option<f64>>> {
    let mut results: BTreeMap<String, BTreeMap<String, Option<f64>>> = BTreeMap::new();
    for run_dir in run_dirs {
        let Some(profile) = load_json(&run_dir.join(PROFILE_JSON)) else {
            eprintln!(
                "Warning: no {PROFILE_JSON} in {}, skipping",
                run_dir.display()
            );
            continue;
        };
        let Some(category) = extract_category(&profile) else {
            eprintln!(
                "Warning: cannot determine category from {}, skipping",
                run_dir.display()
            );
            continue;
        };
        let model = extract_model(&profile);
        let value = match metric {
            "accept_length" | "accept_rate" => {
                match load_json(&run_dir.join(SERVER_METRICS_JSON)) {
                    Some(sm) => {
                        if metric == "accept_length" {
                            extract_accept_length(&sm)
                        } else {
                            extract_accept_rate(&sm)
                        }
                    }
                    None => {
                        eprintln!("Warning: no {SERVER_METRICS_JSON} in {}", run_dir.display());
                        None
                    }
                }
            }
            "throughput" => extract_throughput(&profile),
            _ => None,
        };
        let value = value.filter(|v| !v.is_nan());
        results.entry(model).or_default().insert(category, value);
    }
    results
}

fn detect_columns(results: &BTreeMap<String, BTreeMap<String, Option<f64>>>) -> Vec<String> {
    let all: BTreeSet<&String> = results.values().flat_map(|m| m.keys()).collect();
    let subset_of = |set: &[&str]| all.iter().all(|c| set.contains(&c.as_str()));
    let filtered = |order: &[&str]| -> Vec<String> {
        order
            .iter()
            .filter(|c| all.contains(&c.to_string()))
            .map(|c| c.to_string())
            .collect()
    };
    if subset_of(QUALITATIVE_CATEGORIES) {
        filtered(QUALITATIVE_CATEGORIES)
    } else if subset_of(THROUGHPUT_TIERS) {
        filtered(THROUGHPUT_TIERS)
    } else if subset_of(SPEC_AL_BENCHMARKS) {
        filtered(SPEC_AL_BENCHMARKS)
    } else {
        all.into_iter().cloned().collect()
    }
}

fn fmt2(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.2}")).unwrap_or_default()
}

fn write_csv(
    results: &BTreeMap<String, BTreeMap<String, Option<f64>>>,
    columns: &[String],
    output: &Path,
) -> anyhow::Result<()> {
    let mut out = String::new();
    out.push_str("Model");
    for c in columns {
        out.push(',');
        out.push_str(c);
    }
    out.push_str(",Overall\r\n");
    for (model, data) in results {
        out.push_str(model);
        let mut values = Vec::new();
        for col in columns {
            let v = data.get(col).copied().flatten();
            out.push(',');
            out.push_str(&fmt2(v));
            if let Some(x) = v {
                values.push(x);
            }
        }
        let overall = if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        };
        out.push(',');
        out.push_str(&fmt2(overall));
        out.push_str("\r\n");
    }
    std::fs::write(output, out).map_err(|e| anyhow::anyhow!("write {}: {e}", output.display()))?;
    println!("CSV written to {}", output.display());
    Ok(())
}

fn print_table(results: &BTreeMap<String, BTreeMap<String, Option<f64>>>, columns: &[String]) {
    let mut header = vec!["Model".to_string()];
    header.extend(columns.iter().cloned());
    header.push("Overall".to_string());
    let mut widths: Vec<usize> = header.iter().map(|h| h.len().max(8)).collect();
    widths[0] = widths[0].max(results.keys().map(String::len).max().unwrap_or(8));

    let rjust = |s: &str, w: usize| format!("{s:>w$}");
    let cells: Vec<String> = header
        .iter()
        .zip(&widths)
        .map(|(h, w)| rjust(h, *w))
        .collect();
    println!("{}", cells.join("  "));
    println!(
        "{}",
        widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for (model, data) in results {
        let mut row = vec![model.clone()];
        let mut values = Vec::new();
        for col in columns {
            let v = data.get(col).copied().flatten();
            row.push(
                v.map(|x| format!("{x:.2}"))
                    .unwrap_or_else(|| "-".to_string()),
            );
            if let Some(x) = v {
                values.push(x);
            }
        }
        let overall = if values.is_empty() {
            "-".to_string()
        } else {
            format!("{:.2}", values.iter().sum::<f64>() / values.len() as f64)
        };
        row.push(overall);
        let line: Vec<String> = row.iter().zip(&widths).map(|(c, w)| rjust(c, *w)).collect();
        println!("{}", line.join("  "));
    }
}

/// Run `aiperf speed-bench-report <paths...> [--output P] [--format F] [--metric M]`.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut output = PathBuf::from("speed_bench_report.csv");
    let mut format = "both".to_string();
    let mut metric = "accept_length".to_string();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--output" => {
                output = PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--output needs a value"))?,
                )
            }
            "--format" => {
                format = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--format needs a value"))?
                    .clone()
            }
            "--metric" => {
                metric = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--metric needs a value"))?
                    .clone()
            }
            other if other.starts_with('-') => {
                anyhow::bail!("unknown speed-bench-report flag {other:?}")
            }
            other => paths.push(PathBuf::from(other)),
        }
    }

    let run_dirs = find_run_dirs(&paths);
    if run_dirs.is_empty() {
        eprintln!("Error: no aiperf run directories found.");
        return Ok(1);
    }
    println!("Found {} run directories.", run_dirs.len());
    let results = build_report(&run_dirs, &metric);
    let has_value = results.values().any(|m| m.values().any(Option::is_some));
    if !has_value {
        eprintln!("Error: no SPEED-Bench results extracted.");
        return Ok(1);
    }
    let columns = detect_columns(&results);
    if format == "table" || format == "both" {
        print_table(&results, &columns);
    }
    if format == "csv" || format == "both" {
        write_csv(&results, &columns, &output)?;
    }
    Ok(0)
}
