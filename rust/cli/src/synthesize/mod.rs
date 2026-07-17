// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf synthesize agentic-code`.
//!
//! `dataset.jsonl` preserves seeded draw order, Mooncake field order, compact
//! separators, and one-decimal float formatting. Run-directory timestamps are
//! intentionally outside the deterministic contract.

pub mod config;
pub mod dist;
pub mod prefix;
pub mod synth;

use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;

use config::SessionDistributionConfig;
use synth::{SessionSynthesizer, SynthesizedSession};

/// Parsed `synthesize agentic-code` options.
struct Options {
    num_sessions: usize,
    output: PathBuf,
    config: Option<String>,
    seed: u64,
    max_isl: Option<i64>,
    max_osl: Option<i64>,
}

/// Run `aiperf synthesize <dataset> [flags]`. Only `agentic-code` is native.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut dataset: Option<String> = None;
    let mut num_sessions: usize = 1000;
    let mut output = PathBuf::from(".");
    let mut config: Option<String> = None;
    let mut seed: u64 = 42;
    let mut max_isl: Option<i64> = None;
    let mut max_osl: Option<i64> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--num-sessions" => {
                num_sessions = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--num-sessions needs a value"))?
                    .parse()?;
            }
            "--output" => {
                output = PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--output needs a value"))?,
                );
            }
            "--config" => {
                config = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--config needs a value"))?
                        .clone(),
                );
            }
            "--seed" => {
                seed = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--seed needs a value"))?
                    .parse()?;
            }
            "--max-isl" => {
                max_isl = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--max-isl needs a value"))?
                        .parse()?,
                );
            }
            "--max-osl" => {
                max_osl = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--max-osl needs a value"))?
                        .parse()?,
                );
            }
            other if other.starts_with('-') => {
                anyhow::bail!("unknown synthesize flag {other:?}")
            }
            other => dataset = Some(other.to_string()),
        }
    }

    match dataset.as_deref() {
        Some("agentic-code") => {}
        Some(d) => anyhow::bail!("unknown synthesize dataset {d:?} (expected agentic-code)"),
        None => anyhow::bail!("synthesize requires a dataset name (agentic-code)"),
    }

    synthesize(Options {
        num_sessions,
        output,
        config,
        seed,
        max_isl,
        max_osl,
    })
}

fn synthesize(opts: Options) -> anyhow::Result<i32> {
    let (mut cfg, config_name) = match &opts.config {
        Some(path) => {
            let cfg = SessionDistributionConfig::load(path)?;
            let name = config_stem(path);
            (cfg, name)
        }
        None => (SessionDistributionConfig::default(), "default".to_string()),
    };

    if let Some(isl) = opts.max_isl {
        cfg.max_prompt_tokens = isl;
    }
    if let Some(osl) = opts.max_osl {
        cfg.generation_length.max = Some(osl as f64);
    }

    let timestamp = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let run_dir_name = format!(
        "{config_name}_{}s_seed{}_{timestamp}",
        opts.num_sessions, opts.seed
    );
    let run_dir = opts.output.join(run_dir_name);
    std::fs::create_dir_all(&run_dir)
        .map_err(|e| anyhow::anyhow!("failed to create {}: {e}", run_dir.display()))?;

    let mut synth = SessionSynthesizer::new(&cfg, opts.seed)?;
    println!(
        "Generating {} sessions (seed={})...",
        opts.num_sessions, opts.seed
    );
    let sessions = synth.synthesize_sessions(opts.num_sessions)?;

    let jsonl_path = run_dir.join("dataset.jsonl");
    write_jsonl(&sessions, &jsonl_path, cfg.block_size)?;

    let manifest_path = run_dir.join("manifest.json");
    write_manifest(
        &manifest_path,
        &cfg,
        opts.seed,
        sessions.len(),
        &config_name,
    )?;

    let validated_rows = validate_or_exit(&jsonl_path)?;

    let total_turns: usize = sessions.iter().map(|s| s.turns.len()).sum();
    println!("Run directory: {}", run_dir.display());
    println!(
        "  JSONL:           {} ({total_turns} turns)",
        jsonl_path.display()
    );
    println!("  Manifest:        {}", manifest_path.display());
    println!("  Validation:      Mooncake trace ({validated_rows} rows)");

    Ok(0)
}

/// Write Mooncake JSONL with contract-defined field order and compact separators.
fn write_jsonl(
    sessions: &[SynthesizedSession],
    path: &Path,
    block_size: i64,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("failed to create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(file);
    let mut next_hash_id: i64 = 0;
    let mut buf = String::new();
    for session in sessions {
        for turn in &session.turns {
            let n_blocks = div_ceil(turn.new_tokens, block_size);
            let hash_ids: Vec<i64> = if turn.turn_index == 0 {
                if let Some(&mx) = turn.hash_ids.iter().max() {
                    next_hash_id = next_hash_id.max(mx + 1);
                }
                turn.hash_ids.clone()
            } else {
                let ids: Vec<i64> = (next_hash_id..next_hash_id + n_blocks).collect();
                next_hash_id += n_blocks;
                ids
            };

            buf.clear();
            buf.push_str("{\"session_id\":");
            push_json_str(&mut buf, &session.session_id);
            buf.push_str(",\"input_length\":");
            buf.push_str(&turn.new_tokens.to_string());
            buf.push_str(",\"output_length\":");
            buf.push_str(&turn.output_length.to_string());
            buf.push_str(",\"hash_ids\":");
            push_int_array(&mut buf, &hash_ids);
            if turn.turn_index == 0 {
                buf.push_str(",\"timestamp\":");
                buf.push_str(&fmt_round1(turn.timestamp_ms));
                buf.push_str(",\"group_id\":");
                buf.push_str(&session.group_id.to_string());
                if session.is_restart_continuation {
                    buf.push_str(",\"is_restart\":true");
                }
            } else {
                buf.push_str(",\"delay\":");
                buf.push_str(&fmt_round1(turn.delay_ms));
            }
            buf.push('}');
            buf.push('\n');
            w.write_all(buf.as_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

fn push_json_str(buf: &mut String, s: &str) {
    buf.push('"');
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            _ => buf.push(c),
        }
    }
    buf.push('"');
}

fn push_int_array(buf: &mut String, ids: &[i64]) {
    buf.push('[');
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&id.to_string());
    }
    buf.push(']');
}

/// Format one decimal using round-half-to-even.
fn fmt_round1(x: f64) -> String {
    format!("{x:.1}")
}

/// Emit `manifest.json`; only `dataset.jsonl` is byte-exact.
fn write_manifest(
    path: &Path,
    cfg: &SessionDistributionConfig,
    seed: u64,
    num_sessions: usize,
    config_name: &str,
) -> anyhow::Result<()> {
    let manifest = serde_json::json!({
        "seed": seed,
        "num_sessions": num_sessions,
        "config_name": config_name,
        "generation_params": generation_params_json(cfg),
    });
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(path, bytes)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", path.display()))?;
    Ok(())
}

fn generation_params_json(cfg: &SessionDistributionConfig) -> serde_json::Value {
    let ln = |p: &config::LognormalParams| {
        serde_json::json!({
            "mu": p.mu, "sigma": p.sigma, "mean": p.mean, "median": p.median,
            "min": p.min, "max": p.max,
        })
    };
    serde_json::json!({
        "new_tokens_per_turn": {
            "mu": cfg.new_tokens_per_turn.params.mu,
            "sigma": cfg.new_tokens_per_turn.params.sigma,
            "mean": cfg.new_tokens_per_turn.params.mean,
            "median": cfg.new_tokens_per_turn.params.median,
            "min": cfg.new_tokens_per_turn.params.min,
            "max": cfg.new_tokens_per_turn.params.max,
            "bias": cfg.new_tokens_per_turn.bias,
        },
        "generation_length": ln(&cfg.generation_length),
        "inter_turn_delay": {
            "agentic_fraction": cfg.inter_turn_delay.agentic_fraction,
            "agentic_delay": ln(&cfg.inter_turn_delay.agentic_delay),
            "human_delay": ln(&cfg.inter_turn_delay.human_delay),
            "max": cfg.inter_turn_delay.max,
        },
        "reset": cfg.reset.as_ref().map(|r| serde_json::json!({
            "base_probability": r.base_probability,
            "context_scaling": r.context_scaling,
        })),
        "max_prompt_tokens": cfg.max_prompt_tokens,
        "block_size": cfg.block_size,
        "cache": {
            "layer1_tokens": cfg.cache.layer1_tokens,
            "layer1_5_tokens": cfg.cache.layer1_5_tokens,
            "layer2": ln(&cfg.cache.layer2),
            "layer1_5_groups": {
                "num_groups": cfg.cache.layer1_5_groups.num_groups,
                "zipf_alpha": cfg.cache.layer1_5_groups.zipf_alpha,
            },
        },
        "restart_initial_probability": cfg.restart_initial_probability,
        "restart_turn_range": cfg.restart_turn_range,
    })
}

fn validate_or_exit(path: &Path) -> anyhow::Result<usize> {
    let (line_count, errors) = crate::validate::validate_mooncake_public(path)?;
    if !errors.is_empty() {
        println!("Validation failed with {} error(s):", errors.len());
        for err in &errors {
            println!("  {err}");
        }
        std::process::exit(1);
    }
    Ok(line_count)
}

fn config_stem(path: &str) -> String {
    let p = Path::new(path);
    if p.is_file() {
        p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string())
    } else {
        path.to_string()
    }
}

fn div_ceil(a: i64, b: i64) -> i64 {
    if a <= 0 { 0 } else { (a + b - 1) / b }
}
