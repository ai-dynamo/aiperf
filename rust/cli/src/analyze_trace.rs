// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Mooncake trace analysis for `aiperf analyze-trace`.
//!
//! Percentiles use linear interpolation and standard deviation uses the
//! population formula to preserve the command's JSON contract.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::Serialize;

use crate::stats::percentile_linear as percentile;

#[derive(Serialize)]
struct MetricStats {
    mean: f64,
    std_dev: f64,
    min: f64,
    p25: f64,
    median: f64,
    p75: f64,
    max: f64,
}

/// Field order is part of the serialized report contract.
#[derive(Serialize)]
struct AnalysisStats {
    total_requests: usize,
    unique_prefixes: usize,
    num_prefix_groups: usize,
    cache_hit_rate: f64,
    min_isl: i64,
    max_isl: i64,
    avg_isl: f64,
    min_osl: i64,
    max_osl: i64,
    avg_osl: f64,
    prefix_reuse_ratio: f64,
    isl_stats: Option<MetricStats>,
    osl_stats: Option<MetricStats>,
    context_length_stats: Option<MetricStats>,
    unique_prompt_length_stats: Option<MetricStats>,
    hit_rate_stats: Option<MetricStats>,
}

#[derive(serde::Deserialize)]
struct TraceRecord {
    #[serde(default)]
    input_length: i64,
    #[serde(default)]
    output_length: i64,
    #[serde(default)]
    hash_ids: Vec<i64>,
}

fn metric_stats(values: &[f64]) -> Option<MetricStats> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(MetricStats {
        mean,
        std_dev: var.sqrt(),
        min: sorted[0],
        p25: percentile(&sorted, 25.0),
        median: percentile(&sorted, 50.0),
        p75: percentile(&sorted, 75.0),
        max: sorted[sorted.len() - 1],
    })
}

fn analyze(records: &[TraceRecord], block_size: i64) -> AnalysisStats {
    let isls: Vec<i64> = records.iter().map(|r| r.input_length).collect();
    let osls: Vec<i64> = records.iter().map(|r| r.output_length).collect();
    let hash_ids_per_trace: Vec<&[i64]> = records.iter().map(|r| r.hash_ids.as_slice()).collect();

    let mut prefix_counter: HashMap<Vec<i64>, u64> = HashMap::new();
    let mut hash_position_counter: HashMap<(usize, i64), u64> = HashMap::new();
    for hash_ids in &hash_ids_per_trace {
        for i in 1..=hash_ids.len() {
            *prefix_counter.entry(hash_ids[..i].to_vec()).or_insert(0) += 1;
        }
        for (pos, &hid) in hash_ids.iter().enumerate() {
            *hash_position_counter.entry((pos, hid)).or_insert(0) += 1;
        }
    }

    let repeated: HashSet<(usize, i64)> = hash_position_counter
        .iter()
        .filter(|(_, c)| **c > 1)
        .map(|(k, _)| *k)
        .collect();
    let mut context_lengths: Vec<i64> = Vec::with_capacity(records.len());
    let mut unique_prompt_lengths: Vec<i64> = Vec::with_capacity(records.len());
    for (&isl, hash_ids) in isls.iter().zip(&hash_ids_per_trace) {
        if hash_ids.is_empty() {
            context_lengths.push(0);
            unique_prompt_lengths.push(isl);
            continue;
        }
        let all_repeated = hash_ids
            .iter()
            .enumerate()
            .all(|(pos, &hid)| repeated.contains(&(pos, hid)));
        let (context_len, unique_len) = if all_repeated {
            (isl, 0)
        } else {
            let repeated_count = hash_ids
                .iter()
                .enumerate()
                .filter(|(pos, hid)| repeated.contains(&(*pos, **hid)))
                .count() as i64;
            let context = repeated_count * block_size;
            (context, isl - context)
        };
        context_lengths.push(context_len);
        unique_prompt_lengths.push(unique_len);
    }

    // The first unseen block defines the hit fraction for an infinite cache.
    let mut seen: HashSet<i64> = HashSet::new();
    let mut hit_rates: Vec<f64> = Vec::new();
    for hash_ids in &hash_ids_per_trace {
        if hash_ids.is_empty() {
            continue;
        }
        let first_unseen = hash_ids
            .iter()
            .position(|hid| !seen.contains(hid))
            .unwrap_or(hash_ids.len());
        hit_rates.push(first_unseen as f64 / hash_ids.len() as f64);
        seen.extend(hash_ids.iter().copied());
    }

    let mut first_block_counts: HashMap<i64, u64> = HashMap::new();
    for hash_ids in &hash_ids_per_trace {
        if let Some(&first) = hash_ids.first() {
            *first_block_counts.entry(first).or_insert(0) += 1;
        }
    }
    let num_prefix_groups = first_block_counts.values().filter(|&&c| c > 1).count();

    let total_prefix: u64 = prefix_counter.values().sum();
    let reused_prefix: u64 = prefix_counter.values().filter(|&&c| c > 1).sum();
    let prefix_reuse_ratio = if total_prefix > 0 {
        reused_prefix as f64 / total_prefix as f64
    } else {
        0.0
    };

    let total = isls.len();
    let cache_hit_rate = if hit_rates.is_empty() {
        0.0
    } else {
        hit_rates.iter().sum::<f64>() / hit_rates.len() as f64
    };
    let as_f64 = |v: &[i64]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let avg = |v: &[i64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<i64>() as f64 / v.len() as f64
        }
    };

    AnalysisStats {
        total_requests: total,
        unique_prefixes: prefix_counter.len(),
        num_prefix_groups,
        cache_hit_rate,
        min_isl: isls.iter().copied().min().unwrap_or(0),
        max_isl: isls.iter().copied().max().unwrap_or(0),
        avg_isl: avg(&isls),
        min_osl: osls.iter().copied().min().unwrap_or(0),
        max_osl: osls.iter().copied().max().unwrap_or(0),
        avg_osl: avg(&osls),
        prefix_reuse_ratio,
        isl_stats: metric_stats(&as_f64(&isls)),
        osl_stats: metric_stats(&as_f64(&osls)),
        context_length_stats: metric_stats(&as_f64(&context_lengths)),
        unique_prompt_length_stats: metric_stats(&as_f64(&unique_prompt_lengths)),
        hit_rate_stats: metric_stats(&hit_rates),
    }
}

fn read_trace(path: &std::path::Path) -> anyhow::Result<Vec<TraceRecord>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
    let mut records = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str::<TraceRecord>(line)
                .map_err(|e| anyhow::anyhow!("bad trace line: {e}"))?,
        );
    }
    Ok(records)
}

fn to_pretty_json(stats: &AnalysisStats) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(stats)?)
}

/// Run `aiperf analyze-trace <input_file> [--block-size N] [--output-file P]`.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut input_file: Option<PathBuf> = None;
    let mut block_size: i64 = 512;
    let mut output_file: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--block-size" => {
                block_size = it
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--block-size needs a value"))?
                    .parse()?;
            }
            "--output-file" => {
                output_file =
                    Some(PathBuf::from(it.next().ok_or_else(|| {
                        anyhow::anyhow!("--output-file needs a value")
                    })?));
            }
            other if other.starts_with('-') => {
                anyhow::bail!("unknown analyze-trace flag {other:?}");
            }
            other => input_file = Some(PathBuf::from(other)),
        }
    }
    let input_file =
        input_file.ok_or_else(|| anyhow::anyhow!("analyze-trace requires an input file"))?;
    if !input_file.exists() {
        println!("Error: Input file not found: {}", input_file.display());
        return Ok(0);
    }

    let records = read_trace(&input_file)?;
    let stats = analyze(&records, block_size);

    println!("\nTrace Analysis Report");
    println!("Total requests:   {}", stats.total_requests);
    println!("Unique prefixes:  {}", stats.unique_prefixes);
    println!("Prefix groups:    {}", stats.num_prefix_groups);
    println!("Cache hit rate:   {:.4}", stats.cache_hit_rate);
    println!(
        "ISL avg/min/max:  {:.1} / {} / {}",
        stats.avg_isl, stats.min_isl, stats.max_isl
    );
    println!(
        "OSL avg/min/max:  {:.1} / {} / {}\n",
        stats.avg_osl, stats.min_osl, stats.max_osl
    );

    if let Some(out) = &output_file {
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(out, to_pretty_json(&stats)?)
            .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", out.display()))?;
        println!("Analysis report saved to {}", out.display());
    }
    Ok(0)
}
