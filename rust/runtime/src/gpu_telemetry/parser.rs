// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DCGM Prometheus exposition decoding.
//!
//! The supported-field filtering, one-timestamp-per-scrape rule, `_total`
//! normalization, metadata extraction, finite-value filtering, and collector
//! scaling are implemented here.

use std::collections::BTreeMap;

use crate::gpu_telemetry::fields::dcgm_metric_spec;
use crate::gpu_telemetry::model::{GpuMetadata, GpuScrape, GpuTelemetryRecord};
use crate::gpu_telemetry::source::GpuTelemetryError;

/// Extension seam for turning one metrics payload into normalized GPU records.
pub trait GpuTelemetryDecoder {
    /// Decodes `body` using one Clock timestamp and a credential-free endpoint.
    fn decode(
        &self,
        endpoint_url: &str,
        timestamp_ns: i64,
        body: &str,
    ) -> Result<GpuScrape, GpuTelemetryError>;
}

/// Decoder for the Prometheus text emitted by NVIDIA's DCGM exporter.
#[derive(Debug, Default)]
pub struct DcgmPrometheusDecoder;

#[derive(Debug, Default)]
struct PendingGpu {
    labels: BTreeMap<String, String>,
    metrics: BTreeMap<String, f64>,
}

impl GpuTelemetryDecoder for DcgmPrometheusDecoder {
    fn decode(
        &self,
        endpoint_url: &str,
        timestamp_ns: i64,
        body: &str,
    ) -> Result<GpuScrape, GpuTelemetryError> {
        let mut by_gpu = BTreeMap::<i32, PendingGpu>::new();
        for (offset, raw_line) in body.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some(sample) = parse_sample(line).map_err(|message| GpuTelemetryError::Parse {
                line: offset + 1,
                message,
            })?
            else {
                continue;
            };
            if !sample.value.is_finite() {
                continue;
            }
            let Some(gpu_index) = sample
                .labels
                .get("gpu")
                .and_then(|value| value.parse::<i32>().ok())
            else {
                continue;
            };
            let base_name = sample.name.strip_suffix("_total").unwrap_or(&sample.name);
            let Some(spec) = dcgm_metric_spec(base_name) else {
                continue;
            };
            let gpu = by_gpu.entry(gpu_index).or_default();
            if gpu.labels.is_empty() {
                gpu.labels = sample.labels;
            }
            gpu.metrics
                .insert(spec.name.to_string(), sample.value * spec.scale);
        }

        let records = by_gpu
            .into_iter()
            .filter_map(|(gpu_index, pending)| {
                (!pending.metrics.is_empty()).then(|| {
                    let gpu_uuid = pending
                        .labels
                        .get("UUID")
                        .cloned()
                        .unwrap_or_else(|| format!("GPU-unknown-{gpu_index}"));
                    let gpu_model_name = pending
                        .labels
                        .get("modelName")
                        .cloned()
                        .unwrap_or_else(|| "Unknown GPU".to_string());
                    GpuTelemetryRecord {
                        timestamp_ns,
                        endpoint_url: endpoint_url.to_string(),
                        metadata: GpuMetadata {
                            gpu_index,
                            gpu_uuid,
                            gpu_model_name,
                            pci_bus_id: pending.labels.get("pci_bus_id").cloned(),
                            device: pending.labels.get("device").cloned(),
                            hostname: pending.labels.get("Hostname").cloned(),
                            namespace: pending.labels.get("namespace").cloned(),
                            pod_name: pending.labels.get("pod").cloned(),
                        },
                        metrics: pending.metrics,
                    }
                })
            })
            .collect();

        Ok(GpuScrape {
            timestamp_ns,
            endpoint_url: endpoint_url.to_string(),
            records,
        })
    }
}

struct ParsedSample {
    name: String,
    labels: BTreeMap<String, String>,
    value: f64,
}

fn parse_sample(line: &str) -> Result<Option<ParsedSample>, String> {
    let Some(split) = sample_value_split(line) else {
        return Err("sample has no value".to_string());
    };
    let metric = line[..split].trim();
    let value_text = line[split..]
        .split_whitespace()
        .next()
        .ok_or_else(|| "sample has no value".to_string())?;
    let value = value_text
        .parse::<f64>()
        .map_err(|error| format!("invalid sample value {value_text:?}: {error}"))?;
    let (name, labels) = parse_metric_and_labels(metric)?;
    if name.is_empty() {
        return Ok(None);
    }
    Ok(Some(ParsedSample {
        name,
        labels,
        value,
    }))
}

fn sample_value_split(line: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut escaped = false;
    let mut brace_depth = 0_u32;
    for (index, byte) in line.bytes().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' if in_quotes => escaped = true,
            b'"' => in_quotes = !in_quotes,
            b'{' if !in_quotes => brace_depth += 1,
            b'}' if !in_quotes => brace_depth = brace_depth.saturating_sub(1),
            b' ' | b'\t' if !in_quotes && brace_depth == 0 => return Some(index),
            _ => {}
        }
    }
    None
}

fn parse_metric_and_labels(metric: &str) -> Result<(String, BTreeMap<String, String>), String> {
    let Some(open) = metric.find('{') else {
        return Ok((metric.to_string(), BTreeMap::new()));
    };
    if !metric.ends_with('}') {
        return Err("unterminated label set".to_string());
    }
    let name = metric[..open].to_string();
    let labels = parse_labels(&metric[open + 1..metric.len() - 1])?;
    Ok((name, labels))
}

fn parse_labels(mut input: &str) -> Result<BTreeMap<String, String>, String> {
    let mut labels = BTreeMap::new();
    while !input.trim_start().is_empty() {
        input = input.trim_start();
        let equals = input
            .find('=')
            .ok_or_else(|| "label has no '='".to_string())?;
        let name = input[..equals].trim();
        if name.is_empty() {
            return Err("label name is empty".to_string());
        }
        input = input[equals + 1..].trim_start();
        let Some(rest) = input.strip_prefix('"') else {
            return Err(format!("label {name:?} has an unquoted value"));
        };
        let (value, consumed) = parse_quoted_label(rest)?;
        labels.insert(name.to_string(), value);
        input = rest[consumed..].trim_start();
        if input.is_empty() {
            break;
        }
        let Some(rest) = input.strip_prefix(',') else {
            return Err("labels must be comma-separated".to_string());
        };
        input = rest;
    }
    Ok(labels)
}

fn parse_quoted_label(input: &str) -> Result<(String, usize), String> {
    let mut output = String::new();
    let mut escaped = false;
    for (index, character) in input.char_indices() {
        if escaped {
            output.push(match character {
                'n' => '\n',
                '\\' => '\\',
                '"' => '"',
                other => other,
            });
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '"' => return Ok((output, index + character.len_utf8())),
            other => output.push(other),
        }
    }
    Err("unterminated quoted label".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_scales_groups_and_strips_total() {
        let body = r#"
# HELP DCGM_FI_DEV_POWER_USAGE Power
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-a",modelName="H100",Hostname="n1"} 250
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total{gpu="0",UUID="GPU-a",modelName="H100"} 2000000000
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-a",modelName="H100"} 1024
DCGM_FI_PROF_SM_ACTIVE{gpu="0",UUID="GPU-a",modelName="H100"} 0.75
DCGM_FI_DEV_POWER_VIOLATION_total{gpu="0",UUID="GPU-a",modelName="H100"} 2000
DCGM_FI_DEV_POWER_USAGE{gpu="bad",UUID="ignored"} 1
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-b",modelName="H200"} NaN
"#;
        let scrape = DcgmPrometheusDecoder
            .decode("http://dcgm/metrics", 42, body)
            .unwrap();
        assert_eq!(scrape.records.len(), 1);
        let record = &scrape.records[0];
        assert_eq!(record.timestamp_ns, 42);
        assert_eq!(record.metadata.gpu_uuid, "GPU-a");
        assert_eq!(record.metrics["gpu_power_usage"], 250.0);
        assert_eq!(record.metrics["energy_consumption"], 2.0);
        assert_eq!(record.metrics["gpu_memory_used"], 1.073_741_824);
        assert_eq!(record.metrics["sm_utilization"], 75.0);
        assert_eq!(record.metrics["power_violation"], 2.0);
    }

    #[test]
    fn decoder_handles_escaped_labels_and_rejects_malformed_samples() {
        let body = r#"DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-\"a",modelName="H100\\SXM"} 1"#;
        let scrape = DcgmPrometheusDecoder
            .decode("http://dcgm/metrics", 1, body)
            .unwrap();
        assert_eq!(scrape.records[0].metadata.gpu_uuid, "GPU-\"a");
        assert_eq!(scrape.records[0].metadata.gpu_model_name, "H100\\SXM");

        let error = DcgmPrometheusDecoder
            .decode(
                "http://dcgm/metrics",
                1,
                "DCGM_FI_DEV_POWER_USAGE{gpu=\"0\" 1",
            )
            .unwrap_err();
        assert!(matches!(error, GpuTelemetryError::Parse { line: 1, .. }));
    }
}
