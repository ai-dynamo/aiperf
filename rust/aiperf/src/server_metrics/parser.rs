// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Classic Prometheus and strict OpenMetrics text parsing.
//!
//! Family routing, metadata/summary skips, label de-duplication, sample-level
//! `_created` filtering, and histogram assembly are implemented here. Unlike the
//! inherited ZMQ path, one non-finite histogram field is dropped without
//! tainting the remaining label series, as required by the native design.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

use crate::server_metrics::model::{
    HistogramValue, MetricFamily, MetricSample, PrometheusMetricType,
};

/// Malformed exposition input with a one-based line location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsParseError {
    /// One-based line number, or zero for a body-level OpenMetrics error.
    pub line: usize,
    /// Human-readable parser detail.
    pub message: String,
}

impl Display for MetricsParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        if self.line == 0 {
            formatter.write_str(&self.message)
        } else {
            write!(formatter, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for MetricsParseError {}

/// Object-safe parser seam for exposition dialects.
pub trait MetricsTextParser {
    /// Parses classic Prometheus text exposition.
    fn parse_classic(
        &self,
        body: &str,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError>;

    /// Parses strict OpenMetrics text exposition.
    fn parse_openmetrics(
        &self,
        body: &str,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError>;
}

/// Small parser tailored to AIPerf's server-metrics semantics.
#[derive(Debug, Default)]
pub struct PrometheusTextParser;

impl MetricsTextParser for PrometheusTextParser {
    fn parse_classic(
        &self,
        body: &str,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError> {
        parse_body(body, false)
    }

    fn parse_openmetrics(
        &self,
        body: &str,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError> {
        parse_body(body, true)
    }
}

#[derive(Debug, Clone)]
struct Metadata {
    family_name: String,
    metric_type: PrometheusMetricType,
    description: String,
}

#[derive(Debug, Default)]
struct HistogramBuilder {
    labels: BTreeMap<String, String>,
    buckets: BTreeMap<String, f64>,
    sum: Option<f64>,
    count: Option<f64>,
}

type LabelKey = Vec<(String, String)>;
type ScalarSamples = BTreeMap<LabelKey, (BTreeMap<String, String>, f64)>;
type HistogramSamples = BTreeMap<LabelKey, HistogramBuilder>;

#[derive(Debug)]
enum FamilyBuilderData {
    Scalar(ScalarSamples),
    Histogram(HistogramSamples),
}

#[derive(Debug)]
struct FamilyBuilder {
    metric_type: PrometheusMetricType,
    description: String,
    data: FamilyBuilderData,
}

impl FamilyBuilder {
    fn new(metric_type: PrometheusMetricType, description: String) -> Self {
        let data = if metric_type == PrometheusMetricType::Histogram {
            FamilyBuilderData::Histogram(BTreeMap::new())
        } else {
            FamilyBuilderData::Scalar(BTreeMap::new())
        };
        Self {
            metric_type,
            description,
            data,
        }
    }

    fn finish(self) -> Option<MetricFamily> {
        let samples = match self.data {
            FamilyBuilderData::Scalar(samples) => samples
                .into_values()
                .map(|(labels, value)| MetricSample::Scalar { labels, value })
                .collect::<Vec<_>>(),
            FamilyBuilderData::Histogram(samples) => samples
                .into_values()
                .filter(|sample| {
                    !sample.buckets.is_empty() || sample.sum.is_some() || sample.count.is_some()
                })
                .map(|sample| MetricSample::Histogram {
                    labels: sample.labels,
                    value: HistogramValue {
                        buckets: sample.buckets,
                        sum: sample.sum,
                        count: sample.count,
                    },
                })
                .collect::<Vec<_>>(),
        };
        (!samples.is_empty()).then_some(MetricFamily {
            metric_type: self.metric_type,
            description: self.description,
            samples,
        })
    }
}

fn parse_body(
    body: &str,
    strict_openmetrics: bool,
) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError> {
    if strict_openmetrics {
        let last = body.lines().rev().find(|line| !line.trim().is_empty());
        if last.map(str::trim) != Some("# EOF") {
            return Err(MetricsParseError {
                line: 0,
                message: "OpenMetrics body must terminate with '# EOF'".to_string(),
            });
        }
    }

    let mut declared_types = BTreeMap::<String, PrometheusMetricType>::new();
    let mut descriptions = BTreeMap::<String, String>::new();
    for (offset, raw_line) in body.lines().enumerate() {
        let line = raw_line.trim();
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let mut fields = rest.split_whitespace();
            let name = fields
                .next()
                .ok_or_else(|| parse_error(offset, "TYPE has no name"))?;
            let kind = fields
                .next()
                .ok_or_else(|| parse_error(offset, "TYPE has no metric type"))?;
            let metric_type = parse_metric_type(kind)
                .ok_or_else(|| parse_error(offset, format!("unsupported TYPE {kind:?}")))?;
            declared_types.insert(name.to_string(), metric_type);
        } else if let Some(rest) = line.strip_prefix("# HELP ") {
            let split = rest
                .find(char::is_whitespace)
                .ok_or_else(|| parse_error(offset, "HELP has no description"))?;
            descriptions.insert(
                rest[..split].to_string(),
                unescape_help(rest[split..].trim_start()),
            );
        }
    }

    let metadata = build_metadata(&declared_types, &descriptions);
    let mut builders = BTreeMap::<(String, PrometheusMetricType), FamilyBuilder>::new();
    for (offset, raw_line) in body.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parsed = parse_sample(line).map_err(|message| parse_error(offset, message))?;
        if parsed.name.ends_with("_created") {
            continue;
        }
        let (family_name, metric_type, description) = resolve_family(&parsed.name, &metadata);
        if should_skip_family(&family_name) || metric_type == PrometheusMetricType::Summary {
            continue;
        }
        let builder = builders
            .entry((family_name.clone(), metric_type))
            .or_insert_with(|| FamilyBuilder::new(metric_type, description));
        if !parsed.value.is_finite() {
            continue;
        }

        match &mut builder.data {
            FamilyBuilderData::Scalar(samples) => {
                let key = parsed
                    .labels
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect();
                samples.insert(key, (parsed.labels, parsed.value));
            }
            FamilyBuilderData::Histogram(samples) => {
                let mut labels = parsed.labels;
                let le = labels.remove("le");
                let key = labels
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect();
                let histogram = samples.entry(key).or_insert_with(|| HistogramBuilder {
                    labels,
                    ..HistogramBuilder::default()
                });
                if parsed.name.ends_with("_bucket") {
                    histogram
                        .buckets
                        .insert(le.unwrap_or_else(|| "+Inf".to_string()), parsed.value);
                } else if parsed.name.ends_with("_sum") {
                    histogram.sum = Some(parsed.value);
                } else if parsed.name.ends_with("_count") {
                    histogram.count = Some(parsed.value);
                }
            }
        }
    }

    let mut families = BTreeMap::<String, MetricFamily>::new();
    for ((name, _), builder) in builders {
        let Some(family) = builder.finish() else {
            continue;
        };
        match families.entry(name) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(family);
            }
            std::collections::btree_map::Entry::Occupied(mut entry)
                if family_priority(family.metric_type)
                    > family_priority(entry.get().metric_type) =>
            {
                entry.insert(family);
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
    }
    Ok(families)
}

fn build_metadata(
    declared_types: &BTreeMap<String, PrometheusMetricType>,
    descriptions: &BTreeMap<String, String>,
) -> BTreeMap<String, Metadata> {
    let mut metadata = BTreeMap::new();
    for (raw_name, metric_type) in declared_types {
        let normalized = normalize_declared_name(raw_name, *metric_type);
        let description = descriptions
            .get(raw_name)
            .or_else(|| descriptions.get(&normalized))
            .cloned()
            .unwrap_or_default();
        metadata.insert(
            raw_name.clone(),
            Metadata {
                family_name: normalized,
                metric_type: *metric_type,
                description,
            },
        );
    }
    metadata
}

fn normalize_declared_name(name: &str, metric_type: PrometheusMetricType) -> String {
    if metric_type == PrometheusMetricType::Counter {
        name.strip_suffix("_total").unwrap_or(name).to_string()
    } else {
        name.to_string()
    }
}

fn resolve_family(
    sample_name: &str,
    metadata: &BTreeMap<String, Metadata>,
) -> (String, PrometheusMetricType, String) {
    for suffix in ["_bucket", "_sum", "_count"] {
        if let Some(base) = sample_name.strip_suffix(suffix)
            && metadata
                .get(base)
                .is_some_and(|meta| meta.metric_type == PrometheusMetricType::Histogram)
        {
            let meta = &metadata[base];
            return (
                meta.family_name.clone(),
                PrometheusMetricType::Histogram,
                meta.description.clone(),
            );
        }
    }
    if let Some(meta) = metadata.get(sample_name) {
        return (
            meta.family_name.clone(),
            meta.metric_type,
            meta.description.clone(),
        );
    }
    if let Some(base) = sample_name.strip_suffix("_total")
        && metadata
            .get(base)
            .is_some_and(|meta| meta.metric_type == PrometheusMetricType::Counter)
    {
        let meta = &metadata[base];
        return (
            meta.family_name.clone(),
            PrometheusMetricType::Counter,
            meta.description.clone(),
        );
    }
    if let Some(meta) = metadata
        .values()
        .find(|meta| meta.family_name == sample_name)
    {
        return (
            meta.family_name.clone(),
            meta.metric_type,
            meta.description.clone(),
        );
    }
    (
        sample_name.to_string(),
        PrometheusMetricType::Unknown,
        String::new(),
    )
}

fn family_priority(metric_type: PrometheusMetricType) -> u8 {
    match metric_type {
        PrometheusMetricType::Counter => 4,
        PrometheusMetricType::Histogram => 3,
        PrometheusMetricType::Gauge => 2,
        PrometheusMetricType::Unknown => 1,
        PrometheusMetricType::Summary => 0,
    }
}

fn should_skip_family(name: &str) -> bool {
    name.ends_with("_created") || name.ends_with("_uptime") || name.contains("_uptime_")
}

fn parse_metric_type(value: &str) -> Option<PrometheusMetricType> {
    match value.to_ascii_lowercase().as_str() {
        "counter" => Some(PrometheusMetricType::Counter),
        "gauge" => Some(PrometheusMetricType::Gauge),
        "histogram" | "gaugehistogram" => Some(PrometheusMetricType::Histogram),
        "summary" => Some(PrometheusMetricType::Summary),
        "untyped" | "unknown" => Some(PrometheusMetricType::Unknown),
        _ => None,
    }
}

struct ParsedSample {
    name: String,
    labels: BTreeMap<String, String>,
    value: f64,
}

fn parse_sample(line: &str) -> Result<ParsedSample, String> {
    let split = sample_value_split(line).ok_or_else(|| "sample has no value".to_string())?;
    let metric = line[..split].trim();
    let value_text = line[split..]
        .split_whitespace()
        .next()
        .ok_or_else(|| "sample has no value".to_string())?;
    let value = parse_prometheus_float(value_text)?;
    let (name, labels) = parse_metric_and_labels(metric)?;
    validate_metric_name(&name)?;
    Ok(ParsedSample {
        name,
        labels,
        value,
    })
}

fn parse_prometheus_float(value: &str) -> Result<f64, String> {
    match value {
        "+Inf" | "Inf" => Ok(f64::INFINITY),
        "-Inf" => Ok(f64::NEG_INFINITY),
        "NaN" => Ok(f64::NAN),
        _ => value
            .parse::<f64>()
            .map_err(|error| format!("invalid sample value {value:?}: {error}")),
    }
}

fn validate_metric_name(name: &str) -> Result<(), String> {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return Err("metric name is empty".to_string());
    };
    if !(first.is_ascii_alphabetic() || matches!(first, b'_' | b':'))
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':'))
    {
        return Err(format!("invalid metric name {name:?}"));
    }
    Ok(())
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
    Ok((
        metric[..open].to_string(),
        parse_labels(&metric[open + 1..metric.len() - 1])?,
    ))
}

fn parse_labels(mut input: &str) -> Result<BTreeMap<String, String>, String> {
    let mut labels = BTreeMap::new();
    while !input.trim_start().is_empty() {
        input = input.trim_start();
        let equals = input
            .find('=')
            .ok_or_else(|| "label has no '='".to_string())?;
        let name = input[..equals].trim();
        validate_label_name(name)?;
        input = input[equals + 1..].trim_start();
        let rest = input
            .strip_prefix('"')
            .ok_or_else(|| format!("label {name:?} has an unquoted value"))?;
        let (value, consumed) = parse_quoted_label(rest)?;
        labels.insert(name.to_string(), value);
        input = rest[consumed..].trim_start();
        if input.is_empty() {
            break;
        }
        input = input
            .strip_prefix(',')
            .ok_or_else(|| "labels must be comma-separated".to_string())?;
    }
    Ok(labels)
}

fn validate_label_name(name: &str) -> Result<(), String> {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return Err("label name is empty".to_string());
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(format!("invalid label name {name:?}"));
    }
    Ok(())
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

fn unescape_help(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut escaped = false;
    for character in input.chars() {
        if escaped {
            output.push(if character == 'n' { '\n' } else { character });
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else {
            output.push(character);
        }
    }
    if escaped {
        output.push('\\');
    }
    output
}

fn parse_error(offset: usize, message: impl Into<String>) -> MetricsParseError {
    MetricsParseError {
        line: offset + 1,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_parser_normalizes_deduplicates_and_structures_histograms() {
        let body = r#"
# HELP vllm:prompt_tokens_total Prompt tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model="a"} 10
vllm:prompt_tokens_total{model="a"} 12
vllm:prompt_tokens_created{model="a"} 123
# HELP vllm:request_latency_seconds Request latency in seconds.
# TYPE vllm:request_latency_seconds histogram
vllm:request_latency_seconds_bucket{model="a",le="0.1"} 4
vllm:request_latency_seconds_bucket{model="a",le="1.0"} NaN
vllm:request_latency_seconds_bucket{model="a",le="+Inf"} 5
vllm:request_latency_seconds_sum{model="a"} 0.9
vllm:request_latency_seconds_count{model="a"} 5
# TYPE lifetime summary
lifetime{quantile="0.5"} 1
# TYPE process_uptime_seconds gauge
process_uptime_seconds 10
"#;
        let metrics = PrometheusTextParser.parse_classic(body).unwrap();
        assert_eq!(metrics.len(), 2);
        let counter = &metrics["vllm:prompt_tokens"];
        assert_eq!(counter.metric_type, PrometheusMetricType::Counter);
        assert_eq!(counter.description, "Prompt tokens.");
        assert!(matches!(
            &counter.samples[0],
            MetricSample::Scalar { value, .. } if *value == 12.0
        ));
        let histogram = &metrics["vllm:request_latency_seconds"];
        let MetricSample::Histogram { value, .. } = &histogram.samples[0] else {
            panic!("expected histogram")
        };
        assert_eq!(value.buckets.len(), 2);
        assert_eq!(value.buckets["+Inf"], 5.0);
        assert_eq!(value.sum, Some(0.9));
        assert_eq!(value.count, Some(5.0));
    }

    #[test]
    fn strict_openmetrics_can_fall_back_to_classic_at_the_caller() {
        let body = "# TYPE requests_total counter\nrequests_total 2\n";
        assert!(PrometheusTextParser.parse_openmetrics(body).is_err());
        assert!(PrometheusTextParser.parse_classic(body).is_ok());

        let strict = "# TYPE requests counter\nrequests_total 2\n# EOF\n";
        assert_eq!(
            PrometheusTextParser.parse_openmetrics(strict).unwrap()["requests"].metric_type,
            PrometheusMetricType::Counter
        );
    }

    #[test]
    fn json_and_malformed_labels_are_rejected() {
        assert!(
            PrometheusTextParser
                .parse_classic("[{\"stats\": 1}]")
                .is_err()
        );
        assert!(
            PrometheusTextParser
                .parse_classic("metric{bad=unquoted} 1")
                .is_err()
        );
    }

    #[test]
    fn undeclared_untyped_samples_remain_unknown_gauges() {
        let metrics = PrometheusTextParser
            .parse_classic("node_netstat_Tcp_InSegs 3\n")
            .unwrap();
        assert_eq!(
            metrics["node_netstat_Tcp_InSegs"].metric_type,
            PrometheusMetricType::Unknown
        );
    }

    #[test]
    fn normalized_counter_wins_a_same_name_gauge_collision() {
        let body = r#"
# TYPE sglang:num_retracted_reqs_total counter
sglang:num_retracted_reqs_total 7
# TYPE sglang:num_retracted_reqs gauge
sglang:num_retracted_reqs 99
"#;

        let metrics = PrometheusTextParser.parse_classic(body).unwrap();
        let family = &metrics["sglang:num_retracted_reqs"];

        assert_eq!(family.metric_type, PrometheusMetricType::Counter);
        assert!(matches!(
            family.samples[0],
            MetricSample::Scalar { value: 7.0, .. }
        ));
    }
}
