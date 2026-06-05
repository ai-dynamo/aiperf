use std::collections::HashMap;
use tachometer_writer::Row;

/// Represents a parsed Prometheus metric sample
#[derive(Debug, Clone)]
pub struct ParsedSample {
    pub metric_name: String, // Base metric name (without suffix)
    pub labels: HashMap<String, String>,
    pub value: f64,
    pub metric_type: MetricType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    HistogramBucket,
    HistogramCount,
    HistogramSum,
    Summary,
    SummaryQuantile,
    SummaryCount,
    SummarySum,
    Untyped,
}

/// Parse Prometheus text format and return ParsedSample objects
/// This parser preserves _count and _sum fields for histograms and summaries
pub fn parse_prometheus_samples(text: &str) -> Result<Vec<ParsedSample>, anyhow::Error> {
    let mut samples = Vec::new();
    let mut metric_types: HashMap<String, MetricType> = HashMap::new();

    // First pass: collect metric types from TYPE comments
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("# TYPE ") {
            if let Some(metric_type) = parse_type_line(line) {
                metric_types.insert(metric_type.0, metric_type.1);
            }
        }
    }

    // Second pass: parse metrics
    for line in text.lines() {
        let line = line.trim();

        // Skip empty lines and comments (except TYPE which we already handled)
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse metric line
        match parse_metric_line(line, &metric_types) {
            Ok(sample) => {
                samples.push(sample);
            }
            Err(_e) => {
                // Silently skip lines that fail to parse
                continue;
            }
        }
    }

    Ok(samples)
}

/// Compute percentile from a sorted slice
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let idx = (p / 100.0) * (sorted_values.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let frac = idx - lower as f64;
        sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac
    }
}

/// Aggregate CPU metrics by mode, computing min/max/p10/p90 across all CPUs
fn aggregate_cpu_metrics(samples: &[ParsedSample]) -> Vec<ParsedSample> {
    use std::collections::HashMap;

    // Group CPU seconds by mode
    let mut by_mode: HashMap<String, Vec<f64>> = HashMap::new();

    for sample in samples {
        if sample.metric_name == "node_cpu_seconds_total" {
            if let Some(mode) = sample.labels.get("mode") {
                by_mode.entry(mode.clone()).or_default().push(sample.value);
            }
        }
    }

    // Generate aggregated samples
    let mut aggregated = Vec::new();
    for (mode, mut values) in by_mode {
        if values.is_empty() {
            continue;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = values.first().copied().unwrap_or(0.0);
        let max_val = values.last().copied().unwrap_or(0.0);
        let p10_val = percentile(&values, 10.0);
        let p90_val = percentile(&values, 90.0);

        for (stat, value) in [
            ("min", min_val),
            ("max", max_val),
            ("p10", p10_val),
            ("p90", p90_val),
        ] {
            // Store mode as a label (will become metadata column via filter)
            // Store stat in the metric name
            let mut labels = HashMap::new();
            labels.insert("mode".to_string(), mode.clone());

            aggregated.push(ParsedSample {
                // Use stat as suffix in metric name, mode will be extracted as metadata
                metric_name: format!("node_cpu_time_{}", stat),
                labels,
                value,
                metric_type: MetricType::Counter,
            });
        }
    }

    aggregated
}

/// Convert ParsedSample objects to Row objects with optional filtering
pub fn samples_to_rows_with_filter(
    samples: Vec<ParsedSample>,
    scraper_endpoint: &str,
    filter: &dyn crate::filters::MetricFilter,
) -> Vec<Row> {
    use std::collections::HashMap;

    let mut rows = Vec::new();

    // Aggregate CPU metrics first
    let cpu_aggregated = aggregate_cpu_metrics(&samples);

    // Filter out original CPU metrics, replace with aggregated ones
    let mut processed_samples: Vec<ParsedSample> = samples
        .into_iter()
        .filter(|s| s.metric_name != "node_cpu_seconds_total")
        .collect();
    processed_samples.extend(cpu_aggregated);

    // First pass: collect histogram sum and count values
    let mut histogram_stats: HashMap<String, (Option<f32>, Option<f32>)> = HashMap::new();

    for sample in &processed_samples {
        let key = make_histogram_key(&sample.metric_name, &sample.labels);
        match sample.metric_type {
            MetricType::HistogramSum => {
                histogram_stats.entry(key).or_insert((None, None)).0 = Some(sample.value as f32);
            }
            MetricType::HistogramCount => {
                histogram_stats.entry(key).or_insert((None, None)).1 = Some(sample.value as f32);
            }
            _ => {}
        }
    }

    // Second pass: create rows, skip sum/count metrics
    for sample in processed_samples {
        // Apply filter to get final metric name and extras
        let (filtered_name, extras) = filter.filter(&sample);

        match sample.metric_type {
            MetricType::HistogramBucket => {
                // Histogram bucket: one row with bucket bounds
                let inf_str = "+Inf".to_string();
                let le_str = sample.labels.get("le").unwrap_or(&inf_str);
                let upper_bound = if le_str == "+Inf" {
                    None // +Inf is represented as None
                } else {
                    le_str.parse::<f32>().ok()
                };

                rows.push(Row {
                    scraper_endpoint: scraper_endpoint.to_string(),
                    metric_name: filtered_name,
                    metric_value: sample.value as f32,
                    histogram_bucket_lower: None, // Will be calculated later
                    histogram_bucket_upper: upper_bound,
                    histogram_sum: None,   // Will be set later for all buckets
                    histogram_count: None, // Will be set later for all buckets
                    extras,
                });
            }
            MetricType::HistogramSum | MetricType::HistogramCount => {
                // Skip - these are now part of histogram bucket rows
            }
            _ => {
                // All other metrics: regular row
                rows.push(Row {
                    scraper_endpoint: scraper_endpoint.to_string(),
                    metric_name: filtered_name,
                    metric_value: sample.value as f32,
                    histogram_bucket_lower: None,
                    histogram_bucket_upper: None,
                    histogram_sum: None,
                    histogram_count: None,
                    extras,
                });
            }
        }
    }

    // Post-process histogram buckets to set lower bounds and attach sum/count to all buckets
    fix_histogram_bucket_bounds_and_stats(&mut rows, histogram_stats);

    rows
}

/// Convert ParsedSample objects to Row objects (without filtering)
pub fn samples_to_rows(samples: Vec<ParsedSample>, scraper_endpoint: &str) -> Vec<Row> {
    use std::collections::HashMap;

    let mut rows = Vec::new();

    // First pass: collect histogram sum and count values
    let mut histogram_stats: HashMap<String, (Option<f32>, Option<f32>)> = HashMap::new();

    for sample in &samples {
        let key = make_histogram_key(&sample.metric_name, &sample.labels);
        match sample.metric_type {
            MetricType::HistogramSum => {
                histogram_stats.entry(key).or_insert((None, None)).0 = Some(sample.value as f32);
            }
            MetricType::HistogramCount => {
                histogram_stats.entry(key).or_insert((None, None)).1 = Some(sample.value as f32);
            }
            _ => {}
        }
    }

    // Second pass: create rows, skip sum/count metrics
    for sample in samples {
        let (metric_name, extras) = format_sample_as_metric_name(&sample);

        match sample.metric_type {
            MetricType::HistogramBucket => {
                let inf_str = "+Inf".to_string();
                let le_str = sample.labels.get("le").unwrap_or(&inf_str);
                let upper_bound = if le_str == "+Inf" {
                    None
                } else {
                    le_str.parse::<f32>().ok()
                };

                rows.push(Row {
                    scraper_endpoint: scraper_endpoint.to_string(),
                    metric_name,
                    metric_value: sample.value as f32,
                    histogram_bucket_lower: None,
                    histogram_bucket_upper: upper_bound,
                    histogram_sum: None,   // Will be set later for all buckets
                    histogram_count: None, // Will be set later for all buckets
                    extras,
                });
            }
            MetricType::HistogramSum | MetricType::HistogramCount => {
                // Skip - these are now part of histogram bucket rows
            }
            _ => {
                rows.push(Row {
                    scraper_endpoint: scraper_endpoint.to_string(),
                    metric_name,
                    metric_value: sample.value as f32,
                    histogram_bucket_lower: None,
                    histogram_bucket_upper: None,
                    histogram_sum: None,
                    histogram_count: None,
                    extras,
                });
            }
        }
    }

    // Post-process histogram buckets to set lower bounds and attach sum/count to all buckets
    fix_histogram_bucket_bounds_and_stats(&mut rows, histogram_stats);

    rows
}

/// Format a ParsedSample as a metric name string for display
fn format_sample_as_metric_name(sample: &ParsedSample) -> (String, Vec<(String, String)>) {
    // Add suffix based on metric type
    let base_name = match sample.metric_type {
        MetricType::HistogramCount => format!("{}_count", sample.metric_name),
        MetricType::HistogramSum => format!("{}_sum", sample.metric_name),
        MetricType::HistogramBucket => format!("{}_bucket", sample.metric_name),
        MetricType::SummaryCount => format!("{}_count", sample.metric_name),
        MetricType::SummarySum => format!("{}_sum", sample.metric_name),
        _ => sample.metric_name.clone(),
    };

    let metric_name = if sample.labels.is_empty() {
        base_name
    } else {
        format!("{}{{{}}}", base_name, format_labels(&sample.labels))
    };

    (metric_name, vec![])
}

/// Convenience function to parse and convert in one step
pub fn parse_prometheus_text(
    text: &str,
    scraper_endpoint: &str,
) -> Result<Vec<Row>, anyhow::Error> {
    let samples = parse_prometheus_samples(text)?;
    Ok(samples_to_rows(samples, scraper_endpoint))
}

fn parse_type_line(line: &str) -> Option<(String, MetricType)> {
    // Format: # TYPE metric_name type
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 4 {
        let metric_name = parts[2].to_string();
        let metric_type = match parts[3] {
            "counter" => MetricType::Counter,
            "gauge" => MetricType::Gauge,
            "histogram" => MetricType::Histogram,
            "summary" => MetricType::Summary,
            "untyped" => MetricType::Untyped,
            _ => return None,
        };
        Some((metric_name, metric_type))
    } else {
        None
    }
}

fn parse_metric_line(
    line: &str,
    metric_types: &HashMap<String, MetricType>,
) -> Result<ParsedSample, anyhow::Error> {
    // Format: metric_name{labels} value or metric_name value
    // Need to handle labels that may contain spaces, so find the split point correctly
    let metric_and_labels: &str;
    let value_str: &str;

    if line.contains('{') {
        // Metric has labels: find the closing brace and split after it
        let brace_end = line
            .rfind('}')
            .ok_or_else(|| anyhow::anyhow!("Unclosed brace in labels"))?;
        metric_and_labels = &line[..=brace_end];
        // Value starts after whitespace following the brace
        value_str = line[brace_end + 1..]
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing value after labels"))?;
    } else {
        // No labels: split on first whitespace
        let mut parts = line.split_whitespace();
        metric_and_labels = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty metric line"))?;
        value_str = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing value"))?;
    }

    let value = value_str
        .parse::<f64>()
        .map_err(|e| anyhow::anyhow!("Failed to parse value '{}': {}", value_str, e))?;
    if !value.is_finite() {
        return Err(anyhow::anyhow!("Non-finite metric value '{}'", value_str));
    }

    let (metric_name, labels, metric_type) = if metric_and_labels.contains('{') {
        parse_metric_with_labels(metric_and_labels, metric_types)?
    } else {
        let metric_type = determine_metric_type(metric_and_labels, metric_types)?;
        (strip_suffix(metric_and_labels), HashMap::new(), metric_type)
    };

    Ok(ParsedSample {
        metric_name,
        labels,
        value,
        metric_type,
    })
}

fn parse_metric_with_labels(
    metric_and_labels: &str,
    metric_types: &HashMap<String, MetricType>,
) -> Result<(String, HashMap<String, String>, MetricType), anyhow::Error> {
    // Format: metric_name{label1="value1",label2="value2"}
    let brace_start = metric_and_labels
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("Malformed label section"))?;
    let brace_end = metric_and_labels
        .rfind('}')
        .ok_or_else(|| anyhow::anyhow!("Malformed label section"))?;

    let metric_name_with_suffix = &metric_and_labels[..brace_start];
    let labels_str = &metric_and_labels[brace_start + 1..brace_end];

    let mut labels = HashMap::new();
    if !labels_str.is_empty() {
        for label_pair in labels_str.split(',') {
            let parts: Vec<&str> = label_pair.splitn(2, '=').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let value = parts[1].trim().trim_matches('"');
                labels.insert(key.to_string(), value.to_string());
            }
        }
    }

    // Determine metric type - check if it's a summary quantile
    let metric_type = if labels.contains_key("quantile") {
        let base = strip_suffix(metric_name_with_suffix);
        if metric_types.get(&base) == Some(&MetricType::Summary) {
            MetricType::SummaryQuantile
        } else {
            determine_metric_type(metric_name_with_suffix, metric_types)?
        }
    } else {
        determine_metric_type(metric_name_with_suffix, metric_types)?
    };

    let base_metric_name = strip_suffix(metric_name_with_suffix);

    Ok((base_metric_name, labels, metric_type))
}

fn determine_metric_type(
    metric_name: &str,
    metric_types: &HashMap<String, MetricType>,
) -> Result<MetricType, anyhow::Error> {
    // Check for suffixes first
    if metric_name.ends_with("_bucket") {
        // Check if base metric is a histogram
        let base = metric_name.strip_suffix("_bucket").unwrap();
        if metric_types.get(base) == Some(&MetricType::Histogram) {
            return Ok(MetricType::HistogramBucket);
        }
    } else if metric_name.ends_with("_count") {
        let base = metric_name.strip_suffix("_count").unwrap();
        if metric_types.get(base) == Some(&MetricType::Histogram) {
            return Ok(MetricType::HistogramCount);
        } else if metric_types.get(base) == Some(&MetricType::Summary) {
            return Ok(MetricType::SummaryCount);
        }
    } else if metric_name.ends_with("_sum") {
        let base = metric_name.strip_suffix("_sum").unwrap();
        if metric_types.get(base) == Some(&MetricType::Histogram) {
            return Ok(MetricType::HistogramSum);
        } else if metric_types.get(base) == Some(&MetricType::Summary) {
            return Ok(MetricType::SummarySum);
        }
    }

    // Check base metric type
    if let Some(metric_type) = metric_types.get(metric_name) {
        match metric_type {
            MetricType::Counter => Ok(MetricType::Counter),
            MetricType::Gauge => Ok(MetricType::Gauge),
            MetricType::Histogram => Ok(MetricType::Histogram),
            MetricType::Summary => Ok(MetricType::Summary),
            MetricType::Untyped => Ok(MetricType::Untyped),
            _ => Ok(MetricType::Untyped), // Fallback
        }
    } else {
        Ok(MetricType::Untyped)
    }
}

fn strip_suffix(metric_name: &str) -> String {
    // Remove _bucket, _count, _sum suffixes to get base name
    if metric_name.ends_with("_bucket") {
        metric_name.strip_suffix("_bucket").unwrap().to_string()
    } else if metric_name.ends_with("_count") {
        metric_name.strip_suffix("_count").unwrap().to_string()
    } else if metric_name.ends_with("_sum") {
        metric_name.strip_suffix("_sum").unwrap().to_string()
    } else {
        metric_name.to_string()
    }
}

fn format_labels(labels: &HashMap<String, String>) -> String {
    let mut pairs: Vec<(String, String)> =
        labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    pairs.sort_by_key(|(k, _)| k.clone());
    pairs
        .iter()
        .map(|(k, v)| format!("{}=\"{}\"", k, v))
        .collect::<Vec<_>>()
        .join(",")
}

/// Create a unique key for a histogram based on metric name and labels (excluding le)
fn make_histogram_key(metric_name: &str, labels: &HashMap<String, String>) -> String {
    let mut label_pairs: Vec<(String, String)> = labels
        .iter()
        .filter(|(k, _)| k.as_str() != "le" && k.as_str() != "quantile")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    label_pairs.sort();

    if label_pairs.is_empty() {
        metric_name.to_string()
    } else {
        format!("{}:{:?}", metric_name, label_pairs)
    }
}

fn fix_histogram_bucket_bounds_and_stats(
    rows: &mut [Row],
    histogram_stats: HashMap<String, (Option<f32>, Option<f32>)>,
) {
    // Group histogram buckets by metric name and set lower bounds
    use std::collections::BTreeMap;

    let mut bucket_groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    // Collect indices of histogram bucket rows
    for (i, row) in rows.iter().enumerate() {
        if row.histogram_bucket_upper.is_some()
            || (row.histogram_bucket_upper.is_none() && row.metric_name.contains("_bucket"))
        {
            // Extract base metric name without the le parameter for grouping
            // e.g., "test_hist_bucket{le=\"0.1\"}" -> "test_hist_bucket"
            let base_name = if let Some(brace_pos) = row.metric_name.find('{') {
                row.metric_name[..brace_pos].to_string()
            } else {
                row.metric_name.clone()
            };

            bucket_groups.entry(base_name).or_default().push(i);
        }
    }

    // Set lower bounds for each group and attach sum/count to all buckets
    for (base_name, indices) in bucket_groups.iter_mut() {
        indices.sort_by(|&i, &j| {
            let upper_i = rows[i].histogram_bucket_upper.unwrap_or(f32::INFINITY);
            let upper_j = rows[j].histogram_bucket_upper.unwrap_or(f32::INFINITY);
            // Sort: +Inf comes last, then by value
            match (upper_i == f32::INFINITY, upper_j == f32::INFINITY) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => upper_i
                    .partial_cmp(&upper_j)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        // Extract labels from metric name to create key (use first bucket's name)
        let key = if !indices.is_empty() {
            extract_histogram_key_from_metric_name(&rows[indices[0]].metric_name, base_name)
        } else {
            base_name.to_string()
        };

        let stats = histogram_stats.get(&key);

        for (j, &idx) in indices.iter().enumerate() {
            if j == 0 {
                rows[idx].histogram_bucket_lower = Some(0.0);
            } else {
                let prev_idx = indices[j - 1];
                rows[idx].histogram_bucket_lower = rows[prev_idx].histogram_bucket_upper;
            }

            // Attach sum and count to all buckets
            if let Some((sum, count)) = stats {
                rows[idx].histogram_sum = *sum;
                rows[idx].histogram_count = *count;
            }
        }
    }
}

/// Extract histogram key from a metric name like "test_hist_bucket{method=\"GET\"}"
fn extract_histogram_key_from_metric_name(metric_name: &str, base_name: &str) -> String {
    // Remove "_bucket" suffix from base_name to get the original metric name
    let original_metric_name = base_name.strip_suffix("_bucket").unwrap_or(base_name);

    // If no labels, just return the metric name
    if !metric_name.contains('{') {
        return original_metric_name.to_string();
    }

    // Parse labels from the metric name
    if let Some(brace_start) = metric_name.find('{') {
        if let Some(brace_end) = metric_name.rfind('}') {
            let labels_str = &metric_name[brace_start + 1..brace_end];
            let mut label_pairs: Vec<(String, String)> = Vec::new();

            for label_pair in labels_str.split(',') {
                let parts: Vec<&str> = label_pair.splitn(2, '=').collect();
                if parts.len() == 2 {
                    let key = parts[0].trim();
                    let value = parts[1].trim().trim_matches('"');
                    // Skip 'le' label
                    if key != "le" && key != "quantile" {
                        label_pairs.push((key.to_string(), value.to_string()));
                    }
                }
            }

            label_pairs.sort();

            if label_pairs.is_empty() {
                return original_metric_name.to_string();
            } else {
                return format!("{}:{:?}", original_metric_name, label_pairs);
            }
        }
    }

    original_metric_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_gauge() {
        let text = r#"# TYPE test_gauge gauge
test_gauge 42.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_name, "test_gauge");
        assert_eq!(samples[0].value, 42.5);
        assert_eq!(samples[0].metric_type, MetricType::Gauge);
        assert!(samples[0].labels.is_empty());
    }

    #[test]
    fn test_parse_gauge_with_labels() {
        let text = r#"# TYPE test_metric gauge
test_metric{label1="value1",label2="value2"} 123.45"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_name, "test_metric");
        assert_eq!(samples[0].value, 123.45);
        assert_eq!(samples[0].labels.get("label1"), Some(&"value1".to_string()));
        assert_eq!(samples[0].labels.get("label2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_parse_skips_non_finite_metric_values() {
        let text = r#"# TYPE sglang:fwd_occupancy gauge
sglang:fwd_occupancy{engine_type="unified",model_name="Qwen/Qwen3-0.6B",tp_rank="0",pp_rank="0",moe_ep_rank="0"} NaN
sglang:fwd_occupancy{engine_type="unified",model_name="Qwen/Qwen3-0.6B",tp_rank="0",pp_rank="0",moe_ep_rank="0"} +Inf
sglang:fwd_occupancy{engine_type="unified",model_name="Qwen/Qwen3-0.6B",tp_rank="0",pp_rank="0",moe_ep_rank="0"} -Inf
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{engine_type="unified",model_name="Qwen/Qwen3-0.6B",tp_rank="0",pp_rank="0",moe_ep_rank="0"} 1.0"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_name, "sglang:num_running_reqs");
        assert_eq!(samples[0].value, 1.0);

        let rows = samples_to_rows(samples, "sglang");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].metric_name, "sglang:num_running_reqs{engine_type=\"unified\",model_name=\"Qwen/Qwen3-0.6B\",moe_ep_rank=\"0\",pp_rank=\"0\",tp_rank=\"0\"}");
        assert_eq!(rows[0].metric_value, 1.0);
    }

    #[test]
    fn test_parse_labels_with_spaces() {
        let text = r#"# TYPE test_metric gauge
test_metric{name="NVIDIA GB200",location="rack 1"} 100"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(
            samples[0].labels.get("name"),
            Some(&"NVIDIA GB200".to_string())
        );
        assert_eq!(
            samples[0].labels.get("location"),
            Some(&"rack 1".to_string())
        );
    }

    #[test]
    fn test_parse_histogram_with_count_and_sum() {
        let text = r#"# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.005"} 24054
http_request_duration_seconds_bucket{le="0.01"} 33444
http_request_duration_seconds_bucket{le="+Inf"} 179887
http_request_duration_seconds_count 179887
http_request_duration_seconds_sum 53423"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 5, "Should have 3 buckets + count + sum");

        // Check buckets
        let buckets: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::HistogramBucket)
            .collect();
        assert_eq!(buckets.len(), 3);

        // Check _count
        let count = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramCount);
        assert!(count.is_some(), "Should have _count metric");
        assert_eq!(count.unwrap().value, 179887.0);
        assert_eq!(count.unwrap().metric_name, "http_request_duration_seconds");

        // Check _sum
        let sum = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramSum);
        assert!(sum.is_some(), "Should have _sum metric");
        assert_eq!(sum.unwrap().value, 53423.0);
        assert_eq!(sum.unwrap().metric_name, "http_request_duration_seconds");
    }

    #[test]
    fn test_parse_histogram_with_labels() {
        let text = r#"# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",le="0.1"} 100
http_request_duration_seconds_bucket{method="GET",le="+Inf"} 200
http_request_duration_seconds_count{method="GET"} 200
http_request_duration_seconds_sum{method="GET"} 10.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 4);

        // Verify count has labels
        let count = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramCount);
        assert!(count.is_some());
        assert_eq!(
            count.unwrap().labels.get("method"),
            Some(&"GET".to_string())
        );

        // Verify sum has labels
        let sum = samples
            .iter()
            .find(|s| s.metric_type == MetricType::HistogramSum);
        assert!(sum.is_some());
        assert_eq!(sum.unwrap().labels.get("method"), Some(&"GET".to_string()));
    }

    #[test]
    fn test_parse_summary_with_count_and_sum() {
        let text = r#"# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0"} 0
go_gc_duration_seconds{quantile="0.25"} 0.5
go_gc_duration_seconds{quantile="0.5"} 1.0
go_gc_duration_seconds{quantile="0.75"} 1.5
go_gc_duration_seconds{quantile="1"} 2.0
go_gc_duration_seconds_sum 12.5
go_gc_duration_seconds_count 100"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 7, "Should have 5 quantiles + count + sum");

        // Check quantiles
        let quantiles: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::SummaryQuantile)
            .collect();
        assert_eq!(quantiles.len(), 5);

        // Check _count
        let count = samples
            .iter()
            .find(|s| s.metric_type == MetricType::SummaryCount);
        assert!(count.is_some(), "Should have summary _count");
        assert_eq!(count.unwrap().value, 100.0);

        // Check _sum
        let sum = samples
            .iter()
            .find(|s| s.metric_type == MetricType::SummarySum);
        assert!(sum.is_some(), "Should have summary _sum");
        assert_eq!(sum.unwrap().value, 12.5);
    }

    #[test]
    fn test_samples_to_rows_preserves_count_sum() {
        let text = r#"# TYPE test_histogram histogram
test_histogram_bucket{le="1.0"} 50
test_histogram_bucket{le="+Inf"} 100
test_histogram_count 100
test_histogram_sum 45.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        let rows = samples_to_rows(samples, "test_endpoint");

        // Should have only 2 rows (the buckets), not 4 (buckets + count + sum)
        assert_eq!(rows.len(), 2, "Should have 2 histogram bucket rows");

        // All buckets should have sum and count attached
        for row in &rows {
            assert_eq!(
                row.histogram_sum,
                Some(45.5),
                "All buckets should have histogram_sum"
            );
            assert_eq!(
                row.histogram_count,
                Some(100.0),
                "All buckets should have histogram_count"
            );
        }
    }

    #[test]
    fn test_counter_metric() {
        let text = r#"# TYPE node_cpu_seconds_total counter
node_cpu_seconds_total{cpu="0",mode="idle"} 12345.67"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_type, MetricType::Counter);
        assert_eq!(samples[0].metric_name, "node_cpu_seconds_total");
        assert_eq!(samples[0].value, 12345.67);
    }

    #[test]
    fn test_histogram_bucket_bounds() {
        let text = r#"# TYPE test_hist histogram
test_hist_bucket{le="0.1"} 10
test_hist_bucket{le="1.0"} 50
test_hist_bucket{le="10.0"} 90
test_hist_bucket{le="+Inf"} 100
test_hist_count 100
test_hist_sum 123.4"#;

        let samples = parse_prometheus_samples(text).unwrap();
        let rows = samples_to_rows(samples, "test");

        // Check bucket bounds are set correctly (4 buckets only, no separate count/sum rows)
        let buckets: Vec<_> = rows
            .iter()
            .filter(|r| {
                r.histogram_bucket_upper.is_some()
                    || r.histogram_bucket_upper.is_none() && r.metric_name.contains("_bucket")
            })
            .collect();
        assert_eq!(
            buckets.len(),
            4,
            "Should have 4 histogram buckets, got {}",
            buckets.len()
        );

        // First bucket: 0 to 0.1
        assert_eq!(buckets[0].histogram_bucket_lower, Some(0.0));
        assert_eq!(buckets[0].histogram_bucket_upper, Some(0.1));
        // All buckets should have sum and count
        assert_eq!(buckets[0].histogram_sum, Some(123.4));
        assert_eq!(buckets[0].histogram_count, Some(100.0));

        // Second bucket: 0.1 to 1.0
        assert_eq!(buckets[1].histogram_bucket_lower, Some(0.1));
        assert_eq!(buckets[1].histogram_bucket_upper, Some(1.0));
        // All buckets should have sum and count
        assert_eq!(buckets[1].histogram_sum, Some(123.4));
        assert_eq!(buckets[1].histogram_count, Some(100.0));

        // Last bucket: 10.0 to +Inf
        assert_eq!(buckets[3].histogram_bucket_lower, Some(10.0));
        assert_eq!(buckets[3].histogram_bucket_upper, None); // +Inf
                                                             // All buckets should have sum and count
        assert_eq!(buckets[3].histogram_sum, Some(123.4));
        assert_eq!(buckets[3].histogram_count, Some(100.0));
    }

    #[test]
    fn test_mixed_metric_types() {
        let text = r#"# TYPE counter_metric counter
counter_metric 100
# TYPE gauge_metric gauge
gauge_metric{label="value"} 50.5
# TYPE hist_metric histogram
hist_metric_bucket{le="+Inf"} 10
hist_metric_count 10
hist_metric_sum 5.5"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 5);

        let counters: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::Counter)
            .collect();
        assert_eq!(counters.len(), 1);

        let gauges: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::Gauge)
            .collect();
        assert_eq!(gauges.len(), 1);

        let hist_count: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::HistogramCount)
            .collect();
        assert_eq!(hist_count.len(), 1);

        let hist_sum: Vec<_> = samples
            .iter()
            .filter(|s| s.metric_type == MetricType::HistogramSum)
            .collect();
        assert_eq!(hist_sum.len(), 1);
    }

    #[test]
    fn test_empty_input() {
        let text = "";
        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 0);
    }

    #[test]
    fn test_comments_only() {
        let text = r#"# HELP test_metric This is a test metric
# TYPE test_metric gauge"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 0, "Should have no samples, only metadata");
    }

    #[test]
    fn test_metric_without_type_declaration() {
        let text = r#"untyped_metric 123"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].metric_type, MetricType::Untyped);
        assert_eq!(samples[0].value, 123.0);
    }

    #[test]
    fn test_percentile_calculation() {
        // Test percentile function
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 100.0), 10.0);
        assert!((percentile(&values, 50.0) - 5.5).abs() < 0.01);
        assert!((percentile(&values, 10.0) - 1.9).abs() < 0.01);
        assert!((percentile(&values, 90.0) - 9.1).abs() < 0.01);

        // Edge cases
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_cpu_aggregation() {
        // Simulate CPU metrics from multiple cores
        let text = r#"# TYPE node_cpu_seconds_total counter
node_cpu_seconds_total{cpu="0",mode="idle"} 100.0
node_cpu_seconds_total{cpu="1",mode="idle"} 200.0
node_cpu_seconds_total{cpu="2",mode="idle"} 150.0
node_cpu_seconds_total{cpu="3",mode="idle"} 250.0
node_cpu_seconds_total{cpu="0",mode="user"} 10.0
node_cpu_seconds_total{cpu="1",mode="user"} 20.0
node_cpu_seconds_total{cpu="2",mode="user"} 15.0
node_cpu_seconds_total{cpu="3",mode="user"} 25.0"#;

        let samples = parse_prometheus_samples(text).unwrap();
        assert_eq!(samples.len(), 8, "Should have 8 original CPU samples");

        // Test aggregation
        let aggregated = aggregate_cpu_metrics(&samples);

        // Should have 4 stats (min, max, p10, p90) × 2 modes (idle, user) = 8 samples
        assert_eq!(
            aggregated.len(),
            8,
            "Should have 8 aggregated samples (4 stats × 2 modes)"
        );

        // Check idle mode aggregation - stat is now in metric name, not labels
        let idle_samples: Vec<_> = aggregated
            .iter()
            .filter(|s| s.labels.get("mode") == Some(&"idle".to_string()))
            .collect();
        assert_eq!(idle_samples.len(), 4, "Should have 4 stats for idle mode");

        // Find specific stats for idle mode - stat is in metric_name now
        let idle_min = idle_samples
            .iter()
            .find(|s| s.metric_name == "node_cpu_time_min")
            .expect("Should have min stat");
        let idle_max = idle_samples
            .iter()
            .find(|s| s.metric_name == "node_cpu_time_max")
            .expect("Should have max stat");

        assert_eq!(idle_min.value, 100.0, "Min idle should be 100.0");
        assert_eq!(idle_max.value, 250.0, "Max idle should be 250.0");

        // Check user mode aggregation
        let user_min = aggregated
            .iter()
            .find(|s| {
                s.labels.get("mode") == Some(&"user".to_string())
                    && s.metric_name == "node_cpu_time_min"
            })
            .expect("Should have user min stat");
        let user_max = aggregated
            .iter()
            .find(|s| {
                s.labels.get("mode") == Some(&"user".to_string())
                    && s.metric_name == "node_cpu_time_max"
            })
            .expect("Should have user max stat");

        assert_eq!(user_min.value, 10.0, "Min user should be 10.0");
        assert_eq!(user_max.value, 25.0, "Max user should be 25.0");
    }

    #[test]
    fn test_cpu_aggregation_removes_original_metrics() {
        use crate::filters::get_filter;

        let text = r#"# TYPE node_cpu_seconds_total counter
node_cpu_seconds_total{cpu="0",mode="idle"} 100.0
node_cpu_seconds_total{cpu="1",mode="idle"} 200.0
# TYPE node_memory_MemTotal_bytes gauge
node_memory_MemTotal_bytes 1000000"#;

        let samples = parse_prometheus_samples(text).unwrap();
        let filter = get_filter("node_exporter", None, None);
        let rows = samples_to_rows_with_filter(samples, "test", filter.as_ref());

        // Should NOT have original per-CPU metrics
        let per_cpu_rows: Vec<_> = rows
            .iter()
            .filter(|r| r.metric_name.contains("cpu="))
            .collect();
        assert!(
            per_cpu_rows.is_empty(),
            "Should not have per-CPU metrics, found: {:?}",
            per_cpu_rows
        );

        // Should have aggregated CPU metrics with stats
        let cpu_rows: Vec<_> = rows
            .iter()
            .filter(|r| r.metric_name.starts_with("cpu_time_"))
            .collect();
        assert_eq!(
            cpu_rows.len(),
            4,
            "Should have 4 aggregated CPU metrics (min, max, p10, p90 for idle mode)"
        );

        // Verify metric names are clean (no labels in name) and mode is in extras
        for row in &cpu_rows {
            // Metric name should NOT contain mode label - mode is now a metadata column
            assert!(
                !row.metric_name.contains("mode="),
                "CPU metric should NOT contain mode label in name: {}",
                row.metric_name
            );
            assert!(
                row.metric_name == "cpu_time_min"
                    || row.metric_name == "cpu_time_max"
                    || row.metric_name == "cpu_time_p10"
                    || row.metric_name == "cpu_time_p90",
                "CPU metric should be clean stat name: {}",
                row.metric_name
            );
            // Mode should be in extras as metadata column
            let has_mode = row.extras.iter().any(|(k, _)| k == "mode");
            assert!(
                has_mode,
                "CPU metric should have mode in extras: {:?}",
                row.extras
            );
        }

        // Should still have memory metric
        let mem_rows: Vec<_> = rows
            .iter()
            .filter(|r| r.metric_name.contains("memory"))
            .collect();
        assert_eq!(mem_rows.len(), 1, "Should have 1 memory metric");
    }
}
