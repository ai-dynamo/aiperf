use crate::parse::ParsedSample;
use std::collections::HashMap;

/// Trait for filtering/transforming Prometheus metric names
pub trait MetricFilter: Send + Sync {
    /// Filter a metric name and its labels
    /// Returns a tuple of (filtered metric name, metadata key-value pairs)
    /// Metadata pairs should be sorted by key for consistent ordering
    fn filter(&self, sample: &ParsedSample) -> (String, Vec<(String, String)>);

    /// Return the extra column names that this filter will add to the schema
    /// These are columns that the filter dynamically adds (e.g., from Prometheus labels)
    /// that aren't part of the config metadata
    fn extra_column_names(&self) -> Vec<String> {
        vec![] // Default: no extra columns
    }
}

/// No-op filter that passes through metric names unchanged
pub struct NoOpFilter;

impl MetricFilter for NoOpFilter {
    fn filter(&self, sample: &ParsedSample) -> (String, Vec<(String, String)>) {
        let name = if sample.labels.is_empty() {
            sample.metric_name.clone()
        } else {
            format!(
                "{}{{{}}}",
                sample.metric_name,
                format_labels(&sample.labels)
            )
        };
        (name, vec![])
    }
}

fn format_labels(labels: &HashMap<String, String>) -> String {
    let mut pairs: Vec<(&String, &String)> = labels.iter().collect();
    pairs.sort_by_key(|(k, _)| *k);
    pairs
        .iter()
        .map(|(k, v)| format!("{}=\"{}\"", k, v))
        .collect::<Vec<_>>()
        .join(",")
}

/// Filter for node_exporter metrics
/// Simplifies verbose metric names by:
/// - Removing redundant "node_" prefix
/// - Simplifying common label combinations
/// - Collapsing similar metrics
/// - Attaching node-level metadata when available
pub struct NodeExporterFilter {
    node_metadata: HashMap<String, String>,
}

impl NodeExporterFilter {
    pub fn new(node_metadata: HashMap<String, String>) -> Self {
        Self { node_metadata }
    }
}

impl MetricFilter for NodeExporterFilter {
    fn extra_column_names(&self) -> Vec<String> {
        // NodeExporterFilter adds a "mode" column for CPU metrics
        vec!["mode".to_string()]
    }

    fn filter(&self, sample: &ParsedSample) -> (String, Vec<(String, String)>) {
        let metric = &sample.metric_name;

        // Remove "node_" prefix if present
        let base_metric = metric.strip_prefix("node_").unwrap_or(metric.as_str());

        // Handle common node_exporter patterns
        let filtered_metric = match base_metric {
            // Aggregated CPU metrics (cpu_time_min, cpu_time_max, cpu_time_p10, cpu_time_p90)
            // Mode is extracted as metadata column for split_by support
            m if m.starts_with("cpu_time_") => base_metric.to_string(),
            // Legacy: original per-CPU metrics (shouldn't appear after aggregation)
            m if m.starts_with("cpu_seconds_total") => {
                let mode = sample
                    .labels
                    .get("mode")
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                format!("cpu_time_total{{mode={}}}", mode)
            }
            // Filesystem metrics - simplify device/mountpoint
            m if m.starts_with("filesystem_") => {
                let mut key_labels = Vec::new();
                // Keep fstype and mountpoint as key identifiers
                if let Some(fstype) = sample.labels.get("fstype") {
                    key_labels.push(format!("fstype={}", fstype));
                }
                if let Some(mountpoint) = sample.labels.get("mountpoint") {
                    // Simplify mountpoint paths
                    let simplified = if mountpoint == "/" {
                        "root".to_string()
                    } else {
                        mountpoint.trim_start_matches('/').replace(['/', '-'], "_")
                    };
                    key_labels.push(format!("mount={}", simplified));
                }
                if key_labels.is_empty() {
                    base_metric.to_string()
                } else {
                    format!("{}{{{}}}", base_metric, key_labels.join(","))
                }
            }
            // Network metrics - simplify interface names
            m if m.starts_with("network_") => {
                if let Some(device) = sample.labels.get("device") {
                    // Simplify interface names (e.g., "eth0" -> "eth0", "enp0s3" -> "eth")
                    let simplified_device = if device.starts_with("en") || device.starts_with("wl")
                    {
                        // Extract meaningful part or use generic name
                        if device.len() > 4 {
                            "net".to_string()
                        } else {
                            device.to_string()
                        }
                    } else {
                        device.to_string()
                    };
                    format!("{}{{device={}}}", base_metric, simplified_device)
                } else {
                    base_metric.to_string()
                }
            }
            // Memory metrics - usually don't need labels
            m if m.starts_with("memory_") => base_metric.to_string(),
            // Disk metrics - simplify device names
            m if m.starts_with("disk_") => {
                if let Some(device) = sample.labels.get("device") {
                    // Extract disk identifier (e.g., "sda" from "/dev/sda")
                    let simplified_device = device
                        .trim_start_matches("/dev/")
                        .trim_start_matches("nvme")
                        .trim_start_matches("n")
                        .to_string();
                    format!("{}{{device={}}}", base_metric, simplified_device)
                } else {
                    base_metric.to_string()
                }
            }
            // Default: keep metric name, simplify labels
            _ => {
                if sample.labels.is_empty() {
                    base_metric.to_string()
                } else {
                    // Keep only the most important labels (limit to 2-3)
                    let mut important_labels = Vec::new();
                    let priority_labels =
                        ["job", "instance", "device", "mountpoint", "fstype", "mode"];

                    for key in priority_labels.iter() {
                        if let Some(value) = sample.labels.get(*key) {
                            important_labels.push(format!("{}={}", key, value));
                            if important_labels.len() >= 2 {
                                break;
                            }
                        }
                    }

                    if important_labels.is_empty() {
                        base_metric.to_string()
                    } else {
                        format!("{}{{{}}}", base_metric, important_labels.join(","))
                    }
                }
            }
        };

        // Build metadata
        let mut metadata = Vec::new();

        // Extract mode from CPU metrics as metadata column
        if base_metric.starts_with("cpu_time_") {
            if let Some(mode) = sample.labels.get("mode") {
                metadata.push(("mode".to_string(), mode.clone()));
            }
        }

        // Add node metadata with NEW FIELD NAMES
        if !self.node_metadata.is_empty() {
            for (old_key, value) in self.node_metadata.iter() {
                let new_key = match old_key.as_str() {
                    "node" => "hostname",
                    other => other,
                };
                metadata.push((new_key.to_string(), value.clone()));
            }
        }

        metadata.sort_by(|a, b| a.0.cmp(&b.0));
        (filtered_metric, metadata)
    }
}

/// Filter for NVIDIA DCGM metrics
/// Simplifies verbose DCGM metric names by:
/// - Removing redundant prefixes
/// - Simplifying GPU/device identifiers
/// - Collapsing similar metrics
/// - Attaching GPU metadata when available
pub struct DcgmFilter {
    gpu_metadata: HashMap<String, HashMap<String, String>>,
}

impl DcgmFilter {
    pub fn new(gpu_metadata: HashMap<String, HashMap<String, String>>) -> Self {
        Self { gpu_metadata }
    }

    /// Extract GPU index from sample labels
    fn extract_gpu_id(&self, sample: &ParsedSample) -> Option<String> {
        // Try various label names that might contain GPU index
        sample
            .labels
            .get("gpu")
            .or_else(|| sample.labels.get("GPU"))
            .or_else(|| sample.labels.get("device"))
            .map(|s| s.to_string())
    }
}

impl MetricFilter for DcgmFilter {
    fn extra_column_names(&self) -> Vec<String> {
        // DcgmFilter adds a "gpu" column from Prometheus labels
        vec!["gpu".to_string()]
    }

    fn filter(&self, sample: &ParsedSample) -> (String, Vec<(String, String)>) {
        let metric = &sample.metric_name;

        // Remove common DCGM prefixes
        let base_metric = metric
            .trim_start_matches("DCGM_")
            .trim_start_matches("dcgm_")
            .trim_start_matches("DCGM_FI_")
            .trim_start_matches("dcgm_fi_");

        // Common DCGM metric patterns - simplified without GPU labels in name
        let filtered_metric = match base_metric {
            // GPU utilization metrics
            m if m.contains("GPU_UTIL") || m.contains("gpu_util") => "gpu_util".to_string(),
            // Memory metrics
            m if m.contains("MEMORY") || m.contains("memory") => {
                let mem_type = if m.contains("USED") || m.contains("used") {
                    "used"
                } else if m.contains("FREE") || m.contains("free") {
                    "free"
                } else {
                    "total"
                };
                format!("gpu_mem_{}", mem_type)
            }
            // Temperature metrics
            m if m.contains("TEMP") || m.contains("temp") || m.contains("TEMPERATURE") => {
                "gpu_temp".to_string()
            }
            // Power metrics
            m if m.contains("POWER") || m.contains("power") => "gpu_power".to_string(),
            // Default: use base metric
            _ => base_metric.to_string(),
        };

        // Extract GPU ID and build metadata with NEW FIELD NAMES
        let mut metadata = Vec::new();
        if let Some(gpu_id) = self.extract_gpu_id(sample) {
            // Add GPU index as a separate metadata column
            metadata.push(("gpu".to_string(), gpu_id.clone()));

            if let Some(gpu_meta) = self.gpu_metadata.get(&gpu_id) {
                // Map old field names to new ones
                for (old_key, value) in gpu_meta.iter() {
                    let new_key = match old_key.as_str() {
                        "endpoint" => "worker_role",
                        "endpoint_index" => "worker_index",
                        "rank_index" => "worker_process",
                        "node" => "hostname",
                        other => other,
                    };
                    metadata.push((new_key.to_string(), value.clone()));
                }
                metadata.sort_by(|a, b| a.0.cmp(&b.0));
            }
        }

        (filtered_metric, metadata)
    }
}

/// Filter for backend inference worker metrics
/// Attaches worker metadata (role, instance, process) to all metrics
pub struct BackendFilter {
    node_metadata: HashMap<String, String>,
}

impl BackendFilter {
    pub fn new(node_metadata: HashMap<String, String>) -> Self {
        Self { node_metadata }
    }
}

impl MetricFilter for BackendFilter {
    fn filter(&self, sample: &ParsedSample) -> (String, Vec<(String, String)>) {
        // Keep metric name as-is (with labels if present)
        let metric_name = if sample.labels.is_empty() {
            sample.metric_name.clone()
        } else {
            format!(
                "{}{{{}}}",
                sample.metric_name,
                format_labels(&sample.labels)
            )
        };

        // Return node metadata with NEW FIELD NAMES
        let mut metadata = Vec::new();
        if !self.node_metadata.is_empty() {
            for (old_key, value) in self.node_metadata.iter() {
                let new_key = match old_key.as_str() {
                    "node" => "hostname",
                    "endpoint" => "worker_role",
                    "endpoint_index" => "worker_index",
                    "process_index" => "worker_process",
                    other => other,
                };
                metadata.push((new_key.to_string(), value.clone()));
            }
            metadata.sort_by(|a, b| a.0.cmp(&b.0));
        }

        (metric_name, metadata)
    }
}

/// Filter for frontend metrics
/// Attaches frontend metadata to all metrics
pub struct FrontendFilter {
    node_metadata: HashMap<String, String>,
}

impl FrontendFilter {
    pub fn new(node_metadata: HashMap<String, String>) -> Self {
        Self { node_metadata }
    }
}

impl MetricFilter for FrontendFilter {
    fn filter(&self, sample: &ParsedSample) -> (String, Vec<(String, String)>) {
        // Keep metric name as-is (with labels if present)
        let metric_name = if sample.labels.is_empty() {
            sample.metric_name.clone()
        } else {
            format!(
                "{}{{{}}}",
                sample.metric_name,
                format_labels(&sample.labels)
            )
        };

        // Return node metadata with NEW FIELD NAMES
        let mut metadata = Vec::new();
        if !self.node_metadata.is_empty() {
            for (old_key, value) in self.node_metadata.iter() {
                let new_key = match old_key.as_str() {
                    "node" => "hostname",
                    "index" => "frontend_index",
                    other => other,
                };
                metadata.push((new_key.to_string(), value.clone()));
            }
            metadata.sort_by(|a, b| a.0.cmp(&b.0));
        }

        (metric_name, metadata)
    }
}

/// Get a filter by name with optional metadata
pub fn get_filter(
    name: &str,
    gpu_metadata: Option<HashMap<String, HashMap<String, String>>>,
    node_metadata: Option<HashMap<String, String>>,
) -> Box<dyn MetricFilter> {
    match name.to_lowercase().as_str() {
        "node_exporter" | "node-exporter" | "nodeexporter" => {
            Box::new(NodeExporterFilter::new(node_metadata.unwrap_or_default()))
        }
        "dcgm" | "dcgm_exporter" | "dcgm-exporter" => {
            Box::new(DcgmFilter::new(gpu_metadata.unwrap_or_default()))
        }
        "backend" => Box::new(BackendFilter::new(node_metadata.unwrap_or_default())),
        "frontend" => Box::new(FrontendFilter::new(node_metadata.unwrap_or_default())),
        _ => Box::new(NoOpFilter),
    }
}
