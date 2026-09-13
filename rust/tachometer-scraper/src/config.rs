use crate::filters::get_filter;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub storage: String,
    #[serde(default)]
    pub rows_per_parquet: Option<usize>,
    #[serde(default)]
    pub save_interval_secs: Option<u64>,
    pub endpoints: Vec<EndpointConfig>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct EndpointConfig {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub frequency: Option<f64>,
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub gpu_metadata: Option<HashMap<String, HashMap<String, String>>>,
    #[serde(default)]
    pub node_metadata: Option<HashMap<String, String>>,
}

impl Config {
    /// Collect all unique metadata keys from all endpoints
    /// Returns a sorted vector of column names for the extras schema
    pub fn collect_extra_column_names(&self) -> Vec<String> {
        let mut keys = std::collections::HashSet::new();

        for endpoint in &self.endpoints {
            // Collect GPU metadata keys
            if let Some(ref gpu_metadata) = endpoint.gpu_metadata {
                for gpu_meta in gpu_metadata.values() {
                    for key in gpu_meta.keys() {
                        keys.insert(key.clone());
                    }
                }
            }
            // Collect node metadata keys
            if let Some(ref node_metadata) = endpoint.node_metadata {
                for key in node_metadata.keys() {
                    keys.insert(key.clone());
                }
            }
            // Collect extra column names from filters
            // Filters can dynamically add columns (e.g., DcgmFilter adds "gpu" from Prometheus labels)
            if let Some(ref filter_name) = endpoint.filter {
                let filter = get_filter(
                    filter_name,
                    endpoint.gpu_metadata.clone(),
                    endpoint.node_metadata.clone(),
                );
                let filter_extra_cols = filter.extra_column_names();
                for col_name in filter_extra_cols {
                    keys.insert(col_name);
                }
            }
        }

        let mut sorted_keys: Vec<String> = keys.into_iter().collect();
        sorted_keys.sort();
        sorted_keys
    }
}
