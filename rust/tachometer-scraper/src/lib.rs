pub mod config;
pub mod filters;
pub mod parse;
pub mod runner;

pub use config::Config;
pub use filters::{get_filter, MetricFilter};
pub use parse::ParsedSample;
pub use runner::run_cli;

use tachometer_writer::Row;

/// Scrape a Prometheus endpoint and return rows
pub async fn scrape_endpoint(
    url: &str,
    endpoint_name: &str,
    filter: Option<&dyn MetricFilter>,
) -> Result<Vec<Row>, anyhow::Error> {
    // Handle file:// URLs specially
    let text = if url.starts_with("file://") {
        let path = url.strip_prefix("file://").unwrap();
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read file {}: {}", path, e))?
    } else {
        let response = reqwest::get(url)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch URL {}: {}", url, e))?;
        response
            .text()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read response text: {}", e))?
    };

    // Parse Prometheus text format using our custom parser that preserves _count and _sum
    let samples = parse::parse_prometheus_samples(&text)?;

    // Apply filters and convert to rows
    let rows = if let Some(f) = filter {
        parse::samples_to_rows_with_filter(samples, endpoint_name, f)
    } else {
        parse::samples_to_rows(samples, endpoint_name)
    };

    Ok(rows)
}
