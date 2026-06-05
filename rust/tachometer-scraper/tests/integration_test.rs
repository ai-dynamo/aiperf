use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tachometer_scraper::{get_filter, scrape_endpoint};
use tachometer_writer::{compact_and_upload, DatasetWriter};
use tempfile::TempDir;

/// Helper function to get decompressed sample file path
/// Decompresses .txt.gz files on demand if the .txt file doesn't exist
fn get_sample_file_path(filename: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sample_dir = manifest_dir.join("sample-metrics");
    let txt_path = sample_dir.join(filename);
    let gz_path = sample_dir.join(format!("{}.gz", filename));

    // If .txt doesn't exist but .txt.gz does, decompress it
    if !txt_path.exists() && gz_path.exists() {
        let mut decoder = flate2::read::GzDecoder::new(
            fs::File::open(&gz_path).expect("Failed to open compressed file"),
        );
        let mut decompressed = Vec::new();
        std::io::copy(&mut decoder, &mut decompressed).expect("Failed to decompress file");

        let mut file = fs::File::create(&txt_path).expect("Failed to create decompressed file");
        file.write_all(&decompressed)
            .expect("Failed to write decompressed file");
    }

    txt_path
}

#[tokio::test]
async fn test_dcgm_filter_with_gpu_metadata() {
    // Create test GPU metadata
    let mut gpu_metadata = HashMap::new();
    let mut gpu0_meta = HashMap::new();
    gpu0_meta.insert("endpoint".to_string(), "prefill".to_string());
    gpu0_meta.insert("endpoint_index".to_string(), "0".to_string());
    gpu0_meta.insert("rank_index".to_string(), "0".to_string());
    gpu_metadata.insert("0".to_string(), gpu0_meta);

    let mut gpu2_meta = HashMap::new();
    gpu2_meta.insert("endpoint".to_string(), "decode".to_string());
    gpu2_meta.insert("endpoint_index".to_string(), "0".to_string());
    gpu2_meta.insert("rank_index".to_string(), "0".to_string());
    gpu_metadata.insert("2".to_string(), gpu2_meta);

    let filter = get_filter("dcgm", Some(gpu_metadata), None);

    // Create a sample DCGM metric for GPU 0
    let sample_text = r#"# TYPE DCGM_FI_DEV_SM_CLOCK gauge
DCGM_FI_DEV_SM_CLOCK{gpu="0",UUID="GPU-00000000-0000-0000-0000-000000000000",device="nvidia0"} 120"#;

    let samples = tachometer_scraper::parse::parse_prometheus_samples(sample_text).unwrap();
    assert_eq!(samples.len(), 1);
    let sample = &samples[0];

    // Test filter
    let (metric_name, extras) = filter.filter(sample);

    // Verify metric name is simplified
    assert!(metric_name.contains("SM_CLOCK") || metric_name.contains("sm_clock"));

    // Verify extras contain GPU metadata (now including gpu column)
    assert_eq!(extras.len(), 4); // gpu + worker_role + worker_index + worker_process
    let extras_map: HashMap<String, String> = extras.into_iter().collect();
    assert_eq!(extras_map.get("gpu"), Some(&"0".to_string()));
    assert_eq!(extras_map.get("worker_role"), Some(&"prefill".to_string()));
    assert_eq!(extras_map.get("worker_index"), Some(&"0".to_string()));
    assert_eq!(extras_map.get("worker_process"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_dcgm_filter_without_metadata() {
    let filter = get_filter("dcgm", None, None);

    // Create a sample DCGM metric
    let sample_text = r#"# TYPE DCGM_FI_DEV_SM_CLOCK gauge
DCGM_FI_DEV_SM_CLOCK{gpu="0",UUID="GPU-00000000-0000-0000-0000-000000000000",device="nvidia0"} 120"#;

    let samples = tachometer_scraper::parse::parse_prometheus_samples(sample_text).unwrap();
    let sample = &samples[0];

    let (metric_name, extras) = filter.filter(sample);

    // Metric name should still be filtered
    assert!(!metric_name.is_empty());
    // Should only have gpu column, no worker metadata since none provided
    assert_eq!(extras.len(), 1);
    let extras_map: HashMap<String, String> = extras.into_iter().collect();
    assert_eq!(extras_map.get("gpu"), Some(&"0".to_string()));
}

#[tokio::test]
async fn test_dcgm_filter_extracts_gpu_tag_from_prometheus_labels() {
    // Test that DcgmFilter correctly extracts the gpu=".." label from Prometheus metrics
    let filter = get_filter("dcgm", None, None);

    // Test various GPU indices
    let test_cases = vec![
        (
            r#"DCGM_FI_DEV_MEMORY_TOTAL{gpu="0",device="nvidia0"} 100"#,
            "0",
        ),
        (
            r#"DCGM_FI_DEV_MEMORY_TOTAL{gpu="1",device="nvidia1"} 100"#,
            "1",
        ),
        (
            r#"DCGM_FI_DEV_MEMORY_TOTAL{gpu="2",UUID="GPU-test"} 100"#,
            "2",
        ),
        (r#"DCGM_FI_DEV_GPU_UTIL{gpu="7",device="nvidia7"} 50"#, "7"),
    ];

    for (metric_text, expected_gpu) in test_cases {
        let sample_text = format!("# TYPE test_metric gauge\n{}", metric_text);
        let samples = tachometer_scraper::parse::parse_prometheus_samples(&sample_text).unwrap();
        assert_eq!(samples.len(), 1, "Should parse one sample");
        let sample = &samples[0];

        let (_metric_name, extras) = filter.filter(sample);
        let extras_map: HashMap<String, String> = extras.into_iter().collect();

        assert_eq!(
            extras_map.get("gpu"),
            Some(&expected_gpu.to_string()),
            "Should extract gpu={} from metric: {}",
            expected_gpu,
            metric_text
        );
    }

    // Test that GPU is extracted even when other labels are present
    let complex_metric = r#"# TYPE DCGM_FI_DEV_SM_CLOCK gauge
DCGM_FI_DEV_SM_CLOCK{gpu="3",UUID="GPU-00000000-0000-0000-0000-000000000000",pci_bus_id="00000008:06:00.0",device="nvidia3",modelName="NVIDIA GB200",Hostname="sample-gb200-host.example.com",DCGM_FI_DRIVER_VERSION="580.105.05"} 120"#;

    let samples = tachometer_scraper::parse::parse_prometheus_samples(complex_metric).unwrap();
    let sample = &samples[0];
    let (_metric_name, extras) = filter.filter(sample);
    let extras_map: HashMap<String, String> = extras.into_iter().collect();

    assert_eq!(
        extras_map.get("gpu"),
        Some(&"3".to_string()),
        "Should extract gpu=3 from complex metric with many labels"
    );
}

#[tokio::test]
async fn test_parse_sample_file_and_verify_rows() {
    // Get path to sample file (decompresses if needed)
    let sample_path = get_sample_file_path("dcgm-gb200.txt");

    // Create GPU metadata matching test-config.toml
    let mut gpu_metadata = HashMap::new();

    let mut gpu0_meta = HashMap::new();
    gpu0_meta.insert("endpoint".to_string(), "prefill".to_string());
    gpu0_meta.insert("endpoint_index".to_string(), "0".to_string());
    gpu0_meta.insert("rank_index".to_string(), "0".to_string());
    gpu_metadata.insert("0".to_string(), gpu0_meta);

    let mut gpu1_meta = HashMap::new();
    gpu1_meta.insert("endpoint".to_string(), "prefill".to_string());
    gpu1_meta.insert("endpoint_index".to_string(), "0".to_string());
    gpu1_meta.insert("rank_index".to_string(), "1".to_string());
    gpu_metadata.insert("1".to_string(), gpu1_meta);

    let mut gpu2_meta = HashMap::new();
    gpu2_meta.insert("endpoint".to_string(), "decode".to_string());
    gpu2_meta.insert("endpoint_index".to_string(), "0".to_string());
    gpu2_meta.insert("rank_index".to_string(), "0".to_string());
    gpu_metadata.insert("2".to_string(), gpu2_meta);

    let mut gpu3_meta = HashMap::new();
    gpu3_meta.insert("endpoint".to_string(), "decode".to_string());
    gpu3_meta.insert("endpoint_index".to_string(), "0".to_string());
    gpu3_meta.insert("rank_index".to_string(), "1".to_string());
    gpu_metadata.insert("3".to_string(), gpu3_meta);

    let filter = get_filter("dcgm", Some(gpu_metadata), None);

    // Read and parse the file using our custom parser
    let text = std::fs::read_to_string(&sample_path).unwrap();
    let samples = tachometer_scraper::parse::parse_prometheus_samples(&text).unwrap();

    // Apply filter and track metadata
    let mut gpu0_rows = 0;
    let mut gpu2_rows = 0;
    let mut rows_with_extras = 0;
    let mut rows_without_extras = 0;

    for sample in &samples {
        let (metric_name, extras) = filter.filter(sample);

        // Verify extras structure
        if !extras.is_empty() {
            rows_with_extras += 1;

            // Check sorting (should be sorted by key)
            let sorted_keys: Vec<String> = extras.iter().map(|(k, _)| k.clone()).collect();
            let mut sorted_keys_check = sorted_keys.clone();
            sorted_keys_check.sort();
            assert_eq!(
                sorted_keys, sorted_keys_check,
                "Extras should be sorted by key"
            );

            // Verify specific GPU metadata
            let extras_map: HashMap<String, String> = extras.into_iter().collect();

            // Check GPU 0 metrics
            if let Some(gpu_id) = extras_map.get("gpu") {
                if gpu_id == "0" {
                    gpu0_rows += 1;
                    assert_eq!(extras_map.get("worker_role"), Some(&"prefill".to_string()));
                    assert_eq!(extras_map.get("worker_process"), Some(&"0".to_string()));
                } else if gpu_id == "2" {
                    gpu2_rows += 1;
                    assert_eq!(extras_map.get("worker_role"), Some(&"decode".to_string()));
                    assert_eq!(extras_map.get("worker_process"), Some(&"0".to_string()));
                }
            }
        } else {
            rows_without_extras += 1;
        }

        // Verify metric name isn't empty
        assert!(!metric_name.is_empty());
    }

    // Verify we got rows with metadata
    assert!(gpu0_rows > 0, "Should have found GPU 0 metrics");
    assert!(gpu2_rows > 0, "Should have found GPU 2 metrics");
    assert!(rows_with_extras > 0, "Should have rows with extras");

    println!(
        "Found {} rows with extras, {} without",
        rows_with_extras, rows_without_extras
    );
    println!("GPU 0 rows: {}, GPU 2 rows: {}", gpu0_rows, gpu2_rows);
}

#[tokio::test]
async fn test_scrape_endpoint_with_metadata() {
    // Get path to sample file (decompresses if needed)
    let sample_path = get_sample_file_path("dcgm-gb200.txt");

    let url = format!("file://{}", sample_path.to_str().unwrap());

    // Create GPU metadata
    let mut gpu_metadata = HashMap::new();

    let mut gpu0_meta = HashMap::new();
    gpu0_meta.insert("endpoint".to_string(), "prefill".to_string());
    gpu0_meta.insert("endpoint_index".to_string(), "0".to_string());
    gpu0_meta.insert("rank_index".to_string(), "0".to_string());
    gpu_metadata.insert("0".to_string(), gpu0_meta);

    let filter = get_filter("dcgm", Some(gpu_metadata), None);

    // Test scrape_endpoint function
    let rows = scrape_endpoint(&url, "test_endpoint", Some(filter.as_ref()))
        .await
        .unwrap();

    // Verify we got rows
    assert!(!rows.is_empty(), "Should have parsed some rows");

    // Verify row structure
    let mut gpu0_row_found = false;
    for row in &rows {
        // Verify all required fields are present
        assert!(!row.scraper_endpoint.is_empty());
        assert!(!row.metric_name.is_empty());

        // Check if this is a GPU 0 row (should have extras)
        if !row.extras.is_empty() {
            let extras_map: HashMap<String, String> = row
                .extras
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            if extras_map.get("worker_role") == Some(&"prefill".to_string()) {
                gpu0_row_found = true;
                assert_eq!(extras_map.get("worker_index"), Some(&"0".to_string()));
                assert_eq!(extras_map.get("worker_process"), Some(&"0".to_string()));
            }
        }
    }

    assert!(
        gpu0_row_found,
        "Should have found at least one GPU 0 row with metadata"
    );

    println!("Parsed {} rows total", rows.len());
}

#[tokio::test]
async fn test_node_exporter_filter_simplifies_cpu_metrics() {
    // Get path to sample file (decompresses if needed)
    let sample_path = get_sample_file_path("node-exporter-gb200.txt");

    let url = format!("file://{}", sample_path.to_str().unwrap());

    let filter = get_filter("node_exporter", None, None);

    // Test scrape_endpoint function
    let rows = scrape_endpoint(&url, "test_node", Some(filter.as_ref()))
        .await
        .unwrap();

    // Verify we got rows
    assert!(!rows.is_empty(), "Should have parsed some rows");

    // Count CPU-related metrics - should be simplified
    let mut cpu_metrics = std::collections::HashSet::new();
    let mut total_rows = 0;

    for row in &rows {
        total_rows += 1;
        if row.metric_name.contains("cpu") || row.metric_name.contains("CPU") {
            cpu_metrics.insert(row.metric_name.clone());
        }
    }

    // Verify CPU metrics are aggregated - should have min/max/p10/p90 as metric names
    // Mode is now a metadata column (in extras), not embedded in metric name
    // The filter should convert node_cpu_seconds_total{cpu="0",mode="idle"} to:
    // metric_name: cpu_time_min, extras: [("mode", "idle")]
    let cpu_time_metrics: Vec<_> = cpu_metrics
        .iter()
        .filter(|m| {
            *m == "cpu_time_min"
                || *m == "cpu_time_max"
                || *m == "cpu_time_p10"
                || *m == "cpu_time_p90"
        })
        .collect();

    // Should have clean metric names (no labels embedded)
    for metric_name in &cpu_time_metrics {
        // Should NOT contain any labels in metric name
        assert!(
            !metric_name.contains("cpu="),
            "CPU metrics should not contain cpu= label after filtering. Found: {}",
            metric_name
        );
        // Mode should NOT be in metric name (it's in extras now)
        assert!(
            !metric_name.contains("mode="),
            "CPU metrics should NOT contain mode= label in name (it's in extras). Found: {}",
            metric_name
        );
    }

    // Verify we have exactly 4 unique metric names (min, max, p10, p90)
    // Each will have multiple rows with different mode values in extras
    assert_eq!(
        cpu_time_metrics.len(),
        4,
        "Should have 4 unique CPU time metric names. Found: {:?}",
        cpu_time_metrics
    );

    println!(
        "Found {} CPU-related metrics out of {} total rows",
        cpu_metrics.len(),
        total_rows
    );
    println!("CPU time metrics: {:?}", cpu_time_metrics);
}

#[tokio::test]
async fn test_node_exporter_with_node_metadata() {
    // Test that we can parse node_exporter metrics
    let sample_path = get_sample_file_path("node-exporter-gb200.txt");

    let text = std::fs::read_to_string(&sample_path).unwrap();
    let samples = tachometer_scraper::parse::parse_prometheus_samples(&text).unwrap();

    let filter = get_filter("node_exporter", None, None);

    // Test a CPU metric sample (simulating post-aggregation output)
    // Note: CPU aggregation creates samples with metric_name "node_cpu_time_{stat}"
    // and mode in labels (which becomes extras after filtering)
    let mut cpu_sample_labels = std::collections::HashMap::new();
    cpu_sample_labels.insert("mode".to_string(), "idle".to_string());
    let cpu_sample = tachometer_scraper::parse::ParsedSample {
        metric_name: "node_cpu_time_min".to_string(), // Aggregation puts stat in metric name
        labels: cpu_sample_labels,
        value: 12345.0,
        metric_type: tachometer_scraper::parse::MetricType::Counter,
    };

    let (metric_name, extras) = filter.filter(&cpu_sample);

    // Verify metric is clean (no labels in name)
    assert_eq!(
        metric_name, "cpu_time_min",
        "Metric should be cpu_time_min. Got: {}",
        metric_name
    );
    assert!(
        !metric_name.contains("mode="),
        "Metric should NOT contain mode= in name (mode is in extras). Got: {}",
        metric_name
    );
    assert!(
        !metric_name.contains("cpu="),
        "Metric should NOT contain cpu= label. Got: {}",
        metric_name
    );

    // Mode should be extracted as extras (metadata column)
    assert_eq!(
        extras.len(),
        1,
        "Should have mode in extras. Got: {:?}",
        extras
    );
    assert_eq!(
        extras[0],
        ("mode".to_string(), "idle".to_string()),
        "Mode extra should be ('mode', 'idle'). Got: {:?}",
        extras
    );

    // Test a non-CPU metric (should pass through with simplification)
    let infiband_sample = samples
        .iter()
        .find(|s| s.metric_name.starts_with("node_infiniband"))
        .expect("Should find an infiniband sample");

    let (ib_metric_name, ib_extras) = filter.filter(infiband_sample);

    // Should remove node_ prefix
    assert!(
        !ib_metric_name.starts_with("node_"),
        "Should remove node_ prefix. Got: {}",
        ib_metric_name
    );
    assert_eq!(ib_extras.len(), 0);

    println!("CPU metric filtered to: {}", metric_name);
    println!("Infiniband metric filtered to: {}", ib_metric_name);
}

#[tokio::test]
async fn test_node_exporter_with_node_metadata_columns() {
    // Get path to sample file (decompresses if needed)
    let sample_path = get_sample_file_path("node-exporter-gb200.txt");

    let url = format!("file://{}", sample_path.to_str().unwrap());

    // Create node metadata
    let mut node_metadata = HashMap::new();
    node_metadata.insert("cluster".to_string(), "gb200-cluster".to_string());
    node_metadata.insert("node_id".to_string(), "sample-node-0073".to_string());
    node_metadata.insert("rack".to_string(), "rack-1".to_string());

    let filter = get_filter("node_exporter", None, Some(node_metadata.clone()));

    // Test scrape_endpoint function
    let rows = scrape_endpoint(&url, "test_node", Some(filter.as_ref()))
        .await
        .unwrap();

    // Verify we got rows
    assert!(!rows.is_empty(), "Should have parsed some rows");

    // Verify all rows have node metadata attached
    let mut rows_with_metadata = 0;
    for row in &rows {
        if !row.extras.is_empty() {
            rows_with_metadata += 1;

            // Verify extras contain node metadata
            let extras_map: HashMap<String, String> = row
                .extras
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            // Note: cluster, node_id, rack are NOT renamed, only standard keys like "node" -> "hostname"
            assert_eq!(
                extras_map.get("cluster"),
                Some(&"gb200-cluster".to_string())
            );
            assert_eq!(extras_map.get("node_id"), Some(&"sample-node-0073".to_string()));
            assert_eq!(extras_map.get("rack"), Some(&"rack-1".to_string()));

            // Verify extras are sorted
            let sorted_keys: Vec<String> = row.extras.iter().map(|(k, _)| k.clone()).collect();
            let mut sorted_keys_check = sorted_keys.clone();
            sorted_keys_check.sort();
            assert_eq!(
                sorted_keys, sorted_keys_check,
                "Extras should be sorted by key"
            );
        }
    }

    // All rows should have metadata
    assert_eq!(
        rows_with_metadata,
        rows.len(),
        "All rows should have node metadata"
    );

    // Verify CPU metrics are aggregated with clean names (mode in extras)
    let cpu_rows: Vec<_> = rows
        .iter()
        .filter(|r| {
            r.metric_name == "cpu_time_min"
                || r.metric_name == "cpu_time_max"
                || r.metric_name == "cpu_time_p10"
                || r.metric_name == "cpu_time_p90"
        })
        .collect();

    assert!(!cpu_rows.is_empty(), "Should have CPU metrics");
    for cpu_row in &cpu_rows {
        assert!(
            !cpu_row.metric_name.contains("cpu="),
            "CPU metrics should not contain cpu= label"
        );
        // Mode should NOT be in metric name (it's in extras now)
        assert!(
            !cpu_row.metric_name.contains("mode="),
            "CPU metrics should NOT contain mode= label in name"
        );
        // Mode should be in extras
        let has_mode = cpu_row.extras.iter().any(|(k, _)| k == "mode");
        assert!(has_mode, "CPU metrics should have mode in extras");
    }

    println!("Parsed {} rows, all with node metadata", rows.len());
    println!("Found {} CPU time metrics", cpu_rows.len());
}

#[tokio::test]
async fn test_gpu_column_appears_in_parquet_output() {
    // Test that when using a DCGM filter, the "gpu" column is actually written to parquet
    use arrow::array::{Array, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::sync::Arc;
    use tachometer_scraper::Config;
    use tachometer_writer::DatasetWriter;

    // Create a temporary directory for test output
    let temp_dir = TempDir::new().unwrap();
    let test_path = temp_dir.path().join("test_output");
    std::fs::create_dir_all(&test_path).unwrap();

    // Create a config with DCGM endpoint
    let sample_path = get_sample_file_path("dcgm-gb200.txt");
    let config_toml = format!(
        r#"
storage = "file://{}"

[[endpoints]]
name = "dcgm_test"
url = "file://{}"
frequency = 5.0
filter = "dcgm"

[endpoints.gpu_metadata]
0 = {{worker_index = "0", worker_process = "0", worker_role = "test"}}
"#,
        test_path.to_str().unwrap(),
        sample_path.to_str().unwrap()
    );

    let config: Config = toml::from_str(&config_toml).unwrap();

    // Collect extra column names - should include "gpu"
    let extra_column_names = config.collect_extra_column_names();
    assert!(
        extra_column_names.contains(&"gpu".to_string()),
        "collect_extra_column_names should include 'gpu' for DCGM filter. Got: {:?}",
        extra_column_names
    );

    // Create DatasetWriter with the extra columns
    // Use small rows_per_parquet so we get a file quickly
    let writer = Arc::new(
        DatasetWriter::new(
            test_path.clone(),
            10,   // rows_per_parquet - small so we get a file quickly
            1000, // save_interval_secs (long to avoid periodic saves during test)
            extra_column_names.clone(),
        )
        .unwrap(),
    );

    // Scrape the endpoint and write rows
    let url = format!("file://{}", sample_path.to_str().unwrap());

    let mut gpu_metadata = HashMap::new();
    let mut gpu0_meta = HashMap::new();
    gpu0_meta.insert("worker_index".to_string(), "0".to_string());
    gpu0_meta.insert("worker_process".to_string(), "0".to_string());
    gpu0_meta.insert("worker_role".to_string(), "test".to_string());
    gpu_metadata.insert("0".to_string(), gpu0_meta);

    let filter = get_filter("dcgm", Some(gpu_metadata), None);
    let rows = scrape_endpoint(&url, "dcgm_test", Some(filter.as_ref()))
        .await
        .unwrap();

    assert!(!rows.is_empty(), "Should have scraped some rows");

    // Write rows to parquet
    writer.append_rows(rows).await.unwrap();

    // Flush and shutdown writer to ensure data is written
    writer.shutdown().await.unwrap();

    // Find the parquet file
    let parquet_files: Vec<_> = std::fs::read_dir(&test_path)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str() == Some("parquet") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    assert!(
        !parquet_files.is_empty(),
        "Should have created at least one parquet file"
    );

    // Read the first parquet file and verify "gpu" column exists
    let parquet_path = &parquet_files[0];
    let file = std::fs::File::open(parquet_path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.build().unwrap();

    // Read all batches
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.unwrap());
    }

    assert!(!batches.is_empty(), "Should have at least one batch");

    // Check schema for "gpu" column
    let schema = batches[0].schema();
    let gpu_field = schema.field_with_name("gpu");
    assert!(
        gpu_field.is_ok(),
        "Schema should contain 'gpu' column. Available columns: {:?}",
        schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
    );

    // Verify that some rows have non-empty gpu values
    let mut rows_with_gpu = 0;
    for batch in &batches {
        let gpu_array = batch.column_by_name("gpu").expect("Should have gpu column");
        let gpu_string_array = gpu_array
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("gpu column should be string array");

        for i in 0..gpu_string_array.len() {
            if !gpu_string_array.is_null(i) {
                let gpu_value = gpu_string_array.value(i);
                if !gpu_value.is_empty() {
                    rows_with_gpu += 1;
                    // Verify GPU value is a valid number
                    assert!(
                        gpu_value.parse::<u32>().is_ok(),
                        "GPU value should be a number, got: {}",
                        gpu_value
                    );
                }
            }
        }
    }

    assert!(
        rows_with_gpu > 0,
        "Should have at least one row with a non-empty gpu value"
    );

    println!(
        "Successfully verified 'gpu' column in parquet file with {} rows containing GPU values",
        rows_with_gpu
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_compact_command_end_to_end() {
    // End-to-end test simulating the tachometer-scraper compact subcommand
    // This tests the full workflow: scrape → write → compact
    use arrow::array::Array;
    use object_store::local::LocalFileSystem;
    use object_store::ObjectStore;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let temp_dir = TempDir::new().unwrap();
    let local_dir = temp_dir.path().join("test_run");
    std::fs::create_dir_all(&local_dir).unwrap();

    // Create a DatasetWriter with small threshold to create multiple parquet files
    let extra_columns = vec![
        "gpu".to_string(),
        "worker_index".to_string(),
        "worker_process".to_string(),
        "worker_role".to_string(),
    ];

    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            50,   // rows_per_parquet - creates multiple files from sample data
            1000, // save_interval_secs
            extra_columns,
        )
        .unwrap(),
    );

    // Scrape DCGM sample data with GPU metadata
    let sample_path = get_sample_file_path("dcgm-gb200.txt");
    let url = format!("file://{}", sample_path.to_str().unwrap());

    let mut gpu_metadata = HashMap::new();
    for gpu_id in 0..4 {
        let mut meta = HashMap::new();
        meta.insert("worker_index".to_string(), "0".to_string());
        meta.insert("worker_process".to_string(), gpu_id.to_string());
        meta.insert(
            "worker_role".to_string(),
            if gpu_id < 2 {
                "prefill".to_string()
            } else {
                "decode".to_string()
            },
        );
        gpu_metadata.insert(gpu_id.to_string(), meta);
    }

    let filter = get_filter("dcgm", Some(gpu_metadata), None);
    let rows = scrape_endpoint(&url, "dcgm_test", Some(filter.as_ref()))
        .await
        .unwrap();

    println!("Scraped {} rows from sample file", rows.len());
    assert!(!rows.is_empty(), "Should have scraped some rows");

    // Write rows - do it twice to ensure we get multiple parquet files
    writer.append_rows(rows.clone()).await.unwrap();
    writer.append_rows(rows).await.unwrap();
    writer.shutdown().await.unwrap();

    // Verify intermediate files exist
    let intermediate_files: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.ends_with(".parquet") || name == "current.arrow"
        })
        .collect();

    assert!(
        intermediate_files.len() >= 2,
        "Should have at least 2 intermediate files. Found: {:?}",
        intermediate_files
            .iter()
            .map(|e| e.file_name())
            .collect::<Vec<_>>()
    );

    println!(
        "Created {} intermediate files before compaction",
        intermediate_files.len()
    );

    // Create output storage (same as what run_compact does)
    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();
    let store: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(&output_dir).unwrap());
    let remote_path = "test_run".to_string();

    // Run compaction
    compact_and_upload(&local_dir, store.clone(), &remote_path)
        .await
        .expect("Compaction should succeed");

    // Verify final.parquet was created
    let final_path = output_dir.join("test_run").join("final.parquet");
    assert!(
        final_path.exists(),
        "final.parquet should exist at {:?}",
        final_path
    );

    // Read and verify the compacted file
    let file = std::fs::File::open(&final_path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.build().unwrap();
    let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();

    assert!(!batches.is_empty(), "Should have batches in final.parquet");

    // Verify schema has expected columns
    let schema = batches[0].schema();
    let required_columns = [
        "scraper_endpoint",
        "metric_name",
        "metric_name_clean", // Added by compaction
        "metric_value",
        "time_since_start",
        "gpu",
        "worker_role",
    ];

    for col_name in &required_columns {
        assert!(
            schema.fields().iter().any(|f| f.name() == *col_name),
            "Schema should have column '{}'. Available: {:?}",
            col_name,
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        );
    }

    // Count total rows
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("Compacted {} rows into final.parquet", total_rows);
    assert!(total_rows > 100, "Should have significant number of rows");

    // Verify sorting by checking first few metric_name values
    let metric_names = batches[0]
        .column_by_name("metric_name")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
        .expect("metric_name should be LargeStringArray");

    // Check that rows are sorted (each metric_name should be >= previous)
    let mut prev_name = "";
    for i in 0..std::cmp::min(100, metric_names.len()) {
        if !metric_names.is_null(i) {
            let name = metric_names.value(i);
            assert!(
                name >= prev_name,
                "Data should be sorted by metric_name: '{}' should come after '{}'",
                name,
                prev_name
            );
            prev_name = name;
        }
    }

    // Verify intermediate files were cleaned up
    let remaining_intermediate: Vec<_> = std::fs::read_dir(&local_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("out-") || name == "current.arrow"
        })
        .collect();

    assert!(
        remaining_intermediate.is_empty(),
        "Intermediate files should be cleaned up after compaction. Remaining: {:?}",
        remaining_intermediate
            .iter()
            .map(|e| e.file_name())
            .collect::<Vec<_>>()
    );

    println!("End-to-end compact test passed!");
}
