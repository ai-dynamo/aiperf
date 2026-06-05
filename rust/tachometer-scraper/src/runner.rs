use crate::{get_filter, scrape_endpoint, Config, MetricFilter};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use log::{error, info, warn};
use object_store::{path::Path as ObjectPath, ObjectStore};
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use tachometer_writer::{compact_and_upload, periodic_compact_and_sync, DatasetWriter};
use tokio::signal;
#[cfg(unix)]
use tokio::signal::unix::{signal as unix_signal, SignalKind};
use url::Url;

#[derive(Parser, Debug)]
#[command(name = "tachometer-scraper")]
#[command(about = "Scrapes Prometheus metrics and logs them to Tachometer")]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Configuration file (TOML format) - mutually exclusive with --endpoint
    #[arg(long = "config", value_name = "PATH")]
    config: Option<String>,

    /// Prometheus endpoints in format name=url (for debugging, mutually exclusive with --config)
    #[arg(long = "endpoint", value_name = "NAME=URL", num_args = 0..)]
    endpoints: Vec<String>,

    /// Polling frequency in Hz (e.g., 0.1 for 10 seconds) - used only with --endpoint
    #[arg(long = "freq", default_value = "0.2")]
    frequency: f64,

    /// Storage location (e.g., "s3://bucket/run_0" or "./local/path") - used only with --endpoint
    #[arg(long = "storage", value_name = "PATH")]
    storage: Option<String>,

    /// Number of rows before creating a numbered parquet file - used only with --endpoint
    #[arg(long = "rows-per-parquet", default_value = "1000000")]
    rows_per_parquet: usize,

    /// Interval in seconds for periodic Arrow file saves - used only with --endpoint
    #[arg(long = "save-interval", default_value = "5")]
    save_interval_secs: u64,

    /// Metric filter to apply (e.g., "node_exporter", "dcgm") - used only with --endpoint
    /// Can be specified multiple times, one per endpoint in order
    #[arg(long = "filter", value_name = "FILTER_NAME")]
    filters: Vec<String>,

    /// Local directory for intermediate files (default: /tmp/tachometer-{pid})
    #[arg(long = "local-dir", value_name = "PATH")]
    local_dir: Option<String>,

    /// Interval in seconds for periodic sync to remote storage (0 = disabled)
    /// Compacts current out-*.parquet files into incomplete-N.parquet and uploads to remote
    #[arg(long = "sync-interval", default_value = "0")]
    sync_interval_secs: u64,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compact intermediate files from a local directory into final.parquet
    ///
    /// S3 credentials are read from environment variables:
    ///   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (or AWS_DEFAULT_REGION)
    ///   AWS_ENDPOINT (optional, for S3-compatible storage like MinIO)
    Compact {
        /// Local directory containing intermediate files (current.arrow, out-*.parquet, incomplete-*.parquet)
        #[arg(value_name = "INPUT_DIR")]
        input_dir: PathBuf,

        /// Output storage location (e.g., "s3://bucket/run_0" or "./local/path")
        /// If not specified, writes final.parquet to the input directory
        #[arg(long = "output", short = 'o', value_name = "PATH")]
        output: Option<String>,

        /// Configuration file (TOML) to read storage URL from
        /// If provided, uses the 'storage' field from the config
        #[arg(long = "config", short = 'c', value_name = "CONFIG_FILE")]
        config: Option<String>,
    },
}

struct EndpointConfig {
    name: String,
    url: String,
    frequency: f64,
    filter: Option<Box<dyn MetricFilter>>,
}

fn parse_endpoint(s: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err("Expected format: name=url".into());
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

fn validate_frequency(frequency: f64) -> Result<(), Box<dyn std::error::Error>> {
    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(format!(
            "Polling frequency must be finite and greater than zero, got {}",
            frequency
        )
        .into());
    }
    Ok(())
}

async fn parse_storage(
    storage: &str,
) -> Result<(Box<dyn ObjectStore>, String), Box<dyn std::error::Error>> {
    // Try parsing as URL first
    if let Ok(url) = Url::parse(storage) {
        match url.scheme() {
            "s3" | "s3a" | "s3n" => {
                // S3 URL: s3://bucket/path
                let bucket = url
                    .host_str()
                    .ok_or("S3 URL must include bucket name (e.g., s3://bucket/path)")?;
                let path = url.path().trim_start_matches('/').to_string();

                info!("Configuring S3 storage:");
                info!("  Bucket: {}", bucket);
                info!("  Path: {}", if path.is_empty() { "/" } else { &path });

                let store = object_store::aws::AmazonS3Builder::from_env()
                    .with_bucket_name(bucket)
                    .build()
                    .map_err(|e| format!("Failed to create S3 object store: {}", e))?;

                info!("  S3 client created successfully");
                info!("  S3 client config: {:?}", store);

                // Test S3 connectivity before proceeding
                info!("Testing S3 connectivity...");
                test_s3_connectivity_impl(&store, &path).await?;

                Ok((Box::new(store), path))
            }
            "file" => {
                // File URL: file:///absolute/path or file://relative/path
                let path_str = url.path();
                let path = std::path::Path::new(path_str);
                let absolute_path = if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    std::env::current_dir()?.join(path)
                };

                let normalized_path = absolute_path.components().collect::<std::path::PathBuf>();

                if normalized_path.exists() && normalized_path.is_dir() {
                    let run_path = normalized_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default();

                    let parent = normalized_path
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new("/"));

                    let canonical_prefix = if parent.exists() {
                        parent
                            .canonicalize()
                            .unwrap_or_else(|_| parent.to_path_buf())
                    } else {
                        parent.to_path_buf()
                    };

                    let store =
                        object_store::local::LocalFileSystem::new_with_prefix(&canonical_prefix)
                            .map_err(|e| {
                                format!("Failed to create local filesystem store: {}", e)
                            })?;

                    Ok((Box::new(store), run_path))
                } else {
                    let parent = normalized_path.parent().ok_or("Invalid file path")?;
                    let run_path = normalized_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_string())
                        .ok_or("Invalid file path")?;

                    let canonical_prefix = if parent.exists() {
                        parent
                            .canonicalize()
                            .unwrap_or_else(|_| parent.to_path_buf())
                    } else {
                        parent.to_path_buf()
                    };

                    let store =
                        object_store::local::LocalFileSystem::new_with_prefix(&canonical_prefix)
                            .map_err(|e| {
                                format!("Failed to create local filesystem store: {}", e)
                            })?;

                    Ok((Box::new(store), run_path))
                }
            }
            _ => Err(format!(
                "Unsupported URL scheme: {}. Supported schemes: s3://, file://, or local path",
                url.scheme()
            )
            .into()),
        }
    } else {
        // Treat as local filesystem path (relative or absolute)
        let path = std::path::Path::new(storage);
        let absolute_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };

        let normalized_path = if absolute_path.exists() {
            absolute_path
                .canonicalize()
                .unwrap_or_else(|_| absolute_path.components().collect::<std::path::PathBuf>())
        } else {
            let mut normalized = std::path::PathBuf::new();
            for component in absolute_path.components() {
                match component {
                    std::path::Component::CurDir => continue,
                    _ => normalized.push(component),
                }
            }
            normalized
        };

        let (prefix, run_path) = if normalized_path.exists() && normalized_path.is_dir() {
            // If storage path is an existing directory, require a nested structure
            // Format: YYYY-MM-DD/run-name
            // Return error to guide user to use proper nested path
            return Err(format!(
                "Storage path '{}' is an existing directory. Please specify a nested path like 'YYYY-MM-DD/run-name' (e.g., '2024-12-05/my-run').",
                storage
            ).into());
        } else {
            let parent = normalized_path.parent().ok_or("Invalid storage path")?;
            let file_name = normalized_path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .unwrap_or_default();
            (parent, file_name)
        };

        if !prefix.exists() {
            std::fs::create_dir_all(prefix).map_err(|e| {
                format!(
                    "Failed to create storage directory {}: {}",
                    prefix.display(),
                    e
                )
            })?;
        }

        let canonical_prefix = prefix
            .canonicalize()
            .map_err(|e| format!("Failed to canonicalize path {}: {}", prefix.display(), e))?;

        let store = object_store::local::LocalFileSystem::new_with_prefix(&canonical_prefix)
            .map_err(|e| format!("Failed to create local filesystem store: {}", e))?;

        Ok((Box::new(store), run_path))
    }
}

/// Test S3 connectivity by attempting to write, read, and delete a test file
async fn test_s3_connectivity_impl(
    store: &object_store::aws::AmazonS3,
    base_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_path = if base_path.is_empty() {
        ObjectPath::from(".tachometer_connectivity_test")
    } else {
        ObjectPath::from(format!("{}/.tachometer_connectivity_test", base_path))
    };

    // Try to write a small test file
    let test_data = bytes::Bytes::from("connectivity test");
    match store.put(&test_path, test_data.into()).await {
        Ok(_) => {
            info!("  ✓ Successfully wrote test file to S3");

            // Try to read it back
            match store.get(&test_path).await {
                Ok(_) => {
                    info!("  ✓ Successfully read test file from S3");

                    // Clean up test file
                    if let Err(e) = store.delete(&test_path).await {
                        warn!("  ⚠ Could not delete test file (non-critical): {}", e);
                    } else {
                        info!("  ✓ Successfully deleted test file");
                    }

                    info!("S3 connectivity check passed!");
                    Ok(())
                }
                Err(e) => {
                    error!("  ✗ Failed to read test file from S3: {}", e);
                    Err(format!("Failed to read test file from S3: {}", e).into())
                }
            }
        }
        Err(e) => {
            // Try to list as a fallback (read-only might work)
            info!("  Write test failed, trying read-only list operation...");
            let list_path = if base_path.is_empty() {
                None
            } else {
                Some(ObjectPath::from(base_path))
            };

            match store.list(list_path.as_ref()).next().await {
                Some(Ok(_)) => {
                    error!("  ✗ Can list S3 but cannot write. Check IAM permissions!");
                    Err(format!("S3 write access denied: {}", e).into())
                }
                Some(Err(list_err)) => {
                    error!("  ✗ Cannot write or list S3. Check credentials and bucket access.");
                    Err(format!(
                        "S3 connectivity check failed - Write error: {}, List error: {}",
                        e, list_err
                    )
                    .into())
                }
                None => {
                    // Empty list is still successful
                    error!("  ✗ Can list S3 (empty) but cannot write. Check IAM permissions!");
                    Err(format!("S3 write access denied: {}", e).into())
                }
            }
        }
    }
}

/// Get the default local directory for intermediate files
fn get_default_local_dir() -> PathBuf {
    let pid = std::process::id();
    PathBuf::from(format!("/tmp/tachometer-{}", pid))
}

/// Run the compact command
async fn run_compact(
    input_dir: PathBuf,
    output: Option<String>,
    config_path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting compaction of: {}", input_dir.display());

    // Verify input directory exists
    if !input_dir.exists() {
        return Err(format!("Input directory does not exist: {}", input_dir.display()).into());
    }
    if !input_dir.is_dir() {
        return Err(format!("Input path is not a directory: {}", input_dir.display()).into());
    }

    // List files in the input directory
    let mut found_files = Vec::new();
    for entry in std::fs::read_dir(&input_dir)? {
        let entry = entry?;
        let filename = entry.file_name().to_string_lossy().to_string();
        if filename.ends_with(".parquet") || filename == "current.arrow" {
            found_files.push(filename);
        }
    }
    info!(
        "Found {} intermediate files: {:?}",
        found_files.len(),
        found_files
    );

    if found_files.is_empty() {
        return Err("No intermediate files found in input directory".into());
    }

    // Determine output storage URL (priority: --output > --config > input directory)
    let output_url = if let Some(url) = output {
        Some(url)
    } else if let Some(cfg_path) = config_path {
        // Read storage URL from config file
        let config_content = std::fs::read_to_string(&cfg_path)
            .map_err(|e| format!("Failed to read config file {}: {}", cfg_path, e))?;
        let config: Config = toml::from_str(&config_content)
            .map_err(|e| format!("Failed to parse config file: {}", e))?;
        info!("Read storage URL from config: {}", config.storage);
        Some(config.storage)
    } else {
        None
    };

    // Determine output storage
    let (store, remote_path): (Arc<dyn ObjectStore>, String) = if let Some(output_path) = output_url
    {
        let (store_obj, path) = parse_storage(&output_path).await?;
        (Arc::new(store_obj), path)
    } else {
        // Default: write to input directory
        let canonical = input_dir.canonicalize()?;
        let parent = canonical.parent().ok_or("Invalid input directory")?;
        let run_name = canonical
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("Invalid input directory name")?
            .to_string();

        let store = object_store::local::LocalFileSystem::new_with_prefix(parent)?;
        (Arc::new(store), run_name)
    };

    info!("Output: {} (path: {})", "local/remote", remote_path);

    // Run compaction
    match compact_and_upload(&input_dir, store, &remote_path).await {
        Ok(()) => {
            info!("Compaction completed successfully!");
            Ok(())
        }
        Err(e) => {
            error!("Compaction failed: {}", e);
            Err(format!("Compaction failed: {}", e).into())
        }
    }
}

pub fn run_cli<I, S>(args: I) -> Result<(), Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut argv = vec![OsString::from("tachometer-scraper")];
    argv.extend(args.into_iter().map(Into::into));
    let args = Args::try_parse_from(argv)?;
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(run(args))
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    // Handle subcommands
    if let Some(command) = args.command {
        return match command {
            Commands::Compact {
                input_dir,
                output,
                config,
            } => run_compact(input_dir, output, config).await,
        };
    }

    // Default behavior: scrape mode
    // Validate mutual exclusivity
    if args.config.is_some() && !args.endpoints.is_empty() {
        return Err("Cannot specify both --config and --endpoint. Use one or the other.".into());
    }
    if args.config.is_none() && args.endpoints.is_empty() {
        return Err("Must specify either --config or --endpoint".into());
    }

    // Determine local directory for intermediate files
    let local_dir = args
        .local_dir
        .map(PathBuf::from)
        .unwrap_or_else(get_default_local_dir);

    let (endpoints, storage, remote_path, rows_per_parquet, save_interval_secs, extra_column_names) =
        if let Some(config_path) = &args.config {
            // Load config from file
            let config_content = std::fs::read_to_string(config_path)
                .map_err(|e| format!("Failed to read config file {}: {}", config_path, e))?;
            let config: Config = toml::from_str(&config_content)
                .map_err(|e| format!("Failed to parse config file: {}", e))?;

            // Collect extra column names from all endpoints
            let extra_column_names = config.collect_extra_column_names();

            // Build endpoint configs with filters
            let mut endpoint_configs = Vec::new();
            for cfg_endpoint in &config.endpoints {
                let gpu_metadata = cfg_endpoint.gpu_metadata.clone();
                let node_metadata = cfg_endpoint.node_metadata.clone();
                let filter = cfg_endpoint
                    .filter
                    .as_ref()
                    .map(|f| get_filter(f, gpu_metadata, node_metadata));

                endpoint_configs.push(EndpointConfig {
                    name: cfg_endpoint.name.clone(),
                    url: cfg_endpoint.url.clone(),
                    frequency: {
                        let frequency = cfg_endpoint.frequency.unwrap_or(2.0);
                        validate_frequency(frequency)?;
                        frequency
                    },
                    filter,
                });
            }

            let storage_str = config.storage.clone();
            let (store_obj, remote_path_str) = parse_storage(&storage_str).await?;
            let store: Arc<dyn ObjectStore> = Arc::new(store_obj);

            (
                endpoint_configs,
                store,
                remote_path_str,
                config.rows_per_parquet.unwrap_or(1000000),
                config.save_interval_secs.unwrap_or(5),
                extra_column_names,
            )
        } else {
            // Use CLI args (legacy mode)
            let storage_str = args
                .storage
                .as_ref()
                .ok_or("--storage is required when using --endpoint")?
                .clone();

            // Parse endpoints with optional filters
            let mut endpoint_configs = Vec::new();
            validate_frequency(args.frequency)?;
            for (i, endpoint_str) in args.endpoints.iter().enumerate() {
                let (name, url) = parse_endpoint(endpoint_str)?;
                let filter_name = args.filters.get(i);
                let filter = filter_name.map(|f| get_filter(f, None, None));
                endpoint_configs.push(EndpointConfig {
                    name,
                    url,
                    frequency: args.frequency,
                    filter,
                });
            }

            let (store_obj, remote_path_str) = parse_storage(&storage_str).await?;
            let store: Arc<dyn ObjectStore> = Arc::new(store_obj);

            (
                endpoint_configs,
                store,
                remote_path_str,
                args.rows_per_parquet,
                args.save_interval_secs,
                vec![],
            )
        };

    let sync_interval_secs = args.sync_interval_secs;

    info!("Using remote storage path: {}", remote_path);
    info!(
        "Using local directory for intermediate files: {}",
        local_dir.display()
    );
    info!(
        "Rows per parquet: {}, Save interval: {}s",
        rows_per_parquet, save_interval_secs
    );
    if sync_interval_secs > 0 {
        info!("Periodic sync to remote: every {}s", sync_interval_secs);
    }
    if !extra_column_names.is_empty() {
        info!("Extra columns: {:?}", extra_column_names);
    }

    // Create dataset writer with local directory for intermediate files
    let writer = Arc::new(
        DatasetWriter::new(
            local_dir.clone(),
            rows_per_parquet,
            save_interval_secs,
            extra_column_names,
        )
        .map_err(|e| format!("Failed to create dataset writer: {}", e))?,
    );

    info!("Starting scraper with {} endpoint(s)", endpoints.len());
    info!("Press Ctrl+C or send SIGTERM to gracefully shutdown and compact data...");

    // Use tokio's cross-platform signal handler
    // Create a task per endpoint with its own frequency
    let mut tasks = Vec::new();
    for endpoint in endpoints {
        let writer_clone = writer.clone();
        let endpoint_name = endpoint.name.clone();
        let endpoint_url = endpoint.url.clone();
        let endpoint_filter = endpoint.filter;
        let sleep_duration = std::time::Duration::from_secs_f64(1.0 / endpoint.frequency);

        let task = tokio::spawn(async move {
            loop {
                let filter_ref = endpoint_filter.as_deref();
                match scrape_endpoint(&endpoint_url, &endpoint_name, filter_ref).await {
                    Ok(rows) => {
                        if !rows.is_empty() {
                            if let Err(e) = writer_clone.append_rows(rows).await {
                                error!("Error appending rows for {}: {}", endpoint_name, e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Error scraping {}: {}", endpoint_name, e);
                    }
                }
                tokio::time::sleep(sleep_duration).await;
            }
        });
        tasks.push(task);
    }

    // Spawn periodic sync task if enabled
    if sync_interval_secs > 0 {
        let sync_local_dir = local_dir.clone();
        let sync_storage = storage.clone();
        let sync_remote_path = remote_path.clone();
        let sync_interval = std::time::Duration::from_secs(sync_interval_secs);

        let sync_task = tokio::spawn(async move {
            // Wait for initial interval before first sync
            tokio::time::sleep(sync_interval).await;

            loop {
                info!("Starting periodic sync to remote storage...");
                match periodic_compact_and_sync(
                    &sync_local_dir,
                    sync_storage.clone(),
                    &sync_remote_path,
                )
                .await
                {
                    Ok(num_files) => {
                        if num_files > 0 {
                            info!("Periodic sync completed: compacted {} files", num_files);
                        }
                    }
                    Err(e) => {
                        error!("Periodic sync failed: {}", e);
                    }
                }
                tokio::time::sleep(sync_interval).await;
            }
        });
        tasks.push(sync_task);
    }

    // Set up signal handlers for graceful shutdown
    #[cfg(unix)]
    {
        let mut sigterm = unix_signal(SignalKind::terminate())
            .map_err(|e| format!("Failed to register SIGTERM handler: {}", e))?;

        // Wait for all tasks without consuming their handles so they can be
        // aborted before writer shutdown.
        let tasks_future = async {
            futures::future::join_all(tasks.iter_mut()).await;
        };

        tokio::select! {
            _ = tasks_future => {
                // All tasks completed (shouldn't happen normally)
            }
            _ = signal::ctrl_c() => {
                info!("\nReceived SIGINT (Ctrl+C), shutting down gracefully...");
            }
            _ = sigterm.recv() => {
                info!("\nReceived SIGTERM, shutting down gracefully...");
            }
        }
    }

    #[cfg(not(unix))]
    {
        // Wait for all tasks without consuming their handles so they can be
        // aborted before writer shutdown.
        let tasks_future = async {
            futures::future::join_all(tasks.iter_mut()).await;
        };

        tokio::select! {
            _ = tasks_future => {
                // All tasks completed (shouldn't happen normally)
            }
            _ = signal::ctrl_c() => {
                info!("\nReceived SIGINT (Ctrl+C), shutting down gracefully...");
            }
        }
    }

    for task in &tasks {
        task.abort();
    }
    for task in tasks {
        let _ = task.await;
    }

    // Graceful shutdown: flush remaining data to local disk
    info!("Shutting down writer...");
    let local_dir = match writer.shutdown().await {
        Ok(dir) => dir,
        Err(e) => {
            warn!("Failed to shutdown writer gracefully: {}", e);
            local_dir
        }
    };

    // Compact local files and upload to remote storage
    info!("Compacting dataset from local disk...");
    if let Err(e) = compact_and_upload(&local_dir, storage.clone(), &remote_path).await {
        error!("Error during compaction: {}", e);
        // Even if compaction fails, the intermediate files are still on local disk
        info!("Intermediate files preserved at: {}", local_dir.display());
    }

    info!("Shutdown complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_frequency;

    #[test]
    fn validate_frequency_rejects_non_positive_or_non_finite_values() {
        assert!(validate_frequency(1.0).is_ok());
        assert!(validate_frequency(0.0).is_err());
        assert!(validate_frequency(-1.0).is_err());
        assert!(validate_frequency(f64::NAN).is_err());
        assert!(validate_frequency(f64::INFINITY).is_err());
    }
}
