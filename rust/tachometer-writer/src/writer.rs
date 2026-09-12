use crate::Result;
use arrow::array::*;
use arrow::datatypes::*;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use log::{error, info};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Row {
    pub scraper_endpoint: String,
    pub metric_name: String,
    pub metric_value: f32,
    pub histogram_bucket_lower: Option<f32>,
    pub histogram_bucket_upper: Option<f32>,
    pub histogram_sum: Option<f32>,
    pub histogram_count: Option<f32>,
    pub extras: Vec<(String, String)>, // Sorted metadata key-value pairs
}

pub struct DatasetWriter {
    local_dir: PathBuf,
    buffer: Arc<Mutex<RecordBatchBuffer>>,
    row_count: Arc<Mutex<usize>>,
    parquet_index: Arc<Mutex<usize>>,
    rows_per_parquet: usize,
    save_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    start_time: Instant,
}

struct RecordBatchBuffer {
    scraper_endpoints: Vec<String>,
    metric_names: Vec<String>,
    metric_values: Vec<f32>,
    histogram_bucket_lowers: Vec<Option<f32>>,
    histogram_bucket_uppers: Vec<Option<f32>>,
    histogram_sums: Vec<Option<f32>>,
    histogram_counts: Vec<Option<f32>>,
    time_since_starts: Vec<f64>,
    extras_columns: Vec<Vec<String>>, // One Vec<String> per extra column
    extra_column_names: Vec<String>,  // Names of extra columns (sorted)
}

impl RecordBatchBuffer {
    fn new(extra_column_names: Vec<String>) -> Self {
        let extras_columns = extra_column_names.iter().map(|_| Vec::new()).collect();
        Self {
            scraper_endpoints: Vec::new(),
            metric_names: Vec::new(),
            metric_values: Vec::new(),
            histogram_bucket_lowers: Vec::new(),
            histogram_bucket_uppers: Vec::new(),
            histogram_sums: Vec::new(),
            histogram_counts: Vec::new(),
            time_since_starts: Vec::new(),
            extras_columns,
            extra_column_names,
        }
    }

    fn len(&self) -> usize {
        self.scraper_endpoints.len()
    }

    fn append_row(&mut self, row: &Row, time_since_start: f64) {
        self.scraper_endpoints.push(row.scraper_endpoint.clone());
        self.metric_names.push(row.metric_name.clone());
        self.metric_values.push(row.metric_value);
        self.histogram_bucket_lowers
            .push(row.histogram_bucket_lower);
        self.histogram_bucket_uppers
            .push(row.histogram_bucket_upper);
        self.histogram_sums.push(row.histogram_sum);
        self.histogram_counts.push(row.histogram_count);
        self.time_since_starts.push(time_since_start);

        // Populate extras columns - ensure all columns have values (empty string if missing)
        for (col_idx, col_name) in self.extra_column_names.iter().enumerate() {
            let value = row
                .extras
                .iter()
                .find(|(k, _)| k == col_name)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(String::new);
            self.extras_columns[col_idx].push(value);
        }
    }

    fn clear(&mut self) {
        self.scraper_endpoints.clear();
        self.metric_names.clear();
        self.metric_values.clear();
        self.histogram_bucket_lowers.clear();
        self.histogram_bucket_uppers.clear();
        self.histogram_sums.clear();
        self.histogram_counts.clear();
        self.time_since_starts.clear();
        for col in &mut self.extras_columns {
            col.clear();
        }
    }

    fn to_record_batch(&self) -> Result<RecordBatch> {
        if self.len() == 0 {
            return Err(crate::NoMoreError::InvalidSchema(
                "Cannot create empty record batch".to_string(),
            ));
        }

        let mut fields = vec![
            Field::new("scraper_endpoint", DataType::Utf8, false),
            Field::new("metric_name", DataType::Utf8, false),
            Field::new("metric_value", DataType::Float32, false),
            Field::new("histogram_bucket_lower", DataType::Float32, true),
            Field::new("histogram_bucket_upper", DataType::Float32, true),
            Field::new("histogram_sum", DataType::Float32, true),
            Field::new("histogram_count", DataType::Float32, true),
            Field::new("time_since_start", DataType::Float64, false),
        ];

        // Add extra columns to schema
        for col_name in &self.extra_column_names {
            fields.push(Field::new(col_name, DataType::Utf8, false));
        }

        let schema = Arc::new(Schema::new(fields));

        let mut arrays: Vec<Arc<dyn Array>> = vec![
            Arc::new(StringArray::from(self.scraper_endpoints.clone())),
            Arc::new(StringArray::from(self.metric_names.clone())),
            Arc::new(Float32Array::from(self.metric_values.clone())),
            Arc::new(Float32Array::from(self.histogram_bucket_lowers.clone())),
            Arc::new(Float32Array::from(self.histogram_bucket_uppers.clone())),
            Arc::new(Float32Array::from(self.histogram_sums.clone())),
            Arc::new(Float32Array::from(self.histogram_counts.clone())),
            Arc::new(Float64Array::from(self.time_since_starts.clone())),
        ];

        // Add extra column arrays
        for col in &self.extras_columns {
            arrays.push(Arc::new(StringArray::from(col.clone())));
        }

        RecordBatch::try_new(schema, arrays).map_err(|e| crate::NoMoreError::Arrow(e.to_string()))
    }
}

impl DatasetWriter {
    /// Create a new DatasetWriter that writes intermediate files to a local directory.
    ///
    /// # Arguments
    /// * `local_dir` - Local directory for intermediate files (current.arrow, out-N.parquet)
    /// * `rows_per_parquet` - Number of rows before creating a numbered parquet file
    /// * `save_interval_secs` - Interval in seconds for periodic Arrow file saves
    /// * `extra_column_names` - Names of extra columns to include in the schema
    pub fn new(
        local_dir: PathBuf,
        rows_per_parquet: usize,
        save_interval_secs: u64,
        extra_column_names: Vec<String>,
    ) -> Result<Self> {
        if save_interval_secs == 0 {
            return Err(crate::NoMoreError::InvalidConfiguration(
                "save_interval_secs must be greater than zero".to_string(),
            ));
        }

        // Create the local directory if it doesn't exist
        std::fs::create_dir_all(&local_dir).map_err(|e| {
            crate::NoMoreError::Io(std::io::Error::other(format!(
                "Failed to create local directory {}: {}",
                local_dir.display(),
                e
            )))
        })?;

        let buffer = Arc::new(Mutex::new(RecordBatchBuffer::new(extra_column_names)));
        let row_count = Arc::new(Mutex::new(0));
        let parquet_index = Arc::new(Mutex::new(1));
        let start_time = Instant::now();

        let buffer_clone = buffer.clone();
        let local_dir_clone = local_dir.clone();

        let save_handle = tokio::spawn(async move {
            periodic_save_task(buffer_clone, local_dir_clone, save_interval_secs).await;
        });

        Ok(Self {
            local_dir,
            buffer,
            row_count,
            parquet_index,
            rows_per_parquet,
            save_handle: Mutex::new(Some(save_handle)),
            start_time,
        })
    }

    /// Get the local directory where intermediate files are stored.
    pub fn local_dir(&self) -> &Path {
        &self.local_dir
    }

    pub async fn append_row(&self, row: Row) -> Result<()> {
        let time_since_start = self.start_time.elapsed().as_secs_f64();
        let mut buffer = self.buffer.lock().await;
        let mut row_count = self.row_count.lock().await;

        buffer.append_row(&row, time_since_start);
        *row_count += 1;

        // Check if we need to create a numbered parquet file
        if *row_count >= self.rows_per_parquet {
            let batch = buffer.to_record_batch()?;
            let mut parquet_idx = self.parquet_index.lock().await;
            let filename = format!("out-{}.parquet", *parquet_idx);
            *parquet_idx += 1;
            drop(parquet_idx);

            let num_rows = batch.num_rows();
            write_parquet_file_local(&self.local_dir, &filename, &batch)?;
            info!("Saved parquet file {} with {} rows", filename, num_rows);
            buffer.clear();
            *row_count = 0;
        }

        Ok(())
    }

    pub async fn append_rows(&self, rows: Vec<Row>) -> Result<()> {
        let time_since_start = self.start_time.elapsed().as_secs_f64();
        let mut buffer = self.buffer.lock().await;
        let mut row_count = self.row_count.lock().await;

        for row in rows {
            buffer.append_row(&row, time_since_start);
            *row_count += 1;
        }

        // Check if we need to create a numbered parquet file
        if *row_count >= self.rows_per_parquet {
            let batch = buffer.to_record_batch()?;
            let mut parquet_idx = self.parquet_index.lock().await;
            let filename = format!("out-{}.parquet", *parquet_idx);
            *parquet_idx += 1;
            drop(parquet_idx);

            let num_rows = batch.num_rows();
            write_parquet_file_local(&self.local_dir, &filename, &batch)?;
            info!("Saved parquet file {} with {} rows", filename, num_rows);
            buffer.clear();
            *row_count = 0;
        }

        Ok(())
    }

    /// Shutdown the writer, flushing any remaining data to current.arrow.
    /// Returns the local directory path for subsequent compaction.
    pub async fn shutdown(&self) -> Result<PathBuf> {
        // Stop the periodic save task
        let save_handle = {
            let mut save_handle = self.save_handle.lock().await;
            save_handle.take()
        };
        if let Some(save_handle) = save_handle {
            save_handle.abort();
            let _ = save_handle.await;
        }

        // Flush any remaining data to current.arrow
        let buffer = self.buffer.lock().await;
        if buffer.len() > 0 {
            let batch = buffer.to_record_batch()?;
            let num_rows = batch.num_rows();
            write_arrow_file_local(&self.local_dir, "current.arrow", &batch)?;
            info!("Flushed current.arrow with {} rows on shutdown", num_rows);
        }

        Ok(self.local_dir.clone())
    }
}

async fn periodic_save_task(
    buffer: Arc<Mutex<RecordBatchBuffer>>,
    local_dir: PathBuf,
    save_interval_secs: u64,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(save_interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        interval.tick().await;

        let batch = {
            let buffer_guard = buffer.lock().await;
            if buffer_guard.len() == 0 {
                continue;
            }
            match buffer_guard.to_record_batch() {
                Ok(batch) => batch,
                Err(e) => {
                    error!("Error creating record batch for periodic save: {}", e);
                    continue;
                }
            }
        };

        // Save to current.arrow on local disk
        if let Err(e) = write_arrow_file_local(&local_dir, "current.arrow", &batch) {
            error!("Error saving current.arrow: {}", e);
        } else {
            info!("Saved current.arrow with {} rows", batch.num_rows());
        }
    }
}

/// Write an Arrow IPC stream file to local disk.
fn write_arrow_file_local(local_dir: &Path, filename: &str, batch: &RecordBatch) -> Result<()> {
    let path = local_dir.join(filename);
    let file = File::create(&path).map_err(|e| {
        crate::NoMoreError::Io(std::io::Error::other(format!(
            "Failed to create file {}: {}",
            path.display(),
            e
        )))
    })?;
    let mut writer = BufWriter::new(file);

    let mut stream_writer = StreamWriter::try_new(&mut writer, batch.schema().as_ref())
        .map_err(|e| crate::NoMoreError::Arrow(e.to_string()))?;
    stream_writer
        .write(batch)
        .map_err(|e| crate::NoMoreError::Arrow(e.to_string()))?;
    stream_writer
        .finish()
        .map_err(|e| crate::NoMoreError::Arrow(e.to_string()))?;

    Ok(())
}

/// Write a Parquet file to local disk.
fn write_parquet_file_local(local_dir: &Path, filename: &str, batch: &RecordBatch) -> Result<()> {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_write_batch_size(100_000)
        .build();

    let path = local_dir.join(filename);
    let temp_path = local_dir.join(format!(".{}.tmp", filename));
    let file = File::create(&temp_path).map_err(|e| {
        crate::NoMoreError::Io(std::io::Error::other(format!(
            "Failed to create file {}: {}",
            temp_path.display(),
            e
        )))
    })?;

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| crate::NoMoreError::Parquet(e.to_string()))?;
    writer
        .write(batch)
        .map_err(|e| crate::NoMoreError::Parquet(e.to_string()))?;
    writer
        .close()
        .map_err(|e| crate::NoMoreError::Parquet(e.to_string()))?;

    std::fs::rename(&temp_path, &path).map_err(|e| {
        crate::NoMoreError::Io(std::io::Error::other(format!(
            "Failed to publish parquet file {}: {}",
            path.display(),
            e
        )))
    })?;

    Ok(())
}
