// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Weights & Biases offline `.wandb` transaction-log sink.
//!
//! Writes `wandb/offline-run-<ts>-<id>/` with the W&B 0.28.0 length-prefixed
//! protobuf datastore framing consumed by `wandb sync`.
//!
//! # What is emitted (accepted by the wandb datastore decoder)
//! - `HeaderRecord` (datastore version stamps).
//! - `RunRecord` — run id, project, entity, display name, tags, start time, and
//!   the full config delta (`ConfigRecord`).
//! - `HistoryRecord` + `SummaryRecord` — the summary-metrics table rows as
//!   `summary_metrics/<label>/<stat>` scalar items (one per finite stat), plus
//!   `_step` / `_runtime`.
//! - `FilesRecord` — the `wandb.Table`-format `media/table/summary_metrics.table.json`
//!   file, preserving table columns and rows.
//! - `RunExitRecord`.
//!
//! # Artifact contract
//! - Tags: `aiperf-<version>` (from [`NativeReport::aiperf_version`]),
//!   `benchmark-<id8>` (from the benchmark id), then user tags.
//! - Table columns: `["Metric", *DEFAULT_STAT_KEYS]` where
//!   `DEFAULT_STAT_KEYS = ("avg","min","max","p99","p90","p50","std")`; one row
//!   per metric, each finite stat rounded to two decimals and non-finite stats null.
//! - Config: the full redacted config plus `aiperf.cli_command`, split into
//!   per-key `ConfigItem`s.
//!
//! # Omitted data
//! - The versioned `wandb.Artifact` bundle (manifest, per-file digests,
//!   client-artifact references, `run_table` artifact record) is **not**
//!   reproduced; that machinery requires content-addressed staging outside this
//!   sink's scope. The table data is instead preserved verbatim as the
//!   `media/table/summary_metrics.table.json` run file and as history/summary
//!   scalars. Artifact-glob bundle upload from `artifact_dir` is therefore
//!   **deferred**.
//! - The metric row **label** is the report's stable metric name plus unit
//!   (`"<name> (<unit>)"`); the console `short_header` / `display_order` /
//!   visibility filtering are not carried in [`NativeReport`], so ordering is by
//!   stable name and visibility is whatever the report already includes.
//!
//! # Configuration contract
//! Because [`NativeReport`] carries neither the full config blob, redacted CLI
//! command, nor benchmark id, callers supply:
//! `project`, `entity`, `run_name`, `tags` (user tags only), `benchmark_id`,
//! `config_json` (the serialized redacted config object), `cli_command`
//! (already redacted). See [`WandbExportConfig`].

pub mod datastore;
pub mod proto;

#[cfg(test)]
mod tests;

use std::fs;
use std::path::Path;

use anyhow::Context;
use chrono::{Datelike, Timelike, Utc};

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::{MetricEntry, NativeReport, ReportStats, ReportValue};

/// Summary table stat columns.
const STAT_COLUMN_KEYS: [&str; 7] = ["avg", "min", "max", "p99", "p90", "p50", "std"];

/// Producer version stamp written into the datastore header.
const HEADER_PRODUCER: &str = "aiperf-native-wandb";
/// Minimum consumer able to read the streams we emit.
const HEADER_MIN_CONSUMER: &str = "0.65.0";

/// W&B export policy plus values unavailable from the report. A project enables
/// the sink.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WandbExportConfig {
    /// W&B project (enables the sink when present).
    pub project: Option<String>,
    /// Optional entity/team.
    pub entity: Option<String>,
    /// Optional run name; defaults to `aiperf-<benchmark_id[:8]>`.
    pub run_name: Option<String>,
    /// User tags (the `aiperf-<version>`/`benchmark-<id8>` tags are derived).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Benchmark id used for the default run name and `benchmark-<id8>` tag.
    pub benchmark_id: Option<String>,
    /// AIPerf package version for the `aiperf-<version>` tag; takes precedence
    /// over the report version.
    pub aiperf_version: Option<String>,
    /// Full redacted config object (`cfg.model_dump(mode="json",
    /// exclude_none=True)`), serialized as a JSON object string. Split into
    /// per-key `ConfigItem`s.
    pub config_json: Option<String>,
    /// Redacted CLI command (`redact_cli_command(run.cli_command)`), recorded
    /// under the config key `aiperf.cli_command`.
    pub cli_command: Option<String>,
}

/// The W&B [`Exporter`] (offline `.wandb` transaction log).
pub struct WandbExporter;

impl Exporter for WandbExporter {
    fn name(&self) -> &'static str {
        "wandb"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.wandb.project.is_some()
    }

    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let wandb = &cfg.wandb;
        let project = wandb
            .project
            .as_deref()
            .context("W&B export enabled without a project")?;

        let run_id = generate_run_id();
        let (dir_stamp, start_time) = utc_stamp();
        let run_name = wandb
            .run_name
            .clone()
            .unwrap_or_else(|| default_run_name(wandb.benchmark_id.as_deref()));

        let run_dir = artifact_dir
            .join("wandb")
            .join(format!("offline-run-{dir_stamp}-{run_id}"));
        let files_dir = run_dir.join("files");
        let table_rel = "media/table/summary_metrics.table.json";
        fs::create_dir_all(files_dir.join("media").join("table"))
            .with_context(|| format!("creating W&B run dir {}", run_dir.display()))?;

        // Preserve the W&B table JSON shape on disk.
        let rows = build_metric_rows(report);
        let table_json = table_file_json(&rows)?;
        fs::write(files_dir.join(table_rel), table_json)
            .context("writing summary_metrics table file")?;

        let tags = build_tags(report, wandb);
        let config_items = build_config_items(wandb)?;

        let mut store = datastore::DataStore::new();
        let mut num: i64 = 0;

        // 1. header
        num += 1;
        store.write(
            &proto::Record {
                num,
                header: Some(proto::HeaderRecord {
                    version_info: Some(proto::VersionInfo {
                        producer: HEADER_PRODUCER.to_string(),
                        min_consumer: HEADER_MIN_CONSUMER.to_string(),
                    }),
                    info: Some(record_info(&run_id)),
                }),
                ..Default::default()
            }
            .to_bytes(),
        );

        // 2. run
        num += 1;
        store.write(
            &proto::Record {
                num,
                run: Some(proto::RunRecord {
                    run_id: run_id.clone(),
                    entity: wandb.entity.clone().unwrap_or_default(),
                    project: project.to_string(),
                    config: Some(proto::ConfigRecord {
                        update: config_items,
                        info: None,
                    }),
                    display_name: run_name.clone(),
                    tags,
                    host: hostname(),
                    start_time: Some(start_time),
                    info: Some(record_info(&run_id)),
                }),
                ..Default::default()
            }
            .to_bytes(),
        );

        // 3. files (the media table file)
        num += 1;
        store.write(
            &proto::Record {
                num,
                files: Some(proto::FilesRecord {
                    files: vec![proto::FilesItem {
                        path: table_rel.to_string(),
                        policy: 1, // END
                        r#type: 2, // MEDIA
                    }],
                    info: Some(record_info(&run_id)),
                }),
                ..Default::default()
            }
            .to_bytes(),
        );

        let history_items = build_history_items(&rows);
        num += 1;
        store.write(
            &proto::Record {
                num,
                history: Some(proto::HistoryRecord {
                    item: history_items.clone(),
                    step: Some(proto::HistoryStep { num: 0 }),
                }),
                ..Default::default()
            }
            .to_bytes(),
        );

        num += 1;
        store.write(
            &proto::Record {
                num,
                summary: Some(proto::SummaryRecord {
                    update: history_items
                        .into_iter()
                        .map(|item| proto::SummaryItem {
                            key: item.key,
                            nested_key: item.nested_key,
                            value_json: item.value_json,
                        })
                        .collect(),
                }),
                ..Default::default()
            }
            .to_bytes(),
        );

        // 6. exit
        num += 1;
        store.write(
            &proto::Record {
                num,
                exit: Some(proto::RunExitRecord {
                    exit_code: 0,
                    runtime: 0,
                    info: Some(record_info(&run_id)),
                }),
                control: Some(proto::Control { always_send: true }),
                ..Default::default()
            }
            .to_bytes(),
        );

        let wandb_path = run_dir.join(format!("run-{run_id}.wandb"));
        fs::write(&wandb_path, store.into_bytes())
            .with_context(|| format!("writing {}", wandb_path.display()))?;
        Ok(())
    }
}

/// One rendered metric row: the display label and the 7 stat cells.
struct MetricRow {
    label: String,
    cells: [Option<f64>; STAT_COLUMN_KEYS.len()],
}

/// Build one row per profiling metric.
fn build_metric_rows(report: &NativeReport) -> Vec<MetricRow> {
    report
        .metrics
        .iter()
        .filter_map(|(name, entry)| {
            let stats = entry.series.first().map(|s| &s.stats)?;
            let mut cells = [None; STAT_COLUMN_KEYS.len()];
            for (i, key) in STAT_COLUMN_KEYS.iter().enumerate() {
                cells[i] = stat_value(stats, key).map(round2);
            }
            Some(MetricRow {
                label: metric_label(name, entry),
                cells,
            })
        })
        .collect()
}

/// `"<name> (<unit>)"`, or just `<name>` when the unit is empty.
fn metric_label(name: &str, entry: &MetricEntry) -> String {
    if entry.unit.is_empty() {
        name.to_string()
    } else {
        format!("{name} ({})", entry.unit)
    }
}

/// Extract one stat key from the type-specific report statistics.
fn stat_value(stats: &ReportStats, key: &str) -> Option<f64> {
    let val = |v: &ReportValue| match v {
        ReportValue::Finite(f) => Some(*f),
        ReportValue::NonFinite => None,
    };
    match stats {
        ReportStats::Distribution(d) => match key {
            "avg" => d.avg.as_ref().and_then(val),
            "min" => d.min.as_ref().and_then(val),
            "max" => d.max.as_ref().and_then(val),
            "std" => d.std.as_ref().and_then(val),
            "p50" | "p90" | "p99" => d.percentiles.get(key).and_then(val),
            _ => None,
        },
        ReportStats::Scalar(s) => match key {
            "avg" | "min" | "max" | "p50" | "p90" | "p99" => val(&s.value),
            _ => None,
        },
        ReportStats::Counter(c) => match key {
            "avg" | "min" | "max" | "p50" | "p90" | "p99" => val(&c.total),
            _ => None,
        },
        ReportStats::Histogram(h) => match key {
            "avg" => h.avg.as_ref().and_then(val),
            "p50" | "p90" | "p99" => h.percentiles.get(key).and_then(val),
            _ => None,
        },
    }
}

/// The `wandb.Table` on-disk JSON: `{"columns":[...],"data":[[...],...]}`.
fn table_file_json(rows: &[MetricRow]) -> anyhow::Result<String> {
    let mut columns: Vec<serde_json::Value> = vec![serde_json::json!("Metric")];
    columns.extend(STAT_COLUMN_KEYS.iter().map(|k| serde_json::json!(k)));

    let data: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            let mut cells = vec![serde_json::json!(row.label)];
            for cell in row.cells {
                cells.push(cell_json(cell));
            }
            serde_json::Value::Array(cells)
        })
        .collect();

    serde_json::to_string(&serde_json::json!({ "columns": columns, "data": data }))
        .context("serializing summary_metrics table")
}

/// Build the flat `summary_metrics/<label>/<stat>` history items (finite only).
fn build_history_items(rows: &[MetricRow]) -> Vec<proto::HistoryItem> {
    let mut items = Vec::new();
    for row in rows {
        for (i, key) in STAT_COLUMN_KEYS.iter().enumerate() {
            if let Some(v) = row.cells[i] {
                items.push(proto::HistoryItem {
                    key: format!("summary_metrics/{}/{}", row.label, key),
                    nested_key: Vec::new(),
                    value_json: json_number(v),
                });
            }
        }
    }
    items.push(proto::HistoryItem {
        key: "_step".to_string(),
        nested_key: Vec::new(),
        value_json: "0".to_string(),
    });
    items.push(proto::HistoryItem {
        key: "_runtime".to_string(),
        nested_key: Vec::new(),
        value_json: "0".to_string(),
    });
    items
}

/// Tags: `aiperf-<version>`, `benchmark-<id8>`, then user tags (`_build_tags`).
fn build_tags(report: &NativeReport, cfg: &WandbExportConfig) -> Vec<String> {
    let aiperf_version = cfg
        .aiperf_version
        .clone()
        .unwrap_or_else(|| report.aiperf_version.clone());
    let mut tags = vec![format!("aiperf-{aiperf_version}")];
    if let Some(id) = &cfg.benchmark_id {
        let id8: String = id.chars().take(8).collect();
        if !id8.is_empty() {
            tags.push(format!("benchmark-{id8}"));
        }
    }
    tags.extend(cfg.tags.iter().cloned());
    tags
}

/// Split the projected config JSON object into `ConfigItem`s, prepending the
/// `_wandb` bookkeeping key and appending `aiperf.cli_command`.
fn build_config_items(cfg: &WandbExportConfig) -> anyhow::Result<Vec<proto::ConfigItem>> {
    let mut items = vec![proto::ConfigItem {
        key: "_wandb".to_string(),
        nested_key: Vec::new(),
        value_json: "{}".to_string(),
    }];

    if let Some(raw) = &cfg.config_json {
        let value: serde_json::Value =
            serde_json::from_str(raw).context("parsing projected wandb config_json")?;
        if let serde_json::Value::Object(map) = value {
            for (key, val) in map {
                items.push(proto::ConfigItem {
                    key,
                    nested_key: Vec::new(),
                    value_json: serde_json::to_string(&val)
                        .context("re-serializing config value")?,
                });
            }
        } else {
            anyhow::bail!("wandb config_json must be a JSON object");
        }
    }

    if let Some(cli) = &cfg.cli_command {
        items.push(proto::ConfigItem {
            key: "aiperf.cli_command".to_string(),
            nested_key: Vec::new(),
            value_json: serde_json::to_string(cli).context("serializing cli_command")?,
        });
    }
    Ok(items)
}

/// JSON for one rounded table cell: a number, or `null` when absent/non-finite.
fn cell_json(value: Option<f64>) -> serde_json::Value {
    match value {
        Some(v) => serde_json::Number::from_f64(v)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        None => serde_json::Value::Null,
    }
}

/// JSON encoding of a finite number for a `value_json` field.
fn json_number(value: f64) -> String {
    serde_json::Number::from_f64(value)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "null".to_string())
}

/// Round finite values to two decimals.
fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Default run name `aiperf-<benchmark_id[:8]>`, or `aiperf-run` when no id.
fn default_run_name(benchmark_id: Option<&str>) -> String {
    match benchmark_id {
        Some(id) if !id.is_empty() => {
            let id8: String = id.chars().take(8).collect();
            format!("aiperf-{id8}")
        }
        _ => "aiperf-run".to_string(),
    }
}

/// Per-record routing info stamped with the run/stream id.
fn record_info(run_id: &str) -> proto::RecordInfo {
    proto::RecordInfo {
        stream_id: run_id.to_string(),
    }
}

/// Best-effort host name for the run record.
fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "localhost".to_string())
}

/// An 8-character lowercase alphanumeric run id, matching the wandb id shape.
fn generate_run_id() -> String {
    use rand::Rng;
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    (0..8)
        .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())] as char)
        .collect()
}

/// The `YYYYMMDD_HHMMSS` directory stamp and the matching start-time proto.
fn utc_stamp() -> (String, proto::Timestamp) {
    let now = Utc::now();
    let stamp = format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
    );
    let ts = proto::Timestamp {
        seconds: now.timestamp(),
        nanos: now.timestamp_subsec_nanos() as i32,
    };
    (stamp, ts)
}
