// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dataset loader registry and `load -> compose -> store` orchestration.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde_json::{Map, Value};

use crate::dataset::compose::{ComposeConfig, Composer, apply_common_contexts};
use crate::dataset::dataset::Dataset;
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::fetch::{DatasetFetcher, HttpDatasetFetcher};
use crate::dataset::model::{ConversationContextMode, SessionId};
use crate::dataset::segment::SegmentPool;
use crate::dataset::tokenizer::TextTokenizer;

pub mod asr;
pub mod exgentic;
pub mod public;
pub mod random_pool;
pub mod raw_payload;
pub mod simple;
pub mod synthetic;
pub mod trace;

pub use asr::{HfAsrComposer, HfAsrDatasetLoader};
pub use exgentic::{ExgenticComposer, ExgenticDatasetLoader, ExgenticV2DatasetLoader};
pub use public::{
    AccuracyComposer, AccuracyDatasetLoader, HfConversationComposer, HfConversationDatasetLoader,
    HfInstructionComposer, HfInstructionDatasetLoader, MmvuComposer, MmvuDatasetLoader,
    MtBenchComposer, MtBenchDatasetLoader, ShareGptComposer, ShareGptDatasetLoader,
    SpecBenchComposer, SpecBenchDatasetLoader, SpeedBenchComposer, SpeedBenchDatasetLoader,
    load_raw_rows,
};
pub use random_pool::{RandomPoolComposer, RandomPoolDatasetLoader};
pub use raw_payload::{InputsJsonPayloadLoader, RawPayloadComposer, RawPayloadDatasetLoader};
pub use simple::{
    MultiTurnComposer, MultiTurnDatasetLoader, SingleTurnComposer, SingleTurnDatasetLoader,
};
pub use synthetic::{
    SyntheticComposer, SyntheticDatasetLoader, SyntheticRankingsComposer,
    SyntheticRankingsDatasetLoader,
};
pub use trace::{
    BailianTraceComposer, BailianTraceDatasetLoader, BurstGptTraceComposer,
    BurstGptTraceDatasetLoader, MooncakeTraceComposer, MooncakeTraceDatasetLoader,
    SageMakerDataCaptureComposer, SageMakerDataCaptureDatasetLoader,
};

/// A local path, in-memory JSON value, or exact in-memory byte source.
#[derive(Debug, Clone)]
pub enum DatasetSource {
    /// File or format-defined directory.
    Path(PathBuf),
    /// Parsed inline configuration records.
    Inline(Value),
    /// Exact JSON/JSONL bytes supplied by an embedding application.
    Bytes(Bytes),
    /// Generic remote JSON, JSONL, CSV, or Parquet URL.
    Url(String),
    /// Hugging Face Dataset Viewer rows source.
    HuggingFace {
        /// Repository identifier such as `openai/gsm8k`.
        dataset: String,
        /// Dataset configuration/subset.
        config: String,
        /// Dataset split.
        split: String,
        /// Optional row cap; absent downloads the full reported split.
        max_rows: Option<usize>,
        /// Optional Git revision. A configured revision is resolved to its
        /// immutable commit and loaded from repository artifacts, never the
        /// Dataset Viewer snapshot of `main`.
        revision: Option<String>,
    },
}

impl DatasetSource {
    /// Human-readable source label for diagnostics.
    pub fn label(&self) -> String {
        match self {
            Self::Path(path) => path.display().to_string(),
            Self::Inline(_) => "inline dataset records".into(),
            Self::Bytes(_) => "in-memory dataset bytes".into(),
            Self::Url(url) => url.clone(),
            Self::HuggingFace {
                dataset,
                config,
                split,
                ..
            } => format!("Hugging Face {dataset}/{config}/{split}"),
        }
    }
}

/// Format-neutral loader settings plus an extension map for loader-specific knobs.
#[derive(Clone)]
pub struct LoadConfig {
    /// Dataset source.
    pub source: DatasetSource,
    /// Trace start timestamp, inclusive, in milliseconds.
    pub start_offset_ms: Option<f64>,
    /// Trace end timestamp, inclusive, in milliseconds.
    pub end_offset_ms: Option<f64>,
    /// Maximum retained input length.
    pub max_input_tokens: Option<u64>,
    /// Maximum output length, applied as a cap.
    pub max_output_tokens: Option<u32>,
    /// Maximum number of parsed rows retained before composition.
    pub max_rows: Option<usize>,
    /// User-selected conversation sampler, overriding the format preference.
    pub sampling_strategy: Option<String>,
    /// Format-specific validated options.
    pub options: Map<String, Value>,
    /// Injected remote fetch/cache implementation.
    pub fetcher: Arc<dyn DatasetFetcher>,
    /// Optional bearer token for gated sources. It is never included in cache keys
    /// or diagnostics.
    pub bearer_token: Option<String>,
}

impl LoadConfig {
    /// Construct default settings for one source.
    pub fn new(source: DatasetSource) -> Self {
        Self {
            source,
            start_offset_ms: None,
            end_offset_ms: None,
            max_input_tokens: None,
            max_output_tokens: None,
            max_rows: None,
            sampling_strategy: None,
            options: Map::new(),
            fetcher: Arc::new(HttpDatasetFetcher::default()),
            bearer_token: std::env::var("HF_TOKEN")
                .ok()
                .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok()),
        }
    }

    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("start_offset_ms", self.start_offset_ms),
            ("end_offset_ms", self.end_offset_ms),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(DatasetError::Validation(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if self
            .start_offset_ms
            .zip(self.end_offset_ms)
            .is_some_and(|(start, end)| start > end)
        {
            return Err(DatasetError::Validation(
                "start_offset_ms must be <= end_offset_ms".into(),
            ));
        }
        if self.max_input_tokens == Some(0) || self.max_output_tokens == Some(0) {
            return Err(DatasetError::Validation(
                "max_input_tokens and max_output_tokens must be positive when configured".into(),
            ));
        }
        if self.max_rows == Some(0) {
            return Err(DatasetError::Validation(
                "max_rows must be positive when configured".into(),
            ));
        }
        if self
            .sampling_strategy
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(DatasetError::Validation(
                "sampling_strategy cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for LoadConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadConfig")
            .field("source", &self.source)
            .field("start_offset_ms", &self.start_offset_ms)
            .field("end_offset_ms", &self.end_offset_ms)
            .field("max_input_tokens", &self.max_input_tokens)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("max_rows", &self.max_rows)
            .field("sampling_strategy", &self.sampling_strategy)
            .field("options", &self.options)
            .field("fetcher", &"dyn DatasetFetcher")
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Source coordinate retained on every parsed row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowOrigin {
    /// One-based JSONL or CSV line.
    FileLine {
        /// Source path.
        path: PathBuf,
        /// One-based line number.
        line: usize,
    },
    /// Zero-based inline record index.
    Inline {
        /// Record index.
        index: usize,
    },
    /// JSON pointer within a whole-file object.
    JsonPointer {
        /// Source path when file-backed.
        path: Option<PathBuf>,
        /// RFC 6901-style pointer.
        pointer: String,
    },
}

impl std::fmt::Display for RowOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileLine { path, line } => write!(f, "{}:{line}", path.display()),
            Self::Inline { index } => write!(f, "inline record {index}"),
            Self::JsonPointer { path, pointer } => match path {
                Some(path) => write!(f, "{}:{pointer}", path.display()),
                None => write!(f, "inline JSON:{pointer}"),
            },
        }
    }
}

/// One parsed format row, retaining exact wire bytes where replay requires them.
#[derive(Debug, Clone)]
pub struct RawRow {
    /// Decoded value used for validation and canonical field access.
    pub value: Value,
    /// Exact authored object bytes for raw replay or raw-message interning.
    pub wire: Option<Bytes>,
    /// Authored session identifier, when the format exposes one outside `value`.
    pub session_id: Option<SessionId>,
    /// Loader-private grouping key for rows that form one conversation.
    pub group_key: Option<String>,
    /// Source coordinate.
    pub origin: RowOrigin,
}

/// First-record/path probe passed to format detectors.
#[derive(Debug, Clone)]
pub struct DatasetProbe {
    /// First decoded value, or whole-file value for JSON containers.
    pub value: Option<Value>,
    /// Source path, including directories.
    pub path: Option<PathBuf>,
}

/// Pure parsing/fetching stage for one format.
#[async_trait]
pub trait DatasetLoader: Send + Sync {
    /// Stable registration name.
    fn name(&self) -> &str;

    /// Whether this loader recognizes a source probe.
    fn can_load(&self, probe: &DatasetProbe) -> bool;

    /// Parse/fetch rows without interning or composing them.
    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>>;

    /// Preferred sampler registration name.
    fn preferred_sampling_strategy(&self) -> &str {
        "shuffle"
    }

    /// Format-implied context behavior, when any.
    fn default_context_mode(&self) -> Option<ConversationContextMode> {
        None
    }
}

/// One paired loader/composer plugin registration.
#[derive(Clone)]
pub struct DatasetFormatRegistration {
    /// Stable format name.
    pub name: String,
    /// Pure parser/fetcher.
    pub loader: Arc<dyn DatasetLoader>,
    /// Format-specific canonical composer.
    pub composer: Arc<dyn Composer>,
}

impl DatasetFormatRegistration {
    /// Pair one loader with its composer, rejecting mismatched registration names.
    pub fn new(loader: Arc<dyn DatasetLoader>, composer: Arc<dyn Composer>) -> Self {
        Self {
            name: loader.name().to_string(),
            loader,
            composer,
        }
    }
}

/// Ordered registry used for explicit format lookup and structural auto-detection.
#[derive(Clone, Default)]
pub struct LoaderRegistry {
    formats: Vec<DatasetFormatRegistration>,
    by_name: HashMap<String, usize>,
}

impl LoaderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register every built-in format implemented by this crate.
    pub fn with_builtin_formats() -> Result<Self> {
        let mut registry = Self::new();
        registry.register_builtin_formats()?;
        Ok(registry)
    }

    /// Register every built-in format into an existing registry. Shared by
    /// [`Self::with_builtin_formats`] and the built-in loader `AIPerfExtension`
    /// so both compose the identical set.
    pub fn register_builtin_formats(&mut self) -> Result<()> {
        for registration in [
            DatasetFormatRegistration::new(
                Arc::new(SyntheticRankingsDatasetLoader),
                Arc::new(SyntheticRankingsComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(SyntheticDatasetLoader),
                Arc::new(SyntheticComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(ExgenticV2DatasetLoader),
                Arc::new(ExgenticComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(ExgenticDatasetLoader),
                Arc::new(ExgenticComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(AccuracyDatasetLoader),
                Arc::new(AccuracyComposer),
            ),
            DatasetFormatRegistration::new(Arc::new(HfAsrDatasetLoader), Arc::new(HfAsrComposer)),
            DatasetFormatRegistration::new(
                Arc::new(ShareGptDatasetLoader),
                Arc::new(ShareGptComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(HfInstructionDatasetLoader),
                Arc::new(HfInstructionComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(HfConversationDatasetLoader),
                Arc::new(HfConversationComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(MtBenchDatasetLoader),
                Arc::new(MtBenchComposer),
            ),
            DatasetFormatRegistration::new(Arc::new(MmvuDatasetLoader), Arc::new(MmvuComposer)),
            DatasetFormatRegistration::new(
                Arc::new(SpecBenchDatasetLoader),
                Arc::new(SpecBenchComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(SpeedBenchDatasetLoader),
                Arc::new(SpeedBenchComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(SageMakerDataCaptureDatasetLoader),
                Arc::new(SageMakerDataCaptureComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(BailianTraceDatasetLoader),
                Arc::new(BailianTraceComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(BurstGptTraceDatasetLoader),
                Arc::new(BurstGptTraceComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(MooncakeTraceDatasetLoader),
                Arc::new(MooncakeTraceComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(RandomPoolDatasetLoader),
                Arc::new(RandomPoolComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(InputsJsonPayloadLoader),
                Arc::new(RawPayloadComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(RawPayloadDatasetLoader),
                Arc::new(RawPayloadComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(MultiTurnDatasetLoader),
                Arc::new(MultiTurnComposer),
            ),
            DatasetFormatRegistration::new(
                Arc::new(SingleTurnDatasetLoader),
                Arc::new(SingleTurnComposer),
            ),
        ] {
            self.register(registration)?;
        }
        Ok(())
    }

    /// Register a format; duplicate normalized names are rejected.
    pub fn register(&mut self, registration: DatasetFormatRegistration) -> Result<()> {
        let normalized = normalize_name(&registration.name);
        if normalized.is_empty() {
            return Err(DatasetError::Validation(
                "dataset format registration name cannot be empty".into(),
            ));
        }
        if self.by_name.contains_key(&normalized) {
            return Err(DatasetError::Validation(format!(
                "duplicate dataset loader registration {:?}",
                registration.name
            )));
        }
        self.by_name.insert(normalized, self.formats.len());
        self.formats.push(registration);
        Ok(())
    }

    /// Resolve an explicitly selected format with dash/underscore-insensitive matching.
    pub fn get(&self, name: &str) -> Result<&DatasetFormatRegistration> {
        let normalized = normalize_name(name);
        self.by_name
            .get(&normalized)
            .map(|index| &self.formats[*index])
            .ok_or_else(|| DatasetError::LoaderNotFound(format!("format {name:?}")))
    }

    /// Detect exactly one compatible registration.
    pub fn detect(&self, probe: &DatasetProbe, label: &str) -> Result<&DatasetFormatRegistration> {
        let matches: Vec<_> = self
            .formats
            .iter()
            .filter(|registration| registration.loader.can_load(probe))
            .collect();
        match matches.as_slice() {
            [] => Err(DatasetError::LoaderNotFound(label.to_string())),
            [registration] => Ok(*registration),
            _ => Err(DatasetError::AmbiguousLoader(
                matches
                    .iter()
                    .map(|registration| registration.name.clone())
                    .collect(),
            )),
        }
    }

    /// Build a probe from a source without consuming it.
    pub fn probe(&self, source: &DatasetSource) -> Result<DatasetProbe> {
        match source {
            DatasetSource::Path(path) if path.is_dir() => Ok(DatasetProbe {
                value: None,
                path: Some(path.clone()),
            }),
            DatasetSource::Path(path) => probe_file(path),
            DatasetSource::Inline(value) => Ok(DatasetProbe {
                value: Some(probe_value(value)),
                path: None,
            }),
            DatasetSource::Bytes(bytes) => Ok(DatasetProbe {
                value: first_json_value(bytes)?.map(|value| probe_value(&value)),
                path: None,
            }),
            DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => Ok(DatasetProbe {
                value: None,
                path: None,
            }),
        }
    }

    /// Execute the full linear pipeline and freeze a shareable dataset.
    pub async fn build_dataset(
        &self,
        explicit_format: Option<&str>,
        load_config: &LoadConfig,
        compose_config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
    ) -> Result<Dataset> {
        load_config.validate()?;
        let registration = match explicit_format {
            Some(name) => self.get(name)?,
            None => {
                let probe = self.probe(&load_config.source)?;
                self.detect(&probe, &load_config.source.label())?
            }
        };
        let mut rows = registration.loader.load(load_config).await?;
        if let Some(max_rows) = load_config.max_rows {
            rows.truncate(max_rows);
        }
        let mut pool = SegmentPool::new();
        let mut conversations =
            registration
                .composer
                .compose(rows, compose_config, tokenizer, &mut pool)?;
        apply_common_contexts(&mut conversations, compose_config, tokenizer, &mut pool)?;
        let context_mode = registration
            .loader
            .default_context_mode()
            .unwrap_or(compose_config.default_context_mode);
        Dataset::new(
            conversations,
            Arc::new(pool.freeze()),
            load_config
                .sampling_strategy
                .as_deref()
                .unwrap_or_else(|| registration.loader.preferred_sampling_strategy()),
            context_mode,
        )
    }
}

fn normalize_name(name: &str) -> String {
    name.to_ascii_lowercase().replace('-', "_")
}

fn probe_file(path: &Path) -> Result<DatasetProbe> {
    if path.extension().and_then(|suffix| suffix.to_str()) == Some("csv") {
        return Ok(DatasetProbe {
            value: None,
            path: Some(path.to_path_buf()),
        });
    }
    let bytes = std::fs::read(path)?;
    let value = if path.extension().and_then(|suffix| suffix.to_str()) == Some("json") {
        Some(probe_value(&serde_json::from_slice(&bytes).map_err(
            |error| DatasetError::Validation(format!("{}: {error}", path.display())),
        )?))
    } else {
        first_json_value(&bytes)?
    };
    Ok(DatasetProbe {
        value,
        path: Some(path.to_path_buf()),
    })
}

fn probe_value(value: &Value) -> Value {
    match value {
        Value::Array(values) => values.first().cloned().unwrap_or(Value::Null),
        other => other.clone(),
    }
}

fn first_json_value(bytes: &[u8]) -> Result<Option<Value>> {
    for (line_index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        let line = trim_ascii(line);
        if line.is_empty() {
            continue;
        }
        return serde_json::from_slice(line).map(Some).map_err(|error| {
            DatasetError::Validation(format!(
                "invalid JSON at in-memory line {}: {error}",
                line_index + 1
            ))
        });
    }
    Ok(None)
}

pub(crate) fn trim_ascii(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

pub(crate) fn jsonl_rows(source: &DatasetSource) -> Result<Vec<RawRow>> {
    match source {
        DatasetSource::Path(path) => rows_from_bytes(&std::fs::read(path)?, Some(path)),
        DatasetSource::Bytes(bytes) => rows_from_bytes(bytes, None),
        DatasetSource::Inline(Value::Array(values)) => Ok(values
            .iter()
            .enumerate()
            .map(|(index, value)| RawRow {
                value: value.clone(),
                wire: serde_json::to_vec(value).ok().map(Bytes::from),
                session_id: None,
                group_key: None,
                origin: RowOrigin::Inline { index },
            })
            .collect()),
        DatasetSource::Inline(value) => Ok(vec![RawRow {
            value: value.clone(),
            wire: serde_json::to_vec(value).ok().map(Bytes::from),
            session_id: None,
            group_key: None,
            origin: RowOrigin::Inline { index: 0 },
        }]),
        DatasetSource::Url(_) | DatasetSource::HuggingFace { .. } => Err(DatasetError::Validation(
            "remote sources must be consumed by a remote-capable loader".into(),
        )),
    }
}

pub(crate) fn rows_from_bytes(bytes: &[u8], path: Option<&Path>) -> Result<Vec<RawRow>> {
    let mut rows = Vec::new();
    for (line_index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        let line = trim_ascii(line);
        if line.is_empty() {
            continue;
        }
        let origin = match path {
            Some(path) => RowOrigin::FileLine {
                path: path.to_path_buf(),
                line: line_index + 1,
            },
            None => RowOrigin::Inline { index: line_index },
        };
        let value = serde_json::from_slice(line).map_err(|error| {
            DatasetError::Validation(format!("{origin}: invalid JSON: {error}"))
        })?;
        rows.push(RawRow {
            value,
            wire: Some(Bytes::copy_from_slice(line)),
            session_id: None,
            group_key: None,
            origin,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use crate::rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::dataset::tokenizer::TiktokenTokenizer;

    #[test]
    fn jsonl_reader_preserves_exact_trimmed_wire_and_line_numbers() {
        let rows = rows_from_bytes(b"\n { \"b\": 2, \"a\": 1 } \n", None).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].wire.as_deref(),
            Some(&b"{ \"b\": 2, \"a\": 1 }"[..])
        );
        assert_eq!(rows[0].origin, RowOrigin::Inline { index: 1 });
    }

    #[test]
    fn registry_rejects_unknown_explicit_format() {
        let registry = LoaderRegistry::new();
        assert!(matches!(
            registry.get("missing"),
            Err(DatasetError::LoaderNotFound(_))
        ));
    }

    #[test]
    fn dag_jsonl_is_not_a_generic_dataset_registration() {
        // `dag_jsonl` is a graph source owned by `aiperf-graph`, never a linear
        // loader. It must stay absent from this registry so a scheduled run can
        // never accidentally parse it as a linear dataset.
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        assert!(registry.get("dag_jsonl").is_err());
    }

    #[tokio::test]
    async fn whole_json_arrays_probe_their_first_row_and_auto_detect() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mt-bench.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&json!([{"prompt":["hello"]}])).unwrap(),
        )
        .unwrap();
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let probe = registry.probe(&DatasetSource::Path(path.clone())).unwrap();
        assert_eq!(probe.value.as_ref().unwrap()["prompt"][0], "hello");
        let bytes_probe = registry
            .probe(&DatasetSource::Bytes(Bytes::from_static(
                br#"[{"prompt":["hello"]}]"#,
            )))
            .unwrap();
        assert_eq!(bytes_probe.value.as_ref().unwrap()["prompt"][0], "hello");

        let dataset = registry
            .build_dataset(
                None,
                &LoadConfig::new(DatasetSource::Path(path)),
                &ComposeConfig::new("model", RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.conversations().len(), 1);
    }

    #[tokio::test]
    async fn generic_row_cap_and_sampling_override_apply_before_freeze() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let mut load = LoadConfig::new(DatasetSource::Inline(serde_json::json!([
            {"text":"first"},
            {"text":"second"}
        ])));
        load.max_rows = Some(1);
        load.sampling_strategy = Some("random".into());

        let dataset = registry
            .build_dataset(
                Some("single_turn"),
                &load,
                &ComposeConfig::new("model", RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();

        assert_eq!(dataset.conversations().len(), 1);
        assert_eq!(dataset.metadata().sampling_strategy, "random");
    }

    #[test]
    fn directory_detection_distinguishes_raw_payloads_from_random_pools() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();

        let raw_directory = tempfile::tempdir().unwrap();
        std::fs::write(
            raw_directory.path().join("requests.jsonl"),
            br#"{"messages":[{"role":"user","content":"hello"}]}"#,
        )
        .unwrap();
        let raw_probe = registry
            .probe(&DatasetSource::Path(raw_directory.path().to_path_buf()))
            .unwrap();
        assert_eq!(
            registry.detect(&raw_probe, "raw directory").unwrap().name,
            "raw_payload"
        );

        let pool_directory = tempfile::tempdir().unwrap();
        std::fs::write(
            pool_directory.path().join("prompts.jsonl"),
            br#"{"text":"hello"}"#,
        )
        .unwrap();
        let pool_probe = registry
            .probe(&DatasetSource::Path(pool_directory.path().to_path_buf()))
            .unwrap();
        assert_eq!(
            registry
                .detect(&pool_probe, "random-pool directory")
                .unwrap()
                .name,
            "random_pool"
        );
    }
}
