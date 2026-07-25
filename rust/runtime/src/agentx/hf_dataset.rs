// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace-hosted WEKA trace dataset download.
//!
//! The byte-exact reconstruction pipeline ([`crate::agentx::loader`]) is format-
//! agnostic: it consumes a `Vec<serde_json::Value>` of rows, each one a JSON
//! object that validates as a [`WekaTrace`]. This module is the thin network
//! adapter that yields those rows for a HuggingFace **dataset** (the Python
//! `SemiAnalysisCCTracesWekaLoader` path), so file-based and HF-based replay run
//! the identical downstream code ([`load_hf_traces_from_rows`]).
//!
//! Rather than reimplement the hub protocol, this delegates to the runtime's own
//! public-dataset loader ([`crate::dataset::load_raw_rows`]) — the same loader
//! the graph-ir `graph::recorded` weka path uses. It resolves the dataset's
//! `cardData.data_files` mapping (so a single root `traces.jsonl` mapped to split
//! `train`, the real `semianalysisai/cc-traces-weka-*` layout, is found),
//! streams only the needed rows through the shared clock-injected fetcher /
//! `~/.cache/huggingface` cache, pins revisions, and decodes JSONL, JSON, CSV,
//! and (under the `parquet` feature) Parquet. We take each row's decoded
//! [`RawRow::value`], which round-trips through [`WekaTrace::from_json_bytes`]
//! exactly as the Python `WekaTrace.model_validate(row)` does.

use crate::agentx::selection::SelectionStats;
use crate::agentx::trace::WekaTrace;
use crate::dataset::{DatasetSource, LoadConfig, load_raw_rows};

/// A HuggingFace dataset coordinate: repo id plus the `datasets`-library
/// selectors (`name=` subset, `split=`, pinned `revision`, optional row cap).
#[derive(Debug, Clone)]
pub struct HfDatasetRef {
    /// The `org/name` (or bare `name`) dataset repository id.
    pub name: String,
    /// The dataset config/subset (`load_dataset(name=...)`); `None` = default.
    pub subset: Option<String>,
    /// The split to load; the AgentX corpora use `"train"`.
    pub split: String,
    /// A pinned commit/branch/tag; `None` resolves `main`.
    pub revision: Option<String>,
    /// Optional row cap; `None` downloads the full reported split. The AgentX
    /// filter-then-cap selection ([`load_hf_weka_traces`]) applies on top of the
    /// rows this yields, so leave it `None` for whole-corpus replay.
    pub max_rows: Option<usize>,
}

impl HfDatasetRef {
    /// A dataset reference for `name`, defaulting `split` to `"train"` (the
    /// AgentX corpus split) with no subset, pinned revision, or row cap.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            subset: None,
            split: "train".to_string(),
            revision: None,
            max_rows: None,
        }
    }
}

/// Reject repository ids that are not a bare or `namespace/name` HuggingFace id,
/// before any network call.
///
/// Fails closed on empty input, whitespace or control characters, and empty /
/// `.` / `..` path segments — a crafted id must never reach the hub client or
/// influence the on-disk cache path. (`load_raw_rows` also validates, but this
/// keeps the guard adjacent to the AgentX entry point and its error wording.)
fn validate_dataset_id(name: &str) -> Result<(), String> {
    let valid = !name.is_empty()
        && !name.chars().any(|c| c.is_whitespace() || c.is_control())
        && name
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
    if valid {
        Ok(())
    } else {
        Err(format!("invalid Hugging Face dataset id {name:?}"))
    }
}

/// Build the runtime public-dataset [`LoadConfig`] for a HuggingFace WEKA source.
fn hf_load_config(dataset: &HfDatasetRef) -> LoadConfig {
    LoadConfig::new(DatasetSource::HuggingFace {
        dataset: dataset.name.clone(),
        // `datasets` names the default config "default"; the runtime loader keys
        // its `cardData.data_files` lookup off that literal.
        config: dataset.subset.clone().unwrap_or_else(|| "default".to_string()),
        split: dataset.split.clone(),
        max_rows: dataset.max_rows,
        revision: dataset.revision.clone(),
    })
}

/// Download a HuggingFace dataset's split and return its rows as JSON values.
///
/// Delegates to [`crate::dataset::load_raw_rows`] (streaming, cached, revision-
/// aware, JSONL/JSON/CSV/Parquet) and projects each [`RawRow`](crate::dataset::RawRow)
/// to its decoded value. The returned rows feed [`load_hf_traces_from_rows`]
/// unchanged.
pub async fn fetch_hf_weka_rows(
    dataset: HfDatasetRef,
) -> Result<Vec<serde_json::Value>, String> {
    validate_dataset_id(&dataset.name)?;
    let config = hf_load_config(&dataset);
    let rows = load_raw_rows(&config)
        .await
        .map_err(|error| format!("loading Hugging Face dataset {:?}: {error}", dataset.name))?;
    Ok(rows.into_iter().map(|row| row.value).collect())
}

/// Download a HuggingFace WEKA-trace dataset and reconstruct the selected traces:
/// fetch rows → validate each as a [`WekaTrace`] → filter-then-cap selection.
///
/// The end-to-end online entry point equivalent to Python's
/// `SemiAnalysisCCTracesWekaLoader.load_dataset`, composed from the byte-exact
/// [`load_hf_traces_from_rows`] over network-fetched rows.
pub async fn load_hf_weka_traces(
    dataset: HfDatasetRef,
    num_dataset_entries: Option<usize>,
    max_context_length: Option<i64>,
    max_osl: Option<i64>,
) -> Result<(Vec<(String, WekaTrace)>, SelectionStats), String> {
    let name = dataset.name.clone();
    let rows = fetch_hf_weka_rows(dataset).await?;
    crate::agentx::loader::load_hf_traces_from_rows(
        rows,
        &name,
        num_dataset_entries,
        max_context_length,
        max_osl,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_adversarial_dataset_ids() {
        for bad in [
            "",
            " ",
            "\t",
            ".",
            "..",
            "../etc/passwd",
            "org/../secret",
            "org//name",
            "/leading",
            "trailing/",
            "https://evil.example/repo",
            "has space",
            "line\nbreak",
            "null\0byte",
        ] {
            assert!(validate_dataset_id(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn accepts_valid_dataset_ids() {
        for ok in [
            "semianalysisai/cc-traces-weka-062126",
            "cc-traces",
            "org/name.with.dots",
        ] {
            assert!(validate_dataset_id(ok).is_ok(), "should accept {ok:?}");
        }
    }

    #[test]
    fn load_config_maps_selectors_and_defaults_subset() {
        let d = HfDatasetRef {
            name: "org/ds".to_string(),
            subset: None,
            split: "train".to_string(),
            revision: Some("abc".to_string()),
            max_rows: Some(5),
        };
        match hf_load_config(&d).source {
            DatasetSource::HuggingFace {
                dataset,
                config,
                split,
                max_rows,
                revision,
            } => {
                assert_eq!(dataset, "org/ds");
                assert_eq!(config, "default");
                assert_eq!(split, "train");
                assert_eq!(max_rows, Some(5));
                assert_eq!(revision.as_deref(), Some("abc"));
            }
            _ => panic!("expected a HuggingFace source"),
        }
    }

    #[test]
    fn load_config_passes_explicit_subset() {
        let d = HfDatasetRef {
            subset: Some("sub_a".to_string()),
            ..HfDatasetRef::new("org/ds")
        };
        match hf_load_config(&d).source {
            DatasetSource::HuggingFace { config, .. } => assert_eq!(config, "sub_a"),
            _ => panic!("expected a HuggingFace source"),
        }
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn adversarial_id_short_circuits_before_network() {
        let err = block_on(fetch_hf_weka_rows(HfDatasetRef::new("../etc/passwd"))).unwrap_err();
        assert!(err.contains("invalid Hugging Face dataset id"), "msg: {err}");
    }

    #[test]
    #[ignore = "hits the Hugging Face hub"]
    fn nonexistent_dataset_errors_cleanly() {
        let name = "aiperf-nonexistent-dataset-xyz-000000";
        let err = block_on(fetch_hf_weka_rows(HfDatasetRef::new(name))).unwrap_err();
        assert!(err.contains(name), "msg: {err}");
    }
}
