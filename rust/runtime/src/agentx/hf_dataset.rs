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
//! HuggingFace datasets are stored as Parquet, where each row is one trace. We
//! download the split's Parquet file(s) into the shared `~/.cache/huggingface`
//! cache with the same blocking `hf-hub`/`ureq` client the tokenizer download
//! uses (retry/backoff, `HF_TOKEN`/`HF_HUB_OFFLINE`), then decode each Parquet
//! row to a `serde_json::Value` via [`parquet::record::Row::to_json_value`] —
//! nested `requests`/`hash_ids` lists become JSON arrays with field names
//! preserved, so the value round-trips through [`WekaTrace::from_json_bytes`]
//! exactly as the Python `WekaTrace.model_validate(row)` does.
//!
//! Requires the `parquet` feature (the arrow/parquet reader). Without it, HF
//! dataset inputs are rejected the same way `.parquet` file inputs are.

use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;

use crate::agentx::selection::SelectionStats;
use crate::agentx::trace::WekaTrace;

/// Environment variable carrying a HuggingFace access token.
///
/// `hf-hub`'s `from_env` reads the on-disk token file but not this variable, so
/// it is applied explicitly to support CI where the token is only an env var.
const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Bounded automatic retry for transient hub failures (429/5xx/timeouts).
const DOWNLOAD_RETRIES: usize = 3;

/// A HuggingFace dataset coordinate: repo id plus the `datasets`-library
/// selectors (`name=` subset, `split=`, pinned `revision`).
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
}

impl HfDatasetRef {
    /// A dataset reference for `name`, defaulting `split` to `"train"` (the
    /// AgentX corpus split) with no subset or pinned revision.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            subset: None,
            split: "train".to_string(),
            revision: None,
        }
    }
}

/// Reject repository ids that are not a bare or `namespace/name` HuggingFace id,
/// before any network call.
///
/// Fails closed on empty input, whitespace or control characters, and empty /
/// `.` / `..` path segments — a crafted id must never reach the hub client or
/// influence the on-disk cache path.
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

/// Build a hub client from the ambient environment plus an explicit `HF_TOKEN`.
fn build_api() -> Result<Api, String> {
    let mut builder = ApiBuilder::from_env().with_retries(DOWNLOAD_RETRIES);
    if let Ok(token) = std::env::var(HF_TOKEN_ENV)
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    builder
        .build()
        .map_err(|error| format!("configuring Hugging Face hub: {error}"))
}

/// Select the Parquet files for `split` (and `subset`, if any) from a dataset
/// repo's file listing, matching HuggingFace's on-repo layouts.
///
/// HuggingFace stores dataset splits as Parquet under conventions like
/// `<subset>/<split>-00000-of-00001.parquet`, `data/<split>-*.parquet`, or the
/// auto-converted `<config>/<split>/0000.parquet`. We keep `.parquet` siblings
/// whose path contains the split token (and the subset, when given), preferring
/// the most specific match and falling back to every Parquet file when a repo
/// uses an unconventional name. Results are sorted for deterministic ordering.
fn select_parquet_files(files: &[String], subset: Option<&str>, split: &str) -> Vec<String> {
    let is_parquet = |f: &&String| f.ends_with(".parquet");
    // A path "contains" a token when the token appears as a `/`-, `-`, or `.`-
    // delimited component (so `train` does not match `retrained`). `_` is NOT a
    // delimiter: subset/config names embed underscores (`sub_a`, and the
    // `semianalysis_cc_traces_weka_*` corpora), so splitting on it would prevent
    // the subset token from ever matching a directory segment.
    let contains_token = |path: &str, token: &str| {
        path.split(['/', '-', '.'])
            .any(|seg| seg.eq_ignore_ascii_case(token))
    };

    let parquet: Vec<String> = files.iter().filter(is_parquet).cloned().collect();
    if parquet.is_empty() {
        return parquet;
    }

    // Most specific: matches subset (if given) AND split.
    let mut selected: Vec<String> = parquet
        .iter()
        .filter(|f| {
            subset.is_none_or(|s| contains_token(f, s)) && contains_token(f, split)
        })
        .cloned()
        .collect();

    // Next: split token alone (a repo without per-subset directories).
    if selected.is_empty() {
        selected = parquet
            .iter()
            .filter(|f| contains_token(f, split))
            .cloned()
            .collect();
    }

    // Last resort: a single-split repo whose files carry no split token.
    if selected.is_empty() {
        selected = parquet;
    }

    selected.sort();
    selected
}

/// Decode every row of a locally-cached Parquet file to a `serde_json::Value`.
///
/// Each Parquet row becomes one JSON object with column names preserved and
/// nested list/group columns as JSON arrays/objects — the shape
/// [`WekaTrace::from_json_bytes`] validates.
fn parquet_rows_to_json(path: &std::path::Path) -> Result<Vec<serde_json::Value>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("opening cached parquet {}: {e}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| format!("reading parquet {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    let iter = reader
        .get_row_iter(None)
        .map_err(|e| format!("iterating parquet {}: {e}", path.display()))?;
    for row in iter {
        let row = row.map_err(|e| format!("decoding parquet row in {}: {e}", path.display()))?;
        rows.push(row.to_json_value());
    }
    Ok(rows)
}

/// Download a HuggingFace dataset's split and return its rows as JSON values.
///
/// Runs the blocking `hf-hub` download + Parquet decode on a `spawn_blocking`
/// worker. The returned rows feed [`load_hf_traces_from_rows`] unchanged.
pub async fn fetch_hf_weka_rows(
    dataset: HfDatasetRef,
) -> Result<Vec<serde_json::Value>, String> {
    tokio::task::spawn_blocking(move || fetch_blocking(&dataset))
        .await
        .map_err(|error| format!("Hugging Face dataset download task failed: {error}"))?
}

/// Blocking `hf-hub` dataset download + Parquet decode body.
fn fetch_blocking(dataset: &HfDatasetRef) -> Result<Vec<serde_json::Value>, String> {
    validate_dataset_id(&dataset.name)?;
    let api = build_api()?;
    let repo = match &dataset.revision {
        Some(rev) => Repo::with_revision(dataset.name.clone(), RepoType::Dataset, rev.clone()),
        None => Repo::new(dataset.name.clone(), RepoType::Dataset),
    };
    let repo = api.repo(repo);

    let info = repo.info().map_err(|error| {
        format!(
            "Failed to fetch dataset {:?} from Hugging Face: {error}. \
             Is this a valid dataset id?{}",
            dataset.name,
            if dataset.revision.is_some() {
                " Check the pinned revision."
            } else {
                ""
            }
        )
    })?;
    let files: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .collect();

    let parquet_files =
        select_parquet_files(&files, dataset.subset.as_deref(), &dataset.split);
    if parquet_files.is_empty() {
        return Err(format!(
            "Dataset {:?} exposes no Parquet files for split {:?}{} \
             (found {} repo file(s)). Only Parquet-backed HuggingFace datasets \
             are supported natively; convert to a local trace file otherwise.",
            dataset.name,
            dataset.split,
            dataset
                .subset
                .as_deref()
                .map(|s| format!(" subset {s:?}"))
                .unwrap_or_default(),
            files.len(),
        ));
    }

    let mut rows = Vec::new();
    for filename in parquet_files {
        let path = repo.get(&filename).map_err(|error| {
            format!(
                "Failed to download {filename:?} from dataset {:?}: {error}",
                dataset.name
            )
        })?;
        rows.extend(parquet_rows_to_json(&path)?);
    }
    Ok(rows)
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
    fn selects_subset_and_split_specific_parquet() {
        let files = vec![
            "README.md".to_string(),
            "sub_a/train-00000-of-00002.parquet".to_string(),
            "sub_a/train-00001-of-00002.parquet".to_string(),
            "sub_a/test-00000-of-00001.parquet".to_string(),
            "sub_b/train-00000-of-00001.parquet".to_string(),
        ];
        let got = select_parquet_files(&files, Some("sub_a"), "train");
        assert_eq!(
            got,
            vec![
                "sub_a/train-00000-of-00002.parquet".to_string(),
                "sub_a/train-00001-of-00002.parquet".to_string(),
            ]
        );
    }

    #[test]
    fn selects_split_when_no_subset_dirs() {
        let files = vec![
            "data/train-00000-of-00001.parquet".to_string(),
            "data/validation-00000-of-00001.parquet".to_string(),
        ];
        let got = select_parquet_files(&files, None, "train");
        assert_eq!(got, vec!["data/train-00000-of-00001.parquet".to_string()]);
    }

    #[test]
    fn split_token_is_component_matched_not_substring() {
        // "retrained" must not match the "train" split token.
        let files = vec![
            "retrained/data-00000.parquet".to_string(),
            "train/0000.parquet".to_string(),
        ];
        let got = select_parquet_files(&files, None, "train");
        assert_eq!(got, vec!["train/0000.parquet".to_string()]);
    }

    #[test]
    fn falls_back_to_all_parquet_when_no_split_token() {
        let files = vec![
            "0000.parquet".to_string(),
            "0001.parquet".to_string(),
            "notes.txt".to_string(),
        ];
        let got = select_parquet_files(&files, None, "train");
        assert_eq!(got, vec!["0000.parquet".to_string(), "0001.parquet".to_string()]);
    }

    #[test]
    fn no_parquet_files_yields_empty() {
        let files = vec!["README.md".to_string(), "data.json".to_string()];
        assert!(select_parquet_files(&files, None, "train").is_empty());
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
