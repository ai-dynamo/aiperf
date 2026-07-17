// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace Hub tokenizer download over the `hf-hub` crate.
//!
//! [`crate::dataset::tokenizer::HuggingFaceTokenizer`] loads a tokenizer from a
//! local directory but does not fetch one; this module fills that gap. It uses
//! the same `hf-hub` client that `dynamo-tokenizers` already links transitively
//! (through `fastokens`), pinned to the blocking `ureq` backend so no second
//! async HTTP/TLS stack (`reqwest`/`native-tls`) enters the product graph. `hf-hub`
//! contributes the robustness this download needs for free: bounded retry/backoff,
//! the xet-CDN `302` follow, reuse of the shared `~/.cache/huggingface` cache
//! across runs and processes, and `HF_HUB_OFFLINE` / `HF_TOKEN` handling.
//!
//! The file-selection and cache-directory-resolution logic is ported verbatim
//! from the retired `llm_tokenizer::hub::download_tokenizer_from_hf`
//! (`llm-tokenizer` 1.4.1 `src/hub.rs`) so product download behavior is unchanged
//! by the move off that crate; only the client (async `reqwest` → blocking `ureq`)
//! differs. The blocking `hf-hub` call runs on a `spawn_blocking` worker so the
//! public entry point keeps its `async` contract for both the online resolver and
//! the `aiperf chat` loader.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};

use crate::dataset::error::{DatasetError, Result};

/// Environment variable carrying a HuggingFace access token.
///
/// `hf-hub`'s `from_env` reads the on-disk token file but not this variable, so
/// it is applied explicitly to support CI where the token is only an env var.
const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Bounded automatic retry for transient hub failures (429/5xx/timeouts). This is
/// the robustness the previous single-shot download over AIPerf's fetcher lacked.
const DOWNLOAD_RETRIES: usize = 3;

/// Repository files that are never tokenizer artifacts.
const IGNORED: [&str; 5] = [
    ".gitattributes",
    "LICENSE",
    "LICENSE.txt",
    "README.md",
    "USE_POLICY.md",
];

/// Build a hub client from the ambient environment plus an explicit `HF_TOKEN`.
fn build_api() -> Result<Api> {
    let mut builder = ApiBuilder::from_env().with_retries(DOWNLOAD_RETRIES);
    if let Ok(token) = std::env::var(HF_TOKEN_ENV)
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    builder
        .build()
        .map_err(|error| DatasetError::Tokenizer(format!("configuring Hugging Face hub: {error}")))
}

/// True for model-weight files, which a tokenizer download must never pull.
fn is_weight_file(filename: &str) -> bool {
    filename.ends_with(".bin")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".h5")
        || filename.ends_with(".msgpack")
        || filename.ends_with(".ckpt.index")
}

/// True for image files, which are ignored during tokenizer acquisition.
fn is_image(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg")
}

/// True for a separate Jinja chat-template file shipped alongside the tokenizer.
fn is_chat_template_file(filename: &str) -> bool {
    filename.ends_with(".jinja") || filename == "chat_template.json"
}

/// True for the tokenizer artifact files worth downloading.
fn is_tokenizer_file(filename: &str) -> bool {
    filename.ends_with("tokenizer.json")
        || filename.ends_with("tokenizer_config.json")
        || filename.ends_with("special_tokens_map.json")
        || filename.ends_with("vocab.json")
        || filename.ends_with("merges.txt")
        || filename.ends_with(".model")
        || filename.ends_with(".tiktoken")
        || is_chat_template_file(filename)
}

/// Download `repository`'s tokenizer files into the standard HuggingFace cache and
/// return the snapshot directory for [`crate::dataset::HuggingFaceTokenizer::from_directory`].
///
/// `hf-hub` resolves the `main` revision; pinned-commit acquisition stays on the
/// HTTP fetcher seam ([`crate::engine`]'s `NativeOnlineTokenizerSourceResolver`).
/// A free function so a distribution can wrap or replace it while the tokenizer
/// type stays download-mechanism agnostic.
pub async fn download_hugging_face_tokenizer(repository: &str) -> Result<PathBuf> {
    let repository = repository.to_string();
    tokio::task::spawn_blocking(move || download_blocking(&repository))
        .await
        .map_err(|error| {
            DatasetError::Tokenizer(format!(
                "Hugging Face tokenizer download task failed: {error}"
            ))
        })?
}

/// Blocking `hf-hub` download body, run on a `spawn_blocking` worker.
fn download_blocking(repository: &str) -> Result<PathBuf> {
    let api = build_api()?;
    let repo = api.model(repository.to_string());

    let info = repo.info().map_err(|error| {
        DatasetError::Tokenizer(format!(
            "Failed to fetch model {repository:?} from Hugging Face: {error}. \
             Is this a valid Hugging Face ID?"
        ))
    })?;
    if info.siblings.is_empty() {
        return Err(DatasetError::Tokenizer(format!(
            "Model {repository:?} exists but contains no downloadable files."
        )));
    }

    let tokenizer_files: Vec<&str> = info
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.as_str())
        .filter(|filename| {
            !IGNORED.contains(filename)
                && !is_image(filename)
                && !is_weight_file(filename)
                && is_tokenizer_file(filename)
        })
        .collect();
    if tokenizer_files.is_empty() {
        return Err(DatasetError::Tokenizer(format!(
            "No tokenizer files found for model {repository:?}."
        )));
    }

    let mut cache_dir = None;
    for filename in tokenizer_files {
        let path = repo.get(filename).map_err(|error| {
            DatasetError::Tokenizer(format!(
                "Failed to download tokenizer file {filename:?} from model {repository:?}: {error}"
            ))
        })?;
        if cache_dir.is_none() {
            cache_dir = path.parent().map(Path::to_path_buf);
        }
    }

    // Config files enrich EOS-token loading but are best-effort: downloaded by
    // exact name so a nested `1_Pooling/config.json` cannot shadow `cache_dir`.
    for config_file in ["config.json", "generation_config.json"] {
        let _ = repo.get(config_file);
    }

    match cache_dir {
        Some(directory) => Ok(resolve_model_cache_dir(&directory, repository)),
        None => Err(DatasetError::Tokenizer(format!(
            "Invalid Hugging Face cache path for model {repository:?}"
        ))),
    }
}

/// Resolve the model directory that directly contains the tokenizer files.
///
/// Handles the `original/` weights subfolder and the `models--org--name/snapshots`
/// cache layout so the returned path is the one holding `tokenizer.json`.
fn resolve_model_cache_dir(path: &Path, model_name: &str) -> PathBuf {
    if let Some(parent) = path.parent()
        && path.file_name().is_some_and(|folder| folder == "original")
    {
        return parent.to_path_buf();
    }

    let model_parts: Vec<&str> = model_name.split('/').collect();
    if model_parts.len() >= 2 {
        let expected_pattern = format!(
            "models--{}--{}",
            model_parts[0].replace('-', "--"),
            model_parts[1].replace('-', "--")
        );

        if path.to_string_lossy().contains(&expected_pattern) {
            return path.to_path_buf();
        }

        if path.join("tokenizer.json").exists() || path.join("tokenizer_config.json").exists() {
            return path.to_path_buf();
        }

        let mut current = path.to_path_buf();
        while let Some(parent) = current.parent() {
            if parent.to_string_lossy().contains(&expected_pattern) {
                let snapshots_dir = parent.join("snapshots");
                if snapshots_dir.is_dir()
                    && let Ok(entries) = std::fs::read_dir(&snapshots_dir)
                {
                    for entry in entries.flatten() {
                        let snapshot_path = entry.path();
                        if snapshot_path.is_dir()
                            && (snapshot_path.join("tokenizer.json").exists()
                                || snapshot_path.join("tokenizer_config.json").exists())
                        {
                            return snapshot_path;
                        }
                    }
                }
                return parent.to_path_buf();
            }
            current = parent.to_path_buf();
        }
    }

    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{is_chat_template_file, is_tokenizer_file, is_weight_file};

    #[test]
    fn recognizes_tokenizer_files() {
        assert!(is_tokenizer_file("tokenizer.json"));
        assert!(is_tokenizer_file("tokenizer_config.json"));
        assert!(is_tokenizer_file("special_tokens_map.json"));
        assert!(is_tokenizer_file("vocab.json"));
        assert!(is_tokenizer_file("merges.txt"));
        assert!(is_tokenizer_file("spiece.model"));
        assert!(is_tokenizer_file("chat_template.jinja"));
        assert!(!is_tokenizer_file("model.bin"));
        assert!(!is_tokenizer_file("README.md"));
    }

    #[test]
    fn recognizes_chat_template_files() {
        assert!(is_chat_template_file("chat_template.jinja"));
        assert!(is_chat_template_file("chat_template.json"));
        assert!(!is_chat_template_file("tokenizer.json"));
        assert!(!is_chat_template_file("README.md"));
    }

    #[test]
    fn recognizes_weight_files() {
        assert!(is_weight_file("model.bin"));
        assert!(is_weight_file("model.safetensors"));
        assert!(is_weight_file("pytorch_model.bin"));
        assert!(!is_weight_file("tokenizer.json"));
        assert!(!is_weight_file("config.json"));
    }
}
