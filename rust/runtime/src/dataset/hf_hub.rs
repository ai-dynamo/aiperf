// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace Hub tokenizer download.
//!
//! [`crate::dataset::tokenizer::HuggingFaceTokenizer`] loads from a local directory
//! but does not fetch; this fills that gap with the same `hf-hub` client
//! `dynamo-tokenizers` already links via `fastokens`, pinned to the blocking `ureq`
//! backend so no `reqwest`/`native-tls` stack enters the product graph. That buys
//! retry/backoff, xet-CDN `302`, shared `~/.cache/huggingface` reuse, and
//! `HF_HUB_OFFLINE`/`HF_TOKEN` handling.
//!
//! File selection excludes weights, images, and non-tokenizer repository files;
//! cache resolution returns the directory containing the tokenizer artifacts.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};

use crate::dataset::error::{DatasetError, Result};

/// Environment variable carrying a HuggingFace access token.
///
/// `hf-hub`'s `from_env` reads the on-disk token file but not this variable, so
/// it is applied explicitly to support CI where the token is only an env var.
const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Bounded automatic retry for transient hub failures (429/5xx/timeouts).
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

fn is_weight_file(filename: &str) -> bool {
    filename.ends_with(".bin")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".h5")
        || filename.ends_with(".msgpack")
        || filename.ends_with(".ckpt.index")
}

fn is_image(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg")
}

fn is_chat_template_file(filename: &str) -> bool {
    filename.ends_with(".jinja") || filename == "chat_template.json"
}

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

/// Select tokenizer artifacts from `repo.info()` siblings.
fn is_downloadable_tokenizer_file(filename: &str) -> bool {
    !IGNORED.contains(&filename)
        && !is_image(filename)
        && !is_weight_file(filename)
        && is_tokenizer_file(filename)
}

/// Reject repository ids that are not a bare or `namespace/name` HuggingFace id,
/// before any network call.
///
/// Fails closed on empty input, whitespace or control characters, and empty /
/// `.` / `..` path segments — a crafted id (`""`, `"../etc/passwd"`,
/// `"https://evil.example"`, `"org//name"`) must never reach the hub client or
/// influence the on-disk cache path.
fn validate_repository_id(repository: &str) -> Result<()> {
    let valid = !repository.is_empty()
        && !repository
            .chars()
            .any(|c| c.is_whitespace() || c.is_control())
        && repository
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
    if valid {
        Ok(())
    } else {
        Err(DatasetError::Tokenizer(format!(
            "invalid Hugging Face repository id {repository:?}"
        )))
    }
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
    validate_repository_id(repository)?;
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
        .filter(|filename| is_downloadable_tokenizer_file(filename))
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
    use super::*;
    use std::path::Path;

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

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

    // The download filter must never pull a weight/image/ignored file, even one
    // wearing a tokenizer-ish name (`tokenizer.safetensors`).
    #[test]
    fn downloadable_filter_excludes_weights_images_and_ignored() {
        assert!(is_downloadable_tokenizer_file("tokenizer.json"));
        assert!(is_downloadable_tokenizer_file("tokenizer_config.json"));
        assert!(is_downloadable_tokenizer_file("chat_template.jinja"));

        assert!(!is_downloadable_tokenizer_file("tokenizer.safetensors"));
        assert!(!is_downloadable_tokenizer_file("model.safetensors"));
        assert!(!is_downloadable_tokenizer_file("pytorch_model.bin"));
        assert!(!is_downloadable_tokenizer_file(".gitattributes"));
        assert!(!is_downloadable_tokenizer_file("README.md"));
        assert!(!is_downloadable_tokenizer_file("preview.png"));
        assert!(!is_downloadable_tokenizer_file(""));
        assert!(!is_downloadable_tokenizer_file("TOKENIZER.JSON"));
    }

    // The classifier matches by suffix and admits sub-paths; it is not a
    // path-traversal defense (the hub never serves such siblings, hf-hub owns
    // cache-path safety, and `validate_repository_id` guards the id itself).
    #[test]
    fn classifier_is_suffix_based_not_a_path_guard() {
        assert!(is_tokenizer_file("onnx/tokenizer.json"));
        assert!(is_tokenizer_file("../tokenizer.json"));
        assert!(is_tokenizer_file("evil_tokenizer.json"));
    }

    #[test]
    fn rejects_adversarial_repository_ids() {
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
            assert!(
                validate_repository_id(bad).is_err(),
                "should reject {bad:?}"
            );
        }
    }

    #[test]
    fn accepts_valid_repository_ids() {
        for ok in [
            "gpt2",
            "openai-community/gpt2",
            "Qwen/Qwen2.5-7B-Instruct",
            "org/name.with.dots",
        ] {
            assert!(validate_repository_id(ok).is_ok(), "should accept {ok:?}");
        }
    }

    #[test]
    fn adversarial_id_short_circuits_before_network() {
        let error = block_on(download_hugging_face_tokenizer("../etc/passwd")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid Hugging Face repository id"),
            "msg: {error}"
        );
    }

    #[test]
    fn cache_dir_original_subfolder_climbs_one_level() {
        let dir = resolve_model_cache_dir(Path::new("/cache/x/original"), "org/name");
        assert_eq!(dir, Path::new("/cache/x"));
    }

    #[test]
    fn cache_dir_returns_path_when_pattern_present() {
        let path = Path::new("/cache/models--org--name/snapshots/abcdef");
        assert_eq!(resolve_model_cache_dir(path, "org/name"), path);
    }

    #[test]
    fn cache_dir_returns_dir_holding_tokenizer_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("tokenizer.json"), "{}").unwrap();
        assert_eq!(resolve_model_cache_dir(tmp.path(), "org/name"), tmp.path());
    }

    #[test]
    fn cache_dir_unchanged_for_nonmatching_or_bare_name() {
        let path = Path::new("/some/unrelated/path");
        assert_eq!(resolve_model_cache_dir(path, "org/name"), path);
        assert_eq!(resolve_model_cache_dir(path, "gpt2"), path);
    }

    #[test]
    #[ignore = "hits the Hugging Face hub"]
    fn nonexistent_repository_errors_cleanly() {
        let repo = "aiperf-nonexistent-model-xyz-000000";
        let error = block_on(download_hugging_face_tokenizer(repo)).unwrap_err();
        assert!(error.to_string().contains(repo), "msg: {error}");
    }

    #[test]
    #[ignore = "run with HF_HUB_OFFLINE=1 and a cold cache"]
    fn offline_cold_cache_errors() {
        let result = block_on(download_hugging_face_tokenizer(
            "aiperf-definitely-not-cached-000000",
        ));
        assert!(result.is_err());
    }
}
