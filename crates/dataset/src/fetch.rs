// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware remote dataset fetching over AIPerf's one HTTP stack.
//!
//! Public-data downloads use the same Clock-injected hyper transport as
//! inference dispatch. A dedicated current-thread runtime is created on a
//! blocking worker because the transport is intentionally `!Send`/`Rc`-based.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use aiperf_clock::{Clock, RealClock};
use aiperf_transport_http::config::ClientConfig;
use aiperf_transport_http::models::{RequestConfig, Response};
use aiperf_transport_http::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use bytes::Bytes;

use crate::error::{DatasetError, Result};

/// Remote byte-fetching extension point used by URL and Hugging Face sources.
#[async_trait]
pub trait DatasetFetcher: Send + Sync {
    /// Fetch `url`, optionally authenticating with `bearer_token`, and cache the
    /// exact response bytes under `cache_key`.
    async fn fetch(&self, url: &str, cache_key: &str, bearer_token: Option<&str>) -> Result<Bytes>;
}

/// Native hyper fetcher with a persistent on-disk exact-byte cache.
#[derive(Debug, Clone)]
pub struct HttpDatasetFetcher {
    cache_directory: PathBuf,
}

impl HttpDatasetFetcher {
    /// Store downloads beneath `cache_directory`.
    pub fn new(cache_directory: impl Into<PathBuf>) -> Self {
        Self {
            cache_directory: cache_directory.into(),
        }
    }

    fn cache_path(&self, cache_key: &str) -> PathBuf {
        let digest = blake3::hash(cache_key.as_bytes()).to_hex();
        self.cache_directory.join(digest.as_str())
    }
}

impl Default for HttpDatasetFetcher {
    fn default() -> Self {
        Self::new(".cache/aiperf/datasets-rust")
    }
}

#[async_trait]
impl DatasetFetcher for HttpDatasetFetcher {
    async fn fetch(&self, url: &str, cache_key: &str, bearer_token: Option<&str>) -> Result<Bytes> {
        let cache_path = self.cache_path(cache_key);
        if cache_path.is_file() {
            return Ok(Bytes::from(std::fs::read(cache_path)?));
        }
        let url = url.to_string();
        let token = bearer_token.map(str::to_string);
        let bytes = tokio::task::spawn_blocking(move || fetch_on_local_runtime(url, token))
            .await
            .map_err(|error| {
                DatasetError::Validation(format!("dataset download task failed: {error}"))
            })??;
        persist_cache(&cache_path, &bytes)?;
        Ok(bytes)
    }
}

fn fetch_on_local_runtime(url: String, bearer_token: Option<String>) -> Result<Bytes> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(DatasetError::Io)?;
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(async move {
        let clock: Rc<dyn Clock> = RealClock::new();
        let transport =
            HttpTransport::new(clock, ClientConfig::default()).with_user_agent("aiperf-dataset/0");
        let initial_url = url.clone();
        let mut current_url = url;
        for _ in 0..=8 {
            let mut request = RequestConfig::new(&current_url);
            if let Some(token) = bearer_token
                .as_deref()
                .filter(|_| is_hugging_face(&current_url))
            {
                request
                    .headers
                    .insert("Authorization".into(), format!("Bearer {token}"));
            }
            let record = transport.get(&request).await;
            if let Some(error) = record.error {
                return Err(DatasetError::Validation(format!(
                    "failed to download dataset {current_url:?}: {error:?}"
                )));
            }
            if matches!(record.status, Some(301 | 302 | 303 | 307 | 308)) {
                let location = record.response_headers.get("location").ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "dataset redirect from {current_url:?} has no Location header"
                    ))
                })?;
                current_url = url::Url::parse(&current_url)
                    .and_then(|base| base.join(location))
                    .map_err(|error| {
                        DatasetError::Validation(format!(
                            "invalid redirect from {current_url:?}: {error}"
                        ))
                    })?
                    .into();
                continue;
            }
            if record.status != Some(200) {
                return Err(DatasetError::Validation(format!(
                    "dataset download {current_url:?} returned HTTP {}",
                    record
                        .status
                        .map_or_else(|| "no status".into(), |status| status.to_string())
                )));
            }
            return record
                .responses
                .into_iter()
                .find_map(|response| match response {
                    Response::Text(response) => Some(response.body),
                    Response::Sse(_) => None,
                })
                .ok_or_else(|| {
                    DatasetError::Validation(format!(
                        "dataset download {current_url:?} returned no response body"
                    ))
                });
        }
        Err(DatasetError::Validation(format!(
            "dataset download {initial_url:?} exceeded the redirect limit"
        )))
    }))
}

fn is_hugging_face(value: &str) -> bool {
    url::Url::parse(value)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "huggingface.co" || host.ends_with(".huggingface.co"))
}

fn persist_cache(path: &Path, bytes: &Bytes) -> Result<()> {
    let parent = path.parent().ok_or_else(|| {
        DatasetError::Validation(format!("cache path {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent)?;
    let temporary = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_keys_are_path_safe_and_stable() {
        let fetcher = HttpDatasetFetcher::new("cache");
        let first = fetcher.cache_path("https://example.com/a?token=secret");
        let second = fetcher.cache_path("https://example.com/a?token=secret");
        assert_eq!(first, second);
        assert_eq!(first.file_name().unwrap().to_string_lossy().len(), 64);
        assert!(!first.to_string_lossy().contains("secret"));
    }
}
