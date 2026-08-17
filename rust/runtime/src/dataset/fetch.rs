// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware remote dataset fetching over AIPerf's one HTTP stack.
//!
//! Public-data downloads use the same Clock-injected hyper transport as
//! inference dispatch. A dedicated current-thread runtime is created on a
//! blocking worker because the transport is intentionally `!Send`/`Rc`-based.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::clock::{Clock, RealClock};
use crate::transport::core::{RequestRecord, Response};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::http_transport::HttpTransport;
use async_trait::async_trait;
use bytes::Bytes;

use crate::dataset::error::{DatasetError, Result};

/// Remote byte-fetching extension point used by URL and Hugging Face sources.
#[async_trait]
pub trait DatasetFetcher: Send + Sync {
    /// Fetch `url`, optionally authenticating with `bearer_token`, and cache the
    /// exact response bytes under `cache_key`.
    async fn fetch(&self, url: &str, cache_key: &str, bearer_token: Option<&str>) -> Result<Bytes>;

    /// Whether revision-pinned Parquet/JSONL may stream through `hf-hub` instead
    /// of [`Self::fetch`].
    ///
    /// The default is `false` so injected fetchers (tests, custom caches) remain
    /// the exclusive download path. [`HttpDatasetFetcher`] opts in so production
    /// can stream giant shards without loading them into memory.
    fn allows_hf_hub_streaming(&self) -> bool {
        false
    }
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

    fn allows_hf_hub_streaming(&self) -> bool {
        true
    }
}

/// Per-request GET seam behind the redirect-following fetch loop.
///
/// Extracted so [`fetch_following_redirects`] can be exercised with an injected
/// transport double: the loop's subtle redirect/error ordering (a 3xx must win
/// over the transport's non-2xx-is-error signal) otherwise needs a live server.
/// `?Send` because the underlying [`HttpTransport`] is intentionally `!Send` and
/// runs on a current-thread `LocalSet`.
#[async_trait(?Send)]
trait FetchTransport {
    /// Dispatch one GET and return its terminal record.
    async fn get(&self, request: &RequestConfig) -> RequestRecord;
}

#[async_trait(?Send)]
impl FetchTransport for HttpTransport {
    async fn get(&self, request: &RequestConfig) -> RequestRecord {
        HttpTransport::get(self, request).await
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
        // Dataset downloads are external and may sit behind a forward proxy, so
        // honor the proxy environment (loopback always excluded). This never
        // touches the benchmark transport, whose ClientConfig leaves proxy None.
        let mut client_config = ClientConfig::default();
        if let Ok(parsed) = url::Url::parse(&url) {
            client_config.proxy =
                crate::transport::http::client::proxy::ProxyConfig::from_env_for(&parsed);
        }
        let transport =
            HttpTransport::new(clock, client_config).with_user_agent("aiperf-dataset/0");
        fetch_following_redirects(&transport, url, bearer_token).await
    }))
}

/// Download `url` over `transport`, following up to eight HTTP redirects.
///
/// The redirect status is honored before `record.error` is inspected: the
/// inference-oriented transport reports every non-2xx (including 3xx) as an
/// error while still populating `status` and the `Location` header, so checking
/// the redirect first is what lets tokenizer/dataset downloads follow HF CDN
/// hops instead of failing fatally on the first 302. The `huggingface.co` auth
/// bearer is deliberately not carried across a redirect to a CDN host.
async fn fetch_following_redirects(
    transport: &dyn FetchTransport,
    url: String,
    bearer_token: Option<String>,
) -> Result<Bytes> {
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
        if let Some(error) = record.error {
            return Err(DatasetError::Validation(format!(
                "failed to download dataset {current_url:?}: {error:?}"
            )));
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
    use crate::transport::core::{ErrorDetails, TextResponse};
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    /// Transport double that replays a scripted queue of records and records the
    /// URLs it was asked for, so redirect chains can be asserted without sockets.
    struct ScriptedTransport {
        records: RefCell<Vec<RequestRecord>>,
        requested_urls: RefCell<Vec<String>>,
    }

    #[async_trait(?Send)]
    impl FetchTransport for ScriptedTransport {
        async fn get(&self, request: &RequestConfig) -> RequestRecord {
            self.requested_urls.borrow_mut().push(request.url.clone());
            self.records.borrow_mut().remove(0)
        }
    }

    fn text_body(body: &'static [u8]) -> Response {
        Response::Text(TextResponse {
            perf_ns: 0,
            text: String::from_utf8_lossy(body).into_owned(),
            body: Bytes::from_static(body),
            content_type: None,
        })
    }

    fn run_fetch(transport: &ScriptedTransport, url: &str) -> Result<Bytes> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        runtime.block_on(local.run_until(fetch_following_redirects(
            transport,
            url.to_string(),
            None,
        )))
    }

    #[test]
    fn follows_302_even_when_transport_also_reports_it_as_an_error() {
        // The inference transport sets status, Location header, AND error on a
        // 302; the loop must follow the redirect rather than fail on the error.
        let redirect = RequestRecord {
            status: Some(302),
            error: Some(ErrorDetails::http(302, "")),
            response_headers: BTreeMap::from([(
                "location".to_string(),
                "https://cdn.example.com/tokenizer.json".to_string(),
            )]),
            ..RequestRecord::default()
        };

        let ok = RequestRecord {
            status: Some(200),
            responses: vec![text_body(b"{\"tokenizer\":true}")],
            ..RequestRecord::default()
        };

        let transport = ScriptedTransport {
            records: RefCell::new(vec![redirect, ok]),
            requested_urls: RefCell::new(Vec::new()),
        };

        let bytes = run_fetch(&transport, "https://huggingface.co/model/tokenizer.json").unwrap();
        assert_eq!(bytes.as_ref(), b"{\"tokenizer\":true}");

        let urls = transport.requested_urls.borrow();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://huggingface.co/model/tokenizer.json");
        assert_eq!(urls[1], "https://cdn.example.com/tokenizer.json");
    }

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
