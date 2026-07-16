// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inline-media preparation above the HTTP byte transport.
//!
//! Unique HTTP(S) image URLs are fetched once, validated, and replaced with
//! data URLs before inference dispatch. Fetching and encoding are separate
//! traits so alternate object stores, credential policies, and media codecs do
//! not fork endpoint logic.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::Value;

use crate::transport::http::models::{ErrorDetails, RequestConfig, Response};
use crate::transport::http::transport::http_transport::HttpTransport;

/// Exact bytes and content type returned by a media fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedMedia {
    /// Exact response bytes.
    pub bytes: Bytes,
    /// Response Content-Type, when supplied.
    pub content_type: Option<String>,
}

/// Pluggable asynchronous byte fetcher used by inline-media preparation.
#[async_trait(?Send)]
pub trait MediaFetcher {
    /// Fetch one absolute HTTP(S) media URL.
    async fn fetch(&self, url: &str) -> Result<FetchedMedia, ErrorDetails>;
}

/// Conversion from fetched bytes to an endpoint-ready inline value.
pub trait InlineMediaEncoder {
    /// Validate and encode one fetched media object.
    fn encode(&self, fetched: &FetchedMedia) -> Result<String, ErrorDetails>;
}

/// Media fetcher backed by the shared Clock-injected HTTP transport.
pub struct HttpMediaFetcher<'a> {
    transport: &'a HttpTransport,
}

impl<'a> HttpMediaFetcher<'a> {
    /// Borrow one shared transport for media downloads.
    pub const fn new(transport: &'a HttpTransport) -> Self {
        Self { transport }
    }
}

#[async_trait(?Send)]
impl MediaFetcher for HttpMediaFetcher<'_> {
    async fn fetch(&self, url: &str) -> Result<FetchedMedia, ErrorDetails> {
        let initial = url.to_string();
        let mut current = initial.clone();
        for _ in 0..=8 {
            let record = self.transport.get(&RequestConfig::new(&current)).await;
            if matches!(record.status, Some(301 | 302 | 303 | 307 | 308)) {
                let location = record.response_headers.get("location").ok_or_else(|| {
                    ErrorDetails::other(format!(
                        "media redirect from {current:?} has no Location header"
                    ))
                })?;
                current = url::Url::parse(&current)
                    .and_then(|base| base.join(location))
                    .map_err(|error| {
                        ErrorDetails::other(format!(
                            "invalid media redirect from {current:?}: {error}"
                        ))
                    })?
                    .into();
                continue;
            }
            if let Some(error) = record.error {
                return Err(ErrorDetails::other(format!(
                    "failed to download media URL {current:?}: {}",
                    error.message
                )));
            }
            if record.status != Some(200) {
                return Err(ErrorDetails::http(
                    record.status.unwrap_or(500),
                    format!("media URL {current:?} did not return HTTP 200"),
                ));
            }
            return record
                .responses
                .into_iter()
                .find_map(|response| match response {
                    Response::Text(response) => Some(FetchedMedia {
                        bytes: response.body,
                        content_type: response.content_type,
                    }),
                    Response::Sse(_) => None,
                })
                .ok_or_else(|| {
                    ErrorDetails::other(format!("media URL {current:?} returned no body"))
                });
        }
        Err(ErrorDetails::other(format!(
            "media URL {initial:?} exceeded the redirect limit"
        )))
    }
}

/// PNG/JPEG validator and data-URL encoder used by image retrieval.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageDataUrlEncoder;

impl InlineMediaEncoder for ImageDataUrlEncoder {
    fn encode(&self, fetched: &FetchedMedia) -> Result<String, ErrorDetails> {
        let format = image::guess_format(&fetched.bytes).map_err(|error| {
            ErrorDetails::other(format!(
                "failed to determine downloaded image format: {error}"
            ))
        })?;
        let subtype = match format {
            image::ImageFormat::Png => "png",
            image::ImageFormat::Jpeg => "jpeg",
            other => {
                return Err(ErrorDetails::other(format!(
                    "downloaded image format {other:?} is unsupported; expected PNG or JPEG"
                )));
            }
        };
        image::load_from_memory_with_format(&fetched.bytes, format).map_err(|error| {
            ErrorDetails::other(format!("downloaded image is invalid: {error}"))
        })?;
        Ok(format!(
            "data:image/{subtype};base64,{}",
            STANDARD.encode(&fetched.bytes)
        ))
    }
}

/// Download and replace every flat `image_url` part containing an HTTP(S) URL.
///
/// Duplicate URLs are fetched once and replacement order is deterministic.
pub async fn inline_image_urls(
    payload: &mut Value,
    fetcher: &dyn MediaFetcher,
    encoder: &dyn InlineMediaEncoder,
) -> Result<(), ErrorDetails> {
    let mut urls = BTreeSet::new();
    collect_image_urls(payload, &mut urls);
    let mut replacements = BTreeMap::new();
    for url in urls {
        let fetched = fetcher.fetch(&url).await?;
        replacements.insert(url, encoder.encode(&fetched)?);
    }
    replace_image_urls(payload, &replacements);
    Ok(())
}

fn collect_image_urls(value: &Value, urls: &mut BTreeSet<String>) {
    match value {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("image_url")
                && let Some(url) = object.get("url").and_then(Value::as_str)
                && is_http_url(url)
            {
                urls.insert(url.to_string());
            }
            for value in object.values() {
                collect_image_urls(value, urls);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_image_urls(value, urls);
            }
        }
        _ => {}
    }
}

fn replace_image_urls(value: &mut Value, replacements: &BTreeMap<String, String>) {
    match value {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("image_url")
                && let Some(url) = object.get("url").and_then(Value::as_str)
                && let Some(replacement) = replacements.get(url)
            {
                object.insert("url".into(), Value::String(replacement.clone()));
            }
            for value in object.values_mut() {
                replace_image_urls(value, replacements);
            }
        }
        Value::Array(values) => {
            for value in values {
                replace_image_urls(value, replacements);
            }
        }
        _ => {}
    }
}

fn is_http_url(value: &str) -> bool {
    url::Url::parse(value)
        .ok()
        .is_some_and(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use image::{ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;

    struct StubFetcher {
        calls: Cell<usize>,
        bytes: Bytes,
    }

    #[async_trait(?Send)]
    impl MediaFetcher for StubFetcher {
        async fn fetch(&self, _url: &str) -> Result<FetchedMedia, ErrorDetails> {
            self.calls.set(self.calls.get() + 1);
            Ok(FetchedMedia {
                bytes: self.bytes.clone(),
                content_type: Some("image/png".into()),
            })
        }
    }

    #[tokio::test]
    async fn inline_images_deduplicates_fetches_and_preserves_data_urls() {
        let mut png = Cursor::new(Vec::new());
        ImageBuffer::<Rgb<u8>, _>::from_pixel(1, 1, Rgb([1, 2, 3]))
            .write_to(&mut png, ImageFormat::Png)
            .unwrap();
        let fetcher = StubFetcher {
            calls: Cell::new(0),
            bytes: Bytes::from(png.into_inner()),
        };
        let mut payload = serde_json::json!({
            "input":[
                {"type":"image_url","url":"http://example.test/p.png"},
                {"type":"image_url","url":"http://example.test/p.png"},
                {"type":"image_url","url":"data:image/png;base64,AA=="}
            ]
        });
        inline_image_urls(&mut payload, &fetcher, &ImageDataUrlEncoder)
            .await
            .unwrap();
        assert_eq!(fetcher.calls.get(), 1);
        assert!(
            payload["input"][0]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        assert_eq!(payload["input"][0]["url"], payload["input"][1]["url"]);
        assert_eq!(payload["input"][2]["url"], "data:image/png;base64,AA==");
    }
}
