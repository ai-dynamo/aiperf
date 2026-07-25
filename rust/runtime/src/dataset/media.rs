// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multimodal content resolution for loaders.
//!
//! URLs and already-encoded values pass through byte-identically. Local image,
//! audio, and video files are validated and encoded once at composition time,
//! before the resulting bytes enter the content-addressed store.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread::JoinHandle;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;

use crate::clock::RealClock;
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::model::MediaKind;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::transport::http_transport::HttpTransport;
use crate::transport::http::transport::inline_media::{
    HttpMediaFetcher, ImageDataUrlEncoder, InlineMediaEncoder, MediaFetcher,
};

/// Pluggable conversion from an authored media reference to endpoint-ready bytes.
pub trait MediaResolver: Send + Sync {
    /// Resolve one URL, encoded value, or local path.
    fn resolve(&self, kind: MediaKind, authored: &str) -> Result<Bytes>;
}

/// Default resolver that inlines local files and preserves URLs/encoded values.
#[derive(Debug, Clone, Copy, Default)]
pub struct InlineMediaResolver;

impl MediaResolver for InlineMediaResolver {
    fn resolve(&self, kind: MediaKind, authored: &str) -> Result<Bytes> {
        if kind == MediaKind::Text || is_already_encoded(kind, authored) {
            return Ok(Bytes::copy_from_slice(authored.as_bytes()));
        }
        if authored.contains("://") {
            let url = url::Url::parse(authored).map_err(|error| {
                DatasetError::Validation(format!("invalid media URL {authored:?}: {error}"))
            })?;
            if url.scheme().is_empty() || url.host_str().is_none() {
                return Err(DatasetError::Validation(format!(
                    "media URL must have a scheme and host: {authored:?}"
                )));
            }
            return Ok(Bytes::copy_from_slice(authored.as_bytes()));
        }

        let path = Path::new(authored);
        let raw = std::fs::read(path).map_err(|error| {
            DatasetError::Io(std::io::Error::new(
                error.kind(),
                format!("failed to read media file {}: {error}", path.display()),
            ))
        })?;
        let encoded = match kind {
            MediaKind::Text => authored.to_string(),
            MediaKind::Image => encode_image(path, &raw)?,
            MediaKind::Audio => encode_audio(path, &raw)?,
            MediaKind::Video => encode_video(path, &raw)?,
        };
        Ok(Bytes::from(encoded))
    }
}

/// One prefetch request: a remote image URL and the reply channel that receives
/// the encoded `data:` URL (or an error message).
type PrefetchRequest = (String, mpsc::Sender<std::result::Result<String, String>>);

/// Media resolver that fetches remote `http(s)://` image URLs at dataset
/// generation time and replaces them with inline `data:` URLs.
///
/// This is the opt-in (`--prefetch-media-urls`) counterpart to
/// [`InlineMediaResolver`]. It exists for servers that cannot resolve image URLs
/// themselves: fetching happens once, up front, before any credits are issued —
/// never on the dispatch hot path. Local files, `data:` URLs, and non-image
/// media delegate to [`InlineMediaResolver`] unchanged.
///
/// [`MediaResolver::resolve`] is synchronous, but fetching is asynchronous, so
/// the actual downloads run on a single background thread owning its own
/// current-thread Tokio runtime and a `!Send` [`HttpTransport`]. `resolve`
/// blocks on that thread's reply for the duration of one fetch. Repeated URLs
/// are deduplicated through a shared cache so each distinct URL is fetched once.
pub struct PrefetchMediaResolver {
    /// Delegate for local files, `data:` URLs, and non-image media.
    inner: InlineMediaResolver,
    /// URL → encoded `data:` URL cache; dedups repeated fetches.
    cache: Mutex<HashMap<String, Bytes>>,
    /// Channel to the background fetcher thread. `None` after `Drop` begins.
    request_tx: Option<mpsc::Sender<PrefetchRequest>>,
    /// Background fetcher thread handle, joined on `Drop`.
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for PrefetchMediaResolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PrefetchMediaResolver")
            .finish_non_exhaustive()
    }
}

impl Default for PrefetchMediaResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefetchMediaResolver {
    /// Spawn the background fetcher thread and return a ready resolver.
    pub fn new() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<PrefetchRequest>();
        let worker = std::thread::Builder::new()
            .name("aiperf-media-prefetch".into())
            .spawn(move || fetcher_loop(&request_rx))
            .ok();
        Self {
            inner: InlineMediaResolver,
            cache: Mutex::new(HashMap::new()),
            request_tx: Some(request_tx),
            worker,
        }
    }

    /// Block on the background thread to fetch and encode one remote image URL.
    fn fetch_data_url(&self, url: &str) -> Result<Bytes> {
        if let Some(cached) = self.cache.lock().unwrap().get(url) {
            return Ok(cached.clone());
        }
        let request_tx = self.request_tx.as_ref().ok_or_else(|| {
            DatasetError::Validation("media prefetch worker is unavailable".into())
        })?;
        let (reply_tx, reply_rx) = mpsc::channel();
        request_tx.send((url.to_string(), reply_tx)).map_err(|_| {
            DatasetError::Validation("media prefetch worker has stopped".into())
        })?;
        let data_url = match reply_rx.recv() {
            Ok(Ok(data_url)) => data_url,
            Ok(Err(message)) => {
                return Err(DatasetError::Validation(format!(
                    "failed to prefetch media URL {url:?}: {message}"
                )));
            }
            Err(_) => {
                return Err(DatasetError::Validation(format!(
                    "media prefetch worker dropped the reply for {url:?}"
                )));
            }
        };
        let bytes = Bytes::from(data_url);
        self.cache
            .lock()
            .unwrap()
            .insert(url.to_string(), bytes.clone());
        Ok(bytes)
    }
}

impl MediaResolver for PrefetchMediaResolver {
    fn resolve(&self, kind: MediaKind, authored: &str) -> Result<Bytes> {
        // Only remote http(s) image URLs are prefetched; everything else
        // (local files, already-encoded values, non-image media) delegates.
        if kind == MediaKind::Image && !authored.starts_with("data:") && is_remote_http(authored) {
            return self.fetch_data_url(authored);
        }
        self.inner.resolve(kind, authored)
    }
}

impl Drop for PrefetchMediaResolver {
    fn drop(&mut self) {
        // Dropping the sender ends the fetcher loop's `recv`, so the thread exits.
        self.request_tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Background thread body: serve fetch requests until the sender is dropped.
fn fetcher_loop(request_rx: &mpsc::Receiver<PrefetchRequest>) {
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        // Without a runtime the loop cannot fetch; drain replies with errors.
        while let Ok((_, reply)) = request_rx.recv() {
            let _ = reply.send(Err("media prefetch runtime unavailable".into()));
        }
        return;
    };
    let local = tokio::task::LocalSet::new();
    local.block_on(&runtime, async move {
        // The transport is `!Send`; it lives entirely on this thread.
        let clock = RealClock::new();
        let transport = HttpTransport::new(clock, ClientConfig::default());
        let fetcher = HttpMediaFetcher::new(&transport);
        while let Ok((url, reply)) = request_rx.recv() {
            let result = match fetcher.fetch(&url).await {
                Ok(media) => ImageDataUrlEncoder
                    .encode(&media)
                    .map_err(|error| error.message),
                Err(error) => Err(error.message),
            };
            let _ = reply.send(result);
        }
    });
}

/// True when `value` parses as an absolute `http`/`https` URL with a host.
fn is_remote_http(value: &str) -> bool {
    url::Url::parse(value)
        .ok()
        .is_some_and(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
}

fn is_already_encoded(kind: MediaKind, content: &str) -> bool {
    match kind {
        MediaKind::Text => true,
        MediaKind::Image | MediaKind::Video => content.starts_with("data:"),
        MediaKind::Audio => content.split_once(',').is_some_and(|(format, _)| {
            !format.contains(':') && matches!(format.to_ascii_lowercase().as_str(), "wav" | "mp3")
        }),
    }
}

fn encode_image(path: &Path, raw: &[u8]) -> Result<String> {
    let format = image::guess_format(raw).map_err(|error| {
        DatasetError::Validation(format!(
            "failed to determine image format for {}: {error}",
            path.display()
        ))
    })?;
    let mime = match format {
        image::ImageFormat::Png => "png",
        image::ImageFormat::Jpeg => "jpeg",
        other => {
            return Err(DatasetError::Validation(format!(
                "unsupported image format {other:?} for {}; expected PNG or JPEG",
                path.display()
            )));
        }
    };
    image::load_from_memory_with_format(raw, format).map_err(|error| {
        DatasetError::Validation(format!("invalid image {}: {error}", path.display()))
    })?;
    Ok(format!("data:image/{mime};base64,{}", STANDARD.encode(raw)))
}

fn encode_audio(path: &Path, raw: &[u8]) -> Result<String> {
    let format = match extension(path) {
        "wav" => "wav",
        "mp3" => "mp3",
        other => {
            return Err(DatasetError::Validation(format!(
                "unsupported audio extension {other:?} for {}; expected wav or mp3",
                path.display()
            )));
        }
    };
    Ok(format!("{format},{}", STANDARD.encode(raw)))
}

fn encode_video(path: &Path, raw: &[u8]) -> Result<String> {
    let format = match extension(path) {
        "mp4" => "mp4",
        "webm" => "webm",
        other => {
            return Err(DatasetError::Validation(format!(
                "unsupported video extension {other:?} for {}; expected mp4 or webm",
                path.display()
            )));
        }
    };
    Ok(format!(
        "data:video/{format};base64,{}",
        STANDARD.encode(raw)
    ))
}

fn extension(path: &Path) -> &str {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{ImageBuffer, ImageFormat, Rgb};

    use super::*;

    #[test]
    fn urls_and_encoded_content_pass_through() {
        let resolver = InlineMediaResolver;
        for (kind, value) in [
            (MediaKind::Image, "https://example.com/a.png"),
            (MediaKind::Image, "data:image/png;base64,AA=="),
            (MediaKind::Audio, "wav,AA=="),
            (MediaKind::Video, "data:video/mp4;base64,AA=="),
        ] {
            assert_eq!(resolver.resolve(kind, value).unwrap(), value.as_bytes());
        }
    }

    #[test]
    fn local_png_is_validated_and_inlined() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pixel.png");
        ImageBuffer::<Rgb<u8>, _>::from_pixel(1, 1, Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();
        let encoded = InlineMediaResolver
            .resolve(MediaKind::Image, path.to_str().unwrap())
            .unwrap();
        assert!(encoded.starts_with(b"data:image/png;base64,"));
        let data = encoded.split(|byte| *byte == b',').nth(1).unwrap();
        let decoded = STANDARD.decode(data).unwrap();
        assert_eq!(
            image::guess_format(&decoded).unwrap(),
            image::ImageFormat::Png
        );
    }

    #[test]
    fn windows_style_paths_are_not_misclassified_as_urls() {
        let error = InlineMediaResolver
            .resolve(MediaKind::Image, r"C:\Users\missing.png")
            .unwrap_err();
        assert!(matches!(error, DatasetError::Io(_)));
    }

    /// Serve a single PNG on a dedicated OS thread with its own runtime so the
    /// synchronous, blocking [`PrefetchMediaResolver::resolve`] call cannot
    /// deadlock the server. Returns the bound address and a per-request hit
    /// counter.
    fn spawn_png_server() -> (std::net::SocketAddr, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        use axum::Router;
        use axum::http::header;
        use axum::response::IntoResponse;
        use axum::routing::get;

        let mut png = Cursor::new(Vec::new());
        ImageBuffer::<Rgb<u8>, _>::from_pixel(1, 1, Rgb([4, 5, 6]))
            .write_to(&mut png, ImageFormat::Png)
            .unwrap();
        let png = Bytes::from(png.into_inner());
        let hits = Arc::new(AtomicUsize::new(0));

        let (address_tx, address_rx) = mpsc::channel();
        let thread_hits = hits.clone();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let handler_hits = thread_hits.clone();
                let handler_png = png.clone();
                let app = Router::new().route(
                    "/asset.png",
                    get(move || {
                        let hits = handler_hits.clone();
                        let png = handler_png.clone();
                        async move {
                            hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            ([(header::CONTENT_TYPE, "image/png")], png).into_response()
                        }
                    }),
                );
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                address_tx.send(listener.local_addr().unwrap()).unwrap();
                axum::serve(listener, app).await.unwrap();
            });
        });
        (address_rx.recv().unwrap(), hits)
    }

    #[test]
    fn prefetch_resolver_downloads_dedups_and_delegates() {
        use std::sync::atomic::Ordering;

        let (address, hits) = spawn_png_server();
        let url = format!("http://{address}/asset.png");
        let resolver = PrefetchMediaResolver::new();

        // Remote http(s) image URLs are fetched and encoded to data URLs.
        let first = resolver.resolve(MediaKind::Image, &url).unwrap();
        assert!(first.starts_with(b"data:image/png;base64,"));
        // A repeated URL is served from the dedup cache: still one download.
        let second = resolver.resolve(MediaKind::Image, &url).unwrap();
        assert_eq!(first, second);
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // Already-encoded data URLs and non-image media pass through unfetched.
        let data_url = "data:image/png;base64,AA==";
        assert_eq!(
            resolver.resolve(MediaKind::Image, data_url).unwrap(),
            data_url.as_bytes()
        );
        assert_eq!(
            resolver.resolve(MediaKind::Text, "hello").unwrap(),
            b"hello".as_slice()
        );
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }
}
