// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Static web-UI assets, embedded into the binary at build time.
//!
//! The whole `src/server/ui/` tree (the "best-of-both" Preact + Chart.js SPA) is
//! compiled into the executable via [`include_dir`], so `aiperf serve` needs no
//! external asset directory; the SPA's own deps are not vendored — `index.html`
//! resolves Preact from an import map and Chart.js from a CDN. Unknown non-API
//! paths fall back to `index.html` so the SPA's client-side router owns navigation.

use axum::http::{StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use include_dir::{Dir, include_dir};

/// The embedded UI tree (`rust/cli/src/server/ui`).
static UI: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/src/server/ui");

/// Fallback handler: serve an embedded UI asset for any non-API path, falling back
/// to `index.html` (SPA client routing). `/api/*` misses return 404.
pub async fn serve(uri: Uri) -> Response {
    if uri.path().starts_with("/api/") {
        return (
            StatusCode::NOT_FOUND,
            format!("no such API route: {}", uri.path()),
        )
            .into_response();
    }
    let rel = uri.path().trim_start_matches('/');
    // Root resolves to the app shell; use the resolved name for the content type so
    // `/` is `text/html`, not the extensionless default.
    let name = if rel.is_empty() { "index.html" } else { rel };
    match UI.get_file(name) {
        Some(f) => (
            [(header::CONTENT_TYPE, content_type(name))],
            f.contents().to_vec(),
        )
            .into_response(),
        // SPA fallback: an unknown path is a client route → serve the app shell.
        None => match UI.get_file("index.html") {
            Some(index) => (
                [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                index.contents().to_vec(),
            )
                .into_response(),
            None => (StatusCode::NOT_FOUND, "dashboard UI not embedded").into_response(),
        },
    }
}

/// Content type from a path's extension (small fixed table — the UI ships only these).
fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") | Some("map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("ttf") => "font/ttf",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}
