// crates/aiperf-core/examples/fast_sse.rs
//! A minimal, ultra-cheap OpenAI-compatible streaming server for load-testing
//! the *client* (aiperf graph + transport) without the co-located mock eating
//! cores. Returns a fixed pre-rendered SSE body per request — a stand-in for a
//! fast / remote inference server. Serves both HTTP/1.1 and h2c (prior
//! knowledge) via hyper's auto connection builder.
//!
//! Usage: PORT=18830 cargo run --release -p aiperf-core --example fast_sse

use axum::routing::{get, post};
use axum::{Router, http::header, response::IntoResponse};

// One content token then [DONE] — the shape the transport sink parses.
const BODY: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n";

async fn chat() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/event-stream")], BODY)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/health", get(|| async { "ok" }));

    // UDS_PATH -> listen on a Unix-domain socket (bypasses the TCP/IP loopback
    // softirq tax for co-located high-throughput benchmarking). Else TCP.
    if let Ok(sock) = std::env::var("UDS_PATH") {
        let _ = std::fs::remove_file(&sock);
        let listener = tokio::net::UnixListener::bind(&sock).unwrap();
        eprintln!("fast_sse listening on unix:{sock}");
        axum::serve(listener, app).await.unwrap();
    } else {
        let port = std::env::var("PORT").unwrap_or_else(|_| "18830".to_string());
        let addr = format!("127.0.0.1:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        eprintln!("fast_sse listening on http://{addr}");
        axum::serve(listener, app).await.unwrap();
    }
}
