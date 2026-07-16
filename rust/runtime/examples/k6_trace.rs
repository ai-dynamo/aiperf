// rust/transport-http/examples/k6_trace.rs
//! Dump the full k6/HAR-style per-request trace for real requests via
//! `aiperf-transport-http`. Prints every connection/request phase the transport
//! timestamps off the Clock: blocked (conn-pool wait), DNS, connecting
//! (TCP+TLS), sending, waiting (TTFB), receiving, and total duration — plus
//! bytes/chunks, status, and socket endpoints.
//!
//! Usage: k6_trace <BASE_URL> <MODEL> [COUNT] [OSL]

use std::rc::Rc;

use aiperf_runtime::transport_http::RealClock;
use aiperf_runtime::transport_http::config::ClientConfig;
use aiperf_runtime::transport_http::models::{RequestConfig, Response, TraceData};
use aiperf_runtime::transport_http::transport::http_transport::HttpTransport;

/// First real-token time (ns from request start): the first SSE chunk carrying a
/// non-empty content (or reasoning_content) delta — not the first SSE message,
/// which may be a role-only chunk with no token payload.
fn first_token_ns(rec: &aiperf_runtime::transport_http::models::RequestRecord) -> Option<i64> {
    for r in &rec.responses {
        let Response::Sse(m) = r else { continue };
        let Some(d) = m.data() else { continue };
        if d == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(d) else {
            continue;
        };
        let choices = v.get("choices").and_then(|c| c.as_array());
        let has_token = choices
            .and_then(|c| c.first())
            .map(|c| {
                let delta = &c["delta"];
                let content = delta.get("content").and_then(|x| x.as_str()).unwrap_or("");
                let reasoning = delta
                    .get("reasoning_content")
                    .and_then(|x| x.as_str())
                    .unwrap_or("");
                !content.is_empty() || !reasoning.is_empty()
            })
            .unwrap_or(false);
        if has_token {
            return Some(m.perf_ns - rec.start_ns);
        }
    }
    None
}

fn ms(ns: Option<i64>) -> String {
    match ns {
        Some(v) => format!("{:.3} ms", v as f64 / 1_000_000.0),
        None => "-".to_string(),
    }
}

fn print_trace(i: usize, status: Option<u16>, t: &TraceData) {
    println!("─── request {i}  status={:?} ───", status);
    // k6 http_req_* names mapped to the transport's TraceData accessors.
    println!(
        "  http_req_blocked    (conn-pool wait) : {}",
        ms(t.blocked())
    );
    println!(
        "  http_req_looking_up (dns lookup)     : {}",
        ms(t.dns_lookup())
    );
    println!(
        "  http_req_connecting (tcp+tls)        : {}",
        ms(t.connecting())
    );
    println!(
        "  http_req_tcp        (tcp connect)    : {}",
        ms(t.tcp_connect())
    );
    println!(
        "  http_req_tls        (tls handshake)  : {}",
        ms(t.tls_handshake())
    );
    println!(
        "  http_req_sending    (request write)  : {}",
        ms(t.sending())
    );
    println!(
        "  http_req_ttfh       (to first header): {}",
        ms(t.time_to_first_header())
    );
    println!(
        "  http_req_waiting    (ttfb, 1st body) : {}",
        ms(t.waiting())
    );
    println!(
        "  http_req_receiving  (body transfer)  : {}",
        ms(t.receiving())
    );
    println!(
        "  http_req_duration   (total)          : {}",
        ms(t.duration())
    );
    println!(
        "  bytes   req={} resp={}   chunks req={} resp={}",
        t.request_bytes_total,
        t.response_bytes_total,
        t.request_chunks_count,
        t.response_chunks_count
    );
    println!(
        "  socket  local={}:{}  remote={}:{}  reused={}",
        t.local_ip.as_deref().unwrap_or("-"),
        t.local_port
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into()),
        t.remote_ip.as_deref().unwrap_or("-"),
        t.remote_port
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into()),
        t.connection_reused_ns.is_some(),
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let base = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());
    let model = args.next().unwrap_or_else(|| "model".to_string());
    let count: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(3);
    let osl: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move {
        let clock: Rc<dyn aiperf_runtime::transport_http::Clock> = RealClock::new();
        // Pooled reuse (default) so the 2nd+ requests show blocked/connecting≈0.
        let t = HttpTransport::new(clock, ClientConfig::default());
        let cfg = RequestConfig::new(format!("{base}/v1/chat/completions"));

        for i in 0..count {
            let payload = serde_json::json!({
                "model": model,
                "stream": true,
                "stream_options": {"include_usage": true},
                "max_tokens": osl,
                "messages": [{"role": "user", "content": "In one sentence, what is NVIDIA Dynamo?"}],
            });
            let rec = t.send_request(&cfg, payload, true, |_| {}).await;
            match &rec.trace {
                Some(tr) => print_trace(i, rec.status, tr),
                None => println!("request {i}: no trace (error={:?})", rec.error),
            }
            println!(
                "  ttft (1st REAL token)                : {}   sse_messages={}   error={:?}",
                ms(first_token_ns(&rec)),
                rec.responses.len(),
                rec.error
            );
            println!();
        }
    });
}
