// crates/aiperf-transport/examples/k6_trace.rs
//! Dump the full k6/HAR-style per-request trace for real requests via
//! aiperf-transport. Prints every connection/request phase the transport
//! timestamps off the Clock: blocked (conn-pool wait), DNS, connecting
//! (TCP+TLS), sending, waiting (TTFB), receiving, and total duration — plus
//! bytes/chunks, status, and socket endpoints.
//!
//! Usage: k6_trace <BASE_URL> <MODEL> [COUNT] [OSL]

use std::rc::Rc;

use aiperf_transport::RealClock;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::{RequestConfig, TraceData};
use aiperf_transport::transport::http_transport::HttpTransport;

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
        "  http_req_sending    (request write)  : {}",
        ms(t.sending())
    );
    println!(
        "  http_req_waiting    (ttfb)           : {}",
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
        let clock: Rc<dyn aiperf_transport::Clock> = RealClock::new();
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
            let mut ttft_ns: Option<i64> = None;
            let rec = t.send_request(&cfg, payload, true, |d| ttft_ns = Some(d)).await;
            match &rec.trace {
                Some(tr) => print_trace(i, rec.status, tr),
                None => println!("request {i}: no trace (error={:?})", rec.error),
            }
            println!(
                "  ttft (first-token delta)             : {}   sse_messages={}   error={:?}",
                ms(ttft_ns),
                rec.responses.len(),
                rec.error
            );
            println!();
        }
    });
}
