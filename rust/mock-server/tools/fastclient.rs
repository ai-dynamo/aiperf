// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HTTP/1.1 load generator for fixed-response mock targets.
//
// Persistent connections, prebuilt request batches, and fixed response framing
// avoid per-request allocation and parsing in the hot loop.
//
// Usage: fastclient URL [--connections C] [--duration S] [--pipeline P]
//   URL            e.g. http://127.0.0.1:8131/v1/chat/completions
//   --connections  persistent keep-alive connections / threads (default 128)
//   --duration     seconds to sustain load (default 10)
//   --pipeline     in-flight requests per connection per round-trip (default 1)
//
// Pipeline depth 1 measures ordinary HTTP/1.1 round trips. Higher depths remove
// round-trip waits and batch syscalls, so report them only as server capacity.
//
// Assumes a UNIFORM fixed-length response (true for fastmock / fastmock-uring):
// response framing is by the probed byte length L, not a per-response parser.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

fn content_length(head: &[u8]) -> usize {
    let s = String::from_utf8_lossy(head).to_lowercase();
    for line in s.split("\r\n") {
        if let Some(v) = line.strip_prefix("content-length:") {
            return v.trim().parse().unwrap_or(0);
        }
    }
    0
}

fn parse_url(url: &str) -> (String, u16, String) {
    let rest = url.strip_prefix("http://").unwrap_or(url);
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(80)),
        None => (authority.to_string(), 80),
    };
    (host, port, path.to_string())
}

/// One request → read the full response, returning its total byte length so the
/// hot loop can frame responses by length instead of parsing each one.
fn probe_response_len(addr: &str, req: &[u8]) -> std::io::Result<usize> {
    let mut s = TcpStream::connect(addr)?;
    s.set_nodelay(true).ok();
    s.write_all(req)?;
    let mut acc: Vec<u8> = Vec::with_capacity(65536);
    let mut buf = [0u8; 65536];
    loop {
        let n = s.read(&mut buf)?;
        if n == 0 {
            break;
        }
        acc.extend_from_slice(&buf[..n]);
        if let Some(hpos) = find(&acc, b"\r\n\r\n") {
            let total = hpos + 4 + content_length(&acc[..hpos]);
            if acc.len() >= total {
                return Ok(total);
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "probe: server closed before a full response",
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut url = "http://127.0.0.1:8131/v1/chat/completions".to_string();
    let mut connections = 128usize;
    let mut duration = 10u64;
    let mut pipeline = 1usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--connections" | "-c" => {
                i += 1;
                connections = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(connections);
            }
            "--duration" | "-d" => {
                i += 1;
                duration = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(duration);
            }
            "--pipeline" | "-p" => {
                i += 1;
                pipeline = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(pipeline);
            }
            u if !u.starts_with('-') => url = u.to_string(),
            _ => {}
        }
        i += 1;
    }
    let connections = connections.max(1);
    let pipeline = pipeline.max(1);
    let (host, port, path) = parse_url(&url);
    let addr = format!("{host}:{port}");

    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{{}}"
    )
    .into_bytes();

    let resp_len = match probe_response_len(&addr, &req) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to reach {addr}: {e}");
            std::process::exit(1);
        }
    };

    let batch: Arc<Vec<u8>> = Arc::new(req.iter().cloned().cycle().take(req.len() * pipeline).collect());
    let read_target = resp_len * pipeline;

    println!(
        "fastclient -> {url}  connections={connections} pipeline={pipeline} duration={duration}s  (resp_len={resp_len}B)"
    );

    let stop = Arc::new(AtomicBool::new(false));
    let total_reqs = Arc::new(AtomicU64::new(0));
    let total_errs = Arc::new(AtomicU64::new(0));
    let latency_ns_sum = Arc::new(AtomicU64::new(0));
    let latency_batches = Arc::new(AtomicU64::new(0));

    let start = Instant::now();
    let mut handles = Vec::with_capacity(connections);
    for _ in 0..connections {
        let addr = addr.clone();
        let batch = batch.clone();
        let stop = stop.clone();
        let total_reqs = total_reqs.clone();
        let total_errs = total_errs.clone();
        let latency_ns_sum = latency_ns_sum.clone();
        let latency_batches = latency_batches.clone();
        handles.push(std::thread::spawn(move || {
            let mut stream = match TcpStream::connect(&addr) {
                Ok(s) => s,
                Err(_) => {
                    total_errs.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };
            stream.set_nodelay(true).ok();
            let mut buf = vec![0u8; read_target.max(65536)];
            let mut local: u64 = 0;
            let mut lat_sum: u64 = 0;
            let mut lat_n: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                let t0 = Instant::now();
                if stream.write_all(&batch).is_err() {
                    total_errs.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                let mut got = 0usize;
                let mut broken = false;
                let cap = buf.len();
                while got < read_target {
                    let want = (read_target - got).min(cap);
                    match stream.read(&mut buf[..want]) {
                        Ok(0) => {
                            broken = true;
                            break;
                        }
                        Ok(n) => got += n,
                        Err(_) => {
                            broken = true;
                            break;
                        }
                    }
                }
                if broken {
                    total_errs.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                local += pipeline as u64;
                lat_sum += t0.elapsed().as_nanos() as u64;
                lat_n += 1;
            }
            total_reqs.fetch_add(local, Ordering::Relaxed);
            latency_ns_sum.fetch_add(lat_sum, Ordering::Relaxed);
            latency_batches.fetch_add(lat_n, Ordering::Relaxed);
        }));
    }

    std::thread::sleep(Duration::from_secs(duration));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let reqs = total_reqs.load(Ordering::Relaxed);
    let errs = total_errs.load(Ordering::Relaxed);
    let lat_sum = latency_ns_sum.load(Ordering::Relaxed);
    let batches = latency_batches.load(Ordering::Relaxed).max(1);
    // Mean per-request latency = mean batch latency / pipeline depth.
    let per_req_us = (lat_sum as f64 / batches as f64) / 1000.0 / pipeline as f64;

    println!("elapsed: {elapsed:.2}s");
    println!(
        "requests: {reqs}  errors: {errs}  RPS: {:.0}",
        reqs as f64 / elapsed
    );
    println!("mean latency/req: {per_req_us:.1}us  (pipeline depth {pipeline})");
}
