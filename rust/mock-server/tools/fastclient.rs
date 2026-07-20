// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HTTP/1.1 load generator for fixed-response mock targets.
//
// Persistent connections, prebuilt request batches, and fixed response framing
// avoid per-request allocation and parsing in the hot loop.
//
// Usage: fastclient URL [--connections C] [--duration S] [--pipeline P] [--uds PATH]
//   URL            e.g. http://127.0.0.1:8131/v1/chat/completions
//                  when --uds is given, only URL's path (and optional Host
//                  header override) is used; host:port is ignored.
//   --connections  persistent keep-alive connections / threads (default 128)
//   --duration     seconds to sustain load (default 10)
//   --pipeline     in-flight requests per connection per round-trip (default 1)
//   --uds PATH     connect over a Unix domain socket at PATH instead of TCP
//
// Pipeline depth 1 measures ordinary HTTP/1.1 round trips. Higher depths remove
// round-trip waits and batch syscalls, so report them only as server capacity.
//
// `--pipeline > 1` is NOT ALLOWED as a stand-in for real client RPS: real
// HTTP/1.1 clients issue one request per round trip and do not pipeline, so
// pipelined numbers do not represent achievable request rate against a real
// client population. Any RPS figure quoted from a `--pipeline > 1` run must be
// labeled "server capacity" / "retirement ceiling", never "RPS" or "throughput"
// on its own.
//
// Assumes a UNIFORM fixed-length response (true for fastmock / fastmock-uring):
// response framing is by the probed byte length L, not a per-response parser.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A load-generator connection over either transport. `handle`'s hot loop
/// (probe, per-connection threads) is transport-agnostic beyond connect/
/// TCP_NODELAY, so this is a thin dispatch rather than duplicated logic.
enum Conn {
    Tcp(TcpStream),
    Unix(UnixStream),
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.read(buf),
            Conn::Unix(s) => s.read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.write(buf),
            Conn::Unix(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Tcp(s) => s.flush(),
            Conn::Unix(s) => s.flush(),
        }
    }
}

/// Where to connect: a TCP `host:port` or a Unix domain socket path.
#[derive(Clone)]
enum Target {
    Tcp(String),
    Uds(String),
}

fn connect(target: &Target) -> std::io::Result<Conn> {
    match target {
        Target::Tcp(addr) => {
            let s = TcpStream::connect(addr)?;
            s.set_nodelay(true).ok();
            Ok(Conn::Tcp(s))
        }
        Target::Uds(path) => Ok(Conn::Unix(UnixStream::connect(path)?)),
    }
}

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

/// Extracts the HTTP status code from a response's start line
/// (`HTTP/1.1 <code> <reason>`). The hot loop never parses status — it just
/// frames by byte length — so this check exists solely at probe time to
/// catch "every request is actually a 4xx/5xx" before a whole benchmark run
/// silently measures the wrong thing.
fn status_code(head: &[u8]) -> Option<u16> {
    let line_end = find(head, b"\r\n").unwrap_or(head.len());
    let line = std::str::from_utf8(&head[..line_end]).ok()?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// One request → read the full response, returning its total byte length and
/// HTTP status code so the hot loop can frame responses by length instead of
/// parsing each one, and so the caller can catch a systematically-erroring
/// target before running a whole benchmark against it.
fn probe_response_len(target: &Target, req: &[u8]) -> std::io::Result<(usize, u16)> {
    let mut s = connect(target)?;
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
                let status = status_code(&acc[..hpos]).unwrap_or(0);
                return Ok((total, status));
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
    let mut uds: Option<String> = None;
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
            "--uds" => {
                i += 1;
                uds = args.get(i).cloned();
            }
            u if !u.starts_with('-') => url = u.to_string(),
            _ => {}
        }
        i += 1;
    }
    let connections = connections.max(1);
    let pipeline = pipeline.max(1);
    if pipeline > 1 {
        eprintln!(
            "\n\
             ########################################################################\n\
             ##  ERROR: --pipeline {pipeline} is NOT ALLOWED                                  \n\
             ##                                                                    ##\n\
             ##  Real HTTP/1.1 clients issue one request per round trip and do not ##\n\
             ##  pipeline. A pipelined run does not measure achievable client RPS  ##\n\
             ##  and its numbers must never be reported as \"RPS\" or \"throughput\".  ##\n\
             ##  Refusing to run. Use --pipeline 1 (the default) for real RPS, or  ##\n\
             ##  quote pipelined numbers ONLY as labeled server retirement-ceiling ##\n\
             ##  capacity via a modified build that removes this guard.           ##\n\
             ########################################################################\n"
        );
        std::process::exit(1);
    }
    let (host, port, path) = parse_url(&url);
    let addr = format!("{host}:{port}");
    let target = match &uds {
        Some(sock_path) => Target::Uds(sock_path.clone()),
        None => Target::Tcp(addr.clone()),
    };

    // A real, schema-valid OpenAI-compatible chat-completion body — not an
    // empty `{}`. `fastmock` never parses the body so this didn't matter
    // there, but `aiperf-mock-server`'s `Json<ChatCompletionRequest>`
    // extractor requires `model`+`messages` and 422s on anything else. An
    // empty body silently benchmarked the server's error-rejection path
    // instead of real request handling — `fastclient` doesn't check HTTP
    // status codes, so that failure mode was invisible until profiled.
    let body = br#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}]}"#;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        body.len()
    )
    .into_bytes();
    let req = [req, body.to_vec()].concat();

    let (resp_len, status) = match probe_response_len(&target, &req) {
        Ok(v) => v,
        Err(e) => {
            let dest = uds.as_deref().unwrap_or(&addr);
            eprintln!("failed to reach {dest}: {e}");
            std::process::exit(1);
        }
    };
    if !(200..300).contains(&status) {
        eprintln!(
            "\n\
             ########################################################################\n\
             ##  ERROR: probe request got HTTP {status}, not 2xx.                        \n\
             ##                                                                    ##\n\
             ##  fastclient's hot loop only measures byte-framing round trips — it ##\n\
             ##  does NOT check status codes per request. Benchmarking against a   ##\n\
             ##  target that errors on every request silently measures the error-  ##\n\
             ##  rejection path, not real request handling. Refusing to run.       ##\n\
             ########################################################################\n"
        );
        std::process::exit(1);
    }

    let batch: Arc<Vec<u8>> = Arc::new(req.iter().cloned().cycle().take(req.len() * pipeline).collect());
    let read_target = resp_len * pipeline;

    let dest_desc = uds.as_deref().map(|p| format!("uds:{p}")).unwrap_or_else(|| url.clone());
    println!(
        "fastclient -> {dest_desc}  connections={connections} pipeline={pipeline} duration={duration}s  (resp_len={resp_len}B)"
    );

    let stop = Arc::new(AtomicBool::new(false));
    let total_reqs = Arc::new(AtomicU64::new(0));
    let total_errs = Arc::new(AtomicU64::new(0));
    let latency_ns_sum = Arc::new(AtomicU64::new(0));
    let latency_batches = Arc::new(AtomicU64::new(0));
    // Log2-bucketed latency histogram (bucket i = [2^i, 2^(i+1)) ns), merged
    // across threads at the end for an approximate p50/p99/p999. Cheap and
    // bounded (64 buckets) vs. keeping every raw sample.
    const HIST_BUCKETS: usize = 64;
    let hist_buckets: Vec<Arc<AtomicU64>> =
        (0..HIST_BUCKETS).map(|_| Arc::new(AtomicU64::new(0))).collect();

    let start = Instant::now();
    let mut handles = Vec::with_capacity(connections);
    for _ in 0..connections {
        let target = target.clone();
        let batch = batch.clone();
        let stop = stop.clone();
        let total_reqs = total_reqs.clone();
        let total_errs = total_errs.clone();
        let latency_ns_sum = latency_ns_sum.clone();
        let latency_batches = latency_batches.clone();
        let hist_buckets = hist_buckets.clone();
        handles.push(std::thread::spawn(move || {
            let mut local_hist = [0u64; HIST_BUCKETS];
            let mut stream = match connect(&target) {
                Ok(s) => s,
                Err(_) => {
                    total_errs.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };
            // Sized to the actual expected response, not a fixed 64KB — the
            // old fixed size touched/cached 64KB per connection thread for a
            // 404B response, ~6MB of unnecessary working set across ~64-96
            // threads. `read()` never requests more than `read_target` bytes
            // anyway (see `want` below), so a larger buffer bought nothing.
            let mut buf = vec![0u8; read_target.max(1)];
            let mut local: u64 = 0;
            let mut lat_sum: u64 = 0;
            let mut lat_n: u64 = 0;
            // `stop` is only re-checked and only 1-in-SAMPLE_EVERY round trips
            // is timestamped: `Instant::now()`/`.elapsed()` (2 vDSO reads/req)
            // and the atomic `stop` load are real per-request overhead that
            // isn't needed every single iteration to get accurate RPS (every
            // request is still counted) and a statistically sound percentile
            // histogram (sampling 1/8 of many millions of requests/sec is
            // still a huge sample).
            const SAMPLE_EVERY: u64 = 32;
            let mut iter: u64 = 0;
            loop {
                iter += 1;
                if iter % SAMPLE_EVERY == 0 && stop.load(Ordering::Relaxed) {
                    break;
                }
                let sample = iter % SAMPLE_EVERY == 0;
                let t0 = if sample { Some(Instant::now()) } else { None };
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
                let Some(t0) = t0 else { continue };
                let ns = t0.elapsed().as_nanos() as u64;
                lat_sum += ns;
                lat_n += 1;
                // Bucket by the per-batch elapsed time; per-request p99 under
                // pipeline>1 is an approximation (batch latency, not per-req).
                let bucket = (64 - ns.max(1).leading_zeros() as usize - 1).min(HIST_BUCKETS - 1);
                local_hist[bucket] += 1;
            }
            total_reqs.fetch_add(local, Ordering::Relaxed);
            latency_ns_sum.fetch_add(lat_sum, Ordering::Relaxed);
            latency_batches.fetch_add(lat_n, Ordering::Relaxed);
            for (i, count) in local_hist.iter().enumerate() {
                if *count > 0 {
                    hist_buckets[i].fetch_add(*count, Ordering::Relaxed);
                }
            }
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

    let hist: Vec<u64> = hist_buckets.iter().map(|b| b.load(Ordering::Relaxed)).collect();
    let total: u64 = hist.iter().sum();
    let percentile = |p: f64| -> f64 {
        if total == 0 {
            return 0.0;
        }
        let target = (total as f64 * p).ceil() as u64;
        let mut cum = 0u64;
        for (i, &count) in hist.iter().enumerate() {
            cum += count;
            if cum >= target {
                // Midpoint of [2^i, 2^(i+1)) ns, in microseconds.
                let lo = (1u64 << i) as f64;
                let hi = (1u64 << (i + 1).min(63)) as f64;
                return (lo + hi) / 2.0 / 1000.0;
            }
        }
        0.0
    };
    println!(
        "latency/batch: p50={:.1}us  p90={:.1}us  p99={:.1}us  p999={:.1}us  (n={total}, pipeline depth {pipeline})",
        percentile(0.50),
        percentile(0.90),
        percentile(0.99),
        percentile(0.999),
    );
}
