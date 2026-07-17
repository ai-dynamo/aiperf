// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// io_uring thread-per-core fixed-response server for I/O microbenchmarks.
//
// Each monoio runtime owns an SO_REUSEPORT listener on the same port, isolating
// the I/O engine as the benchmark variable.
//
// Usage: fastmock-uring [PORT] [--cores N]   (N default = available parallelism)
use bytes::Bytes;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::{ListenerOpts, TcpListener, TcpStream};

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

fn build_responses() -> (Bytes, Bytes) {
    let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"x\"}}]}\n\ndata: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", body.len());
    let chat = Bytes::from([head.as_bytes(), body].concat());

    let models = b"{\"object\":\"list\",\"data\":[{\"id\":\"mock-model\",\"object\":\"model\"}]}";
    let mhead = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", models.len());
    let models_resp = Bytes::from([mhead.as_bytes(), models.as_ref()].concat());
    (chat, models_resp)
}

/// Use monoio-owned buffers so io_uring retains buffer ownership across I/O.
async fn handle(mut stream: TcpStream, chat: Bytes, models: Bytes) {
    let mut acc: Vec<u8> = Vec::with_capacity(65536);
    let mut buf: Vec<u8> = vec![0u8; 65536];
    loop {
        let (res, b) = stream.read(buf).await;
        buf = b;
        let n = match res {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        acc.extend_from_slice(&buf[..n]);
        loop {
            let Some(hpos) = find(&acc, b"\r\n\r\n") else { break };
            let head = &acc[..hpos];
            let cl = if head.starts_with(b"GET") {
                0
            } else {
                content_length(head)
            };
            let total = hpos + 4 + cl;
            if acc.len() < total {
                break;
            }
            let resp = if head.starts_with(b"GET") {
                models.clone()
            } else {
                chat.clone()
            };
            let (wres, _) = stream.write_all(resp).await;
            if wres.is_err() {
                return;
            }
            acc.drain(..total);
        }
    }
}

async fn serve(port: u16, chat: Bytes, models: Bytes) {
    let addr = format!("127.0.0.1:{port}");
    let opts = ListenerOpts::default().reuse_port(true).reuse_addr(true);
    let listener = TcpListener::bind_with_config(addr.as_str(), &opts)
        .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let _ = stream.set_nodelay(true);
                monoio::spawn(handle(stream, chat.clone(), models.clone()));
            }
            Err(_) => continue,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut port: u16 = 8131;
    let mut cores: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--cores" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse().ok()) {
                    cores = v;
                }
            }
            p if !p.starts_with("--") => {
                if let Ok(v) = p.parse() {
                    port = v;
                }
            }
            _ => {}
        }
        i += 1;
    }
    let cores = cores.max(1);
    let (chat, models) = build_responses();
    println!("fastmock-uring: {cores} io_uring runtimes (SO_REUSEPORT) on {port}");

    let mut handles = Vec::with_capacity(cores);
    for _ in 0..cores {
        let chat = chat.clone();
        let models = models.clone();
        handles.push(std::thread::spawn(move || {
            let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
                .with_entries(4096)
                .enable_timer()
                .build()
                .expect("build io_uring runtime");
            rt.block_on(serve(port, chat, models));
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}
