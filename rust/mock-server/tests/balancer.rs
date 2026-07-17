// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire tests for the multi-process round-robin balancer.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// RAII wrapper so the spawned balancer (and, via `PR_SET_PDEATHSIG`, its
/// children) is always killed when the test ends, even on panic.
struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

/// Block until `GET /health` on `addr` returns HTTP 200, or panic after `timeout`.
fn wait_healthy(addr: SocketAddr, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if http_get(addr, "/health")
            .map(|(status, _)| status == 200)
            .unwrap_or(false)
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("balancer at {addr} never became healthy within {timeout:?}");
}

/// `Connection: close` permits reading the HTTP/1.0 response to EOF.
fn http_get(addr: SocketAddr, path: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    write!(
        stream,
        "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw)?;
    Ok(parse_response(&raw))
}

fn http_post_json(addr: SocketAddr, path: &str, body: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    write!(
        stream,
        "POST {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw)?;
    Ok(parse_response(&raw))
}

fn parse_response(raw: &str) -> (u16, String) {
    let status = raw
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let body = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (status, body)
}

fn spawn_balancer(processes: usize) -> (Server, SocketAddr) {
    let port = free_port();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_aiperf-mock-server"))
        .args([
            "--processes",
            &processes.to_string(),
            "--fast",
            "--no-tokenizer",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .spawn()
        .expect("spawn balancer");
    let server = Server(child);
    wait_healthy(addr, Duration::from_secs(30));
    (server, addr)
}

#[test]
fn balancer_serves_health_and_chat_through_one_port() {
    let (_server, addr) = spawn_balancer(3);

    let (status, body) = http_get(addr, "/health").unwrap();
    assert_eq!(status, 200, "health status");
    assert!(body.contains("healthy"), "health body: {body}");

    let payload =
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    for i in 0..30 {
        let (status, body) = http_post_json(addr, "/v1/chat/completions", payload)
            .unwrap_or_else(|e| panic!("request {i} failed: {e}"));
        assert_eq!(status, 200, "request {i} status; body={body}");
        assert!(
            body.contains("chat.completion"),
            "request {i} body missing completion object: {body}"
        );
    }
}

#[test]
fn balancer_distributes_connections_across_backends() {
    let (_server, addr) = spawn_balancer(4);
    let payload =
        r#"{"model":"mock-model","messages":[{"role":"user","content":"x"}],"max_tokens":2}"#;

    let mut handles = Vec::new();
    for _ in 0..64 {
        let payload = payload.to_string();
        handles.push(std::thread::spawn(move || {
            http_post_json(addr, "/v1/chat/completions", &payload).map(|(s, _)| s)
        }));
    }
    let mut ok = 0usize;
    for h in handles {
        if matches!(h.join().unwrap(), Ok(200)) {
            ok += 1;
        }
    }
    assert_eq!(
        ok, 64,
        "all 64 concurrent requests should succeed through the balancer"
    );
}

#[test]
fn single_process_serves_directly() {
    let (_server, addr) = spawn_balancer(1);
    let (status, _) = http_get(addr, "/health").unwrap();
    assert_eq!(status, 200);
}
