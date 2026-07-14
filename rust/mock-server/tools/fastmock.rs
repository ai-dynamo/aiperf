// Minimal blazing-fast OpenAI-ish mock: fixed streaming chat response.
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

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

fn main() {
    let port = std::env::args().nth(1).unwrap_or_else(|| "8131".into());
    let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"x\"}}]}\n\ndata: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", body.len());
    let chat: Arc<Vec<u8>> = Arc::new([head.as_bytes(), body].concat());

    let models = b"{\"object\":\"list\",\"data\":[{\"id\":\"mock-model\",\"object\":\"model\"}]}";
    let mhead = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", models.len());
    let models_resp: Arc<Vec<u8>> = Arc::new([mhead.as_bytes(), models.as_ref()].concat());

    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).unwrap();
    println!("fastmock listening on {port}");
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        stream.set_nodelay(true).ok();
        let chat = chat.clone();
        let models_resp = models_resp.clone();
        thread::spawn(move || {
            let mut buf = vec![0u8; 65536];
            let mut acc: Vec<u8> = Vec::with_capacity(65536);
            loop {
                let n = match stream.read(&mut buf) {
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
                        &models_resp
                    } else {
                        &chat
                    };
                    if stream.write_all(resp).is_err() {
                        return;
                    }
                    acc.drain(..total);
                }
            }
        });
    }
}
