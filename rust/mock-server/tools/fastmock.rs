// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal fixed-response OpenAI-compatible mock.
//
// Compile standalone (no cargo, no crates):  rustc -O fastmock.rs -o /tmp/fastmock
//
// Usage: fastmock [PORT] [--threads M] [--procs N]
//   PORT        listen port (default 8131)
//   --threads M M concurrent accept threads sharing one listener (default 1;
//               0 = auto = available parallelism). Lifts the single-accept-loop
//               ceiling with zero added latency — no proxy, same process.
//   --procs N   N independent server processes sharing PORT via SO_REUSEPORT
//               (default 1; 0 = auto = available parallelism). The kernel spreads
//               new connections across the processes without a proxy hop.
//
// --threads and --procs compose: `--procs 4 --threads 2` = 4 processes, each with
// 2 accept threads.
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
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

struct Responses {
    chat: Arc<Vec<u8>>,
    models: Arc<Vec<u8>>,
}

fn build_responses() -> Responses {
    let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"x\"}}]}\n\ndata: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", body.len());
    let chat: Arc<Vec<u8>> = Arc::new([head.as_bytes(), body].concat());

    let models = b"{\"object\":\"list\",\"data\":[{\"id\":\"mock-model\",\"object\":\"model\"}]}";
    let mhead = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", models.len());
    let models_resp: Arc<Vec<u8>> = Arc::new([mhead.as_bytes(), models.as_ref()].concat());

    Responses {
        chat,
        models: models_resp,
    }
}

fn handle(mut stream: TcpStream, chat: Arc<Vec<u8>>, models: Arc<Vec<u8>>) {
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
                &models
            } else {
                &chat
            };
            if stream.write_all(resp).is_err() {
                return;
            }
            acc.drain(..total);
        }
    }
}

/// The kernel serializes concurrent `accept` calls on the shared listener.
fn accept_loop(listener: Arc<TcpListener>, resp: &Responses) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        stream.set_nodelay(true).ok();
        let chat = resp.chat.clone();
        let models = resp.models.clone();
        thread::spawn(move || handle(stream, chat, models));
    }
}

fn serve(listener: TcpListener, threads: usize) {
    let listener = Arc::new(listener);
    let resp = Arc::new(build_responses());
    let threads = threads.max(1);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let l = listener.clone();
        let r = resp.clone();
        handles.push(thread::spawn(move || accept_loop(l, &r)));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn auto_parallelism() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut port = "8131".to_string();
    let mut threads: usize = 1;
    let mut procs: usize = 1;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "--procs" => {
                i += 1;
                procs = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            p if !p.starts_with("--") => port = p.to_string(),
            _ => {}
        }
        i += 1;
    }
    if threads == 0 {
        threads = auto_parallelism();
    }
    if procs == 0 {
        procs = auto_parallelism();
    }

    let is_child = std::env::var_os("FASTMOCK_CHILD").is_some();
    if procs > 1 && !is_child {
        run_supervisor(&port, threads, procs);
        return;
    }

    let listener = if procs > 1 {
        bind_reuseport(&port)
    } else {
        TcpListener::bind(format!("127.0.0.1:{port}")).unwrap()
    };
    if is_child {
        println!("fastmock child serving on {port} ({threads} accept threads)");
    } else {
        println!("fastmock listening on {port} ({threads} accept threads)");
    }
    serve(listener, threads);
}

/// Supervise `procs` SO_REUSEPORT workers and kill them with their parent.
fn run_supervisor(port: &str, threads: usize, procs: usize) {
    use std::process::Command;
    let exe = std::env::current_exe().expect("current_exe");
    println!("fastmock balancer: {procs} SO_REUSEPORT processes on {port} ({threads} accept threads each)");
    let mut children = Vec::with_capacity(procs);
    for _ in 0..procs {
        let mut cmd = Command::new(&exe);
        cmd.env("FASTMOCK_CHILD", "1").args([
            port,
            "--threads",
            &threads.to_string(),
            "--procs",
            &procs.to_string(),
        ]);
        set_parent_death_signal(&mut cmd);
        match cmd.spawn() {
            Ok(c) => children.push(c),
            Err(e) => eprintln!("fastmock: failed to spawn child: {e}"),
        }
    }
    for mut c in children {
        let _ = c.wait();
    }
}

/// Set SO_REUSEPORT before bind because `TcpListener` exposes no post-bind API.
#[cfg(target_os = "linux")]
fn bind_reuseport(port: &str) -> TcpListener {
    use std::os::unix::io::FromRawFd;
    extern "C" {
        fn socket(domain: i32, ty: i32, protocol: i32) -> i32;
        fn setsockopt(
            fd: i32,
            level: i32,
            optname: i32,
            optval: *const core::ffi::c_void,
            optlen: u32,
        ) -> i32;
        fn bind(fd: i32, addr: *const u8, len: u32) -> i32;
        fn listen(fd: i32, backlog: i32) -> i32;
    }
    const AF_INET: i32 = 2;
    const SOCK_STREAM: i32 = 1;
    const SOL_SOCKET: i32 = 1;
    const SO_REUSEADDR: i32 = 2;
    const SO_REUSEPORT: i32 = 15;

    let port: u16 = port.parse().expect("port must be a number");
    unsafe {
        let fd = socket(AF_INET, SOCK_STREAM, 0);
        assert!(fd >= 0, "socket() failed");
        let one: i32 = 1;
        let optval = &one as *const i32 as *const core::ffi::c_void;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, optval, 4);
        assert!(
            setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, optval, 4) == 0,
            "SO_REUSEPORT failed"
        );
        // struct sockaddr_in (16 bytes, little-endian host):
        //   sin_family = AF_INET (u16), sin_port (u16, network order),
        //   sin_addr = 127.0.0.1 (4 bytes, network order), 8 bytes padding.
        let mut sa = [0u8; 16];
        sa[0] = AF_INET as u8;
        sa[2..4].copy_from_slice(&port.to_be_bytes());
        sa[4..8].copy_from_slice(&[127, 0, 0, 1]);
        assert!(
            bind(fd, sa.as_ptr(), 16) == 0,
            "bind() failed (port {} already taken?)",
            port
        );
        assert!(listen(fd, 1024) == 0, "listen() failed");
        TcpListener::from_raw_fd(fd)
    }
}

/// Without SO_REUSEPORT, a second process cannot bind the same address.
#[cfg(not(target_os = "linux"))]
fn bind_reuseport(port: &str) -> TcpListener {
    TcpListener::bind(format!("127.0.0.1:{port}")).unwrap()
}

/// Prevent workers from orphaning and retaining the shared port.
#[cfg(target_os = "linux")]
fn set_parent_death_signal(cmd: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            extern "C" {
                fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
            }
            const PR_SET_PDEATHSIG: i32 = 1;
            const SIGKILL: u64 = 9;
            if prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn set_parent_death_signal(_cmd: &mut std::process::Command) {}
