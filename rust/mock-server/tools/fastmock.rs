// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal fixed-response OpenAI-compatible mock.
//
// Compile standalone (no cargo, no crates):  rustc -O fastmock.rs -o /tmp/fastmock
//
// Usage: fastmock [PORT] [--threads M] [--procs N] [--uds PATH]
//   PORT        listen port (default 8131); ignored when --uds is given
//   --threads M M concurrent accept threads sharing one listener (default 1;
//               0 = auto = available parallelism). Lifts the single-accept-loop
//               ceiling with zero added latency — no proxy, same process.
//   --procs N   N independent server processes sharing PORT via SO_REUSEPORT
//               (default 1; 0 = auto = available parallelism). The kernel spreads
//               new connections across the processes without a proxy hop.
//               NOT supported with --uds (SO_REUSEPORT-for-AF_UNIX is skipped
//               here to keep the raw-syscall bind path TCP-only); --procs is
//               forced to 1 when --uds is set.
//   --uds PATH  listen on a Unix domain socket at PATH instead of TCP. Any
//               stale socket file at PATH is removed before binding.
//   --mode M    UDS-only run mode, chosen once at startup (never re-checked
//               per request/loop-iteration) (default: blocking):
//                 blocking  thread-per-connection, blocking read()/write().
//                 epoll     N worker threads, each with its own epoll
//                           instance multiplexing many connections, so one
//                           epoll_wait() can report many ready fds instead
//                           of needing one parked thread per connection.
//               Both modes block/park while idle (epoll_wait with a bounded
//               timeout, blocking read() otherwise) — no mode here spins a
//               CPU core at 100% waiting for data.
//
// --threads and --procs compose: `--procs 4 --threads 2` = 4 processes, each with
// 2 accept threads.
use std::io::{Read, Write};
use std::net::TcpListener;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}
/// Byte-level, allocation-free header scan. The naive version this replaced
/// (`String::from_utf8_lossy(head).to_lowercase()`) allocated two Strings on
/// every single request — at millions of req/s that's millions of heap
/// allocations/sec of pure overhead versus this zero-alloc version.
fn content_length(head: &[u8]) -> usize {
    for line in head.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(colon) = line.iter().position(|&b| b == b':') else {
            continue;
        };
        let (name, val) = line.split_at(colon);
        if name.eq_ignore_ascii_case(b"content-length") {
            let val = std::str::from_utf8(&val[1..]).unwrap_or("").trim();
            return val.parse().unwrap_or(0);
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

/// Parses and answers every complete request found in `chunk` starting at
/// offset 0. Returns the byte offset past the last complete request handled
/// (== chunk.len() when nothing is left over). Shared by the zero-copy fast
/// path and the accumulator fallback below — same parse logic, different
/// backing buffer.
fn drain_requests<S: Write>(
    chunk: &[u8],
    stream: &mut S,
    chat: &Arc<Vec<u8>>,
    models: &Arc<Vec<u8>>,
) -> Result<usize, ()> {
    let mut off = 0usize;
    loop {
        let rest = &chunk[off..];
        let Some(hpos) = find(rest, b"\r\n\r\n") else {
            break;
        };
        let head = &rest[..hpos];
        let cl = if head.starts_with(b"GET") {
            0
        } else {
            content_length(head)
        };
        let total = hpos + 4 + cl;
        if rest.len() < total {
            break;
        }
        let resp = if head.starts_with(b"GET") { models } else { chat };
        if stream.write_all(resp).is_err() {
            return Err(());
        }
        off += total;
    }
    Ok(off)
}

fn handle<S: Read + Write>(mut stream: S, chat: Arc<Vec<u8>>, models: Arc<Vec<u8>>) {
    let mut buf = vec![0u8; 65536];
    // Only populated when a request spans more than one `read()` call (rare
    // at pipeline depth 1: the client waits for a response before sending
    // its next request, so a read almost always contains exactly one
    // complete request already). Keeping the fast path copy-free avoids a
    // Vec extend + drain per request in the overwhelmingly common case.
    let mut acc: Vec<u8> = Vec::new();
    loop {
        let n = match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        if acc.is_empty() {
            let off = match drain_requests(&buf[..n], &mut stream, &chat, &models) {
                Ok(off) => off,
                Err(()) => return,
            };
            if off < n {
                acc.extend_from_slice(&buf[off..n]);
            }
        } else {
            acc.extend_from_slice(&buf[..n]);
            let off = match drain_requests(&acc, &mut stream, &chat, &models) {
                Ok(off) => off,
                Err(()) => return,
            };
            acc.drain(..off);
        }
    }
}

/// Minimal raw `epoll` bindings (std exposes no epoll API). Layout matches
/// the kernel/glibc `struct epoll_event`, which is `packed` on x86_64/aarch64
/// (natural alignment would otherwise insert padding the kernel doesn't
/// expect).
mod raw_epoll {
    #[repr(C, packed)]
    pub struct EpollEvent {
        pub events: u32,
        pub data: u64,
    }
    pub const EPOLL_CTL_ADD: i32 = 1;
    pub const EPOLL_CTL_DEL: i32 = 2;
    pub const EPOLLIN: u32 = 0x001;
    extern "C" {
        pub fn epoll_create1(flags: i32) -> i32;
        pub fn epoll_ctl(epfd: i32, op: i32, fd: i32, event: *mut EpollEvent) -> i32;
        pub fn epoll_wait(epfd: i32, events: *mut EpollEvent, maxevents: i32, timeout: i32) -> i32;
    }
}

/// One epoll worker: owns an epoll instance and a private slice of
/// connections (assigned round-robin by the acceptor via `new_conns`). A
/// single `epoll_wait` call can report many ready connections at once,
/// unlike the blocking model where each ready connection needs its own
/// parked OS thread to wake.
fn epoll_worker(new_conns: mpsc::Receiver<UnixStream>, resp: Arc<Responses>) {
    use raw_epoll::*;
    let epfd = unsafe { epoll_create1(0) };
    assert!(epfd >= 0, "epoll_create1 failed");

    let mut conns: std::collections::HashMap<RawFd, (UnixStream, Vec<u8>)> =
        std::collections::HashMap::new();
    let mut events: Vec<EpollEvent> = (0..1024).map(|_| EpollEvent { events: 0, data: 0 }).collect();
    let mut scratch = vec![0u8; 65536];

    loop {
        // Drain newly-assigned connections (non-blocking check) before each
        // wait so a burst of new connects doesn't wait a full epoll_wait
        // timeout to start being served.
        while let Ok(stream) = new_conns.try_recv() {
            stream.set_nonblocking(true).ok();
            let fd = stream.as_raw_fd();
            let mut ev = EpollEvent { events: EPOLLIN, data: fd as u64 };
            let rc = unsafe { epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &mut ev) };
            if rc == 0 {
                conns.insert(fd, (stream, Vec::new()));
            }
        }

        // 20ms timeout so a quiet worker still periodically checks
        // `new_conns` rather than only waking on socket readiness.
        let n = unsafe { epoll_wait(epfd, events.as_mut_ptr(), events.len() as i32, 20) };
        if n < 0 {
            continue;
        }
        for ev in &events[..n as usize] {
            let fd = ev.data as RawFd;
            let Some((stream, acc)) = conns.get_mut(&fd) else {
                continue;
            };
            let mut closed = false;
            loop {
                match stream.read(&mut scratch) {
                    Ok(0) => {
                        closed = true;
                        break;
                    }
                    Ok(n) => {
                        acc.extend_from_slice(&scratch[..n]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(_) => {
                        closed = true;
                        break;
                    }
                }
            }
            if !closed && !acc.is_empty() {
                match drain_requests(acc, stream, &resp.chat, &resp.models) {
                    Ok(off) => {
                        acc.drain(..off);
                    }
                    Err(()) => closed = true,
                }
            }
            if closed {
                let mut ev = EpollEvent { events: 0, data: 0 };
                unsafe { epoll_ctl(epfd, EPOLL_CTL_DEL, fd, &mut ev) };
                conns.remove(&fd);
            }
        }
    }
}

fn serve_unix_epoll(listener: UnixListener, workers: usize) {
    let resp = Arc::new(build_responses());
    let workers = workers.max(1);
    let mut senders = Vec::with_capacity(workers);
    for _ in 0..workers {
        let (tx, rx) = mpsc::channel::<UnixStream>();
        let resp = resp.clone();
        thread::spawn(move || epoll_worker(rx, resp));
        senders.push(tx);
    }
    let mut next = 0usize;
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let _ = senders[next % workers].send(stream);
        next = next.wrapping_add(1);
    }
}

/// The kernel serializes concurrent `accept` calls on the shared listener.
fn accept_loop_tcp(listener: Arc<TcpListener>, resp: &Responses) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        stream.set_nodelay(true).ok();
        let chat = resp.chat.clone();
        let models = resp.models.clone();
        thread::spawn(move || handle(stream, chat, models));
    }
}

fn accept_loop_unix(listener: Arc<UnixListener>, resp: &Responses) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let chat = resp.chat.clone();
        let models = resp.models.clone();
        thread::spawn(move || handle(stream, chat, models));
    }
}

fn serve_tcp(listener: TcpListener, threads: usize) {
    let listener = Arc::new(listener);
    let resp = Arc::new(build_responses());
    let threads = threads.max(1);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let l = listener.clone();
        let r = resp.clone();
        handles.push(thread::spawn(move || accept_loop_tcp(l, &r)));
    }
    for h in handles {
        let _ = h.join();
    }
}

/// Run mode, resolved once from `--mode` before any listener/thread is
/// created — never re-checked inside a hot loop.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Blocking,
    Epoll,
}

impl Mode {
    fn parse(s: &str) -> Self {
        match s {
            "blocking" => Mode::Blocking,
            "epoll" => Mode::Epoll,
            other => panic!("unknown --mode '{}' (expected blocking|epoll)", other),
        }
    }
}

fn serve_unix(listener: UnixListener, threads: usize, mode: Mode) {
    if mode == Mode::Epoll {
        serve_unix_epoll(listener, threads);
        return;
    }
    let listener = Arc::new(listener);
    let resp = Arc::new(build_responses());
    let threads = threads.max(1);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let l = listener.clone();
        let r = resp.clone();
        handles.push(thread::spawn(move || accept_loop_unix(l, &r)));
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
    let mut uds: Option<String> = None;
    let mut mode = Mode::Blocking;
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
            "--uds" => {
                i += 1;
                uds = args.get(i).cloned();
            }
            "--mode" => {
                i += 1;
                mode = Mode::parse(args.get(i).map(|s| s.as_str()).unwrap_or("blocking"));
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

    if let Some(sock_path) = &uds {
        if procs > 1 {
            eprintln!("fastmock: --procs is not supported with --uds, ignoring (forcing --procs 1)");
        }
        // Remove a stale socket file from a previous unclean shutdown; bind()
        // fails with EADDRINUSE against an existing path otherwise.
        let _ = std::fs::remove_file(sock_path);
        let listener = UnixListener::bind(sock_path)
            .unwrap_or_else(|e| panic!("bind({}) failed: {}", sock_path, e));
        let mode_desc = match mode {
            Mode::Blocking => format!("{threads} accept threads, blocking"),
            Mode::Epoll => format!("{threads} epoll workers"),
        };
        println!("fastmock listening on uds:{sock_path} ({mode_desc})");
        serve_unix(listener, threads, mode);
        return;
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
    serve_tcp(listener, threads);
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
