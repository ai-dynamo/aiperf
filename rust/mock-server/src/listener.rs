// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared TCP listener construction.
//!
//! Both the single-process server (`main::serve`) and the multi-process
//! round-robin balancer (`balancer`) need a listener tuned identically —
//! `SO_REUSEADDR` + `SO_REUSEPORT` and a deep accept backlog — so the socket
//! setup lives here rather than being duplicated at each bind site.

use std::net::SocketAddr;

use socket2::{Domain, Protocol, Socket, Type};

/// Listen backlog. Large enough that a C10K burst of SYNs doesn't get dropped;
/// the kernel silently clamps to `/proc/sys/net/core/somaxconn` (4096 on modern
/// Linux).
pub const LISTEN_BACKLOG: i32 = 16_384;

/// Build a non-blocking, `SO_REUSEADDR`/`SO_REUSEPORT` TCP listener bound to
/// `addr` with a deep backlog, returned as a tokio `TcpListener`.
///
/// `SO_REUSEPORT` (Linux) lets several independent processes bind the *same*
/// port — the balancer relies on it only as a defensive fallback; its backends
/// each own a distinct loopback port and it round-robins across them itself.
pub fn build_listener(addr: SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    let domain = if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_nonblocking(true)?;
    socket.set_reuse_address(true)?;
    // SO_REUSEPORT on Linux lets multiple sockets bind the same port so the
    // kernel load-balances accepts across them. Safe fallback if unsupported.
    #[cfg(target_os = "linux")]
    let _ = socket.set_reuse_port(true);
    socket.bind(&addr.into())?;
    socket.listen(LISTEN_BACKLOG)?;
    let std_listener: std::net::TcpListener = socket.into();
    let listener = tokio::net::TcpListener::from_std(std_listener)?;
    Ok(listener)
}

/// Bind a `UnixListener` at `path`, first unlinking any stale socket file left
/// by a previous (crashed or un-cleaned) run so the `bind(2)` does not fail with
/// `EADDRINUSE`. Only a pre-existing *socket* is removed; a non-socket file at
/// the path is left in place so the bind fails loudly rather than clobbering an
/// unrelated file. The runner's UDS transport speaks HTTP/1.1 over this socket
/// (`transport_http/client/connection.rs` -> `UnixStream::connect` + h1
/// handshake), so callers serve the axum router over it with an HTTP/1-capable
/// connection builder.
#[cfg(unix)]
pub fn bind_unix_listener(path: &str) -> anyhow::Result<tokio::net::UnixListener> {
    use std::os::unix::fs::FileTypeExt;

    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => {
            std::fs::remove_file(path)?;
        }
        Ok(_) => {
            anyhow::bail!("--uds path {path} exists and is not a socket; refusing to remove it");
        }
        // Nothing at the path (the common case) — nothing to unlink.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let listener = tokio::net::UnixListener::bind(path)?;
    Ok(listener)
}

/// Bind a Unix-domain socket at `path` and serve `router` over it as HTTP/1.1
/// until the process exits (the accept loop never returns `Ok`).
///
/// This is the shipped UDS serve path — the single-process binary spawns it as
/// a background task alongside the TCP frontend, and the e2e suite drives it
/// directly. One tokio task per accepted connection runs hyper's HTTP/1
/// handshake: the runner's UDS transport
/// (`transport_http/client/connection.rs`) negotiates HTTP/1.1 only, so no h2
/// upgrade is offered. There is no `TCP_NODELAY` / `SO_REUSEPORT` tuning — those
/// are TCP socket options with no Unix-domain analogue.
#[cfg(unix)]
pub async fn serve_router_uds(router: axum::Router, path: &str) -> anyhow::Result<()> {
    use hyper_util::rt::TokioIo;
    use tower::Service;

    let listener = bind_unix_listener(path)?;
    tracing::info!(uds_path = %path, "Listening (Unix domain socket, HTTP/1.1)");

    // `into_make_service` (no connect-info) accepts any target, so the unnamed
    // UDS peer address is passed through unchanged.
    let make_service = router.into_make_service();
    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("UDS accept error: {e}");
                continue;
            }
        };

        let tower_service = match make_service.clone().call(()).await {
            Ok(svc) => svc,
            Err(e) => {
                tracing::warn!("UDS make_service error: {e}");
                continue;
            }
        };

        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let hyper_service =
                hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                    tower_service.clone().call(req)
                });
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, hyper_service)
                .with_upgrades()
                .await
            {
                tracing::warn!("UDS connection error: {e}");
            }
        });
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// A collision-free temp path under the system temp dir (no `tempfile`
    /// dependency in this crate). The caller removes it; parent dir is the OS
    /// temp dir, which always exists.
    fn temp_socket_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "aiperf-mock-{tag}-{}-{nanos}-{n}.sock",
            std::process::id()
        ))
    }

    /// A stale socket file at the path is unlinked so a fresh bind succeeds.
    #[tokio::test]
    async fn bind_unix_listener_unlinks_stale_socket() {
        let path = temp_socket_path("stale");
        let path_str = path.to_str().unwrap().to_owned();

        // First bind creates the socket file.
        let first = bind_unix_listener(&path_str).expect("first bind");
        assert!(path.exists(), "socket file should exist after bind");
        drop(first);
        // Dropping the tokio listener does not remove the socket file on disk,
        // so it is genuinely stale for the second bind.
        assert!(path.exists(), "stale socket file remains after drop");

        // Second bind must succeed by unlinking the stale socket first.
        let second = bind_unix_listener(&path_str).expect("second bind unlinks stale socket");
        drop(second);
        let _ = std::fs::remove_file(&path);
    }

    /// A non-socket file at the path is refused rather than clobbered.
    #[tokio::test]
    async fn bind_unix_listener_refuses_non_socket() {
        let path = temp_socket_path("regular");
        std::fs::write(&path, b"not a socket").expect("write file");
        let err =
            bind_unix_listener(path.to_str().unwrap()).expect_err("must refuse a non-socket path");
        assert!(err.to_string().contains("not a socket"), "err: {err}");
        assert!(path.exists(), "the regular file must be left intact");
        let _ = std::fs::remove_file(&path);
    }

    /// End-to-end: the shipped serve loop answers an HTTP/1.1 request over UDS.
    #[tokio::test]
    async fn serve_router_uds_answers_over_unix_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let path = temp_socket_path("serve");
        let path_str = path.to_str().unwrap().to_owned();

        let cfg = crate::MockServerConfig {
            fast: true,
            no_tokenizer: true,
            ..crate::MockServerConfig::default()
        }
        .apply_flags();
        let state = crate::app::build_state(cfg);
        let router = crate::build_router(state);

        let serve_path = path_str.clone();
        let handle = tokio::spawn(async move {
            let _ = serve_router_uds(router, &serve_path).await;
        });

        // Wait for the socket to appear.
        for _ in 0..50 {
            if path.exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let mut stream = tokio::net::UnixStream::connect(&path_str)
            .await
            .expect("connect UDS");
        let req = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
        stream.write_all(req.as_bytes()).await.expect("write");
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.expect("read");
        let resp = String::from_utf8_lossy(&buf);
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "expected 200 over UDS, got:\n{resp}"
        );

        handle.abort();
        let _ = std::fs::remove_file(&path);
    }
}
