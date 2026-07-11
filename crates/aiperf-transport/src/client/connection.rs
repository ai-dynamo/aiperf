// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Establish one connection: DNS -> TCP(+socket opts) -> optional TLS/ALPN ->
//! httpN handshake. Every phase timestamped via the Clock into TraceData.

use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;

use std::cell::Cell;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};
use http_body_util::Full;
use hyper::Response;
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;
use url::Url;

use aiperf_clock::Clock;

use crate::client::resolver::resolve;
use crate::config::{ClientConfig, apply_socket_opts};
use crate::models::{ErrorDetails, ErrorKind, HttpVersion, TraceData};

/// A local (`!Send`) executor: drives the connection future on the current
/// thread via `spawn_local`. Used for the HTTP/2 connection so that neither the
/// IO nor the request body must be `Send` (the crate is `Rc`-based / `!Send`).
#[derive(Clone)]
pub struct LocalExec;

impl<F> hyper::rt::Executor<F> for LocalExec
where
    F: Future + 'static,
{
    fn execute(&self, fut: F) {
        tokio::task::spawn_local(async move {
            let _ = fut.await;
        });
    }
}

/// Request body that records — via the [`Clock`] — the instant it is fully
/// written (end-of-stream), into a shared cell. This is the "send complete"
/// hook: hyper's `send_request().await` resolves at *response headers*, so the
/// only way to time when the request body finished being handed to the
/// transport is to observe the body stream reaching its end.
pub struct TimedBody {
    inner: Full<Bytes>,
    clock: Rc<dyn Clock>,
    sent_ns: Rc<Cell<Option<i64>>>,
}

impl TimedBody {
    pub fn new(bytes: Bytes, clock: Rc<dyn Clock>, sent_ns: Rc<Cell<Option<i64>>>) -> Self {
        Self {
            inner: Full::new(bytes),
            clock,
            sent_ns,
        }
    }
}

impl Body for TimedBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        let this = self.get_mut();
        let r = Pin::new(&mut this.inner).poll_frame(cx);
        // Stamp the "send complete" instant once the body is fully handed to the
        // encoder: either an explicit end-of-stream, or the last data frame
        // (hyper skips the trailing `None` poll when `is_end_stream` is already
        // true, e.g. for a single-frame `Full` body).
        if this.sent_ns.get().is_none() {
            let done = match &r {
                Poll::Ready(None) => true,
                Poll::Ready(Some(Ok(_))) => this.inner.is_end_stream(),
                _ => false,
            };
            if done {
                this.sent_ns.set(Some(this.clock.now_ns()));
            }
        }
        r
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// A protocol-specific request sender.
pub enum Sender {
    H1(hyper::client::conn::http1::SendRequest<TimedBody>),
    H2(hyper::client::conn::http2::SendRequest<TimedBody>),
}

impl Sender {
    pub fn is_ready(&self) -> bool {
        match self {
            Sender::H1(s) => s.is_ready(),
            Sender::H2(s) => s.is_ready(),
        }
    }
    /// True once the underlying connection has closed and can no longer carry
    /// requests.
    pub fn is_closed(&self) -> bool {
        match self {
            Sender::H1(s) => s.is_closed(),
            Sender::H2(s) => s.is_closed(),
        }
    }
    /// Clone this sender for concurrent multiplexed requests over the same
    /// connection. HTTP/2 senders are cheaply clonable — each clone opens an
    /// independent stream — so this is how many in-flight requests share one
    /// connection. HTTP/1 has no multiplexing, so returns `None`.
    pub fn clone_multiplex(&self) -> Option<Sender> {
        match self {
            Sender::H2(s) => Some(Sender::H2(s.clone())),
            Sender::H1(_) => None,
        }
    }
    pub async fn send(
        &mut self,
        req: hyper::Request<TimedBody>,
    ) -> Result<Response<Incoming>, ErrorDetails> {
        let r = match self {
            Sender::H1(s) => s.send_request(req).await,
            Sender::H2(s) => s.send_request(req).await,
        };
        r.map_err(|e| ErrorDetails {
            kind: ErrorKind::Other,
            code: None,
            message: format!("send: {e}"),
        })
    }
}

/// Socket endpoint info captured for TraceData.
pub struct SocketInfo {
    pub local: SocketAddr,
    pub remote: SocketAddr,
}

fn rustls_config(_ssl_verify: bool) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut cfg = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    // A no-verify path can be added if needed; default verifies.
    Arc::new(cfg)
}

/// Establish a connection to `url`. `trace` is filled with connect timings.
pub async fn establish(
    url: &Url,
    cfg: &ClientConfig,
    clock: Rc<dyn Clock>,
    trace: &mut TraceData,
) -> Result<(Sender, SocketInfo), ErrorDetails> {
    // Unix-domain socket path: connect over UDS (HTTP/1.1), bypassing the
    // TCP/IP stack. The request URL still supplies the HTTP path + Host header.
    #[cfg(unix)]
    if let Some(path) = &cfg.uds_path {
        trace.tcp_connect_start_ns = Some(clock.now_ns());
        let stream = tokio::net::UnixStream::connect(path)
            .await
            .map_err(ErrorDetails::from)?;
        trace.tcp_connect_end_ns = Some(clock.now_ns());
        let sender = handshake(TokioIo::new(stream), false).await?;
        let dummy = SocketAddr::from(([127, 0, 0, 1], 0));
        return Ok((
            sender,
            SocketInfo {
                local: dummy,
                remote: dummy,
            },
        ));
    }

    let host = url
        .host_str()
        .ok_or_else(|| ErrorDetails::other("missing host"))?;
    let is_tls = url.scheme() == "https";
    let port = url
        .port_or_known_default()
        .unwrap_or(if is_tls { 443 } else { 80 });

    let remote = resolve(host, port, &clock, trace).await?;

    trace.tcp_connect_start_ns = Some(clock.now_ns());
    let tcp = TcpStream::connect(remote)
        .await
        .map_err(ErrorDetails::from)?;
    trace.tcp_connect_end_ns = Some(clock.now_ns());
    let local = tcp.local_addr().map_err(ErrorDetails::from)?;
    // Apply low-latency socket options through a borrowed socket2 ref.
    {
        let sref = socket2::SockRef::from(&tcp);
        let _ = apply_socket_opts(&sref);
    }

    // Decide protocol.
    let force_h2 = matches!(cfg.http_version, HttpVersion::Http2PriorKnowledge);
    let force_h1 = matches!(cfg.http_version, HttpVersion::Http1Only);

    let sender = if is_tls {
        use tokio_rustls::TlsConnector;
        let connector = TlsConnector::from(rustls_config(cfg.ssl_verify));
        let server_name =
            rustls::pki_types::ServerName::try_from(host.to_string()).map_err(|e| {
                ErrorDetails {
                    kind: ErrorKind::Connect,
                    code: None,
                    message: format!("tls name: {e}"),
                }
            })?;
        trace.tls_connect_start_ns = Some(clock.now_ns());
        let tls = connector
            .connect(server_name, tcp)
            .await
            .map_err(ErrorDetails::from)?;
        trace.tls_connect_end_ns = Some(clock.now_ns());
        let alpn_h2 = tls.get_ref().1.alpn_protocol() == Some(b"h2");
        let use_h2 = force_h2 || (alpn_h2 && !force_h1);
        handshake(TokioIo::new(tls), use_h2).await?
    } else {
        let use_h2 = force_h2; // cleartext: h2 only via prior-knowledge
        handshake(TokioIo::new(tcp), use_h2).await?
    };

    trace.local_ip = Some(local.ip().to_string());
    trace.local_port = Some(local.port());
    trace.remote_ip = Some(remote.ip().to_string());
    trace.remote_port = Some(remote.port());

    Ok((sender, SocketInfo { local, remote }))
}

async fn handshake<I>(io: I, use_h2: bool) -> Result<Sender, ErrorDetails>
where
    I: hyper::rt::Read + hyper::rt::Write + Unpin + 'static,
{
    if use_h2 {
        let (sender, conn) = hyper::client::conn::http2::handshake(LocalExec, io)
            .await
            .map_err(|e| ErrorDetails {
                kind: ErrorKind::Connect,
                code: None,
                message: format!("h2 handshake: {e}"),
            })?;
        tokio::task::spawn_local(async move {
            let _ = conn.await;
        });
        Ok(Sender::H2(sender))
    } else {
        let (sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .map_err(|e| ErrorDetails {
                kind: ErrorKind::Connect,
                code: None,
                message: format!("h1 handshake: {e}"),
            })?;
        tokio::task::spawn_local(async move {
            let _ = conn.await;
        });
        Ok(Sender::H1(sender))
    }
}
