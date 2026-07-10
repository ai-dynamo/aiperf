// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Establish one connection: DNS -> TCP(+socket opts) -> optional TLS/ALPN ->
//! httpN handshake. Every phase timestamped via the Clock into TraceData.

use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;

use bytes::Bytes;
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

/// A protocol-specific request sender.
pub enum Sender {
    H1(hyper::client::conn::http1::SendRequest<Full<Bytes>>),
    H2(hyper::client::conn::http2::SendRequest<Full<Bytes>>),
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
        req: hyper::Request<Full<Bytes>>,
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
        let sender = handshake(TokioIo::new(stream), false, clock.clone()).await?;
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
        let tls = connector
            .connect(server_name, tcp)
            .await
            .map_err(ErrorDetails::from)?;
        let alpn_h2 = tls.get_ref().1.alpn_protocol() == Some(b"h2");
        trace.tcp_connect_end_ns = Some(clock.now_ns());
        let use_h2 = force_h2 || (alpn_h2 && !force_h1);
        handshake(TokioIo::new(tls), use_h2, clock.clone()).await?
    } else {
        trace.tcp_connect_end_ns = Some(clock.now_ns());
        let use_h2 = force_h2; // cleartext: h2 only via prior-knowledge
        handshake(TokioIo::new(tcp), use_h2, clock.clone()).await?
    };

    trace.local_ip = Some(local.ip().to_string());
    trace.local_port = Some(local.port());
    trace.remote_ip = Some(remote.ip().to_string());
    trace.remote_port = Some(remote.port());

    Ok((sender, SocketInfo { local, remote }))
}

async fn handshake<I>(io: I, use_h2: bool, _clock: Rc<dyn Clock>) -> Result<Sender, ErrorDetails>
where
    I: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
{
    if use_h2 {
        let (sender, conn) =
            hyper::client::conn::http2::handshake(hyper_util::rt::TokioExecutor::new(), io)
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
