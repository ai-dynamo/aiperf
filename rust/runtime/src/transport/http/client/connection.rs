// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-injected DNS, TCP, TLS, and HTTP connection establishment.

use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::OnceLock;

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
use tokio::sync::Notify;
use url::Url;

use crate::clock::Clock;

use crate::transport::core::{ErrorDetails, ErrorKind, TraceData};
use crate::transport::http::client::resolver::{CachingDnsResolver, DnsResolver};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::config::defaults::{SocketOptions, apply_socket_options};
use crate::transport::http::models::HttpVersion;

/// Drives `!Send` HTTP/2 connections on the current thread.
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

/// Shared send-completion signal stamped by [`TimedBody`].
///
/// Cancellation waits on this signal before arming its deadline. The timestamp
/// is retained so a woken task schedules against the actual send-complete
/// instant rather than the later instant at which the executor happened to poll
/// it.
pub struct SendCompletion {
    headers_ns: Cell<Option<i64>>,
    sent_ns: Rc<Cell<Option<i64>>>,
    notify: Notify,
}

impl SendCompletion {
    /// Create an untriggered signal.
    pub fn new() -> Self {
        Self {
            headers_ns: Cell::new(None),
            sent_ns: Rc::new(Cell::new(None)),
            notify: Notify::new(),
        }
    }

    fn with_cell(sent_ns: Rc<Cell<Option<i64>>>) -> Self {
        Self {
            headers_ns: Cell::new(None),
            sent_ns,
            notify: Notify::new(),
        }
    }

    fn mark_headers(&self, headers_ns: i64) {
        if self.headers_ns.get().is_none() {
            self.headers_ns.set(Some(headers_ns));
        }
    }

    fn mark(&self, sent_ns: i64) {
        if self.sent_ns.get().is_none() {
            self.sent_ns.set(Some(sent_ns));
            // One cancellation waiter exists per request. `notify_one` stores a
            // permit when that waiter has not yet polled, avoiding a lost wakeup.
            self.notify.notify_one();
        }
    }

    /// Return the captured timestamp when send completion has already fired.
    pub fn sent_ns(&self) -> Option<i64> {
        self.sent_ns.get()
    }

    /// Return the instant the encoder first requested the body, immediately
    /// after the request head was accepted for writing.
    ///
    /// Hyper exposes no request-head callback, so the first body poll is the
    /// closest observable lifecycle boundary.
    pub fn headers_ns(&self) -> Option<i64> {
        self.headers_ns.get()
    }

    /// Wait until the complete body has been handed to the HTTP transport and
    /// return that exact clock timestamp.
    pub async fn wait(&self) -> i64 {
        loop {
            // Register before checking the cell so a mark between the check and
            // await cannot be lost.
            let notified = self.notify.notified();
            if let Some(sent_ns) = self.sent_ns.get() {
                return sent_ns;
            }
            notified.await;
        }
    }
}

impl Default for SendCompletion {
    fn default() -> Self {
        Self::new()
    }
}

/// Request body that records — via the [`Clock`] — the instant it is fully
/// written (end-of-stream). This is the "send complete" hook: hyper's
/// `send_request().await` resolves at *response headers*, so cancellation must
/// instead observe the body stream reaching its end.
pub struct TimedBody {
    inner: Full<Bytes>,
    clock: Rc<dyn Clock>,
    completion: Rc<SendCompletion>,
}

impl TimedBody {
    /// Build a timed body that writes its timestamp into `sent_ns`.
    pub fn new(bytes: Bytes, clock: Rc<dyn Clock>, sent_ns: Rc<Cell<Option<i64>>>) -> Self {
        Self::with_completion(bytes, clock, Rc::new(SendCompletion::with_cell(sent_ns)))
    }

    pub(crate) fn with_completion(
        bytes: Bytes,
        clock: Rc<dyn Clock>,
        completion: Rc<SendCompletion>,
    ) -> Self {
        Self {
            inner: Full::new(bytes),
            clock,
            completion,
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
        this.completion.mark_headers(this.clock.now_ns());
        let r = Pin::new(&mut this.inner).poll_frame(cx);
        // Stamp the "send complete" instant once the body is fully handed to the
        // encoder: either an explicit end-of-stream, or the last data frame
        // (hyper skips the trailing `None` poll when `is_end_stream` is already
        // true, e.g. for a single-frame `Full` body).
        if this.completion.sent_ns().is_none() {
            let done = match &r {
                Poll::Ready(None) => true,
                Poll::Ready(Some(Ok(_))) => this.inner.is_end_stream(),
                _ => false,
            };
            if done {
                this.completion.mark(this.clock.now_ns());
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
    /// Whether this sender can carry concurrent independent streams.
    pub fn is_multiplexed(&self) -> bool {
        matches!(self, Sender::H2(_))
    }

    /// Whether the sender currently accepts a request.
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
        let ready = match self {
            Sender::H1(sender) => sender.ready().await,
            Sender::H2(sender) => sender.ready().await,
        };
        ready.map_err(|error| ErrorDetails {
            kind: ErrorKind::Other,
            code: None,
            message: format!("send readiness: {error}"),
        })?;
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
#[derive(Debug, Clone, Copy)]
pub struct SocketInfo {
    pub local: SocketAddr,
    pub remote: SocketAddr,
}

#[derive(Debug)]
struct NoCertificateVerification {
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl rustls::client::danger::ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        signed: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            signed,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        signed: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            signed,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Build TLS policy for one transport.
///
/// `ssl_verify=false` disables certificate-chain and hostname validation while
/// retaining cryptographic handshake-signature verification.
fn rustls_config(client: &ClientConfig) -> Arc<rustls::ClientConfig> {
    if let Some(prepared) = &client.prepared_tls {
        return prepared.rustls_config();
    }
    // The non-prepared config depends only on `ssl_verify` (roots, provider, and
    // ALPN are fixed), so memoize the two variants instead of rebuilding the
    // aws-lc provider and re-extending the full webpki root store per connection.
    static CACHE: [OnceLock<Arc<rustls::ClientConfig>>; 2] = [OnceLock::new(), OnceLock::new()];
    CACHE[usize::from(client.ssl_verify)]
        .get_or_init(|| build_non_prepared_rustls_config(client.ssl_verify))
        .clone()
}

/// Reuse the resolved HTTP trust and client-identity policy for a WebSocket
/// HTTP/1.1 upgrade. HTTP/2 ALPN is deliberately removed because RFC 6455 uses
/// an HTTP/1.1 upgrade on this connector.
#[cfg(feature = "websocket")]
pub(crate) fn websocket_rustls_config(client: &ClientConfig) -> Arc<rustls::ClientConfig> {
    let mut config = (*rustls_config(client)).clone();
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Arc::new(config)
}

/// Build the non-prepared client TLS config for one `ssl_verify` mode.
fn build_non_prepared_rustls_config(ssl_verify: bool) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    // Feature unification links multiple crypto providers, so select one
    // explicitly instead of relying on rustls process-global inference.
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("aws-lc supports rustls safe default protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    if !ssl_verify {
        cfg.dangerous()
            .set_certificate_verifier(Arc::new(NoCertificateVerification { provider }));
    }
    cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Arc::new(cfg)
}

/// Shared `ssl_verify=false` verifier for gRPC.
///
/// Handshake signatures remain verified; certificate-chain and hostname
/// validation are skipped.
#[cfg(feature = "grpc")]
pub(crate) fn insecure_server_cert_verifier() -> Arc<dyn rustls::client::danger::ServerCertVerifier>
{
    Arc::new(NoCertificateVerification {
        provider: Arc::new(rustls::crypto::aws_lc_rs::default_provider()),
    })
}

/// Race `fut` against a [`Clock`] timer of `timeout_ns`. If the future resolves
/// first its output is returned; if the timer fires first, `on_timeout()` builds
/// the error. A `None` / non-positive `timeout_ns` disables the timer entirely
/// (a pure pass-through — the un-raced hot path). The timer is driven through the
/// crate's [`Clock`] (SimClock in tests), never hardcoded `tokio::time`, so
/// virtual-time runs stay deterministic.
pub(crate) async fn with_timeout<T, E>(
    clock: Rc<dyn Clock>,
    timeout_ns: Option<i64>,
    fut: impl Future<Output = Result<T, E>>,
    on_timeout: impl FnOnce() -> E,
) -> Result<T, E> {
    match timeout_ns.filter(|&t| t > 0) {
        None => fut.await,
        Some(t) => {
            let timer = clock.sleep(t);
            futures::pin_mut!(fut);
            tokio::select! {
                biased;
                res = &mut fut => res,
                _ = timer => Err(on_timeout()),
            }
        }
    }
}

/// Establish a connection to `url`, enforcing `cfg.connect_timeout_ns` (when set
/// to a positive value) around each DNS -> TCP -> TLS -> handshake attempt by
/// racing it against a [`Clock`] timer. A `None`/non-positive timeout means "no
/// deadline". Up to `cfg.max_connect_retries` further attempts follow a
/// connect-phase failure, so that deadline bounds one attempt and not the whole
/// call. `trace` is filled with connect timings.
pub async fn establish(
    url: &Url,
    cfg: &ClientConfig,
    clock: Rc<dyn Clock>,
    trace: &mut TraceData,
) -> Result<(Sender, SocketInfo), ErrorDetails> {
    let resolver = CachingDnsResolver::default();
    establish_with_resolver(url, cfg, clock, trace, &resolver).await
}

/// Establish using an injected DNS policy.
///
/// Connection managers reuse a resolver across connections so trace records can
/// distinguish cache hits from misses; standalone callers retain
/// [`establish`]'s request-local behavior.
pub async fn establish_with_resolver(
    url: &Url,
    cfg: &ClientConfig,
    clock: Rc<dyn Clock>,
    trace: &mut TraceData,
    resolver: &dyn DnsResolver,
) -> Result<(Sender, SocketInfo), ErrorDetails> {
    // One initial attempt plus up to `max_connect_retries` further tries. Only
    // pre-send `ErrorKind::Connect` failures (DNS/TCP/TLS/handshake) are worth
    // retrying, because the request bytes never left the client and the server
    // cannot have observed a partial request. Connect timeouts and every
    // post-send outcome are handed back to the caller unchanged. The retry loop
    // and clock-driven linear backoff are shared with the gRPC transport via
    // `transport::retry::retry_connect`.
    crate::transport::retry::retry_connect(
        &clock,
        cfg.max_connect_retries,
        cfg.connect_retry_backoff_ns,
        |err: &ErrorDetails| err.kind == ErrorKind::Connect,
        async || establish_once(url, cfg, &clock, trace, resolver).await,
    )
    .await
}

/// Establish a connection with the connect-phase deadline applied, but without
/// retry. `establish_with_resolver` layers retry-with-backoff on top of this.
async fn establish_once(
    url: &Url,
    cfg: &ClientConfig,
    clock: &Rc<dyn Clock>,
    trace: &mut TraceData,
    resolver: &dyn DnsResolver,
) -> Result<(Sender, SocketInfo), ErrorDetails> {
    let timeout_ns = cfg.connect_timeout_ns;
    with_timeout(
        clock.clone(),
        timeout_ns,
        establish_inner(url, cfg, clock.clone(), trace, resolver),
        || ErrorDetails {
            kind: ErrorKind::Timeout,
            code: None,
            message: format!("connect timeout after {}ns", timeout_ns.unwrap_or_default()),
        },
    )
    .await
}

async fn establish_inner(
    url: &Url,
    cfg: &ClientConfig,
    clock: Rc<dyn Clock>,
    trace: &mut TraceData,
    resolver: &dyn DnsResolver,
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

    // When a proxy is configured, tunnel via
    // HTTP CONNECT: the proxy resolves the origin, so we must not resolve it
    // locally. Otherwise resolve and connect to the origin directly, unchanged —
    // DNS stays outside the TCP-connect span exactly as before.
    let (tcp, remote) = if let Some(proxy) = &cfg.proxy {
        trace.tcp_connect_start_ns = Some(clock.now_ns());
        let stream =
            crate::transport::http::client::proxy::connect_via_proxy(proxy, host, port).await?;
        let peer = stream.peer_addr().map_err(ErrorDetails::from)?;
        (stream, peer)
    } else {
        let remote = resolver.resolve(host, port, cfg, &clock, trace).await?;
        trace.tcp_connect_start_ns = Some(clock.now_ns());
        let stream = TcpStream::connect(remote)
            .await
            .map_err(ErrorDetails::from)?;
        (stream, remote)
    };
    let local = tcp.local_addr().map_err(ErrorDetails::from)?;
    {
        let sref = socket2::SockRef::from(&tcp);
        continue_after_socket_options(&sref, || async { Ok(()) }).await?;
    }

    let force_h2 = matches!(cfg.http_version, HttpVersion::Http2PriorKnowledge);
    let force_h1 = matches!(cfg.http_version, HttpVersion::Http1Only);

    let sender = if is_tls {
        use tokio_rustls::TlsConnector;
        let connector = TlsConnector::from(rustls_config(cfg));
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
        let connected_ns = clock.now_ns();
        trace.tls_connect_end_ns = Some(connected_ns);
        // The public TCP span includes TLS while the dedicated TLS fields retain
        // the narrower handshake bracket.
        trace.tcp_connect_end_ns = Some(connected_ns);
        let alpn_h2 = tls.get_ref().1.alpn_protocol() == Some(b"h2");
        let use_h2 = force_h2 || (alpn_h2 && !force_h1);
        handshake(TokioIo::new(tls), use_h2).await?
    } else {
        trace.tcp_connect_end_ns = Some(clock.now_ns());
        let use_h2 = force_h2; // cleartext: h2 only via prior-knowledge
        handshake(TokioIo::new(tcp), use_h2).await?
    };

    trace.local_ip = Some(local.ip().to_string());
    trace.local_port = Some(local.port());
    trace.remote_ip = Some(remote.ip().to_string());
    trace.remote_port = Some(remote.port());

    Ok((sender, SocketInfo { local, remote }))
}

async fn continue_after_socket_options<O, F, Fut, T>(
    options: &O,
    continuation: F,
) -> Result<T, ErrorDetails>
where
    O: SocketOptions + ?Sized,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T, ErrorDetails>>,
{
    apply_socket_options(options).map_err(|error| ErrorDetails {
        kind: ErrorKind::Connect,
        code: None,
        message: format!("required TCP socket setup failed: {error}"),
    })?;
    continuation().await
}

async fn handshake<I>(io: I, use_h2: bool) -> Result<Sender, ErrorDetails>
where
    I: hyper::rt::Read + hyper::rt::Write + Unpin + 'static,
{
    if use_h2 {
        // h2 remembers only `max_concurrent_reset_streams` locally reset streams
        // (default 50); past that the oldest is purged, and a late frame for a
        // purged stream is a connection-level protocol error that terminates the
        // connection. Large cancellation bursts can hit that, so stress tests may
        // raise the cap with `AIPERF_H2_MAX_RESET_STREAMS`.
        let mut builder = hyper::client::conn::http2::Builder::new(LocalExec);
        if let Some(cap) = std::env::var("AIPERF_H2_MAX_RESET_STREAMS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            builder.max_concurrent_reset_streams(cap);
        }
        let (sender, conn) = builder.handshake(io).await.map_err(|e| ErrorDetails {
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

#[cfg(test)]
mod tests {
    use super::{
        continue_after_socket_options, establish_with_resolver, rustls_config, with_timeout,
    };
    use crate::clock::{Clock, SimClock, drive_sim};
    use crate::transport::core::{ErrorDetails, ErrorKind, TraceData};
    use crate::transport::http::client::resolver::DnsResolver;
    use crate::transport::http::config::ClientConfig;
    use crate::transport::http::config::defaults::SocketOptions;
    use async_trait::async_trait;
    use std::cell::Cell;
    use std::net::SocketAddr;
    use std::rc::Rc;
    use std::sync::Arc;

    fn run_local<F: std::future::Future>(future: F) -> F::Output {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, future)
    }

    struct FakeSocketOptions {
        nodelay_error: Option<&'static str>,
        has_optional_tuning_error: bool,
    }

    impl FakeSocketOptions {
        fn fail_nodelay(message: &'static str) -> Self {
            Self {
                nodelay_error: Some(message),
                has_optional_tuning_error: false,
            }
        }

        fn fail_optional_tuning() -> Self {
            Self {
                nodelay_error: None,
                has_optional_tuning_error: true,
            }
        }

        fn optional_result(&self) -> std::io::Result<()> {
            if self.has_optional_tuning_error {
                Err(std::io::Error::other("synthetic optional tuning failure"))
            } else {
                Ok(())
            }
        }
    }

    impl SocketOptions for FakeSocketOptions {
        fn set_nodelay(&self, _nodelay: bool) -> std::io::Result<()> {
            match self.nodelay_error {
                Some(message) => Err(std::io::Error::other(message)),
                None => Ok(()),
            }
        }

        fn set_keepalive(&self, _keepalive: bool) -> std::io::Result<()> {
            Ok(())
        }

        fn set_reuse_address(&self, _reuse: bool) -> std::io::Result<()> {
            self.optional_result()
        }

        #[cfg(target_os = "linux")]
        fn set_recv_buffer_size(&self, _size: usize) -> std::io::Result<()> {
            self.optional_result()
        }

        #[cfg(target_os = "linux")]
        fn set_send_buffer_size(&self, _size: usize) -> std::io::Result<()> {
            self.optional_result()
        }
    }

    #[test]
    fn required_socket_option_failure_is_connect_error_before_handshake() {
        let handshake_called = Cell::new(false);
        let options = FakeSocketOptions::fail_nodelay("synthetic TCP_NODELAY failure");
        let error = run_local(continue_after_socket_options(&options, || async {
            handshake_called.set(true);
            Ok(())
        }))
        .expect_err("required setup must fail");
        assert_eq!(error.kind, ErrorKind::Connect);
        assert!(error.message.contains("TCP_NODELAY"));
        assert!(!handshake_called.get());
    }

    #[test]
    fn optional_socket_tuning_failure_still_reaches_handshake() {
        let handshake_called = Cell::new(false);
        let options = FakeSocketOptions::fail_optional_tuning();
        run_local(continue_after_socket_options(&options, || async {
            handshake_called.set(true);
            Ok(())
        }))
        .unwrap();
        assert!(handshake_called.get());
    }

    #[test]
    fn non_prepared_rustls_config_is_memoized_per_verify_mode() {
        let verify = ClientConfig {
            ssl_verify: true,
            ..ClientConfig::default()
        };
        let insecure = ClientConfig {
            ssl_verify: false,
            ..ClientConfig::default()
        };
        // Repeated non-prepared builds return the same cached Arc, not a fresh
        // aws-lc provider + webpki root store each time.
        assert!(Arc::ptr_eq(
            &rustls_config(&verify),
            &rustls_config(&verify)
        ));
        assert!(Arc::ptr_eq(
            &rustls_config(&insecure),
            &rustls_config(&insecure)
        ));
        // The two verify modes are distinct configs.
        assert!(!Arc::ptr_eq(
            &rustls_config(&verify),
            &rustls_config(&insecure)
        ));
    }

    /// A resolver that fails the connect phase for its first `fail_first`
    /// invocations, then resolves. Used to exercise connect-retry policy
    /// without a live socket: a resolver `Connect` error is a genuine
    /// pre-send establishment failure, so it drives the same retry path as a
    /// refused TCP handshake while remaining fully deterministic.
    struct FlakyResolver {
        calls: Cell<u32>,
        fail_first: u32,
        addr_port: u16,
        kind: ErrorKind,
    }

    #[async_trait(?Send)]
    impl DnsResolver for FlakyResolver {
        async fn resolve(
            &self,
            _host: &str,
            _port: u16,
            _cfg: &ClientConfig,
            _clock: &Rc<dyn Clock>,
            _trace: &mut TraceData,
        ) -> Result<SocketAddr, ErrorDetails> {
            let n = self.calls.get() + 1;
            self.calls.set(n);
            if n <= self.fail_first {
                return Err(ErrorDetails {
                    kind: self.kind,
                    code: None,
                    message: format!("synthetic {:?} failure #{n}", self.kind),
                });
            }
            Ok(SocketAddr::from(([127, 0, 0, 1], self.addr_port)))
        }
    }

    #[test]
    fn connect_retries_exhaust_then_surface_last_error() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let cfg = ClientConfig {
            max_connect_retries: 2,
            connect_retry_backoff_ns: 1_000,
            ..ClientConfig::default()
        };
        let resolver = Rc::new(FlakyResolver {
            calls: Cell::new(0),
            fail_first: u32::MAX,
            addr_port: 9,
            kind: ErrorKind::Connect,
        });
        let probe = resolver.clone();
        let url = url::Url::parse("http://example.test:80/").unwrap();
        let out = drive_sim(clock.clone(), async move {
            let mut trace = TraceData::default();
            establish_with_resolver(&url, &cfg, clk, &mut trace, &*probe)
                .await
                .map(|_| ())
        });
        let err = out.expect_err("all attempts fail");
        assert_eq!(err.kind, ErrorKind::Connect);
        // 1 initial attempt + 2 retries.
        assert_eq!(resolver.calls.get(), 3);
        // Linear backoff between retries: 1000*1 + 1000*2 = 3000ns virtual.
        assert_eq!(clock.now_ns(), 3_000);
    }

    #[test]
    fn connect_retries_fire_even_with_connect_timeout_set() {
        // A per-attempt `connect_timeout_ns` must bound each attempt without
        // capping the whole retry sequence: the resolver fails instantly (well
        // within the 10ms per-attempt deadline), so every retry proceeds and
        // only the linear backoff advances virtual time.
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let cfg = ClientConfig {
            connect_timeout_ns: Some(10_000_000),
            max_connect_retries: 2,
            connect_retry_backoff_ns: 1_000,
            ..ClientConfig::default()
        };
        let resolver = Rc::new(FlakyResolver {
            calls: Cell::new(0),
            fail_first: u32::MAX,
            addr_port: 9,
            kind: ErrorKind::Connect,
        });
        let probe = resolver.clone();
        let url = url::Url::parse("http://example.test:80/").unwrap();
        let out = drive_sim(clock.clone(), async move {
            let mut trace = TraceData::default();
            establish_with_resolver(&url, &cfg, clk, &mut trace, &*probe)
                .await
                .map(|_| ())
        });
        let err = out.expect_err("all attempts fail");
        assert_eq!(err.kind, ErrorKind::Connect);
        // 3 attempts despite the connect timeout being set.
        assert_eq!(resolver.calls.get(), 3);
        // Only backoff advanced the clock: 1000*1 + 1000*2 = 3000ns.
        assert_eq!(clock.now_ns(), 3_000);
    }

    #[test]
    fn connect_retry_recovers_after_transient_failures() {
        // A real cleartext HTTP/1 handshake succeeds as soon as the TCP
        // connection is accepted (hyper's client handshake needs no server
        // bytes), so a live listener lets us prove the *recovery* path: the
        // resolver refuses twice, then points establishment at the listener,
        // and the third attempt establishes a real sender.
        use crate::clock::RealClock;
        use tokio::net::TcpListener;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&runtime, async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::task::spawn_local(async move {
                // Accept and hold connections so handshakes complete.
                loop {
                    let _ = listener.accept().await;
                }
            });

            let clock: Rc<dyn Clock> = RealClock::new();
            let cfg = ClientConfig {
                max_connect_retries: 3,
                connect_retry_backoff_ns: 1_000,
                // A per-attempt connect deadline must not defeat recovery.
                connect_timeout_ns: Some(5_000_000_000),
                http_version: crate::transport::http::models::HttpVersion::Http1Only,
                ..ClientConfig::default()
            };
            let resolver = FlakyResolver {
                calls: Cell::new(0),
                fail_first: 2,
                addr_port: port,
                kind: ErrorKind::Connect,
            };
            let url = url::Url::parse(&format!("http://127.0.0.1:{port}/")).unwrap();
            let mut trace = TraceData::default();
            let result = establish_with_resolver(&url, &cfg, clock, &mut trace, &resolver).await;
            assert!(result.is_ok(), "recovered after transient connect failures");
            assert_eq!(resolver.calls.get(), 3);
        });
    }

    #[test]
    fn non_connect_errors_are_not_retried() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let cfg = ClientConfig {
            max_connect_retries: 5,
            connect_retry_backoff_ns: 1_000,
            ..ClientConfig::default()
        };
        let resolver = Rc::new(FlakyResolver {
            calls: Cell::new(0),
            fail_first: u32::MAX,
            addr_port: 9,
            kind: ErrorKind::Other,
        });
        let probe = resolver.clone();
        let url = url::Url::parse("http://example.test:80/").unwrap();
        let out = drive_sim(clock.clone(), async move {
            let mut trace = TraceData::default();
            establish_with_resolver(&url, &cfg, clk, &mut trace, &*probe)
                .await
                .map(|_| ())
        });
        let err = out.expect_err("non-connect failure");
        assert_eq!(err.kind, ErrorKind::Other);
        assert_eq!(resolver.calls.get(), 1);
        assert_eq!(clock.now_ns(), 0);
    }

    #[test]
    fn zero_retries_makes_a_single_attempt() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let cfg = ClientConfig::default(); // max_connect_retries == 0
        let resolver = Rc::new(FlakyResolver {
            calls: Cell::new(0),
            fail_first: u32::MAX,
            addr_port: 9,
            kind: ErrorKind::Connect,
        });
        let probe = resolver.clone();
        let url = url::Url::parse("http://example.test:80/").unwrap();
        let out = drive_sim(clock.clone(), async move {
            let mut trace = TraceData::default();
            establish_with_resolver(&url, &cfg, clk, &mut trace, &*probe)
                .await
                .map(|_| ())
        });
        assert!(out.is_err());
        assert_eq!(resolver.calls.get(), 1);
        assert_eq!(clock.now_ns(), 0);
    }

    fn timeout_err() -> ErrorDetails {
        ErrorDetails {
            kind: ErrorKind::Timeout,
            code: None,
            message: "connect timeout after 1000000ns".to_string(),
        }
    }

    #[test]
    fn times_out_a_never_completing_future() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let out: Result<u32, ErrorDetails> = drive_sim(clock.clone(), async move {
            let never = futures::future::pending::<Result<u32, ErrorDetails>>();
            with_timeout(clk, Some(1_000_000), never, timeout_err).await
        });
        let err = out.expect_err("should time out");
        assert_eq!(err.kind, ErrorKind::Timeout);
    }

    #[test]
    fn passes_through_when_future_resolves_before_timer() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let out: Result<u32, ErrorDetails> = drive_sim(clock.clone(), async move {
            let ready = async { Ok::<u32, ErrorDetails>(42) };
            with_timeout(clk, Some(1_000_000), ready, timeout_err).await
        });
        assert_eq!(out.unwrap(), 42);
    }

    #[test]
    fn none_timeout_is_a_pure_passthrough() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let out: Result<u32, ErrorDetails> = drive_sim(clock.clone(), async move {
            let ready = async { Ok::<u32, ErrorDetails>(7) };
            with_timeout(clk, None, ready, timeout_err).await
        });
        assert_eq!(out.unwrap(), 7);
    }

    #[test]
    fn non_positive_timeout_disables_the_timer() {
        // A zero/negative timeout means "no deadline" — the inner future's own
        // result (here an error) must pass through untouched, not become Timeout.
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let out: Result<u32, ErrorDetails> = drive_sim(clock.clone(), async move {
            let failing = async { Err::<u32, ErrorDetails>(ErrorDetails::other("inner")) };
            with_timeout(clk, Some(0), failing, timeout_err).await
        });
        let err = out.expect_err("inner error");
        assert_eq!(err.kind, ErrorKind::Other);
        assert_eq!(err.message, "inner");
    }

    #[test]
    fn inner_error_passes_through_before_deadline() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let out: Result<u32, ErrorDetails> = drive_sim(clock.clone(), async move {
            let failing = async { Err::<u32, ErrorDetails>(ErrorDetails::other("boom")) };
            with_timeout(clk, Some(1_000_000_000), failing, timeout_err).await
        });
        let err = out.expect_err("inner error");
        assert_eq!(err.kind, ErrorKind::Other);
        assert_eq!(err.message, "boom");
    }
}
