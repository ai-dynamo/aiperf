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
use tokio::sync::Notify;
use url::Url;

use crate::clock::Clock;

use crate::transport_http::client::resolver::{CachingDnsResolver, DnsResolver};
use crate::transport_http::config::{ClientConfig, apply_socket_opts};
use crate::transport_http::models::{ErrorDetails, ErrorKind, HttpVersion, TraceData};

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
    /// Hyper exposes no request-head callback. The first body poll is its
    /// closest lifecycle boundary to aiohttp's distinct
    /// `on_request_headers_sent` event.
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
    ///
    /// This compatibility constructor is useful to timing-only callers. The
    /// cancellable request path uses the crate-private completion constructor so
    /// it can also await the event.
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
/// `ssl_verify=false` deliberately disables only certificate-chain and hostname
/// validation. Handshake signatures remain cryptographically verified. This is
/// the Rust equivalent of the Python connector's `ssl=False` behavior.
fn rustls_config(client: &ClientConfig) -> Arc<rustls::ClientConfig> {
    if let Some(prepared) = &client.prepared_tls {
        return prepared.rustls_config();
    }
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    // Select the provider explicitly. The complete runner links both the HTTP
    // transport's aws-lc default and tonic/reqwest's ring features, so rustls
    // cannot infer a process-global provider safely from feature unification.
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("aws-lc supports rustls safe default protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    if !client.ssl_verify {
        cfg.dangerous()
            .set_certificate_verifier(Arc::new(NoCertificateVerification { provider }));
    }
    cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Arc::new(cfg)
}

/// A rustls verifier that accepts ANY server certificate — the shared
/// `ssl_verify=false` policy. Exposed so the gRPC (tonic) transport can install
/// the same danger verifier for `grpcs` against a self-signed / untrusted
/// server, matching the HTTP transport's behavior. Signatures stay
/// cryptographically verified; only chain/hostname validation is skipped.
///
/// The HTTP transport uses [`NoCertificateVerification`] directly; this wrapper
/// exists solely for the gRPC transport, so it is gated on the `grpc` feature.
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
/// to a positive value) around the whole DNS -> TCP -> TLS -> handshake phase by
/// racing it against a [`Clock`] timer. A `None`/non-positive timeout means "no
/// deadline". `trace` is filled with connect timings.
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
    let timeout_ns = cfg.connect_timeout_ns;
    with_timeout(
        clock.clone(),
        timeout_ns,
        establish_inner(url, cfg, clock, trace, resolver),
        || ErrorDetails {
            kind: ErrorKind::Timeout,
            code: None,
            message: format!("connect timeout after {}ns", timeout_ns.unwrap_or_default()),
        },
    )
    .await
}

/// The un-timed connection-establishment body raced by [`establish`].
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

    let remote = resolver.resolve(host, port, cfg, &clock, trace).await?;

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
        // Python's connection-create trace folds TLS into tcp_connect_*.
        // Preserve that public span while retaining the Rust-only TLS bracket.
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

async fn handshake<I>(io: I, use_h2: bool) -> Result<Sender, ErrorDetails>
where
    I: hyper::rt::Read + hyper::rt::Write + Unpin + 'static,
{
    if use_h2 {
        // hyper caps locally-reset streams at 1024 (the Rapid-Reset / CVE-2023-44487
        // guard). At very high in-flight concurrency (100k+), a duration bound or
        // per-request timeout cancels many streams at once and trips that cap,
        // tearing the connection down and failing every remaining stream. Raise it
        // via `AIPERF_H2_MAX_RESET_STREAMS` for concurrent-request stress tests;
        // absent the env var the hyper default (1024) is preserved. Mirrors the
        // mock server's `--max-concurrent-streams`.
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
    use super::with_timeout;
    use crate::clock::{Clock, SimClock, drive_sim};
    use crate::transport_http::models::{ErrorDetails, ErrorKind};
    use std::rc::Rc;

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
            // Resolves immediately, before the 1ms deadline.
            let ready = async { Ok::<u32, ErrorDetails>(42) };
            with_timeout(clk, Some(1_000_000), ready, timeout_err).await
        });
        assert_eq!(out.unwrap(), 42);
    }

    #[test]
    fn none_timeout_is_a_pure_passthrough() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        // No timer is armed; a ready future passes straight through.
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
        // With a generous deadline the inner error is returned as-is (the timer
        // never fires), proving errors aren't masked as timeouts.
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
