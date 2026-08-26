// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One bounded Kubernetes API transport for user commands and controller reporting.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::Request;

use super::auth::{KubeAuthOptions, KubeCredentials};
use super::error::KubeError;

/// Default upper bound for one API request, including connection and response drain.
pub const DEFAULT_REQUEST_DEADLINE: Duration = Duration::from_secs(10);
/// Default upper bound for one API watch response.
pub const DEFAULT_WATCH_DEADLINE: Duration = Duration::from_secs(30);
/// AIPerf custom-resource API group.
pub const AIPERF_GROUP: &str = "aiperf.nvidia.com";
/// AIPerf custom-resource API version.
pub const AIPERF_VERSION: &str = "v1alpha1";
/// AIPerf custom-resource plural.
pub const AIPERF_PLURAL: &str = "aiperfjobs";
/// Largest accepted newline-delimited Kubernetes watch record.
pub const MAX_WATCH_RECORD_BYTES: usize = 1024 * 1024;
const WATCH_CHANNEL_CAPACITY: usize = 32;

/// Maximum accepted body of one bounded Kubernetes API response.
pub const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
/// Annotation written after results publication completes.
pub const BENCHMARK_COMPLETE_ANNOTATION: &str = "aiperf.nvidia.com/benchmark-complete";

/// A Kubernetes HTTP request after authentication has been resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KubeRequest {
    /// HTTP method.
    pub method: String,
    /// Absolute Kubernetes API path.
    pub path: String,
    /// Request content type.
    pub content_type: String,
    /// Serialized request body.
    pub body: Vec<u8>,
    /// Deadline for the entire request.
    pub deadline: Duration,
    /// Maximum bytes the response collector may retain.
    pub response_limit: usize,
}

/// One bounded Kubernetes API response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KubeResponse {
    /// HTTP status of the completed request.
    pub status: u16,
    /// Complete response body; a body past the request's `response_limit` is refused.
    pub body: Vec<u8>,
}

impl KubeResponse {
    /// Return whether the response carries a 2xx status.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// Bounded stream of Kubernetes watch events. Dropping it cancels its receiver.
pub struct KubeWatch {
    receiver: std::sync::mpsc::Receiver<WatchMessage>,
}

/// One bounded receive outcome from a Kubernetes watch stream.
#[derive(Debug, Eq, PartialEq)]
pub enum KubeWatchPoll {
    /// One complete newline-delimited response record.
    Record(Vec<u8>),
    /// No record arrived within the caller's local receive budget.
    Idle,
    /// The HTTP response reached clean EOF and can be reconnected.
    Closed,
}

type WatchMessage = (
    Result<Vec<u8>, KubeError>,
    tokio::sync::OwnedSemaphorePermit,
);

impl KubeWatch {
    #[cfg(test)]
    pub(super) fn closed_for_test() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<WatchMessage>();
        drop(sender);
        Self { receiver }
    }

    #[cfg(test)]
    pub(crate) fn events_for_test(events: Vec<Vec<u8>>) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let capacity = Arc::new(tokio::sync::Semaphore::new(events.len().max(1)));
        for event in events {
            let permit = Arc::clone(&capacity)
                .try_acquire_owned()
                .expect("test channel has one permit per event");
            sender
                .send((Ok(event), permit))
                .expect("test receiver is open");
        }
        drop(sender);
        Self { receiver }
    }

    /// Distinguish a record, local idle timeout, and clean response EOF.
    pub fn poll(&self, timeout: Duration) -> Result<KubeWatchPoll, KubeError> {
        match self.receiver.recv_timeout(timeout) {
            Ok((event, _permit)) => event.map(KubeWatchPoll::Record),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(KubeWatchPoll::Idle),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Ok(KubeWatchPoll::Closed),
        }
    }

    /// Wait no longer than `timeout` for the next raw watch event.
    pub fn next(&self, timeout: Duration) -> Result<Option<Vec<u8>>, KubeError> {
        match self.poll(timeout)? {
            KubeWatchPoll::Record(record) => Ok(Some(record)),
            KubeWatchPoll::Idle => Ok(None),
            KubeWatchPoll::Closed => Err(KubeError::Transport(
                "Kubernetes watch stream closed".to_string(),
            )),
        }
    }
}

struct WatchSender {
    sender: std::sync::mpsc::Sender<WatchMessage>,
    capacity: Arc<tokio::sync::Semaphore>,
}

impl WatchSender {
    fn bounded(capacity: usize) -> (Self, KubeWatch) {
        // A permit is acquired before enqueue, so the underlying channel can
        // never retain more than `capacity` messages.
        let (sender, receiver) = std::sync::mpsc::channel();
        (
            Self {
                sender,
                capacity: Arc::new(tokio::sync::Semaphore::new(capacity)),
            },
            KubeWatch { receiver },
        )
    }

    async fn send(&self, event: Result<Vec<u8>, KubeError>) -> Result<(), KubeError> {
        let permit = Arc::clone(&self.capacity)
            .acquire_owned()
            .await
            .map_err(|_| KubeError::Transport("Kubernetes watch queue closed".to_string()))?;
        self.sender
            .send((event, permit))
            .map_err(|_| KubeError::Transport("Kubernetes watch receiver closed".to_string()))
    }

    fn try_send(&self, event: Result<Vec<u8>, KubeError>) {
        let Ok(permit) = Arc::clone(&self.capacity).try_acquire_owned() else {
            return;
        };
        let _ = self.sender.send((event, permit));
    }
}

/// Injectable synchronous boundary around the HTTP/TLS implementation.
pub trait KubeTransport: Send + Sync {
    /// Send a bounded request and return its status with a bounded body.
    fn send(
        &self,
        credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeResponse, KubeError>;
    /// Open a bounded stream of Kubernetes watch events. Callers own reconnect policy.
    fn watch(
        &self,
        credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeWatch, KubeError>;
}

/// Kubernetes API client with finite request and watch deadlines.
pub struct KubeClient {
    credentials: KubeCredentials,
    transport: Arc<dyn KubeTransport>,
    request_deadline: Duration,
    watch_deadline: Duration,
}

impl KubeCredentials {
    fn http_authority(&self) -> String {
        if self.host.contains(':') {
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

impl KubeClient {
    /// Create a client using kubeconfig discovery and the native TLS transport.
    pub fn from_options(options: &KubeAuthOptions) -> Result<Self, KubeError> {
        Self::from_credentials(options.resolve()?)
    }

    /// Create a client from already-resolved credentials.
    pub fn from_credentials(credentials: KubeCredentials) -> Result<Self, KubeError> {
        Ok(Self::with_transport(
            credentials,
            Arc::new(HyperKubeTransport),
        ))
    }

    /// Create a client with an injected transport for hermetic tests.
    pub fn with_transport(credentials: KubeCredentials, transport: Arc<dyn KubeTransport>) -> Self {
        Self {
            credentials,
            transport,
            request_deadline: DEFAULT_REQUEST_DEADLINE,
            watch_deadline: DEFAULT_WATCH_DEADLINE,
        }
    }

    /// Override finite deadlines for callers with a tighter lifecycle budget.
    pub fn with_deadlines(
        mut self,
        request_deadline: Duration,
        watch_deadline: Duration,
    ) -> Result<Self, KubeError> {
        if request_deadline.is_zero() || watch_deadline.is_zero() {
            return Err(KubeError::Transport(
                "Kubernetes deadlines must be finite and positive".to_string(),
            ));
        }
        self.request_deadline = request_deadline;
        self.watch_deadline = watch_deadline;
        Ok(self)
    }

    /// Return the configured request deadline.
    pub fn request_deadline(&self) -> Duration {
        self.request_deadline
    }

    /// Return the configured watch deadline.
    pub fn watch_deadline(&self) -> Duration {
        self.watch_deadline
    }

    /// Open one bounded watch request. Reconnect policy remains at the caller.
    pub fn watch(&self, path: &str) -> Result<KubeWatch, KubeError> {
        self.transport.watch(
            &self.credentials,
            KubeRequest {
                method: "GET".to_string(),
                path: path.to_string(),
                content_type: String::new(),
                body: Vec::new(),
                deadline: self.watch_deadline,
                response_limit: MAX_RESPONSE_BYTES,
            },
        )
    }

    /// Submit one bounded JSON API request and return only its HTTP status.
    pub fn request(
        &self,
        method: &str,
        path: &str,
        content_type: &str,
        body: Vec<u8>,
    ) -> Result<u16, KubeError> {
        Ok(self.execute(method, path, content_type, body)?.status)
    }

    /// Submit one bounded JSON API request and retain its bounded response body.
    pub fn execute(
        &self,
        method: &str,
        path: &str,
        content_type: &str,
        body: Vec<u8>,
    ) -> Result<KubeResponse, KubeError> {
        self.execute_with_response_limit(method, path, content_type, body, MAX_RESPONSE_BYTES)
    }

    /// Submit a JSON API request and collect at most `response_limit` response bytes.
    pub fn execute_with_response_limit(
        &self,
        method: &str,
        path: &str,
        content_type: &str,
        body: Vec<u8>,
        response_limit: usize,
    ) -> Result<KubeResponse, KubeError> {
        if response_limit == 0 {
            return Err(KubeError::Transport(
                "Kubernetes response limit must be positive".to_string(),
            ));
        }
        self.transport.send(
            &self.credentials,
            KubeRequest {
                method: method.to_string(),
                path: path.to_string(),
                content_type: content_type.to_string(),
                body,
                deadline: self.request_deadline,
                response_limit,
            },
        )
    }

    /// Submit a JSON merge patch using the shared authenticated transport.
    pub fn merge_patch(&self, path: &str, body: &serde_json::Value) -> Result<u16, KubeError> {
        let body =
            serde_json::to_vec(body).map_err(|error| KubeError::Decode(error.to_string()))?;
        self.request("PATCH", path, "application/merge-patch+json", body)
    }
}

struct HyperKubeTransport;

async fn run_watch_until_deadline<F>(deadline: Duration, sender: &WatchSender, stream: F)
where
    F: std::future::Future<Output = Result<(), KubeError>>,
{
    let result = tokio::time::timeout(deadline, stream)
        .await
        .map_err(|_| KubeError::Transport("Kubernetes watch timed out".to_string()))
        .and_then(std::convert::identity);
    if let Err(error) = result {
        let _ = sender.send(Err(error)).await;
    }
}

#[derive(Debug)]
struct InsecureVerifier;

impl rustls::client::danger::ServerCertVerifier for InsecureVerifier {
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
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}

impl KubeTransport for HyperKubeTransport {
    fn send(
        &self,
        credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeResponse, KubeError> {
        send_bounded(credentials, request)
    }

    fn watch(
        &self,
        credentials: &KubeCredentials,
        request: KubeRequest,
    ) -> Result<KubeWatch, KubeError> {
        let credentials = credentials.clone();
        let (sender, watch) = WatchSender::bounded(WATCH_CHANNEL_CAPACITY);
        std::thread::Builder::new()
            .name("aiperf-k8s-watch".to_string())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| KubeError::Transport(error.to_string()));
                match runtime {
                    Ok(runtime) => runtime.block_on(run_watch_until_deadline(
                        request.deadline,
                        &sender,
                        stream_watch(&credentials, request, &sender),
                    )),
                    Err(error) => sender.try_send(Err(error)),
                }
            })
            .map_err(KubeError::Io)?;
        Ok(watch)
    }
}

fn send_bounded(
    credentials: &KubeCredentials,
    request: KubeRequest,
) -> Result<KubeResponse, KubeError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| KubeError::Transport(error.to_string()))?;
    runtime.block_on(async {
        tokio::time::timeout(request.deadline, send_request(credentials, request))
            .await
            .map_err(|_| KubeError::Transport("Kubernetes API request timed out".to_string()))?
    })
}

fn client_auth(
    credentials: &KubeCredentials,
) -> Result<
    Option<(
        Vec<rustls::pki_types::CertificateDer<'static>>,
        rustls::pki_types::PrivateKeyDer<'static>,
    )>,
    KubeError,
> {
    match (
        &credentials.client_certificate_pem,
        &credentials.client_key_pem,
    ) {
        (None, None) => Ok(None),
        (Some(certificate), Some(key)) => {
            let mut certificate = certificate.as_slice();
            let certificates = rustls_pemfile::certs(&mut certificate)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| {
                    KubeError::Tls(format!(
                        "failed to parse Kubernetes client certificate: {error}"
                    ))
                })?;
            if certificates.is_empty() {
                return Err(KubeError::Tls(
                    "Kubernetes client certificate contains no certificates".to_string(),
                ));
            }
            let mut key = key.as_slice();
            let key = rustls_pemfile::private_key(&mut key)
                .map_err(|error| {
                    KubeError::Tls(format!("failed to parse Kubernetes client key: {error}"))
                })?
                .ok_or_else(|| {
                    KubeError::Tls("Kubernetes client key contains no private key".to_string())
                })?;
            Ok(Some((certificates, key)))
        }
        _ => Err(KubeError::Authentication(
            "Kubernetes client certificate and key must be configured together".to_string(),
        )),
    }
}

async fn send_request(
    credentials: &KubeCredentials,
    request: KubeRequest,
) -> Result<KubeResponse, KubeError> {
    let response_limit = request.response_limit;
    let mut response = open_response(credentials, request).await?;
    let status = response.status().as_u16();
    // Bodies are read frame by frame so an unbounded API response cannot exhaust memory.
    let mut body = Vec::new();
    while let Some(frame) = response.next_frame().await? {
        if let Ok(data) = frame.into_data() {
            if body.len() + data.len() > response_limit {
                return Err(KubeError::Transport(format!(
                    "Kubernetes API response exceeds {} bytes",
                    response_limit,
                )));
            }
            body.extend_from_slice(&data);
        }
    }
    Ok(KubeResponse { status, body })
}

async fn stream_watch(
    credentials: &KubeCredentials,
    request: KubeRequest,
    sender: &WatchSender,
) -> Result<(), KubeError> {
    let mut response = open_response(credentials, request).await?;
    if !response.status().is_success() {
        return Err(KubeError::Transport(format!(
            "Kubernetes watch returned {}",
            response.status()
        )));
    }
    let mut pending = Vec::new();
    while let Some(frame) = response.next_frame().await? {
        if let Ok(data) = frame.into_data() {
            pending.extend_from_slice(&data);
            if pending.len() > MAX_WATCH_RECORD_BYTES {
                return Err(KubeError::Transport(format!(
                    "Kubernetes watch record exceeds {MAX_WATCH_RECORD_BYTES} bytes",
                )));
            }
            while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
                let record: Vec<_> = pending.drain(..=newline).collect();
                sender.send(Ok(record)).await?;
            }
        }
    }
    if !pending.is_empty() {
        sender.send(Ok(pending)).await?;
    }
    Ok(())
}

#[derive(Debug)]
enum ResponseProgress {
    ConnectionClosed,
    Frame(Option<hyper::body::Frame<Bytes>>),
}

async fn next_response_progress<F>(
    connection: &mut tokio::task::JoinHandle<Result<(), KubeError>>,
    frame: F,
) -> Result<ResponseProgress, KubeError>
where
    F: std::future::Future<Output = Option<Result<hyper::body::Frame<Bytes>, hyper::Error>>>,
{
    tokio::select! {
        biased;
        result = connection => match result {
            Ok(Ok(())) => Ok(ResponseProgress::ConnectionClosed),
            Ok(Err(error)) => Err(error),
            Err(error) => Err(KubeError::Transport(format!(
                "Kubernetes HTTP connection monitor closed: {error}"
            ))),
        },
        frame = frame => frame
            .transpose()
            .map(ResponseProgress::Frame)
            .map_err(|error| KubeError::Transport(error.to_string())),
    }
}

struct OpenResponse {
    response: hyper::Response<hyper::body::Incoming>,
    connection: Option<tokio::task::JoinHandle<Result<(), KubeError>>>,
}

impl OpenResponse {
    fn status(&self) -> hyper::StatusCode {
        self.response.status()
    }

    async fn next_frame(&mut self) -> Result<Option<hyper::body::Frame<Bytes>>, KubeError> {
        loop {
            let progress = match self.connection.as_mut() {
                Some(connection) => {
                    next_response_progress(connection, self.response.body_mut().frame()).await?
                }
                None => {
                    return self
                        .response
                        .body_mut()
                        .frame()
                        .await
                        .transpose()
                        .map_err(|error| KubeError::Transport(error.to_string()));
                }
            };
            match progress {
                ResponseProgress::ConnectionClosed => self.connection = None,
                ResponseProgress::Frame(frame) => return Ok(frame),
            }
        }
    }
}

async fn open_response(
    credentials: &KubeCredentials,
    request: KubeRequest,
) -> Result<OpenResponse, KubeError> {
    let client_auth = client_auth(credentials)?;
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config =
        if credentials.insecure_skip_tls_verify {
            let builder = rustls::ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .map_err(|error| {
                    KubeError::Tls(format!("rustls provider initialization failed: {error}"))
                })?
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(InsecureVerifier));
            match client_auth {
                Some((certificates, key)) => builder
                    .with_client_auth_cert(certificates, key)
                    .map_err(|error| {
                        KubeError::Tls(format!("invalid Kubernetes client certificate: {error}"))
                    })?,
                None => builder.with_no_client_auth(),
            }
        } else {
            let mut ca_pem = credentials.ca_pem.as_deref().ok_or_else(|| {
                KubeError::Tls(
                    "Kubernetes API credentials omitted a certificate authority".to_string(),
                )
            })?;
            let mut roots = rustls::RootCertStore::empty();
            let certificates = rustls_pemfile::certs(&mut ca_pem)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| {
                    KubeError::Tls(format!("failed to parse Kubernetes CA PEM: {error}"))
                })?;
            if certificates.is_empty() {
                return Err(KubeError::Tls(
                    "Kubernetes CA PEM contains no certificates".to_string(),
                ));
            }
            for certificate in certificates {
                roots.add(certificate).map_err(|error| {
                    KubeError::Tls(format!("failed to add Kubernetes CA: {error}"))
                })?;
            }
            let builder = rustls::ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .map_err(|error| {
                    KubeError::Tls(format!("rustls provider initialization failed: {error}"))
                })?
                .with_root_certificates(roots);
            match client_auth {
                Some((certificates, key)) => builder
                    .with_client_auth_cert(certificates, key)
                    .map_err(|error| {
                        KubeError::Tls(format!("invalid Kubernetes client certificate: {error}"))
                    })?,
                None => builder.with_no_client_auth(),
            }
        };
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let tcp = tokio::net::TcpStream::connect((credentials.host.as_str(), credentials.port))
        .await
        .map_err(|error| KubeError::Transport(error.to_string()))?;
    let server_name = rustls::pki_types::ServerName::try_from(credentials.server_name.clone())
        .map_err(|error| KubeError::Tls(format!("invalid Kubernetes server name: {error}")))?;
    let tls = connector
        .connect(server_name, tcp)
        .await
        .map_err(|error| KubeError::Tls(error.to_string()))?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(tls))
            .await
            .map_err(|error| KubeError::Transport(error.to_string()))?;
    let connection = tokio::spawn(async move {
        connection.await.map_err(|error| {
            KubeError::Transport(format!("Kubernetes HTTP connection failed: {error}"))
        })
    });
    let mut builder = Request::builder()
        .method(request.method.as_str())
        .uri(request.path)
        .header("host", credentials.http_authority())
        .header("content-type", request.content_type)
        .header("accept", "application/json");
    if let Some(token) = &credentials.token {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    let response = sender
        .send_request(
            builder
                .body(Full::<Bytes>::new(Bytes::from(request.body)))
                .map_err(|error| KubeError::Transport(error.to_string()))?,
        )
        .await
        .map_err(|error| KubeError::Transport(error.to_string()))?;
    Ok(OpenResponse {
        response,
        connection: Some(connection),
    })
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;

    use hyper::body::Frame;

    use super::*;

    #[test]
    fn http_authority_brackets_only_ipv6_hosts() {
        let mut credentials = KubeCredentials {
            host: "::1".to_string(),
            port: 6443,
            server_name: "::1".to_string(),
            token: None,
            client_certificate_pem: None,
            client_key_pem: None,
            ca_pem: None,
            insecure_skip_tls_verify: true,
        };

        assert_eq!(credentials.http_authority(), "[::1]:6443");
        credentials.host = "127.0.0.1".to_string();
        assert_eq!(credentials.http_authority(), "127.0.0.1:6443");
        credentials.host = "kubernetes.default.svc".to_string();
        assert_eq!(credentials.http_authority(), "kubernetes.default.svc:6443");
    }

    #[test]
    fn full_watch_channel_does_not_block_the_watch_deadline() {
        let (sender, watch) = WatchSender::bounded(WATCH_CHANNEL_CAPACITY);
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let producer = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build test runtime");
            runtime.block_on(run_watch_until_deadline(
                Duration::from_millis(50),
                &sender,
                async {
                    started_sender.send(()).expect("test remains active");
                    for index in 0..WATCH_CHANNEL_CAPACITY * 2 {
                        sender.send(Ok(index.to_string().into_bytes())).await?;
                    }
                    std::future::pending().await
                },
            ));
        });

        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("watch producer starts");
        sleep(Duration::from_millis(100));
        for expected in 0..WATCH_CHANNEL_CAPACITY {
            let record = watch
                .next(Duration::from_secs(1))
                .expect("buffered watch record")
                .expect("record is available");
            assert_eq!(record, expected.to_string().as_bytes());
            if expected == 0 {
                sleep(Duration::from_millis(50));
            }
        }
        let error = watch
            .next(Duration::from_secs(1))
            .expect_err("deadline must precede records that did not fit the bounded channel");
        assert!(error.to_string().contains("watch timed out"), "{error}");
        producer.join().expect("watch producer exits");
    }

    #[test]
    fn terminal_connection_error_reaches_the_response_reader() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build test runtime");
        let error = runtime
            .block_on(async {
                let mut connection = tokio::spawn(async {
                    Err(KubeError::Transport(
                        "terminal connection failure".to_string(),
                    ))
                });
                next_response_progress(
                    &mut connection,
                    std::future::pending::<Option<Result<Frame<Bytes>, hyper::Error>>>(),
                )
                .await
            })
            .expect_err("terminal connection error must interrupt a pending body");
        assert!(error.to_string().contains("terminal connection failure"));
    }
}
