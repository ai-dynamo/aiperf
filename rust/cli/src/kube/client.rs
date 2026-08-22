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
}

/// Bounded stream of Kubernetes watch events. Dropping it cancels its receiver.
pub struct KubeWatch {
    receiver: std::sync::mpsc::Receiver<Result<Vec<u8>, KubeError>>,
}

impl KubeWatch {
    /// Wait no longer than `timeout` for the next raw watch event.
    pub fn next(&self, timeout: Duration) -> Result<Option<Vec<u8>>, KubeError> {
        match self.receiver.recv_timeout(timeout) {
            Ok(event) => event.map(Some),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Ok(None),
        }
    }
}

/// Injectable synchronous boundary around the HTTP/TLS implementation.
pub trait KubeTransport: Send + Sync {
    /// Send a bounded request and return its HTTP status.
    fn send(&self, credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError>;
    /// Open a bounded stream of Kubernetes watch events. Callers own reconnect policy.
    fn watch(&self, credentials: &KubeCredentials, request: KubeRequest) -> Result<KubeWatch, KubeError>;
}

/// Kubernetes API client with finite request and watch deadlines.
pub struct KubeClient {
    credentials: KubeCredentials,
    transport: Arc<dyn KubeTransport>,
    request_deadline: Duration,
    watch_deadline: Duration,
}

impl KubeClient {
    /// Create a client using kubeconfig discovery and the native TLS transport.
    pub fn from_options(options: &KubeAuthOptions) -> Result<Self, KubeError> {
        Self::from_credentials(options.resolve()?)
    }

    /// Create a client from already-resolved credentials.
    pub fn from_credentials(credentials: KubeCredentials) -> Result<Self, KubeError> {
        Ok(Self::with_transport(credentials, Arc::new(HyperKubeTransport)))
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
    pub fn with_deadlines(mut self, request_deadline: Duration, watch_deadline: Duration) -> Result<Self, KubeError> {
        if request_deadline.is_zero() || watch_deadline.is_zero() {
            return Err(KubeError::Transport("Kubernetes deadlines must be finite and positive".to_string()));
        }
        self.request_deadline = request_deadline;
        self.watch_deadline = watch_deadline;
        Ok(self)
    }

    /// Return the configured request deadline.
    pub fn request_deadline(&self) -> Duration { self.request_deadline }

    /// Return the configured watch deadline.
    pub fn watch_deadline(&self) -> Duration { self.watch_deadline }

    /// Open one bounded watch request. Reconnect policy remains at the caller.
    pub fn watch(&self, path: &str) -> Result<KubeWatch, KubeError> {
        self.transport.watch(&self.credentials, KubeRequest {
            method: "GET".to_string(), path: path.to_string(), content_type: String::new(), body: Vec::new(), deadline: self.watch_deadline,
        })
    }

    /// Submit a JSON merge patch using the shared authenticated transport.
    pub fn merge_patch(&self, path: &str, body: &serde_json::Value) -> Result<u16, KubeError> {
        let body = serde_json::to_vec(body).map_err(|error| KubeError::Decode(error.to_string()))?;
        self.transport.send(&self.credentials, KubeRequest {
            method: "PATCH".to_string(),
            path: path.to_string(),
            content_type: "application/merge-patch+json".to_string(),
            body,
            deadline: self.request_deadline,
        })
    }
}

struct HyperKubeTransport;

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
        vec![rustls::SignatureScheme::ECDSA_NISTP256_SHA256, rustls::SignatureScheme::ECDSA_NISTP384_SHA384, rustls::SignatureScheme::ED25519, rustls::SignatureScheme::RSA_PSS_SHA256, rustls::SignatureScheme::RSA_PSS_SHA384, rustls::SignatureScheme::RSA_PSS_SHA512]
    }
}

impl KubeTransport for HyperKubeTransport {
    fn send(&self, credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError> { send_bounded(credentials, request) }

    fn watch(&self, credentials: &KubeCredentials, request: KubeRequest) -> Result<KubeWatch, KubeError> {
        let credentials = credentials.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::Builder::new().name("aiperf-k8s-watch".to_string()).spawn(move || {
            let result = tokio::runtime::Builder::new_current_thread().enable_all().build()
                .map_err(|error| KubeError::Transport(error.to_string()))
                .and_then(|runtime| runtime.block_on(async {
                    tokio::time::timeout(request.deadline, stream_watch(&credentials, request, &sender)).await
                        .map_err(|_| KubeError::Transport("Kubernetes watch timed out".to_string()))?
                }));
            if let Err(error) = result { let _ = sender.send(Err(error)); }
        }).map_err(KubeError::Io)?;
        Ok(KubeWatch { receiver })
    }
}

fn send_bounded(credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError> {
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()
        .map_err(|error| KubeError::Transport(error.to_string()))?;
    runtime.block_on(async {
        tokio::time::timeout(request.deadline, send_request(credentials, request)).await
            .map_err(|_| KubeError::Transport("Kubernetes API request timed out".to_string()))?
    })
}

fn client_auth(credentials: &KubeCredentials) -> Result<Option<(Vec<rustls::pki_types::CertificateDer<'static>>, rustls::pki_types::PrivateKeyDer<'static>)>, KubeError> {
    match (&credentials.client_certificate_pem, &credentials.client_key_pem) {
        (None, None) => Ok(None),
        (Some(certificate), Some(key)) => {
            let mut certificate = certificate.as_slice();
            let certificates = rustls_pemfile::certs(&mut certificate).collect::<Result<Vec<_>, _>>()
                .map_err(|error| KubeError::Tls(format!("failed to parse Kubernetes client certificate: {error}")))?;
            if certificates.is_empty() { return Err(KubeError::Tls("Kubernetes client certificate contains no certificates".to_string())); }
            let mut key = key.as_slice();
            let key = rustls_pemfile::private_key(&mut key)
                .map_err(|error| KubeError::Tls(format!("failed to parse Kubernetes client key: {error}")))?
                .ok_or_else(|| KubeError::Tls("Kubernetes client key contains no private key".to_string()))?;
            Ok(Some((certificates, key)))
        }
        _ => Err(KubeError::Authentication("Kubernetes client certificate and key must be configured together".to_string())),
    }
}

async fn send_request(credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError> {
    let response = open_response(credentials, request).await?;
    let status = response.status().as_u16();
    let _ = response.into_body().collect().await;
    Ok(status)
}

async fn stream_watch(
    credentials: &KubeCredentials,
    request: KubeRequest,
    sender: &std::sync::mpsc::Sender<Result<Vec<u8>, KubeError>>,
) -> Result<(), KubeError> {
    let mut response = open_response(credentials, request).await?;
    if !response.status().is_success() {
        return Err(KubeError::Transport(format!("Kubernetes watch returned {}", response.status())));
    }
    while let Some(frame) = response.body_mut().frame().await {
        let frame = frame.map_err(|error| KubeError::Transport(error.to_string()))?;
        if let Ok(data) = frame.into_data() {
            if sender.send(Ok(data.to_vec())).is_err() { break; }
        }
    }
    Ok(())
}

async fn open_response(
    credentials: &KubeCredentials,
    request: KubeRequest,
) -> Result<hyper::Response<hyper::body::Incoming>, KubeError> {
    let client_auth = client_auth(credentials)?;
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = if credentials.insecure_skip_tls_verify {
        let builder = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|error| KubeError::Tls(format!("rustls provider initialization failed: {error}")))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(InsecureVerifier));
        match client_auth {
            Some((certificates, key)) => builder.with_client_auth_cert(certificates, key)
                .map_err(|error| KubeError::Tls(format!("invalid Kubernetes client certificate: {error}")))?,
            None => builder.with_no_client_auth(),
        }
    } else {
        let mut ca_pem = credentials.ca_pem.as_deref().ok_or_else(|| KubeError::Tls("Kubernetes API credentials omitted a certificate authority".to_string()))?;
        let mut roots = rustls::RootCertStore::empty();
        let certificates = rustls_pemfile::certs(&mut ca_pem).collect::<Result<Vec<_>, _>>()
            .map_err(|error| KubeError::Tls(format!("failed to parse Kubernetes CA PEM: {error}")))?;
        if certificates.is_empty() { return Err(KubeError::Tls("Kubernetes CA PEM contains no certificates".to_string())); }
        for certificate in certificates { roots.add(certificate).map_err(|error| KubeError::Tls(format!("failed to add Kubernetes CA: {error}")))?; }
        let builder = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|error| KubeError::Tls(format!("rustls provider initialization failed: {error}")))?
            .with_root_certificates(roots);
        match client_auth {
            Some((certificates, key)) => builder.with_client_auth_cert(certificates, key)
                .map_err(|error| KubeError::Tls(format!("invalid Kubernetes client certificate: {error}")))?,
            None => builder.with_no_client_auth(),
        }
    };
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let tcp = tokio::net::TcpStream::connect((credentials.host.as_str(), credentials.port))
        .await
        .map_err(|error| KubeError::Transport(error.to_string()))?;
    let server_name = rustls::pki_types::ServerName::try_from(credentials.server_name.clone())
        .map_err(|error| KubeError::Tls(format!("invalid Kubernetes server name: {error}")))?;
    let tls = connector.connect(server_name, tcp).await.map_err(|error| KubeError::Tls(error.to_string()))?;
    let (mut sender, connection) = hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(tls))
        .await
        .map_err(|error| KubeError::Transport(error.to_string()))?;
    tokio::spawn(async move { let _ = connection.await; });
    let mut builder = Request::builder()
        .method(request.method.as_str())
        .uri(request.path)
        .header("host", format!("{}:{}", credentials.host, credentials.port))
        .header("content-type", request.content_type)
        .header("accept", "application/json");
    if let Some(token) = &credentials.token {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    sender.send_request(builder.body(Full::<Bytes>::new(Bytes::from(request.body))).map_err(|error| KubeError::Transport(error.to_string()))?)
        .await
        .map_err(|error| KubeError::Transport(error.to_string()))
}
