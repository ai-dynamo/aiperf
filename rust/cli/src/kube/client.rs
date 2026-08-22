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

/// Injectable synchronous boundary around the HTTP/TLS implementation.
pub trait KubeTransport: Send + Sync {
    /// Send a bounded request and return its HTTP status.
    fn send(&self, credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError>;
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
        if credentials.insecure_skip_tls_verify {
            return Err(KubeError::Tls(
                "insecure TLS requires the explicit native insecure transport, which is unavailable in native-k8s/v1".to_string(),
            ));
        }
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

impl KubeTransport for HyperKubeTransport {
    fn send(&self, credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError> {
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
}

async fn send_request(credentials: &KubeCredentials, request: KubeRequest) -> Result<u16, KubeError> {
    let ca_pem = credentials.ca_pem.as_deref().ok_or_else(|| {
        KubeError::Tls("Kubernetes API credentials omitted a certificate authority".to_string())
    })?;
    let mut roots = rustls::RootCertStore::empty();
    let certs = rustls_pemfile::certs(&mut ca_pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| KubeError::Tls(format!("failed to parse Kubernetes CA PEM: {error}")))?;
    if certs.is_empty() {
        return Err(KubeError::Tls("Kubernetes CA PEM contains no certificates".to_string()));
    }
    for certificate in certs {
        roots.add(certificate).map_err(|error| KubeError::Tls(format!("failed to add Kubernetes CA: {error}")))?;
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| KubeError::Tls(format!("rustls provider initialization failed: {error}")))?
        .with_root_certificates(roots)
        .with_no_client_auth();
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
    let response = sender.send_request(builder.body(Full::<Bytes>::new(Bytes::from(request.body))).map_err(|error| KubeError::Transport(error.to_string()))?)
        .await
        .map_err(|error| KubeError::Transport(error.to_string()))?;
    let status = response.status().as_u16();
    let _ = response.into_body().collect().await;
    Ok(status)
}
