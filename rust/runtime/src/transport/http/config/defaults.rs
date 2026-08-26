// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client and socket defaults.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};

/// Opaque, already validated TLS client policy injected by a provider.
///
/// This is deliberately not a serializable configuration surface. Callers can
/// construct it only from provider-resolved PEM entities, and its `Debug`
/// representation never exposes certificates or private-key bytes.
#[derive(Clone)]
pub struct PreparedTlsClientConfig {
    inner: Arc<rustls::ClientConfig>,
}

impl PreparedTlsClientConfig {
    /// Build a verifying rustls client policy from provider-held material.
    ///
    /// `trust_roots_pem = None` selects the built-in WebPKI roots. A supplied
    /// trust bundle replaces that set so a named private trust domain does not
    /// silently retain public-CA authority. Client certificate and private key
    /// must either both be supplied or both be absent.
    pub fn from_provider_pem(
        trust_roots_pem: Option<&[u8]>,
        client_certificate_chain_pem: Option<&[u8]>,
        client_private_key_pem: Option<&[u8]>,
    ) -> Result<Self, PreparedTlsClientConfigError> {
        let roots = match trust_roots_pem {
            Some(pem) => provider_roots(pem)?,
            None => {
                let mut roots = rustls::RootCertStore::empty();
                roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
                roots
            }
        };
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let builder = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|_| PreparedTlsClientConfigError::UnsupportedProtocolProfile)?
            .with_root_certificates(roots);
        let mut config = match (client_certificate_chain_pem, client_private_key_pem) {
            (None, None) => builder.with_no_client_auth(),
            (Some(chain_pem), Some(key_pem)) => {
                let chain = CertificateDer::pem_slice_iter(chain_pem)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| PreparedTlsClientConfigError::InvalidClientCertificate)?;
                if chain.is_empty() {
                    return Err(PreparedTlsClientConfigError::InvalidClientCertificate);
                }
                let key = PrivateKeyDer::from_pem_slice(key_pem)
                    .map_err(|_| PreparedTlsClientConfigError::InvalidClientPrivateKey)?;
                builder
                    .with_client_auth_cert(chain, key)
                    .map_err(|_| PreparedTlsClientConfigError::InvalidClientIdentity)?
            }
            _ => return Err(PreparedTlsClientConfigError::IncompleteClientIdentity),
        };
        config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        Ok(Self {
            inner: Arc::new(config),
        })
    }

    pub(crate) fn rustls_config(&self) -> Arc<rustls::ClientConfig> {
        Arc::clone(&self.inner)
    }
}

impl Debug for PreparedTlsClientConfig {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("PreparedTlsClientConfig([REDACTED])")
    }
}

impl PartialEq for PreparedTlsClientConfig {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for PreparedTlsClientConfig {}

fn provider_roots(pem: &[u8]) -> Result<rustls::RootCertStore, PreparedTlsClientConfigError> {
    let certificates = CertificateDer::pem_slice_iter(pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| PreparedTlsClientConfigError::InvalidTrustRoots)?;
    if certificates.is_empty() {
        return Err(PreparedTlsClientConfigError::InvalidTrustRoots);
    }
    let mut roots = rustls::RootCertStore::empty();
    for certificate in certificates {
        roots
            .add(certificate)
            .map_err(|_| PreparedTlsClientConfigError::InvalidTrustRoots)?;
    }
    Ok(roots)
}

/// Stable, secret-free reason provider TLS material was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedTlsClientConfigError {
    /// The trust provider did not return a non-empty valid certificate bundle.
    InvalidTrustRoots,
    /// The mTLS provider did not return a non-empty valid certificate chain.
    InvalidClientCertificate,
    /// The mTLS provider did not return a supported private-key PEM entity.
    InvalidClientPrivateKey,
    /// The certificate and private key do not form a usable rustls identity.
    InvalidClientIdentity,
    /// Exactly one half of the client identity was supplied.
    IncompleteClientIdentity,
    /// The linked crypto provider cannot implement the required TLS profile.
    UnsupportedProtocolProfile,
}

impl std::fmt::Display for PreparedTlsClientConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidTrustRoots => "TLS trust provider returned invalid certificate material",
            Self::InvalidClientCertificate => {
                "mTLS provider returned invalid client certificate material"
            }
            Self::InvalidClientPrivateKey => {
                "mTLS provider returned invalid client private-key material"
            }
            Self::InvalidClientIdentity => {
                "mTLS provider returned an unusable client certificate identity"
            }
            Self::IncompleteClientIdentity => {
                "mTLS client certificate and private key must be supplied together"
            }
            Self::UnsupportedProtocolProfile => {
                "linked TLS provider cannot implement the required protocol profile"
            }
        })
    }
}

impl std::error::Error for PreparedTlsClientConfigError {}

use crate::transport::http::models::HttpVersion;

/// Client-wide configuration. Timeouts are clock-nanoseconds.
///
/// `connect_timeout_ns` is enforced in `client::connection::establish` (races
/// each DNS/TCP/TLS/handshake attempt against a Clock timer, so with
/// `max_connect_retries` set it bounds an attempt, not the call) and `request_timeout_ns`
/// is enforced in `client::http_client::HttpClient::dispatch` (races the
/// send + response phase). `total_timeout_ns` wraps connection acquisition,
/// send, and the complete response lifecycle with one deadline, matching
/// Config-v2's endpoint request timeout. For all three, `None` or a non-positive
/// value means "no deadline".
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientConfig {
    /// Deadline for DNS, TCP, TLS, and HTTP handshake establishment.
    pub connect_timeout_ns: Option<i64>,
    /// Number of additional attempts to (re)establish a connection after a
    /// connect-phase failure (DNS/TCP/TLS/handshake —
    /// [`crate::transport::core::ErrorKind::Connect`]).
    ///
    /// Retries apply *only* to the pre-response connect phase, before any
    /// request bytes are sent, so a request the server may have partially
    /// processed is never re-issued. Application HTTP errors, post-send
    /// failures, and connect *timeouts* are never retried. `0` (the default)
    /// preserves the historical fail-fast behavior.
    pub max_connect_retries: u32,
    /// Base linear backoff, in clock-nanoseconds, slept between connect
    /// retries. Attempt `n` (1-based) waits `connect_retry_backoff_ns * n`
    /// before retrying, so successive waits grow linearly. The sleep is driven
    /// through the injected [`crate::clock::Clock`] so virtual-time replay
    /// stays deterministic. Non-positive (the default) disables the wait, so
    /// with the default zero retries the whole feature reads as "off".
    pub connect_retry_backoff_ns: i64,
    /// Deadline for request send plus the complete response body.
    pub request_timeout_ns: Option<i64>,
    /// One end-to-end request deadline including connection establishment.
    pub total_timeout_ns: Option<i64>,
    /// Maximum response-body bytes accepted per request.
    ///
    /// The bound is enforced on every received chunk for streaming, ordinary,
    /// and non-2xx bodies. `None` leaves the body unbounded.
    pub max_response_body_bytes: Option<u64>,
    /// Verify the server certificate and hostname for HTTPS connections.
    pub ssl_verify: bool,
    /// Provider-resolved custom trust and optional mTLS identity.
    ///
    /// This cannot be authored through endpoint configuration. When present it
    /// is a fully verifying policy and replaces the built-in WebPKI roots.
    pub prepared_tls: Option<PreparedTlsClientConfig>,
    /// HTTP protocol selection and cleartext prior-knowledge policy.
    pub http_version: HttpVersion,
    /// Maximum idle lifetime of a pooled connection. `None` disables expiry.
    pub keepalive_ns: Option<i64>,
    /// Maximum number of simultaneous HTTP/1 connections per origin.
    ///
    /// HTTP/2 uses one multiplexed connection per origin; this bound applies to
    /// protocols that require an exclusive connection while a request is live.
    pub max_connections_per_origin: usize,
    /// Whether hostname resolutions are cached by the transport.
    pub use_dns_cache: bool,
    /// DNS cache lifetime. `None` retains entries until the transport is dropped.
    pub dns_cache_ttl_ns: Option<i64>,
    /// Retain per-wire-chunk `(clock_ns, size_bytes)` trace vectors.
    ///
    /// Counts, byte totals, and first/last timestamps are always collected.
    pub collect_trace_chunks: bool,
    /// When set, connect over this Unix-domain socket path instead of TCP
    /// (co-located high-throughput: bypasses the TCP/IP loopback softirq tax).
    /// HTTP/1.1 is used over UDS. The request URL still supplies the path + Host.
    pub uds_path: Option<String>,
    /// When set, tunnel TCP through this forward proxy via HTTP `CONNECT` before
    /// TLS. Dataset/tokenizer downloads set it from the proxy environment;
    /// benchmark traffic sets it only when `--proxy`/`--proxy-from-env` opted in,
    /// and otherwise leaves it `None` so its connect is unchanged.
    pub proxy: Option<crate::transport::http::client::proxy::ProxyConfig>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout_ns: None,
            max_connect_retries: 0,
            connect_retry_backoff_ns: 0,
            request_timeout_ns: None,
            total_timeout_ns: None,
            max_response_body_bytes: None,
            ssl_verify: true,
            prepared_tls: None,
            http_version: HttpVersion::Auto,
            keepalive_ns: Some(300_000_000_000),
            max_connections_per_origin: 2_500,
            use_dns_cache: true,
            dns_cache_ttl_ns: Some(300_000_000_000),
            collect_trace_chunks: false,
            uds_path: None,
            proxy: None,
        }
    }
}

/// Apply low-latency streaming socket options without taking ownership of the
/// file descriptor.
pub fn apply_socket_opts(sock: &socket2::SockRef<'_>) -> std::io::Result<()> {
    apply_socket_options(sock)
}

pub(crate) trait SocketOptions {
    fn set_nodelay(&self, nodelay: bool) -> std::io::Result<()>;
    fn set_keepalive(&self, keepalive: bool) -> std::io::Result<()>;
    fn set_reuse_address(&self, reuse: bool) -> std::io::Result<()>;
    #[cfg(target_os = "linux")]
    fn set_recv_buffer_size(&self, size: usize) -> std::io::Result<()>;
    #[cfg(target_os = "linux")]
    fn set_send_buffer_size(&self, size: usize) -> std::io::Result<()>;
}

impl SocketOptions for socket2::SockRef<'_> {
    fn set_nodelay(&self, nodelay: bool) -> std::io::Result<()> {
        (**self).set_nodelay(nodelay)
    }

    fn set_keepalive(&self, keepalive: bool) -> std::io::Result<()> {
        (**self).set_keepalive(keepalive)
    }

    fn set_reuse_address(&self, reuse: bool) -> std::io::Result<()> {
        (**self).set_reuse_address(reuse)
    }

    #[cfg(target_os = "linux")]
    fn set_recv_buffer_size(&self, size: usize) -> std::io::Result<()> {
        (**self).set_recv_buffer_size(size)
    }

    #[cfg(target_os = "linux")]
    fn set_send_buffer_size(&self, size: usize) -> std::io::Result<()> {
        (**self).set_send_buffer_size(size)
    }
}

pub(crate) fn apply_socket_options<O: SocketOptions + ?Sized>(sock: &O) -> std::io::Result<()> {
    sock.set_nodelay(true)?;
    sock.set_keepalive(true)?;
    let _ = sock.set_reuse_address(true);
    #[cfg(target_os = "linux")]
    {
        // Buffer tuning is best-effort; ignore failures.
        let _ = sock.set_recv_buffer_size(1 << 20);
        let _ = sock.set_send_buffer_size(1 << 20);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::http::models::HttpVersion;

    #[test]
    fn defaults_match_transport_policy() {
        let c = ClientConfig::default();
        assert!(c.ssl_verify);
        assert!(c.prepared_tls.is_none());
        assert_eq!(c.http_version, HttpVersion::Auto);
        assert_eq!(c.connect_timeout_ns, None);
        assert_eq!(c.max_connect_retries, 0);
        assert_eq!(c.connect_retry_backoff_ns, 0);
        assert_eq!(c.request_timeout_ns, None);
        assert_eq!(c.total_timeout_ns, None);
        assert_eq!(c.max_response_body_bytes, None);
        assert_eq!(c.keepalive_ns, Some(300_000_000_000));
        assert_eq!(c.max_connections_per_origin, 2_500);
        assert!(c.use_dns_cache);
        assert_eq!(c.dns_cache_ttl_ns, Some(300_000_000_000));
        assert!(!c.collect_trace_chunks);
    }
}
