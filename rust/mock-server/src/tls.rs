// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TLS/HTTPS termination for the mock server's HTTP and gRPC listeners.
//!
//! The mock has always served cleartext. This module adds an optional rustls
//! frontend so it is a valid target for AIPerf's HTTPS/`grpcs` transports:
//! the runner's HTTP client
//! (`aiperf_runtime::transport_http::client::connection::rustls_config`) negotiates
//! ALPN `h2` then `http/1.1` and, with `endpoint.ssl_verify=false`, installs a
//! `NoCertificateVerification` verifier — so a self-signed cert is accepted and
//! the server side only needs to advertise the same two ALPN protocols. The
//! tonic `grpcs` client (`aiperf_runtime::transport_grpc::transport`) negotiates `h2`.
//!
//! Provider selection mirrors the runner: rustls cannot infer a process-global
//! crypto provider when both `aws-lc-rs` and `ring` are linked (the full runner
//! links both), so every `ServerConfig` is built with an explicit
//! `aws_lc_rs::default_provider()` — the same provider the client's
//! `rustls_config` selects.
//!
//! Extensibility: the acceptor is a `tokio_rustls::TlsAcceptor` shared by both
//! the HTTP ([`serve_http`]) and gRPC ([`crate::grpc::serve_grpc`]) accept
//! loops, so a future third listener (or mTLS via a client-auth verifier) is a
//! new `ServerConfig` builder arm here, not a change at each call site.

use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio_rustls::TlsAcceptor;
use tower::Service;

use crate::config::MockServerConfig;
use crate::listener::LISTEN_BACKLOG;

/// ALPN protocols advertised by every TLS listener, in server-preference order.
/// `h2` first (HTTP/2 over TLS, and the only protocol tonic `grpcs` clients
/// negotiate) then `http/1.1`, matching the runner client's
/// `alpn_protocols = vec![b"h2", b"http/1.1"]`.
const ALPN_PROTOCOLS: &[&[u8]] = &[b"h2", b"http/1.1"];

/// Subject-alternative names baked into a `--tls-self-signed` certificate.
/// Both loopback forms the runner may dial (`https://127.0.0.1:...` and
/// `https://localhost:...`); with `ssl_verify=false` the SANs are not checked,
/// but a rustls client that *does* verify (the e2e's danger-accept fallback
/// aside) still resolves the hostname.
const SELF_SIGNED_SANS: &[&str] = &["127.0.0.1", "localhost"];

/// Build the TLS acceptor implied by `config`, or `None` for cleartext.
///
/// Precedence: an explicit `--tls-cert`/`--tls-key` pair wins; otherwise
/// `--tls-self-signed` mints an in-memory cert. Supplying only one of
/// cert/key is a configuration error. Returns `Ok(None)` when no TLS flag is
/// set (the unchanged cleartext path).
pub fn build_acceptor(config: &MockServerConfig) -> anyhow::Result<Option<TlsAcceptor>> {
    match (&config.tls_cert, &config.tls_key) {
        (Some(cert), Some(key)) => Ok(Some(acceptor_from_files(cert, key)?)),
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("--tls-cert and --tls-key must be provided together")
        }
        (None, None) if config.tls_self_signed => Ok(Some(self_signed_acceptor()?)),
        (None, None) => Ok(None),
    }
}

/// A [`TlsAcceptor`] backed by a freshly-minted in-memory self-signed cert for
/// `127.0.0.1`/`localhost`. Public so integration/e2e tests can stand up an
/// HTTPS mock without a cert file on disk, driving the exact server-side path
/// `--tls-self-signed` uses.
pub fn self_signed_acceptor() -> anyhow::Result<TlsAcceptor> {
    let (certs, key) = self_signed_material()?;
    acceptor_from_material(certs, key)
}

/// A [`TlsAcceptor`] built from PEM cert-chain and private-key files.
pub fn acceptor_from_files(cert_path: &str, key_path: &str) -> anyhow::Result<TlsAcceptor> {
    let (certs, key) = load_pem(Path::new(cert_path), Path::new(key_path))?;
    acceptor_from_material(certs, key)
}

/// Generate a self-signed cert + PKCS#8 key for [`SELF_SIGNED_SANS`].
fn self_signed_material() -> anyhow::Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)>
{
    let sans: Vec<String> = SELF_SIGNED_SANS.iter().map(|s| s.to_string()).collect();
    let generated =
        rcgen::generate_simple_self_signed(sans).context("generate self-signed certificate")?;
    let cert = generated.cert.der().clone();
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(generated.key_pair.serialize_der()));
    Ok((vec![cert], key))
}

/// Parse a PEM cert-chain file and a PEM private-key file (PKCS#8, RSA/PKCS#1,
/// or SEC1) into DER material.
fn load_pem(
    cert_path: &Path,
    key_path: &Path,
) -> anyhow::Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    let cert_pem = std::fs::read(cert_path)
        .with_context(|| format!("read TLS cert {}", cert_path.display()))?;
    let certs = rustls_pemfile::certs(&mut BufReader::new(&cert_pem[..]))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("parse TLS cert PEM {}", cert_path.display()))?;
    if certs.is_empty() {
        anyhow::bail!("no certificates found in {}", cert_path.display());
    }

    let key_pem =
        std::fs::read(key_path).with_context(|| format!("read TLS key {}", key_path.display()))?;
    let key = rustls_pemfile::private_key(&mut BufReader::new(&key_pem[..]))
        .with_context(|| format!("parse TLS key PEM {}", key_path.display()))?
        .with_context(|| format!("no private key found in {}", key_path.display()))?;
    Ok((certs, key))
}

/// Assemble a rustls [`ServerConfig`] (ALPN `h2`+`http/1.1`, no client auth,
/// explicit aws-lc-rs provider) and wrap it as a [`TlsAcceptor`].
fn acceptor_from_material(
    certs: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
) -> anyhow::Result<TlsAcceptor> {
    // Explicit provider: with both aws-lc-rs and ring linked, rustls refuses to
    // pick a process-global default. Matches the client's `rustls_config`.
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut server_config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("aws-lc supports rustls safe default protocol versions")?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .context("install TLS certificate + key")?;
    server_config.alpn_protocols = ALPN_PROTOCOLS.iter().map(|p| p.to_vec()).collect();
    Ok(TlsAcceptor::from(Arc::new(server_config)))
}

/// The mock's HTTP accept loop, shared by the cleartext and TLS frontends.
///
/// Factored out of `main::serve` so the single connection-handling policy —
/// `TCP_NODELAY`, per-connection hyper auto HTTP/1+2 handshake, optional h2
/// `max_concurrent_streams`, upgrade support — is identical whether or not TLS
/// is terminated, and so integration/e2e tests can drive the same loop over a
/// self-signed acceptor. When `acceptor` is `Some`, each accepted TCP stream is
/// wrapped in a rustls handshake (ALPN selecting h2 vs http/1.1) before hyper
/// sees it; when `None`, cleartext is served exactly as before.
pub async fn serve_http(
    listener: tokio::net::TcpListener,
    router: axum::Router,
    acceptor: Option<TlsAcceptor>,
    max_concurrent_streams: u32,
) -> anyhow::Result<()> {
    let make_service = router.into_make_service();
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("accept error: {e}");
                continue;
            }
        };
        // Disable Nagle for low-latency streaming.
        let _ = stream.set_nodelay(true);

        let tower_service = match make_service.clone().call(peer).await {
            Ok(svc) => svc,
            Err(e) => {
                tracing::warn!("make_service error: {e}");
                continue;
            }
        };
        let acceptor = acceptor.clone();

        tokio::spawn(async move {
            let hyper_service =
                hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                    tower_service.clone().call(req)
                });
            let mut builder = ConnBuilder::new(TokioExecutor::new());
            if max_concurrent_streams > 0 {
                builder
                    .http2()
                    .max_concurrent_streams(max_concurrent_streams);
            }
            // The TLS and cleartext arms serve structurally-identical hyper
            // connections over different stream types; the handshake failure
            // handling is the same WARN the cleartext path already emits.
            match acceptor {
                Some(acceptor) => {
                    let tls_stream = match acceptor.accept(stream).await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(%peer, "TLS handshake error: {e}");
                            return;
                        }
                    };
                    let io = TokioIo::new(tls_stream);
                    if let Err(e) = builder
                        .serve_connection_with_upgrades(io, hyper_service)
                        .await
                    {
                        tracing::warn!(%peer, "connection error: {e}");
                    }
                }
                None => {
                    let io = TokioIo::new(stream);
                    if let Err(e) = builder
                        .serve_connection_with_upgrades(io, hyper_service)
                        .await
                    {
                        tracing::warn!(%peer, "connection error: {e}");
                    }
                }
            }
        });
    }
}

/// Build a listener bound to `addr` with the same tuning as the cleartext path.
/// Convenience for tests that pair [`serve_http`] with a fresh listener.
pub fn bind(addr: std::net::SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    crate::listener::build_listener(addr)
}

/// The listen backlog, re-exported so a caller pairing [`serve_http`] with its
/// own listener can log the same value `main::serve` does.
pub const BACKLOG: i32 = LISTEN_BACKLOG;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_signed_material_yields_one_cert_and_key() {
        let (certs, _key) = self_signed_material().expect("generate self-signed");
        assert_eq!(certs.len(), 1, "one leaf certificate");
    }

    #[test]
    fn self_signed_acceptor_builds() {
        // Exercises the full aws-lc-rs ServerConfig assembly + ALPN wiring.
        self_signed_acceptor().expect("build self-signed acceptor");
    }

    #[test]
    fn build_acceptor_none_without_tls_flags() {
        let cfg = MockServerConfig::default();
        assert!(build_acceptor(&cfg).expect("no-tls").is_none());
    }

    #[test]
    fn build_acceptor_self_signed() {
        let cfg = MockServerConfig {
            tls_self_signed: true,
            ..MockServerConfig::default()
        };
        assert!(build_acceptor(&cfg).expect("self-signed").is_some());
    }

    #[test]
    fn build_acceptor_rejects_half_pair() {
        let cfg = MockServerConfig {
            tls_cert: Some("/tmp/cert.pem".to_string()),
            ..MockServerConfig::default()
        };
        assert!(build_acceptor(&cfg).is_err());
    }
}
