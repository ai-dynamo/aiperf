// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-loopback TLS verification tests.

mod common;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_runtime::transport::core::ErrorKind;
use aiperf_runtime::transport::http::client::pool::ConnectionPool;
use aiperf_runtime::transport::http::client::resolver::{CachingDnsResolver, HostLookup};
use aiperf_runtime::transport::http::config::{ClientConfig, PreparedTlsClientConfig};
use aiperf_runtime::transport::http::models::{HttpVersion, RequestConfig};
use aiperf_runtime::transport::http::transport::http_transport::HttpTransport;
use aiperf_runtime::transport::http::{Clock, RealClock};
use bytes::Bytes;
use common::run_local;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

// Test-only certificate chain and key from tokio-rustls's public test fixture.
// The chain is intentionally outside the web PKI trust store.
const CHAIN: &str = r#"-----BEGIN CERTIFICATE-----
MIIBszCCAVmgAwIBAgIUUg3keFcU1xXWK8BNVb1KynPulV8wCgYIKoZIzj0EAwIw
JjEkMCIGA1UEAwwbUnVzdGxzIFJvYnVzdCBSb290IC0gUnVuZyAyMCAXDTc1MDEw
MTAwMDAwMFoYDzQwOTYwMTAxMDAwMDAwWjAhMR8wHQYDVQQDDBZyY2dlbiBzZWxm
IHNpZ25lZCBjZXJ0MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEud6w4gtZ0xbw
J3E69SSMy5TZfdIifl9L5ZY+hgEe4UiUsBWS32f6Y5NR5Jo8FO1f6o13b3+FvVHR
EHCGdvppL6NoMGYwFQYDVR0RBA4wDIIKZm9vYmFyLmNvbTAdBgNVHSUEFjAUBggr
BgEFBQcDAQYIKwYBBQUHAwIwHQYDVR0OBBYEFELvxbj5tD75n4pYFvJyr+c8qVEi
MA8GA1UdEwEB/wQFMAMBAQAwCgYIKoZIzj0EAwIDSAAwRQIhALxSSdUsrRFnwNMu
/doBqI8i8u5HdohVAheFTDwObkOMAiASSjULUtkWSD15u/7Sr01Wm9J1MpqW1pob
BVqU3CNRlA==
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIIBiTCCATCgAwIBAgIUHWiVYIvMMWoZEFYvSz46COf2FqowCgYIKoZIzj0EAwIw
HTEbMBkGA1UEAwwSUnVzdGxzIFJvYnVzdCBSb290MCAXDTc1MDEwMTAwMDAwMFoY
DzQwOTYwMTAxMDAwMDAwWjAmMSQwIgYDVQQDDBtSdXN0bHMgUm9idXN0IFJvb3Qg
LSBSdW5nIDIwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAATAOCcBD7dXjmAZ3te5
D47cCJ9ec93PWv7BKYIL826CJsKfXQOGrBTthLm77hXLhHu6uv8E5QXNLZpfowLQ
Do1ao0MwQTAPBgNVHQ8BAf8EBQMDB4QAMB0GA1UdDgQWBBRdza76r11Ok9vRmlg6
Nn/wL/N+jTAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49BAMCA0cAMEQCIFmZrXeK
hnfkahocvkhhNT3cDv1LWf6WBoFaCiBwZXFPAiARaKRiSCMG7PCHmSqFe82TBVmL
odHGogAVax1Dh/aYAA==
-----END CERTIFICATE-----
"#;

const KEY: &str = r#"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgTbAQpfjAT46fgF4B
mP15n37woNG5ZNJmwcqsred/7tmhRANCAAS53rDiC1nTFvAncTr1JIzLlNl90iJ+
X0vllj6GAR7hSJSwFZLfZ/pjk1HkmjwU7V/qjXdvf4W9UdEQcIZ2+mkv
-----END PRIVATE KEY-----
"#;

struct FixedLookup(SocketAddr);

#[async_trait::async_trait(?Send)]
impl HostLookup for FixedLookup {
    async fn lookup(
        &self,
        _host: &str,
        _port: u16,
    ) -> Result<SocketAddr, aiperf_runtime::transport::core::ErrorDetails> {
        Ok(self.0)
    }
}

fn transport_with_fixed_lookup(
    clock: Rc<dyn Clock>,
    config: ClientConfig,
    address: SocketAddr,
) -> HttpTransport {
    let resolver = CachingDnsResolver::new(Rc::new(FixedLookup(address)));
    HttpTransport::with_connection_manager(
        clock,
        config,
        Rc::new(ConnectionPool::with_resolver(Rc::new(resolver))),
    )
}

async fn spawn_untrusted_https(
    require_client_auth: bool,
) -> (String, SocketAddr, tokio::task::JoinHandle<()>) {
    // HTTP and gRPC features link different rustls crypto providers, so install
    // the HTTP provider explicitly for process-global test builders.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let certificates = CertificateDer::pem_slice_iter(CHAIN.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let key = PrivateKeyDer::from_pem_slice(KEY.as_bytes()).unwrap();
    let builder = rustls::ServerConfig::builder();
    let mut server_config = if require_client_auth {
        let mut roots = rustls::RootCertStore::empty();
        for certificate in CertificateDer::pem_slice_iter(CHAIN.as_bytes()) {
            roots.add(certificate.unwrap()).unwrap();
        }
        let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots))
            .build()
            .unwrap();
        builder
            .with_client_cert_verifier(verifier)
            .with_single_cert(certificates, key)
            .unwrap()
    } else {
        builder
            .with_no_client_auth()
            .with_single_cert(certificates, key)
            .unwrap()
    };
    server_config.alpn_protocols = vec![b"http/1.1".to_vec()];
    let acceptor = TlsAcceptor::from(Arc::new(server_config));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let task = tokio::task::spawn_local(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let acceptor = acceptor.clone();
            tokio::task::spawn_local(async move {
                let Ok(tls) = acceptor.accept(stream).await else {
                    return;
                };
                let service = service_fn(|_request: Request<hyper::body::Incoming>| async {
                    Ok::<_, Infallible>(
                        Response::builder()
                            .header("content-type", "application/json")
                            .body(Full::new(Bytes::from_static(b"{\"ok\":true}")))
                            .unwrap(),
                    )
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(tls), service)
                    .await;
            });
        }
    });
    (
        format!("https://foobar.com:{}/health", address.port()),
        address,
        task,
    )
}

#[test]
fn ssl_verify_rejects_untrusted_chain_and_no_verify_accepts_it() {
    run_local(async {
        let (url, address, server) = spawn_untrusted_https(false).await;
        let request = RequestConfig::new(url);
        let verified_clock: Rc<dyn Clock> = RealClock::new();
        let verified = transport_with_fixed_lookup(
            verified_clock,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                ssl_verify: true,
                ..ClientConfig::default()
            },
            address,
        )
        .get(&request)
        .await;
        assert_eq!(
            verified.error.as_ref().map(|error| error.kind),
            Some(ErrorKind::Connect)
        );
        assert!(verified.status.is_none());

        let insecure_clock: Rc<dyn Clock> = RealClock::new();
        let insecure = transport_with_fixed_lookup(
            insecure_clock,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                ssl_verify: false,
                ..ClientConfig::default()
            },
            address,
        )
        .get(&request)
        .await;
        assert!(
            !insecure.has_error(),
            "no-verify request failed: {:?}",
            insecure.error
        );
        assert_eq!(insecure.status, Some(200));

        server.abort();
    });
}

#[test]
fn provider_trust_roots_are_injected_into_the_live_rustls_connector() {
    run_local(async {
        let (url, address, server) = spawn_untrusted_https(false).await;
        let prepared =
            PreparedTlsClientConfig::from_provider_pem(Some(CHAIN.as_bytes()), None, None).unwrap();
        assert!(!format!("{prepared:?}").contains("BEGIN CERTIFICATE"));
        let transport = transport_with_fixed_lookup(
            RealClock::new(),
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                prepared_tls: Some(prepared),
                ..ClientConfig::default()
            },
            address,
        );
        let record = transport.get(&RequestConfig::new(url)).await;
        assert!(
            !record.has_error(),
            "custom-trust request failed: {:?}",
            record.error
        );
        assert_eq!(record.status, Some(200));
        server.abort();
    });
}

#[test]
fn provider_mtls_identity_is_required_and_injected() {
    run_local(async {
        let (url, address, server) = spawn_untrusted_https(true).await;
        let trust_only =
            PreparedTlsClientConfig::from_provider_pem(Some(CHAIN.as_bytes()), None, None).unwrap();
        let without_identity = transport_with_fixed_lookup(
            RealClock::new(),
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                prepared_tls: Some(trust_only),
                ..ClientConfig::default()
            },
            address,
        )
        .get(&RequestConfig::new(url.clone()))
        .await;
        assert!(without_identity.has_error());

        let mtls = PreparedTlsClientConfig::from_provider_pem(
            Some(CHAIN.as_bytes()),
            Some(CHAIN.as_bytes()),
            Some(KEY.as_bytes()),
        )
        .unwrap();
        let with_identity = transport_with_fixed_lookup(
            RealClock::new(),
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                prepared_tls: Some(mtls),
                ..ClientConfig::default()
            },
            address,
        )
        .get(&RequestConfig::new(url))
        .await;
        assert!(
            !with_identity.has_error(),
            "mTLS request failed: {:?}",
            with_identity.error
        );
        assert_eq!(with_identity.status, Some(200));
        server.abort();
    });
}
