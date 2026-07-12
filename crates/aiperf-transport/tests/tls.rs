// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-loopback parity for Python's SSL verification switch, source-tested in
//! `tests/unit/transports/test_tcp_connector.py:337-445`.

mod common;

use std::convert::Infallible;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::{ErrorKind, HttpVersion, RequestConfig};
use aiperf_transport::transport::http_transport::HttpTransport;
use aiperf_transport::{Clock, RealClock};
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

async fn spawn_untrusted_https() -> (String, tokio::task::JoinHandle<()>) {
    let certificates = CertificateDer::pem_slice_iter(CHAIN.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let key = PrivateKeyDer::from_pem_slice(KEY.as_bytes()).unwrap();
    let mut server_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certificates, key)
        .unwrap();
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
    (format!("https://127.0.0.1:{}/health", address.port()), task)
}

#[test]
fn ssl_verify_rejects_untrusted_chain_and_no_verify_accepts_it() {
    run_local(async {
        let (url, server) = spawn_untrusted_https().await;
        let request = RequestConfig::new(url);
        let verified_clock: Rc<dyn Clock> = RealClock::new();
        let verified = HttpTransport::new(
            verified_clock,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                ssl_verify: true,
                ..ClientConfig::default()
            },
        )
        .get(&request)
        .await;
        assert_eq!(
            verified.error.as_ref().map(|error| error.kind),
            Some(ErrorKind::Connect)
        );
        assert!(verified.status.is_none());

        let insecure_clock: Rc<dyn Clock> = RealClock::new();
        let insecure = HttpTransport::new(
            insecure_clock,
            ClientConfig {
                http_version: HttpVersion::Http1Only,
                ssl_verify: false,
                ..ClientConfig::default()
            },
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
