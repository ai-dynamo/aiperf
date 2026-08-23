// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "engine")]

mod common;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::rc::Rc;

use aiperf_runtime::transport::core::{ErrorKind, Response};
use aiperf_runtime::transport::http::client::http_client::HttpClient;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::{Clock, RealClock};
use bytes::Bytes;
use common::run_local;
use futures::stream;
use http::StatusCode;
use http_body::Frame;
use http_body_util::StreamBody;
use hyper::service::service_fn;
use hyper::{Request, Response as HttpResponse};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

struct ChunkedServer {
    base_url: String,
    accept_task: JoinHandle<()>,
}

impl ChunkedServer {
    async fn spawn(status: StatusCode, content_type: &'static str, chunks: Vec<Bytes>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let accept_task = tokio::task::spawn_local(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let chunks = chunks.clone();
                let service = service_fn(move |_request: Request<hyper::body::Incoming>| {
                    let chunks = chunks.clone();
                    async move {
                        let frames = chunks
                            .into_iter()
                            .map(|chunk| Ok::<_, Infallible>(Frame::data(chunk)));
                        Ok::<_, Infallible>(
                            HttpResponse::builder()
                                .status(status)
                                .header("content-type", content_type)
                                .body(StreamBody::new(stream::iter(frames)))
                                .unwrap(),
                        )
                    }
                });
                tokio::task::spawn_local(async move {
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
            }
        });
        Self {
            base_url: format!("http://{address}"),
            accept_task,
        }
    }
}

impl Drop for ChunkedServer {
    fn drop(&mut self) {
        self.accept_task.abort();
    }
}

async fn request(
    server: &ChunkedServer,
    config: ClientConfig,
) -> aiperf_runtime::transport::core::RequestRecord {
    let clock: Rc<dyn Clock> = RealClock::new();
    let client = HttpClient::new(clock, config);
    let url = url::Url::parse(&format!("{}/metrics", server.base_url)).unwrap();
    client
        .request(&url, &BTreeMap::new(), Bytes::new(), false, |_| {})
        .await
}

#[test]
fn chunked_response_stops_at_the_incremental_body_limit() {
    run_local(async {
        let server = ChunkedServer::spawn(
            StatusCode::OK,
            "application/octet-stream",
            vec![
                Bytes::from_static(b"1234"),
                Bytes::from_static(b"5678"),
                Bytes::from_static(b"9"),
            ],
        )
        .await;
        let record = request(
            &server,
            ClientConfig {
                max_response_body_bytes: Some(8),
                ..ClientConfig::default()
            },
        )
        .await;

        assert_eq!(record.status, Some(200));
        assert!(record.responses.is_empty());
        let error = record.error.expect("oversize response must fail");
        assert_eq!(error.kind, ErrorKind::Other);
        assert!(error.message.contains("configured 8-byte limit"));
        assert!(error.message.contains("9 bytes"));
        assert_eq!(
            record.trace.unwrap().response_bytes_total,
            9,
            "the offending wire chunk remains observable without being retained"
        );
    });
}

#[test]
fn metric_looking_http_500_retains_exact_body_and_typed_http_error() {
    run_local(async {
        const BODY: &[u8] = b"# TYPE error_metric gauge\nerror_metric 42\n";
        let server = ChunkedServer::spawn(
            StatusCode::INTERNAL_SERVER_ERROR,
            "text/plain; version=0.0.4",
            vec![
                Bytes::from_static(&BODY[..17]),
                Bytes::from_static(&BODY[17..]),
            ],
        )
        .await;
        let record = request(
            &server,
            ClientConfig {
                max_response_body_bytes: Some(1024),
                ..ClientConfig::default()
            },
        )
        .await;

        assert_eq!(record.status, Some(500));
        let error = record
            .error
            .as_ref()
            .expect("HTTP 500 must remain an error");
        assert_eq!(error.kind, ErrorKind::Http);
        assert_eq!(error.code, Some(500));
        assert_eq!(error.message.as_bytes(), BODY);
        assert_eq!(record.responses.len(), 1);
        let Response::Text(response) = &record.responses[0] else {
            panic!("non-2xx body must be retained as exact text response bytes")
        };
        assert_eq!(response.body.as_ref(), BODY);
        assert_eq!(response.text.as_bytes(), BODY);
        assert_eq!(
            response.content_type.as_deref(),
            Some("text/plain; version=0.0.4")
        );
    });
}
