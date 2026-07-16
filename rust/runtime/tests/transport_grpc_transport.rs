// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end native Tonic transport tests against an in-process OIP server.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::endpoints::EndpointId;
use aiperf_runtime::transport_grpc::proto::model_infer_response::InferOutputTensor;
use aiperf_runtime::transport_grpc::proto::{
    InferTensorContents, ModelInferRequest, ModelInferResponse, ModelReadyRequest,
    ModelReadyResponse, ModelStreamInferResponse,
};
use aiperf_runtime::transport_grpc::{
    ConnectionReuseStrategy, GrpcBindingRegistry, GrpcClientConfig, GrpcErrorKind,
    GrpcRequestConfig, GrpcTransport,
};
use bytes::{Buf, BufMut, Bytes};
use futures::Stream;
use prost::Message;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::body::Body;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::codegen::{Body as HttpBody, BoxFuture, Service, StdError};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Code, Request, Response, Status};

#[derive(Clone, Copy, Debug, Default)]
struct RawCodec;

#[derive(Clone, Copy, Debug, Default)]
struct RawEncoder;

#[derive(Clone, Copy, Debug, Default)]
struct RawDecoder;

impl Codec for RawCodec {
    type Encode = Bytes;
    type Decode = Bytes;
    type Encoder = RawEncoder;
    type Decoder = RawDecoder;

    fn encoder(&mut self) -> Self::Encoder {
        RawEncoder
    }

    fn decoder(&mut self) -> Self::Decoder {
        RawDecoder
    }
}

impl Encoder for RawEncoder {
    type Item = Bytes;
    type Error = Status;

    fn encode(&mut self, item: Bytes, destination: &mut EncodeBuf<'_>) -> Result<(), Status> {
        destination.put(item);
        Ok(())
    }
}

impl Decoder for RawDecoder {
    type Item = Bytes;
    type Error = Status;

    fn decode(&mut self, source: &mut DecodeBuf<'_>) -> Result<Option<Bytes>, Status> {
        let remaining = source.remaining();
        Ok(Some(source.copy_to_bytes(remaining)))
    }
}

#[derive(Debug, Default)]
struct ServerState {
    metadata: Mutex<Vec<BTreeMap<String, String>>>,
}

#[derive(Clone, Debug)]
struct MockOipService {
    state: Arc<ServerState>,
}

impl<B> Service<http::Request<B>> for MockOipService
where
    B: HttpBody + Send + 'static,
    B::Error: Into<StdError> + Send + 'static,
{
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        match request.uri().path() {
            "/inference.GRPCInferenceService/ModelInfer" => {
                let method = ModelInferSvc(self.state.clone());
                Box::pin(async move {
                    Ok(tonic::server::Grpc::new(RawCodec)
                        .unary(method, request)
                        .await)
                })
            }
            "/inference.GRPCInferenceService/ModelStreamInfer" => {
                let method = ModelStreamInferSvc(self.state.clone());
                Box::pin(async move {
                    Ok(tonic::server::Grpc::new(RawCodec)
                        .server_streaming(method, request)
                        .await)
                })
            }
            "/inference.GRPCInferenceService/ModelReady" => {
                let method = ModelReadySvc(self.state.clone());
                Box::pin(async move {
                    Ok(tonic::server::Grpc::new(RawCodec)
                        .unary(method, request)
                        .await)
                })
            }
            _ => Box::pin(async move {
                let mut response = http::Response::new(Body::default());
                response
                    .headers_mut()
                    .insert(Status::GRPC_STATUS, (Code::Unimplemented as i32).into());
                response.headers_mut().insert(
                    http::header::CONTENT_TYPE,
                    tonic::metadata::GRPC_CONTENT_TYPE,
                );
                Ok(response)
            }),
        }
    }
}

impl NamedService for MockOipService {
    const NAME: &'static str = "inference.GRPCInferenceService";
}

struct ModelInferSvc(Arc<ServerState>);

impl UnaryService<Bytes> for ModelInferSvc {
    type Response = Bytes;
    type Future = BoxFuture<Response<Self::Response>, Status>;

    fn call(&mut self, request: Request<Bytes>) -> Self::Future {
        let state = self.0.clone();
        Box::pin(async move {
            record_metadata(&state, request.metadata());
            let request = ModelInferRequest::decode(request.into_inner())
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            match request.model_name.as_str() {
                "rpc-error" => return Err(Status::resource_exhausted("capacity exhausted")),
                "slow" => tokio::time::sleep(Duration::from_secs(5)).await,
                _ => {}
            }
            Ok(Response::new(Bytes::from(
                infer_response("unary", &request).encode_to_vec(),
            )))
        })
    }
}

struct ModelReadySvc(Arc<ServerState>);

impl UnaryService<Bytes> for ModelReadySvc {
    type Response = Bytes;
    type Future = BoxFuture<Response<Self::Response>, Status>;

    fn call(&mut self, request: Request<Bytes>) -> Self::Future {
        let state = self.0.clone();
        Box::pin(async move {
            record_metadata(&state, request.metadata());
            let request = ModelReadyRequest::decode(request.into_inner())
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            Ok(Response::new(Bytes::from(
                ModelReadyResponse {
                    ready: request.name != "not-ready",
                }
                .encode_to_vec(),
            )))
        })
    }
}

struct ModelStreamInferSvc(Arc<ServerState>);

impl ServerStreamingService<Bytes> for ModelStreamInferSvc {
    type Response = Bytes;
    type ResponseStream = Pin<Box<dyn Stream<Item = Result<Bytes, Status>> + Send + 'static>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Bytes>) -> Self::Future {
        let state = self.0.clone();
        Box::pin(async move {
            record_metadata(&state, request.metadata());
            let request = ModelInferRequest::decode(request.into_inner())
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            if request.model_name == "slow-stream" {
                let stream = futures::stream::pending();
                return Ok(Response::new(Box::pin(stream) as Self::ResponseStream));
            }
            let mut messages = vec![Ok(stream_response("", &request))];
            if request.model_name == "stream-error" {
                messages.push(Ok(Bytes::from(
                    ModelStreamInferResponse {
                        error_message: "stream backend failed".to_string(),
                        infer_response: None,
                    }
                    .encode_to_vec(),
                )));
            } else {
                messages.push(Ok(stream_response("token", &request)));
            }
            Ok(Response::new(
                Box::pin(futures::stream::iter(messages)) as Self::ResponseStream
            ))
        })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn native_tonic_transport_covers_unary_streaming_reuse_metadata_readiness_and_errors() {
    let (url, state, shutdown, server) = start_server().await;
    let transport = GrpcTransport::new(RealClock::new(), GrpcClientConfig::default(), [url])
        .unwrap()
        .with_user_agent("aiperf-test/1")
        .with_session_header("x-session-id");
    let registry = GrpcBindingRegistry::builtin().unwrap();
    let binding = registry
        .prepare(&EndpointId::new("kserve_v2_infer").unwrap())
        .unwrap();
    let payload = json!({
        "inputs": [{
            "name": "text_input", "shape": [1], "datatype": "BYTES", "data": ["prompt"]
        }]
    });

    let request = GrpcRequestConfig::new("model")
        .metadata("Authorization", "Bearer secret")
        .request_id("request-1")
        .correlation_id("session-1");
    let mut ignored = |_: i64, _: &serde_json::Value| true;
    let unary = transport
        .send_request(binding.as_ref(), &request, &payload, &mut ignored)
        .await;
    assert_eq!(unary.status, Some(200));
    assert_eq!(
        unary.responses[0].json["outputs"][0]["data"],
        json!(["unary"])
    );
    assert_eq!(unary.trace.trace_type, "grpc");
    assert_eq!(unary.trace.grpc_status_code, Some(0));
    assert!(unary.trace.request_send_end_ns.is_some());
    assert_eq!(transport.pooled_channel_count(), 1);

    let second = transport
        .send_request(binding.as_ref(), &request, &payload, &mut ignored)
        .await;
    assert_eq!(second.status, Some(200));
    assert!(second.trace.channel_reused_ns.is_some());
    assert_eq!(transport.pooled_channel_count(), 1);

    {
        let metadata = state.metadata.lock().unwrap();
        assert_eq!(metadata[0]["authorization"], "Bearer secret");
        assert!(metadata[0]["user-agent"].starts_with("aiperf-test/1"));
        assert_eq!(metadata[0]["x-request-id"], "request-1");
        assert_eq!(metadata[0]["x-correlation-id"], "session-1");
        assert_eq!(metadata[0]["x-session-id"], "session-1");
    }

    assert!(
        transport
            .model_ready(binding.as_ref(), "model", None, BTreeMap::new())
            .await
            .unwrap()
    );
    assert!(
        !transport
            .model_ready(binding.as_ref(), "not-ready", None, BTreeMap::new())
            .await
            .unwrap()
    );

    let streaming = GrpcRequestConfig::new("model").streaming(true);
    let mut filter_calls = 0;
    let mut filter = |_: i64, response: &serde_json::Value| {
        filter_calls += 1;
        response["outputs"][0]["data"][0] == "token"
    };
    let stream = transport
        .send_request(binding.as_ref(), &streaming, &payload, &mut filter)
        .await;
    assert_eq!(stream.status, Some(200));
    assert_eq!(stream.responses.len(), 2);
    assert_eq!(filter_calls, 2);
    assert_eq!(stream.trace.response_chunks_count, 2);

    let stream_error = GrpcRequestConfig::new("stream-error").streaming(true);
    let mut ignored = |_: i64, _: &serde_json::Value| true;
    let stream = transport
        .send_request(binding.as_ref(), &stream_error, &payload, &mut ignored)
        .await;
    assert_eq!(stream.status, Some(500));
    assert_eq!(stream.responses.len(), 1);
    assert_eq!(stream.error.as_ref().unwrap().kind, GrpcErrorKind::Stream);
    assert!(stream.trace.response_receive_end_ns.is_some());

    let sticky = GrpcRequestConfig::new("model")
        .reuse(ConnectionReuseStrategy::StickyUserSessions)
        .correlation_id("sticky")
        .final_turn(false);
    let sticky_first = transport
        .send_request(binding.as_ref(), &sticky, &payload, &mut ignored)
        .await;
    assert_eq!(sticky_first.status, Some(200));
    assert_eq!(transport.sticky_channel_count(), 1);
    let sticky_final = sticky.clone().final_turn(true);
    let sticky_last = transport
        .send_request(binding.as_ref(), &sticky_final, &payload, &mut ignored)
        .await;
    assert_eq!(sticky_last.status, Some(200));
    assert!(sticky_last.trace.channel_reused_ns.is_some());
    assert_eq!(transport.sticky_channel_count(), 0);

    let rpc_error = transport
        .send_request(
            binding.as_ref(),
            &GrpcRequestConfig::new("rpc-error"),
            &payload,
            &mut ignored,
        )
        .await;
    assert_eq!(rpc_error.status, Some(429));
    assert_eq!(rpc_error.trace.grpc_status_code, Some(8));
    assert_eq!(
        rpc_error.trace.response_reason.as_deref(),
        Some("RESOURCE_EXHAUSTED")
    );
    assert_eq!(rpc_error.error.as_ref().unwrap().kind, GrpcErrorKind::Rpc);

    let cancelled = transport
        .send_request(
            binding.as_ref(),
            &GrpcRequestConfig::new("slow").cancel_after_ns(20_000_000),
            &payload,
            &mut ignored,
        )
        .await;
    assert_eq!(cancelled.status, Some(499));
    assert!(cancelled.cancellation_ns.is_some());
    assert_eq!(
        cancelled.error.as_ref().unwrap().kind,
        GrpcErrorKind::RequestCancellation
    );

    let _ = shutdown.send(());
    server.await.unwrap();
}

fn infer_response(text: &str, request: &ModelInferRequest) -> ModelInferResponse {
    ModelInferResponse {
        model_name: request.model_name.clone(),
        id: request.id.clone(),
        outputs: vec![InferOutputTensor {
            name: "text_output".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![1],
            contents: Some(InferTensorContents {
                bytes_contents: vec![text.as_bytes().to_vec()],
                ..InferTensorContents::default()
            }),
            ..InferOutputTensor::default()
        }],
        ..ModelInferResponse::default()
    }
}

fn stream_response(text: &str, request: &ModelInferRequest) -> Bytes {
    Bytes::from(
        ModelStreamInferResponse {
            error_message: String::new(),
            infer_response: Some(infer_response(text, request)),
        }
        .encode_to_vec(),
    )
}

fn record_metadata(state: &ServerState, metadata: &tonic::metadata::MetadataMap) {
    let values = [
        "authorization",
        "user-agent",
        "x-request-id",
        "x-correlation-id",
        "x-session-id",
    ]
    .into_iter()
    .filter_map(|name| {
        metadata
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(|value| (name.to_string(), value.to_string()))
    })
    .collect();
    state.metadata.lock().unwrap().push(values);
}

async fn start_server() -> (
    String,
    Arc<ServerState>,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let state = Arc::new(ServerState::default());
    let service = MockOipService {
        state: state.clone(),
    };
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });
    (format!("grpc://{address}"), state, shutdown_tx, server)
}
