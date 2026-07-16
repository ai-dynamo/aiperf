// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end native Tonic transport coverage for Riva ASR bidi streaming.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::endpoints::EndpointId;
use aiperf_runtime::transport::grpc::riva_proto::streaming_recognize_request::StreamingRequest;
use aiperf_runtime::transport::grpc::riva_proto::{
    SpeechRecognitionAlternative, StreamingRecognitionResult, StreamingRecognizeRequest,
    StreamingRecognizeResponse,
};
use aiperf_runtime::transport::grpc::{
    GrpcBindingRegistry, GrpcClientConfig, GrpcRequestConfig, GrpcTransport,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
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
use tonic::server::{NamedService, StreamingService};
use tonic::{Code, Request, Response, Status, Streaming};

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
    requests: Mutex<Vec<StreamingRecognizeRequest>>,
}

#[derive(Clone, Debug)]
struct MockRivaAsrService {
    state: Arc<ServerState>,
}

impl<B> Service<http::Request<B>> for MockRivaAsrService
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
            "/nvidia.riva.asr.RivaSpeechRecognition/StreamingRecognize" => {
                let method = StreamingRecognizeSvc(self.state.clone());
                Box::pin(async move {
                    Ok(tonic::server::Grpc::new(RawCodec)
                        .streaming(method, request)
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

impl NamedService for MockRivaAsrService {
    const NAME: &'static str = "nvidia.riva.asr.RivaSpeechRecognition";
}

struct StreamingRecognizeSvc(Arc<ServerState>);

impl StreamingService<Bytes> for StreamingRecognizeSvc {
    type Response = Bytes;
    type ResponseStream = Pin<Box<dyn Stream<Item = Result<Bytes, Status>> + Send + 'static>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Streaming<Bytes>>) -> Self::Future {
        let state = self.0.clone();
        Box::pin(async move {
            let mut inbound = request.into_inner();
            let mut decoded = Vec::new();
            while let Some(message) = inbound.message().await? {
                decoded.push(
                    StreamingRecognizeRequest::decode(message)
                        .map_err(|error| Status::invalid_argument(error.to_string()))?,
                );
            }
            *state.requests.lock().unwrap() = decoded;
            let messages = [
                response("interim", false, 0.5),
                response("final transcript", true, 1.0),
            ]
            .into_iter()
            .map(|response| Ok(Bytes::from(response.encode_to_vec())));
            Ok(Response::new(
                Box::pin(futures::stream::iter(messages)) as Self::ResponseStream
            ))
        })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn native_transport_sends_riva_config_first_bidi_stream_and_records_every_chunk() {
    let (url, state, shutdown, server) = start_server().await;
    let transport =
        GrpcTransport::new(RealClock::new(), GrpcClientConfig::default(), [url]).unwrap();
    let binding = GrpcBindingRegistry::builtin()
        .unwrap()
        .prepare(&EndpointId::new("riva_asr").unwrap())
        .unwrap();
    let payload = json!({
        "language_code": "en-US",
        "sample_rate_hertz": 16000,
        "encoding": "LINEAR_PCM",
        "interim_results": true,
        "audio_chunks": [STANDARD.encode([1_u8, 2]), STANDARD.encode([3_u8, 4])],
    });
    let request = GrpcRequestConfig::new("asr-model")
        .streaming(true)
        .request_id("request-id");
    let mut accepted = Vec::new();
    let mut filter = |_: i64, response: &serde_json::Value| {
        accepted.push(response["transcript"].as_str().unwrap().to_string());
        response["is_final"] == true
    };
    let record = transport
        .send_request(binding.as_ref(), &request, &payload, &mut filter)
        .await;

    assert_eq!(record.status, Some(200), "{record:?}");
    assert!(record.error.is_none());
    assert_eq!(record.responses.len(), 2);
    assert_eq!(record.responses[1].json["transcript"], "final transcript");
    assert_eq!(accepted, ["interim", "final transcript"]);
    assert_eq!(record.request_messages.len(), 3);
    assert_eq!(record.trace.request_chunks_count, 3);
    assert_eq!(record.trace.response_chunks_count, 2);
    assert_eq!(
        record.trace.request_bytes_total,
        record
            .request_messages
            .iter()
            .map(|message| message.len() as u64)
            .sum::<u64>()
    );

    {
        let requests = state.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(matches!(
            requests[0].streaming_request,
            Some(StreamingRequest::StreamingConfig(_))
        ));
        assert_eq!(requests[0].id.as_ref().unwrap().value, "request-id");
        assert!(matches!(
            &requests[1].streaming_request,
            Some(StreamingRequest::AudioContent(audio)) if audio == &[1, 2]
        ));
        assert!(matches!(
            &requests[2].streaming_request,
            Some(StreamingRequest::AudioContent(audio)) if audio == &[3, 4]
        ));
    }

    let _ = shutdown.send(());
    server.await.unwrap();
}

fn response(transcript: &str, is_final: bool, stability: f32) -> StreamingRecognizeResponse {
    StreamingRecognizeResponse {
        results: vec![StreamingRecognitionResult {
            alternatives: vec![SpeechRecognitionAlternative {
                transcript: transcript.to_string(),
                confidence: 1.0,
            }],
            is_final,
            stability,
        }],
        id: None,
    }
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
    let service = MockRivaAsrService {
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
