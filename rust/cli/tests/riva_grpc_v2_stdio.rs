// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-process proof for protocol-v2 Riva ASR over native bidi gRPC.

use std::convert::Infallible;
use std::io::Write;
use std::pin::Pin;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use aiperf_runtime::transport::grpc::riva_proto::streaming_recognize_request::StreamingRequest;
use aiperf_runtime::transport::grpc::riva_proto::{
    SpeechRecognitionAlternative, StreamingRecognitionResult, StreamingRecognizeRequest,
    StreamingRecognizeResponse,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::{Buf, BufMut, Bytes};
use prost::Message;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::{Stream, wrappers::TcpListenerStream};
use tonic::body::Body;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::codegen::{Body as HttpBody, BoxFuture, Service, StdError};
use tonic::server::{NamedService, StreamingService};
use tonic::{Code, Request, Response, Status, Streaming};

const RIVA_ENDPOINTS: [&str; 9] = [
    "riva_analyze_entities",
    "riva_analyze_intent",
    "riva_asr",
    "riva_natural_query",
    "riva_punctuate_text",
    "riva_text_classify",
    "riva_token_classify",
    "riva_transform_text",
    "riva_tts",
];

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_aiperf")
}

fn one_json_line(output: &Output) -> Value {
    let lines = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(
        lines.len(),
        1,
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(lines[0]).unwrap()
}

fn capabilities() -> Value {
    // Capabilities is an in-process call now — one binary, no subprocess.
    serde_json::to_value(
        aiperf_cli::execute_mode::capabilities_catalog().expect("capabilities catalog"),
    )
    .expect("catalog to Value")
}

fn benchmark_run(legacy: Value) -> Value {
    let mut endpoint = legacy["resources"]["endpoints"]["profiles"][0].clone();
    endpoint.as_object_mut().unwrap().remove("id");
    json!({
        "benchmark_id": legacy["identity"]["benchmark_id"],
        "artifact_dir": legacy["artifact_target"],
        "random_seed": legacy["identity"]["random_seed"],
        "cfg": {
            "models": legacy["resources"]["models"],
            "endpoint": endpoint,
            "datasets": [legacy["workload"]["config"]["dataset"]],
            "phases": legacy["workload"]["config"]["phases"],
            "tokenizer": legacy["workload"]["config"]["tokenizer"],
            "transport": {"type": legacy["transport"]["type"]},
            "runtime": {"workers": legacy["workload"]["config"]["worker_count"]}
        }
    })
}

fn run_child(request: &Value) -> Output {
    let mut request = request.clone();
    request["run"] = benchmark_run(request["run"].take());
    let mut child = Command::new(binary())
        .arg("--execute")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&request).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn operation(operation: &str, target: &std::path::Path, url: &str) -> Value {
    // Valid mono 16-bit PCM WAV containing four silent 16 kHz samples.
    let wav = STANDARD.encode(
        b"RIFF,\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80>\0\0\0}\0\0\x02\0\x10\0data\x08\0\0\0\0\0\0\0\0\0\0\0",
    );
    json!({
        "protocol_version": 2,
        "operation": operation,
        "run": {
            "identity": {"benchmark_id": "native-riva-asr-v2", "random_seed": 11},
            "artifact_target": target,
            "transport": {"type": "grpc", "config": {}},
            "workload": {"type": "scheduled", "config": {
                "worker_count": 1,
                "dataset": {
                    "type": "file",
                    "format": "single_turn",
                    "sampling": "sequential",
                    "records": [{"audio": format!("wav,{wav}"), "output_length": 1}]
                },
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 1,
                    "concurrency": 1
                }]
            }},
            "resources": {
                "models": {
                    "strategy": "round_robin",
                    "items": [{"name": "fixture-riva-asr"}]
                },
                "endpoints": {"profiles": [{
                    "id": "default",
                    "type": "riva_asr",
                    "urls": [url],
                    "streaming": true,
                    "use_server_token_count": false,
                    "timeout_seconds": 10.0,
                    "connection_reuse": "pooled",
                    "headers": {"authorization": "Bearer fixture"},
                    "extra": {
                        "language_code": "en-US",
                        "sample_rate_hertz": 16000,
                        "encoding": "LINEAR_PCM",
                        "chunk_size": 16
                    },
                    "http2": false,
                    "wait_for_model_timeout": 0.0
                }]},
                "metrics": {},
                "artifacts": {},
                "sidecars": {}
            }
        }
    })
}

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

#[derive(Clone, Debug)]
struct RivaAsrService {
    requests: Arc<Mutex<Vec<Vec<StreamingRecognizeRequest>>>>,
}

impl<B> Service<http::Request<B>> for RivaAsrService
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
                let method = StreamingRecognizeSvc(self.requests.clone());
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

impl NamedService for RivaAsrService {
    const NAME: &'static str = "nvidia.riva.asr.RivaSpeechRecognition";
}

struct StreamingRecognizeSvc(Arc<Mutex<Vec<Vec<StreamingRecognizeRequest>>>>);

impl StreamingService<Bytes> for StreamingRecognizeSvc {
    type Response = Bytes;
    type ResponseStream = Pin<Box<dyn Stream<Item = Result<Bytes, Status>> + Send + 'static>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Streaming<Bytes>>) -> Self::Future {
        let captured = self.0.clone();
        Box::pin(async move {
            assert_eq!(
                request
                    .metadata()
                    .get("authorization")
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "Bearer fixture"
            );
            let mut inbound = request.into_inner();
            let mut messages = Vec::new();
            while let Some(message) = inbound.message().await? {
                messages.push(
                    StreamingRecognizeRequest::decode(message)
                        .map_err(|error| Status::invalid_argument(error.to_string()))?,
                );
            }
            captured.lock().unwrap().push(messages);
            let response = StreamingRecognizeResponse {
                results: vec![StreamingRecognitionResult {
                    alternatives: vec![SpeechRecognitionAlternative {
                        transcript: "native riva transcript".to_string(),
                        confidence: 1.0,
                    }],
                    is_final: true,
                    stability: 1.0,
                }],
                id: None,
            };
            Ok(Response::new(Box::pin(tokio_stream::iter([Ok(Bytes::from(
                response.encode_to_vec(),
            ))])) as Self::ResponseStream))
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runner_capabilities_validate_and_execute_native_riva_asr_bidi() {
    let (url, captured, shutdown, server) = start_server().await;
    let capabilities = capabilities();
    for endpoint in RIVA_ENDPOINTS {
        assert!(
            capabilities["endpoint"].get(endpoint).is_some(),
            "missing {endpoint} from {capabilities}"
        );
    }
    assert!(
        capabilities["transport"].get("grpc").is_some(),
        "{capabilities}"
    );

    let temporary = tempfile::tempdir().unwrap();
    let target = temporary.path().join("riva-run");
    let validation = operation("validate", &target, &url);
    let validation_output = tokio::task::spawn_blocking(move || run_child(&validation))
        .await
        .unwrap();
    let validation_response = one_json_line(&validation_output);
    assert!(
        validation_output.status.success(),
        "validation={validation_response} stderr={}",
        String::from_utf8_lossy(&validation_output.stderr)
    );
    assert_eq!(validation_response["event"], "run_validation");
    assert_eq!(validation_response["success"], true);
    assert!(!target.exists(), "validation created artifacts");

    let execution = operation("execute", &target, &url);
    let output = tokio::task::spawn_blocking(move || run_child(&execution))
        .await
        .unwrap();
    let terminal = one_json_line(&output);
    assert!(
        output.status.success(),
        "terminal={terminal} stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["provenance"]["transport"], "grpc");
    assert_eq!(terminal["provenance"]["transport"], "grpc");

    {
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].len() > 2, "expected config plus audio chunks");
        let Some(StreamingRequest::StreamingConfig(streaming)) = &requests[0][0].streaming_request
        else {
            panic!("runner did not send Riva ASR config first")
        };
        let config = streaming.config.as_ref().unwrap();
        assert_eq!(config.model, "fixture-riva-asr");
        assert_eq!(config.language_code, "en-US");
        assert_eq!(config.sample_rate_hertz, 16000);
        assert!(streaming.interim_results);
        assert!(requests[0][0].id.is_some());
        let audio = requests[0][1..]
            .iter()
            .flat_map(
                |message| match message.streaming_request.as_ref().unwrap() {
                    StreamingRequest::AudioContent(audio) => audio.as_slice(),
                    StreamingRequest::StreamingConfig(_) => {
                        panic!("duplicate ASR stream config")
                    }
                },
            )
            .copied()
            .collect::<Vec<_>>();
        assert!(
            audio.starts_with(b"RIFF") && audio.get(8..12) == Some(b"WAVE"),
            "inline audio fixture was not a WAV file"
        );
    }

    let report: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
    assert_eq!(report["run"]["transport"], "grpc");
    assert_eq!(
        report["run"]["endpoint_profiles"],
        json!([{"profile_id": "default", "endpoint_id": "riva_asr"}])
    );
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        1.0
    );

    let _ = shutdown.send(());
    server.await.unwrap();
}

async fn start_server() -> (
    String,
    Arc<Mutex<Vec<Vec<StreamingRecognizeRequest>>>>,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let service = RivaAsrService {
        requests: captured.clone(),
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
    (format!("grpc://{address}"), captured, shutdown_tx, server)
}
