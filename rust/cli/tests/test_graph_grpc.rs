// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end graph dispatch over KServe gRPC.

use std::convert::Infallible;
use std::io::Write;
use std::path::Path;
use std::pin::Pin;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use aiperf_runtime::transport::grpc::proto::model_infer_response::InferOutputTensor;
use aiperf_runtime::transport::grpc::proto::{
    InferTensorContents, ModelInferRequest, ModelInferResponse, ModelStreamInferResponse,
};
use bytes::{Buf, BufMut, Bytes};
use prost::Message;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::{Stream, wrappers::TcpListenerStream};
use tonic::body::Body;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::codegen::{Body as HttpBody, BoxFuture, Service, StdError};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Code, Request, Response, Status};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_aiperf")
}

fn capabilities() -> Value {
    serde_json::to_value(
        aiperf_cli::execute_mode::capabilities_catalog().expect("capabilities catalog"),
    )
    .expect("catalog to Value")
}

fn run_child(request: &Value) -> Output {
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
        .write_all(serde_json::to_string(&request["run"]).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
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

fn graph_request(artifact_dir: &Path, url: &str) -> Value {
    json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "graph-grpc-v2",
            "artifact_dir": artifact_dir,
            "random_seed": 7,
            "cfg": {
                "models": {"strategy": "round_robin", "items": [{"name": "graph-model"}]},
                "endpoint": {
                    "type": "kserve_v2_infer",
                    "urls": [url],
                    "streaming": true,
                    "use_server_token_count": false,
                    "timeout": 10.0,
                    "connection_reuse": "pooled",
                    "http2": false,
                    "wait_for_model_timeout": 0.0
                },
                "datasets": [{
                    "type": "file",
                    "format": "dag_jsonl",
                    "sampling": "sequential",
                    "records": [
                        {
                            "session_id": "root",
                            "turns": [{
                                "messages": [{"role": "user", "content": "root prompt"}],
                                "max_tokens": 4,
                                "forks": ["child_a", "child_b"]
                            }]
                        },
                        {
                            "session_id": "child_a",
                            "turns": [{
                                "messages": [{"role": "user", "content": "child a prompt"}],
                                "max_tokens": 4
                            }]
                        },
                        {
                            "session_id": "child_b",
                            "turns": [{
                                "messages": [{"role": "user", "content": "child b prompt"}],
                                "max_tokens": 4
                            }]
                        }
                    ]
                }],
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
                    "sessions": 1,
                    "concurrency": 1
                }],
                "transport": {"type": "grpc"},
                "runtime": {"workers": 1}
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
struct OipService {
    requests: Arc<Mutex<Vec<ModelInferRequest>>>,
}

impl<B> Service<http::Request<B>> for OipService
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
                let method = ModelInferSvc(self.requests.clone());
                return Box::pin(async move {
                    Ok(tonic::server::Grpc::new(RawCodec)
                        .unary(method, request)
                        .await)
                });
            }
            "/inference.GRPCInferenceService/ModelStreamInfer" => {
                let method = ModelStreamInferSvc(self.requests.clone());
                return Box::pin(async move {
                    Ok(tonic::server::Grpc::new(RawCodec)
                        .server_streaming(method, request)
                        .await)
                });
            }
            _ => {}
        }
        Box::pin(async move {
            let mut response = http::Response::new(Body::default());
            response
                .headers_mut()
                .insert(Status::GRPC_STATUS, (Code::Unimplemented as i32).into());
            response.headers_mut().insert(
                http::header::CONTENT_TYPE,
                tonic::metadata::GRPC_CONTENT_TYPE,
            );
            Ok(response)
        })
    }
}

impl NamedService for OipService {
    const NAME: &'static str = "inference.GRPCInferenceService";
}

struct ModelInferSvc(Arc<Mutex<Vec<ModelInferRequest>>>);

impl UnaryService<Bytes> for ModelInferSvc {
    type Response = Bytes;
    type Future = BoxFuture<Response<Self::Response>, Status>;

    fn call(&mut self, request: Request<Bytes>) -> Self::Future {
        let captured = self.0.clone();
        Box::pin(async move {
            let request = ModelInferRequest::decode(request.into_inner())
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            captured.lock().unwrap().push(request.clone());
            let response = infer_response("node", &request);
            Ok(Response::new(Bytes::from(response.encode_to_vec())))
        })
    }
}

struct ModelStreamInferSvc(Arc<Mutex<Vec<ModelInferRequest>>>);

impl ServerStreamingService<Bytes> for ModelStreamInferSvc {
    type Response = Bytes;
    type ResponseStream = Pin<Box<dyn Stream<Item = Result<Bytes, Status>> + Send + 'static>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Bytes>) -> Self::Future {
        let captured = self.0.clone();
        Box::pin(async move {
            let request = ModelInferRequest::decode(request.into_inner())
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            captured.lock().unwrap().push(request.clone());
            let messages = ["node", "done"].map(|text| {
                Ok(Bytes::from(
                    ModelStreamInferResponse {
                        error_message: String::new(),
                        infer_response: Some(infer_response(text, &request)),
                    }
                    .encode_to_vec(),
                ))
            });
            Ok(Response::new(
                Box::pin(tokio_stream::iter(messages)) as Self::ResponseStream
            ))
        })
    }
}

fn infer_response(text: &str, request: &ModelInferRequest) -> ModelInferResponse {
    ModelInferResponse {
        model_name: request.model_name.clone(),
        id: request.id.clone(),
        outputs: vec![InferOutputTensor {
            name: "text_output".to_owned(),
            datatype: "BYTES".to_owned(),
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

async fn start_server() -> (
    String,
    Arc<Mutex<Vec<ModelInferRequest>>>,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let service = OipService {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_dag_dispatches_over_native_grpc() {
    let (url, captured, shutdown, server) = start_server().await;

    let catalog = capabilities();
    assert!(
        catalog["transport"].get("grpc").is_some(),
        "catalog must expose the gRPC transport: {catalog}"
    );
    assert!(
        catalog["endpoint"].get("kserve_v2_infer").is_some(),
        "catalog must expose the KServe v2 endpoint: {catalog}"
    );
    assert!(
        catalog.get("supported_pairs").is_none(),
        "capabilities must not expose supported_pairs: {catalog}"
    );

    let temporary = tempfile::tempdir().unwrap();
    let artifact_dir = temporary.path().join("graph-grpc-run");
    let request = graph_request(&artifact_dir, &url);
    let output = tokio::task::spawn_blocking(move || run_child(&request))
        .await
        .unwrap();
    let terminal = one_json_line(&output);
    assert!(
        output.status.success(),
        "terminal={terminal} stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["protocol_version"], 2);
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["run_metadata"]["transport"], "grpc");
    assert_eq!(terminal["run_metadata"]["workload"], "graph");

    let requests = captured.lock().unwrap().clone();
    assert_eq!(
        requests.len(),
        3,
        "root + two forked children must each produce one gRPC dispatch, got {}",
        requests.len()
    );
    assert!(
        requests
            .iter()
            .all(|request| request.model_name == "graph-model"),
        "all graph nodes dispatch the configured model over gRPC"
    );
    assert!(
        requests.iter().all(|request| !request.id.is_empty()),
        "each gRPC dispatch carries a request id"
    );
    assert!(
        requests
            .iter()
            .all(|request| request.inputs[0].name == "text_input"
                && request.inputs[0].datatype == "BYTES"),
        "graph node bodies materialize into the KServe text_input tensor"
    );

    let report: Value =
        serde_json::from_slice(&std::fs::read(artifact_dir.join("native-v2.json")).unwrap())
            .unwrap();
    assert_eq!(report["run"]["transport"], "grpc");
    assert_eq!(report["run"]["workload"], "graph");
    assert_eq!(report["run"]["mode"], "graph");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"], 3.0,
        "graph-over-gRPC report must carry one record per dispatched node"
    );

    let _ = shutdown.send(());
    server.await.unwrap();
}
