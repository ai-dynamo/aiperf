// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-v2 native KServe gRPC process coverage.

use std::convert::Infallible;
use std::io::Write;
use std::path::{Path, PathBuf};
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

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn aiperf_cli() -> PathBuf {
    workspace_root().join(".venv/bin/aiperf")
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
    serde_json::to_value(
        aiperf_cli::execute_mode::capabilities_catalog().expect("capabilities catalog"),
    )
    .expect("catalog to Value")
}

fn run_child(request: &Value) -> Output {
    let flag = match request["operation"].as_str() {
        Some("validate") => "--validate",
        _ => "--execute",
    };
    let mut child = Command::new(binary())
        .arg(flag)
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

fn find_files(root: &Path, name: &str, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_files(&path, name, found);
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            found.push(path);
        }
    }
}

fn request(operation: &str, artifact_dir: &std::path::Path, url: &str) -> Value {
    json!({
        "protocol_version": 2,
        "operation": operation,
        "run": {
            "benchmark_id": "native-grpc-v2",
            "artifact_dir": artifact_dir,
            "random_seed": 7,
            "cfg": {
                "models": {
                    "strategy": "round_robin",
                    "items": [{"name": "fixture-model"}, {"name": "second-model"}]
                },
                "endpoint": {
                    "type": "kserve_v2_infer",
                    "urls": [url],
                    "streaming": true,
                    "use_server_token_count": false,
                    "timeout": 10.0,
                    "connection_reuse": "pooled",
                    "headers": {"authorization": "Bearer fixture"},
                    "http2": false,
                    "wait_for_model_timeout": 0.0
                },
                "datasets": [{
                    "type": "synthetic",
                    "entries": 2,
                    "sampling": "sequential",
                    "prompts": {
                        "isl": {"value": 4.0},
                        "osl": {"value": 1.0}
                    }
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
                    "requests": 2,
                    "concurrency": 2
                }],
                "transport": {"type": "grpc"},
                "runtime": {"workers": 2}
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
            assert_eq!(
                request
                    .metadata()
                    .get("authorization")
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "Bearer fixture"
            );
            let request = ModelInferRequest::decode(request.into_inner())
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            if request.model_name != "cli-model" {
                return Err(Status::failed_precondition(
                    "fixture runner requests must use ModelStreamInfer",
                ));
            }
            captured.lock().unwrap().push(request.clone());
            let response = infer_response("answer", &request);
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
            let messages = ["ans", "wer"].map(|text| {
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

/// Return one deterministic `FP32` embedding tensor.
fn embedding_infer_response(request: &ModelInferRequest, dim: usize) -> ModelInferResponse {
    let embedding: Vec<f32> = (0..dim).map(|index| index as f32 * 0.5).collect();
    ModelInferResponse {
        model_name: request.model_name.clone(),
        id: request.id.clone(),
        outputs: vec![InferOutputTensor {
            name: "text_embeddings".to_owned(),
            datatype: "FP32".to_owned(),
            shape: vec![1, dim as i64],
            contents: Some(InferTensorContents {
                fp32_contents: embedding,
                ..InferTensorContents::default()
            }),
            ..InferOutputTensor::default()
        }],
        ..ModelInferResponse::default()
    }
}

#[derive(Clone, Debug)]
struct EmbeddingService {
    requests: Arc<Mutex<Vec<ModelInferRequest>>>,
    dim: usize,
}

impl<B> Service<http::Request<B>> for EmbeddingService
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
        if request.uri().path() == "/inference.GRPCInferenceService/ModelInfer" {
            let method = EmbeddingInferSvc(self.requests.clone(), self.dim);
            return Box::pin(async move {
                Ok(tonic::server::Grpc::new(RawCodec)
                    .unary(method, request)
                    .await)
            });
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

impl NamedService for EmbeddingService {
    const NAME: &'static str = "inference.GRPCInferenceService";
}

struct EmbeddingInferSvc(Arc<Mutex<Vec<ModelInferRequest>>>, usize);

impl UnaryService<Bytes> for EmbeddingInferSvc {
    type Response = Bytes;
    type Future = BoxFuture<Response<Self::Response>, Status>;

    fn call(&mut self, request: Request<Bytes>) -> Self::Future {
        let captured = self.0.clone();
        let dim = self.1;
        Box::pin(async move {
            let request = ModelInferRequest::decode(request.into_inner())
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            captured.lock().unwrap().push(request.clone());
            let response = embedding_infer_response(&request, dim);
            Ok(Response::new(Bytes::from(response.encode_to_vec())))
        })
    }
}

async fn start_embedding_server(
    dim: usize,
) -> (
    String,
    Arc<Mutex<Vec<ModelInferRequest>>>,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let service = EmbeddingService {
        requests: captured.clone(),
        dim,
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

fn embedding_request(operation: &str, artifact_dir: &std::path::Path, url: &str) -> Value {
    json!({
        "protocol_version": 2,
        "operation": operation,
        "run": {
            "benchmark_id": "native-grpc-embeddings-v2",
            "artifact_dir": artifact_dir,
            "random_seed": 7,
            "cfg": {
                "models": {"strategy": "round_robin", "items": [{"name": "clip-l14"}]},
                "endpoint": {
                    "type": "kserve_v2_embeddings",
                    "urls": [url],
                    "streaming": false,
                    "use_server_token_count": false,
                    "timeout": 10.0,
                    "connection_reuse": "pooled",
                    "http2": false,
                    "wait_for_model_timeout": 0.0,
                    "extra": {"v2_input_name": "query", "v2_output_name": "text_embeddings"}
                },
                "datasets": [{
                    "type": "synthetic",
                    "entries": 2,
                    "sampling": "sequential",
                    "prompts": {"isl": {"value": 8.0}, "osl": {"value": 1.0}}
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
                    "requests": 2,
                    "concurrency": 2
                }],
                "transport": {"type": "grpc"},
                "runtime": {"workers": 2}
            }
        }
    })
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
async fn scheduled_pair_validates_and_executes_over_native_grpc_stdio() {
    let (url, captured, shutdown, server) = start_server().await;
    assert!(
        capabilities()["transport"].get("grpc").is_some(),
        "catalog must expose the gRPC transport"
    );
    assert!(
        capabilities()["endpoint"].get("kserve_v2_infer").is_some(),
        "catalog must expose the KServe v2 endpoint"
    );

    let temporary = tempfile::tempdir().unwrap();
    let target = temporary.path().join("grpc-run");
    let validation = request("validate", &target, &url);
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
    assert!(!target.exists(), "v2 validation created artifacts");

    let execution = request("execute", &target, &url);
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
    assert_eq!(terminal["protocol_version"], 2);
    assert_eq!(terminal["success"], true);
    assert_eq!(terminal["provenance"]["transport"], "grpc");
    assert_eq!(terminal["provenance"]["workload"], "scheduled");
    assert_eq!(terminal["provenance"]["transport"], "grpc");

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    let mut model_names = requests
        .iter()
        .map(|request| request.model_name.as_str())
        .collect::<Vec<_>>();
    model_names.sort_unstable();
    assert_eq!(model_names, ["fixture-model", "second-model"]);
    assert!(requests.iter().all(|request| !request.id.is_empty()));
    assert!(
        requests
            .iter()
            .all(|request| request.inputs[0].name == "text_input")
    );
    assert!(
        requests
            .iter()
            .all(|request| request.inputs[0].datatype == "BYTES")
    );

    let report: Value =
        serde_json::from_slice(&std::fs::read(target.join("native-v2.json")).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["transport"], "grpc");
    assert_eq!(report["run"]["workload"], "scheduled");
    assert_eq!(
        report["run"]["endpoint_profiles"],
        json!([{"profile_id": "default", "endpoint_id": "kserve_v2_infer"}])
    );
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        2.0
    );

    let _ = shutdown.send(());
    server.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn user_facing_aiperf_cli_executes_config_v2_against_mock_grpc_server() {
    let (url, captured, shutdown, server) = start_server().await;
    let temporary = tempfile::tempdir().unwrap();
    let artifact_root = temporary.path().join("cli-artifacts");
    let config_path = temporary.path().join("grpc.yaml");
    std::fs::write(
        &config_path,
        format!(
            r#"schemaVersion: "2.0"
benchmark:
  models: [cli-model]
  endpoint:
    urls: ["{url}"]
    type: kserve_v2_infer
    streaming: false
    waitForModelTimeout: 0.0
    headers:
      authorization: Bearer fixture
  dataset:
    type: synthetic
    entries: 1
    prompts: {{isl: 4, osl: 1}}
  phases:
    - name: profiling
      type: concurrency
      requests: 1
      concurrency: 1
  tokenizer:
    name: cl100k_base
    trustRemoteCode: false
    applyChatTemplate: false
  gpuTelemetry: {{enabled: false}}
  serverMetrics: {{enabled: false}}
  artifacts:
    dir: "{}"
  runtime:
    ui: none
  transport:
    type: grpc
"#,
            artifact_root.display()
        ),
    )
    .unwrap();

    let root = workspace_root();
    let cli = aiperf_cli();
    assert!(
        cli.is_file(),
        "missing user-facing CLI at {}",
        cli.display()
    );
    let runner = binary().to_string();
    let python_path = root.join("src");
    let cache = temporary.path().join("cache");
    let output = tokio::task::spawn_blocking(move || {
        Command::new(cli)
            .arg("profile")
            .arg("--config")
            .arg(config_path)
            .env("AIPERF_EXEC_BIN", runner)
            .env("PYTHONPATH", python_path)
            .env("AIPERF_CACHE_DIR", cache)
            .env("NO_COLOR", "1")
            .current_dir(root)
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "aiperf stdout={}\naiperf stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    {
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model_name, "cli-model");
        assert!(!requests[0].id.is_empty());
        assert_eq!(requests[0].inputs[0].name, "text_input");
    }

    let mut reports = Vec::new();
    find_files(&artifact_root, "native-v2.json", &mut reports);
    assert_eq!(
        reports.len(),
        1,
        "native reports under {}: {reports:?}",
        artifact_root.display()
    );
    let report: Value = serde_json::from_slice(&std::fs::read(&reports[0]).unwrap()).unwrap();
    assert_eq!(report["run"]["transport"], "grpc");
    assert_eq!(report["run"]["workload"], "scheduled");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        1.0
    );

    let _ = shutdown.send(());
    server.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embeddings_pair_validates_and_executes_over_native_grpc_stdio() {
    const DIM: usize = 8;
    let (url, captured, shutdown, server) = start_embedding_server(DIM).await;
    assert!(
        capabilities()["endpoint"]
            .get("kserve_v2_embeddings")
            .is_some(),
        "catalog must expose the KServe v2 embeddings endpoint"
    );

    let temporary = tempfile::tempdir().unwrap();
    let target = temporary.path().join("grpc-embeddings-run");

    let validation = embedding_request("validate", &target, &url);
    let validation_output = tokio::task::spawn_blocking(move || run_child(&validation))
        .await
        .unwrap();
    let validation_response = one_json_line(&validation_output);
    assert!(
        validation_output.status.success(),
        "validation={validation_response} stderr={}",
        String::from_utf8_lossy(&validation_output.stderr)
    );
    assert_eq!(validation_response["success"], true);

    let execution = embedding_request("execute", &target, &url);
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
    assert_eq!(terminal["provenance"]["workload"], "scheduled");

    let requests = captured.lock().unwrap().clone();
    assert_eq!(requests.len(), 2, "one ModelInfer per synthetic entry");
    assert!(
        requests
            .iter()
            .all(|request| request.model_name == "clip-l14"),
        "model name threaded through"
    );
    assert!(
        requests
            .iter()
            .all(|request| request.inputs[0].name == "query"),
        "v2_input_name selector renamed the input tensor to 'query'"
    );
    assert!(
        requests
            .iter()
            .all(|request| request.inputs[0].datatype == "BYTES"),
        "STRING input carried as a BYTES tensor"
    );

    let _ = shutdown.send(());
    server.await.unwrap();
}
