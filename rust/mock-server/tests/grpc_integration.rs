// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end wire test for the mock server's KServe gRPC target.
//!
//! Drives a real tonic client over h2 against a running `serve_grpc`, using
//! AIPerf's own KServe encode/decode helpers (`aiperf_runtime::transport::grpc`) — the
//! exact codec the product client uses — so this closes the client↔server loop
//! and exercises the real HTTP/2 framing, trailers, and prost round-trip that
//! the in-crate unit tests (which call handlers directly) do not.

use std::net::SocketAddr;
use std::time::Duration;

use aiperf_mock_server::config::MockServerConfig;
use aiperf_mock_server::grpc::serve_grpc;
use aiperf_runtime::transport::grpc::{
    decode_model_infer_response, decode_model_ready_response, decode_model_stream_infer_response,
    encode_model_infer_request, encode_model_ready_request,
};
use bytes::{Buf, Bytes};
use futures::StreamExt;
use http::uri::PathAndQuery;
use serde_json::{Value, json};
use tonic::client::Grpc;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::transport::Channel;
use tonic::{Request, Status};

/// Identity codec matching the product client's `RawBytesCodec`: the request is
/// pre-encoded protobuf bytes and the response is raw protobuf bytes.
#[derive(Clone, Copy, Default)]
struct RawBytesCodec;

impl Codec for RawBytesCodec {
    type Encode = Bytes;
    type Decode = Bytes;
    type Encoder = RawEnc;
    type Decoder = RawDec;
    fn encoder(&mut self) -> RawEnc {
        RawEnc
    }
    fn decoder(&mut self) -> RawDec {
        RawDec
    }
}

#[derive(Clone, Copy)]
struct RawEnc;
impl Encoder for RawEnc {
    type Item = Bytes;
    type Error = Status;
    fn encode(&mut self, item: Bytes, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        use bytes::BufMut;
        dst.put(item);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct RawDec;
impl Decoder for RawDec {
    type Item = Bytes;
    type Error = Status;
    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Bytes>, Status> {
        Ok(Some(src.copy_to_bytes(src.remaining())))
    }
}

/// Bind an ephemeral port, then start `serve_grpc` on it. Returns the address.
async fn spawn_grpc() -> SocketAddr {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let addr = probe.local_addr().expect("local addr");
    drop(probe);
    let config = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        grpc_port: Some(addr.port()),
        ..MockServerConfig::default()
    }
    .apply_flags();
    let state = aiperf_mock_server::app::build_state(config);
    tokio::spawn(async move {
        let _ = serve_grpc(addr, state).await;
    });
    // Give the accept loop a moment to bind.
    tokio::time::sleep(Duration::from_millis(100)).await;
    addr
}

/// Spawn a mock gRPC server in NON-LLM embedding mode: `ModelInfer` returns an
/// `FP32` embedding tensor of `dim` values instead of generated text.
async fn spawn_grpc_embedding(dim: usize) -> SocketAddr {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let addr = probe.local_addr().expect("local addr");
    drop(probe);
    let config = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        grpc_port: Some(addr.port()),
        grpc_embedding_dim: Some(dim),
        ..MockServerConfig::default()
    }
    .apply_flags();
    let state = aiperf_mock_server::app::build_state(config);
    tokio::spawn(async move {
        let _ = serve_grpc(addr, state).await;
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    addr
}

async fn connect(addr: SocketAddr) -> Channel {
    Channel::from_shared(format!("http://{addr}"))
        .expect("valid uri")
        .connect()
        .await
        .expect("connect to mock grpc")
}

fn infer_payload(prompt: &str) -> Value {
    json!({
        "inputs": [
            {"name": "text_input", "datatype": "BYTES", "shape": [1], "data": [prompt]}
        ]
    })
}

fn output_text(response: &Value, name: &str) -> String {
    let outputs = response
        .get("outputs")
        .and_then(Value::as_array)
        .expect("outputs array");
    let tensor = outputs
        .iter()
        .find(|o| o.get("name").and_then(Value::as_str) == Some(name))
        .expect("text_output tensor");
    tensor
        .get("data")
        .and_then(Value::as_array)
        .and_then(|d| d.first())
        .and_then(Value::as_str)
        .expect("string data")
        .to_string()
}

#[tokio::test]
async fn model_infer_round_trip() {
    let addr = spawn_grpc().await;
    let channel = connect(addr).await;
    let mut grpc = Grpc::new(channel);
    grpc.ready().await.expect("channel ready");

    let body = encode_model_infer_request(
        &infer_payload("generate a native gRPC reply here"),
        "test-model",
        "",
        "req-42",
    )
    .expect("encode request");
    let path = PathAndQuery::from_static("/inference.GRPCInferenceService/ModelInfer");
    let response = grpc
        .unary(Request::new(body), path, RawBytesCodec)
        .await
        .expect("unary ok");
    let decoded = decode_model_infer_response(&response.into_inner()).expect("decode response");

    assert_eq!(decoded.get("id").and_then(Value::as_str), Some("req-42"));
    assert_eq!(
        decoded.get("model_name").and_then(Value::as_str),
        Some("test-model")
    );
    assert!(!output_text(&decoded, "text_output").is_empty());
}

#[tokio::test]
async fn model_stream_infer_round_trip() {
    let addr = spawn_grpc().await;
    let channel = connect(addr).await;
    let mut grpc = Grpc::new(channel);
    grpc.ready().await.expect("channel ready");

    let body = encode_model_infer_request(
        &infer_payload("stream me one token at a time please"),
        "test-model",
        "",
        "req-stream",
    )
    .expect("encode request");
    let path = PathAndQuery::from_static("/inference.GRPCInferenceService/ModelStreamInfer");
    let response = grpc
        .server_streaming(Request::new(body), path, RawBytesCodec)
        .await
        .expect("streaming ok");
    let mut stream = response.into_inner();

    let mut chunks = 0usize;
    let mut assembled = String::new();
    while let Some(item) = stream.next().await {
        let bytes = item.expect("stream item ok");
        let (err, value) = decode_model_stream_infer_response(&bytes).expect("decode chunk");
        assert!(err.is_none(), "unexpected in-band error: {err:?}");
        let value = value.expect("infer_response present");
        assembled.push_str(&output_text(&value, "text_output"));
        chunks += 1;
    }
    assert!(chunks > 0, "expected at least one streamed chunk");
    assert!(!assembled.is_empty());
}

/// Read the FP32 values of a named output tensor.
fn output_fp32(response: &Value, name: &str) -> Vec<f64> {
    let outputs = response
        .get("outputs")
        .and_then(Value::as_array)
        .expect("outputs array");
    let tensor = outputs
        .iter()
        .find(|o| o.get("name").and_then(Value::as_str) == Some(name))
        .expect("named output tensor");
    assert_eq!(
        tensor.get("datatype").and_then(Value::as_str),
        Some("FP32"),
        "embedding output must be FP32"
    );
    tensor
        .get("data")
        .and_then(Value::as_array)
        .expect("data array")
        .iter()
        .map(|v| v.as_f64().expect("numeric FP32 value"))
        .collect()
}

/// A non-LLM embedding model: STRING input tensor named `query` in, FP32
/// `text_embeddings`-shaped vector out, over the real gRPC `ModelInfer` wire.
/// Mirrors a Triton `python`-backend embedder (Yingge's `clip-l14` case).
#[tokio::test]
async fn model_infer_embedding_round_trip() {
    const DIM: usize = 768;
    let addr = spawn_grpc_embedding(DIM).await;
    let channel = connect(addr).await;
    let mut grpc = Grpc::new(channel);
    grpc.ready().await.expect("channel ready");

    // Input tensor named "query" (not the default "text_input"), exactly as the
    // product `kserve_v2_embeddings` endpoint sends it with `v2_input_name=query`.
    let payload = json!({
        "inputs": [
            {"name": "query", "datatype": "BYTES", "shape": [1], "data": ["a photo of a cat"]}
        ]
    });
    let body = encode_model_infer_request(&payload, "clip-l14", "", "emb-1").expect("encode");
    let path = PathAndQuery::from_static("/inference.GRPCInferenceService/ModelInfer");
    let response = grpc
        .unary(Request::new(body.clone()), path.clone(), RawBytesCodec)
        .await
        .expect("unary ok");
    let decoded = decode_model_infer_response(&response.into_inner()).expect("decode");

    assert_eq!(decoded.get("id").and_then(Value::as_str), Some("emb-1"));
    assert_eq!(
        decoded.get("model_name").and_then(Value::as_str),
        Some("clip-l14")
    );
    let embedding = output_fp32(&decoded, "text_output");
    assert_eq!(
        embedding.len(),
        DIM,
        "embedding width matches configured dim"
    );
    assert!(
        embedding.iter().all(|v| v.is_finite()),
        "all embedding values finite"
    );

    // Determinism: the same query yields the same vector (hash-seeded generator).
    grpc.ready().await.expect("channel ready again");
    let response2 = grpc
        .unary(Request::new(body), path, RawBytesCodec)
        .await
        .expect("unary ok");
    let decoded2 = decode_model_infer_response(&response2.into_inner()).expect("decode");
    assert_eq!(
        embedding,
        output_fp32(&decoded2, "text_output"),
        "identical input produces identical embedding"
    );
}

#[tokio::test]
async fn model_ready_round_trip() {
    let addr = spawn_grpc().await;
    let channel = connect(addr).await;
    let mut grpc = Grpc::new(channel);
    grpc.ready().await.expect("channel ready");

    let body = encode_model_ready_request("test-model", "");
    let path = PathAndQuery::from_static("/inference.GRPCInferenceService/ModelReady");
    let response = grpc
        .unary(Request::new(body), path, RawBytesCodec)
        .await
        .expect("unary ok");
    let ready = decode_model_ready_response(&response.into_inner()).expect("decode ready");
    assert!(ready);
}
