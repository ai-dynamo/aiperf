// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KServe Open Inference Protocol (OIP) v2 gRPC target.
//!
//! Serves the KServe `GRPCInferenceService` so AIPerf's native gRPC KServe
//! client (`aiperf::transport_grpc`) has a mock inference target, mirroring
//! ai-dynamo's frontend at
//! `dynamo-aiperf-native/lib/llm/src/grpc/service/kserve.rs` (dispatching tensor
//! requests to a chat/completion flavor). The five methods AIPerf dials are
//! implemented: `ModelInfer` (unary), `ModelStreamInfer` (server-streaming),
//! `ModelReady`, plus trivial `ServerLive` / `ServerReady` health.
//!
//! The wire contract is guaranteed by construction: the request/response
//! messages are the *same* prost structs the client encodes/decodes
//! (`aiperf::transport_grpc::proto`), so there is no second schema to drift.
//! There is no build-time `protoc` / `tonic-build`; the service is a
//! hand-routed `tower` service dispatched by method path (the server mirror of
//! the client's hand-rolled `RawBytesCodec` + `PathAndQuery`), served over the
//! same hyper h2 stack as the HTTP frontend.
//!
//! Content comes from the mock's existing generation seam: a KServe
//! `ModelInferRequest` is lowered to a synthetic [`ChatCompletionRequest`] and
//! run through [`crate::handlers::RequestCtx`], so token generation, latency /
//! prefix-cache / scheduler pacing, and `/metrics` accounting are shared with
//! the HTTP handlers rather than re-implemented. `text_input` (BYTES) carries
//! the prompt and an optional `max_tokens` (INT32) tensor caps output; the reply
//! is a `text_output` (BYTES) tensor.
//!
//! Extensibility: routing is by method path, so a second gRPC dialect (e.g.
//! Riva) is an added `(path -> handler)` arm plus its prost messages, reusing
//! the same lower-to-[`ChatCompletionRequest`] → generate → tensor seam. Only
//! KServe is served today; the structure does not hardcode it as the only one.

use std::convert::Infallible;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::{Buf, Bytes};
use clap::ValueEnum;
use futures::Stream;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tonic::body::Body;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::server::Grpc;
use tonic::{Request, Response, Status};

use aiperf::transport_grpc::proto::model_infer_request::InferInputTensor;
use aiperf::transport_grpc::proto::model_infer_response::InferOutputTensor;
use aiperf::transport_grpc::proto::{
    InferTensorContents, ModelInferRequest, ModelInferResponse, ModelReadyRequest,
    ModelReadyResponse, ModelStreamInferResponse,
};

use crate::handlers::RequestCtx;
use crate::listener::build_listener;
use crate::metrics::LLMLatencyInfo;
use crate::models::{ChatCompletionRequest, Message};
use crate::state::AppState;
use crate::tokens::{GenRequest, TokenizedText};

/// KServe `GRPCInferenceService` method paths AIPerf's client dials.
const MODEL_INFER: &str = "/inference.GRPCInferenceService/ModelInfer";
const MODEL_STREAM_INFER: &str = "/inference.GRPCInferenceService/ModelStreamInfer";
const MODEL_READY: &str = "/inference.GRPCInferenceService/ModelReady";
const SERVER_LIVE: &str = "/inference.GRPCInferenceService/ServerLive";
const SERVER_READY: &str = "/inference.GRPCInferenceService/ServerReady";

/// Default KServe v2 tensor names (`V2InferBehavior` in `aiperf::endpoints`).
const DEFAULT_INPUT_NAME: &str = "text_input";
const DEFAULT_OUTPUT_NAME: &str = "text_output";
/// Model name reported when the request omits one.
const DEFAULT_MODEL: &str = "mock-kserve";

/// Default KServe v2 rankings tensor names (`V2RankingsBehavior` in
/// `aiperf::endpoints`): a `query` BYTES input, a `passages` BYTES input, and a
/// numeric `scores` output the runner reads back per-passage.
const RANKINGS_QUERY_NAME: &str = "query";
const RANKINGS_PASSAGES_NAME: &str = "passages";
const RANKINGS_OUTPUT_NAME: &str = "scores";
/// Default KServe v2 image-generation tensor names (`V2ImagesBehavior`): a
/// `prompt` BYTES input and a `generated_image` BYTES output.
const IMAGES_PROMPT_NAME: &str = "prompt";
const IMAGES_OUTPUT_NAME: &str = "generated_image";
/// Default KServe v2 VLM image input tensor name (`V2VlmBehavior`).
const VLM_IMAGE_NAME: &str = "image";

/// Which output tensor(s) the KServe inference handler emits for a request.
///
/// `Auto` (the default) keys the decision off the request's INPUT tensor names,
/// so a single mock instance serves every KServe v2 dialect the runner drives
/// (`kserve_v2_infer` / `_vlm` / `_rankings` / `_images`) without per-run
/// reconfiguration — the runner's own endpoint factories name their inputs
/// distinctly (`text_input`, `query`+`passages`, `prompt`). The explicit
/// variants force one behavior regardless of the inputs, for a single-purpose
/// target or a non-AIPerf client whose tensor names differ. See
/// [`crate::config::MockServerConfig::grpc_behavior`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "snake_case")]
pub enum GrpcBehavior {
    /// Detect the behavior from the request's input tensor names.
    #[default]
    Auto,
    /// Always generate text and emit a `text_output` BYTES tensor.
    Text,
    /// Always emit a numeric `scores` FP32 tensor (one score per passage).
    Rankings,
    /// Always emit a `generated_image` BYTES tensor (a base64 mock JPEG).
    Images,
}

impl GrpcBehavior {
    /// Resolve `Auto` against the request's input tensor names; the explicit
    /// variants pass through unchanged. Rankings wins when a `passages` tensor is
    /// present; images when a `prompt` tensor is present without a `text_input`;
    /// text otherwise (covers `kserve_v2_infer` and `kserve_v2_vlm`).
    fn resolve(self, msg: &ModelInferRequest) -> GrpcBehavior {
        if self != GrpcBehavior::Auto {
            return self;
        }
        let has = |name: &str| msg.inputs.iter().any(|tensor| tensor.name == name);
        if has(RANKINGS_PASSAGES_NAME) {
            GrpcBehavior::Rankings
        } else if has(IMAGES_PROMPT_NAME) && !has(DEFAULT_INPUT_NAME) {
            GrpcBehavior::Images
        } else {
            GrpcBehavior::Text
        }
    }
}

/// Health messages KServe defines but AIPerf's client never decodes, so they
/// live here instead of the shared `aiperf::transport_grpc::proto` (which only
/// carries the messages the client uses).
#[derive(Clone, PartialEq, ::prost::Message)]
struct ServerLiveRequest {}

#[derive(Clone, Copy, PartialEq, ::prost::Message)]
struct ServerLiveResponse {
    #[prost(bool, tag = "1")]
    live: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ServerReadyRequest {}

#[derive(Clone, Copy, PartialEq, ::prost::Message)]
struct ServerReadyResponse {
    #[prost(bool, tag = "1")]
    ready: bool,
}

/// Server-streaming response stream for `ModelStreamInfer`.
type InferStream = Pin<Box<dyn Stream<Item = Result<ModelStreamInferResponse, Status>> + Send>>;

// ===========================================================================
// Prost codec (server mirror of the client's `RawBytesCodec`; no `tonic-prost`)
// ===========================================================================

/// A tonic [`Codec`] that decodes `D` and encodes `E` via `prost`. Generic over
/// the two message types because each RPC has a distinct request/response pair.
struct ProstCodec<D, E>(PhantomData<(D, E)>);

impl<D, E> Default for ProstCodec<D, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D, E> Codec for ProstCodec<D, E>
where
    D: prost::Message + Default + Send + 'static,
    E: prost::Message + Send + 'static,
{
    type Encode = E;
    type Decode = D;
    type Encoder = ProstEncoder<E>;
    type Decoder = ProstDecoder<D>;

    fn encoder(&mut self) -> Self::Encoder {
        ProstEncoder(PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        ProstDecoder(PhantomData)
    }
}

struct ProstEncoder<E>(PhantomData<E>);

impl<E: prost::Message> Encoder for ProstEncoder<E> {
    type Item = E;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        item.encode(dst)
            .map_err(|error| Status::internal(format!("encode KServe protobuf: {error}")))
    }
}

struct ProstDecoder<D>(PhantomData<D>);

impl<D: prost::Message + Default> Decoder for ProstDecoder<D> {
    type Item = D;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Status> {
        // tonic hands us exactly one complete length-delimited frame; read it
        // whole (like the client's `RawBytesDecoder`) and prost-decode.
        let bytes: Bytes = src.copy_to_bytes(src.remaining());
        D::decode(bytes)
            .map(Some)
            .map_err(|error| Status::internal(format!("decode KServe protobuf: {error}")))
    }
}

// ===========================================================================
// Request lowering + response construction
// ===========================================================================

/// Extract the prompt (`text_input` BYTES) and optional `max_tokens` (INT32)
/// from a KServe `ModelInferRequest`, honoring typed contents first and falling
/// back to `raw_input_contents` for non-AIPerf clients (e.g. `grpcurl`).
fn decode_infer_inputs(msg: &ModelInferRequest) -> Result<(String, Option<usize>), Status> {
    let mut prompt: Option<String> = None;
    let mut fallback_text: Option<String> = None;
    let mut max_tokens: Option<usize> = None;

    for (index, tensor) in msg.inputs.iter().enumerate() {
        let raw = msg.raw_input_contents.get(index);
        if tensor.name == "max_tokens" {
            if let Some(value) = tensor_first_int(tensor, raw)
                && value > 0
            {
                max_tokens = Some(value as usize);
            }
            continue;
        }
        if tensor.name == DEFAULT_INPUT_NAME {
            prompt = tensor_first_text(tensor, raw);
            continue;
        }
        // The VLM `image` tensor (base64 image bytes) is consumed but never a
        // prompt source — skip it so it can't masquerade as the text fallback.
        if tensor.name == VLM_IMAGE_NAME {
            continue;
        }
        // Remember the first text-bearing tensor as a fallback prompt source in
        // case the input tensor was renamed via `v2_input_name`.
        if fallback_text.is_none() {
            fallback_text = tensor_first_text(tensor, raw);
        }
    }

    let prompt = prompt.or(fallback_text).ok_or_else(|| {
        Status::invalid_argument("KServe ModelInferRequest is missing a text_input BYTES tensor")
    })?;
    Ok((prompt, max_tokens))
}

/// All BYTES values of the first input tensor named `name`, decoded as UTF-8.
/// Reads typed `bytes_contents` first, falling back to the length-prefixed
/// `raw_input_contents` frame for non-AIPerf clients.
fn tensor_all_text(msg: &ModelInferRequest, name: &str) -> Vec<String> {
    for (index, tensor) in msg.inputs.iter().enumerate() {
        if tensor.name != name {
            continue;
        }
        if let Some(contents) = &tensor.contents
            && !contents.bytes_contents.is_empty()
        {
            return contents
                .bytes_contents
                .iter()
                .map(|value| String::from_utf8_lossy(value).into_owned())
                .collect();
        }
        if let Some(raw) = msg.raw_input_contents.get(index) {
            return decode_raw_bytes_tensor(raw);
        }
    }
    Vec::new()
}

/// Decode a length-prefixed KServe raw BYTES tensor frame (repeated 4-byte
/// little-endian length + payload) into UTF-8 strings.
fn decode_raw_bytes_tensor(raw: &[u8]) -> Vec<String> {
    let mut values = Vec::new();
    let mut offset = 0usize;
    while offset + 4 <= raw.len() {
        let length =
            u32::from_le_bytes(raw[offset..offset + 4].try_into().expect("checked len")) as usize;
        offset += 4;
        let end = raw.len().min(offset + length);
        values.push(String::from_utf8_lossy(&raw[offset..end]).into_owned());
        offset = end;
    }
    values
}

/// First value of a tensor as text: typed BYTES contents, else a length-prefixed
/// raw BYTES payload (4-byte little-endian length + bytes).
fn tensor_first_text(tensor: &InferInputTensor, raw: Option<&Vec<u8>>) -> Option<String> {
    if let Some(contents) = &tensor.contents
        && let Some(first) = contents.bytes_contents.first()
    {
        return Some(String::from_utf8_lossy(first).into_owned());
    }
    if let Some(raw) = raw
        && raw.len() >= 4
    {
        let length = u32::from_le_bytes(raw[0..4].try_into().expect("checked len")) as usize;
        let end = (4 + length).min(raw.len());
        return Some(String::from_utf8_lossy(&raw[4..end]).into_owned());
    }
    None
}

/// First value of a tensor as an integer: typed INT32/INT64/UINT32 contents,
/// else the leading 4 raw bytes as little-endian INT32.
fn tensor_first_int(tensor: &InferInputTensor, raw: Option<&Vec<u8>>) -> Option<i64> {
    if let Some(contents) = &tensor.contents {
        if let Some(value) = contents.int_contents.first() {
            return Some(i64::from(*value));
        }
        if let Some(value) = contents.int64_contents.first() {
            return Some(*value);
        }
        if let Some(value) = contents.uint_contents.first() {
            return Some(i64::from(*value));
        }
    }
    if let Some(raw) = raw
        && raw.len() >= 4
    {
        return Some(i64::from(i32::from_le_bytes(
            raw[0..4].try_into().expect("checked len"),
        )));
    }
    None
}

/// Requested output tensor name, defaulting to `text_output`.
fn requested_output_name(msg: &ModelInferRequest) -> String {
    msg.outputs
        .first()
        .map(|output| output.name.clone())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| DEFAULT_OUTPUT_NAME.to_string())
}

/// Resolved model name for the request.
fn model_name(msg: &ModelInferRequest) -> String {
    if msg.model_name.is_empty() {
        DEFAULT_MODEL.to_string()
    } else {
        msg.model_name.clone()
    }
}

/// Lower a KServe inference request to the shared chat generation input.
fn synth_chat(model: &str, prompt: &str, max_tokens: Option<usize>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: Value::String(prompt.to_string()),
        }],
        stream: false,
        stream_options: None,
        max_tokens,
        max_completion_tokens: None,
        ignore_eos: false,
        min_tokens: None,
        reasoning_effort: None,
        priority: None,
    }
}

/// A `text_output` BYTES tensor carrying one generated string.
fn text_output_tensor(name: &str, text: &str) -> InferOutputTensor {
    InferOutputTensor {
        name: name.to_string(),
        datatype: "BYTES".to_string(),
        shape: vec![1],
        parameters: Default::default(),
        contents: Some(InferTensorContents {
            bytes_contents: vec![text.as_bytes().to_vec()],
            ..Default::default()
        }),
    }
}

/// Build a `ModelInferResponse` with a single `text_output` tensor.
fn build_infer_response(
    id: &str,
    model: &str,
    output_name: &str,
    text: &str,
) -> ModelInferResponse {
    ModelInferResponse {
        model_name: model.to_string(),
        model_version: String::new(),
        id: id.to_string(),
        parameters: Default::default(),
        outputs: vec![text_output_tensor(output_name, text)],
        raw_output_contents: Vec::new(),
    }
}

/// An `FP32` embedding output tensor of shape `[1, dim]`, mirroring a Triton
/// `python`-backend embedder's single `text_embeddings` output. The KServe wire
/// carries `FP32` as `fp32_contents`, so the client decodes the vector directly
/// (matching `aiperf::transport_grpc::codec` typed-contents decode).
fn embedding_output_tensor(name: &str, embedding: &[f32]) -> InferOutputTensor {
    InferOutputTensor {
        name: name.to_string(),
        datatype: "FP32".to_string(),
        shape: vec![1, embedding.len() as i64],
        parameters: Default::default(),
        contents: Some(InferTensorContents {
            fp32_contents: embedding.to_vec(),
            ..Default::default()
        }),
    }
}

/// Build a `ModelInferResponse` carrying one `FP32` embedding tensor.
fn build_embedding_response(
    id: &str,
    model: &str,
    output_name: &str,
    embedding: &[f32],
) -> ModelInferResponse {
    ModelInferResponse {
        model_name: model.to_string(),
        model_version: String::new(),
        id: id.to_string(),
        parameters: Default::default(),
        outputs: vec![embedding_output_tensor(output_name, embedding)],
        raw_output_contents: Vec::new(),
    }
}

/// An `FP32` numeric output tensor of shape `[len]`, carrying one relevance
/// score per passage in the same order the passages arrived. The runner's
/// `V2RankingsBehavior::parse_response` reads `data` positionally, assigning
/// each score to its passage index, so the order — not a sort — is the contract.
fn scores_output_tensor(name: &str, scores: &[f32]) -> InferOutputTensor {
    InferOutputTensor {
        name: name.to_string(),
        datatype: "FP32".to_string(),
        shape: vec![scores.len() as i64],
        parameters: Default::default(),
        contents: Some(InferTensorContents {
            fp32_contents: scores.to_vec(),
            ..Default::default()
        }),
    }
}

/// Build a `ModelInferResponse` carrying one numeric `scores` tensor.
fn build_scores_response(
    id: &str,
    model: &str,
    output_name: &str,
    scores: &[f32],
) -> ModelInferResponse {
    ModelInferResponse {
        model_name: model.to_string(),
        model_version: String::new(),
        id: id.to_string(),
        parameters: Default::default(),
        outputs: vec![scores_output_tensor(output_name, scores)],
        raw_output_contents: Vec::new(),
    }
}

/// A `generated_image` BYTES output tensor carrying one base64-encoded mock
/// JPEG. The runner's `V2ImagesBehavior::parse_response` reads each `data`
/// element as a base64 image string (`b64_json`).
fn image_output_tensor(name: &str, b64_image: &str) -> InferOutputTensor {
    InferOutputTensor {
        name: name.to_string(),
        datatype: "BYTES".to_string(),
        shape: vec![1],
        parameters: Default::default(),
        contents: Some(InferTensorContents {
            bytes_contents: vec![b64_image.as_bytes().to_vec()],
            ..Default::default()
        }),
    }
}

/// Build a `ModelInferResponse` carrying one `generated_image` tensor.
fn build_image_response(
    id: &str,
    model: &str,
    output_name: &str,
    b64_image: &str,
) -> ModelInferResponse {
    ModelInferResponse {
        model_name: model.to_string(),
        model_version: String::new(),
        id: id.to_string(),
        parameters: Default::default(),
        outputs: vec![image_output_tensor(output_name, b64_image)],
        raw_output_contents: Vec::new(),
    }
}

/// The full generated token sequence for a KServe `text_output`: reasoning
/// tokens (if any) followed by output tokens.
///
/// KServe text has no separate reasoning channel, so a reasoning model's
/// thinking is folded into the single text output. This also keeps the
/// server-streaming response non-empty when a small `max_tokens` budget was
/// fully consumed by reasoning (leaving zero output tokens): an empty gRPC
/// server stream is rejected by strict clients — including AIPerf's own runner —
/// as a failed request, so a reasoning model over streaming gRPC must still emit
/// at least the reasoning tokens.
fn generated_tokens(tokenized: &TokenizedText) -> Vec<&str> {
    tokenized
        .reasoning_content_tokens
        .iter()
        .chain(tokenized.tokens.iter())
        .map(String::as_str)
        .collect()
}

/// Wrap one incremental chunk as a streaming envelope.
fn stream_chunk(id: &str, model: &str, output_name: &str, text: &str) -> ModelStreamInferResponse {
    ModelStreamInferResponse {
        error_message: String::new(),
        infer_response: Some(build_infer_response(id, model, output_name, text)),
    }
}

// ===========================================================================
// RPC handlers
// ===========================================================================

/// `ModelInfer` (unary): generate the full text and return it in one response.
async fn model_infer(
    state: Arc<AppState>,
    request: Request<ModelInferRequest>,
) -> Result<Response<ModelInferResponse>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let msg = request.into_inner();
    let model = model_name(&msg);

    // Non-text output-tensor variants: keyed off the request's input tensor
    // names (or a forced `--grpc-behavior`). Rankings/images carry no
    // `text_input`, so this must precede `decode_infer_inputs` (which requires
    // one). See [`GrpcBehavior`].
    match state.config.grpc_behavior.resolve(&msg) {
        GrpcBehavior::Rankings => return model_infer_rankings(state, &msg, &model).await,
        GrpcBehavior::Images => return model_infer_images(state, &msg, &model).await,
        GrpcBehavior::Text | GrpcBehavior::Auto => {}
    }

    let (prompt, max_tokens) = decode_infer_inputs(&msg)?;
    let output_name = requested_output_name(&msg);

    // Non-LLM embedding mode: consume the input text and return a single FP32
    // embedding tensor (one encoder forward pass, no token generation).
    if let Some(dim) = state.config.grpc_embedding_dim {
        return model_infer_embedding(state, &msg, &prompt, max_tokens, &output_name, &model, dim)
            .await;
    }

    let start = Instant::now();
    state.recorder.init_model_config(&model);
    let chat = synth_chat(&model, &prompt, max_tokens);
    let req_gen = GenRequest::Chat(&chat);
    let ctx = RequestCtx::build("grpcinfer", &req_gen, MODEL_INFER, start, &state);

    let tokens = generated_tokens(&ctx.tokenized);
    state.recorder.record_request_start(MODEL_INFER, &ctx.model);
    state.recorder.record_llm_inflight_start(&ctx.model);
    let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(tokens.len()).await;
    let latency = start.elapsed();
    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: latency.saturating_sub(prefill),
    };
    let text = tokens.concat();
    let response = build_infer_response(&msg.id, &ctx.model, &output_name, &text);

    state
        .recorder
        .record_request_bytes(MODEL_INFER, prompt.len() as u64, text.len() as u64);
    state.recorder.record_llm_success(
        MODEL_INFER,
        &ctx.model,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
    );
    state.recorder.record_llm_inflight_end(&ctx.model);
    state.recorder.record_request_end(MODEL_INFER);

    Ok(Response::new(response))
}

/// `ModelInfer` embedding variant: run one encoder forward pass (prefill only)
/// and return a deterministic `FP32` embedding vector. Reuses the HTTP
/// embeddings generator so the same input yields the same vector regardless of
/// transport, and charges only the prefill (TTFT) latency — an embedding has no
/// decode steps.
async fn model_infer_embedding(
    state: Arc<AppState>,
    msg: &ModelInferRequest,
    prompt: &str,
    max_tokens: Option<usize>,
    output_name: &str,
    model: &str,
    dim: usize,
) -> Result<Response<ModelInferResponse>, Status> {
    let start = Instant::now();
    state.recorder.init_model_config(model);
    let chat = synth_chat(model, prompt, max_tokens);
    let req_gen = GenRequest::Chat(&chat);
    let ctx = RequestCtx::build("grpcembed", &req_gen, MODEL_INFER, start, &state);

    state.recorder.record_request_start(MODEL_INFER, &ctx.model);
    state.recorder.record_llm_inflight_start(&ctx.model);
    let (prefill, _decode) = ctx.latency_sim.wait_for_tokens(0).await;
    let latency = start.elapsed();

    let embedding: Vec<f32> = crate::handlers::generate_embedding(prompt, dim)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    let response = build_embedding_response(&msg.id, &ctx.model, output_name, &embedding);

    let info = LLMLatencyInfo {
        e2e: latency,
        prefill,
        decode: Duration::ZERO,
    };
    state.recorder.record_request_bytes(
        MODEL_INFER,
        prompt.len() as u64,
        (embedding.len() * std::mem::size_of::<f32>()) as u64,
    );
    state.recorder.record_llm_success(
        MODEL_INFER,
        &ctx.model,
        latency.as_secs_f64(),
        &ctx.usage,
        &info,
    );
    state.recorder.record_llm_inflight_end(&ctx.model);
    state.recorder.record_request_end(MODEL_INFER);

    Ok(Response::new(response))
}

/// `ModelInfer` rankings variant: read the `query` and `passages` BYTES input
/// tensors and return a numeric `scores` FP32 tensor with one relevance score
/// per passage (positional, unsorted — the runner assigns the passage index).
/// Scores reuse the HTTP reranker's deterministic `(query, passage)` hash so the
/// same inputs yield the same scores across transports.
async fn model_infer_rankings(
    state: Arc<AppState>,
    msg: &ModelInferRequest,
    model: &str,
) -> Result<Response<ModelInferResponse>, Status> {
    let endpoint = MODEL_INFER;
    let start = Instant::now();
    let query = tensor_all_text(msg, RANKINGS_QUERY_NAME)
        .into_iter()
        .next()
        .unwrap_or_default();
    let passages = tensor_all_text(msg, RANKINGS_PASSAGES_NAME);
    let output_name = msg
        .outputs
        .first()
        .map(|output| output.name.clone())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| RANKINGS_OUTPUT_NAME.to_string());

    state.recorder.record_request_start(endpoint, model);
    let scores: Vec<f32> = passages
        .iter()
        .map(|passage| crate::handlers::compute_mock_score(&query, passage) as f32)
        .collect();
    let latency = start.elapsed();
    let response = build_scores_response(&msg.id, model, &output_name, &scores);
    state
        .recorder
        .record_basic_success(endpoint, latency.as_secs_f64());
    state.recorder.record_request_end(endpoint);
    Ok(Response::new(response))
}

/// `ModelInfer` images variant: read the `prompt` BYTES input tensor and return
/// a `generated_image` BYTES tensor carrying one base64 mock JPEG (the same
/// deterministic generator the HTTP `/v1/images/generations` route uses).
async fn model_infer_images(
    state: Arc<AppState>,
    msg: &ModelInferRequest,
    model: &str,
) -> Result<Response<ModelInferResponse>, Status> {
    let endpoint = MODEL_INFER;
    let start = Instant::now();
    let prompt = tensor_all_text(msg, IMAGES_PROMPT_NAME)
        .into_iter()
        .next()
        .or_else(|| decode_infer_inputs(msg).ok().map(|(prompt, _)| prompt))
        .unwrap_or_default();
    let output_name = msg
        .outputs
        .first()
        .map(|output| output.name.clone())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| IMAGES_OUTPUT_NAME.to_string());

    state.recorder.record_request_start(endpoint, model);
    let b64_image = crate::handlers::mock_jpeg_b64(&prompt, 0);
    let latency = start.elapsed();
    let response = build_image_response(&msg.id, model, &output_name, &b64_image);
    state
        .recorder
        .record_basic_success(endpoint, latency.as_secs_f64());
    state.recorder.record_request_end(endpoint);
    Ok(Response::new(response))
}

/// `ModelStreamInfer` (server-streaming): one envelope per generated token,
/// paced by the shared latency simulator so the client measures real TTFT/ITL.
async fn model_stream_infer(
    state: Arc<AppState>,
    request: Request<ModelInferRequest>,
) -> Result<Response<InferStream>, Status> {
    if state.inject_error() {
        return Err(Status::internal("Simulated error"));
    }
    let msg = request.into_inner();
    let (prompt, max_tokens) = decode_infer_inputs(&msg)?;
    let output_name = requested_output_name(&msg);
    let model = model_name(&msg);
    let id = msg.id.clone();

    let stream = async_stream::stream! {
        let start = Instant::now();
        state.recorder.init_model_config(&model);
        let chat = synth_chat(&model, &prompt, max_tokens);
        let req_gen = GenRequest::Chat(&chat);
        let ctx = RequestCtx::build("grpcinfer", &req_gen, MODEL_STREAM_INFER, start, &state);
        let labeled = state.recorder.labeled(MODEL_STREAM_INFER, &ctx.model);
        state.recorder.record_streaming_start(MODEL_STREAM_INFER, &ctx.model);
        state.recorder.record_request_start(MODEL_STREAM_INFER, &ctx.model);
        state.recorder.record_llm_inflight_start(&ctx.model);

        let tokens = generated_tokens(&ctx.tokenized);
        let mut first_emit: Option<Instant> = None;
        let mut last_emit: Option<Instant> = None;
        for (index, token) in tokens.iter().enumerate() {
            let emit_at = ctx.latency_sim.wait_for_index(index).await;
            if first_emit.is_none() {
                first_emit = Some(emit_at);
                state
                    .recorder
                    .record_ttft_fast(&labeled, emit_at.duration_since(start).as_secs_f64());
            } else if let Some(last) = last_emit {
                state
                    .recorder
                    .record_itl_fast(&labeled, emit_at.duration_since(last).as_secs_f64());
            }
            last_emit = Some(emit_at);
            state.recorder.record_streamed_token_fast(&labeled);
            yield Ok(stream_chunk(&id, &ctx.model, &output_name, token));
        }

        let latency = start.elapsed();
        let prefill = first_emit
            .map(|instant| instant.duration_since(start))
            .unwrap_or_default();
        let info = LLMLatencyInfo {
            e2e: latency,
            prefill,
            decode: latency.saturating_sub(prefill),
        };
        state.recorder.record_llm_success(
            MODEL_STREAM_INFER,
            &ctx.model,
            latency.as_secs_f64(),
            &ctx.usage,
            &info,
        );
        state.recorder.record_llm_inflight_end(&ctx.model);
        state.recorder.record_request_end(MODEL_STREAM_INFER);
    };

    Ok(Response::new(Box::pin(stream)))
}

/// `ModelReady`: the mock always has its model ready.
async fn model_ready(
    _state: Arc<AppState>,
    _request: Request<ModelReadyRequest>,
) -> Result<Response<ModelReadyResponse>, Status> {
    Ok(Response::new(ModelReadyResponse { ready: true }))
}

async fn server_live(
    _state: Arc<AppState>,
    _request: Request<ServerLiveRequest>,
) -> Result<Response<ServerLiveResponse>, Status> {
    Ok(Response::new(ServerLiveResponse { live: true }))
}

async fn server_ready(
    _state: Arc<AppState>,
    _request: Request<ServerReadyRequest>,
) -> Result<Response<ServerReadyResponse>, Status> {
    Ok(Response::new(ServerReadyResponse { ready: true }))
}

// ===========================================================================
// Routing + serving
// ===========================================================================

/// Route one gRPC request to its handler by method path. Unknown methods get a
/// gRPC `Unimplemented` status. Returns `Infallible` because every path — RPC or
/// not — resolves to a well-formed gRPC HTTP response.
pub async fn route(
    state: Arc<AppState>,
    req: http::Request<hyper::body::Incoming>,
) -> Result<http::Response<Body>, Infallible> {
    let response = match req.uri().path() {
        MODEL_INFER => {
            let service = tower::service_fn(move |r: Request<ModelInferRequest>| {
                let state = state.clone();
                async move { model_infer(state, r).await }
            });
            Grpc::new(ProstCodec::<ModelInferRequest, ModelInferResponse>::default())
                .unary(service, req)
                .await
        }
        MODEL_STREAM_INFER => {
            let service = tower::service_fn(move |r: Request<ModelInferRequest>| {
                let state = state.clone();
                async move { model_stream_infer(state, r).await }
            });
            Grpc::new(ProstCodec::<ModelInferRequest, ModelStreamInferResponse>::default())
                .server_streaming(service, req)
                .await
        }
        MODEL_READY => {
            let service = tower::service_fn(move |r: Request<ModelReadyRequest>| {
                let state = state.clone();
                async move { model_ready(state, r).await }
            });
            Grpc::new(ProstCodec::<ModelReadyRequest, ModelReadyResponse>::default())
                .unary(service, req)
                .await
        }
        SERVER_LIVE => {
            let service = tower::service_fn(move |r: Request<ServerLiveRequest>| {
                let state = state.clone();
                async move { server_live(state, r).await }
            });
            Grpc::new(ProstCodec::<ServerLiveRequest, ServerLiveResponse>::default())
                .unary(service, req)
                .await
        }
        SERVER_READY => {
            let service = tower::service_fn(move |r: Request<ServerReadyRequest>| {
                let state = state.clone();
                async move { server_ready(state, r).await }
            });
            Grpc::new(ProstCodec::<ServerReadyRequest, ServerReadyResponse>::default())
                .unary(service, req)
                .await
        }
        other => Status::unimplemented(format!("unknown gRPC method: {other}")).into_http(),
    };
    Ok(response)
}

/// Serve the KServe gRPC service on `addr` until the process exits. Runs its own
/// accept loop on the shared runtime with `TCP_NODELAY`, sharing `state` (and
/// thus recorder / prefix-cache / scheduler) with the HTTP frontend. gRPC is
/// h2c; hyper's auto builder serves the HTTP/2-prior-knowledge preface tonic
/// clients send.
pub async fn serve_grpc(addr: SocketAddr, state: Arc<AppState>) -> anyhow::Result<()> {
    let listener = build_listener(addr)?;
    tracing::info!(%addr, "KServe gRPC listening");
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!("grpc accept error: {error}");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        let state = state.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let service = hyper::service::service_fn(move |req| {
                let state = state.clone();
                async move { route(state, req).await }
            });
            if let Err(error) = ConnBuilder::new(TokioExecutor::new())
                .serve_connection(io, service)
                .await
            {
                tracing::debug!(%peer, "grpc connection error: {error}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiperf::transport_grpc::proto::InferTensorContents;

    fn fast_state() -> Arc<AppState> {
        let config = crate::config::MockServerConfig {
            fast: true,
            no_tokenizer: true,
            ..crate::config::MockServerConfig::default()
        }
        .apply_flags();
        AppState::build(config)
    }

    fn infer_request(prompt: &str, max_tokens: Option<i32>) -> ModelInferRequest {
        let mut inputs = vec![InferInputTensor {
            name: DEFAULT_INPUT_NAME.to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![1],
            parameters: Default::default(),
            contents: Some(InferTensorContents {
                bytes_contents: vec![prompt.as_bytes().to_vec()],
                ..Default::default()
            }),
        }];
        if let Some(max_tokens) = max_tokens {
            inputs.push(InferInputTensor {
                name: "max_tokens".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![1],
                parameters: Default::default(),
                contents: Some(InferTensorContents {
                    int_contents: vec![max_tokens],
                    ..Default::default()
                }),
            });
        }
        ModelInferRequest {
            model_name: "test-model".to_string(),
            model_version: String::new(),
            id: "req-1".to_string(),
            parameters: Default::default(),
            inputs,
            outputs: Vec::new(),
            raw_input_contents: Vec::new(),
        }
    }

    fn output_text(response: &ModelInferResponse, name: &str) -> String {
        let tensor = response
            .outputs
            .iter()
            .find(|output| output.name == name)
            .expect("output tensor present");
        let bytes = tensor
            .contents
            .as_ref()
            .expect("typed contents")
            .bytes_contents
            .first()
            .expect("one value");
        String::from_utf8_lossy(bytes).into_owned()
    }

    #[test]
    fn decode_inputs_reads_prompt_and_max_tokens() {
        let msg = infer_request("hello world this is a prompt", Some(7));
        let (prompt, max_tokens) = decode_infer_inputs(&msg).expect("decode");
        assert_eq!(prompt, "hello world this is a prompt");
        assert_eq!(max_tokens, Some(7));
    }

    #[test]
    fn decode_inputs_missing_text_input_errors() {
        let msg = ModelInferRequest {
            model_name: "m".to_string(),
            inputs: Vec::new(),
            ..Default::default()
        };
        let error = decode_infer_inputs(&msg).expect_err("must error");
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn decode_inputs_falls_back_to_raw_bytes() {
        // A renamed input tensor carrying its text in raw_input_contents
        // (4-byte LE length prefix + bytes), as a non-AIPerf client might send.
        let prompt = "raw prompt";
        let mut raw = (prompt.len() as u32).to_le_bytes().to_vec();
        raw.extend_from_slice(prompt.as_bytes());
        let msg = ModelInferRequest {
            model_name: "m".to_string(),
            inputs: vec![InferInputTensor {
                name: "prompt".to_string(),
                datatype: "BYTES".to_string(),
                shape: vec![1],
                parameters: Default::default(),
                contents: None,
            }],
            raw_input_contents: vec![raw],
            ..Default::default()
        };
        let (decoded, _) = decode_infer_inputs(&msg).expect("decode");
        assert_eq!(decoded, prompt);
    }

    #[tokio::test]
    async fn model_infer_returns_text_output() {
        let state = fast_state();
        let response = model_infer(
            state,
            Request::new(infer_request("generate some text here", None)),
        )
        .await
        .expect("infer ok")
        .into_inner();
        assert_eq!(response.id, "req-1");
        assert_eq!(response.model_name, "test-model");
        let text = output_text(&response, DEFAULT_OUTPUT_NAME);
        assert!(!text.is_empty(), "expected non-empty generated text");
    }

    #[tokio::test]
    async fn model_infer_honors_max_tokens() {
        let state = fast_state();
        // ignore_eos is off, so max_tokens is an upper bound; assert the reply
        // does not exceed it. Use a long prompt so the natural length would be
        // large without the cap.
        let long = "word ".repeat(200);
        let response = model_infer(state, Request::new(infer_request(&long, Some(5))))
            .await
            .expect("infer ok")
            .into_inner();
        let text = output_text(&response, DEFAULT_OUTPUT_NAME);
        let token_count = crate::tokens::count_tokens(&text);
        assert!(token_count <= 5, "expected <= 5 tokens, got {token_count}");
    }

    #[tokio::test]
    async fn model_stream_infer_yields_chunks() {
        use futures::StreamExt;
        let state = fast_state();
        let unary = model_infer(
            state.clone(),
            Request::new(infer_request("stream this prompt text", None)),
        )
        .await
        .expect("unary ok")
        .into_inner();
        let expected = output_text(&unary, DEFAULT_OUTPUT_NAME);

        let mut stream = model_stream_infer(
            state,
            Request::new(infer_request("stream this prompt text", None)),
        )
        .await
        .expect("stream ok")
        .into_inner();
        let mut chunks = 0usize;
        let mut assembled = String::new();
        while let Some(item) = stream.next().await {
            let envelope = item.expect("chunk ok");
            assert!(envelope.error_message.is_empty());
            let infer = envelope.infer_response.expect("infer_response present");
            assembled.push_str(&output_text(&infer, DEFAULT_OUTPUT_NAME));
            chunks += 1;
        }
        assert!(chunks > 0, "expected at least one streamed chunk");
        // Deterministic generation: the concatenated stream equals the unary text.
        assert_eq!(assembled, expected);
    }

    #[tokio::test]
    async fn model_stream_infer_reasoning_model_is_not_empty() {
        // Regression: a reasoning model with a small max_tokens budget spends it
        // all on reasoning tokens, leaving zero *output* tokens. The stream must
        // still emit the reasoning tokens — an empty gRPC server stream is a
        // failed request to strict clients (the runner). Caught only end-to-end.
        use futures::StreamExt;
        let state = fast_state();
        let mut msg = infer_request("think hard about this prompt please", Some(4));
        "openai/gpt-oss-120b".clone_into(&mut msg.model_name);
        let mut stream = model_stream_infer(state, Request::new(msg))
            .await
            .expect("stream ok")
            .into_inner();
        let mut chunks = 0usize;
        while let Some(item) = stream.next().await {
            item.expect("chunk ok");
            chunks += 1;
        }
        assert!(
            chunks > 0,
            "reasoning model must not produce an empty gRPC stream"
        );
    }

    fn bytes_tensor(name: &str, values: &[&str]) -> InferInputTensor {
        InferInputTensor {
            name: name.to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![values.len() as i64],
            parameters: Default::default(),
            contents: Some(InferTensorContents {
                bytes_contents: values.iter().map(|v| v.as_bytes().to_vec()).collect(),
                ..Default::default()
            }),
        }
    }

    fn scores_of(response: &ModelInferResponse, name: &str) -> Vec<f32> {
        response
            .outputs
            .iter()
            .find(|output| output.name == name)
            .expect("output tensor present")
            .contents
            .as_ref()
            .expect("typed contents")
            .fp32_contents
            .clone()
    }

    #[test]
    fn auto_behavior_detects_rankings_and_images() {
        let rankings = ModelInferRequest {
            inputs: vec![
                bytes_tensor("query", &["q"]),
                bytes_tensor("passages", &["p0", "p1"]),
            ],
            ..Default::default()
        };
        assert_eq!(
            GrpcBehavior::Auto.resolve(&rankings),
            GrpcBehavior::Rankings
        );

        let images = ModelInferRequest {
            inputs: vec![bytes_tensor("prompt", &["draw a cat"])],
            ..Default::default()
        };
        assert_eq!(GrpcBehavior::Auto.resolve(&images), GrpcBehavior::Images);

        let vlm = ModelInferRequest {
            inputs: vec![
                bytes_tensor("text_input", &["describe"]),
                bytes_tensor("image", &["base64data"]),
            ],
            ..Default::default()
        };
        assert_eq!(GrpcBehavior::Auto.resolve(&vlm), GrpcBehavior::Text);

        // Explicit override wins over the input tensor names.
        assert_eq!(GrpcBehavior::Text.resolve(&rankings), GrpcBehavior::Text);
    }

    #[tokio::test]
    async fn model_infer_rankings_returns_one_score_per_passage() {
        let state = fast_state();
        let msg = ModelInferRequest {
            model_name: "reranker".to_string(),
            id: "rk-1".to_string(),
            inputs: vec![
                bytes_tensor("query", &["what is ai"]),
                bytes_tensor("passages", &["p0", "p1", "p2"]),
            ],
            ..Default::default()
        };
        let response = model_infer(state, Request::new(msg))
            .await
            .expect("rankings ok")
            .into_inner();
        let scores = scores_of(&response, "scores");
        assert_eq!(scores.len(), 3, "one score per passage");
        // Deterministic: the score matches the HTTP reranker's hash.
        for (passage, score) in ["p0", "p1", "p2"].iter().zip(&scores) {
            let expected = crate::handlers::compute_mock_score("what is ai", passage) as f32;
            assert!((score - expected).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn model_infer_images_returns_generated_image() {
        let state = fast_state();
        let msg = ModelInferRequest {
            model_name: "diffusion".to_string(),
            id: "im-1".to_string(),
            inputs: vec![bytes_tensor("prompt", &["a red bicycle"])],
            ..Default::default()
        };
        let response = model_infer(state, Request::new(msg))
            .await
            .expect("images ok")
            .into_inner();
        let image = output_text(&response, "generated_image");
        assert!(!image.is_empty(), "expected a base64 image string");
        assert_eq!(image, crate::handlers::mock_jpeg_b64("a red bicycle", 0));
    }

    #[tokio::test]
    async fn model_ready_is_true() {
        let state = fast_state();
        let response = model_ready(state, Request::new(ModelReadyRequest::default()))
            .await
            .expect("ready ok")
            .into_inner();
        assert!(response.ready);
    }
}
