// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-owned ingress for evaluator-side auxiliary model calls.
//!
//! Some canonical task packages contain an LLM-backed user simulator or judge.
//! Their sandboxes can only speak the OpenAI HTTP wire format, so Rust exposes
//! an authenticated local ingress and advertises it through the evaluator
//! protocol. The ingress never forwards to a model server. It converts each
//! request into an [`AgenticModelCall`], sends it to [`AgenticWorkload`](crate::agentic::AgenticWorkload),
//! and waits while that workload issues the call through the ordinary
//! `ScheduledRuntime` / endpoint / transport / metrics path.

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use aiperf_accuracy::{
    AgenticInferenceGatewayConfig, AgenticInferenceStatus, AgenticMessage, AgenticModelCall,
    AgenticModelResult, EpisodeId, EvaluatorGenerationConfig, ModelCallId,
};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

/// Evaluator component that requested an auxiliary inference call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgenticInferencePurpose {
    /// LLM-backed user, environment, or tool simulation.
    Environment,
    /// LLM-backed canonical verification or judging.
    Verifier,
}

impl AgenticInferencePurpose {
    fn as_str(self) -> &'static str {
        match self {
            Self::Environment => "environment",
            Self::Verifier => "verifier",
        }
    }
}

impl FromStr for AgenticInferencePurpose {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "environment" => Ok(Self::Environment),
            "verifier" => Ok(Self::Verifier),
            other => Err(format!("unknown auxiliary inference purpose {other:?}")),
        }
    }
}

/// One authenticated sandbox request waiting for ordinary Rust dispatch.
pub struct AgenticAuxiliaryInferenceRequest {
    /// Evaluator component that authored the request.
    pub purpose: AgenticInferencePurpose,
    /// Normal evaluator call lowered by the shared agentic turn builder.
    pub call: AgenticModelCall,
    /// Whether the sandbox requested an SSE response.
    pub stream: bool,
    response: oneshot::Sender<AgenticModelResult>,
}

impl AgenticAuxiliaryInferenceRequest {
    /// Build a request and the response future consumed by an ingress adapter.
    pub fn new(
        purpose: AgenticInferencePurpose,
        call: AgenticModelCall,
        stream: bool,
    ) -> (Self, oneshot::Receiver<AgenticModelResult>) {
        let (response, receiver) = oneshot::channel();
        (
            Self {
                purpose,
                call,
                stream,
                response,
            },
            receiver,
        )
    }

    /// Resume the waiting sandbox with one terminal Rust inference result.
    pub fn respond(self, result: AgenticModelResult) -> Result<()> {
        self.response
            .send(result)
            .map_err(|_| anyhow!("auxiliary inference HTTP caller disconnected"))
    }
}

/// Pluggable ingress boundary for evaluator-side model callers.
///
/// A future non-HTTP sandbox bridge can implement this trait without changing
/// task scheduling or inference dispatch.
#[async_trait(?Send)]
pub trait AgenticInferenceGateway {
    /// Callback address and bearer credential sent to the evaluator worker.
    fn evaluator_config(&self) -> &AgenticInferenceGatewayConfig;

    /// Transfer exclusive ownership of the request stream to one workload.
    fn take_requests(
        &mut self,
    ) -> Result<mpsc::UnboundedReceiver<AgenticAuxiliaryInferenceRequest>>;

    /// Stop accepting requests and join the ingress task.
    async fn shutdown(&mut self) -> Result<()>;
}

#[derive(Clone)]
struct GatewayState {
    api_key: Arc<str>,
    max_tokens: usize,
    sequence: Arc<AtomicU64>,
    requests: mpsc::UnboundedSender<AgenticAuxiliaryInferenceRequest>,
}

/// Authenticated OpenAI Chat Completions ingress implemented entirely in Rust.
pub struct HttpAgenticInferenceGateway {
    config: AgenticInferenceGatewayConfig,
    requests: Option<mpsc::UnboundedReceiver<AgenticAuxiliaryInferenceRequest>>,
    shutdown: Option<oneshot::Sender<()>>,
    server: Option<JoinHandle<std::io::Result<()>>>,
}

impl HttpAgenticInferenceGateway {
    /// Bind on all local interfaces and advertise one sandbox-reachable host.
    pub async fn bind(advertised_host: &str, max_tokens: usize) -> Result<Self> {
        ensure!(
            max_tokens > 0,
            "agentic gateway max_tokens must be positive"
        );
        let advertised_host = normalize_advertised_host(advertised_host)?;
        let listener =
            tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))
                .await
                .context("binding Rust agentic inference gateway")?;
        let port = listener
            .local_addr()
            .context("reading Rust agentic inference gateway address")?
            .port();
        let api_key = format!("aiperf-{}", Uuid::new_v4().simple());
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let state = GatewayState {
            api_key: Arc::from(api_key.as_str()),
            max_tokens,
            sequence: Arc::new(AtomicU64::new(0)),
            requests: request_tx,
        };
        let app = Router::new()
            .route(
                "/episodes/{episode_id}/{purpose}/v1/chat/completions",
                post(chat_completions),
            )
            .with_state(state);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
        });
        Ok(Self {
            config: AgenticInferenceGatewayConfig {
                base_url: format!("http://{advertised_host}:{port}"),
                api_key,
            },
            requests: Some(request_rx),
            shutdown: Some(shutdown_tx),
            server: Some(server),
        })
    }
}

/// Resolve the host address advertised to evaluator sandboxes.
///
/// An explicit hostname or IP is validated verbatim. Otherwise a UDP route
/// probe asks the kernel which non-loopback interface would reach an external
/// address; `connect` on an unbound UDP socket performs no network request.
pub fn resolve_advertised_host(explicit: Option<&str>) -> Result<String> {
    if let Some(explicit) = explicit {
        return normalize_advertised_host(explicit);
    }
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
        .context("binding agentic gateway address-discovery socket")?;
    socket
        .connect((Ipv4Addr::new(192, 0, 2, 1), 9))
        .context("discovering a sandbox-reachable agentic gateway address")?;
    let address = socket
        .local_addr()
        .context("reading discovered agentic gateway address")?
        .ip();
    ensure!(
        !address.is_loopback() && !address.is_unspecified(),
        "could not discover a non-loopback agentic gateway address; pass --agentic-inference-gateway-host"
    );
    normalize_advertised_host(&address.to_string())
}

#[async_trait(?Send)]
impl AgenticInferenceGateway for HttpAgenticInferenceGateway {
    fn evaluator_config(&self) -> &AgenticInferenceGatewayConfig {
        &self.config
    }

    fn take_requests(
        &mut self,
    ) -> Result<mpsc::UnboundedReceiver<AgenticAuxiliaryInferenceRequest>> {
        self.requests
            .take()
            .ok_or_else(|| anyhow!("agentic inference gateway request stream was already taken"))
    }

    async fn shutdown(&mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(server) = self.server.take() {
            server
                .await
                .context("joining Rust agentic inference gateway task")?
                .context("serving Rust agentic inference gateway")?;
        }
        Ok(())
    }
}

impl Drop for HttpAgenticInferenceGateway {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(server) = self.server.take() {
            server.abort();
        }
    }
}

async fn chat_completions(
    State(state): State<GatewayState>,
    Path((episode_id, purpose)): Path<(String, String)>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    match handle_chat_completions(state, episode_id, purpose, headers, body).await {
        Ok(response) => response,
        Err(error) => openai_error(error.status, error.kind, anyhow!(error.message)),
    }
}

#[derive(Debug)]
struct GatewayError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl GatewayError {
    fn invalid(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: error.to_string(),
        }
    }

    fn unavailable(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "aiperf_gateway_unavailable",
            message: error.to_string(),
        }
    }

    fn response_lost(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            kind: "aiperf_inference_error",
            message: error.to_string(),
        }
    }
}

async fn handle_chat_completions(
    state: GatewayState,
    episode_id: String,
    purpose: String,
    headers: HeaderMap,
    body: Value,
) -> std::result::Result<Response, GatewayError> {
    authenticate(&headers, &state.api_key).map_err(|error| GatewayError {
        status: StatusCode::UNAUTHORIZED,
        kind: "authentication_error",
        message: error.to_string(),
    })?;
    let episode_id = EpisodeId::new(episode_id).map_err(GatewayError::invalid)?;
    let purpose = AgenticInferencePurpose::from_str(&purpose).map_err(GatewayError::invalid)?;
    let parsed = parse_chat_request(body, state.max_tokens).map_err(GatewayError::invalid)?;
    let sequence = state.sequence.fetch_add(1, Ordering::Relaxed);
    let call_id = ModelCallId::new(format!(
        "{}:aux:{}:{sequence:016x}",
        episode_id.as_str(),
        purpose.as_str()
    ))
    .map_err(GatewayError::invalid)?;
    let (request, response_rx) = AgenticAuxiliaryInferenceRequest::new(
        purpose,
        AgenticModelCall {
            episode_id: episode_id.clone(),
            call_id,
            turn_index: usize::try_from(sequence).unwrap_or(usize::MAX),
            model: Some(parsed.model.clone()),
            prompt: String::new(),
            messages: parsed.messages,
            generation: parsed.generation,
            tools: parsed.tools,
            tool_choice: parsed.tool_choice,
            response_format: parsed.response_format,
            extra_body: parsed.extra_body,
        },
        parsed.stream,
    );
    state.requests.send(request).map_err(|_| {
        GatewayError::unavailable("agentic workload is no longer accepting auxiliary calls")
    })?;
    let result = response_rx.await.map_err(|error| {
        GatewayError::response_lost(format!(
            "agentic workload dropped an auxiliary inference response: {error}"
        ))
    })?;
    Ok(openai_result(result, parsed.model, parsed.stream))
}

fn authenticate(headers: &HeaderMap, api_key: &str) -> Result<()> {
    let expected = format!("Bearer {api_key}");
    let authored = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    ensure!(
        authored == Some(expected.as_str()),
        "missing or invalid agentic inference gateway bearer token"
    );
    Ok(())
}

struct ParsedChatRequest {
    model: String,
    messages: Vec<AgenticMessage>,
    generation: EvaluatorGenerationConfig,
    tools: Vec<Value>,
    tool_choice: Option<Value>,
    response_format: Option<Value>,
    extra_body: Map<String, Value>,
    stream: bool,
}

fn parse_chat_request(value: Value, default_max_tokens: usize) -> Result<ParsedChatRequest> {
    let mut object = value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("chat completion request must be a JSON object"))?;
    let model = take_string(&mut object, "model")?
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("chat completion request requires a non-empty model"))?;
    let messages = object
        .remove("messages")
        .and_then(|value| value.as_array().cloned())
        .ok_or_else(|| anyhow!("chat completion request requires a messages array"))?
        .into_iter()
        .map(parse_message)
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !messages.is_empty(),
        "chat completion messages must not be empty"
    );
    let max_tokens = take_usize(&mut object, "max_completion_tokens")?
        .or(take_usize(&mut object, "max_tokens")?)
        .unwrap_or(default_max_tokens);
    ensure!(
        max_tokens > 0,
        "chat completion max_tokens must be positive"
    );
    let temperature = take_f64(&mut object, "temperature")?.unwrap_or(0.0);
    let top_p = take_f64(&mut object, "top_p")?.unwrap_or(1.0);
    ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "chat completion temperature must be finite and non-negative"
    );
    ensure!(
        top_p.is_finite() && (0.0..=1.0).contains(&top_p),
        "chat completion top_p must be in [0, 1]"
    );
    let stop = parse_stop(object.remove("stop"))?;
    let tools = match object.remove("tools") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(anyhow!("chat completion tools must be an array or null")),
    };
    let tool_choice = non_null(object.remove("tool_choice"));
    let response_format = non_null(object.remove("response_format"));
    let stream = match object.remove("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => value,
        Some(_) => return Err(anyhow!("chat completion stream must be a boolean")),
    };
    if let Some(n) = take_usize(&mut object, "n")? {
        ensure!(
            n == 1,
            "agentic inference gateway supports exactly one choice"
        );
    }
    object.remove("stream_options");
    Ok(ParsedChatRequest {
        model,
        messages,
        generation: EvaluatorGenerationConfig {
            max_tokens,
            temperature,
            top_p,
            stop,
        },
        tools,
        tool_choice,
        response_format,
        extra_body: object,
        stream,
    })
}

fn parse_message(value: Value) -> Result<AgenticMessage> {
    let mut object = value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("chat completion message must be an object"))?;
    let role = take_string(&mut object, "role")?
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("chat completion message requires a non-empty role"))?;
    let content = object.remove("content").unwrap_or(Value::Null);
    Ok(AgenticMessage {
        role,
        content,
        extra: object.into_iter().collect::<BTreeMap<_, _>>(),
    })
}

fn take_string(object: &mut Map<String, Value>, name: &str) -> Result<Option<String>> {
    match object.remove(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(anyhow!("chat completion {name} must be a string or null")),
    }
}

fn take_usize(object: &mut Map<String, Value>, name: &str) -> Result<Option<usize>> {
    match object.remove(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| anyhow!("chat completion {name} must be a non-negative integer")),
        Some(_) => Err(anyhow!("chat completion {name} must be an integer or null")),
    }
}

fn take_f64(object: &mut Map<String, Value>, name: &str) -> Result<Option<f64>> {
    match object.remove(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| anyhow!("chat completion {name} must be numeric")),
        Some(_) => Err(anyhow!("chat completion {name} must be numeric or null")),
    }
}

fn parse_stop(value: Option<Value>) -> Result<Vec<String>> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(value)) => Ok(vec![value]),
        Some(Value::Array(values)) => values
            .into_iter()
            .map(|value| {
                value
                    .as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| anyhow!("chat completion stop items must be strings"))
            })
            .collect(),
        Some(_) => Err(anyhow!(
            "chat completion stop must be a string, array, or null"
        )),
    }
}

fn non_null(value: Option<Value>) -> Option<Value> {
    value.filter(|value| !value.is_null())
}

fn openai_result(result: AgenticModelResult, model: String, stream: bool) -> Response {
    if result.status != AgenticInferenceStatus::Completed {
        let message = result
            .error_message
            .unwrap_or_else(|| "Rust inference did not complete".to_string());
        return openai_error(
            StatusCode::BAD_GATEWAY,
            result.error_kind.as_deref().unwrap_or("inference_error"),
            anyhow!(message),
        );
    }
    let id = result
        .response_id
        .clone()
        .unwrap_or_else(|| result.call_id.as_str().to_string());
    let response_text = result.response.clone();
    let mut message = result
        .assistant_message
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_else(|| {
            json!({"role": "assistant", "content": response_text})
                .as_object()
                .expect("assistant message literal")
                .clone()
        });
    message
        .entry("role")
        .or_insert_with(|| Value::String("assistant".to_string()));
    if !message.contains_key("content") {
        message.insert("content".to_string(), Value::String(result.response));
    }
    if let Some(reasoning) = result.reasoning {
        message
            .entry("reasoning_content")
            .or_insert(Value::String(reasoning));
    }
    let finish_reason = result.finish_reason.unwrap_or_else(|| "stop".to_string());
    let usage = json!({
        "prompt_tokens": result.prompt_tokens.unwrap_or(0),
        "completion_tokens": result.completion_tokens.unwrap_or(0),
        "total_tokens": result.prompt_tokens.unwrap_or(0).saturating_add(result.completion_tokens.unwrap_or(0)),
        "prompt_tokens_details": {"cached_tokens": result.cached_tokens.unwrap_or(0)},
    });
    if !stream {
        return Json(json!({
            "id": id,
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }))
        .into_response();
    }
    let chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": message, "finish_reason": finish_reason}],
    });
    let usage_chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [],
        "usage": usage,
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(format!(
            "data: {chunk}\n\ndata: {usage_chunk}\n\ndata: [DONE]\n\n"
        )))
        .expect("static SSE response")
}

fn openai_error(status: StatusCode, kind: &str, error: anyhow::Error) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": error.to_string(),
                "type": kind,
                "code": status.as_u16(),
            }
        })),
    )
        .into_response()
}

fn normalize_advertised_host(value: &str) -> Result<String> {
    let value = value.trim();
    ensure!(
        !value.is_empty(),
        "agentic inference gateway host must not be empty"
    );
    ensure!(
        !value.contains('/') && !value.contains('@'),
        "agentic inference gateway host must be a hostname or IP address without a URL scheme"
    );
    if let Ok(address) = value.parse::<IpAddr>() {
        return Ok(match address {
            IpAddr::V4(address) => address.to_string(),
            IpAddr::V6(address) => format!("[{address}]"),
        });
    }
    ensure!(
        !value.contains(':'),
        "agentic inference gateway host must not include a port"
    );
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lossless_tool_history_and_generation_extras() {
        let parsed = parse_chat_request(
            json!({
                "model": "judge-model",
                "messages": [
                    {"role": "assistant", "content": null, "tool_calls": [{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "done"}
                ],
                "max_completion_tokens": 77,
                "temperature": 0.2,
                "top_p": 0.9,
                "stop": "END",
                "reasoning_effort": "low",
                "stream": true
            }),
            4096,
        )
        .unwrap();
        assert_eq!(parsed.model, "judge-model");
        assert_eq!(parsed.generation.max_tokens, 77);
        assert_eq!(parsed.generation.stop, ["END"]);
        assert_eq!(parsed.messages[0].extra["tool_calls"][0]["id"], "c1");
        assert_eq!(parsed.messages[1].extra["tool_call_id"], "c1");
        assert_eq!(parsed.extra_body["reasoning_effort"], "low");
        assert!(parsed.stream);
    }

    #[test]
    fn rejects_gateway_hosts_that_smuggle_a_url_or_port() {
        assert!(normalize_advertised_host("https://host").is_err());
        assert!(normalize_advertised_host("host:8000").is_err());
        assert_eq!(normalize_advertised_host("10.0.0.99").unwrap(), "10.0.0.99");
        assert_eq!(normalize_advertised_host("::1").unwrap(), "[::1]");
    }

    #[tokio::test]
    async fn authenticated_http_adapter_preserves_tools_and_correlates_response() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let state = GatewayState {
            api_key: Arc::from("secret"),
            max_tokens: 512,
            sequence: Arc::new(AtomicU64::new(0)),
            requests: sender,
        };
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        let body = json!({
            "model": "canonical-user-simulator",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "prior", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "prior", "content": "ready"}
            ],
            "tools": [{"type": "function", "function": {"name": "answer", "parameters": {"type": "object"}}}],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
            "max_tokens": 32,
            "stream": false
        });
        let handler = handle_chat_completions(
            state,
            "episode-1".to_string(),
            "environment".to_string(),
            headers,
            body,
        );
        let broker = async {
            let request = receiver.recv().await.unwrap();
            assert_eq!(request.purpose, AgenticInferencePurpose::Environment);
            assert_eq!(
                request.call.model.as_deref(),
                Some("canonical-user-simulator")
            );
            assert_eq!(
                request.call.messages[0].extra["tool_calls"][0]["id"],
                "prior"
            );
            assert_eq!(request.call.messages[1].extra["tool_call_id"], "prior");
            assert_eq!(request.call.tools[0]["function"]["name"], "answer");
            let result = AgenticModelResult {
                episode_id: request.call.episode_id.clone(),
                call_id: request.call.call_id.clone(),
                status: AgenticInferenceStatus::Completed,
                response: String::new(),
                reasoning: None,
                prompt_tokens: Some(9),
                completion_tokens: Some(4),
                cached_tokens: Some(2),
                response_id: Some("response-1".to_string()),
                finish_reason: Some("tool_calls".to_string()),
                assistant_message: Some(json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "next", "type": "function", "function": {"name": "answer", "arguments": "{\"value\":1}"}}]
                })),
                error_kind: None,
                error_message: None,
            };
            request.respond(result).unwrap();
        };
        let (response, ()) = tokio::join!(handler, broker);
        let response = response.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["id"], "response-1");
        assert_eq!(body["model"], "canonical-user-simulator");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["choices"][0]["message"]["tool_calls"][0]["id"], "next");
        assert_eq!(body["usage"]["prompt_tokens"], 9);
        assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
    }
}
