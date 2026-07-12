// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation-aware evaluator inference materialization and scheduled execution.
//!
//! Provider payloads are semantic, typed JSON values. This module rejects
//! transport authority, resolves only a logical service, lowers through a
//! worker-local prepared endpoint binding, and issues the resulting turn through
//! [`ScheduledRuntime`]. It therefore shares endpoint formatting, transport,
//! cancellation, metrics, and usage reconciliation with ordinary workloads.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use aiperf_dataset::{
    AccuracyAssociation, ContentGroup, Conversation, ConversationContextMode, CorrelationId,
    Dataset, MediaKind, ModelId, Role, SegmentPool, TextTokenizer, Turn,
};
use aiperf_endpoints::PreparedEndpointTable;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use serde_json::{Map, Value, json};
use tokio::sync::oneshot;

use super::host::{
    EvaluationRoute, HostExecutionEventSink, HostExecutionTerminal, HostOperationEnvelope,
    HostOperationExecutor, RegisteredOperationId,
};
use super::ledger::HostTerminalClass;
use super::retry::OperationCancellation;
use crate::multiturn::{
    ConversationSource, NativeDatasetConversationSource, PreparedEndpointReference, TurnToSend,
};
use crate::scheduled::{ScheduledRuntime, TurnDispatchOutcome};

/// One logical route paired with its worker-local prepared endpoint binding.
#[derive(Clone)]
pub struct PreparedEvaluationRoute {
    /// Secret-free route identity.
    pub route: EvaluationRoute,
    /// Worker-local dense prepared endpoint table.
    pub endpoint_table: Rc<PreparedEndpointTable>,
    /// Dense key plus canonical endpoint ID.
    pub endpoint: PreparedEndpointReference,
}

impl std::fmt::Debug for PreparedEvaluationRoute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedEvaluationRoute")
            .field("route", &self.route)
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

impl PreparedEvaluationRoute {
    /// Validate table key/ID agreement before evaluator work begins.
    pub fn validate(&self) -> Result<()> {
        self.route.validate()?;
        let endpoint = self.endpoint_table.get(self.endpoint.key)?;
        ensure!(
            endpoint.descriptor().id == self.endpoint.endpoint_id.as_str(),
            "evaluation route {:?} endpoint key {} resolved to {:?}, expected {:?}",
            self.route.service_id,
            self.endpoint.key.index(),
            endpoint.descriptor().id,
            self.endpoint.endpoint_id.as_str()
        );
        Ok(())
    }
}

/// Replaceable semantic-operation to normal-turn materializer.
pub trait EvaluationInferenceMaterializer {
    /// Build one transport-ready ordinary scheduled turn without sending it.
    fn materialize(&self, operation: &HostOperationEnvelope) -> Result<TurnToSend>;
}

/// Unified-dataset materializer for built-in evaluator inference operations.
pub struct DatasetEvaluationInferenceMaterializer {
    tokenizer: Arc<dyn TextTokenizer>,
    routes: BTreeMap<String, PreparedEvaluationRoute>,
}

impl DatasetEvaluationInferenceMaterializer {
    /// Freeze route bindings in deterministic logical-service order.
    pub fn new(
        tokenizer: Arc<dyn TextTokenizer>,
        routes: impl IntoIterator<Item = PreparedEvaluationRoute>,
    ) -> Result<Self> {
        let mut by_service = BTreeMap::new();
        for route in routes {
            route.validate()?;
            let service_id = route.route.service_id.clone();
            ensure!(
                by_service.insert(service_id.clone(), route).is_none(),
                "duplicate prepared evaluation route {service_id:?}"
            );
        }
        ensure!(
            !by_service.is_empty(),
            "evaluation inference materializer requires at least one route"
        );
        Ok(Self {
            tokenizer,
            routes: by_service,
        })
    }

    fn route(&self, service_id: &str) -> Result<&PreparedEvaluationRoute> {
        self.routes
            .get(service_id)
            .ok_or_else(|| anyhow!("unknown prepared evaluation route {service_id:?}"))
    }
}

impl EvaluationInferenceMaterializer for DatasetEvaluationInferenceMaterializer {
    fn materialize(&self, operation: &HostOperationEnvelope) -> Result<TurnToSend> {
        let route = self.route(&operation.service_id)?;
        let parsed = ParsedInferenceOperation::parse(operation)?;
        let mut segments = SegmentPool::new();
        let mut parent = None;
        let mut text_handles = Vec::with_capacity(parsed.texts.len());
        let mut input_tokens = 0u64;
        for text in &parsed.texts {
            let tokens = self.tokenizer.encode(text).with_context(|| {
                format!(
                    "tokenizing evaluator operation {} input",
                    operation.operation_id
                )
            })?;
            input_tokens = input_tokens
                .checked_add(u64::try_from(tokens.len()).unwrap_or(u64::MAX))
                .ok_or_else(|| anyhow!("evaluator input token count overflow"))?;
            let handle = segments.intern_text(
                parent,
                Role::from("user"),
                text.as_bytes().to_vec(),
                tokens.into_boxed_slice(),
            )?;
            parent = Some(handle);
            text_handles.push(handle);
        }
        let raw_messages = parsed
            .messages
            .as_ref()
            .map(|messages| segments.intern_raw(parent, serde_json::to_vec(messages)?))
            .transpose()?;
        if raw_messages.is_some() {
            parent = raw_messages;
        }
        let tools = parsed
            .tools
            .as_ref()
            .map(|tools| segments.intern_raw(parent, serde_json::to_vec(tools)?))
            .transpose()?;
        if tools.is_some() {
            parent = tools;
        }
        let extra_body = (!parsed.extra_body.is_empty())
            .then(|| segments.intern_raw(parent, serde_json::to_vec(&parsed.extra_body)?))
            .transpose()?;

        let mut content = ContentGroup {
            kind: MediaKind::Text,
            name: String::new(),
            handles: Default::default(),
        };
        content.handles.extend(text_handles);
        let mut turn = Turn {
            role: Some(Role::from("user")),
            model: Some(ModelId::from(route.route.model.clone())),
            max_tokens: Some(parsed.max_tokens),
            streaming: Some(parsed.streaming),
            input_tokens: input_tokens.max(1),
            raw_messages,
            tools,
            extra_body,
            ..Turn::default()
        };
        turn.content.push(content);
        let mut conversation = Conversation::new(operation.logical_call_id.clone());
        conversation.turns.push(turn);
        conversation.accuracy = Some(AccuracyAssociation {
            correlation_id: CorrelationId::from(operation.operation_id.clone()),
            task: operation.semantic_operation_id.as_str().to_string(),
        });
        let dataset = Dataset::new(
            vec![conversation],
            Arc::new(segments.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )?;
        let source = NativeDatasetConversationSource::sequential_with_prepared_endpoint(
            dataset,
            route.route.model.clone(),
            usize::try_from(parsed.max_tokens)
                .unwrap_or(usize::MAX)
                .max(1),
            route.endpoint_table.clone(),
            route.endpoint.clone(),
        )?;
        let session = source.session_for(&operation.logical_call_id, operation.unit_id.clone())?;
        let turn = session.build_first_turn(Some(1))?;
        ensure!(
            turn.request_correlation_id == operation.operation_id,
            "evaluation operation lost request correlation"
        );
        ensure!(
            turn.x_correlation_id == operation.unit_id,
            "evaluation operation lost unit correlation"
        );
        Ok(turn)
    }
}

struct ParsedInferenceOperation {
    texts: Vec<String>,
    messages: Option<Vec<Value>>,
    tools: Option<Vec<Value>>,
    extra_body: Map<String, Value>,
    max_tokens: u32,
    streaming: bool,
}

impl ParsedInferenceOperation {
    fn parse(operation: &HostOperationEnvelope) -> Result<Self> {
        let object = operation.payload.as_object().ok_or_else(|| {
            anyhow!(
                "evaluator operation {} payload must be an object",
                operation.semantic_operation_id
            )
        })?;
        reject_transport_authority(object)?;
        let mut extra_body = object
            .get("parameters")
            .map(|value| {
                value
                    .as_object()
                    .cloned()
                    .ok_or_else(|| anyhow!("inference parameters must be an object"))
            })
            .transpose()?
            .unwrap_or_default();
        let generation = object
            .get("generation")
            .map(|value| {
                value
                    .as_object()
                    .ok_or_else(|| anyhow!("generation must be an object"))
            })
            .transpose()?;
        if let Some(generation) = generation {
            for (name, value) in generation {
                if name == "parameters" {
                    let parameters = value
                        .as_object()
                        .ok_or_else(|| anyhow!("generation.parameters must be an object"))?;
                    for (parameter, value) in parameters {
                        ensure!(
                            extra_body
                                .insert(parameter.clone(), value.clone())
                                .is_none(),
                            "generation parameter {parameter:?} conflicts with parameters"
                        );
                    }
                } else if name != "max_tokens" {
                    ensure!(
                        extra_body.insert(name.clone(), value.clone()).is_none(),
                        "generation field {name:?} conflicts with parameters"
                    );
                }
            }
        }
        let max_tokens = generation
            .and_then(|value| value.get("max_tokens"))
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| anyhow!("generation.max_tokens must be a positive integer"))
                    .and_then(|value| {
                        u32::try_from(value)
                            .map_err(|_| anyhow!("generation.max_tokens exceeds u32"))
                    })
            })
            .transpose()?;
        let operation_id = operation.semantic_operation_id.as_str();
        let (texts, messages, tools, default_max_tokens, may_stream) = match operation_id {
            "model.generate" => {
                let messages = required_array(object, "messages")?.clone();
                ensure!(
                    !messages.is_empty(),
                    "model.generate messages must not be empty"
                );
                let texts = message_texts(&messages);
                let tools = optional_array(object, "tools")?;
                copy_optional(&mut extra_body, object, "tool_choice")?;
                copy_optional(&mut extra_body, object, "response_format")?;
                copy_optional(&mut extra_body, object, "response_schema")?;
                copy_optional(&mut extra_body, object, "structured_output")?;
                (texts, Some(messages), tools, None, true)
            }
            "model.complete" => {
                let texts = string_or_string_array(
                    object
                        .get("prompt")
                        .ok_or_else(|| anyhow!("model.complete requires prompt"))?,
                    "prompt",
                )?;
                (texts, None, None, None, true)
            }
            "model.responses" => {
                let input = required_array(object, "input")?.clone();
                ensure!(!input.is_empty(), "model.responses input must not be empty");
                let texts = message_texts(&input);
                let tools = optional_array(object, "tools")?;
                copy_optional(&mut extra_body, object, "instructions")?;
                copy_optional(&mut extra_body, object, "tool_choice")?;
                copy_optional(&mut extra_body, object, "response_format")?;
                copy_optional(&mut extra_body, object, "response_schema")?;
                copy_optional(&mut extra_body, object, "structured_output")?;
                (texts, Some(input), tools, None, true)
            }
            "model.embed" => {
                let texts = string_or_string_array(
                    object
                        .get("input")
                        .ok_or_else(|| anyhow!("model.embed requires input"))?,
                    "input",
                )?;
                (texts, None, None, Some(1), false)
            }
            _ => {
                return Err(anyhow!(
                    "no dataset inference materializer for semantic operation {operation_id:?}"
                ));
            }
        };
        ensure!(
            !texts.is_empty(),
            "inference operation selected zero input values"
        );
        let max_tokens = max_tokens
            .or(default_max_tokens)
            .ok_or_else(|| anyhow!("{operation_id} requires generation.max_tokens"))?;
        ensure!(max_tokens > 0, "generation.max_tokens must be positive");
        ensure!(
            !operation.stream || may_stream,
            "{operation_id} does not support streaming"
        );
        Ok(Self {
            texts,
            messages,
            tools,
            extra_body,
            max_tokens,
            streaming: operation.stream,
        })
    }
}

fn reject_transport_authority(object: &Map<String, Value>) -> Result<()> {
    for field in [
        "model", "base_url", "url", "endpoint", "api_key", "token", "headers", "retry", "retries",
        "timeout", "cache",
    ] {
        ensure!(
            !object.contains_key(field),
            "evaluator inference payload must not contain transport authority field {field:?}"
        );
    }
    Ok(())
}

fn required_array<'a>(object: &'a Map<String, Value>, field: &str) -> Result<&'a Vec<Value>> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{field} must be an array"))
}

fn optional_array(object: &Map<String, Value>, field: &str) -> Result<Option<Vec<Value>>> {
    object
        .get(field)
        .map(|value| {
            value
                .as_array()
                .cloned()
                .ok_or_else(|| anyhow!("{field} must be an array"))
        })
        .transpose()
}

fn copy_optional(
    target: &mut Map<String, Value>,
    source: &Map<String, Value>,
    field: &str,
) -> Result<()> {
    if let Some(value) = source.get(field) {
        ensure!(
            target.insert(field.to_string(), value.clone()).is_none(),
            "field {field:?} conflicts with parameters"
        );
    }
    Ok(())
}

fn string_or_string_array(value: &Value, field: &str) -> Result<Vec<String>> {
    match value {
        Value::String(value) if !value.is_empty() => Ok(vec![value.clone()]),
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("{field} array entries must be non-empty strings"))
            })
            .collect(),
        _ => Err(anyhow!(
            "{field} must be a non-empty string or string array"
        )),
    }
}

fn message_texts(messages: &[Value]) -> Vec<String> {
    let mut texts = Vec::new();
    for message in messages {
        collect_text(message, None, &mut texts);
    }
    if texts.is_empty() {
        texts.push(serde_json::to_string(messages).unwrap_or_else(|_| "[]".into()));
    }
    texts
}

fn collect_text(value: &Value, key: Option<&str>, output: &mut Vec<String>) {
    match value {
        Value::String(text)
            if !text.is_empty()
                && key.is_some_and(|key| {
                    matches!(key, "content" | "text" | "input_text" | "instructions")
                }) =>
        {
            output.push(text.clone());
        }
        Value::Array(values) => {
            for value in values {
                collect_text(value, key, output);
            }
        }
        Value::Object(object) => {
            for (key, value) in object {
                collect_text(value, Some(key), output);
            }
        }
        _ => {}
    }
}

/// Inference executor backed by one ordinary [`ScheduledRuntime`].
pub struct ScheduledInferenceHostExecutor {
    runtime: Rc<ScheduledRuntime>,
    materializer: Rc<dyn EvaluationInferenceMaterializer>,
}

impl ScheduledInferenceHostExecutor {
    /// Compose a materializer with the already-running normal scheduler.
    pub fn new(
        runtime: Rc<ScheduledRuntime>,
        materializer: Rc<dyn EvaluationInferenceMaterializer>,
    ) -> Self {
        Self {
            runtime,
            materializer,
        }
    }
}

#[async_trait(?Send)]
impl HostOperationExecutor for ScheduledInferenceHostExecutor {
    async fn execute(
        &self,
        operation: &HostOperationEnvelope,
        _events: &dyn HostExecutionEventSink,
        cancellation: OperationCancellation,
    ) -> Result<HostExecutionTerminal> {
        let turn = self.materializer.materialize(operation)?;
        let (terminal_tx, terminal_rx) = oneshot::channel();
        let issued = self.runtime.issue_turn_cancellable(
            turn,
            self.runtime.now_ns(),
            None,
            Box::new(move |_credit, outcome| {
                Box::pin(async move {
                    let _ = terminal_tx.send(outcome);
                })
            }),
            Rc::new(cancellation),
        );
        ensure!(
            issued,
            "Rust scheduling policy rejected evaluator inference"
        );
        let outcome = terminal_rx
            .await
            .context("scheduled evaluator inference lost its terminal callback")?;
        Ok(normalized_terminal(outcome))
    }
}

fn normalized_terminal(outcome: TurnDispatchOutcome) -> HostExecutionTerminal {
    let class = match outcome.terminal {
        ReplayTerminalStatus::Completed => HostTerminalClass::Completed,
        ReplayTerminalStatus::Canceled => HostTerminalClass::Cancelled,
        ReplayTerminalStatus::Rejected => HostTerminalClass::Rejected,
        ReplayTerminalStatus::Failed => HostTerminalClass::Failed,
    };
    HostExecutionTerminal {
        class,
        payload: json!({
            "status": match class {
                HostTerminalClass::Completed => "completed",
                HostTerminalClass::Failed => "failed",
                HostTerminalClass::Rejected => "rejected",
                HostTerminalClass::Cancelled => "cancelled",
            },
            "content": outcome.model_response.content.or_else(|| (!outcome.response_text.is_empty()).then_some(outcome.response_text)),
            "reasoning": outcome.model_response.reasoning,
            "assistant_message": outcome.model_response.assistant_message,
            "response_id": outcome.model_response.response_id,
            "finish_reason": outcome.model_response.finish_reason,
            "usage": {
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
                "cached_tokens": outcome.model_response.cached_prompt_tokens,
            },
            "error": outcome.model_response.error_kind.map(|kind| json!({
                "kind": kind,
                "message": outcome.model_response.error_message,
            })),
        }),
    }
}

/// Built-in semantic operation IDs supported by the dataset materializer.
pub fn builtin_inference_operation_ids() -> Result<BTreeSet<RegisteredOperationId>> {
    [
        "model.generate",
        "model.complete",
        "model.responses",
        "model.embed",
    ]
    .into_iter()
    .map(RegisteredOperationId::new)
    .collect()
}

#[cfg(test)]
mod tests {
    use aiperf_dataset::TiktokenTokenizer;
    use aiperf_endpoints::{EndpointId, EndpointRegistry, RawEndpointConfig};

    use super::*;

    fn materializer() -> DatasetEvaluationInferenceMaterializer {
        let registry = EndpointRegistry::builtin().unwrap();
        let mut table = PreparedEndpointTable::new();
        let mut routes = Vec::new();
        for (service, endpoint_name, model, capabilities) in [
            ("primary", "chat", "candidate", &["chat"][..]),
            ("judge", "messages", "judge-model", &["chat"][..]),
            (
                "completion",
                "completions",
                "completion-model",
                &["completion"][..],
            ),
            (
                "responses",
                "responses",
                "responses-model",
                &["responses"][..],
            ),
            (
                "embeddings",
                "embeddings",
                "embed-model",
                &["embedding"][..],
            ),
        ] {
            let endpoint_id = EndpointId::new(endpoint_name).unwrap();
            let endpoint = registry
                .prepare(
                    &endpoint_id,
                    RawEndpointConfig {
                        streaming: endpoint_name != "embeddings",
                        use_server_token_count: true,
                        ..RawEndpointConfig::default()
                    },
                )
                .unwrap();
            let key = table.push(endpoint).unwrap();
            routes.push((
                EvaluationRoute {
                    service_id: service.into(),
                    purpose: service.into(),
                    model: model.into(),
                    endpoint_profile: format!("{service}_profile"),
                    prepared_identity_sha256: "a".repeat(64),
                    endpoint_capabilities: capabilities
                        .iter()
                        .map(|value| (*value).into())
                        .collect(),
                },
                PreparedEndpointReference { key, endpoint_id },
            ));
        }
        let table = Rc::new(table);
        DatasetEvaluationInferenceMaterializer::new(
            Arc::new(TiktokenTokenizer::builtin()),
            routes
                .into_iter()
                .map(|(route, endpoint)| PreparedEvaluationRoute {
                    route,
                    endpoint_table: table.clone(),
                    endpoint,
                }),
        )
        .unwrap()
    }

    fn operation(service: &str, semantic: &str, payload: Value) -> HostOperationEnvelope {
        HostOperationEnvelope {
            operation_id: format!("operation-{service}"),
            unit_id: "unit".into(),
            case_id: "case".into(),
            semantic_attempt_id: "attempt".into(),
            logical_call_id: format!("call-{service}"),
            service_id: service.into(),
            semantic_operation_id: RegisteredOperationId::new(semantic).unwrap(),
            purpose: service.into(),
            payload,
            restricted: service == "judge",
            stream: semantic != "model.embed",
        }
    }

    #[test]
    fn chat_and_judge_routes_preserve_messages_tools_and_route_models() {
        let materializer = materializer();
        let payload = json!({
            "messages": [
                {"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]},
                {"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"call-1","content":"done"}
            ],
            "tools": [{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
            "tool_choice": "auto",
            "generation": {"max_tokens":32,"temperature":0.2,"top_p":0.9}
        });
        let primary = materializer
            .materialize(&operation("primary", "model.generate", payload.clone()))
            .unwrap();
        let body: Value = serde_json::from_slice(primary.request_body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "candidate");
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call-1");
        assert_eq!(body["tools"][0]["function"]["name"], "lookup");
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(primary.request_correlation_id, "operation-primary");

        let judge = materializer
            .materialize(&operation("judge", "model.generate", payload))
            .unwrap();
        let body: Value = serde_json::from_slice(judge.request_body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "judge-model");
        assert!(body.get("messages").is_some());
    }

    #[test]
    fn completions_responses_and_embedding_arrays_use_prepared_dialects() {
        let materializer = materializer();
        let completion = materializer
            .materialize(&operation(
                "completion",
                "model.complete",
                json!({"prompt":["one","two"],"generation":{"max_tokens":7}}),
            ))
            .unwrap();
        let body: Value =
            serde_json::from_slice(completion.request_body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "completion-model");
        assert_eq!(body["prompt"], json!(["one", "two"]));

        let responses = materializer
            .materialize(&operation(
                "responses",
                "model.responses",
                json!({
                    "input":[{"type":"message","role":"user","content":"hello"}],
                    "instructions":"answer briefly",
                    "generation":{"max_tokens":9}
                }),
            ))
            .unwrap();
        let body: Value = serde_json::from_slice(responses.request_body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "responses-model");
        assert_eq!(body["instructions"], "answer briefly");

        let embedding = materializer
            .materialize(&operation(
                "embeddings",
                "model.embed",
                json!({"input":["alpha","beta"]}),
            ))
            .unwrap();
        let body: Value = serde_json::from_slice(embedding.request_body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "embed-model");
        assert_eq!(body["input"], json!(["alpha", "beta"]));
        assert!(!embedding.streaming);
    }

    #[test]
    fn transport_authority_and_provider_model_override_fail_before_materialization() {
        let materializer = materializer();
        for forbidden in ["model", "base_url", "api_key", "headers", "retry", "cache"] {
            let mut payload = json!({
                "messages":[{"role":"user","content":"hello"}],
                "generation":{"max_tokens":4}
            });
            payload
                .as_object_mut()
                .unwrap()
                .insert(forbidden.into(), json!("secret-or-authority"));
            let error = materializer
                .materialize(&operation("primary", "model.generate", payload))
                .unwrap_err();
            assert!(error.to_string().contains("transport authority"));
            assert!(!error.to_string().contains("secret-or-authority"));
        }
    }

    #[test]
    fn unsupported_operation_and_streaming_embedding_fail_closed() {
        let materializer = materializer();
        assert!(
            materializer
                .materialize(&operation("primary", "model.unknown", json!({})))
                .is_err()
        );
        let mut embedding = operation("embeddings", "model.embed", json!({"input":"hello"}));
        embedding.stream = true;
        assert!(materializer.materialize(&embedding).is_err());
    }
}
