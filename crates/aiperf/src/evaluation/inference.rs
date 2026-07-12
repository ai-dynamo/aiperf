// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operation-aware evaluator inference materialization and scheduled execution.
//!
//! Provider payloads are semantic, typed JSON values. This module rejects
//! transport authority, resolves only a logical service, lowers through a
//! worker-local prepared endpoint binding, and issues the resulting turn through
//! [`ScheduledRuntime`]. It therefore shares endpoint formatting, transport,
//! cancellation, metrics, and usage reconciliation with ordinary workloads.
//! The accepted request/response/event algebra is ported exactly from
//! `src/aiperf/accuracy/evaluation/operation_schemas.py:11-376`.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use aiperf_accuracy::{HostOperationUsage, STOCK_EVALUATION_OPERATION_SCHEMAS};
use aiperf_dataset::{
    AccuracyAssociation, ContentGroup, Conversation, ConversationContextMode, CorrelationId,
    Dataset, MediaKind, ModelId, Role, SegmentPool, TextTokenizer, Turn,
};
use aiperf_endpoints::PreparedEndpointTable;
use aiperf_endpoints::{ParsedResponse, ResponseData};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, oneshot};

use super::host::{
    EvaluationRoute, HostExecutionDelta, HostExecutionEventSink, HostExecutionTerminal,
    HostExecutorRegistryBuilder, HostExecutorRuntime, HostOperationDescriptor,
    HostOperationEnvelope, HostOperationExecutor, HostOperationExecutorFactory,
    HostOperationFamily, HostOperationSchemaValidator, RegisteredOperationId,
};
use super::ledger::HostTerminalClass;
use super::retry::{
    AttemptExecution, ClockedInferenceAttemptExecutor, ExponentialTransportRetryPolicy,
    InferenceAttemptExecutor, OneAttemptInference, OperationCancellation,
};
use crate::multiturn::{
    ConversationSource, NativeDatasetConversationSource, PreparedEndpointReference, TurnToSend,
};
use crate::scheduled::TurnResponseObserver;
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
    /// Validate that a route has one worker-local binding for this operation.
    fn validate_route(
        &self,
        route: &EvaluationRoute,
        operation_id: &RegisteredOperationId,
    ) -> Result<()>;

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
    fn validate_route(
        &self,
        route: &EvaluationRoute,
        operation_id: &RegisteredOperationId,
    ) -> Result<()> {
        let prepared = self.route(&route.service_id)?;
        ensure!(
            &prepared.route == route,
            "prepared evaluation route identity changed after registry freeze"
        );
        ensure!(
            route
                .endpoint_capabilities
                .contains(operation_endpoint_capability(operation_id.as_str())?),
            "route {:?} cannot execute operation {operation_id}",
            route.service_id
        );
        Ok(())
    }

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
        Value::String(value) => Ok(vec![value.clone()]),
        Value::Array(values) if !values.is_empty() => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("{field} array entries must be strings"))
            })
            .collect(),
        _ => Err(anyhow!(
            "{field} must be a string or non-empty string array"
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

fn stock_operation_schema(
    operation_id: &str,
) -> Result<&'static aiperf_accuracy::StockEvaluationOperationSchema> {
    STOCK_EVALUATION_OPERATION_SCHEMAS
        .iter()
        .find(|schema| schema.operation_id == operation_id)
        .ok_or_else(|| anyhow!("no canonical evaluator schema for operation {operation_id:?}"))
}

fn operation_endpoint_capability(operation_id: &str) -> Result<&'static str> {
    Ok(stock_operation_schema(operation_id)?.endpoint_capability)
}

struct InferenceOperationValidator {
    operation_id: RegisteredOperationId,
}

impl HostOperationSchemaValidator for InferenceOperationValidator {
    fn validate_request(&self, payload: &Value) -> Result<()> {
        validate_inference_request(self.operation_id.as_str(), payload)
    }

    fn validate_stream(&self, payload: &Value) -> Result<()> {
        validate_inference_stream(self.operation_id.as_str(), payload)
    }

    fn validate_response(&self, payload: &Value) -> Result<()> {
        validate_inference_response(self.operation_id.as_str(), payload)
    }
}

/// Factory for one canonical model operation over the normal scheduler.
pub struct ScheduledInferenceHostExecutorFactory {
    descriptor: HostOperationDescriptor,
    validator: InferenceOperationValidator,
    materializer: Rc<dyn EvaluationInferenceMaterializer>,
}

impl ScheduledInferenceHostExecutorFactory {
    /// Bind one exact stock operation schema to a prepared-route materializer.
    pub fn new(
        operation_id: RegisteredOperationId,
        materializer: Rc<dyn EvaluationInferenceMaterializer>,
    ) -> Result<Self> {
        let schema = stock_operation_schema(operation_id.as_str())?;
        let endpoint_capability = schema.endpoint_capability.to_string();
        Ok(Self {
            descriptor: HostOperationDescriptor {
                operation_id: operation_id.clone(),
                family: HostOperationFamily::new("inference")?,
                request_schema_fingerprint: schema.request_schema_sha256.to_string(),
                response_schema_fingerprint: schema.response_schema_sha256.to_string(),
                stream_schema_fingerprint: schema
                    .true_streaming
                    .then(|| schema.canonical_stream_schema_sha256.to_string()),
                true_streaming: schema.true_streaming,
                max_request_bytes: 8 * 1024 * 1024,
                max_response_bytes: 8 * 1024 * 1024,
                endpoint_capabilities: BTreeSet::from([endpoint_capability]),
            },
            validator: InferenceOperationValidator { operation_id },
            materializer,
        })
    }
}

impl HostOperationExecutorFactory for ScheduledInferenceHostExecutorFactory {
    fn descriptor(&self) -> &HostOperationDescriptor {
        &self.descriptor
    }

    fn validator(&self) -> &dyn HostOperationSchemaValidator {
        &self.validator
    }

    fn prepare(
        &self,
        runtime: &HostExecutorRuntime,
        route: &EvaluationRoute,
    ) -> Result<Rc<dyn HostOperationExecutor>> {
        self.materializer
            .validate_route(route, &self.descriptor.operation_id)?;
        let runtime = runtime.require_scheduled()?;
        Ok(Rc::new(ScheduledInferenceHostExecutor::new_for_operation(
            runtime,
            self.materializer.clone(),
            self.descriptor.operation_id.clone(),
        )))
    }
}

/// Register every linked canonical model operation against one materializer.
pub fn register_scheduled_inference_host_executors(
    builder: &mut HostExecutorRegistryBuilder,
    materializer: Rc<dyn EvaluationInferenceMaterializer>,
) -> Result<()> {
    for schema in STOCK_EVALUATION_OPERATION_SCHEMAS {
        builder.register(Rc::new(ScheduledInferenceHostExecutorFactory::new(
            RegisteredOperationId::new(schema.operation_id)?,
            materializer.clone(),
        )?))?;
    }
    Ok(())
}

fn validate_inference_request(operation_id: &str, payload: &Value) -> Result<()> {
    let object = payload
        .as_object()
        .ok_or_else(|| anyhow!("{operation_id} request must be an object"))?;
    reject_transport_authority(object)?;
    match operation_id {
        "model.generate" => {
            require_only_fields(
                object,
                &[
                    "messages",
                    "generation",
                    "tools",
                    "tool_choice",
                    "response_format",
                    "parameters",
                ],
                operation_id,
            )?;
            validate_messages(required_array(object, "messages")?)?;
            validate_generation(object.get("generation"))?;
            validate_tools(object.get("tools"))?;
            validate_tool_choice(object.get("tool_choice"))?;
            validate_optional_object(object, "response_format")?;
            validate_parameters(object.get("parameters"))?;
        }
        "model.complete" => {
            require_only_fields(
                object,
                &["prompt", "generation", "parameters"],
                operation_id,
            )?;
            string_or_string_array(
                object
                    .get("prompt")
                    .ok_or_else(|| anyhow!("model.complete requires prompt"))?,
                "prompt",
            )?;
            validate_generation(object.get("generation"))?;
            validate_parameters(object.get("parameters"))?;
        }
        "model.responses" => {
            require_only_fields(
                object,
                &["input", "instructions", "generation", "tools", "parameters"],
                operation_id,
            )?;
            validate_messages(required_array(object, "input")?)?;
            if let Some(instructions) = object.get("instructions") {
                ensure!(instructions.is_string(), "instructions must be a string");
            }
            validate_generation(object.get("generation"))?;
            validate_tools(object.get("tools"))?;
            validate_parameters(object.get("parameters"))?;
        }
        "model.embed" => {
            require_only_fields(object, &["input", "parameters"], operation_id)?;
            string_or_string_array(
                object
                    .get("input")
                    .ok_or_else(|| anyhow!("model.embed requires input"))?,
                "input",
            )?;
            if let Some(parameters) = object.get("parameters") {
                let parameters = parameters
                    .as_object()
                    .ok_or_else(|| anyhow!("parameters must be an object"))?;
                require_only_fields(
                    parameters,
                    &["dimensions", "encoding_format"],
                    "model.embed parameters",
                )?;
                if let Some(dimensions) = parameters.get("dimensions") {
                    ensure!(
                        dimensions.as_u64().is_some_and(|value| value > 0),
                        "dimensions must be positive"
                    );
                }
                if let Some(format) = parameters.get("encoding_format") {
                    ensure!(
                        format.as_str() == Some("float"),
                        "encoding_format must be float"
                    );
                }
            }
        }
        _ => return Err(anyhow!("unsupported inference operation {operation_id:?}")),
    }
    Ok(())
}

fn validate_inference_stream(operation_id: &str, payload: &Value) -> Result<()> {
    if payload.is_null() {
        return match operation_id {
            "model.generate" | "model.complete" | "model.responses" => Ok(()),
            "model.embed" => Ok(()),
            _ => Err(anyhow!("unsupported inference operation {operation_id:?}")),
        };
    }
    let object = payload
        .as_object()
        .ok_or_else(|| anyhow!("{operation_id} stream event must be an object"))?;
    match operation_id {
        "model.generate" => {
            require_only_fields(object, &["choice_index", "delta"], "model.generate stream")?;
            ensure!(
                object.get("choice_index").and_then(Value::as_u64).is_some(),
                "choice_index must be non-negative"
            );
            validate_message(
                object
                    .get("delta")
                    .ok_or_else(|| anyhow!("model.generate stream requires delta"))?,
            )
        }
        "model.complete" => {
            require_only_fields(object, &["choice_index", "text"], "model.complete stream")?;
            ensure!(
                object.get("choice_index").and_then(Value::as_u64).is_some(),
                "choice_index must be non-negative"
            );
            ensure!(
                object.get("text").is_some_and(Value::is_string),
                "stream text must be a string"
            );
            Ok(())
        }
        "model.responses" => {
            require_only_fields(object, &["event_type", "item"], "model.responses stream")?;
            ensure!(
                object
                    .get("event_type")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.is_empty()),
                "event_type must be non-empty"
            );
            ensure!(
                object.contains_key("item"),
                "responses stream requires item"
            );
            Ok(())
        }
        "model.embed" => Err(anyhow!("model.embed does not support streaming")),
        _ => Err(anyhow!("unsupported inference operation {operation_id:?}")),
    }
}

fn validate_inference_response(operation_id: &str, payload: &Value) -> Result<()> {
    let object = payload
        .as_object()
        .ok_or_else(|| anyhow!("{operation_id} response must be an object"))?;
    match operation_id {
        "model.generate" => {
            require_only_fields(object, &["choices", "usage"], operation_id)?;
            let choices = required_array(object, "choices")?;
            ensure!(
                !choices.is_empty(),
                "model.generate choices must not be empty"
            );
            for choice in choices {
                let choice = choice
                    .as_object()
                    .ok_or_else(|| anyhow!("generate choice must be an object"))?;
                require_only_fields(
                    choice,
                    &["message", "stop_reason", "finish_reason", "logprobs"],
                    "generate choice",
                )?;
                validate_message(
                    choice
                        .get("message")
                        .ok_or_else(|| anyhow!("generate choice requires message"))?,
                )?;
                ensure!(
                    matches!(
                        choice.get("stop_reason").and_then(Value::as_str),
                        Some(
                            "stop"
                                | "max_tokens"
                                | "model_length"
                                | "tool_calls"
                                | "content_filter"
                                | "unknown"
                        )
                    ),
                    "generate choice stop_reason is invalid"
                );
                validate_optional_string(choice, "finish_reason")?;
                validate_optional_object_or_null(choice, "logprobs")?;
            }
            validate_usage(object.get("usage"))
        }
        "model.complete" => {
            require_only_fields(object, &["choices", "usage"], operation_id)?;
            let choices = required_array(object, "choices")?;
            ensure!(
                !choices.is_empty(),
                "model.complete choices must not be empty"
            );
            for choice in choices {
                let choice = choice
                    .as_object()
                    .ok_or_else(|| anyhow!("completion choice must be an object"))?;
                require_only_fields(
                    choice,
                    &["text", "finish_reason", "logprobs"],
                    "completion choice",
                )?;
                ensure!(
                    choice.get("text").is_some_and(Value::is_string),
                    "completion choice requires text"
                );
                ensure!(
                    choice.get("finish_reason").is_some_and(Value::is_string),
                    "completion choice requires finish_reason"
                );
                validate_optional_object_or_null(choice, "logprobs")?;
            }
            validate_usage(object.get("usage"))
        }
        "model.responses" => {
            require_only_fields(object, &["output", "usage", "status"], operation_id)?;
            for message in required_array(object, "output")? {
                validate_message(message)?;
            }
            ensure!(
                matches!(
                    object.get("status").and_then(Value::as_str),
                    Some("completed" | "incomplete" | "failed")
                ),
                "invalid responses status"
            );
            validate_usage(object.get("usage"))
        }
        "model.embed" => {
            require_only_fields(object, &["embeddings", "usage"], operation_id)?;
            let embeddings = required_array(object, "embeddings")?;
            for embedding in embeddings {
                ensure!(
                    embedding
                        .as_array()
                        .is_some_and(|values| values.iter().all(Value::is_number)),
                    "embedding must contain numeric values"
                );
            }
            validate_usage(object.get("usage"))
        }
        _ => Err(anyhow!("unsupported inference operation {operation_id:?}")),
    }
}

fn require_only_fields(object: &Map<String, Value>, allowed: &[&str], context: &str) -> Result<()> {
    for field in object.keys() {
        ensure!(
            allowed.contains(&field.as_str()),
            "{context} contains unknown field {field:?}"
        );
    }
    Ok(())
}

fn validate_generation(value: Option<&Value>) -> Result<()> {
    let generation = value
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("generation must be an object"))?;
    require_only_fields(
        generation,
        &["max_tokens", "temperature", "top_p", "stop"],
        "generation",
    )?;
    ensure!(
        generation
            .get("max_tokens")
            .is_some_and(|value| is_integer(value) && integer_is_at_least(value, 1)),
        "generation.max_tokens must be a positive integer"
    );
    for field in ["temperature", "top_p"] {
        if let Some(value) = generation.get(field) {
            ensure!(value.is_number(), "generation.{field} must be numeric");
        }
    }
    if let Some(stop) = generation.get("stop") {
        ensure!(
            stop.is_string()
                || stop
                    .as_array()
                    .is_some_and(|values| values.iter().all(Value::is_string)),
            "generation.stop must be a string or string array"
        );
    }
    Ok(())
}

fn validate_parameters(value: Option<&Value>) -> Result<()> {
    let Some(value) = value else { return Ok(()) };
    let parameters = value
        .as_object()
        .ok_or_else(|| anyhow!("parameters must be an object"))?;
    require_only_fields(
        parameters,
        &[
            "best_of",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "seed",
            "top_k",
            "num_choices",
            "logprobs",
            "top_logprobs",
            "parallel_tool_calls",
            "internal_tools",
            "max_tool_output",
            "reasoning_effort",
            "reasoning_tokens",
            "reasoning_summary",
            "reasoning_history",
        ],
        "parameters",
    )?;
    for field in ["best_of", "top_k", "num_choices", "max_tool_output"] {
        if let Some(value) = parameters.get(field) {
            ensure!(
                is_integer(value) && integer_is_at_least(value, 1),
                "parameters.{field} must be a positive integer"
            );
        }
    }
    for field in ["top_logprobs", "reasoning_tokens"] {
        if let Some(value) = parameters.get(field) {
            ensure!(
                is_integer(value) && integer_is_at_least(value, 0),
                "parameters.{field} must be a non-negative integer"
            );
        }
    }
    if let Some(seed) = parameters.get("seed") {
        ensure!(is_integer(seed), "parameters.seed must be an integer");
    }
    for field in ["frequency_penalty", "presence_penalty"] {
        if let Some(value) = parameters.get(field) {
            ensure!(value.is_number(), "parameters.{field} must be numeric");
        }
    }
    for field in ["logprobs", "parallel_tool_calls", "internal_tools"] {
        if let Some(value) = parameters.get(field) {
            ensure!(value.is_boolean(), "parameters.{field} must be boolean");
        }
    }
    if let Some(logit_bias) = parameters.get("logit_bias") {
        let logit_bias = logit_bias
            .as_object()
            .ok_or_else(|| anyhow!("parameters.logit_bias must be an object"))?;
        ensure!(
            logit_bias.values().all(Value::is_number),
            "parameters.logit_bias values must be numeric"
        );
    }
    validate_enum(
        parameters,
        "reasoning_effort",
        &["minimal", "low", "medium", "high"],
    )?;
    validate_enum(
        parameters,
        "reasoning_summary",
        &["concise", "detailed", "auto"],
    )?;
    validate_enum(
        parameters,
        "reasoning_history",
        &["none", "all", "last", "auto"],
    )
}

fn validate_messages(messages: &[Value]) -> Result<()> {
    ensure!(!messages.is_empty(), "messages must not be empty");
    for message in messages {
        validate_message(message)?;
    }
    Ok(())
}

fn validate_message(message: &Value) -> Result<()> {
    let message = message
        .as_object()
        .ok_or_else(|| anyhow!("message must be an object"))?;
    require_only_fields(
        message,
        &["role", "content", "name", "tool_call_id", "tool_calls"],
        "message",
    )?;
    ensure!(
        matches!(
            message.get("role").and_then(Value::as_str),
            Some("system" | "developer" | "user" | "assistant" | "tool")
        ),
        "message role is invalid"
    );
    let content = message
        .get("content")
        .ok_or_else(|| anyhow!("message requires content"))?;
    match content {
        Value::String(_) => {}
        Value::Array(blocks) => {
            for block in blocks {
                validate_content_block(block)?;
            }
        }
        _ => return Err(anyhow!("message content must be text or content blocks")),
    }
    validate_optional_string(message, "name")?;
    validate_optional_string(message, "tool_call_id")?;
    if let Some(tool_calls) = message.get("tool_calls") {
        let tool_calls = tool_calls
            .as_array()
            .ok_or_else(|| anyhow!("message.tool_calls must be an array"))?;
        for tool_call in tool_calls {
            validate_tool_call(tool_call)?;
        }
    }
    Ok(())
}

fn validate_content_block(block: &Value) -> Result<()> {
    let block = block
        .as_object()
        .ok_or_else(|| anyhow!("message content block must be an object"))?;
    match block.get("type").and_then(Value::as_str) {
        Some("text") => {
            require_only_fields(block, &["type", "text"], "text content block")?;
            ensure!(
                block.get("text").is_some_and(Value::is_string),
                "text content block requires text"
            );
        }
        Some("reasoning") => {
            require_only_fields(
                block,
                &["type", "reasoning", "signature"],
                "reasoning content block",
            )?;
            ensure!(
                block.get("reasoning").is_some_and(Value::is_string),
                "reasoning content block requires reasoning"
            );
            validate_optional_string(block, "signature")?;
        }
        Some(kind @ ("image" | "audio" | "video" | "document" | "data")) => {
            require_only_fields(
                block,
                &["type", "asset_id", "media_type", "detail"],
                "asset content block",
            )?;
            ensure!(
                block
                    .get("asset_id")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.is_empty()),
                "{kind} content block requires non-empty asset_id"
            );
            validate_optional_nonempty_string(block, "media_type")?;
            validate_enum(block, "detail", &["auto", "low", "high"])?;
        }
        Some("tool_result") => {
            require_only_fields(
                block,
                &["type", "tool_call_id", "content", "is_error"],
                "tool result content block",
            )?;
            ensure!(
                block
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.is_empty()),
                "tool result requires non-empty tool_call_id"
            );
            ensure!(
                block.contains_key("content"),
                "tool result requires content"
            );
            if let Some(is_error) = block.get("is_error") {
                ensure!(
                    is_error.is_boolean(),
                    "tool result is_error must be boolean"
                );
            }
        }
        Some(kind) => return Err(anyhow!("unsupported message content block type {kind:?}")),
        None => return Err(anyhow!("message content block requires string type")),
    }
    Ok(())
}

fn validate_tool_call(value: &Value) -> Result<()> {
    let call = value
        .as_object()
        .ok_or_else(|| anyhow!("tool call must be an object"))?;
    require_only_fields(call, &["id", "type", "function"], "tool call")?;
    ensure!(
        call.get("id")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty()),
        "tool call requires non-empty id"
    );
    ensure!(
        call.get("type").and_then(Value::as_str) == Some("function"),
        "tool call type must be function"
    );
    let function = call
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("tool call requires function object"))?;
    require_only_fields(function, &["name", "arguments"], "tool call function")?;
    ensure!(
        function
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty()),
        "tool call function requires non-empty name"
    );
    ensure!(
        function.get("arguments").is_some_and(Value::is_object),
        "tool call function arguments must be an object"
    );
    Ok(())
}

fn validate_tools(value: Option<&Value>) -> Result<()> {
    let Some(value) = value else { return Ok(()) };
    let tools = value
        .as_array()
        .ok_or_else(|| anyhow!("tools must be an array"))?;
    for tool in tools {
        let tool = tool
            .as_object()
            .ok_or_else(|| anyhow!("tool must be an object"))?;
        require_only_fields(tool, &["type", "function"], "tool")?;
        ensure!(
            tool.get("type").and_then(Value::as_str) == Some("function"),
            "tool type must be function"
        );
        let function = tool
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("tool requires function object"))?;
        require_only_fields(
            function,
            &["name", "description", "parameters"],
            "tool function",
        )?;
        ensure!(
            function
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty()),
            "tool function requires non-empty name"
        );
        validate_optional_string(function, "description")?;
        ensure!(
            function.get("parameters").is_some_and(Value::is_object),
            "tool function parameters must be an object"
        );
    }
    Ok(())
}

fn validate_tool_choice(value: Option<&Value>) -> Result<()> {
    let Some(value) = value else { return Ok(()) };
    ensure!(
        value.is_object() || matches!(value.as_str(), Some("auto" | "none" | "required")),
        "tool_choice must be an object or auto/none/required"
    );
    Ok(())
}

fn validate_optional_object(object: &Map<String, Value>, field: &str) -> Result<()> {
    if let Some(value) = object.get(field) {
        ensure!(value.is_object(), "{field} must be an object");
    }
    Ok(())
}

fn validate_optional_object_or_null(object: &Map<String, Value>, field: &str) -> Result<()> {
    if let Some(value) = object.get(field) {
        ensure!(
            value.is_null() || value.is_object(),
            "{field} must be object or null"
        );
    }
    Ok(())
}

fn validate_optional_string(object: &Map<String, Value>, field: &str) -> Result<()> {
    if let Some(value) = object.get(field) {
        ensure!(value.is_string(), "{field} must be a string");
    }
    Ok(())
}

fn validate_optional_nonempty_string(object: &Map<String, Value>, field: &str) -> Result<()> {
    if let Some(value) = object.get(field) {
        ensure!(
            value.as_str().is_some_and(|value| !value.is_empty()),
            "{field} must be a non-empty string"
        );
    }
    Ok(())
}

fn validate_enum(object: &Map<String, Value>, field: &str, allowed: &[&str]) -> Result<()> {
    if let Some(value) = object.get(field) {
        ensure!(
            value.as_str().is_some_and(|value| allowed.contains(&value)),
            "{field} has an invalid value"
        );
    }
    Ok(())
}

fn is_integer(value: &Value) -> bool {
    value.as_i64().is_some() || value.as_u64().is_some()
}

fn integer_is_at_least(value: &Value, minimum: i64) -> bool {
    value.as_i64().map_or_else(
        || {
            u64::try_from(minimum)
                .ok()
                .is_some_and(|minimum| value.as_u64().is_some_and(|value| value >= minimum))
        },
        |value| value >= minimum,
    )
}

fn validate_usage(value: Option<&Value>) -> Result<()> {
    let usage = value
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("response usage must be an object"))?;
    require_only_fields(
        usage,
        &[
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_tokens",
        ],
        "usage",
    )?;
    ensure!(
        usage.values().all(|value| value.as_u64().is_some()),
        "usage values must be non-negative integers"
    );
    Ok(())
}

/// Inference executor backed by one ordinary [`ScheduledRuntime`].
pub struct ScheduledInferenceHostExecutor {
    runtime: Rc<ScheduledRuntime>,
    materializer: Rc<dyn EvaluationInferenceMaterializer>,
    operation_id: RegisteredOperationId,
    attempt_executor: Rc<dyn InferenceAttemptExecutor>,
}

impl ScheduledInferenceHostExecutor {
    /// Compose a materializer with the already-running normal scheduler.
    pub fn new(
        runtime: Rc<ScheduledRuntime>,
        materializer: Rc<dyn EvaluationInferenceMaterializer>,
    ) -> Self {
        Self::new_for_operation(
            runtime,
            materializer,
            RegisteredOperationId::new("model.generate").expect("built-in operation ID is valid"),
        )
    }

    /// Compose a scheduler/materializer for one exact semantic operation.
    pub fn new_for_operation(
        runtime: Rc<ScheduledRuntime>,
        materializer: Rc<dyn EvaluationInferenceMaterializer>,
        operation_id: RegisteredOperationId,
    ) -> Self {
        let retry_policy = Rc::new(
            ExponentialTransportRetryPolicy::new(
                3,
                100_000_000,
                1_000_000_000,
                [HostTerminalClass::Failed],
            )
            .expect("built-in inference retry policy is valid"),
        );
        let attempt_executor = Rc::new(ClockedInferenceAttemptExecutor::new(
            runtime.clock(),
            retry_policy,
        ));
        Self::new_for_operation_with_attempt_executor(
            runtime,
            materializer,
            operation_id,
            attempt_executor,
        )
    }

    /// Compose one exact operation with an injected logical-attempt executor.
    pub fn new_for_operation_with_attempt_executor(
        runtime: Rc<ScheduledRuntime>,
        materializer: Rc<dyn EvaluationInferenceMaterializer>,
        operation_id: RegisteredOperationId,
        attempt_executor: Rc<dyn InferenceAttemptExecutor>,
    ) -> Self {
        Self {
            runtime,
            materializer,
            operation_id,
            attempt_executor,
        }
    }

    async fn execute_one_attempt(
        &self,
        operation: &HostOperationEnvelope,
        events: &dyn HostExecutionEventSink,
        cancellation: OperationCancellation,
        attempt_id: &str,
    ) -> Result<(HostExecutionTerminal, bool)> {
        let mut turn = self.materializer.materialize(operation)?;
        turn.request_correlation_id = attempt_id.to_string();
        let (terminal_tx, terminal_rx) = oneshot::channel();
        let completion = Box::new(move |_credit, outcome| {
            Box::pin(async move {
                let _ = terminal_tx.send(outcome);
            }) as crate::scheduled::CompletionTask
        });
        if !operation.stream {
            let issued = self.runtime.issue_turn_cancellable(
                turn,
                self.runtime.now_ns(),
                None,
                completion,
                Rc::new(cancellation),
            );
            ensure!(
                issued,
                "Rust scheduling policy rejected evaluator inference"
            );
            let outcome = terminal_rx
                .await
                .context("scheduled evaluator inference lost its terminal callback")?;
            return Ok((normalized_terminal(outcome, &self.operation_id)?, false));
        }

        ensure!(
            self.runtime.supports_response_streaming(),
            "selected inference backend cannot provide true streaming"
        );
        let (response_tx, mut response_rx) = mpsc::channel(64);
        let response_failure = Rc::new(RefCell::new(None));
        let response_observer = Rc::new(InferenceResponseObserver {
            sender: response_tx,
            cancellation: cancellation.clone(),
            failure: response_failure.clone(),
        });
        let issued = self.runtime.issue_turn_streaming_cancellable(
            turn,
            self.runtime.now_ns(),
            None,
            response_observer,
            completion,
            Rc::new(cancellation.clone()),
        );
        ensure!(
            issued,
            "Rust scheduling policy rejected evaluator inference"
        );
        let mut ordinal = 0usize;
        tokio::pin!(terminal_rx);
        let outcome = loop {
            tokio::select! {
                terminal = &mut terminal_rx => {
                    let outcome = terminal.context("scheduled evaluator inference lost its terminal callback")?;
                    while let Ok(response) = response_rx.try_recv() {
                        publish_stream_response(
                            self.operation_id.as_str(),
                            response,
                            events,
                            &mut ordinal,
                        ).await?;
                    }
                    break outcome;
                }
                response = response_rx.recv() => {
                    if let Some(response) = response {
                        publish_stream_response(
                            self.operation_id.as_str(),
                            response,
                            events,
                            &mut ordinal,
                        ).await?;
                    }
                }
            }
        };
        if let Some(failure) = response_failure.borrow_mut().take() {
            return Err(anyhow!(failure));
        }
        Ok((
            normalized_terminal(outcome, &self.operation_id)?,
            ordinal > 0,
        ))
    }
}

struct ScheduledOneAttempt<'a> {
    executor: &'a ScheduledInferenceHostExecutor,
    operation: &'a HostOperationEnvelope,
    events: &'a dyn HostExecutionEventSink,
}

#[async_trait(?Send)]
impl OneAttemptInference for ScheduledOneAttempt<'_> {
    async fn execute_attempt(
        &self,
        _operation_id: &str,
        attempt_id: &str,
        _attempt_ordinal: usize,
        cancellation: OperationCancellation,
    ) -> Result<AttemptExecution> {
        let (terminal, output_observed) = self
            .executor
            .execute_one_attempt(self.operation, self.events, cancellation, attempt_id)
            .await?;
        Ok(AttemptExecution {
            terminal: terminal.class,
            output_observed,
            retryable: terminal.retryable,
            payload: terminal.payload,
            usage: terminal.usage,
        })
    }
}

#[async_trait(?Send)]
impl HostOperationExecutor for ScheduledInferenceHostExecutor {
    async fn execute(
        &self,
        operation: &HostOperationEnvelope,
        events: &dyn HostExecutionEventSink,
        cancellation: OperationCancellation,
    ) -> Result<HostExecutionTerminal> {
        ensure!(
            operation.semantic_operation_id == self.operation_id,
            "prepared inference executor received operation {}, expected {}",
            operation.semantic_operation_id,
            self.operation_id
        );
        let attempt = ScheduledOneAttempt {
            executor: self,
            operation,
            events,
        };
        let result = self
            .attempt_executor
            .execute(&operation.operation_id, false, &attempt, cancellation)
            .await?;
        Ok(HostExecutionTerminal {
            class: result.terminal,
            payload: result.payload,
            usage: result.usage,
            retryable: false,
            transport_attempts: result.attempts,
        })
    }
}

struct InferenceResponseObserver {
    sender: mpsc::Sender<ParsedResponse>,
    cancellation: OperationCancellation,
    failure: Rc<RefCell<Option<String>>>,
}

impl TurnResponseObserver for InferenceResponseObserver {
    fn on_response(&self, response: ParsedResponse) {
        if let Err(error) = self.sender.try_send(response) {
            let message = match error {
                mpsc::error::TrySendError::Full(_) => {
                    "bounded evaluator streaming response queue is full"
                }
                mpsc::error::TrySendError::Closed(_) => {
                    "evaluator streaming response consumer closed"
                }
            };
            self.failure
                .borrow_mut()
                .get_or_insert_with(|| message.into());
            self.cancellation.cancel();
        }
    }
}

async fn publish_stream_response(
    operation_id: &str,
    response: ParsedResponse,
    events: &dyn HostExecutionEventSink,
    ordinal: &mut usize,
) -> Result<()> {
    let Some(payload) = normalize_stream_response(operation_id, response)? else {
        return Ok(());
    };
    events
        .publish(HostExecutionDelta {
            ordinal: *ordinal,
            payload,
        })
        .await?;
    *ordinal = ordinal
        .checked_add(1)
        .ok_or_else(|| anyhow!("evaluator stream ordinal overflow"))?;
    Ok(())
}

fn normalize_stream_response(
    operation_id: &str,
    response: ParsedResponse,
) -> Result<Option<Value>> {
    let Some(data) = response.data else {
        return Ok(None);
    };
    let payload = match operation_id {
        "model.generate" => {
            let content = match data {
                ResponseData::Text { text } => Value::String(text),
                ResponseData::Reasoning { content, reasoning } => {
                    let mut blocks = vec![json!({"type":"reasoning","reasoning":reasoning})];
                    if let Some(content) = content {
                        blocks.push(json!({"type":"text","text":content}));
                    }
                    Value::Array(blocks)
                }
                ResponseData::ToolCall {
                    tool_call_text,
                    content,
                } => Value::String(content.unwrap_or(tool_call_text)),
                other => {
                    return Err(anyhow!(
                        "model.generate received incompatible stream data {other:?}"
                    ));
                }
            };
            json!({
                "choice_index": 0,
                "delta": {"role":"assistant","content":content},
            })
        }
        "model.complete" => json!({"choice_index":0,"text":data.get_text()}),
        "model.responses" => json!({
            "event_type": match data {
                ResponseData::Text { .. } => "response.output_text.delta",
                ResponseData::Reasoning { .. } => "response.reasoning.delta",
                ResponseData::ToolCall { .. } => "response.function_call_arguments.delta",
                _ => "response.output_item.delta",
            },
            "item": serde_json::to_value(data)?,
        }),
        "model.embed" => return Err(anyhow!("model.embed emitted an incremental response")),
        _ => return Err(anyhow!("no stream normalizer for {operation_id:?}")),
    };
    Ok(Some(payload))
}

fn normalized_terminal(
    outcome: TurnDispatchOutcome,
    operation_id: &RegisteredOperationId,
) -> Result<HostExecutionTerminal> {
    let class = match outcome.terminal {
        ReplayTerminalStatus::Completed => HostTerminalClass::Completed,
        ReplayTerminalStatus::Canceled => HostTerminalClass::Cancelled,
        ReplayTerminalStatus::Rejected => HostTerminalClass::Rejected,
        ReplayTerminalStatus::Failed => HostTerminalClass::Failed,
    };
    let payload = if class == HostTerminalClass::Completed {
        normalize_completed_response(operation_id.as_str(), &outcome)?
    } else {
        json!({
            "status": match class {
                HostTerminalClass::Completed => unreachable!("handled above"),
                HostTerminalClass::Failed => "failed",
                HostTerminalClass::Rejected => "rejected",
                HostTerminalClass::Cancelled => "cancelled",
            },
            "error": outcome.model_response.error_kind.as_ref().map(|kind| json!({
                "kind": kind,
                "message": outcome.model_response.error_message,
            })),
        })
    };
    Ok(HostExecutionTerminal {
        class,
        payload,
        usage: HostOperationUsage {
            prompt_tokens: outcome.prompt_tokens,
            completion_tokens: outcome.completion_tokens,
            reasoning_tokens: None,
            cached_tokens: outcome.model_response.cached_prompt_tokens,
        },
        retryable: class == HostTerminalClass::Failed,
        transport_attempts: Vec::new(),
    })
}

fn normalize_completed_response(
    operation_id: &str,
    outcome: &TurnDispatchOutcome,
) -> Result<Value> {
    let usage = normalized_usage(outcome);
    match operation_id {
        "model.generate" => Ok(json!({
            "choices": normalized_generate_choices(outcome)?,
            "usage": usage,
        })),
        "model.complete" => Ok(json!({
            "choices": normalized_completion_choices(outcome)?,
            "usage": usage,
        })),
        "model.responses" => Ok(json!({
            "output": normalized_responses_output(outcome)?,
            "usage": usage,
            "status": "completed",
        })),
        "model.embed" => Ok(json!({
            "embeddings": normalized_embeddings(outcome)?,
            "usage": usage,
        })),
        _ => Err(anyhow!("no terminal normalizer for {operation_id:?}")),
    }
}

fn normalized_usage(outcome: &TurnDispatchOutcome) -> Value {
    let mut usage = Map::new();
    if let Some(value) = outcome.prompt_tokens {
        usage.insert("prompt_tokens".into(), Value::from(value));
    }
    if let Some(value) = outcome.completion_tokens {
        usage.insert("completion_tokens".into(), Value::from(value));
    }
    if let Some(value) = outcome.model_response.cached_prompt_tokens {
        usage.insert("cached_tokens".into(), Value::from(value));
    }
    let reasoning_tokens = outcome
        .model_response
        .wire_responses
        .iter()
        .rev()
        .find_map(|value| {
            value
                .pointer("/usage/completion_tokens_details/reasoning_tokens")
                .or_else(|| value.pointer("/usage/output_tokens_details/reasoning_tokens"))
                .or_else(|| value.pointer("/usage/reasoning_tokens"))
                .and_then(Value::as_u64)
        });
    if let Some(value) = reasoning_tokens {
        usage.insert("reasoning_tokens".into(), Value::from(value));
    }
    Value::Object(usage)
}

fn normalized_generate_choices(outcome: &TurnDispatchOutcome) -> Result<Vec<Value>> {
    if let Some(choices) = outcome
        .model_response
        .wire_responses
        .iter()
        .rev()
        .find_map(|value| value.get("choices").and_then(Value::as_array))
        .filter(|choices| choices.iter().any(|choice| choice.get("message").is_some()))
    {
        return choices
            .iter()
            .map(|choice| {
                let message = choice
                    .get("message")
                    .ok_or_else(|| anyhow!("generate choice omitted message"))?;
                let finish = choice
                    .get("finish_reason")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let mut normalized = Map::from_iter([
                    ("message".into(), normalize_message(message)?),
                    (
                        "stop_reason".into(),
                        Value::String(stop_reason(finish).into()),
                    ),
                ]);
                if let Some(value) = choice.get("finish_reason") {
                    normalized.insert("finish_reason".into(), value.clone());
                }
                if let Some(value) = choice.get("logprobs") {
                    normalized.insert("logprobs".into(), value.clone());
                }
                Ok(Value::Object(normalized))
            })
            .collect();
    }

    let message = outcome
        .model_response
        .assistant_message
        .as_ref()
        .map(normalize_message)
        .transpose()?
        .unwrap_or_else(|| {
            json!({
                "role": "assistant",
                "content": outcome.model_response.content.clone().unwrap_or_else(|| outcome.response_text.clone()),
            })
        });
    let finish = outcome
        .model_response
        .finish_reason
        .as_deref()
        .unwrap_or("unknown");
    Ok(vec![json!({
        "message": message,
        "stop_reason": stop_reason(finish),
        "finish_reason": finish,
    })])
}

fn normalized_completion_choices(outcome: &TurnDispatchOutcome) -> Result<Vec<Value>> {
    if let Some(choices) = outcome
        .model_response
        .wire_responses
        .iter()
        .rev()
        .find_map(|value| value.get("choices").and_then(Value::as_array))
        .filter(|choices| choices.iter().any(|choice| choice.get("text").is_some()))
    {
        return choices
            .iter()
            .map(|choice| {
                let mut normalized = Map::from_iter([
                    (
                        "text".into(),
                        Value::String(
                            choice
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .into(),
                        ),
                    ),
                    (
                        "finish_reason".into(),
                        Value::String(
                            choice
                                .get("finish_reason")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown")
                                .into(),
                        ),
                    ),
                ]);
                if let Some(value) = choice.get("logprobs") {
                    normalized.insert("logprobs".into(), value.clone());
                }
                Ok(Value::Object(normalized))
            })
            .collect();
    }
    Ok(vec![json!({
        "text": outcome.model_response.content.clone().unwrap_or_else(|| outcome.response_text.clone()),
        "finish_reason": outcome.model_response.finish_reason.as_deref().unwrap_or("unknown"),
    })])
}

fn normalized_responses_output(outcome: &TurnDispatchOutcome) -> Result<Vec<Value>> {
    if let Some(output) = outcome
        .model_response
        .wire_responses
        .iter()
        .rev()
        .find_map(|value| value.get("output").and_then(Value::as_array))
    {
        let messages = output
            .iter()
            .filter(|item| {
                item.get("type").and_then(Value::as_str) == Some("message")
                    || item.get("role").is_some()
            })
            .map(normalize_responses_message)
            .collect::<Result<Vec<_>>>()?;
        if !messages.is_empty() {
            return Ok(messages);
        }
    }
    let message = outcome
        .model_response
        .assistant_message
        .as_ref()
        .map(normalize_message)
        .transpose()?
        .unwrap_or_else(|| {
            json!({
                "role":"assistant",
                "content": outcome.model_response.content.clone().unwrap_or_else(|| outcome.response_text.clone()),
            })
        });
    Ok(vec![message])
}

fn normalized_embeddings(outcome: &TurnDispatchOutcome) -> Result<Vec<Value>> {
    for value in outcome.model_response.wire_responses.iter().rev() {
        if let Some(embeddings) = value.get("embeddings").and_then(Value::as_array) {
            return Ok(embeddings.clone());
        }
        if let Some(data) = value.get("data").and_then(Value::as_array) {
            let embeddings = data
                .iter()
                .map(|item| {
                    item.get("embedding")
                        .cloned()
                        .ok_or_else(|| anyhow!("embedding response item omitted embedding"))
                })
                .collect::<Result<Vec<_>>>()?;
            if !embeddings.is_empty() {
                return Ok(embeddings);
            }
        }
    }
    Err(anyhow!("completed embedding response contained no vectors"))
}

fn normalize_responses_message(value: &Value) -> Result<Value> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("Responses output message must be an object"))?;
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("assistant");
    let content = object.get("content").cloned().unwrap_or_default();
    let content = match content {
        Value::Array(blocks) => Value::Array(
            blocks
                .into_iter()
                .filter_map(|block| {
                    let kind = block.get("type").and_then(Value::as_str);
                    let text = block.get("text").and_then(Value::as_str)?;
                    matches!(kind, Some("output_text" | "input_text" | "text"))
                        .then(|| json!({"type":"text","text":text}))
                })
                .collect(),
        ),
        other => other,
    };
    normalize_message(&json!({"role":role,"content":content}))
}

fn normalize_message(value: &Value) -> Result<Value> {
    let source = value
        .as_object()
        .ok_or_else(|| anyhow!("normalized model message must be an object"))?;
    let role = source
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("assistant");
    let mut message = Map::new();
    message.insert("role".into(), Value::String(role.into()));
    let mut content = source
        .get("content")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    if content.is_null() {
        content = Value::String(String::new());
    }
    if let Some(reasoning) = source
        .get("reasoning_content")
        .or_else(|| source.get("reasoning"))
        .and_then(Value::as_str)
    {
        let mut blocks = vec![json!({"type":"reasoning","reasoning":reasoning})];
        if let Some(signature) = source
            .get("reasoning_signature")
            .or_else(|| source.get("signature"))
            .and_then(Value::as_str)
        {
            blocks[0]
                .as_object_mut()
                .expect("reasoning block is an object")
                .insert("signature".into(), Value::String(signature.into()));
        }
        match content {
            Value::String(text) if !text.is_empty() => {
                blocks.push(json!({"type":"text","text":text}));
            }
            Value::Array(existing) => blocks.extend(existing),
            _ => {}
        }
        content = Value::Array(blocks);
    }
    message.insert("content".into(), content);
    for field in ["name", "tool_call_id"] {
        if let Some(value) = source.get(field) {
            message.insert(field.into(), value.clone());
        }
    }
    if let Some(tool_calls) = source.get("tool_calls") {
        message.insert("tool_calls".into(), normalize_tool_calls(tool_calls)?);
    }
    Ok(Value::Object(message))
}

fn normalize_tool_calls(value: &Value) -> Result<Value> {
    let calls = value
        .as_array()
        .ok_or_else(|| anyhow!("upstream tool_calls must be an array"))?;
    calls
        .iter()
        .map(|call| {
            let call = call
                .as_object()
                .ok_or_else(|| anyhow!("upstream tool call must be an object"))?;
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| anyhow!("upstream tool call omitted id"))?;
            ensure!(
                call.get("type").and_then(Value::as_str) == Some("function"),
                "upstream tool call type must be function"
            );
            let function = call
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| anyhow!("upstream tool call omitted function"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| anyhow!("upstream tool call omitted function name"))?;
            let arguments = function
                .get("arguments")
                .ok_or_else(|| anyhow!("upstream tool call omitted function arguments"))?;
            let arguments = match arguments {
                Value::Object(_) => arguments.clone(),
                Value::String(arguments) => {
                    let parsed: Value = serde_json::from_str(arguments)
                        .context("decoding upstream tool-call arguments")?;
                    ensure!(
                        parsed.is_object(),
                        "upstream tool-call arguments must decode to an object"
                    );
                    parsed
                }
                _ => {
                    return Err(anyhow!(
                        "upstream tool-call arguments must be object or JSON"
                    ));
                }
            };
            Ok(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }))
        })
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}

fn stop_reason(value: &str) -> &'static str {
    match value {
        "stop" | "end_turn" | "stop_sequence" => "stop",
        "length" | "max_tokens" => "max_tokens",
        "model_length" => "model_length",
        "tool_calls" | "tool_use" => "tool_calls",
        "content_filter" => "content_filter",
        _ => "unknown",
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
                {"role":"user","content":[{"type":"text","text":"hello"},{"type":"image","asset_id":"image-1","media_type":"image/png"}]},
                {"role":"assistant","content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":{}}}]},
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
                    "input":[{"role":"user","content":"hello"}],
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

    #[test]
    fn registered_factories_pin_exact_provider_schema_triplets() {
        let materializer: Rc<dyn EvaluationInferenceMaterializer> = Rc::new(materializer());
        let mut builder = HostExecutorRegistryBuilder::default();
        register_scheduled_inference_host_executors(&mut builder, materializer).unwrap();
        let registry = builder.freeze().unwrap();
        let descriptors = registry.descriptors().collect::<Vec<_>>();
        assert_eq!(descriptors.len(), STOCK_EVALUATION_OPERATION_SCHEMAS.len());
        for schema in STOCK_EVALUATION_OPERATION_SCHEMAS {
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.operation_id.as_str() == schema.operation_id)
                .unwrap();
            assert_eq!(
                descriptor.request_schema_fingerprint,
                schema.request_schema_sha256
            );
            assert_eq!(
                descriptor.response_schema_fingerprint,
                schema.response_schema_sha256
            );
            assert_eq!(
                descriptor.stream_schema_fingerprint.as_deref(),
                schema
                    .true_streaming
                    .then_some(schema.canonical_stream_schema_sha256)
            );
            assert_eq!(
                descriptor.endpoint_capabilities,
                BTreeSet::from([schema.endpoint_capability.to_string()])
            );
        }
    }

    #[test]
    fn canonical_schema_validators_reject_unknown_authority_and_bad_terminals() {
        let materializer: Rc<dyn EvaluationInferenceMaterializer> = Rc::new(materializer());
        let factory = ScheduledInferenceHostExecutorFactory::new(
            RegisteredOperationId::new("model.generate").unwrap(),
            materializer,
        )
        .unwrap();
        factory
            .validator()
            .validate_request(&json!({
                "messages":[
                    {"role":"user","content":[
                        {"type":"text","text":"hello"},
                        {"type":"image","asset_id":"image-1","media_type":"image/png","detail":"high"},
                        {"type":"tool_result","tool_call_id":"call-0","content":{"ok":true}}
                    ]},
                    {"role":"assistant","content":"","tool_calls":[{
                        "id":"call-1","type":"function","function":{"name":"lookup","arguments":{"q":"x"}}
                    }]}
                ],
                "generation":{"max_tokens":8},
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                "tool_choice":"auto",
                "response_format":{"type":"json_object"},
                "parameters":{"seed":7,"reasoning_effort":"low","parallel_tool_calls":true}
            }))
            .unwrap();
        assert!(
            factory
                .validator()
                .validate_request(&json!({
                    "messages":[{"role":"user","content":"hello"}],
                    "generation":{"max_tokens":8},
                    "base_url":"https://forbidden.invalid"
                }))
                .is_err()
        );
        assert!(
            factory
                .validator()
                .validate_request(&json!({
                    "messages":[{"role":"assistant","content":"","tool_calls":[{
                        "id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}
                    }]}],
                    "generation":{"max_tokens":8}
                }))
                .is_err()
        );
        assert!(
            factory
                .validator()
                .validate_request(&json!({
                    "messages":[{"role":"user","content":[{"type":"image_url","url":"secret"}]}],
                    "generation":{"max_tokens":8}
                }))
                .is_err()
        );
        factory.validator().validate_stream(&Value::Null).unwrap();
        factory
            .validator()
            .validate_response(&json!({
                "choices":[{
                    "message":{"role":"assistant","content":"ok"},
                    "stop_reason":"stop",
                    "finish_reason":"stop"
                }],
                "usage":{"prompt_tokens":3,"completion_tokens":1}
            }))
            .unwrap();
        assert!(
            factory
                .validator()
                .validate_response(&json!({
                    "choices":[],
                    "usage":{"completion_tokens":1}
                }))
                .is_err()
        );
        assert!(
            factory
                .validator()
                .validate_response(&json!({
                    "choices":[{
                        "message":{"role":"assistant","content":"ok"},
                        "stop_reason":"provider_specific"
                    }],
                    "usage":{}
                }))
                .is_err()
        );
    }

    #[test]
    fn terminal_normalizers_preserve_choices_logprobs_embeddings_and_usage() {
        let generated = normalized_terminal(
            TurnDispatchOutcome {
                start_ns: 1,
                end_ns: 2,
                terminal: ReplayTerminalStatus::Completed,
                response_text: "ignored".into(),
                model_response: crate::scheduled::ModelResponseMetadata {
                    wire_responses: vec![json!({
                        "choices":[
                            {"message":{"role":"assistant","content":"one","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},"finish_reason":"tool_calls","logprobs":{"tokens":[1]}},
                            {"message":{"role":"assistant","content":"two"},"finish_reason":"length","logprobs":null}
                        ],
                        "usage":{"completion_tokens_details":{"reasoning_tokens":2}}
                    })],
                    ..Default::default()
                },
                prompt_tokens: Some(4),
                completion_tokens: Some(3),
                http: Default::default(),
            },
            &RegisteredOperationId::new("model.generate").unwrap(),
        )
        .unwrap();
        assert_eq!(generated.payload["choices"].as_array().unwrap().len(), 2);
        assert_eq!(
            generated.payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]["q"],
            "x"
        );
        assert_eq!(generated.payload["choices"][0]["stop_reason"], "tool_calls");
        assert_eq!(generated.payload["choices"][1]["stop_reason"], "max_tokens");
        assert_eq!(generated.payload["usage"]["reasoning_tokens"], 2);

        let embedded = normalized_terminal(
            TurnDispatchOutcome {
                start_ns: 1,
                end_ns: 2,
                terminal: ReplayTerminalStatus::Completed,
                response_text: String::new(),
                model_response: crate::scheduled::ModelResponseMetadata {
                    wire_responses: vec![json!({
                        "data":[{"embedding":[0.1,0.2]},{"embedding":[0.3]}],
                        "usage":{"prompt_tokens":2}
                    })],
                    ..Default::default()
                },
                prompt_tokens: Some(2),
                completion_tokens: None,
                http: Default::default(),
            },
            &RegisteredOperationId::new("model.embed").unwrap(),
        )
        .unwrap();
        assert_eq!(embedded.payload["embeddings"][0], json!([0.1, 0.2]));
    }

    #[test]
    fn normalized_stream_frames_are_typed_and_never_raw_sse() {
        let response = ParsedResponse {
            perf_ns: 7,
            data: Some(ResponseData::Reasoning {
                content: Some("answer".into()),
                reasoning: "think".into(),
            }),
            usage: None,
            sources: None,
        };
        let payload = normalize_stream_response("model.generate", response)
            .unwrap()
            .unwrap();
        assert_eq!(payload["choice_index"], 0);
        assert_eq!(payload["delta"]["content"][0]["type"], "reasoning");
        assert!(!payload.to_string().contains("data:"));
    }
}
