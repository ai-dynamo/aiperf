// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request/response models.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn tokens(&self) -> usize {
        match self {
            ReasoningEffort::Low => 100,
            ReasoningEffort::Medium => 250,
            ReasoningEffort::High => 500,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct StreamOptions {
    pub include_usage: bool,
    pub continuous_usage_stats: bool,
}

fn default_first_chunk_tokens() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub ignore_eos: bool,
    pub min_tokens: Option<usize>,
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Request priority for the `priority` KV-cache eviction policy (SGLang
    /// `--radix-eviction-policy priority`). Higher survives eviction longer;
    /// absent => 0 (so the policy behaves as LRU).
    #[serde(default)]
    pub priority: Option<i64>,
    /// Test seam for servers that bundle several output tokens in their first
    /// streamed content chunk.
    #[serde(default = "default_first_chunk_tokens")]
    pub mock_first_chunk_tokens: usize,
}

impl ChatCompletionRequest {
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false)
    }

    pub fn max_output_tokens(&self) -> Option<usize> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub fn continuous_usage_stats(&self) -> bool {
        self.stream_options
            .as_ref()
            .is_some_and(|options| options.continuous_usage_stats)
    }

    pub fn first_chunk_tokens(&self) -> usize {
        self.mock_first_chunk_tokens.max(1)
    }
}

/// Supported vLLM/Dynamo token-native sampling parameters.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct VllmSamplingParams {
    pub max_tokens: Option<usize>,
    pub min_tokens: Option<usize>,
    pub ignore_eos: bool,
}

/// vLLM/Dynamo token-native Generate request (`POST /inference/v1/generate`).
///
/// `token_ids` are both the prompt and the source for deterministic output IDs.
/// The endpoint is non-streaming.
#[derive(Debug, Clone, Deserialize)]
pub struct VllmGenerateRequest {
    pub model: String,
    #[serde(default)]
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub sampling_params: VllmSamplingParams,
    #[serde(default)]
    pub stream: bool,
    pub request_id: Option<String>,
}

/// OpenAI Responses API request (`POST /v1/responses`).
///
/// `input` accepts a string or message items with string or typed-part content.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub input: Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub instructions: Value,
    pub max_output_tokens: Option<usize>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ResponsesRequest {
    /// Flattens `instructions` and text-bearing `input` parts into one prompt.
    pub fn prompt_text(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        collect_responses_text(&self.instructions, &mut parts);
        collect_responses_text(&self.input, &mut parts);
        parts.join("\n")
    }
}

fn collect_responses_text(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(s) if !s.is_empty() => out.push(s.clone()),
        Value::Array(items) => {
            for item in items {
                collect_responses_text(item, out);
            }
        }
        Value::Object(obj) => {
            if let Some(text) = obj.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                out.push(text.to_string());
            }
            if let Some(content) = obj.get("content") {
                collect_responses_text(content, out);
            }
        }
        _ => {}
    }
}

/// Anthropic Messages API request accepted by the local mock.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    /// Request priority for the `priority` KV-cache eviction policy.
    #[serde(default)]
    pub priority: Option<i64>,
    /// Test seam for servers that bundle several output tokens in their first
    /// streamed content chunk.
    #[serde(default = "default_first_chunk_tokens")]
    pub mock_first_chunk_tokens: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrList {
    String(String),
    List(Vec<String>),
}

impl StringOrList {
    pub fn as_vec(&self) -> Vec<String> {
        match self {
            StringOrList::String(s) => vec![s.clone()],
            StringOrList::List(v) => v.clone(),
        }
    }

    pub fn joined(&self, sep: &str) -> String {
        match self {
            StringOrList::String(s) => s.clone(),
            StringOrList::List(v) => v
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(sep),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: StringOrList,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub ignore_eos: bool,
    pub min_tokens: Option<usize>,
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Request priority for the `priority` KV-cache eviction policy (SGLang
    /// `--radix-eviction-policy priority`). Higher survives eviction longer;
    /// absent => 0 (so the policy behaves as LRU).
    #[serde(default)]
    pub priority: Option<i64>,
    /// Test seam for servers that bundle several output tokens in their first
    /// streamed content chunk.
    #[serde(default = "default_first_chunk_tokens")]
    pub mock_first_chunk_tokens: usize,
}

impl CompletionRequest {
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false)
    }

    pub fn prompt_text(&self) -> String {
        self.prompt.joined("\n")
    }

    pub fn continuous_usage_stats(&self) -> bool {
        self.stream_options
            .as_ref()
            .is_some_and(|options| options.continuous_usage_stats)
    }

    pub fn first_chunk_tokens(&self) -> usize {
        self.mock_first_chunk_tokens.max(1)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: StringOrList,
}

impl EmbeddingRequest {
    pub fn inputs(&self) -> Vec<String> {
        self.input.as_vec()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueryText {
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PassageText {
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RankingRequest {
    pub model: String,
    pub query: QueryText,
    pub passages: Vec<PassageText>,
}

impl RankingRequest {
    pub fn query_text(&self) -> &str {
        &self.query.text
    }

    pub fn passage_texts(&self) -> Vec<&str> {
        self.passages.iter().map(|p| p.text.as_str()).collect()
    }
}

fn default_tei_model() -> String {
    "tei-reranker".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct HFTEIRerankRequest {
    pub query: String,
    pub texts: Option<Vec<String>>,
    pub documents: Option<Vec<String>>,
    #[serde(default = "default_tei_model")]
    pub model: String,
}

impl HFTEIRerankRequest {
    pub fn query_text(&self) -> &str {
        &self.query
    }

    pub fn passage_texts(&self) -> Vec<&str> {
        match (&self.texts, &self.documents) {
            (Some(v), _) => v.iter().map(String::as_str).collect(),
            (None, Some(v)) => v.iter().map(String::as_str).collect(),
            _ => Vec::new(),
        }
    }
}

fn default_cohere_model() -> String {
    "cohere-reranker".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct CohereRerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default = "default_cohere_model")]
    pub model: String,
}

impl CohereRerankRequest {
    pub fn query_text(&self) -> &str {
        &self.query
    }

    pub fn passage_texts(&self) -> Vec<&str> {
        self.documents.iter().map(String::as_str).collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TGIParameters {
    #[serde(default = "tgi_default_max_new_tokens")]
    pub max_new_tokens: usize,
}

fn tgi_default_max_new_tokens() -> usize {
    50
}

impl Default for TGIParameters {
    fn default() -> Self {
        Self { max_new_tokens: 50 }
    }
}

fn default_tgi_model() -> String {
    "tgi".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TGIGenerateRequest {
    pub inputs: Option<String>,
    #[serde(default)]
    pub parameters: TGIParameters,
    #[serde(default = "default_tgi_model")]
    pub model: String,
    #[serde(default)]
    pub ignore_eos: bool,
    pub min_tokens: Option<usize>,
}

impl TGIGenerateRequest {
    pub fn prompt_text(&self) -> String {
        self.inputs.clone().unwrap_or_else(|| "Hello!".to_string())
    }

    pub fn max_tokens(&self) -> Option<usize> {
        Some(self.parameters.max_new_tokens)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResponseFormat {
    Url,
    B64Json,
}

fn default_image_model() -> String {
    "black-forest-labs/FLUX.1-dev".to_string()
}

fn default_image_n() -> u32 {
    1
}

fn default_image_response_format() -> ImageResponseFormat {
    ImageResponseFormat::B64Json
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[serde(default = "default_image_model")]
    pub model: String,
    #[serde(default = "default_image_n")]
    pub n: u32,
    #[serde(default = "default_image_response_format")]
    pub response_format: ImageResponseFormat,
    #[serde(default)]
    pub stream: bool,
    pub size: Option<String>,
    pub quality: Option<String>,
    pub style: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageRetrievalInput {
    #[serde(rename = "type")]
    pub kind: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageRetrievalRequest {
    pub input: Vec<ImageRetrievalInput>,
}

fn default_solido_inference_model() -> String {
    "default-model".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct SolidoRAGRequest {
    pub query: Vec<String>,
    #[serde(default)]
    pub filters: Value,
    #[serde(default = "default_solido_inference_model")]
    pub inference_model: String,
    #[serde(default)]
    pub ignore_eos: bool,
    pub min_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Prompt tokens written into the provider KV cache. Emitted only when
    /// `--usage-cache-write-tokens` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<usize>,
    /// Prompt cache misses configured by `--usage-cache-miss-tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<usize>,
    /// Gemini `toolUsePromptTokenCount`, configured by
    /// `--usage-tool-use-prompt-tokens`.
    #[serde(
        rename = "toolUsePromptTokenCount",
        skip_serializing_if = "Option::is_none"
    )]
    pub tool_use_prompt_token_count: Option<usize>,
    /// Prompt-audio duration in seconds, configured by
    /// `--usage-prompt-audio-seconds`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_audio_seconds: Option<f64>,
    /// Anthropic disjoint cache-read count. Serialized only into the Anthropic
    /// `messages` usage object (see [`crate::handlers`]), never the OpenAI usage
    /// (OpenAI reports cache reads via `prompt_tokens_details.cached_tokens`).
    #[serde(skip)]
    pub cache_read_input_tokens: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
    /// Output audio tokens configured by `--usage-completion-audio-tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<usize>,
    /// Accepted predicted-output tokens configured by
    /// `--usage-accepted-prediction-tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<usize>,
    /// Rejected predicted-output tokens configured by
    /// `--usage-rejected-prediction-tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<usize>,
}

/// OpenAI/vLLM-style prompt token breakdown. `cached_tokens` is the prefix
/// served from the KV cache (read by AIPerf as `usage_prompt_cache_read_tokens`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
    /// Prompt audio tokens configured by `--usage-prompt-audio-tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_parses_max_completion_tokens() {
        let raw = r#"{"model": "x", "messages": [], "max_completion_tokens": 42}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.max_output_tokens(), Some(42));
    }

    #[test]
    fn completion_prompt_joins_list() {
        let raw = r#"{"model": "x", "prompt": ["a", "", "b"]}"#;
        let req: CompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.prompt_text(), "a\nb");
    }

    #[test]
    fn embedding_input_list_or_string() {
        let single: EmbeddingRequest =
            serde_json::from_str(r#"{"model": "x", "input": "hello"}"#).unwrap();
        assert_eq!(single.inputs(), vec!["hello".to_string()]);

        let list: EmbeddingRequest =
            serde_json::from_str(r#"{"model": "x", "input": ["a", "b"]}"#).unwrap();
        assert_eq!(list.inputs(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn include_usage_reads_stream_options() {
        let raw = r#"{"model": "x", "messages": [], "stream": true, "stream_options": {"include_usage": true}}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert!(req.include_usage());
    }

    #[test]
    fn completion_requests_read_continuous_usage_and_first_chunk_policy() {
        let chat: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model": "x",
                "messages": [],
                "stream": true,
                "stream_options": {
                    "include_usage": true,
                    "continuous_usage_stats": true
                },
                "mock_first_chunk_tokens": 3
            }"#,
        )
        .unwrap();
        assert!(chat.continuous_usage_stats());
        assert_eq!(chat.first_chunk_tokens(), 3);

        let text: CompletionRequest = serde_json::from_str(
            r#"{
                "model": "x",
                "prompt": "hello",
                "stream": true,
                "stream_options": {"continuous_usage_stats": true},
                "mock_first_chunk_tokens": 0
            }"#,
        )
        .unwrap();
        assert!(text.continuous_usage_stats());
        assert_eq!(text.first_chunk_tokens(), 1);

        let chat_defaults: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"x","messages":[]}"#).unwrap();
        assert!(!chat_defaults.continuous_usage_stats());
        assert_eq!(chat_defaults.first_chunk_tokens(), 1);

        let text_defaults: CompletionRequest =
            serde_json::from_str(r#"{"model":"x","prompt":"hello"}"#).unwrap();
        assert!(!text_defaults.continuous_usage_stats());
        assert_eq!(text_defaults.first_chunk_tokens(), 1);
    }

    #[test]
    fn tei_prefers_texts_over_documents() {
        let req: HFTEIRerankRequest =
            serde_json::from_str(r#"{"query": "q", "texts": ["t1"], "documents": ["d1"]}"#)
                .unwrap();
        assert_eq!(req.passage_texts(), vec!["t1"]);
    }

    #[test]
    fn reasoning_effort_token_counts() {
        assert_eq!(ReasoningEffort::Low.tokens(), 100);
        assert_eq!(ReasoningEffort::Medium.tokens(), 250);
        assert_eq!(ReasoningEffort::High.tokens(), 500);
    }
}
