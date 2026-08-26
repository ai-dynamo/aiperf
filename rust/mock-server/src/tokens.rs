// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokenization and response generation.
//!
//! Character-based tokenization, bounded caching, deterministic generation,
//! token budgets, and reasoning-token accounting.

use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use dashmap::DashMap;
use once_cell::sync::{Lazy, OnceCell};
use serde_json::Value;

use crate::models::{
    ChatCompletionRequest, CohereRerankRequest, CompletionRequest, EmbeddingRequest,
    HFTEIRerankRequest, ImageGenerationRequest, ImageRetrievalRequest, MessagesRequest,
    RankingRequest, ReasoningEffort, SolidoRAGRequest, TGIGenerateRequest,
};

/// Upper bound before we stop caching new entries. At steady state the mock
/// server sees a bounded set of synthetic prompts; if it blows past this we
/// just stop inserting (we never evict — eviction under DashMap contention
/// would negate the win).
const TOKENIZE_CACHE_CAP: usize = 16_384;

/// Don't cache prompts larger than this (~64k tokens at ~4 chars/token). A
/// cached entry holds the source string plus its full `Vec<String>` of tokens,
/// so a handful of unique huge prompts (e.g. 1M-token payloads) would otherwise
/// blow the process memory. Above this size we recompute every time —
/// `tokenize_uncached` is O(n) and cheap next to the request's simulated latency.
const TOKENIZE_CACHE_MAX_BYTES: usize = 256 * 1024;

/// Lock-free sharded tokenization cache.
static TOKENIZE_CACHE: Lazy<DashMap<String, Vec<String>>> =
    Lazy::new(|| DashMap::with_capacity_and_shard_amount(1024, 64));

/// Populated exactly once at startup, read on every generation call after
/// that — `OnceCell` gives lock-free reads (just a loaded-check + pointer
/// deref) instead of an `RwLock` acquisition on every request, since there's
/// never a writer contending with readers in steady state.
static CORPUS_TOKENS: OnceCell<Option<Vec<String>>> = OnceCell::new();

/// Loads the optional Shakespeare corpus once.
pub fn load_corpus() {
    if CORPUS_TOKENS.get().is_some() {
        return;
    }

    let path = find_corpus_path();
    let tokens = match path {
        Some(p) => match std::fs::read_to_string(&p) {
            Ok(text) => {
                let normalized: String = text
                    .lines()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                let tokens = tokenize_uncached(&normalized);
                tracing::info!(
                    "Corpus loaded: {} tokens from {}",
                    tokens.len(),
                    p.display()
                );
                Some(tokens)
            }
            Err(e) => {
                tracing::warn!("Failed to read corpus at {}: {}", p.display(), e);
                None
            }
        },
        None => {
            tracing::warn!("Corpus file not found, falling back to prompt cycling");
            None
        }
    };

    // `set` fails if another thread raced us and already populated it; that's
    // fine, the loser's `tokens` is simply dropped — first writer wins, same
    // as the old read-check-then-write-lock pattern.
    let _ = CORPUS_TOKENS.set(tokens);
}

fn find_corpus_path() -> Option<PathBuf> {
    if let Ok(override_path) = std::env::var("AIPERF_MOCK_CORPUS_PATH") {
        let p = PathBuf::from(override_path);
        if p.exists() {
            return Some(p);
        }
    }

    let candidates = [
        workspace_corpus_path(),
        PathBuf::from("src/aiperf/dataset/generator/assets/shakespeare.txt"),
        PathBuf::from("../src/aiperf/dataset/generator/assets/shakespeare.txt"),
        PathBuf::from("../../src/aiperf/dataset/generator/assets/shakespeare.txt"),
        PathBuf::from("../../../src/aiperf/dataset/generator/assets/shakespeare.txt"),
    ];
    candidates.into_iter().find(|path| path.exists())
}

fn workspace_corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/aiperf/dataset/generator/assets/shakespeare.txt")
}

/// Splits at roughly four characters per token, preferring nearby whitespace.
fn tokenize_uncached(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    // Byte scanning is safe for ASCII; Unicode input uses character boundaries.
    let bytes = text.as_bytes();

    if bytes.iter().all(|b| b.is_ascii()) {
        tokenize_ascii(bytes)
    } else {
        tokenize_unicode(text)
    }
}

fn tokenize_ascii(bytes: &[u8]) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut i = 0;
    let len = bytes.len();
    while i < len {
        let mut end = (i + 4).min(len);
        if end < len && !(bytes[end] as char).is_whitespace() {
            let limit = (end + 2).min(len);
            #[allow(clippy::needless_range_loop)]
            for j in end..limit {
                if (bytes[j] as char).is_whitespace() {
                    end = j + 1;
                    break;
                }
            }
        }
        tokens.push(std::str::from_utf8(&bytes[i..end]).unwrap().to_string());
        i = end;
    }
    tokens
}

fn tokenize_unicode(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    let len = chars.len();
    while i < len {
        let mut end = (i + 4).min(len);
        if end < len && !chars[end].is_whitespace() {
            let limit = (end + 2).min(len);
            #[allow(clippy::needless_range_loop)]
            for j in end..limit {
                if chars[j].is_whitespace() {
                    end = j + 1;
                    break;
                }
            }
        }
        tokens.push(chars[i..end].iter().collect());
        i = end;
    }
    tokens
}

pub fn tokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    // Skip the cache for very short strings — the hash+lookup costs more than
    // just computing the 1–2 tokens directly — and for very large prompts,
    // whose cached token vectors would otherwise exhaust memory under load.
    if text.len() <= 8 || text.len() > TOKENIZE_CACHE_MAX_BYTES {
        return tokenize_uncached(text);
    }
    if let Some(entry) = TOKENIZE_CACHE.get(text) {
        return entry.clone();
    }
    let tokens = tokenize_uncached(text);
    if TOKENIZE_CACHE.len() < TOKENIZE_CACHE_CAP {
        TOKENIZE_CACHE.insert(text.to_string(), tokens.clone());
    }
    tokens
}

pub fn count_tokens(text: &str) -> usize {
    tokenize(text).len()
}

/// Tokenized output for a generated request.
#[derive(Debug, Clone, Default)]
pub struct TokenizedText {
    pub text: String,
    pub tokens: Vec<String>,
    pub prompt_token_count: usize,
    pub reasoning_tokens: usize,
    pub reasoning_content_tokens: Vec<String>,
    pub finish_reason: &'static str,
}

impl TokenizedText {
    pub fn count(&self) -> usize {
        self.tokens.len()
    }

    pub fn content(&self) -> String {
        self.tokens.concat()
    }

    pub fn reasoning_content(&self) -> Option<String> {
        if self.reasoning_content_tokens.is_empty() {
            None
        } else {
            Some(self.reasoning_content_tokens.concat())
        }
    }

    pub fn usage(&self) -> crate::models::Usage {
        let completion = self.count() + self.reasoning_tokens;
        crate::models::Usage {
            prompt_tokens: self.prompt_token_count,
            completion_tokens: completion,
            total_tokens: self.prompt_token_count + completion,
            completion_tokens_details: if self.reasoning_tokens > 0 {
                Some(crate::models::CompletionTokensDetails {
                    reasoning_tokens: self.reasoning_tokens,
                    ..Default::default()
                })
            } else {
                None
            },
            // Always populated by RequestCtx::build, with zero cached tokens
            // when the prefix cache is off.
            prompt_tokens_details: None,
            // Optional extended usage accounting, populated by RequestCtx::build
            // from the `--usage-*` config knobs (absent by default).
            cache_creation_input_tokens: None,
            prompt_cache_miss_tokens: None,
            tool_use_prompt_token_count: None,
            prompt_audio_seconds: None,
            cache_read_input_tokens: None,
        }
    }

    pub fn stop() -> Self {
        Self {
            finish_reason: "stop",
            ..Self::default()
        }
    }
}

pub enum GenRequest<'a> {
    Chat(&'a ChatCompletionRequest),
    Messages(&'a MessagesRequest),
    Completion(&'a CompletionRequest),
    TGI(&'a TGIGenerateRequest),
    Embedding(&'a EmbeddingRequest),
    Ranking(&'a RankingRequest),
    HFTEIRerank(&'a HFTEIRerankRequest),
    CohereRerank(&'a CohereRerankRequest),
    ImageGeneration(&'a ImageGenerationRequest),
    ImageRetrieval(&'a ImageRetrievalRequest),
    SolidoRAG(&'a SolidoRAGRequest),
}

impl GenRequest<'_> {
    pub fn model(&self) -> &str {
        match self {
            GenRequest::Chat(r) => &r.model,
            GenRequest::Messages(r) => &r.model,
            GenRequest::Completion(r) => &r.model,
            GenRequest::TGI(r) => &r.model,
            GenRequest::Embedding(r) => &r.model,
            GenRequest::Ranking(r) => &r.model,
            GenRequest::HFTEIRerank(r) => &r.model,
            GenRequest::CohereRerank(r) => &r.model,
            GenRequest::ImageGeneration(r) => &r.model,
            GenRequest::ImageRetrieval(_) => "image-retrieval",
            GenRequest::SolidoRAG(r) => &r.inference_model,
        }
    }

    /// Request priority for the `priority` KV-cache eviction policy.
    pub fn priority(&self) -> i64 {
        match self {
            GenRequest::Chat(r) => r.priority.unwrap_or(0),
            GenRequest::Messages(r) => r.priority.unwrap_or(0),
            GenRequest::Completion(r) => r.priority.unwrap_or(0),
            _ => 0,
        }
    }
}

fn extract_chat_text(messages: &[crate::models::Message]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for m in messages {
        if m.role != "user" {
            continue;
        }
        match &m.content {
            Value::String(s) => parts.push(s.clone()),
            Value::Array(items) => {
                for item in items {
                    if let Some(obj) = item.as_object()
                        && obj.get("type").and_then(Value::as_str) == Some("text")
                        && let Some(t) = obj.get("text").and_then(Value::as_str)
                    {
                        parts.push(t.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    parts.join("\n")
}

fn extract_content(req: &GenRequest<'_>) -> (String, Option<usize>) {
    match req {
        GenRequest::Chat(r) => (extract_chat_text(&r.messages), r.max_output_tokens()),
        GenRequest::Messages(r) => (extract_chat_text(&r.messages), Some(r.max_tokens)),
        GenRequest::Completion(r) => (r.prompt_text(), r.max_tokens),
        GenRequest::TGI(r) => (r.prompt_text(), r.max_tokens()),
        GenRequest::Embedding(r) => (r.inputs().join("\n"), None),
        GenRequest::Ranking(r) => {
            let mut t = r.query_text().to_string();
            for p in r.passage_texts() {
                t.push('\n');
                t.push_str(p);
            }
            (t, None)
        }
        GenRequest::HFTEIRerank(r) => {
            let mut t = r.query_text().to_string();
            for p in r.passage_texts() {
                t.push('\n');
                t.push_str(p);
            }
            (t, None)
        }
        GenRequest::CohereRerank(r) => {
            let mut t = r.query_text().to_string();
            for p in r.passage_texts() {
                t.push('\n');
                t.push_str(p);
            }
            (t, None)
        }
        GenRequest::ImageGeneration(r) => (r.prompt.clone(), None),
        GenRequest::ImageRetrieval(_) => (String::new(), None),
        GenRequest::SolidoRAG(r) => (r.query.join(" "), None),
    }
}

/// Generate a deterministic seed from the first five prompt tokens.
fn generate_seed(prompt_tokens: &[String]) -> u64 {
    if prompt_tokens.is_empty() {
        return 0;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for t in prompt_tokens.iter().take(5) {
        t.hash(&mut hasher);
    }
    hasher.finish() % 1000
}

fn cycle_tokens(prompt_tokens: &[String], num_tokens: usize, offset: usize) -> Vec<String> {
    if num_tokens == 0 {
        return Vec::new();
    }
    if let Some(corpus) = CORPUS_TOKENS.get().and_then(|o| o.as_ref()) {
        if corpus.is_empty() {
            return cycle_prompt(prompt_tokens, num_tokens, offset);
        }
        let seed = generate_seed(prompt_tokens) as usize;
        let start = (seed + offset) % corpus.len();
        let mut out = Vec::with_capacity(num_tokens);
        for i in 0..num_tokens {
            out.push(corpus[(start + i) % corpus.len()].clone());
        }
        out
    } else {
        cycle_prompt(prompt_tokens, num_tokens, offset)
    }
}

fn cycle_prompt(prompt_tokens: &[String], num_tokens: usize, offset: usize) -> Vec<String> {
    // When there is no prompt to cycle (e.g. NativeGraph start node with empty
    // inputs), fall back to a static placeholder so the mock still emits valid
    // streaming content. An empty return causes parsed_content=false downstream,
    // which the transport classifies as a failed reply even on HTTP 200.
    static FALLBACK: &[&str] = &["ok", " "];
    let source: &[String];
    let fallback_owned: Vec<String>;
    if prompt_tokens.is_empty() {
        fallback_owned = FALLBACK.iter().map(|s| s.to_string()).collect();
        source = &fallback_owned;
    } else {
        source = prompt_tokens;
    }
    (0..num_tokens)
        .map(|i| source[(offset + i) % source.len()].clone())
        .collect()
}

fn cycle_tokens_reversed(prompt_tokens: &[String], num_tokens: usize) -> Vec<String> {
    if num_tokens == 0 {
        return Vec::new();
    }
    let offset = CORPUS_TOKENS
        .get()
        .and_then(|o| o.as_ref())
        .map(|c| c.len() / 2)
        .unwrap_or(0);
    cycle_tokens(prompt_tokens, num_tokens, offset)
}

#[derive(Debug)]
struct TokenBudget {
    min_tokens: usize,
    max_tokens: usize,
}

fn calculate_budget(
    prompt_token_count: usize,
    max_tokens: Option<usize>,
    min_tokens: Option<usize>,
) -> TokenBudget {
    let max_budget = match max_tokens {
        Some(m) => m,
        None => (prompt_token_count * 2).max(16),
    };
    let min_budget = match min_tokens {
        Some(m) => m.min(max_budget),
        None => {
            let floor = ((prompt_token_count as f64) * 0.8) as usize;
            let base = floor.max(1);
            base.min(max_budget)
        }
    };
    TokenBudget {
        min_tokens: min_budget,
        max_tokens: max_budget,
    }
}

fn calculate_variable_token_count(
    prompt_tokens: &[String],
    prompt_token_count: usize,
    min_tokens: Option<usize>,
    max_tokens: Option<usize>,
    budget: &TokenBudget,
) -> usize {
    let seed = generate_seed(prompt_tokens);
    let target_max = if max_tokens.is_some() || min_tokens.is_some() {
        budget.max_tokens
    } else {
        let ceiling = ((prompt_token_count as f64) * 1.2) as usize;
        let capped = ceiling.min(budget.max_tokens);
        capped.max(budget.min_tokens)
    };
    let range_size = target_max
        .saturating_sub(budget.min_tokens)
        .saturating_add(1);
    if range_size == 0 {
        return budget.min_tokens;
    }
    budget.min_tokens + (seed as usize % range_size)
}

fn generate_output_tokens(
    prompt_tokens: &[String],
    prompt_token_count: usize,
    max_tokens: Option<usize>,
    min_tokens: Option<usize>,
    ignore_eos: bool,
) -> (Vec<String>, &'static str) {
    let budget = calculate_budget(prompt_token_count, max_tokens, min_tokens);
    if ignore_eos {
        return (cycle_tokens(prompt_tokens, budget.max_tokens, 0), "length");
    }
    let num = calculate_variable_token_count(
        prompt_tokens,
        prompt_token_count,
        min_tokens,
        max_tokens,
        &budget,
    );
    let reason = if num == budget.max_tokens {
        "length"
    } else {
        "stop"
    };
    (cycle_tokens(prompt_tokens, num, 0), reason)
}

/// Deterministically choose the output length and emit integer output token IDs
/// for the token-native vLLM Generate endpoint.
///
/// Reuses the exact same budget/variable-count logic as the text path
/// (`calculate_budget` / `calculate_variable_token_count`) keyed on the input
/// token IDs, so a tuned run with `sampling_params.max_tokens = N` and a
/// sufficiently long prompt (`0.8 * ISL >= N`) yields exactly `N` output tokens —
/// the same exact-OSL property the chat/completions e2e paths rely on. `ignore_eos`
/// fills to `max_tokens`. Output IDs cycle the input IDs (a stable, in-range
/// sequence); an empty input falls back to a monotonic sequence so the array is
/// never empty when tokens are requested.
pub fn generate_output_token_ids(
    input_token_ids: &[u32],
    max_tokens: Option<usize>,
    min_tokens: Option<usize>,
    ignore_eos: bool,
) -> (Vec<u32>, &'static str) {
    let prompt_token_count = input_token_ids.len();
    // Render IDs as strings so text and token-native generation share seed logic.
    let seed_repr: Vec<String> = input_token_ids
        .iter()
        .take(5)
        .map(|id| id.to_string())
        .collect();
    let budget = calculate_budget(prompt_token_count, max_tokens, min_tokens);
    let (count, reason) = if ignore_eos {
        (budget.max_tokens, "length")
    } else {
        let num = calculate_variable_token_count(
            &seed_repr,
            prompt_token_count,
            min_tokens,
            max_tokens,
            &budget,
        );
        let reason = if num == budget.max_tokens {
            "length"
        } else {
            "stop"
        };
        (num, reason)
    };
    let out = if input_token_ids.is_empty() {
        (0..count).map(|i| i as u32).collect()
    } else {
        (0..count)
            .map(|i| input_token_ids[i % input_token_ids.len()])
            .collect()
    };
    (out, reason)
}

struct ReasoningResult {
    token_count: usize,
    content_tokens: Vec<String>,
    remaining_budget: Option<usize>,
}

fn is_reasoning_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("gpt-oss") || lower.contains("qwen")
}

fn generate_reasoning_tokens(
    req: &GenRequest<'_>,
    prompt_tokens: &[String],
    prompt_token_count: usize,
    max_tokens: Option<usize>,
) -> ReasoningResult {
    let chat = match req {
        GenRequest::Chat(r) => r,
        _ => {
            return ReasoningResult {
                token_count: 0,
                content_tokens: Vec::new(),
                remaining_budget: max_tokens,
            };
        }
    };
    if !is_reasoning_model(&chat.model) {
        return ReasoningResult {
            token_count: 0,
            content_tokens: Vec::new(),
            remaining_budget: max_tokens,
        };
    }
    let requested = chat
        .reasoning_effort
        .as_ref()
        .map(ReasoningEffort::tokens)
        .unwrap_or(250);
    let total_budget = max_tokens.unwrap_or_else(|| (prompt_token_count * 2).max(16));
    let actual = requested.min(total_budget);
    let tokens = cycle_tokens_reversed(prompt_tokens, actual);
    ReasoningResult {
        token_count: actual,
        content_tokens: tokens,
        remaining_budget: Some(total_budget - actual),
    }
}

fn min_tokens_of(req: &GenRequest<'_>) -> Option<usize> {
    match req {
        GenRequest::Chat(r) => r.min_tokens,
        GenRequest::Messages(_) => None,
        GenRequest::Completion(r) => r.min_tokens,
        GenRequest::TGI(r) => r.min_tokens,
        GenRequest::SolidoRAG(r) => r.min_tokens,
        _ => None,
    }
}

fn ignore_eos_of(req: &GenRequest<'_>) -> bool {
    match req {
        GenRequest::Chat(r) => r.ignore_eos,
        GenRequest::Messages(_) => false,
        GenRequest::Completion(r) => r.ignore_eos,
        GenRequest::TGI(r) => r.ignore_eos,
        GenRequest::SolidoRAG(r) => r.ignore_eos,
        _ => false,
    }
}

pub fn tokenize_request(req: &GenRequest<'_>) -> TokenizedText {
    tokenize_request_with_fixed_output_tokens(req, None)
}

/// Tokenize a request and optionally fix its generated response length.
///
/// The override is a mock-server test seam. It does not rewrite the parsed
/// request cap, which remains available to request-capture assertions.
pub fn tokenize_request_with_fixed_output_tokens(
    req: &GenRequest<'_>,
    fixed_output_tokens: Option<usize>,
) -> TokenizedText {
    let (text, max_tokens) = extract_content(req);
    let prompt_tokens = tokenize(&text);
    let prompt_token_count = prompt_tokens.len();

    if matches!(
        req,
        GenRequest::Embedding(_)
            | GenRequest::Ranking(_)
            | GenRequest::HFTEIRerank(_)
            | GenRequest::CohereRerank(_)
            | GenRequest::ImageGeneration(_)
            | GenRequest::ImageRetrieval(_)
    ) {
        return TokenizedText {
            text,
            tokens: Vec::new(),
            prompt_token_count,
            reasoning_tokens: 0,
            reasoning_content_tokens: Vec::new(),
            finish_reason: "stop",
        };
    }

    if let Some(output_tokens) = fixed_output_tokens {
        return TokenizedText {
            text,
            tokens: cycle_tokens(&prompt_tokens, output_tokens, 0),
            prompt_token_count,
            reasoning_tokens: 0,
            reasoning_content_tokens: Vec::new(),
            finish_reason: "length",
        };
    }

    if prompt_tokens.is_empty() {
        // Requests with no extractable text (e.g. NativeGraph start node with
        // empty messages) still carry max_tokens from the model binding, so
        // honour it: emit output tokens seeded from an empty prompt so the
        // transport sees non-zero content and classifies the reply as
        // Completed rather than Failed.
        let (output_tokens, finish_reason) = generate_output_tokens(
            &prompt_tokens,
            0,
            max_tokens,
            min_tokens_of(req),
            ignore_eos_of(req),
        );
        return TokenizedText {
            text,
            tokens: output_tokens,
            prompt_token_count: 0,
            reasoning_tokens: 0,
            reasoning_content_tokens: Vec::new(),
            finish_reason,
        };
    }

    let reasoning = generate_reasoning_tokens(req, &prompt_tokens, prompt_token_count, max_tokens);
    let (output_tokens, finish_reason) = generate_output_tokens(
        &prompt_tokens,
        prompt_token_count,
        reasoning.remaining_budget,
        min_tokens_of(req),
        ignore_eos_of(req),
    );

    TokenizedText {
        text,
        tokens: output_tokens,
        prompt_token_count,
        reasoning_tokens: reasoning.token_count,
        reasoning_content_tokens: reasoning.content_tokens,
        finish_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ChatCompletionRequest, Message};

    fn chat(model: &str, content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.into(),
            messages: vec![Message {
                role: "user".into(),
                content: Value::String(content.into()),
            }],
            stream: false,
            stream_options: None,
            max_tokens: None,
            max_completion_tokens: None,
            ignore_eos: false,
            min_tokens: None,
            reasoning_effort: None,
            priority: None,
            mock_first_chunk_tokens: 1,
        }
    }

    #[test]
    fn tokenize_simple_ascii_splits_on_4_chars() {
        let tokens = tokenize("hello world");
        assert!(!tokens.is_empty());
        assert_eq!(tokens.concat(), "hello world");
    }

    #[test]
    fn tokenize_empty_returns_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn workspace_corpus_path_is_discoverable() {
        let path = workspace_corpus_path();
        assert!(path.ends_with("src/aiperf/dataset/generator/assets/shakespeare.txt"));
        assert!(path.is_file());
    }

    #[test]
    fn reasoning_model_produces_reasoning_tokens() {
        let mut req = chat("openai/gpt-oss-120b", "Hello there! Think about this hard.");
        req.max_tokens = Some(1000);
        let req_gen = GenRequest::Chat(&req);
        let out = tokenize_request(&req_gen);
        assert_eq!(out.reasoning_tokens, 250);
    }

    #[test]
    fn reasoning_high_effort() {
        let mut req = chat("qwen", "short");
        req.reasoning_effort = Some(ReasoningEffort::High);
        req.max_tokens = Some(1000);
        let req_gen = GenRequest::Chat(&req);
        let out = tokenize_request(&req_gen);
        assert_eq!(out.reasoning_tokens, 500);
    }

    #[test]
    fn reasoning_clamps_to_budget() {
        let req = chat("qwen", "short");
        let req_gen = GenRequest::Chat(&req);
        let out = tokenize_request(&req_gen);
        assert_eq!(out.reasoning_tokens, 16);
    }

    #[test]
    fn non_reasoning_model_no_reasoning_tokens() {
        let req = chat("gpt-4", "hello");
        let req_gen = GenRequest::Chat(&req);
        let out = tokenize_request(&req_gen);
        assert_eq!(out.reasoning_tokens, 0);
    }

    #[test]
    fn ignore_eos_fills_max_tokens() {
        let mut req = chat("gpt-4", "hello world this is content");
        req.max_tokens = Some(100);
        req.ignore_eos = true;
        let req_gen = GenRequest::Chat(&req);
        let out = tokenize_request(&req_gen);
        assert_eq!(out.tokens.len(), 100);
        assert_eq!(out.finish_reason, "length");
    }

    #[test]
    fn deterministic_for_same_input() {
        let req = chat("gpt-4", "same input");
        let req_gen = GenRequest::Chat(&req);
        let a = tokenize_request(&req_gen);
        let b = tokenize_request(&req_gen);
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.count(), b.count());
    }

    #[test]
    fn empty_messages_still_yields_output_tokens() {
        // A request with no extractable text (e.g. NativeGraph start node with
        // empty messages or an empty-string user message) must still return
        // output tokens so the transport classifies the reply as Completed
        // rather than Failed. The explicit 64-token cap below bounds the reply.
        let mut req = chat("gpt-4", "");
        req.max_tokens = Some(64);
        let req_gen = GenRequest::Chat(&req);
        let out = tokenize_request(&req_gen);
        assert!(
            !out.tokens.is_empty(),
            "empty prompt must still generate output tokens"
        );
        assert!(out.tokens.len() <= 64, "must respect max_tokens");
        assert_eq!(out.prompt_token_count, 0);
    }

    #[test]
    fn embedding_has_no_output_tokens() {
        let req = EmbeddingRequest {
            model: "emb".into(),
            input: crate::models::StringOrList::String("hello world".into()),
        };
        let req_gen = GenRequest::Embedding(&req);
        let out = tokenize_request(&req_gen);
        assert!(out.tokens.is_empty());
        assert!(out.prompt_token_count > 0);
    }
}
