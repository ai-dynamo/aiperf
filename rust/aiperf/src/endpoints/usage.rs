// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider usage-field normalization.
//!
//! Anthropic and Bedrock use disjoint input accounting: their ordinary input
//! count excludes cache reads and writes. The normalization ensures
//! `prompt_tokens` retains one meaning across endpoint dialects.

use serde_json::{Map, Value};

/// Borrowed view over one provider usage object.
#[derive(Debug, Clone, Copy)]
pub struct UsageView<'a> {
    usage: &'a Map<String, Value>,
}

impl<'a> UsageView<'a> {
    /// Wrap a JSON usage object, returning `None` for all other shapes.
    pub fn from_value(value: &'a Value) -> Option<Self> {
        value.as_object().map(|usage| Self { usage })
    }

    /// Total prompt tokens, re-totalizing disjoint Anthropic/Bedrock cache fields.
    pub fn prompt_tokens(self) -> Option<u64> {
        for key in [
            "prompt_tokens",
            "input_tokens",
            "promptTokenCount",
            "inputTokens",
            "input_token_count",
        ] {
            let Some(value) = self.value(key) else {
                continue;
            };
            let value = value.as_u64()?;
            if matches!(key, "input_tokens" | "inputTokens")
                && DISJOINT_CACHE_KEYS
                    .iter()
                    .any(|cache_key| self.value(cache_key).is_some())
            {
                return Some(DISJOINT_CACHE_KEYS.iter().fold(value, |total, cache_key| {
                    total.saturating_add(self.value(cache_key).and_then(Value::as_u64).unwrap_or(0))
                }));
            }
            return Some(value);
        }
        None
    }

    /// Raw uncached input remainder for disjoint-accounting providers.
    pub fn prompt_uncached_tokens(self) -> Option<u64> {
        for key in [
            "prompt_tokens",
            "input_tokens",
            "promptTokenCount",
            "inputTokens",
            "input_token_count",
        ] {
            if self.value(key).is_none() {
                continue;
            }
            if matches!(key, "input_tokens" | "inputTokens")
                && DISJOINT_CACHE_KEYS
                    .iter()
                    .any(|cache_key| self.value(cache_key).is_some())
            {
                return self.value(key).and_then(Value::as_u64);
            }
            return None;
        }
        None
    }

    /// Completion/output token count.
    pub fn completion_tokens(self) -> Option<u64> {
        self.first_u64(&[
            "completion_tokens",
            "output_tokens",
            "candidatesTokenCount",
            "outputTokens",
            "generated_token_count",
        ])
    }

    /// Provider-reported total token count, when explicitly present.
    pub fn total_tokens(self) -> Option<u64> {
        self.first_u64(&["total_tokens", "totalTokenCount", "totalTokens"])
    }

    /// Cached prompt tokens read from a prior prefix.
    pub fn prompt_cache_read_tokens(self) -> Option<u64> {
        self.first_nested_u64(
            &["prompt_tokens_details", "input_tokens_details"],
            "cached_tokens",
        )
        .or_else(|| {
            self.first_u64(&[
                "cache_read_input_tokens",
                "prompt_cache_hit_tokens",
                "cachedContentTokenCount",
                "cacheReadInputTokens",
                "cached_tokens",
            ])
        })
    }

    /// Prompt tokens written into a provider cache.
    pub fn prompt_cache_write_tokens(self) -> Option<u64> {
        self.first_u64(&["cache_creation_input_tokens", "cacheWriteInputTokens"])
    }

    /// Explicit prompt cache-miss count.
    pub fn prompt_cache_miss_tokens(self) -> Option<u64> {
        self.first_u64(&["prompt_cache_miss_tokens"])
    }

    /// Provider-reported reasoning token count.
    pub fn reasoning_tokens(self) -> Option<u64> {
        self.first_nested_u64(
            &["completion_tokens_details", "output_tokens_details"],
            "reasoning_tokens",
        )
        .or_else(|| self.first_u64(&["thoughtsTokenCount"]))
    }

    /// Audio-token count attributed to the prompt.
    pub fn prompt_audio_tokens(self) -> Option<u64> {
        self.first_nested_u64(
            &["prompt_tokens_details", "input_tokens_details"],
            "audio_tokens",
        )
    }

    /// Audio-token count attributed to model output.
    pub fn completion_audio_tokens(self) -> Option<u64> {
        self.first_nested_u64(
            &["completion_tokens_details", "output_tokens_details"],
            "audio_tokens",
        )
    }

    /// Accepted predicted-output tokens reported by OpenAI-compatible APIs.
    pub fn accepted_prediction_tokens(self) -> Option<u64> {
        self.first_nested_u64(
            &["completion_tokens_details", "output_tokens_details"],
            "accepted_prediction_tokens",
        )
    }

    /// Rejected predicted-output tokens reported by OpenAI-compatible APIs.
    pub fn rejected_prediction_tokens(self) -> Option<u64> {
        self.first_nested_u64(
            &["completion_tokens_details", "output_tokens_details"],
            "rejected_prediction_tokens",
        )
    }

    /// Gemini tool-definition tokens reported separately from prompt content.
    pub fn tool_use_prompt_tokens(self) -> Option<u64> {
        self.first_u64(&["toolUsePromptTokenCount"])
    }

    /// Mistral prompt-audio duration in seconds, distinct from audio tokens.
    pub fn prompt_audio_seconds(self) -> Option<f64> {
        self.first_value(&["prompt_audio_seconds"])
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
    }

    fn first_u64(self, keys: &[&str]) -> Option<u64> {
        keys.iter()
            .find_map(|key| self.value(key).map(Value::as_u64))
            .flatten()
    }

    fn first_value(self, keys: &[&str]) -> Option<&'a Value> {
        keys.iter().find_map(|key| self.value(key))
    }

    fn first_nested_u64(self, detail_keys: &[&str], field: &str) -> Option<u64> {
        detail_keys
            .iter()
            .find_map(|key| {
                self.value(key)
                    .and_then(Value::as_object)
                    .and_then(|details| details.get(field).map(Value::as_u64))
            })
            .flatten()
    }

    fn value(self, key: &str) -> Option<&'a Value> {
        self.usage
            .get(key)
            .or_else(|| nested_object(self.usage, "usageMetadata")?.get(key))
            .or_else(|| {
                let meta = nested_object(self.usage, "meta")?;
                nested_object(meta, "tokens")
                    .and_then(|tokens| tokens.get(key))
                    .or_else(|| (key == "cached_tokens").then(|| meta.get(key)).flatten())
            })
            .or_else(|| nested_object(self.usage, "tokens")?.get(key))
    }
}

const DISJOINT_CACHE_KEYS: &[&str] = &[
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
];

fn nested_object<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a Map<String, Value>> {
    object.get(key).and_then(Value::as_object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_cache_accounting_is_disjoint() {
        let value = json!({
            "input_tokens": 25,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 3,
            "output_tokens": 15
        });
        let usage = UsageView::from_value(&value).unwrap();
        assert_eq!(usage.prompt_tokens(), Some(35));
        assert_eq!(usage.prompt_uncached_tokens(), Some(25));
        assert_eq!(usage.prompt_cache_read_tokens(), Some(7));
        assert_eq!(usage.prompt_cache_write_tokens(), Some(3));
        assert_eq!(usage.completion_tokens(), Some(15));
    }

    #[test]
    fn subset_cache_accounting_is_not_double_counted() {
        let value = json!({
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 40}
        });
        let usage = UsageView::from_value(&value).unwrap();
        assert_eq!(usage.prompt_tokens(), Some(100));
        assert_eq!(usage.prompt_uncached_tokens(), None);
        assert_eq!(usage.prompt_cache_read_tokens(), Some(40));
    }

    #[test]
    fn synonym_lookup_stops_at_the_first_present_key() {
        let value = json!({
            "completion_tokens": null,
            "output_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": null},
            "input_tokens_details": {"cached_tokens": 7}
        });
        let usage = UsageView::from_value(&value).unwrap();
        assert_eq!(usage.completion_tokens(), None);
        assert_eq!(usage.prompt_cache_read_tokens(), None);
    }

    #[test]
    fn extended_usage_fields_preserve_zero_and_detail_precedence() {
        let value = json!({
            "prompt_tokens_details": {"audio_tokens": 0},
            "input_tokens_details": {"audio_tokens": 99},
            "completion_tokens_details": {
                "audio_tokens": 20,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 5
            },
            "toolUsePromptTokenCount": 30,
            "prompt_audio_seconds": 12
        });
        let usage = UsageView::from_value(&value).unwrap();
        assert_eq!(usage.prompt_audio_tokens(), Some(0));
        assert_eq!(usage.completion_audio_tokens(), Some(20));
        assert_eq!(usage.accepted_prediction_tokens(), Some(0));
        assert_eq!(usage.rejected_prediction_tokens(), Some(5));
        assert_eq!(usage.tool_use_prompt_tokens(), Some(30));
        assert_eq!(usage.prompt_audio_seconds(), Some(12.0));
    }

    #[test]
    fn gemini_and_cohere_envelopes_are_read_without_materializing_a_copy() {
        let value = json!({
            "usageMetadata": {
                "promptTokenCount": 10,
                "toolUsePromptTokenCount": 30
            },
            "meta": {
                "tokens": {"output_tokens": 52},
                "cached_tokens": 25
            }
        });
        let usage = UsageView::from_value(&value).unwrap();
        assert_eq!(usage.prompt_tokens(), Some(10));
        assert_eq!(usage.completion_tokens(), Some(52));
        assert_eq!(usage.prompt_cache_read_tokens(), Some(25));
        assert_eq!(usage.tool_use_prompt_tokens(), Some(30));

        let top_level_wins = json!({
            "promptTokenCount": 999,
            "usageMetadata": {"promptTokenCount": 10}
        });
        assert_eq!(
            UsageView::from_value(&top_level_wins)
                .unwrap()
                .prompt_tokens(),
            Some(999)
        );
    }

    #[test]
    fn prompt_audio_seconds_rejects_non_numeric_sentinels() {
        for value in [json!({}), json!([]), json!("12.5"), Value::Null] {
            let usage_value = json!({"prompt_audio_seconds": value});
            let usage = UsageView::from_value(&usage_value).unwrap();
            assert_eq!(usage.prompt_audio_seconds(), None);
        }
    }
}
