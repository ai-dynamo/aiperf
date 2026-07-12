// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider usage-field normalization.
//!
//! Anthropic and Bedrock use disjoint input accounting: their ordinary input
//! count excludes cache reads and writes. The normalization follows
//! `src/aiperf/common/models/usage_models.py:56-401` from PR 731 so
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
            let Some(value) = self.usage.get(key) else {
                continue;
            };
            let value = value.as_u64()?;
            if matches!(key, "input_tokens" | "inputTokens")
                && DISJOINT_CACHE_KEYS
                    .iter()
                    .any(|cache_key| self.usage.contains_key(*cache_key))
            {
                return Some(DISJOINT_CACHE_KEYS.iter().fold(value, |total, cache_key| {
                    total.saturating_add(
                        self.usage
                            .get(*cache_key)
                            .and_then(Value::as_u64)
                            .unwrap_or(0),
                    )
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
            if !self.usage.contains_key(key) {
                continue;
            }
            if matches!(key, "input_tokens" | "inputTokens")
                && DISJOINT_CACHE_KEYS
                    .iter()
                    .any(|cache_key| self.usage.contains_key(*cache_key))
            {
                return self.usage.get(key).and_then(Value::as_u64);
            }
            return None;
        }
        None
    }

    /// Completion/output token count.
    pub fn completion_tokens(self) -> Option<u64> {
        first_u64(
            self.usage,
            &[
                "completion_tokens",
                "output_tokens",
                "candidatesTokenCount",
                "outputTokens",
                "generated_token_count",
            ],
        )
    }

    /// Provider-reported total token count, when explicitly present.
    pub fn total_tokens(self) -> Option<u64> {
        first_u64(
            self.usage,
            &["total_tokens", "totalTokenCount", "totalTokens"],
        )
    }

    /// Cached prompt tokens read from a prior prefix.
    pub fn prompt_cache_read_tokens(self) -> Option<u64> {
        first_nested_u64(
            self.usage,
            &["prompt_tokens_details", "input_tokens_details"],
            "cached_tokens",
        )
        .or_else(|| {
            first_u64(
                self.usage,
                &[
                    "cache_read_input_tokens",
                    "prompt_cache_hit_tokens",
                    "cachedContentTokenCount",
                    "cacheReadInputTokens",
                    "cached_tokens",
                ],
            )
        })
    }

    /// Prompt tokens written into a provider cache.
    pub fn prompt_cache_write_tokens(self) -> Option<u64> {
        first_u64(
            self.usage,
            &["cache_creation_input_tokens", "cacheWriteInputTokens"],
        )
    }

    /// Explicit prompt cache-miss count.
    pub fn prompt_cache_miss_tokens(self) -> Option<u64> {
        first_u64(self.usage, &["prompt_cache_miss_tokens"])
    }

    /// Provider-reported reasoning token count.
    pub fn reasoning_tokens(self) -> Option<u64> {
        first_nested_u64(
            self.usage,
            &["completion_tokens_details", "output_tokens_details"],
            "reasoning_tokens",
        )
        .or_else(|| first_u64(self.usage, &["thoughtsTokenCount"]))
    }
}

const DISJOINT_CACHE_KEYS: &[&str] = &[
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
];

fn first_u64(usage: &Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Value::as_u64))
}

fn first_nested_u64(usage: &Map<String, Value>, detail_keys: &[&str], field: &str) -> Option<u64> {
    detail_keys.iter().find_map(|key| {
        usage
            .get(*key)
            .and_then(Value::as_object)
            .and_then(|details| details.get(field))
            .and_then(Value::as_u64)
    })
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
}
