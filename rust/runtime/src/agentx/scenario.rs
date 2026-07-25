// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scenario-lock runtime helpers (Slice 4), ported from
//! `src/aiperf/common/scenario/`.
//!
//! So far: the context-overflow classifier (`context_overflow.py`). The
//! substring allowlist is passed in (Python reads it from
//! `Environment.AGENTX.CONTEXT_OVERFLOW_SUBSTRINGS`).

/// Classify whether an error response indicates a context-overflow (Python
/// `is_context_overflow_response`).
///
/// Case-insensitive substring match against (1) the raw body text and (2) the
/// OpenAI-style nested `error.message` field when the body parses as JSON.
/// Callers pre-filter to error responses and pre-decode bytes bodies. `None` /
/// empty body or empty allowlist → false.
pub fn is_context_overflow_response(body: Option<&str>, substrings: &[String]) -> bool {
    let text = match body {
        Some(t) if !t.is_empty() => t,
        _ => return false,
    };
    let needles: Vec<String> = substrings
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    if needles.is_empty() {
        return false;
    }

    let lowered = text.to_lowercase();
    if needles.iter().any(|n| lowered.contains(n)) {
        return true;
    }

    if let Some(msg) = extract_openai_error_message(text) {
        let nested = msg.to_lowercase();
        if needles.iter().any(|n| nested.contains(n)) {
            return true;
        }
    }
    false
}

/// Return the OpenAI-style `error.message` from a JSON body (Python
/// `_extract_openai_error_message`). Tolerates a string-shaped `error` field.
fn extract_openai_error_message(text: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(text).ok()?;
    let obj = parsed.as_object()?;
    match obj.get("error") {
        Some(serde_json::Value::Object(err)) => {
            err.get("message").and_then(|m| m.as_str()).map(String::from)
        }
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subs() -> Vec<String> {
        vec!["context length".into(), "maximum context".into()]
    }

    #[test]
    fn matches_raw_body_case_insensitive() {
        assert!(is_context_overflow_response(
            Some("Error: Maximum Context exceeded"),
            &subs()
        ));
    }

    #[test]
    fn matches_openai_error_message() {
        let body = r#"{"error": {"message": "This model's maximum context length is 8192 tokens"}}"#;
        assert!(is_context_overflow_response(Some(body), &subs()));
    }

    #[test]
    fn string_error_field_is_used() {
        let body = r#"{"error": "maximum context reached"}"#;
        assert!(is_context_overflow_response(Some(body), &subs()));
    }

    #[test]
    fn no_match_and_empty_cases() {
        assert!(!is_context_overflow_response(Some("rate limit exceeded"), &subs()));
        assert!(!is_context_overflow_response(None, &subs()));
        assert!(!is_context_overflow_response(Some(""), &subs()));
        assert!(!is_context_overflow_response(Some("context length"), &[]));
    }
}
