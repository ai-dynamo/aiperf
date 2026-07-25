// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Structural layout inference for arbitrary Hugging Face dataset rows.
//!
//! Public datasets on the Hub do not share a schema: some carry a chat-message
//! array, some a flat prompt/answer pair, some split a task into a context field
//! plus a question field. When a run points at a dataset by id without naming its
//! columns, [`infer_row_layout`] inspects one representative row and decides which
//! field supplies the prompt, whether that field is a message array, and which
//! field (if any) supplies the reference completion. The logic is intentionally
//! pure `serde_json` — it holds no loader or compose state — so every branch is
//! covered by unit tests below.

use serde_json::Value;

/// Fields that, when present, hold a chat-message array. Checked before the flat
/// prompt fields so a conversational dataset is never mistaken for plain text.
const MESSAGE_FIELDS: &[&str] = &["conversation", "conversations", "messages"];
/// Flat prompt fields, most specific first. A row carrying several of these is
/// read from the earliest match.
const PROMPT_FIELDS: &[&str] = &[
    "prompt",
    "question",
    "problem",
    "input",
    "text",
    "content",
    "instruction",
];
/// Fields that supply the reference completion used to size the output length.
const COMPLETION_FIELDS: &[&str] = &[
    "completion",
    "response",
    "answer",
    "output",
    "solution",
    "answers",
];

/// How a dataset row exposes its prompt (and optional completion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RowLayout {
    /// A chat-message array field (`{role,content}` or `{from,value}` entries).
    Messages(String),
    /// A single prompt field, optionally paired with a completion field.
    Prompt {
        /// Field holding the prompt text.
        prompt_field: String,
        /// Field holding the reference completion, when one was found.
        completion_field: Option<String>,
    },
    /// Several fields concatenated (blank-line separated) into one prompt.
    Joined {
        /// Fields concatenated, in order, to form the prompt.
        fields: Vec<String>,
        /// Field holding the reference completion, when one was found.
        completion_field: Option<String>,
    },
}

/// Whether a JSON value is a single chat message (either role/content or the
/// ShareGPT-style from/value shape).
fn is_message_object(value: &Value) -> bool {
    value.as_object().is_some_and(|entry| {
        (entry.contains_key("role") && entry.contains_key("content"))
            || (entry.contains_key("from") && entry.contains_key("value"))
    })
}

/// Whether a JSON value is a non-empty array of chat messages.
fn is_message_list(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|items| !items.is_empty() && items.iter().all(is_message_object))
}

/// Comma-joined field names of `entry`, for "available fields: …" diagnostics.
fn available_fields(entry: &serde_json::Map<String, Value>) -> String {
    entry.keys().cloned().collect::<Vec<_>>().join(", ")
}

/// Infer the prompt/completion layout of a dataset from one sample row.
///
/// When `prompt_override` names a field, that field is used as the prompt (still
/// recognising a message array there); otherwise message fields win over the
/// flat prompt fields, with a `context` + `input` pair recognised as a joined
/// prompt. Returns a caller-facing message (listing the row's fields) when no
/// prompt field can be found.
pub(crate) fn infer_row_layout(
    row: &Value,
    prompt_override: Option<&str>,
) -> Result<RowLayout, String> {
    let entry = row
        .as_object()
        .ok_or_else(|| "Hugging Face row is not a JSON object".to_string())?;

    let completion_field = || -> Option<String> {
        COMPLETION_FIELDS
            .iter()
            .find(|field| entry.contains_key(**field))
            .map(|field| (*field).to_string())
    };

    if let Some(field) = prompt_override {
        let value = entry.get(field).ok_or_else(|| {
            format!(
                "prompt field {field:?} is missing; available fields: {}",
                available_fields(entry)
            )
        })?;
        if is_message_list(value) {
            return Ok(RowLayout::Messages(field.to_string()));
        }
        return Ok(RowLayout::Prompt {
            prompt_field: field.to_string(),
            completion_field: completion_field(),
        });
    }

    for field in MESSAGE_FIELDS {
        if entry.get(*field).is_some_and(is_message_list) {
            return Ok(RowLayout::Messages((*field).to_string()));
        }
    }

    // A `turns` array of strings is a multi-turn prompt; the first turn seeds the
    // single-turn request the composer builds.
    if entry
        .get("turns")
        .and_then(Value::as_array)
        .is_some_and(|items| items.first().is_some_and(Value::is_string))
    {
        return Ok(RowLayout::Prompt {
            prompt_field: "turns".to_string(),
            completion_field: completion_field(),
        });
    }

    if entry.contains_key("context") && entry.contains_key("input") {
        return Ok(RowLayout::Joined {
            fields: vec!["context".to_string(), "input".to_string()],
            completion_field: completion_field(),
        });
    }

    for field in PROMPT_FIELDS {
        if entry.contains_key(*field) {
            return Ok(RowLayout::Prompt {
                prompt_field: (*field).to_string(),
                completion_field: completion_field(),
            });
        }
    }

    Err(format!(
        "could not infer a prompt field; available fields: {}. Pass --hf-text-column to name it.",
        available_fields(entry)
    ))
}

/// The role label of a message, under either the role/content or from/value shape.
fn message_role(message: &Value) -> Option<&str> {
    message
        .get("role")
        .and_then(Value::as_str)
        .or_else(|| message.get("from").and_then(Value::as_str))
}

/// The text of a message, under either the role/content or from/value shape.
fn message_text(message: &Value) -> Option<&str> {
    message
        .get("content")
        .and_then(Value::as_str)
        .or_else(|| message.get("value").and_then(Value::as_str))
}

/// Text of the first requester turn (`user`/`human`) in a message array.
pub(crate) fn first_user_message(messages: &[Value]) -> Option<String> {
    messages.iter().find_map(
        |message| match (message_role(message), message_text(message)) {
            (Some(role), Some(text)) if role == "user" || role == "human" => Some(text.to_string()),
            _ => None,
        },
    )
}

/// Text of the first responder turn (`assistant`/`gpt`) in a message array.
pub(crate) fn first_assistant_message(messages: &[Value]) -> Option<String> {
    messages.iter().find_map(
        |message| match (message_role(message), message_text(message)) {
            (Some(role), Some(text)) if role == "assistant" || role == "gpt" => {
                Some(text.to_string())
            }
            _ => None,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn message_array_field_wins_over_flat_fields() {
        let row = json!({
            "id": 7,
            "conversation": [
                {"role": "user", "content": "Which crate owns the clock seam?"},
                {"role": "assistant", "content": "aiperf_runtime::clock"}
            ]
        });
        assert!(matches!(infer_row_layout(&row, None).unwrap(),
            RowLayout::Messages(field) if field == "conversation"));
    }

    #[test]
    fn sharegpt_from_value_array_is_messages() {
        let row = json!({"conversations": [
            {"from": "human", "value": "ping"}, {"from": "gpt", "value": "pong"}]});
        assert!(matches!(infer_row_layout(&row, None).unwrap(),
            RowLayout::Messages(field) if field == "conversations"));
    }

    #[test]
    fn flat_prompt_and_completion_pair() {
        let row = json!({"prompt": "Summarise the dispatch seam.", "completion": "It is transport-neutral."});
        match infer_row_layout(&row, None).unwrap() {
            RowLayout::Prompt {
                prompt_field,
                completion_field,
            } => {
                assert_eq!(prompt_field, "prompt");
                assert_eq!(completion_field.as_deref(), Some("completion"));
            }
            other => panic!("expected Prompt, got {other:?}"),
        }
    }

    #[test]
    fn question_field_maps_to_prompt_with_answer_completion() {
        let row = json!({"question": "What does SimClock provide?", "answer": "Virtual nanosecond time."});
        match infer_row_layout(&row, None).unwrap() {
            RowLayout::Prompt {
                prompt_field,
                completion_field,
            } => {
                assert_eq!(prompt_field, "question");
                assert_eq!(completion_field.as_deref(), Some("answer"));
            }
            other => panic!("expected Prompt, got {other:?}"),
        }
    }

    #[test]
    fn context_plus_input_joins_into_one_prompt() {
        let row = json!({
            "context": "The worker sink drives one request to terminal.",
            "input": "What does the worker sink do?",
            "answers": ["drive a request to terminal"]
        });
        match infer_row_layout(&row, None).unwrap() {
            RowLayout::Joined {
                fields,
                completion_field,
            } => {
                assert_eq!(fields, vec!["context".to_string(), "input".to_string()]);
                assert_eq!(completion_field.as_deref(), Some("answers"));
            }
            other => panic!("expected Joined, got {other:?}"),
        }
    }

    #[test]
    fn override_forces_a_named_field() {
        let row = json!({"body": "custom prompt body", "answer": "reply"});
        match infer_row_layout(&row, Some("body")).unwrap() {
            RowLayout::Prompt {
                prompt_field,
                completion_field,
            } => {
                assert_eq!(prompt_field, "body");
                assert_eq!(completion_field.as_deref(), Some("answer"));
            }
            other => panic!("expected Prompt, got {other:?}"),
        }
    }

    #[test]
    fn override_on_message_field_is_messages() {
        let row = json!({"dialog": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]});
        assert!(matches!(infer_row_layout(&row, Some("dialog")).unwrap(),
            RowLayout::Messages(field) if field == "dialog"));
    }

    #[test]
    fn override_naming_absent_field_errors() {
        let row = json!({"text": "hello"});
        assert!(infer_row_layout(&row, Some("absent")).is_err());
    }

    #[test]
    fn unrecognised_row_lists_its_fields() {
        let row = json!({"identifier": 1, "sentiment": "positive"});
        let message = infer_row_layout(&row, None).unwrap_err();
        assert!(message.contains("identifier") && message.contains("sentiment"));
    }

    #[test]
    fn picks_first_requester_and_responder_turns() {
        let messages = vec![
            json!({"role": "system", "content": "stay terse"}),
            json!({"role": "user", "content": "define TTFT"}),
            json!({"role": "assistant", "content": "first token observation"}),
        ];
        assert_eq!(
            first_user_message(&messages),
            Some("define TTFT".to_string())
        );
        assert_eq!(
            first_assistant_message(&messages),
            Some("first token observation".to_string())
        );
    }

    #[test]
    fn requester_and_responder_under_from_value() {
        let messages = vec![
            json!({"from": "human", "value": "ping"}),
            json!({"from": "gpt", "value": "pong"}),
        ];
        assert_eq!(first_user_message(&messages), Some("ping".to_string()));
        assert_eq!(first_assistant_message(&messages), Some("pong".to_string()));
    }

    #[test]
    fn unknown_roles_match_neither_side() {
        let messages = vec![json!({"role": "model", "content": "n/a"})];
        assert_eq!(first_user_message(&messages), None);
        assert_eq!(first_assistant_message(&messages), None);
    }
}
