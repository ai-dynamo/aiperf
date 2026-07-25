// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Structural column/format detection for arbitrary Hugging Face dataset rows.
//!
//! A port of vLLM's `detect_column_format`: given the first row of an unknown
//! dataset, decide whether prompts live in a chat-message column, a combined
//! context+input pair, or a single text column, and which column (if any) holds
//! the reference completion. This module is pure `serde_json` logic with no
//! dependency on the loader/compose plane so it can be exhaustively unit-tested.

use serde_json::Value;

/// Chat-array column names, in priority order.
const CHAT_COLUMNS: &[&str] = &["conversation", "conversations", "messages"];
/// Single-text prompt column names, in priority order.
const TEXT_COLUMNS: &[&str] =
    &["prompt", "question", "problem", "input", "text", "content", "instruction"];
/// Reference-output column names, in priority order.
const OUTPUT_COLUMNS: &[&str] =
    &["completion", "response", "answer", "output", "solution", "answers"];

/// Detected shape of an HF dataset row's prompt/output columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ColumnFormat {
    /// A chat-message array column (`{role,content}` or `{from,value}`).
    Chat(String),
    /// A single prompt column plus an optional output column.
    Text {
        /// Column holding the prompt text.
        prompt_col: String,
        /// Column holding the reference completion, when detected.
        output_col: Option<String>,
    },
    /// Multiple text columns joined with `\n\n` plus an optional output column.
    Combined {
        /// Columns joined to form the prompt.
        cols: Vec<String>,
        /// Column holding the reference completion, when detected.
        output_col: Option<String>,
    },
}

fn is_chat_message(val: &Value) -> bool {
    val.as_object().is_some_and(|obj| {
        (obj.contains_key("role") && obj.contains_key("content"))
            || (obj.contains_key("from") && obj.contains_key("value"))
    })
}

fn is_chat_array(val: &Value) -> bool {
    val.as_array().is_some_and(|arr| !arr.is_empty() && arr.iter().all(is_chat_message))
}

/// Detect the prompt/output column layout from one row. `text_column_override`
/// forces the prompt column (still auto-detecting chat vs text for it).
pub(crate) fn detect_column_format(
    row: &Value,
    text_column_override: Option<&str>,
) -> Result<ColumnFormat, String> {
    let obj = row.as_object().ok_or_else(|| "HF dataset row is not a JSON object".to_string())?;

    let find_output_col = || -> Option<String> {
        OUTPUT_COLUMNS.iter().find(|c| obj.contains_key(**c)).map(|c| (*c).to_string())
    };

    if let Some(name) = text_column_override {
        let val = obj.get(name).ok_or_else(|| {
            format!(
                "column {name:?} not found; available columns: {}",
                obj.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        if is_chat_array(val) {
            return Ok(ColumnFormat::Chat(name.to_string()));
        }
        return Ok(ColumnFormat::Text {
            prompt_col: name.to_string(),
            output_col: find_output_col(),
        });
    }

    for col in CHAT_COLUMNS {
        if obj.get(*col).is_some_and(is_chat_array) {
            return Ok(ColumnFormat::Chat((*col).to_string()));
        }
    }

    if obj
        .get("turns")
        .and_then(Value::as_array)
        .is_some_and(|a| a.first().is_some_and(Value::is_string))
    {
        return Ok(ColumnFormat::Text {
            prompt_col: "turns".to_string(),
            output_col: find_output_col(),
        });
    }

    if obj.contains_key("context") && obj.contains_key("input") {
        return Ok(ColumnFormat::Combined {
            cols: vec!["context".to_string(), "input".to_string()],
            output_col: find_output_col(),
        });
    }

    for col in TEXT_COLUMNS {
        if obj.contains_key(*col) {
            return Ok(ColumnFormat::Text {
                prompt_col: (*col).to_string(),
                output_col: find_output_col(),
            });
        }
    }

    Err(format!(
        "could not auto-detect a prompt column; available columns: {}. Use --hf-text-column to pick one.",
        obj.keys().cloned().collect::<Vec<_>>().join(", ")
    ))
}

fn role_of(msg: &Value) -> Option<&str> {
    msg.get("role").and_then(Value::as_str).or_else(|| msg.get("from").and_then(Value::as_str))
}
fn content_of(msg: &Value) -> Option<&str> {
    msg.get("content").and_then(Value::as_str).or_else(|| msg.get("value").and_then(Value::as_str))
}

/// First `user`/`human` message content in a chat array.
pub(crate) fn extract_chat_prompt(messages: &[Value]) -> Option<String> {
    messages.iter().find_map(|m| match (role_of(m), content_of(m)) {
        (Some(r), Some(c)) if r == "user" || r == "human" => Some(c.to_string()),
        _ => None,
    })
}

/// First `assistant`/`gpt` message content in a chat array.
pub(crate) fn extract_chat_completion(messages: &[Value]) -> Option<String> {
    messages.iter().find_map(|m| match (role_of(m), content_of(m)) {
        (Some(r), Some(c)) if r == "assistant" || r == "gpt" => Some(c.to_string()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_chat_conversation_column() {
        let row = json!({"conversation": [
            {"role": "user", "content": "Hello there friend"},
            {"role": "assistant", "content": "Hi"}]});
        assert!(matches!(detect_column_format(&row, None).unwrap(),
            ColumnFormat::Chat(c) if c == "conversation"));
    }

    #[test]
    fn detects_sharegpt_from_value_chat() {
        let row = json!({"conversations": [
            {"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi"}]});
        assert!(matches!(detect_column_format(&row, None).unwrap(),
            ColumnFormat::Chat(c) if c == "conversations"));
    }

    #[test]
    fn detects_plain_prompt_completion() {
        let row = json!({"prompt": "What is 2+2?", "completion": "4"});
        match detect_column_format(&row, None).unwrap() {
            ColumnFormat::Text { prompt_col, output_col } => {
                assert_eq!(prompt_col, "prompt");
                assert_eq!(output_col.as_deref(), Some("completion"));
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn detects_question_answer() {
        let row = json!({"question": "What is AI?", "answer": "Artificial intelligence"});
        match detect_column_format(&row, None).unwrap() {
            ColumnFormat::Text { prompt_col, output_col } => {
                assert_eq!(prompt_col, "question");
                assert_eq!(output_col.as_deref(), Some("answer"));
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn detects_combined_context_input() {
        let row = json!({"context": "The fox", "input": "what animal?", "answers": ["fox"]});
        match detect_column_format(&row, None).unwrap() {
            ColumnFormat::Combined { cols, output_col } => {
                assert_eq!(cols, vec!["context".to_string(), "input".to_string()]);
                assert_eq!(output_col.as_deref(), Some("answers"));
            }
            _ => panic!("expected Combined"),
        }
    }

    #[test]
    fn override_selects_named_text_column() {
        let row = json!({"my_col": "Hello world", "answer": "resp"});
        match detect_column_format(&row, Some("my_col")).unwrap() {
            ColumnFormat::Text { prompt_col, output_col } => {
                assert_eq!(prompt_col, "my_col");
                assert_eq!(output_col.as_deref(), Some("answer"));
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn override_missing_column_errors() {
        let row = json!({"text": "hello"});
        assert!(detect_column_format(&row, Some("nope")).is_err());
    }

    #[test]
    fn no_known_columns_errors_listing_available() {
        let row = json!({"id": 1, "label": "positive"});
        let err = detect_column_format(&row, None).unwrap_err();
        assert!(err.contains("id") && err.contains("label"));
    }

    #[test]
    fn extracts_first_user_and_assistant() {
        let msgs = vec![
            json!({"role": "system", "content": "be nice"}),
            json!({"role": "user", "content": "joke?"}),
            json!({"role": "assistant", "content": "haha"}),
        ];
        assert_eq!(extract_chat_prompt(&msgs), Some("joke?".to_string()));
        assert_eq!(extract_chat_completion(&msgs), Some("haha".to_string()));
    }

    #[test]
    fn extracts_from_value_style() {
        let msgs = vec![json!({"from": "human", "value": "Hi"}), json!({"from": "gpt", "value": "Yo"})];
        assert_eq!(extract_chat_prompt(&msgs), Some("Hi".to_string()));
        assert_eq!(extract_chat_completion(&msgs), Some("Yo".to_string()));
    }

    #[test]
    fn model_role_is_not_user_or_assistant() {
        let msgs = vec![json!({"role": "model", "content": "x"})];
        assert_eq!(extract_chat_prompt(&msgs), None);
        assert_eq!(extract_chat_completion(&msgs), None);
    }
}
