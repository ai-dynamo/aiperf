// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral SSE message model and parser.
//!
//! These are the SSE *data* types (the parsed message and its fields), not the
//! streaming reader. They live in `transport::core` so [`super::response::Response`]
//! can hold an [`SseMessage`] without creating a `core → http` dependency; the SSE
//! *reader* that produces them stays in `transport::http::sse`.

use smallvec::SmallVec;

/// Inline capacity for an SSE message's fields. Almost every frame carries a
/// single `data:` field; `event`/`id`/comment stay inline too, so the common
/// per-token frame parses without a heap allocation for the field list.
type SseFields = SmallVec<[SseField; 4]>;

/// A named SSE field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseFieldName {
    Data,
    Event,
    Id,
    Retry,
    Comment,
    Other(String),
}

impl SseFieldName {
    fn from_str(s: &str) -> Self {
        match s {
            "data" => Self::Data,
            "event" => Self::Event,
            "id" => Self::Id,
            "retry" => Self::Retry,
            "" => Self::Comment,
            other => Self::Other(other.to_string()),
        }
    }
}

/// A single field within an SSE message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseField {
    pub name: SseFieldName,
    pub value: Option<String>,
}

/// One SSE message (delimited by a blank line), timestamped at arrival.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseMessage {
    /// Clock-nanoseconds when this message arrived.
    pub perf_ns: i64,
    /// Parsed fields, in order.
    pub packets: SseFields,
}

impl SseMessage {
    /// Parse a raw (already delimiter-stripped) SSE message.
    ///
    /// Splits each line at the first colon. An empty name denotes a comment and
    /// a colon-less line is name-only. A continuation after an unterminated
    /// `data: {` value is appended with a literal `\n`.
    pub fn parse(raw: &str, perf_ns: i64) -> Self {
        let mut packets: SseFields = SmallVec::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(last) = packets.last_mut()
                && let Some(prev) = &last.value
                && prev.starts_with('{')
                && !prev.ends_with('}')
                && !line.starts_with("data:")
            {
                last.value = Some(format!("{prev}\\n{line}"));
                continue;
            }

            match line.split_once(':') {
                None => packets.push(SseField {
                    name: SseFieldName::from_str(line.trim()),
                    value: None,
                }),
                Some((name, value)) => {
                    let name = SseFieldName::from_str(name.trim());
                    packets.push(SseField {
                        name,
                        value: Some(value.trim().to_string()),
                    });
                }
            }
        }
        Self { perf_ns, packets }
    }

    /// True if any `data:` field is the `[DONE]` sentinel.
    pub fn is_done(&self) -> bool {
        self.packets
            .iter()
            .any(|p| p.name == SseFieldName::Data && p.value.as_deref() == Some("[DONE]"))
    }

    /// Return an `event: error` comment or a fallback message.
    pub fn error_message(&self) -> Option<String> {
        let is_error = self
            .packets
            .iter()
            .any(|p| p.name == SseFieldName::Event && p.value.as_deref() == Some("error"));
        if !is_error {
            return None;
        }
        let comment = self
            .packets
            .iter()
            .find(|p| p.name == SseFieldName::Comment)
            .and_then(|p| p.value.clone());
        Some(comment.unwrap_or_else(|| "Unknown error in SSE response".to_string()))
    }

    /// The first `data:` payload (used to extract JSON chunks).
    pub fn data(&self) -> Option<&str> {
        self.packets
            .iter()
            .find(|p| p.name == SseFieldName::Data)
            .and_then(|p| p.value.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_data_field() {
        let m = SseMessage::parse("data: {\"a\":1}", 100);
        assert_eq!(m.perf_ns, 100);
        assert_eq!(m.packets.len(), 1);
        assert_eq!(m.packets[0].name, SseFieldName::Data);
        assert_eq!(m.packets[0].value.as_deref(), Some("{\"a\":1}"));
    }

    #[test]
    fn field_without_colon_has_no_value() {
        let m = SseMessage::parse("data", 1);
        assert_eq!(m.packets[0].name, SseFieldName::Data);
        assert_eq!(m.packets[0].value, None);
    }

    #[test]
    fn empty_field_name_is_comment() {
        let m = SseMessage::parse(": this is a comment", 1);
        assert_eq!(m.packets[0].name, SseFieldName::Comment);
        assert_eq!(m.packets[0].value.as_deref(), Some("this is a comment"));
    }

    #[test]
    fn json_continuation_line_is_appended_with_escaped_newline() {
        let m = SseMessage::parse("data: {\"x\":\n1}", 1);
        assert_eq!(m.packets.len(), 1);
        assert_eq!(m.packets[0].value.as_deref(), Some("{\"x\":\\n1}"));
    }

    #[test]
    fn done_sentinel_detected() {
        let m = SseMessage::parse("data: [DONE]", 1);
        assert!(m.is_done());
    }

    #[test]
    fn error_event_message_extracted_from_comment() {
        let raw = "event: error\n: boom";
        let m = SseMessage::parse(raw, 1);
        assert_eq!(m.error_message().as_deref(), Some("boom"));
    }
}
