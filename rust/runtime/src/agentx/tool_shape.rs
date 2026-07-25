// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in OpenAI tool-call wire shaping for Weka trace replay, ported from
//! `src/aiperf/dataset/loader/weka_tool_shape.py`.
//!
//! Shaping operates on the reconstructor's emitted segment window so it applies
//! uniformly to append deltas AND `reset_context` full re-emissions — a reset
//! replaces the wire context, so emission-time-only shaping would retroactively
//! unshape every previously-sent tool turn.

use crate::agentx::synth::{ChatMessage, Role, RoleSegment, ToolCall};

/// Clear `tool_result_turn` marks that cannot pair in this window.
///
/// A marked user segment without an assistant directly before it in the window
/// ships plain — and once sent plain it must STAY plain, so the mark is demoted
/// at first emission. Idempotent: a mark that pairs in its first window keeps
/// pairing in every later full window.
pub fn demote_unpaired_tool_marks(segments: &mut [RoleSegment]) {
    for i in 0..segments.len() {
        if segments[i].tool_result_turn.is_none() {
            continue;
        }
        if segments[i].role != Role::User {
            continue;
        }
        if i == 0 || segments[i - 1].role != Role::Assistant {
            segments[i].tool_result_turn = None;
        }
    }
}

/// Shape an emitted message window from its source segments.
///
/// `messages[i]` corresponds 1:1 to `segments[i]`. A user segment marked with
/// `tool_result_turn` becomes a `role: "tool"` message, and the assistant
/// segment immediately before it (in the same window) gains the matching
/// synthetic `tool_calls` entry. The call id is keyed to the recorded turn so
/// re-emissions reproduce the id the turn was first sent with. A marked segment
/// without a directly-preceding assistant falls back to the plain user shape.
pub fn tool_shape_segment_messages(
    mut messages: Vec<ChatMessage>,
    segments: &[RoleSegment],
) -> Vec<ChatMessage> {
    for i in 0..segments.len() {
        let turn = match segments[i].tool_result_turn {
            Some(t) => t,
            None => continue,
        };
        if segments[i].role != Role::User {
            continue;
        }
        if i == 0 || segments[i - 1].role != Role::Assistant {
            continue;
        }
        let call_id = format!("call_turn_{turn}");
        messages[i - 1].tool_calls = Some(vec![ToolCall {
            id: call_id.clone(),
            name: "recorded_tool".to_string(),
            arguments: "{}".to_string(),
        }]);
        let content = messages[i].content.clone();
        messages[i] = ChatMessage {
            role: "tool".to_string(),
            content,
            tool_calls: None,
            tool_call_id: Some(call_id),
        };
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(role: Role, mark: Option<i64>) -> RoleSegment {
        RoleSegment {
            role,
            block_start: 0,
            block_count: 1,
            tokens: vec![],
            content: format!("{}-content", role.as_str()),
            tool_result_turn: mark,
        }
    }

    #[test]
    fn pairs_marked_user_after_assistant() {
        let segs = vec![seg(Role::Assistant, None), seg(Role::User, Some(3))];
        let msgs = vec![
            ChatMessage::plain("assistant", "a"),
            ChatMessage::plain("user", "u"),
        ];
        let out = tool_shape_segment_messages(msgs, &segs);
        assert_eq!(out[0].tool_calls.as_ref().unwrap()[0].id, "call_turn_3");
        assert_eq!(out[1].role, "tool");
        assert_eq!(out[1].tool_call_id.as_deref(), Some("call_turn_3"));
        assert_eq!(out[1].content, "u");
    }

    #[test]
    fn demotes_unpaired_mark() {
        // Marked user at index 0 (no preceding assistant) -> demoted, stays plain.
        let mut segs = vec![seg(Role::User, Some(1))];
        demote_unpaired_tool_marks(&mut segs);
        assert_eq!(segs[0].tool_result_turn, None);
        let msgs = vec![ChatMessage::plain("user", "u")];
        let out = tool_shape_segment_messages(msgs, &segs);
        assert_eq!(out[0].role, "user");
        assert!(out[0].tool_call_id.is_none());
    }
}
