// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared inter-token-latency decode-token accounting.

use std::sync::atomic::{AtomicBool, Ordering};

static HAS_WARNED_FIRST_CHUNK_MISMATCH: AtomicBool = AtomicBool::new(false);

/// Returns the tokens decoded after the first content-bearing streamed chunk.
///
/// Missing usage preserves the legacy `OSL - 1` divisor. A reported zero or a
/// count at least as large as OSL is inconsistent, so it also falls back while
/// warning once per process.
pub(crate) fn decode_tokens_after_first_chunk(
    output_sequence_length: u64,
    first_content_chunk_tokens: Option<u64>,
) -> Option<u64> {
    if output_sequence_length < 2 {
        return None;
    }
    match first_content_chunk_tokens {
        None => Some(output_sequence_length - 1),
        Some(first_content_chunk_tokens)
            if first_content_chunk_tokens > 0
                && first_content_chunk_tokens < output_sequence_length =>
        {
            Some(output_sequence_length - first_content_chunk_tokens)
        }
        Some(first_content_chunk_tokens) => {
            if !HAS_WARNED_FIRST_CHUNK_MISMATCH.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    first_content_chunk_tokens,
                    output_sequence_length,
                    "server-reported first content chunk token count is inconsistent with output sequence length; falling back to OSL minus one; check per-chunk usage server support"
                );
            }
            Some(output_sequence_length - 1)
        }
    }
}
