// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental SSE stream parsing.

pub mod reader;
// The OpenAI chat SSE chunk codec is a dialect concern owned by `aiperf-endpoints`;
// re-exported here so streaming callers keep importing it from the transport `sse` module.
pub use crate::endpoints::chat_chunk::{ChatChoice, ChatChunk, Delta, TokenDetails, Usage};
pub use reader::{SseMessageHandler, read_sse, read_sse_with_handler};
