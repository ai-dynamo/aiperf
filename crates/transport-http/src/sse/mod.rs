// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental SSE stream parsing.

pub mod chat_chunk;
pub mod reader;
pub use chat_chunk::{ChatChoice, ChatChunk, Delta, TokenDetails, Usage};
pub use reader::{SseMessageHandler, read_sse, read_sse_with_handler};
