// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental SSE stream parsing.

pub mod reader;
// Keep the endpoint-owned chat codec available through the transport API.
pub use crate::endpoints::chat_chunk::{ChatChoice, ChatChunk, Delta, TokenDetails, Usage};
pub use reader::{SseMessageHandler, read_sse, read_sse_with_handler};
