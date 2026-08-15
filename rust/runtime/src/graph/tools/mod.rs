// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-dispatch contracts used by recorded-agent trace drivers.

pub mod dispatch;

pub use dispatch::{
    AgentObservationFormatter, AgentObservationFormatterFactory, AgentToolCall,
    AgentToolCallDecoder, AgentToolCallDecoderFactory, InMemoryAgentObservationFormatter,
    InMemoryAgentObservationFormatterFactory, InMemoryAgentToolCallDecoder,
    InMemoryAgentToolCallDecoderFactory, InMemoryToolDispatcher, InMemoryToolDispatcherFactory,
    ToolDispatchError, ToolDispatchRequest, ToolDispatchResult, ToolDispatcher,
    ToolDispatcherFactory,
};
