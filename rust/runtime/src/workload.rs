// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic workload generation for the walking skeleton.

use uuid::Uuid;

use crate::transport::core::Request;

/// A synthetic workload: `num_requests` chat requests of approximately
/// `input_tokens` prompt length, each asking for `output_tokens` output.
#[derive(Clone, Debug)]
pub struct SkeletonWorkload {
    /// Number of requests to generate.
    pub num_requests: usize,
    /// Approximate prompt length in tokens.
    pub input_tokens: usize,
    /// Requested output length in tokens.
    pub output_tokens: usize,
    /// Number of turns per synthetic conversation.
    pub turns: usize,
    /// Optional delay before continuation turns, in milliseconds.
    pub think_time_ms: Option<u64>,
}

impl SkeletonWorkload {
    /// Mint one fresh [`Request`] with a new correlation id. Stateless, so the
    /// run loop can pull requests on demand and let the stop conditions (not a fixed
    /// list length) decide when to stop.
    ///
    /// The prompt is `input_tokens` whitespace-separated words; tokenizer-exact
    /// input/output lengths are deferred to a later increment.
    pub fn make_request(&self) -> Request {
        Request {
            uuid: Uuid::new_v4(),
            input_length: self.input_tokens,
            max_output_tokens: self.output_tokens,
            prompt_text: Some(vec!["lorem"; self.input_tokens].join(" ")),
            request_body: None,
            request_body_bytes: None,
            headers: std::collections::BTreeMap::new(),
            parameters: std::collections::BTreeMap::new(),
            endpoint_path: None,
            streaming: true,
            x_correlation_id: None,
            is_final_turn: true,
            cancel_after_ns: None,
            url_index: None,
        }
    }
}
