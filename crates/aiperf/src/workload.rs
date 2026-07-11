// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic workload generation for the walking skeleton.

use uuid::Uuid;

use crate::http::HttpRequest;

/// A synthetic workload: `num_requests` chat requests of approximately
/// `input_tokens` prompt length, each asking for `output_tokens` output.
pub struct SkeletonWorkload {
    /// Number of requests to generate.
    pub num_requests: usize,
    /// Approximate prompt length in tokens.
    pub input_tokens: usize,
    /// Requested output length in tokens.
    pub output_tokens: usize,
}

impl SkeletonWorkload {
    /// Mint one fresh [`HttpRequest`] with a new correlation id. Stateless, so the
    /// run loop can pull requests on demand and let the stop conditions (not a fixed
    /// list length) decide when to stop.
    ///
    /// The prompt is `input_tokens` whitespace-separated words; tokenizer-exact
    /// input/output lengths are deferred to a later increment.
    pub fn make_request(&self) -> HttpRequest {
        HttpRequest {
            uuid: Uuid::new_v4(),
            input_length: self.input_tokens,
            max_output_tokens: self.output_tokens,
            prompt_text: Some(vec!["lorem"; self.input_tokens].join(" ")),
        }
    }

    /// Materialize `num_requests` requests at once (used by tests).
    pub fn generate(&self) -> Vec<HttpRequest> {
        (0..self.num_requests).map(|_| self.make_request()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_requested_count_with_text() {
        let wl = SkeletonWorkload {
            num_requests: 3,
            input_tokens: 10,
            output_tokens: 5,
        };
        let reqs = wl.generate();
        assert_eq!(reqs.len(), 3);
        assert_eq!(reqs[0].max_output_tokens, 5);
        assert!(
            reqs[0]
                .prompt_text
                .as_ref()
                .unwrap()
                .split_whitespace()
                .count()
                >= 8
        );
    }
}
