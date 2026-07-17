// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral dispatch seam.
//!
//! A scheduler drives transport-native requests through [`RequestSink`];
//! measurements flow through
//! [`RequestObserver`] into a [`TraceCollector`](crate::collector::TraceCollector).
//!
//! Each request type implements [`Dispatchable`], keeping engine and wire types
//! behind the trait.

use uuid::Uuid;

use crate::collector::ReplayTerminalStatus;

/// Semantic class of one streamed token-like content delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedTokenKind {
    /// User-visible output content (including tool-call content).
    Output,
    /// Reasoning-only content emitted before user-visible output.
    Reasoning,
}

/// Authoritative server-reported token usage for one request.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ObservedUsage {
    /// Prompt-token count, when the endpoint returned it.
    pub prompt_tokens: Option<usize>,
    /// Completion-token count, when the endpoint returned it.
    pub completion_tokens: Option<usize>,
    /// Explicit provider total, when reported.
    pub total_tokens: Option<usize>,
    /// Provider-reported reasoning-token count.
    pub reasoning_tokens: Option<usize>,
    /// Prompt tokens served from a cache.
    pub prompt_cache_read_tokens: Option<usize>,
    /// Prompt tokens written into a cache.
    pub prompt_cache_write_tokens: Option<usize>,
    /// Explicit prompt cache misses, when reported separately.
    pub prompt_cache_miss_tokens: Option<usize>,
    /// Prompt audio tokens reported in provider usage details.
    pub prompt_audio_tokens: Option<usize>,
    /// Completion audio tokens reported in provider usage details.
    pub completion_audio_tokens: Option<usize>,
    /// Accepted predicted-output tokens.
    pub accepted_prediction_tokens: Option<usize>,
    /// Rejected predicted-output tokens.
    pub rejected_prediction_tokens: Option<usize>,
    /// Tool-definition prompt tokens reported separately by the provider.
    pub tool_use_prompt_tokens: Option<usize>,
    /// Prompt audio duration in seconds, distinct from audio-token counts.
    pub prompt_audio_seconds: Option<f64>,
}

/// Endpoint-specific modality facts that feed native metrics without exposing
/// endpoint wire models to the transport-neutral observer seam.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ObservedEndpointMetrics {
    /// Number of image inputs in the exact request payload.
    pub num_images: Option<usize>,
    /// Server-reported video inference duration in seconds.
    pub video_inference_seconds: Option<f64>,
    /// Server-reported peak video-generation memory in MiB.
    pub video_peak_memory_mb: Option<f64>,
}

/// Measurement hook fed by any sink. Timestamps are milliseconds relative to
/// run start. The observer is intentionally local-loop friendly: it has no
/// `Send`/`Sync` supertraits, so thread-per-core workers can accumulate through
/// `Rc<RefCell<_>>` without a per-token mutex. A consumer that genuinely crosses
/// threads can still provide a thread-safe implementation.
///
/// TTFT is derived by the collector as the first
/// [`on_token`](RequestObserver::on_token), classified-token, or output-token
/// batch callback for a request, so sinks do not emit a separate first-token
/// event.
pub trait RequestObserver {
    /// Record request arrival with its input length and requested output length.
    fn on_arrival(
        &self,
        uuid: Uuid,
        arrival_ms: f64,
        input_length: usize,
        requested_output_length: usize,
    );
    /// Record admission (scheduling start), with the count of prefix-cache-reused input tokens.
    fn on_admit(&self, uuid: Uuid, admit_ms: f64, reused_input_tokens: usize);
    /// Record one output token observed at `at_ms`.
    fn on_token(&self, uuid: Uuid, at_ms: f64);
    /// Record one classified token delta.
    ///
    /// The default preserves compatibility for transports that cannot distinguish
    /// reasoning from user-visible output. Reasoning-aware transports override the
    /// call site, while observer tees forward the classification unchanged.
    fn on_classified_token(&self, uuid: Uuid, at_ms: f64, _kind: ObservedTokenKind) {
        self.on_token(uuid, at_ms);
    }
    /// Record an ordered batch of user-visible output-token timestamps.
    ///
    /// Coalescing transports can override this hook downstream to amortize
    /// correlation and mutable-state access without changing the semantic
    /// callback sequence. The default deliberately replays the ordinary
    /// classified-token callback so every existing observer remains correct.
    fn on_output_tokens(&self, uuid: Uuid, at_ms: &[f64]) {
        for &timestamp in at_ms {
            self.on_classified_token(uuid, timestamp, ObservedTokenKind::Output);
        }
    }
    /// Record the terminal server-usage observation.
    ///
    /// Individual fields remain absent when the endpoint reports no usage. The
    /// default is a no-op; observers that reconcile counts override it.
    fn on_usage(&self, _uuid: Uuid, _usage: ObservedUsage) {}
    /// Record endpoint-specific image/video facts.
    ///
    /// The default is a no-op so token-only transports and observers do not pay
    /// for modality handling. Endpoint-aware observer tees forward it unchanged.
    fn on_endpoint_metrics(&self, _uuid: Uuid, _metrics: ObservedEndpointMetrics) {}
    /// Record terminal status for the request.
    fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus);
}

/// Transport-independent request metadata exposed to sinks and observers.
pub trait Dispatchable: Send + Sync {
    /// Stable per-request identifier used to correlate observer events.
    fn uuid(&self) -> Uuid;
    /// Prompt length in tokens, for measurement accounting.
    fn input_length(&self) -> usize;
    /// Maximum number of output tokens to request.
    fn max_output_tokens(&self) -> usize;
}

/// Dispatch one request of type `R` to a terminal state, resolving on completion.
///
/// Implementations emit measurement events through `obs`. `dispatch` returns
/// `Err` only on a transport/dispatch failure the caller should surface; a
/// request that completes with an error terminal status returns `Ok(())` after
/// emitting `obs.on_terminal(..)`.
///
/// `?Send`: the sink is driven on a single-threaded `LocalSet` (the hyper
/// transport is `!Send`, holding `Rc<dyn Clock>`), so neither the sink nor its
/// dispatch future is required to be `Send`.
#[async_trait::async_trait(?Send)]
pub trait RequestSink<R: Dispatchable> {
    /// Dispatch `req`, awaiting terminal completion.
    async fn dispatch(&self, req: R, obs: &dyn RequestObserver) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct TinyRequest {
        uuid: Uuid,
        input_length: usize,
        max_output_tokens: usize,
    }
    impl Dispatchable for TinyRequest {
        fn uuid(&self) -> Uuid {
            self.uuid
        }
        fn input_length(&self) -> usize {
            self.input_length
        }
        fn max_output_tokens(&self) -> usize {
            self.max_output_tokens
        }
    }

    #[derive(Default)]
    struct RecordingObserver {
        tokens: Mutex<Vec<(Uuid, f64)>>,
    }
    impl RequestObserver for RecordingObserver {
        fn on_arrival(&self, _u: Uuid, _a: f64, _i: usize, _o: usize) {}
        fn on_admit(&self, _u: Uuid, _a: f64, _r: usize) {}
        fn on_token(&self, u: Uuid, at: f64) {
            self.tokens.lock().unwrap().push((u, at));
        }
        fn on_terminal(&self, _u: Uuid, _s: ReplayTerminalStatus) {}
    }

    struct EchoSink;
    #[async_trait::async_trait(?Send)]
    impl RequestSink<TinyRequest> for EchoSink {
        async fn dispatch(
            &self,
            req: TinyRequest,
            obs: &dyn RequestObserver,
        ) -> anyhow::Result<()> {
            obs.on_arrival(req.uuid(), 0.0, req.input_length(), req.max_output_tokens());
            for i in 0..req.max_output_tokens() {
                obs.on_token(req.uuid(), i as f64);
            }
            obs.on_terminal(req.uuid(), ReplayTerminalStatus::Completed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn sink_emits_one_token_per_output() {
        let obs = RecordingObserver::default();
        let req = TinyRequest {
            uuid: Uuid::nil(),
            input_length: 3,
            max_output_tokens: 5,
        };
        EchoSink.dispatch(req, &obs).await.unwrap();
        assert_eq!(obs.tokens.lock().unwrap().len(), 5);
    }

    #[test]
    fn output_token_batch_defaults_to_ordered_token_callbacks() {
        let obs = RecordingObserver::default();
        let uuid = Uuid::from_u128(7);
        obs.on_output_tokens(uuid, &[1.0, 2.5, 4.0]);
        assert_eq!(
            *obs.tokens.lock().unwrap(),
            vec![(uuid, 1.0), (uuid, 2.5), (uuid, 4.0)]
        );
    }
}
