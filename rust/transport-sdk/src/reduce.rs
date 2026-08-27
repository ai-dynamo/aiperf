// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral reduction of decoded responses into observer facts.
//!
//! A transport decodes wire records into its own endpoint-shaped values; this
//! module owns the parts of the fold that are the *same* for every transport —
//! the once-only first-token latch, batched output-token emission, text
//! accumulation, and terminal usage reconciliation. Those exist exactly once
//! here so HTTP, gRPC, and WebSocket cannot drift apart on what a token
//! observation means.

use std::cell::Cell;

use aiperf_core::dispatch::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use smallvec::SmallVec;
use uuid::Uuid;

/// Token-emission context supplied by a transport for one request.
///
/// The first-token latch is borrowed rather than owned because a streaming
/// dispatch may fold responses from more than one call site; TTFT is the first
/// token observation across all of them, not the first per site.
pub struct TokenEmitter<'a> {
    /// Per-request measurement identity.
    pub uuid: Uuid,
    /// Whether this endpoint produces streamable tokens at all.
    pub produces_tokens: bool,
    /// Clock-ns when this request's dispatch started, subtracted to form the
    /// first-token delta.
    pub start_ns: i64,
    /// The measurement observer.
    pub obs: &'a dyn RequestObserver,
    /// Map an absolute clock-ns instant to run-relative milliseconds.
    pub to_ms: &'a dyn Fn(i64) -> f64,
    /// Shared once-only first-token latch.
    pub first_token_released: &'a Cell<bool>,
    /// First-token callback taking a dispatch-relative ns delta, i.e. TTFT.
    pub on_first_token: &'a dyn Fn(i64),
}

impl TokenEmitter<'_> {
    /// Release the first-token gate exactly once, at `at_ns`.
    ///
    /// Returns whether this call was the one that released it.
    pub fn release_first_token(&self, at_ns: i64) -> bool {
        if self.first_token_released.replace(true) {
            return false;
        }
        (self.on_first_token)(at_ns.saturating_sub(self.start_ns));
        true
    }

    /// Emit one chunk of token ids observed at `at_ns`.
    ///
    /// All ids in a chunk share the same arrival instant. The timestamps are
    /// built in a stack-inline buffer so the common single-token streaming chunk
    /// costs no heap allocation; the observations are identical either way.
    pub fn emit_token_ids(&self, token_count: usize, at_ns: i64) {
        if !self.produces_tokens || token_count == 0 {
            return;
        }
        self.release_first_token(at_ns);
        let at_ms = (self.to_ms)(at_ns);
        let timestamps: SmallVec<[f64; 8]> = smallvec::smallvec![at_ms; token_count];
        self.obs.on_output_tokens(self.uuid, &timestamps);
    }

    /// Emit one text chunk observed at `at_ns`, classified as `kind`.
    pub fn emit_text(&self, text: &str, kind: ObservedTokenKind, at_ns: i64) {
        if !self.produces_tokens || text.is_empty() {
            return;
        }
        self.release_first_token(at_ns);
        self.obs
            .on_classified_token(self.uuid, (self.to_ms)(at_ns), kind);
    }
}

/// Append `text` to an optional accumulator, creating it on first use.
///
/// The `Option` is meaningful: an absent field and an empty field are different
/// facts at the serialization boundary, so a response that contributed nothing
/// must leave the field `None`.
pub fn append_text(target: &mut Option<String>, text: &str) {
    target.get_or_insert_with(String::new).push_str(text);
}

/// Fold one response's usage into the terminal observation.
///
/// Later reports win, matching the streaming convention that the last usage
/// frame is the authoritative one. Absent fields stay absent: an endpoint that
/// never reported a count must not be given a zero.
pub fn absorb_observed_usage(reported: &ObservedUsage, observed: &mut ObservedUsage) {
    if reported.first_content_chunk_tokens.is_some() {
        observed.first_content_chunk_tokens = reported.first_content_chunk_tokens;
    }
}

/// Fold one response's endpoint-specific metrics into the terminal observation.
///
/// Non-finite values are dropped rather than propagated: a `NaN` reaching the
/// serialization boundary is not representable as a metric.
pub fn absorb_observed_endpoint_metrics(
    reported: &ObservedEndpointMetrics,
    metrics: &mut ObservedEndpointMetrics,
) {
    metrics.video_inference_seconds = reported
        .video_inference_seconds
        .filter(|value| value.is_finite())
        .or(metrics.video_inference_seconds);
    metrics.video_peak_memory_mb = reported
        .video_peak_memory_mb
        .filter(|value| value.is_finite())
        .or(metrics.video_peak_memory_mb);
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use aiperf_core::dispatch::ReplayTerminalStatus;

    use super::*;

    #[derive(Default)]
    struct TokenLog {
        batches: RefCell<Vec<Vec<f64>>>,
        classified: RefCell<Vec<(f64, ObservedTokenKind)>>,
    }

    impl RequestObserver for TokenLog {
        fn on_arrival(&self, _uuid: Uuid, _ms: f64, _input: usize, _requested: usize) {}
        fn on_admit(&self, _uuid: Uuid, _ms: f64, _reused: usize) {}
        fn on_token(&self, _uuid: Uuid, _at_ms: f64) {}
        fn on_classified_token(&self, _uuid: Uuid, at_ms: f64, kind: ObservedTokenKind) {
            self.classified.borrow_mut().push((at_ms, kind));
        }
        fn on_output_tokens(&self, _uuid: Uuid, at_ms: &[f64]) {
            self.batches.borrow_mut().push(at_ms.to_vec());
        }
        fn on_terminal(&self, _uuid: Uuid, _status: ReplayTerminalStatus) {}
    }

    fn emitter<'a>(
        obs: &'a TokenLog,
        latch: &'a Cell<bool>,
        ttft: &'a dyn Fn(i64),
        to_ms: &'a dyn Fn(i64) -> f64,
    ) -> TokenEmitter<'a> {
        TokenEmitter {
            uuid: Uuid::from_u128(1),
            produces_tokens: true,
            start_ns: 1_000,
            obs,
            to_ms,
            first_token_released: latch,
            on_first_token: ttft,
        }
    }

    #[test]
    fn first_token_fires_once_across_batches_and_text() {
        let obs = TokenLog::default();
        let latch = Cell::new(false);
        let ttfts: RefCell<Vec<i64>> = RefCell::new(Vec::new());
        let record = |delta: i64| ttfts.borrow_mut().push(delta);
        let to_ms = |ns: i64| ns as f64 / 1e6;
        let emit = emitter(&obs, &latch, &record, &to_ms);

        emit.emit_token_ids(3, 3_000);
        emit.emit_text("more", ObservedTokenKind::Output, 5_000);

        assert_eq!(ttfts.borrow().as_slice(), &[2_000]);
        assert_eq!(obs.batches.borrow().as_slice(), &[vec![0.003, 0.003, 0.003]]);
        assert_eq!(obs.classified.borrow().len(), 1);
    }

    #[test]
    fn empty_chunks_release_nothing() {
        let obs = TokenLog::default();
        let latch = Cell::new(false);
        let ttfts: RefCell<Vec<i64>> = RefCell::new(Vec::new());
        let record = |delta: i64| ttfts.borrow_mut().push(delta);
        let to_ms = |ns: i64| ns as f64 / 1e6;
        let emit = emitter(&obs, &latch, &record, &to_ms);

        emit.emit_token_ids(0, 3_000);
        emit.emit_text("", ObservedTokenKind::Output, 3_000);

        assert!(ttfts.borrow().is_empty());
        assert!(!latch.get());
    }

    #[test]
    fn non_finite_endpoint_metrics_are_dropped() {
        let mut metrics = ObservedEndpointMetrics::default();
        absorb_observed_endpoint_metrics(
            &ObservedEndpointMetrics {
                video_inference_seconds: Some(f64::NAN),
                video_peak_memory_mb: Some(12.0),
                ..ObservedEndpointMetrics::default()
            },
            &mut metrics,
        );
        assert!(metrics.video_inference_seconds.is_none());
        assert_eq!(metrics.video_peak_memory_mb, Some(12.0));
    }
}
