// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for recorded-agent replay timing normalization.

use aiperf_runtime::graph::replay::{
    ReplayCallMeasurement, ReplayMetricsPolicy, StockReplayMetricsPolicy,
};

#[test]
fn anomalous_call_keeps_raw_values_but_makes_aggregate_normalized_values_none() {
    let policy = StockReplayMetricsPolicy;
    let metrics = policy
        .analyze_call(&ReplayCallMeasurement {
            trace_id: "trace-a".into(),
            call_index: 0,
            raw_end_to_end_ms: 12.0,
            raw_inference_ms: 10.0,
            raw_generation_ms: 0.0,
            ttft_ms: Some(4.0),
            stream_total_ms: Some(10.0),
            observed_isl: 9,
            observed_osl: 4,
            target_osl: 8,
            recorded_prompt_isl: Some(9),
            sse_event_count: 1,
            has_meaningful_output: true,
            has_done: true,
            has_required_usage: true,
        })
        .expect("safe raw timing is retained as an anomalous call");

    assert_eq!(metrics.raw_end_to_end_ms, 12.0);
    assert_eq!(metrics.raw_inference_ms, 10.0);
    assert!(metrics.normalized_generation_ms.is_none());
    assert!(metrics.normalized_end_to_end_ms.is_none());
    assert!(
        metrics
            .anomaly_reasons
            .iter()
            .any(|reason| reason == "zero_generation_time")
    );
}
