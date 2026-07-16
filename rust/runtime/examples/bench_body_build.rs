// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Microbench for the graph dispatch body-build hot path (Opt 1 + Opt 4).
//!
//! Baseline = the pre-optimization path: rebuild `Overrides` and re-`serde_json`
//! the model/stream/include_usage/max_tokens tail on *every* request, validating
//! each message slice. Optimized = pre-serialize the tail once (per distinct
//! `max_tokens`) and byte-splice with no validation.
//!
//! Run: `cargo run --release -p aiperf --example bench_body_build`

use std::hint::black_box;
use std::time::Instant;

use aiperf_runtime::dataset::{
    Overrides, build_message_body_from_wire_parts, build_message_body_from_wires,
};
use bytes::Bytes;

fn overrides_for(max_tokens: u32) -> Overrides {
    let mut o = Overrides::new();
    o.set_model("meta-llama/Llama-3.1-8B-Instruct");
    o.set_stream(true);
    o.set_include_usage(true);
    o.set_max_tokens("max_tokens", max_tokens);
    o
}

fn main() {
    // Two pre-serialized message wires, as the materializer would produce.
    let messages = vec![
        Bytes::from_static(br#"{"role":"system","content":"You are a helpful assistant."}"#),
        Bytes::from_static(
            br#"{"role":"user","content":"Summarize the following text in one sentence."}"#,
        ),
    ];
    let iters: u64 = 5_000_000;
    let max_tokens = 1_u32;

    // --- Baseline: Overrides rebuilt + serde + validation per request. ---
    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        let overrides = overrides_for(max_tokens);
        let body = build_message_body_from_wires(black_box(&messages), &overrides).unwrap();
        sink = sink.wrapping_add(body.len());
    }
    let base_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // --- Optimized: pre-serialize the tail once, byte-splice per request. ---
    let tail = Bytes::from(overrides_for(max_tokens).inner_bytes().unwrap());
    let t1 = Instant::now();
    let mut sink2 = 0usize;
    for _ in 0..iters {
        let body = build_message_body_from_wire_parts(black_box(&messages), &tail);
        sink2 = sink2.wrapping_add(body.len());
    }
    let opt_ns = t1.elapsed().as_nanos() as f64 / iters as f64;

    assert_eq!(
        sink, sink2,
        "byte lengths must match — outputs are identical"
    );
    // Prove byte-identical output, not just equal length.
    let a = build_message_body_from_wires(&messages, &overrides_for(max_tokens)).unwrap();
    let b = build_message_body_from_wire_parts(&messages, &tail);
    assert_eq!(a, b, "optimized body must be byte-identical to baseline");

    println!("body-build hot path ({iters} iters, 2 messages, OSL cap):");
    println!("  baseline  (Overrides+serde+validate/req): {base_ns:8.1} ns/op");
    println!("  optimized (pre-serialized tail, no valid): {opt_ns:8.1} ns/op");
    println!(
        "  speedup: {:.2}x  ({:.1}% less time)  [sink={sink}]",
        base_ns / opt_ns,
        100.0 * (base_ns - opt_ns) / base_ns
    );
}
