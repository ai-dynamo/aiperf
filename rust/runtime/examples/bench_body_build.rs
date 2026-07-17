// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Microbenchmark comparing validated and pre-serialized request-body assembly.
//!
//! The validated strategy rebuilds `Overrides` and serializes the request tail
//! per request. The pre-serialized strategy builds the tail once per distinct
//! `max_tokens` value and byte-splices it without validation.
//!
//! Run: `cargo run --release -p aiperf-runtime --example bench_body_build`

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

    // Rebuild overrides and validate each request.
    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        let overrides = overrides_for(max_tokens);
        let body = build_message_body_from_wires(black_box(&messages), &overrides).unwrap();
        sink = sink.wrapping_add(body.len());
    }
    let base_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // Reuse one serialized tail for every request.
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
    let a = build_message_body_from_wires(&messages, &overrides_for(max_tokens)).unwrap();
    let b = build_message_body_from_wire_parts(&messages, &tail);
    assert_eq!(a, b, "body strategies must be byte-identical");

    println!("body-build hot path ({iters} iters, 2 messages, OSL cap):");
    println!("  validated      (Overrides+serde/req): {base_ns:8.1} ns/op");
    println!("  pre-serialized (tail splice):         {opt_ns:8.1} ns/op");
    println!(
        "  speedup: {:.2}x  ({:.1}% less time)  [sink={sink}]",
        base_ns / opt_ns,
        100.0 * (base_ns - opt_ns) / base_ns
    );
}
