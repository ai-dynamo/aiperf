// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Seeded Poisson inter-arrival schedule, emitted as JSONL for cross-language parity.
//!
//! Drives the real `aiperf_runtime::timing` Poisson interval generator (the one the scheduled
//! runtime uses), seeding it from the `timing.request.poisson_interval` stream derived
//! off a root seed with the BLAKE3 algebra. The Python side
//! (`tools/poisson_intervals.py`, backed by `aiperf.common.rng_parity`) reproduces this
//! byte-for-byte. Interval nanoseconds are integers, so JSONL lines compare exactly.
//!
//! Usage: `cargo run -p aiperf --example poisson_intervals -- <root_seed> <rate> <count>`

use aiperf_runtime::rng::{RngRoot, namespace};
use aiperf_runtime::timing::{ArrivalPattern, make_interval_generator};

fn main() {
    let mut args = std::env::args().skip(1);
    let root_seed: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(42);
    let rate: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(50.0);
    let count: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(64);

    // Same derivation the reproducible run uses: the poisson-interval stream seed is
    // BLAKE3-derived from (root_seed, "timing.request.poisson_interval").
    let seed = RngRoot::new(Some(root_seed))
        .derive_seed(namespace::TIMING_REQUEST_POISSON_INTERVAL)
        .expect("seeded root yields a concrete poisson-interval seed");

    let mut generator = make_interval_generator(ArrivalPattern::Poisson, Some(rate), None, seed);

    let mut cumulative_ns: i64 = 0;
    for i in 0..count {
        let interval_ns = generator.next_interval_ns();
        cumulative_ns += interval_ns;
        // Compact JSONL: one object per line, integer nanoseconds (exact across langs).
        println!("{{\"i\":{i},\"interval_ns\":{interval_ns},\"cumulative_ns\":{cumulative_ns}}}");
    }
    eprintln!("poisson: root_seed={root_seed} derived_seed={seed} rate={rate} count={count}");
}
