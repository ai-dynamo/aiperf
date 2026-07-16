// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden-vector generator for the Python parity RNG.
//!
//! Emits a JSON document of seeded RNG outputs to stdout. The Python parity backend
//! (`aiperf.common.rng_parity`) replays the identical operation script and asserts
//! byte-exact equality in `tests/unit/common/test_rng_parity.py`. Floats are emitted as
//! their raw IEEE-754 `u64` bit patterns (`f64::to_bits`) so the comparison is exact and
//! never subject to decimal-formatting drift.
//!
//! Regenerate the committed golden with:
//! ```bash
//! cargo run -p aiperf --example rng_parity_vectors \
//!   > rust/aiperf/tests/data/rng_parity_vectors.json
//! ```

use aiperf_runtime::rng::{HashIdRandomGenerator, RandomGenerator, RngRoot, namespace};
use serde_json::{Value, json};

fn bits(values: &[f64]) -> Vec<String> {
    // Emit as decimal-string u64 so JSON stays exact for the full u64 range.
    values.iter().map(|v| v.to_bits().to_string()).collect()
}

fn main() {
    let mut out = serde_json::Map::new();

    // --- BLAKE3 derivation vectors (seed algebra) ---
    let mut derive = serde_json::Map::new();
    for id in namespace::ALL {
        derive.insert(
            id.to_string(),
            json!(
                RngRoot::new(Some(42))
                    .derive_seed(id)
                    .map(|s| s.to_string())
            ),
        );
    }
    derive.insert(
        "empty_id".into(),
        json!(RngRoot::new(Some(42)).derive_seed("").unwrap().to_string()),
    );
    derive.insert(
        "root0_a".into(),
        json!(RngRoot::new(Some(0)).derive_seed("a").unwrap().to_string()),
    );
    derive.insert(
        "variation".into(),
        json!(
            RngRoot::new(Some(42))
                .derive_variation_seed("concurrency=4")
                .unwrap()
                .to_string()
        ),
    );
    out.insert("derive".into(), Value::Object(derive));

    // --- raw u64 stream ---
    let mut g = RandomGenerator::from_seed(Some(42));
    let u64s: Vec<String> = (0..12).map(|_| g.random_u64().to_string()).collect();
    out.insert("u64".into(), json!(u64s));

    // --- f64 [0,1) stream (as bits) ---
    let mut g = RandomGenerator::from_seed(Some(43));
    let f64s: Vec<f64> = (0..12).map(|_| g.random()).collect();
    out.insert("f64_bits".into(), json!(bits(&f64s)));

    // --- integer ranges ---
    let mut g = RandomGenerator::from_seed(Some(44));
    let mut ranges: Vec<i64> = Vec::new();
    for _ in 0..5 {
        ranges.push(g.randrange(2, 10, 2).unwrap());
    }
    for _ in 0..5 {
        ranges.push(g.randrange(10, 2, -3).unwrap());
    }
    for _ in 0..5 {
        ranges.push(g.randint(1, 3).unwrap());
    }
    for _ in 0..5 {
        ranges.push(g.randbelow(100).unwrap());
    }
    let u64_ranges: Vec<String> = (0..5)
        .map(|_| g.randrange_u64(1000, 2000).unwrap().to_string())
        .collect();
    out.insert("randrange".into(), json!(ranges));
    out.insert("randrange_u64".into(), json!(u64_ranges));

    // --- choice / shuffle / sample ---
    let pool: Vec<u64> = (0..10).collect();
    let mut g = RandomGenerator::from_seed(Some(45));
    let choices: Vec<u64> = (0..12).map(|_| *g.choice(&pool).unwrap()).collect();
    out.insert("choice".into(), json!(choices));

    let mut g = RandomGenerator::from_seed(Some(46));
    let mut shuffled = pool.clone();
    g.shuffle(&mut shuffled);
    out.insert("shuffle".into(), json!(shuffled));

    let mut g = RandomGenerator::from_seed(Some(47));
    let sampled = g.sample(&pool, 5).unwrap();
    out.insert("sample".into(), json!(sampled));

    // --- weighted choice ---
    let vals: Vec<u64> = vec![0, 1, 2, 3];
    let weights = [1.0_f64, 2.0, 3.0, 4.0];
    let mut g = RandomGenerator::from_seed(Some(48));
    let weighted: Vec<u64> = (0..12)
        .map(|_| g.weighted_choice(&vals, Some(&weights)).unwrap())
        .collect();
    out.insert("weighted_choice".into(), json!(weighted));

    // --- numpy_choice ---
    let five: Vec<u64> = (0..5).collect();
    let w5 = [1.0_f64, 0.0, 2.0, 1.0, 4.0];
    let mut g = RandomGenerator::from_seed(Some(49));
    let np_replace = g.numpy_choice(&five, 8, Some(&w5), true).unwrap();
    out.insert("numpy_choice_replace".into(), json!(np_replace));

    let mut g = RandomGenerator::from_seed(Some(50));
    let np_noreplace = g.numpy_choice(&five, 3, Some(&w5), false).unwrap();
    out.insert("numpy_choice_noreplace".into(), json!(np_noreplace));

    // --- continuous distributions (as bits) ---
    let mut g = RandomGenerator::from_seed(Some(51));
    let exp: Vec<f64> = (0..12).map(|_| g.expovariate(4.0).unwrap()).collect();
    out.insert("expovariate_bits".into(), json!(bits(&exp)));

    let mut g = RandomGenerator::from_seed(Some(52));
    let mut gamma: Vec<f64> = Vec::new();
    for _ in 0..8 {
        gamma.push(g.gammavariate(2.0, 3.0).unwrap());
    }
    for _ in 0..8 {
        gamma.push(g.gammavariate(0.5, 2.0).unwrap());
    }
    for _ in 0..4 {
        gamma.push(g.gammavariate(1.0, 2.0).unwrap());
    }
    out.insert("gammavariate_bits".into(), json!(bits(&gamma)));

    let mut g = RandomGenerator::from_seed(Some(53));
    let normal: Vec<f64> = (0..12).map(|_| g.normal(4.0, 2.0).unwrap()).collect();
    out.insert("normal_bits".into(), json!(bits(&normal)));

    let mut g = RandomGenerator::from_seed(Some(54));
    let bnormal: Vec<f64> = (0..12)
        .map(|_| g.sample_normal(10.0, 2.0, 8.0, 12.0).unwrap())
        .collect();
    out.insert("sample_normal_bits".into(), json!(bits(&bnormal)));

    let mut g = RandomGenerator::from_seed(Some(55));
    let pint: Vec<i64> = (0..12)
        .map(|_| g.sample_positive_normal_integer(100.0, 10.0).unwrap())
        .collect();
    out.insert("positive_normal_int".into(), json!(pint));

    // --- hash-id reseed stream ---
    let mut base = RandomGenerator::from_seed(Some(42));
    let mut hid = HashIdRandomGenerator::from_base(&mut base);
    let mut hid_out: Vec<String> = Vec::new();
    for (scope, hash_id) in [("trace-a", 7_i64), ("trace-b", -3), ("trace-a", 99)] {
        hid.reseed_for_hash_id(hash_id, Some(scope));
        hid_out.push(hid.random_u64().to_string());
        hid_out.push(hid.random_u64().to_string());
    }
    out.insert("hash_id_u64".into(), json!(hid_out));

    // --- fill_bytes ---
    let mut g = RandomGenerator::from_seed(Some(56));
    let mut buf = [0u8; 37];
    g.fill_bytes(&mut buf);
    out.insert(
        "fill_bytes_hex".into(),
        json!(buf.iter().map(|b| format!("{b:02x}")).collect::<String>()),
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(out)).unwrap()
    );
}
