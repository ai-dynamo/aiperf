// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-internal fixed-seed canary for the complete RNG substrate.

use std::fmt::Write;

use aiperf_runtime::rng::namespace;
use aiperf_runtime::rng::{
    EmpiricalPoint, HashIdRandomGenerator, PeakEntry, RngRoot, SamplingDistribution,
    SequenceLengthDistribution, SequenceLengthPair,
};

fn bits(values: impl IntoIterator<Item = f64>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect()
}

fn seeded_profile() -> String {
    let root = RngRoot::new(Some(42));
    let mut output = String::new();

    for name in [
        namespace::DATASET_PROMPT_LENGTH,
        namespace::DATASET_AUDIO_DURATION,
        namespace::MODELS_SEQUENCE_DISTRIBUTION,
        namespace::TIMING_REQUEST_GAMMA_INTERVAL,
    ] {
        writeln!(
            &mut output,
            "seed {name} {}",
            root.derive_seed(name).unwrap()
        )
        .unwrap();
    }

    let mut prompt_rng = root.derive(namespace::DATASET_PROMPT_LENGTH);
    let prompt = SamplingDistribution::normal(100.0, 10.0).unwrap();
    let prompt_lengths: Vec<_> = (0..12)
        .map(|_| prompt.sample_int(&mut prompt_rng).unwrap())
        .collect();
    writeln!(&mut output, "prompt {prompt_lengths:?}").unwrap();

    let sequence = SequenceLengthDistribution::new(vec![
        SequenceLengthPair::new_with_stddev(128, 8.0, 64, 4.0, 35.0).unwrap(),
        SequenceLengthPair::new_with_stddev(512, 16.0, 256, 8.0, 65.0).unwrap(),
    ])
    .unwrap();
    let mut sequence_rng = root.derive(namespace::MODELS_SEQUENCE_DISTRIBUTION);
    writeln!(
        &mut output,
        "sequence {:?}",
        sequence.sample_batch(&mut sequence_rng, 12).unwrap()
    )
    .unwrap();

    let mixture = SamplingDistribution::multimodal(vec![
        PeakEntry::new(SamplingDistribution::fixed(4.0).unwrap(), 1.0).unwrap(),
        PeakEntry::new(SamplingDistribution::lognormal(20.0, 10.0).unwrap(), 3.0).unwrap(),
    ])
    .unwrap()
    .with_bounds(Some(2.0), Some(40.0))
    .unwrap();
    let empirical = SamplingDistribution::empirical(vec![
        EmpiricalPoint::new(1.0, 1.0).unwrap(),
        EmpiricalPoint::new(3.0, 2.0).unwrap(),
        EmpiricalPoint::new(9.0, 1.0).unwrap(),
    ])
    .unwrap();
    let mut distribution_rng = root.derive(namespace::DATASET_SYNTHESIS_EMPIRICAL_SAMPLER);
    let mixture_bits = bits((0..8).map(|_| mixture.sample(&mut distribution_rng).unwrap()));
    let empirical_values: Vec<_> = (0..12)
        .map(|_| empirical.sample(&mut distribution_rng).unwrap() as i64)
        .collect();
    writeln!(&mut output, "mixture {mixture_bits:?}").unwrap();
    writeln!(&mut output, "empirical {empirical_values:?}").unwrap();

    let mut generator = root.derive(namespace::DATASET_SAMPLER_RANDOM);
    let random_bits = bits(generator.random_batch(4));
    let integers = generator.integers(-5, Some(8), 8).unwrap();
    let choices = generator.choices(&['a', 'b', 'c'], 8).unwrap();
    let weighted = generator
        .numpy_choice(&[10, 20, 30], 6, Some(&[1.0, 2.0, 7.0]), true)
        .unwrap();
    let mut shuffled = [0, 1, 2, 3, 4, 5];
    generator.shuffle(&mut shuffled);
    let sampled = generator.sample(&shuffled, 3).unwrap();
    let continuous = bits([
        generator.uniform(-1.0, 2.0),
        generator.expovariate(2.0).unwrap(),
        generator.gammavariate(3.0, 0.25).unwrap(),
        generator.normal(5.0, 2.0).unwrap(),
    ]);
    let mut random_bytes = [0_u8; 12];
    generator.fill_bytes(&mut random_bytes);
    writeln!(&mut output, "uniforms {random_bits:?}").unwrap();
    writeln!(&mut output, "integers {integers:?}").unwrap();
    writeln!(&mut output, "choices {choices:?} weighted {weighted:?}").unwrap();
    writeln!(&mut output, "shuffle {shuffled:?} sample {sampled:?}").unwrap();
    writeln!(&mut output, "continuous {continuous:?}").unwrap();
    writeln!(&mut output, "bytes {random_bytes:02x?}").unwrap();

    let mut hash_base = root.derive(namespace::DATASET_PROMPT_CORPUS);
    let mut hash_rng = HashIdRandomGenerator::from_base(&mut hash_base);
    for (scope, hash_id) in [("trace-a", 7), ("trace-b", -3), ("", i64::MAX)] {
        hash_rng.reseed_for_hash_id(hash_id, Some(scope));
        writeln!(
            &mut output,
            "hash {scope:?} {hash_id} {} {:?}",
            hash_rng.generator().seed().unwrap(),
            [
                hash_rng.random_u64(),
                hash_rng.random_u64(),
                hash_rng.random_u64()
            ]
        )
        .unwrap();
    }

    output
}

#[test]
fn seed_42_profile_is_byte_stable() {
    let actual = seeded_profile();
    assert_eq!(actual, include_str!("fixtures/rng_profile_seed_42.txt"));
}

#[test]
fn profile_repeats_exactly_within_one_process() {
    assert_eq!(seeded_profile(), seeded_profile());
}
