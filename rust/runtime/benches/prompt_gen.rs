// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Throughput benchmark for the synthetic prompt generator
//! ([`CorpusPromptGeneratorFactory`] / [`CorpusPromptGenerator`]).
//!
//! Answers "how fast does the Rust dataset manager generate synthetic prompts?"
//! by timing the per-prompt hot path (`build_token_ids` → optional tokenizer
//! decode) in isolation, with corpus tokenization moved into one-time setup so
//! the measured region is only the steady-state generation cost.
//!
//! Groups:
//! - `generate/tokens` — full path: sample tokens + decode to text (what a text
//!   endpoint pays per prompt). `Throughput::Elements(isl)` makes criterion
//!   report the estimate as tokens/second in addition to time/prompt.
//! - `generate_token_ids/tokens` — no-decode raw-token path (what a token-native
//!   endpoint pays): sample + EOS scan, no tokenizer round-trip.
//! - `generate/prefix_reuse` — steady-state prefix-block reuse (hash-id cache
//!   warm): copies cached blocks instead of resampling, the KV-cache-reuse case.
//!
//! Run: `cargo bench -p aiperf-runtime --bench prompt_gen`
//! (add `--profile release` semantics are already applied — benches build in the
//! release profile by default).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use aiperf_runtime::dataset::{
    CorpusPromptGeneratorFactory, PromptGenerator, PromptGeneratorFactory, TiktokenTokenizer,
};
use aiperf_runtime::rng::RngRoot;

/// Input sequence lengths (in tokens) swept by the length-parameterized groups.
/// Spans a short chat turn (16) through a large RAG-style context (4096).
const ISL_SWEEP: &[usize] = &[16, 128, 512, 2048, 4096];

/// Build one generator over the default Shakespeare corpus and a fixed seed. The
/// corpus is tokenized inside `create`, so calling this in setup (not in the
/// timed closure) keeps that one-time cost out of the per-prompt measurement.
fn generator<'a>(
    factory: &CorpusPromptGeneratorFactory,
    tokenizer: &'a TiktokenTokenizer,
) -> Box<dyn PromptGenerator + 'a> {
    factory
        .create(tokenizer, RngRoot::new(Some(1234)))
        .expect("default corpus tokenizes to a non-empty token stream")
}

/// Full text path: sample `isl` tokens from the corpus and decode them to a
/// prompt string, across the ISL sweep.
fn bench_generate_text(c: &mut Criterion) {
    let tokenizer = TiktokenTokenizer::builtin();
    let factory = CorpusPromptGeneratorFactory::default();

    let mut group = c.benchmark_group("generate/tokens");
    for &isl in ISL_SWEEP {
        // Report elements/sec == tokens/sec for the estimate.
        group.throughput(Throughput::Elements(isl as u64));
        group.bench_with_input(BenchmarkId::from_parameter(isl), &isl, |b, &isl| {
            let mut g = generator(&factory, &tokenizer);
            b.iter(|| {
                let prompt = g.generate(black_box(isl), &[], 1).expect("generate");
                black_box(prompt);
            });
        });
    }
    group.finish();
}

/// No-decode raw-token path: sample `isl` tokens plus the EOS-replacement scan,
/// skipping the tokenizer decode, across the ISL sweep.
fn bench_generate_token_ids(c: &mut Criterion) {
    let tokenizer = TiktokenTokenizer::builtin();
    let factory = CorpusPromptGeneratorFactory::default();

    let mut group = c.benchmark_group("generate_token_ids/tokens");
    for &isl in ISL_SWEEP {
        group.throughput(Throughput::Elements(isl as u64));
        group.bench_with_input(BenchmarkId::from_parameter(isl), &isl, |b, &isl| {
            let mut g = generator(&factory, &tokenizer);
            b.iter(|| {
                let ids = g
                    .generate_token_ids(black_box(isl), &[], 1)
                    .expect("generate_token_ids");
                black_box(ids);
            });
        });
    }
    group.finish();
}

/// Steady-state prefix-block reuse: a fixed 2048-token prompt framed as
/// `block_size`-token hash-id blocks. The first timed iteration samples and
/// caches every block; every subsequent iteration copies the cached blocks — the
/// KV-cache/shared-prefix reuse path that dedups repeated prefixes.
fn bench_prefix_reuse(c: &mut Criterion) {
    let tokenizer = TiktokenTokenizer::builtin();
    let factory = CorpusPromptGeneratorFactory::default();

    const ISL: usize = 2048;
    const BLOCK_SIZE: usize = 64;
    // `isl == n_blocks * block_size` so the final block is full (2048 / 64 = 32).
    let hash_ids: Vec<i64> = (0..(ISL / BLOCK_SIZE) as i64).collect();

    let mut group = c.benchmark_group("generate/prefix_reuse");
    group.throughput(Throughput::Elements(ISL as u64));
    group.bench_function(BenchmarkId::new("block_size", BLOCK_SIZE), |b| {
        let mut g = generator(&factory, &tokenizer);
        b.iter(|| {
            let prompt = g
                .generate(black_box(ISL), &hash_ids, BLOCK_SIZE)
                .expect("generate with hash ids");
            black_box(prompt);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_generate_text,
    bench_generate_token_ids,
    bench_prefix_reuse
);
criterion_main!(benches);
