// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Throughput benchmark for the synthetic prompt generator
//! ([`CorpusPromptGeneratorFactory`] / [`PreparedCorpusPromptGeneratorFactory`]).
//!
//! Two measurement layers:
//!
//! 1. **Setup cost** (`setup/*`) — one-shot Shakespeare corpus tokenization via
//!    [`CorpusPromptGeneratorFactory::prepare`], and the cheap follow-on
//!    [`PromptGeneratorFactory::create`] from prepared tokens. These are
//!    deliberately separate from per-prompt generation so dataset-load timing
//!    can keep preparation outside its measured region while still quantifying
//!    the absolute setup cost.
//! 2. **Per-prompt hot path** (`generate/*`, `generate_token_ids/*`) — steady-
//!    state `build_token_ids` → optional tokenizer decode, with corpus
//!    tokenization moved into one-time setup so the measured region excludes
//!    prepare.
//!
//! Groups:
//! - `setup/prepare_corpus` — cold `prepare` (chunked Shakespeare tokenization).
//! - `setup/create_from_prepared` — cheap generator construction from prepared
//!   `Arc<[u32]>` tokens (no corpus encode).
//! - `generate/tokens` — full path: sample tokens + decode to text (what a text
//!   endpoint pays per prompt). `Throughput::Elements(isl)` makes criterion
//!   report the estimate as tokens/second in addition to time/prompt.
//! - `generate_token_ids/tokens` — no-decode raw-token path (what a token-native
//!   endpoint pays): sample + EOS scan, no tokenizer round-trip.
//! - `generate/prefix_reuse` — steady-state prefix-block reuse (hash-id cache
//!   warm): copies cached blocks instead of resampling, the KV-cache-reuse case.
//!
//! Run all groups:
//! ```bash
//! cargo bench -p aiperf-runtime --bench prompt_gen
//! ```
//!
//! Setup cost only:
//! ```bash
//! cargo bench -p aiperf-runtime --bench prompt_gen -- setup
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use aiperf_runtime::dataset::{
    CorpusPromptGeneratorFactory, PreparedCorpusPromptGeneratorFactory, PromptGenerator,
    PromptGeneratorFactory, TiktokenTokenizer,
};
use aiperf_runtime::rng::RngRoot;

/// Input sequence lengths (in tokens) swept by the length-parameterized groups.
/// Spans a short chat turn (16) through a large RAG-style context (4096).
const ISL_SWEEP: &[usize] = &[16, 128, 512, 2048, 4096];

/// Build one generator over prepared corpus tokens and a fixed seed. Calling this
/// in setup (not in the timed closure) keeps corpus tokenization out of the
/// per-prompt measurement.
fn generator<'a>(
    prepared: &PreparedCorpusPromptGeneratorFactory,
    tokenizer: &'a TiktokenTokenizer,
) -> Box<dyn PromptGenerator + 'a> {
    prepared
        .create(tokenizer, RngRoot::new(Some(1234)))
        .expect("prepared corpus yields a non-empty generator")
}

/// Cold corpus tokenization: `CorpusPromptGeneratorFactory::prepare` over the
/// embedded Shakespeare sonnet text. This is the cost dataset-load adapters
/// move outside their timed region.
fn bench_prepare_corpus(c: &mut Criterion) {
    let tokenizer = TiktokenTokenizer::builtin();
    let factory = CorpusPromptGeneratorFactory::default();

    c.bench_function("setup/prepare_corpus", |b| {
        b.iter(|| {
            let prepared = factory
                .prepare(black_box(&tokenizer))
                .expect("sonnet corpus tokenizes");
            black_box(prepared);
        });
    });
}

/// Cheap generator construction from tokens already prepared once. Measures
/// only `Arc` clone + RNG/block-cache init, not corpus encode.
fn bench_create_from_prepared(c: &mut Criterion) {
    let tokenizer = TiktokenTokenizer::builtin();
    let prepared = CorpusPromptGeneratorFactory::default()
        .prepare(&tokenizer)
        .expect("sonnet corpus tokenizes");

    c.bench_function("setup/create_from_prepared", |b| {
        b.iter(|| {
            let g = prepared
                .create(black_box(&tokenizer), RngRoot::new(Some(1234)))
                .expect("create from prepared");
            black_box(g);
        });
    });
}

/// Full text path: sample `isl` tokens from the corpus and decode them to a
/// prompt string, across the ISL sweep.
fn bench_generate_text(c: &mut Criterion) {
    let tokenizer = TiktokenTokenizer::builtin();
    let prepared = CorpusPromptGeneratorFactory::default()
        .prepare(&tokenizer)
        .expect("sonnet corpus tokenizes");

    let mut group = c.benchmark_group("generate/tokens");
    for &isl in ISL_SWEEP {
        // Report elements/sec == tokens/sec for the estimate.
        group.throughput(Throughput::Elements(isl as u64));
        group.bench_with_input(BenchmarkId::from_parameter(isl), &isl, |b, &isl| {
            let mut g = generator(&prepared, &tokenizer);
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
    let prepared = CorpusPromptGeneratorFactory::default()
        .prepare(&tokenizer)
        .expect("sonnet corpus tokenizes");

    let mut group = c.benchmark_group("generate_token_ids/tokens");
    for &isl in ISL_SWEEP {
        group.throughput(Throughput::Elements(isl as u64));
        group.bench_with_input(BenchmarkId::from_parameter(isl), &isl, |b, &isl| {
            let mut g = generator(&prepared, &tokenizer);
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
    let prepared = CorpusPromptGeneratorFactory::default()
        .prepare(&tokenizer)
        .expect("sonnet corpus tokenizes");

    const ISL: usize = 2048;
    const BLOCK_SIZE: usize = 64;
    // `isl == n_blocks * block_size` so the final block is full (2048 / 64 = 32).
    let hash_ids: Vec<i64> = (0..(ISL / BLOCK_SIZE) as i64).collect();

    let mut group = c.benchmark_group("generate/prefix_reuse");
    group.throughput(Throughput::Elements(ISL as u64));
    group.bench_function(BenchmarkId::new("block_size", BLOCK_SIZE), |b| {
        let mut g = generator(&prepared, &tokenizer);
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
    bench_prepare_corpus,
    bench_create_from_prepared,
    bench_generate_text,
    bench_generate_token_ids,
    bench_prefix_reuse
);
criterion_main!(benches);
