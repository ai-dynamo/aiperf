<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python/Rust dataset-load comparison harness

## Purpose

Add a small developer benchmark that compares Python and Rust dataset loading
for formats implemented with equivalent semantics in both stacks. The measured
phase is load and composition through a frozen dataset, including tokenization
only for formats whose composition semantics require it. Request scheduling,
transport, export, process startup, fixture generation, tokenizer
initialization, and result serialization are outside the timed region.

The harness is diagnostic rather than a CI performance gate. It verifies output
shape before presenting performance numbers so a faster but semantically
different result is never reported as a valid win.

## Architecture

`dev/benchmarks/dataset_load_compare.py` is the user-facing orchestrator. It:

1. selects the requested formats from an explicit Python/Rust intersection;
2. generates deterministic fixtures or reads a user manifest;
3. starts the Python and Rust adapters, alternating their order between runs;
4. performs warmup and measured iterations;
5. verifies parity invariants; and
6. prints a console summary and writes machine-readable JSON.

The Python adapter calls the existing Python loader and composer APIs. A small
Rust binary calls `aiperf_runtime::dataset::LoaderRegistry` and the registered
loader/composer directly. Each adapter initializes its tokenizer and performs a
warm `encode("warm")` before the measurement. For authored-length trace formats
(`mooncake_trace`, `bailian_trace`, `burst_gpt_trace`), both adapters also
construct or prepare the corpus prompt generator outside the timed region:
Python builds `PromptGenerator` before `perf_counter_ns`, and Rust calls
`CorpusPromptGeneratorFactory::prepare` then injects the prepared factory into
`ComposeConfig.prompt_generator` so timed `compose` only clones prepared
`Arc<[u32]>` tokens. Each adapter then times only dataset load through frozen
composition and emits one JSON result record.

The shared result record contains:

- implementation and dataset format;
- fixture identity and row count;
- conversation and turn counts;
- total input-token count;
- elapsed nanoseconds; and
- a structured error when the format cannot be measured.

Formats without equivalent implementations or fixture semantics are skipped
with an explicit reason. The harness does not infer equivalence from similar
registry names.

## Dataset catalog and inputs

The verified built-in catalog is exactly `single_turn`, `multi_turn`,
`raw_payload`, `inputs_json`, `random_pool`, `mooncake_trace`,
`bailian_trace`, `burst_gpt_trace`, and `sagemaker_data_capture`. The Rust
adapter owns that catalog as `BENCHMARK_FORMATS` in
`dataset_load_bench.rs`: each entry records the Python-canonical name, the
Rust `LoaderRegistry` name (so `burst_gpt_trace` maps to `burst_gpt` only in
the adapter), and whether compose always creates a corpus prompt generator.
No other name similarity implies equivalence.

Built-in generators create deterministic, local fixtures for all nine verified
formats. Random pool uses one literal-text row so output does not depend on
cross-language RNG stream parity. Mooncake uses literal `text_input` rows
without `hash_ids`. Bailian and BurstGPT use authored input lengths; Bailian
rows omit `hash_ids`, while BurstGPT uses the minimal three-column CSV schema.
SageMaker uses JSON-encoded captured message input and omits
`usage.prompt_tokens`, requiring both adapters to tokenize the messages.
`--rows` controls parsed rows except for the deliberately one-row random pool,
and `--tokens-per-row` controls literal prompt size or authored input length.

Python trace loaders receive the same `PromptGenerator` dependency used by
`CustomDatasetComposer._create_loader_instance`; Bailian also receives the
plugin metadata default block size of 16. That constructor tokenizes the
embedded Shakespeare corpus once before timing. Rust mirrors the same untimed
setup with `CorpusPromptGeneratorFactory::prepare` for the three trace formats
above. Token accounting matches Rust `turn.input_tokens`: authored lengths for
Bailian and BurstGPT, literal text for Mooncake and random pool, and
`raw_messages` for SageMaker. `raw_payload` and `inputs_json` preserve ordinary
endpoint request bodies as opaque bytes and report `null` /
`None` input tokens; neither adapter tokenizes their message text.

Rust `RawPayloadComposer` follows the same opaque-body rule in product code.
When the selected endpoint does not set `requires_raw_token_ids`, composition
interns only the exact authored request bytes and leaves `Turn::input_tokens`
as `None`, even if the opaque object happens to contain a `token_ids` member.
When the endpoint does require raw token IDs (for example `vllm_generate`),
composition validates and interns the authored `token_ids`, omits the raw body,
and records `Some(length)` in `Turn::input_tokens`. It never derives token
IDs by BPE-tokenizing text.

Absolute corpus prompt-generator setup cost (cold `prepare` and cheap
`create` from prepared tokens) is measured separately by the Criterion bench:

```bash
source .venv/bin/activate
cd rust
cargo bench -p aiperf-runtime --bench prompt_gen -- setup
```

Filter `setup` selects `setup/prepare_corpus` and `setup/create_from_prepared`.
Per-prompt generation groups continue to exclude setup. Full prompt_gen run:

```bash
cargo bench -p aiperf-runtime --bench prompt_gen
```

`--manifest PATH` accepts real datasets in the same nine-format verified
catalog. Each schema-version-1 manifest entry contains exactly `format`, `path`,
and `options`. Manifest paths bypass fixture generation but use the same
adapters, validation, timing, and reporting. Entries outside the verified
catalog are explicitly skipped rather than measured under unverified semantic
equivalence. Non-empty `options` objects are also skipped with an explicit
reason until a cross-stack option mapping has been proven; both adapters reject
non-empty options early with the same structured error instead of forwarding
unknown kwargs into loaders.

Public and Hugging Face datasets, synthetic datasets, and accuracy datasets
remain explicit skips because equivalent generated local Python/Rust pipelines
have not yet been proven. The harness does not add speculative comparisons for
those categories.

## Measurement and reporting

Defaults are one warmup and five measured iterations. The orchestrator
alternates Python-first and Rust-first order to reduce systematic cache,
frequency, and thermal bias. It retains all samples and summarizes each
implementation using median and p95 elapsed time, rows per second, and input
tokens per second. Rust speedup is Python median divided by Rust median.

The console report is compact and human-readable. The JSON report includes all
raw samples, summaries, command options, platform metadata, Python version,
Rust binary identity, and skipped-format reasons.

Before reporting a speedup, the harness requires the two adapters to agree on
row, conversation, turn, and input-token totals. A mismatch fails that format
and includes both result records in the JSON report. Samples that claim success
with non-positive `elapsed_ns` also fail only that format rather than aborting
report construction. Adapter failures affect only their format unless no
selected format succeeds, in which case the command exits nonzero.

## Command surface

The primary controls are:

- `--formats NAME[,NAME...]` (all equivalent formats by default);
- `--rows N`;
- `--tokens-per-row N`;
- `--warmups N` (default 1);
- `--runs N` (default 5);
- `--manifest PATH`;
- `--output PATH`; and
- `--keep-fixtures`.

Generated fixtures live in a temporary directory and are removed unless
`--keep-fixtures` is set.

From the repository root, run the benchmark with:

```bash
source .venv/bin/activate
.venv/bin/python dev/benchmarks/dataset_load_compare.py \
  --rows 1000 --tokens-per-row 128 \
  --output dataset-load-comparison.json
```

The command always runs an incremental
`cargo build -p aiperf-runtime --release --example dataset_load_bench` before
benchmark execution (outside all timed samples) so a stale release example is
never measured, then runs `.venv/bin/python
dev/benchmarks/dataset_load_python.py` and
`rust/target/release/examples/dataset_load_bench` for each selected format.
Each adapter subprocess is bounded by a timeout; a timeout is recorded as an
adapter error for that format.

## Verification

Unit tests cover deterministic fixture generation, format selection, sample
aggregation, p95 calculation, parity mismatch detection, skipped formats,
adapter errors, and JSON serialization. A small end-to-end smoke test runs both
adapters on all nine generated formats and verifies matching counts plus
positive timing.

The benchmark itself is run manually because timing assertions are unsuitable
for normal CI. Correctness tests assert structure and parity, never that one
implementation is faster.
