<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tracker 56 semantic audit 1: RNG and reference stream

## Scope and method

This audit compares every random-number, bounds, and stream-order change in exact
upstream target `94fee7338b2cdddf0d69c10526d9f3f81afa64dd` with native implementation
tree `3af08f8807`. Line references name files as they exist at that exact upstream
commit or native tree. Generated documentation and tests are evidence of the same
behavior and are not counted as additional runtime semantics.

## Behavior inventory

| Upstream behavior | Native equivalent | Executable evidence | Unresolved divergence |
| --- | --- | --- | --- |
| `RandomCorpusStyle` selects vLLM or SGLang semantics (`src/aiperf/common/enums/enums.py:783-816`). | `RandomCorpusStyle` at `rust/runtime/src/dataset/random_range.rs:14-23`. | `style_bounds_and_validation_match_reference_contracts`; CLI style parsing test. | None. |
| vLLM accepts one ratio or independent input/output ratios in `[0,1)` (`src/aiperf/common/models/sequence_distribution.py:686-725`). | `RandomRangeRatioInput` and checked vLLM endpoints at `random_range.rs:25-124`. Strict serde also rejects booleans and unknown object keys. | `ratio_input_rejects_bool_and_unknown_object_fields`; protocol-v2 scalar/split test; CLI/YAML tests. | None. |
| SGLang accepts one common ratio in `[0,1]` and refuses split ratios (`sequence_distribution.py:752-806`). | Style-specific equal-ratio and endpoint checks at `random_range.rs:103-124`. | `style_bounds_and_validation_match_reference_contracts`; Python/native E2E includes ratios 0, 0.5, and 1. | None. |
| vLLM bounds subtract tokenizer special tokens from the input mean, then use inclusive `floor(mean*(1-r))..ceil(mean*(1+r))`; OSL is positive (`sequence_distribution.py:711-719`). | Checked construction at `random_range.rs:136-181`. | Native golden bounds `(68,128)` and `(10,30)` in `rust/runtime/tests/random_range.rs:9-53`; two-special-token E2E case. | None. |
| SGLang bounds are `max(1,floor(mean*r))..mean`, then special tokens are subtracted per draw with a floor of one (`sequence_distribution.py:799-805,1025-1083`). | SGLang bounds and `adjust_input` at `random_range.rs:161-174,213-219`. | Native golden bounds/adjustment and SGLang special-token E2E case. | None. |
| The sequence sampler is a protocol so ratio distributions and mixture distributions share one consumer (`sequence_distribution.py:55-68`; `src/aiperf/dataset/protocols.py:44-67`). | Native ownership is the typed `RandomRangePlan`/`SeededRandomRangePlan` dataset seam, consumed by the existing synthetic composer. | `random_range_ratio_reaches_composed_native_lengths_in_reference_order`. | None; Rust uses a concrete typed seam rather than Python structural typing. |
| vLLM uses NumPy `default_rng`/PCG64 and draws all ISLs, then all OSLs (`sequence_distribution.py:808-866`). | Existing byte-compatible `NumpyGenerator` selected at `random_range.rs:352-386`; loops at `239-248`. | `preseed_matches_numpy_draw_order_for_each_style`; generated Python vectors. | None. |
| SGLang uses a private NumPy `RandomState` MT19937 stream and the same all-ISL→all-OSL order (`sequence_distribution.py:985-1050,1085-1112`). | Private MT19937 plus NumPy bounded-integer rejection sampling at `random_range.rs:343-477`. | Pinned SGLang vectors in `preseed_matches_numpy_draw_order_for_each_style`; Python vector parity. | None. |
| Offset draws follow every ISL and OSL draw and are bounded by tokenizer `vocab_size`, not filtered-pool length (`sequence_distribution.py:841-865`; `src/aiperf/dataset/generator/prompt.py:222-249`). | Offset loop at `random_range.rs:246-247` and retained `vocab_size`; prompt materializer indexes the selected pool separately. | `preseed_matches_numpy_draw_order_for_each_style`; `python_numpy_vectors_match_native_lengths_offsets_and_tokens`. | None. |
| The shared stream continues after offsets for prefix and BPE top-up draws (`sequence_distribution.py:841-866`; `prompt.py:190-249,373-425`). | Authored seed/vocab are retained and `continuation()` replays exactly through offsets at `random_range.rs:260-303`. | `continuation_starts_after_all_lengths_and_offsets`; `reference_prefix_continues_shared_stream_without_consuming_body_ordinal`. | None. |
| SGLang folds a wide seed as `(seed ^ (seed >> 32)) & 0xffffffff` (`src/aiperf/common/random_generator.py:428-447`). | `fold_seed` at `random_range.rs:322-324`. | `sglang_folds_wide_seeds_by_xor_words`; wide-seed E2E case 4,294,967,300. | None. |
| A wide SGLang seed warns once because folds can alias, while vLLM retains the full seed (`sequence_distribution.py:937-983`). | Structured `tracing::warn!` guarded by a once-per-authored-seed set at `random_range.rs:326-340`; PCG64 receives the `u64` unchanged. | Wide-seed integration test exercises the fold; code audit verifies the warning guard and structured fields. | None. |
| Preseed cache exhaustion falls back deterministically and warns once (`sequence_distribution.py:867-895`; `prompt.py:704-723`). | `pair` returns `None`, composer falls back through its seeded worker-local generator, and its existing one-shot exhaustion flag emits the warning. | Prompt/composer fallback unit coverage plus E2E stays within the reference cache. | None; both explicitly end reference parity after exhaustion. |
| Invalid or unrepresentable ranges fail at setup rather than panic or silently wrap (`sequence_distribution.py:686-806`). | Checked finite/endpoints, checked integer conversions, u32 interval limits, and empty-vocab refusal at `random_range.rs:93-124,136-150,365-415`. | `unsupported_reference_bounds_fail_instead_of_panicking`; finite/style boundary integration test. | None. |
| vLLM refuses a minimum total input below one unless a prefix rescues it (`src/aiperf/dataset/composer/base.py:344-396`). | `validate_minimum_input` at `random_range.rs:194-211`. | `vllm_empty_minimum_requires_an_additive_prefix`. | None. |

## Complete upstream target ledger

The RNG-focused inventory above is supplemented by this complete 37-file target ledger;
there are no unaccounted changes outside the audit's detailed focus.

| Exact upstream files/change group | Native equivalent and evidence | Unresolved divergence |
| --- | --- | --- |
| Public enum export and definitions: `src/aiperf/common/enums/__init__.py`, `enums.py:250-260,783-848`. | Native public corpus/style enums and CLI/protocol serde tests. | None. |
| Distribution/RNG: `src/aiperf/common/models/sequence_distribution.py:55-68,686-1112`, `common/random_generator.py:428-447`. | Detailed above: checked plan, PCG64, private MT19937, fold, cache, continuation, and golden/vector/E2E tests. | None. |
| Tokenizer metadata: `src/aiperf/common/tokenizer.py:33-66,830-906`. | Native tokenizer vocabulary/special/allowed-pool APIs; pool unit tests and zero/two-special E2E fixtures. | None. |
| Config and generated schema: `src/aiperf/config/dataset/config.py:673-683`, `dataset/content.py:121-125,201-357`, `flags/_converter_dataset.py:429-465`, `flags/cli_config.py:1420-1465`, `schema/aiperf-config.schema.json`. | Native strict Config-v2, CLI, YAML, and projection; exact upstream Python schema retained by the target merge. | None; native file traces use a defined random materializer instead of Python's absent-corpus refusal. |
| Composition: `src/aiperf/dataset/composer/base.py:78-107,183-206,324-438`, `composer/synthetic.py:103-123,219-257`. | Native special-token compensation, degenerate guard, seeded-pair consumption, token-additive prefix composition; focused composer/engine tests. | None. |
| Generator/protocol plumbing: `src/aiperf/dataset/generator/coding_content.py:28,697`, `generator/corpus.py:10-49`, `generator/prompt.py:82-249,373-548,681-727`, `dataset/protocols.py:19-67`. | One native `PromptMaterializer` trait covers sonnet/coding/random text, raw IDs, and prefixes; detailed prompt tests and vector parity. | None. |
| Mock recorder: `tests/aiperf_mock_server/request_recorder.py:313-328,508-527`. | Rust mock records raw request bytes; E2E additionally normalizes either string or text-part arrays for token checks. | None. |
| RNG/tokenizer upstream tests: `tests/unit/common/models/test_sequence_distribution.py`, `common/test_random_generator.py`, `common/test_tokenizer.py`. | Native range integration, Python-vector parity, tokenizer pool tests, and full E2E. | None. |
| Config upstream tests: `tests/unit/config/test_converter_random_corpus_style.py`, `test_prompt_config_range_ratio.py`. | Native CLI/YAML/protocol/dataset-build tests. | None. |
| Composer/generator upstream tests: `tests/unit/dataset/composer/test_base_composer.py`, `test_degenerate_range_ratio.py`, `test_isl_budget_compensation.py`, `test_synthetic_composer.py`, `tests/unit/dataset/generator/test_prompt_generator.py`. | Native degenerate, special-token, exact-length, pool, prefix, raw-token, stream, fallback, and engine tests. | None. |
| Property upstream tests: `tests/unit/property/test_cli_help_references.py`, `test_finite_invariants.py`. | Native clap help derives from the typed flags; checked finite validation is exercised in range tests. | None. |
| Documentation: `docs/cli-options.md`, `docs/index.yml`, `docs/reference/isl-budget-compensation.md`, `isl-distribution-examples.html`, `isl-example-gpt2.svg`, `isl-example-llama.svg`, `prompt-corpus.md`, `validating-isl-distribution.md`. | Exact target documentation is retained by merge `cd31c0ae5a`; native spec/plan/audits describe the Rust architecture. Docs-current hook passed. | None. |

## Commands and observed outcomes

All commands used `RUSTC_WRAPPER=/usr/bin/sccache` and
`CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-056-target`.

- `cargo test --manifest-path rust/Cargo.toml -p aiperf-runtime --features engine random_range --lib`
  — 6 passed, 0 failed.
- `cargo test --manifest-path rust/Cargo.toml -p aiperf-runtime --test random_range`
  — initial RED: 2 passed, 1 failed because whole-plan equality included the retained
  authored seed; corrected behavioral assertions then GREEN: 3 passed, 0 failed.
- `cargo test --manifest-path rust/Cargo.toml -p aiperf-runtime --test random_range_python_parity`
  — 1 passed, 0 failed.
- `VIRTUAL_ENV=/home/anthony/nvidia/projects/aiperf/ajc/rust/.venv PYTHONPATH=/mnt/4tb/aiperf-origin-port-056/src /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/python tools/generate_random_range_python_vectors.py --check`
  — exit 0, fixture current.

## Audit conclusion

The final rebuilt production E2E also passed 13/13 tests and all 48 ordered captures.
Every upstream RNG algorithm, bound, seed, draw-order, continuation, and boundary
behavior has a native equivalent and concrete evidence. Unresolved divergences: none.
