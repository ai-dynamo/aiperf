<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tracker 56 semantic audit 2: config, dataset, formatter, and prefix semantics

## Scope and method

This audit compares the authored configuration, tokenizer metadata, dataset composition,
prompt generation, and formatter-facing changes in upstream `94fee7338b` with native
implementation tree `3af08f8807`.

## Behavior inventory

| Upstream behavior | Native equivalent | Executable evidence | Unresolved divergence |
| --- | --- | --- | --- |
| `PromptCorpus.RANDOM` is a public corpus value (`src/aiperf/common/enums/enums.py:250-260`). | Native corpus parser/factory accepts `random` and selects `CorpusPromptGenerator::random*`. | CLI `prompt_corpus_flag_parses_supported_values`; protocol-v2 and YAML corpus tests; native dataset-build tests. | None. |
| Config exposes `random_range_ratio` and `random_corpus_style`, defaulting style to vLLM (`src/aiperf/config/dataset/content.py:121-125,224-246`). | Strict serde fields on native synthetic prompt input and defaulted `RandomCorpusStyle::Vllm`. | `synthetic_random_range_ratio_decodes_scalar_and_split_protocol_v2`; YAML authoring test. | None. |
| Ratio mode requires authored fixed ISL/OSL, rejects nonzero stddev and sequence distribution, and validates style/ratio shape before execution (`content.py:258-309`). | Native projection checks fixed distributions, exclusivity, required means, and checked ratio policy before dataset composition. | Engine dataset-build range tests and CLI/YAML/protocol tests. | None. |
| CLI accepts scalar or JSON-object ratio and style; style must survive even without a ratio (`src/aiperf/config/flags/cli_config.py:1420-1465`; `_converter_dataset.py:429-465`). | `parse_random_range_ratio` at `rust/cli/src/load.rs:649-660`; style is projected independently. | `random_range_ratio_and_corpus_style_flags_parse`; `random_style_selects_pool_without_reference_offsets`. | None. |
| YAML and generated schema/help expose the new enum/fields (`content.py:121-125`; generated schema and `docs/cli-options.md` in the target delta). | Native Config-v2 YAML DTO and clap help expose the fields; exact upstream Python docs/schema are retained by the target-only merge. | `synthetic_random_range_ratio_and_style_are_yaml_authorable`; CLI parser test. | None. |
| Random file-trace selection is rejected in Python because its hash materializer has no text corpus (`src/aiperf/config/dataset/config.py:673-683`). | Native file traces have an explicit random hash/count materializer, so `random` is projected to that supported implementation and never hashes into an absent corpus. | `prompt_corpus_flag_projects_file_dataset_prompts`; file-trace corpus dataset-build tests. | None; this is a native product adaptation satisfying the failure-avoidance invariant with a defined superset rather than a Python-only refusal. |
| Tokenizer exposes real vocabulary size, special IDs, valid non-special IDs, all token IDs, and server-added prompt-special-token count, including sparse tiktoken handling (`src/aiperf/common/tokenizer.py:33-66,810-906`). | `TextTokenizer` already exposes vocab size, special IDs, allowed random IDs, and `num_special_tokens_to_add`; HF/tiktoken implementations provide checked pools. | Allowed-pool/raw-token prompt tests; zero/two-special checked-in tokenizer E2E fixtures. | None. |
| vLLM random bodies use the valid-token pool; SGLang uses the full dense ID space (`src/aiperf/dataset/generator/prompt.py:154-186`). | Style selects allowed or dense pool in native prompt materializer. | `reference_random_offsets_add_request_ordinal_and_style_selects_pool`; `random_style_selects_pool_without_reference_offsets`. | None. |
| Request `i` emits `pool[(offset+i+j) % pool.len()]` (`prompt.py:681-727`). | Native indexed construction uses the same offset, request ordinal, and token ordinal. | Native offset/style test and generated Python token-vector parity. | None. |
| Decode→encode repair uses the re-encoded sequence, trims or tops up, and has a 10-attempt budget (`prompt.py:484-548`). | Native text materializer uses the re-encoded result and `RANDOM_TEXT_REPAIR_ATTEMPTS = 10` at `rust/runtime/src/dataset/prompt.rs:530,714-749`. | Existing exact-length repair tests; all production E2E bodies re-tokenize identically. | None. |
| Top-up tokens come from the full vocabulary and consume the continuation stream without advancing request ordinal or offset cache (`prompt.py:190-220`). | Native reference stream is retained in the prompt generator; repair top-ups draw full-vocab indices while body ordinal is advanced once. | `reference_prefix_continues_shared_stream_without_consuming_body_ordinal`; Python vector and E2E parity. | None. |
| Prefix pools initialize after preseed, draw from the continuation stream, do not consume body request ordinals, and are token-additive (`prompt.py:373-425`; `src/aiperf/dataset/composer/synthetic.py:103-123`). | Native builds reference prefixes from `SeededRandomRangePlan::continuation`, assembles prefix+body IDs before one decode, and leaves body ordinal unchanged. | `reference_prefix_continues_shared_stream_without_consuming_body_ordinal`; `reusable_prefix_is_a_shared_parent_of_the_first_turn_only`; raw prefix tests. | None. |
| Prefix composition does not inject a separator token, and total requested ISL remains exact (`tests/unit/dataset/generator/test_prompt_generator.py:1190-1255`). | Native concatenates token vectors, not strings, then decodes once. | Text and raw prefix composition tests; exact-length tests. | None. |
| A zero random body is allowed only when a nonempty prefix supplies the request; negative/empty total input is rejected (`src/aiperf/dataset/composer/synthetic.py:219-257`; prompt tests at 1111-1121). | Native range guard accepts a rescuing prefix and the materializer refuses an empty final prompt. | `vllm_empty_minimum_requires_an_additive_prefix`; prefix-only composer coverage. | None. |
| Server-added special tokens are subtracted for independently sampled non-range ISL as well as ratio mode; BOS is not double-counted in chat-template overhead (`src/aiperf/dataset/composer/base.py:78-107,397-438`). | `SyntheticPromptConfig::input_token_subtraction` is set from the tokenizer only for non-ratio sampling at `dataset_build.rs:525-536`, and applied after the draw at `loader/synthetic.rs:267-284`; ratio modes retain style-specific compensation. | `independently_sampled_isl_subtracts_server_special_tokens`; special-token E2E cases. | None. |
| Synthetic composer preseeds before prefix initialization and caches paired ISL/OSL per turn (`synthetic.py:103-123,219-257`). | Dataset build creates one seeded plan and composer consumes its paired vectors in ordinal order before prompt materialization. | `random_range_ratio_reaches_composed_native_lengths_in_reference_order`; production E2E sequential profiles. | None. |
| Coding generator and corpus generator protocol gain the prefix/random-compatible method surface (`src/aiperf/dataset/generator/coding_content.py`; `corpus.py`; `src/aiperf/dataset/protocols.py`). | Native `PromptMaterializer` already owns the equivalent text/raw/prefix methods for sonnet, coding, and random factories. | Existing coding/sonnet tests plus random prefix/raw tests. | None; Rust uses one trait rather than duplicating Python protocols. |
| Request formatter receives the composed prompt through the ordinary endpoint body plan; random mode adds no transport-specific branch. | Native composition yields normal conversation turns/token IDs; existing chat endpoint body plan serializes them. | Full production capture compares exact outbound bytes and contract-bearing method/route/content-type. | None. |
| Mock request recording normalizes text-part arrays before tokenizer chat-template application (`tests/aiperf_mock_server/request_recorder.py:313-328,508-527`). | Rust mock captures the raw body before parsing, and the E2E test independently reads both string and text-part-array content when re-tokenizing. | `prompt_text` at `rust/e2e-tests/tests/test_random_range_e2e_parity.rs:67-86`; 48 exact captures. | None; raw byte capture is stronger than normalized-only recording. |
| New upstream reference pages, diagrams, help links, finite-invariant tests, and schema describe/protect these behaviors. | Exact target documentation/schema/tests remain in the second-parent merge; native spec, Sol plan, three audits, and Rust executable tests establish native behavior. | Exact ancestry `cd31c0ae5a`; docs-current hook passed. | None. |

## Complete upstream target ledger

The detailed config/dataset analysis above covers most of the target directly. This
ledger independently accounts for every one of its 37 changed files.

| Exact upstream files/change group | Native equivalent and evidence | Unresolved divergence |
| --- | --- | --- |
| Public enum export/definitions: `src/aiperf/common/enums/__init__.py`, `enums.py:250-260,783-848`. | Native corpus/style enums, serde, and CLI/protocol tests. | None. |
| Distribution/RNG: `src/aiperf/common/models/sequence_distribution.py:55-68,686-1112`, `common/random_generator.py:428-447`. | Native checked range policy, PCG64/MT19937 streams, wide-seed fold, continuation; golden and Python-vector tests. | None. |
| Tokenizer: `src/aiperf/common/tokenizer.py:33-66,830-906`. | Native vocab/special/allowed APIs and checked-in special-token E2E fixtures. | None. |
| Config/schema: `src/aiperf/config/dataset/config.py:673-683`, `dataset/content.py:121-125,201-357`, `flags/_converter_dataset.py:429-465`, `flags/cli_config.py:1420-1465`, `schema/aiperf-config.schema.json`. | Detailed above; CLI/YAML/protocol/projection tests and exact target schema ancestry. | None. |
| Composers: `src/aiperf/dataset/composer/base.py:78-107,183-206,324-438`, `composer/synthetic.py:103-123,219-257`. | Detailed above; native special compensation, bounds, preseed order, prefix and engine tests. | None. |
| Generators/protocol: `src/aiperf/dataset/generator/coding_content.py:28,697`, `generator/corpus.py:10-49`, `generator/prompt.py:82-249,373-548,681-727`, `dataset/protocols.py:19-67`. | Native `PromptMaterializer` and sonnet/coding/random factories, including text/raw/prefix behavior. | None. |
| Mock recorder: `tests/aiperf_mock_server/request_recorder.py:313-328,508-527`. | Raw server-side Rust capture plus independent string/text-part token extraction. | None. |
| Common tests: `tests/unit/common/models/test_sequence_distribution.py`, `common/test_random_generator.py`, `common/test_tokenizer.py`. | Native golden bounds/streams/fold, generated Python vectors, tokenizer/pool tests. | None. |
| Config tests: `tests/unit/config/test_converter_random_corpus_style.py`, `test_prompt_config_range_ratio.py`. | Native CLI/YAML/protocol/dataset-build surface tests. | None. |
| Composer/generator tests: `tests/unit/dataset/composer/test_base_composer.py`, `test_degenerate_range_ratio.py`, `test_isl_budget_compensation.py`, `test_synthetic_composer.py`, `tests/unit/dataset/generator/test_prompt_generator.py`. | Native special-token, degenerate, prefix, pool, raw-token, repair, continuation, fallback, and composed-dataset tests. | None. |
| Property tests: `tests/unit/property/test_cli_help_references.py`, `test_finite_invariants.py`. | Typed clap help and checked finite ratio tests. | None. |
| Eight docs assets: `docs/cli-options.md`, `docs/index.yml`, `docs/reference/isl-budget-compensation.md`, `isl-distribution-examples.html`, `isl-example-gpt2.svg`, `isl-example-llama.svg`, `prompt-corpus.md`, `validating-isl-distribution.md`. | Exact target delta retained by the two-parent merge; native spec/plan/audits added separately. | None. |

## Commands and observed outcomes

All Cargo commands used sccache and the isolated target.

- `cargo test --manifest-path rust/Cargo.toml -p aiperf-cli random_range` — 2 passed,
  0 failed (ratio/style CLI and YAML).
- `cargo test --manifest-path rust/Cargo.toml -p aiperf-cli prompt_corpus` — 7 passed,
  0 failed (CLI/YAML/projection across dataset forms).
- `cargo test --manifest-path rust/Cargo.toml -p aiperf-runtime --features engine random_range --lib`
  — 6 passed, 0 failed.
- `cargo test --manifest-path rust/Cargo.toml -p aiperf-runtime independently_sampled_isl_subtracts_server_special_tokens --lib`
  — 1 passed, 0 failed.
- `cargo test --manifest-path rust/Cargo.toml -p aiperf-runtime --test random_range_python_parity`
  — 1 passed, 0 failed.

## Audit conclusion

The final rebuilt production E2E passed 13/13 tests and all 48 captures after the final
implementation commit. Every upstream config, tokenizer, dataset, prefix, repair, and
formatter-facing semantic has a tested native equivalent. Unresolved divergences: none.
