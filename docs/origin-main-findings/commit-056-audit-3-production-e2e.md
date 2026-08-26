<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tracker 56 semantic audit 3: Python-to-native production E2E

## Scope and proof boundary

Upstream `94fee7338b` changes the request population produced by the Python profile path.
Formatter-local DTO equality cannot prove parity because it bypasses configuration,
projection, dataset composition, endpoint formatting, and transport. The required proof
therefore launches both products against the same deterministic Rust mock and compares
what the server actually received. Native implementation tree is `3af08f8807`.

## Full-flow inventory

| Upstream production behavior | Native/E2E equivalent | Evidence | Unresolved divergence |
| --- | --- | --- | --- |
| Authored Config v2 selects `corpus: random`, ratio, style, seed, fixed means, and tokenizer (`src/aiperf/config/dataset/content.py:201-246,258-357`). | Each case writes one real Config-v2 YAML and passes it to both profile processes (`rust/e2e-tests/tests/test_random_range_e2e_parity.rs:88-131`). | Both subprocesses must exit successfully before captures are inspected. | None. |
| Python `aiperf profile` runs the Python engine and native `aiperf profile` runs Config parsing→projection→dataset build→prompt generation→body plan→HTTP transport. | Harness sets `AIPERF_RUNTIME_ENGINE=python` only for the first run (`test_random_range_e2e_parity.rs:133-160`), clears captures, then invokes the built native binary normally (`162-176`). | The binary is rebuilt from the final implementation immediately before the gate. | None. |
| vLLM uses PCG64, accepts scalar/split ratios below one, excludes special IDs, and applies pre-bound special compensation (`sequence_distribution.py:686-725,808-895`; `prompt.py:154-186`). | Matrix includes seed 0 ratio 0, seed 42 split ratios, and near-boundary 0.899999 with two special tokens (`test_random_range_e2e_parity.rs:223-249`). | 24 ordered vLLM request captures compare exactly. | None. |
| SGLang uses private RandomState MT19937, scalar ratios including one, full vocab, per-sample special subtraction, and wide-seed fold (`sequence_distribution.py:752-806,958-1112`). | Matrix includes seed 0 ratio 0, seed 42 ratio 0.5 with two specials, and wide folded seed 4,294,967,300 at ratio 1 (`test_random_range_e2e_parity.rs:250-276`). | 24 ordered SGLang request captures compare exactly. | None. |
| Each case draws eight sequential requests, preserving all-ISL→all-OSL→offset order and request ordinal in emitted prompt content. | `REQUESTS = 8`, one worker, concurrency one, sequential dataset (`test_random_range_e2e_parity.rs:15,98-123`). | Capture order is compared without sorting. | None. |
| Production endpoint contract is POST `/v1/chat/completions` with JSON content type and the endpoint formatter's actual body. | Mock records request method, route, headers, and raw body; the test filters only the contract route (`59-65`). | Per-request method, route, content-type, and body assertions at lines 178-209. | None. |
| Python and native prompt token IDs must match after real formatter serialization, including text-part-array normalization (`tests/aiperf_mock_server/request_recorder.py:313-328,508-527`). | Test extracts the received prompt from either string or text-part arrays and re-encodes with the same checked-in tokenizer (`67-86,178-216`). | Per-request emitted token-ID equality, in addition to raw byte equality. | None. |
| Tokenizer special-token behavior must be reproducible without a network/cache dependency (`src/aiperf/common/tokenizer.py:810-906`). | Two complete checked-in Hugging Face tokenizer directories model zero and two auto-added special tokens (`27-57`); offline environment is explicit (`133-145`). | All six cases run without network access. | None. |
| The mock's response behavior must not alter request generation or ordering. | One in-process fast no-tokenizer mock is shared for Python then native in each case, with capacity for both runs (`88-93`). | Exactly eight captures are required for each process before comparison (`153-176`). | None. |

## Complete upstream target ledger

The production-flow proof above is complemented by a complete accounting of the exact
37-file upstream delta, viewed through externally observable output.

| Exact upstream files/change group | Production-path native proof | Unresolved divergence |
| --- | --- | --- |
| Enum export/definitions: `src/aiperf/common/enums/__init__.py`, `enums.py:250-260,783-848`. | Real YAML parses `random`, `vllm`, and `sglang` in both products before traffic. | None. |
| Distribution/RNG: `src/aiperf/common/models/sequence_distribution.py:55-68,686-1112`, `common/random_generator.py:428-447`. | Six cases distinguish PCG64/MT19937, scalar/split/boundary ratios, special adjustment, and wide-seed fold across 48 ordered bodies. Golden/vector tests isolate the streams. | None. |
| Tokenizer: `src/aiperf/common/tokenizer.py:33-66,830-906`. | Two network-independent tokenizers exercise zero/two auto-special counts; every received prompt is re-tokenized. | None. |
| Config/schema: `src/aiperf/config/dataset/config.py:673-683`, `dataset/content.py:121-125,201-357`, `flags/_converter_dataset.py:429-465`, `flags/cli_config.py:1420-1465`, `schema/aiperf-config.schema.json`. | Both real profiles parse equivalent Config-v2 YAML; separate CLI/YAML/protocol tests cover authored variants and refusal boundaries. | None; native file-trace random is a defined supported materializer. |
| Composers: `src/aiperf/dataset/composer/base.py:78-107,183-206,324-438`, `composer/synthetic.py:103-123,219-257`. | Captured bodies/OSL fields prove paired lengths, special compensation, and ordinary body planning; focused prefix/degenerate tests cover non-matrix boundaries. | None. |
| Generators/protocol: `src/aiperf/dataset/generator/coding_content.py:28,697`, `generator/corpus.py:10-49`, `generator/prompt.py:82-249,373-548,681-727`, `dataset/protocols.py:19-67`. | Exact captured bytes and token IDs prove random generator selection, pool/index arithmetic, and repair. Native trait tests cover coding and prefix surface parity. | None. |
| Mock recorder: `tests/aiperf_mock_server/request_recorder.py:313-328,508-527`. | Rust server captures raw bytes before parsing; E2E token extraction handles both content representations. | None. |
| Common tests: `tests/unit/common/models/test_sequence_distribution.py`, `common/test_random_generator.py`, `common/test_tokenizer.py`. | Native golden/vector/tokenizer gates plus production captures cover their behavioral assertions. | None. |
| Config tests: `tests/unit/config/test_converter_random_corpus_style.py`, `test_prompt_config_range_ratio.py`. | Native CLI/YAML/protocol tests cover the same surface before E2E. | None. |
| Composer/generator tests: `tests/unit/dataset/composer/test_base_composer.py`, `test_degenerate_range_ratio.py`, `test_isl_budget_compensation.py`, `test_synthetic_composer.py`, `tests/unit/dataset/generator/test_prompt_generator.py`. | Native unit/integration tests cover prefix, repair, raw IDs, pools, compensation, continuation, fallback, and degeneracy; E2E proves their production composition. | None. |
| Property tests: `tests/unit/property/test_cli_help_references.py`, `test_finite_invariants.py`. | Typed clap surface and finite validation pass focused gates; invalid values cannot reach transport. | None. |
| Eight docs assets: `docs/cli-options.md`, `docs/index.yml`, `docs/reference/isl-budget-compensation.md`, `isl-distribution-examples.html`, `isl-example-gpt2.svg`, `isl-example-llama.svg`, `prompt-corpus.md`, `validating-isl-distribution.md`. | Exact target assets retained by merge `cd31c0ae5a`; native design/plan/audits map the Rust product. | None. |

## Exact command and observed receipt

The native binary was first rebuilt from `3af08f8807`:

```text
RUSTC_WRAPPER=/usr/bin/sccache \
CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-056-target \
cargo build --manifest-path rust/Cargo.toml -p aiperf-cli
```

Observed: exit 0, dev binary rebuilt.

Then the mandatory production gate ran:

```text
RUSTC_WRAPPER=/usr/bin/sccache \
CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-056-target \
VIRTUAL_ENV=/home/anthony/nvidia/projects/aiperf/ajc/rust/.venv \
PYTHONPATH=/mnt/4tb/aiperf-origin-port-056/src \
AIPERF_E2E_BIN=/mnt/4tb/aiperf-origin-port-056-target/debug/aiperf \
cargo test --manifest-path rust/Cargo.toml -p aiperf-e2e-tests \
  --test test_random_range_e2e_parity -- --nocapture
```

Observed: 13 passed, 0 failed. The named parity test passed all six matrix cases × eight
requests = 48 ordered captures. Every capture matched method, route, content-type, exact
outbound UTF-8 body bytes, and re-tokenized prompt IDs.

## Audit conclusion

This is production-path A/B evidence, not a test-local reconstruction. Every upstream
Python random-range behavior reaches the native transport with byte-exact observable
output across the required matrix. Unresolved divergences: none.
