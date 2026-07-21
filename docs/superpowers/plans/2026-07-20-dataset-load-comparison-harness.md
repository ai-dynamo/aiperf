<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dataset Load Comparison Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one developer command that measures Python and Rust load → compose performance on equivalent deterministic dataset fixtures and reports validated speedups.

**Architecture:** A Python orchestrator owns fixture generation, repetitions, parity checks, summaries, and JSON output. It invokes a thin Python adapter and a Rust runtime example; both emit the same one-line JSON sample record and time load and composition through dataset freeze, including tokenization only where composition semantics require it.

**Tech Stack:** Python 3.12, pytest, argparse, subprocess, Rust 2024, Tokio current-thread runtime, serde/serde_json, `aiperf-runtime`.

## Global Constraints

- Measure load and compose through dataset freeze; include tokenization only
  where composition semantics require it. Exclude process startup, fixture
  generation, tokenizer initialization, corpus prompt-generator preparation
  for authored-length traces, and result serialization.
- Use the built-in `o200k_base` tokenizer in both implementations.
- Use seed `42` and model `test-model` for generated fixtures.
- A speedup is valid only when row, conversation, turn, and total-input-token counts agree.
- Unsupported or semantically non-equivalent formats must be reported as skipped with a reason.
- Default to one warmup and five measured runs.
- Do not create performance assertions in CI.
- Add NVIDIA SPDX and Apache-2.0 headers to every new source file.

---

### Task 1: Shared fixtures, aggregation, and orchestrator

**Files:**
- Create: `dev/benchmarks/dataset_load_compare.py`
- Create: `tests/unit/dev/test_dataset_load_compare.py`

**Interfaces:**
- Produces: `Sample`, `FormatCase`, `generate_fixtures()`, `summarize_samples()`, `validate_parity()`, and `main()`.
- Invokes adapters with `--format`, `--path`, `--options-json`, `--fixture-id`, `--seed`, and `--model`.
- Consumes one-line JSON records with keys `implementation`, `format`, `fixture_id`, `row_count`, `conversation_count`, `turn_count`, `total_input_tokens`, `elapsed_ns`, and `error`.

- [ ] **Step 1: Write failing unit tests**

Cover deterministic fixture bytes, the generated catalog, nearest-rank p95,
median throughput, parity rejection, skipped-format serialization, alternating
adapter order, and nonzero exit when no format succeeds. Generated fixtures
must include `single_turn`, `multi_turn`, `raw_payload`, `inputs_json`,
`random_pool`, `mooncake_trace`, `bailian_trace`, `burst_gpt_trace`, and
`sagemaker_data_capture`.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
source .venv/bin/activate
pytest tests/unit/dev/test_dataset_load_compare.py -q
```

Expected: collection fails because `dev.benchmarks.dataset_load_compare` does
not exist.

- [ ] **Step 3: Implement the orchestrator**

Use immutable dataclasses:

```python
@dataclass(frozen=True)
class Sample:
    implementation: str
    format: str
    fixture_id: str
    row_count: int
    conversation_count: int
    turn_count: int
    total_input_tokens: int
    elapsed_ns: int
    error: str | None = None

@dataclass(frozen=True)
class FormatCase:
    format: str
    path: Path
    fixture_id: str
    options: dict[str, object]
```

The generated fixtures use exact local shapes:

```python
single_turn = [
    {"text": "alpha beta gamma"},
    {"session_id": "s-a", "text": "turn one"},
    {"session_id": "s-a", "text": "turn two"},
]
multi_turn = [
    {"session_id": "m1", "turns": [{"text": "q1"}, {"text": "q2"}]},
    {"session_id": "m2", "turns": [{"text": "only"}]},
]
raw_payload = [
    {"messages": [{"role": "user", "content": "hi"}], "model": "test-model", "max_tokens": 16},
    {"messages": [{"role": "user", "content": "bye"}], "model": "test-model", "max_tokens": 16},
]
inputs_json = {
    "data": [
        {"session_id": "session-001", "payloads": raw_payload},
        {"session_id": "session-002", "payloads": [raw_payload[0]]},
    ]
}
```

`validate_parity()` compares the four count fields exactly. `summarize_samples()`
uses `statistics.median`; p95 uses sorted nearest-rank index
`ceil(0.95 * len(samples)) - 1`. The orchestrator alternates
`("python", "rust")` and `("rust", "python")`, removes warmup samples, prints a
compact table, and writes a versioned JSON report containing raw samples,
summaries, environment metadata, and skips.

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/unit/dev/test_dataset_load_compare.py -q
```

Expected: all tests pass.

### Task 2: Python timed adapter

**Files:**
- Create: `dev/benchmarks/dataset_load_python.py`
- Create: `tests/unit/dev/test_dataset_load_python.py`

**Interfaces:**
- Consumes the adapter arguments defined in Task 1.
- Produces the exact `Sample` JSON schema defined in Task 1.
- Supports the nine-format generated catalog from Task 1.

- [ ] **Step 1: Write failing adapter tests**

Test each format using a temporary fixture. Assert implementation=`python`,
positive elapsed time, expected row/conversation/turn counts, positive token
count, deterministic counts across two runs, and structured JSON errors for an
unknown format.

- [ ] **Step 2: Verify the tests fail**

```bash
source .venv/bin/activate
pytest tests/unit/dev/test_dataset_load_python.py -q
```

Expected: collection fails because `dataset_load_python.py` does not exist.

- [ ] **Step 3: Implement the adapter**

Initialize `Tokenizer.from_pretrained("builtin")` before the timer. Construct the
existing loader classes with a minimal `BenchmarkRun`, then time
`load_dataset()`, `convert_to_conversations()`, and explicit built-in
tokenization of composed turn content. Do not include `DatasetManager`,
backing-store mmap, request scheduling, or transport setup.

Use one dispatch map:

```python
LOADERS = {
    "single_turn": SingleTurnDatasetLoader,
    "multi_turn": MultiTurnDatasetLoader,
    "raw_payload": RawPayloadDatasetLoader,
    "inputs_json": InputsJsonPayloadLoader,
    "random_pool": RandomPoolDatasetLoader,
    "mooncake_trace": MooncakeTraceDatasetLoader,
    "bailian_trace": BailianTraceDatasetLoader,
    "burst_gpt_trace": BurstGPTTraceDatasetLoader,
    "sagemaker_data_capture": SageMakerDataCaptureLoader,
}
```

Count parsed rows from the loader output before conversion, conversations from
the returned list, and turns from each conversation's `turns`. Construct trace
loaders with the `PromptGenerator` used by
`CustomDatasetComposer._create_loader_instance`, including Bailian's plugin
metadata `default_block_size=16`. Match Rust `turn.input_tokens`: use authored
lengths for Bailian and BurstGPT, tokenize literal text for Mooncake and random
pool, and tokenize `raw_messages` for SageMaker. Catch exceptions at the CLI
boundary and emit a sample record with `error` populated and zero elapsed/count
fields.

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
pytest tests/unit/dev/test_dataset_load_python.py -q
```

Expected: all tests pass.

### Task 3: Rust timed adapter

**Files:**
- Create: `rust/runtime/examples/dataset_load_bench.rs`
- Create: `rust/runtime/tests/dataset_load_bench.rs`

**Interfaces:**
- Consumes the adapter arguments defined in Task 1.
- Produces the exact `Sample` JSON schema defined in Task 1.
- Uses `LoaderRegistry::with_builtin_formats()`, `LoadConfig`,
  `ComposeConfig`, `RngRoot::new(Some(seed))`, and
  `TiktokenTokenizer::builtin()`.

- [ ] **Step 1: Write failing Rust integration tests**

Exercise the public load/compose path for all nine generated formats and assert
positive elapsed/count totals. Add a serialization test for the one-line sample
schema, Python `burst_gpt_trace` to Rust `burst_gpt` alias resolution, and an
unknown-format error test.

- [ ] **Step 2: Verify the tests fail**

```bash
source .venv/bin/activate
cd rust
cargo test -p aiperf-runtime --test dataset_load_bench
```

Expected: compilation fails because the adapter module/example does not exist.

- [ ] **Step 3: Implement the Rust example**

Use `#[tokio::main(flavor = "current_thread")]`. Initialize and warm
`TiktokenTokenizer::builtin()` before `Instant::now()`. For
`mooncake_trace`, `bailian_trace`, and `burst_gpt_trace`, call
`CorpusPromptGeneratorFactory::default().prepare(&tokenizer)` and inject the
prepared factory into `ComposeConfig.prompt_generator` before starting the
timer (matching Python constructing `PromptGenerator` outside timing). Inside
the timed region, resolve the registration, load rows, retain `row_count`,
compose into a `SegmentPool`, freeze via `Dataset::new`, and sum
`turn.input_tokens`.

Absolute corpus setup cost lives in Criterion, not in the harness samples:

```bash
cargo bench -p aiperf-runtime --bench prompt_gen -- setup
```

That filter runs `setup/prepare_corpus` (cold Shakespeare tokenization) and
`setup/create_from_prepared` (cheap generator construction from prepared
tokens). Per-prompt `generate/*` groups continue to exclude setup.

Serialize:

```rust
#[derive(serde::Serialize)]
struct Sample {
    implementation: &'static str,
    format: String,
    fixture_id: String,
    row_count: usize,
    conversation_count: usize,
    turn_count: usize,
    total_input_tokens: u64,
    elapsed_ns: u128,
    error: Option<String>,
}
```

Argument parsing stays dependency-free and rejects missing/unknown arguments
with a structured error record.

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate
cd rust
cargo test -p aiperf-runtime --test dataset_load_bench
```

Expected: all tests pass.

### Task 4: Cross-language integration and documentation

**Files:**
- Modify: `dev/benchmarks/dataset_load_compare.py`
- Modify: `tests/unit/dev/test_dataset_load_compare.py`
- Modify: `docs/superpowers/specs/2026-07-20-dataset-load-comparison-harness-design.md`

**Interfaces:**
- Consumes the two adapters from Tasks 2 and 3.
- Produces a working one-command benchmark and a small end-to-end smoke test.

- [ ] **Step 1: Write the failing cross-language smoke test**

Generate all nine local fixtures, build the Rust example once, run both adapters
for every format, and assert count parity plus positive timing. Mark the test
integration/slow according to existing repository conventions, but do not
assert which implementation is faster.

- [ ] **Step 2: Verify the smoke test fails**

```bash
source .venv/bin/activate
pytest tests/unit/dev/test_dataset_load_compare.py -q -k cross_language
```

Expected: fails until adapter command construction and schema integration are
complete.

- [ ] **Step 3: Complete integration**

Build the Rust adapter once with:

```bash
cargo build -p aiperf-runtime --release --example dataset_load_bench
```

Invoke `.venv/bin/python dev/benchmarks/dataset_load_python.py` and
`rust/target/release/examples/dataset_load_bench`. Add `--manifest` support with
schema version 1 and entries containing `format`, `path`, and `options`.
Document the exact run command and clarify that the built-in generated catalog
is the nine-format semantically verified local subset. Public/Hugging Face,
synthetic, and accuracy formats remain explicit skips because equivalent
generated local Python/Rust pipelines are not yet proven; do not add speculative
comparisons through name similarity.

- [ ] **Step 4: Run focused verification**

```bash
source .venv/bin/activate
pytest tests/unit/dev/test_dataset_load_compare.py tests/unit/dev/test_dataset_load_python.py -q
cd rust
cargo test -p aiperf-runtime --test dataset_load_bench
cargo fmt --check
```

Expected: all tests pass and formatting is clean.

### Task 5: Keep ordinary raw payloads opaque

**Files:**
- Modify: `rust/runtime/src/dataset/loader/raw_payload.rs`
- Modify: `dev/benchmarks/dataset_load_python.py`
- Modify: `tests/unit/dev/test_dataset_load_python.py`
- Modify: `docs/superpowers/specs/2026-07-20-dataset-load-comparison-harness-design.md`

**Interfaces:**
- `RawPayloadComposer::compose` continues to accept the shared
  `TextTokenizer` interface but must not call it for ordinary raw payloads.
- `requires_raw_token_ids=true` continues to require authored `token_ids`,
  intern them as `Payload::TokenIds`, and set `Turn::input_tokens` to their
  length.
- Python benchmark samples report `total_input_tokens=null` for `raw_payload` and
  `inputs_json`, matching Rust product composition.

- [ ] **Step 1: Write failing tests**

Add a tokenizer that fails if `encode` is called, then compose an ordinary raw
payload containing both messages and a `token_ids` member. Assert composition
succeeds, `turn.input_tokens == 0`, the body contains exactly one raw segment,
and no token-ID segment was interned. Retain the existing token-native test as
coverage that `requires_raw_token_ids=true` records authored token count.

In `test_dataset_load_python.py`, change the raw-format expectation to:

```python
expected_tokens = 0 if format_name in {"raw_payload", "inputs_json"} else None
if expected_tokens is not None:
    assert first.total_input_tokens == expected_tokens
else:
    assert first.total_input_tokens > 0
```

- [ ] **Step 2: Verify tests fail for the missing behavior**

```bash
source .venv/bin/activate
cd rust
cargo test -p aiperf-runtime raw_payload_does_not_tokenize_opaque_body --lib
cd ..
pytest tests/unit/dev/test_dataset_load_python.py -q -k deterministic_counts
```

Expected: Rust fails because the ordinary composer calls `encode`; Python fails
because both raw formats still report positive token totals.

- [ ] **Step 3: Implement opaque raw composition**

Remove the parallel BPE pre-pass and `raw_input_tokens`. In the ordinary branch,
intern only `row.wire`, set `body` from that raw handle, and leave
`input_tokens` as `None`. Parse, validate, intern, and count `token_ids` only
inside the `requires_raw_token_ids` branch as `Some(length)`.

In the Python adapter, short-circuit raw-format accounting:

```python
if format_name in {"raw_payload", "inputs_json"}:
    return None
```

- [ ] **Step 4: Run focused verification and the benchmark**

```bash
source .venv/bin/activate
cd rust
cargo test -p aiperf-runtime raw_payload --lib
cargo test -p aiperf-runtime --test dataset_load_bench
cargo fmt --check
cd ..
pytest tests/unit/dev/test_dataset_load_python.py tests/unit/dev/test_dataset_load_compare.py -q
.venv/bin/python dev/benchmarks/dataset_load_compare.py \
  --formats raw_payload,inputs_json --warmups 1 --runs 5 \
  --output /tmp/dataset-load-opaque-raw.json
```

Expected: tests and formatting pass, cross-language token totals are zero, and
both formats produce validated benchmark summaries.
