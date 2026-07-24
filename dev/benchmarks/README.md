# Dataset Load Benchmarks

This directory holds the Python/Rust dataset-load parity harness and its adapter
scripts:

- `dataset_load_compare.py`: cross-language harness that generates or reads the
  same fixture catalog, runs the Python and Rust adapters in alternating order,
  checks semantic parity, and writes a machine-readable report.
- `dataset_load_python.py`: standalone Python adapter used by the harness.
- `rust/runtime/examples/dataset_load_bench.rs`: standalone Rust adapter used by
  the harness.

These tools are for development-time parity and performance work. They are not
part of the public `aiperf` CLI surface.

## Environment

Run the harness from the repository root with the project environment active:

```bash
source .venv/bin/activate
PYTHONPATH=. python dev/benchmarks/dataset_load_compare.py --formats synthetic
```

`PYTHONPATH=.` is required because the script imports `dev.benchmarks.*`
modules directly from the repository tree.

## Shared Tokenizer Modes

`dataset_load_compare.py` forwards the same `--tokenizer` string to both
adapters, so use a tokenizer spec both sides understand.

Safe cross-language forms:

- `builtin`: zero-network `o200k_base` tokenizer on both sides.
- Explicit tiktoken encoding names:
  `o200k_base`, `o200k_harmony`, `cl100k_base`, `p50k_base`, `p50k_edit`,
  `r50k_base`.
- Hugging Face repository IDs, for example
  `HuggingFaceTB/SmolLM2-135M-Instruct`.
- A local tokenizer directory that contains `tokenizer.json` (and, for
  template-aware counting, `tokenizer_config.json` and/or
  `chat_template.jinja`).

Rust's standalone adapter accepts a few extra debugging-only forms, such as an
explicit `tokenizer.json` file path or a native `tiktoken.model` /
`*.tiktoken` file or directory. Do not use those extra Rust-only forms through
`dataset_load_compare.py`, because the Python adapter will not interpret them
the same way.

## Token Counting Modes

The benchmark can measure several different notions of "input token count". The
choice matters both for parity and for measured throughput.

### Default mode

By default the harness tries to use already-authored counts when they are part
of the dataset contract.

- Synthetic formats prefer the stored authoritative `turn.input_tokens`
  produced at composition time.
- Formats whose inputs are intentionally opaque at benchmark time, such as
  `raw_payload` and `inputs_json`, may report `total_input_tokens = None`.

This is the cheapest mode and is the right baseline when you want to measure
load/compose performance without adding extra recount work.

### `--apply-chat-template`

`--apply-chat-template` asks both adapters to count chat-shaped payloads through
the Hugging Face tokenizer's chat template using the equivalent of:

```python
apply_chat_template(tokenize=True, add_generation_prompt=True)
```

Semantics:

- Applies only to payloads that expose a chat-style message array.
- Counts template-added wrappers such as role headers, BOS/EOT markers, and the
  generation-prompt suffix.
- Counts top-level tool schema text outside the role/content template, matching
  the runtime record-parser behavior.
- Falls back to bare-text counting if the tokenizer has no chat template or the
  template cannot be rendered.

Use this when you want the benchmark to reflect the same template-aware ISL that
the runtime uses for chat-shape payloads.

### `--exact-isl`

`--exact-isl` is a synthetic-format validation knob.

For `synthetic` and `synthetic_rankings` it disables the fast stored-count path
and forces the adapters to re-tokenize the final rendered text payload instead.
That is useful when you want to validate the exact token count of the text that
survives decode/content-fitting, not just the authoritative token sequence that
generated it.

Notes:

- It is intentionally slower than the default synthetic path.
- It is mainly useful for exactness checks and regressions around
  BPE-merge/content-fitting behavior.
- It does not change non-synthetic formats today.

### Real tokenizer recounts

If you request either:

- a non-`builtin` tokenizer, or
- `--apply-chat-template`

the adapters recount from extracted payload text instead of blindly trusting the
cheap stored-count path. That is deliberate: it lets the benchmark include the
real tokenizer/template cost that a user-visible ISL path would pay.

## Synthetic Parity Default

For `inline_synthetic` sources, `dataset_load_compare.py` sets
`AIPERF_RNG_BACKEND=python` for both adapters unless you already exported
`AIPERF_RNG_BACKEND` yourself. This keeps corpus sampling, prefix reuse, and
other synthetic RNG draws aligned across Python and Rust during parity runs.

If you want to benchmark a different RNG backend intentionally, export
`AIPERF_RNG_BACKEND` before launching the harness.

## Common Commands

Baseline synthetic comparison with the default builtin tokenizer:

```bash
source .venv/bin/activate
PYTHONPATH=. python dev/benchmarks/dataset_load_compare.py \
  --formats synthetic \
  --rows 4 \
  --tokens-per-row 128 \
  --warmups 1 \
  --runs 5
```

Template-aware synthetic comparison with a real Hugging Face tokenizer:

```bash
source .venv/bin/activate
PYTHONPATH=. python dev/benchmarks/dataset_load_compare.py \
  --formats synthetic \
  --rows 4 \
  --tokens-per-row 128 \
  --warmups 0 \
  --runs 3 \
  --tokenizer HuggingFaceTB/SmolLM2-135M-Instruct \
  --apply-chat-template
```

Synthetic exactness check that recounts final rendered text:

```bash
source .venv/bin/activate
PYTHONPATH=. python dev/benchmarks/dataset_load_compare.py \
  --formats synthetic,synthetic_rankings \
  --rows 4 \
  --tokens-per-row 128 \
  --warmups 0 \
  --runs 1 \
  --exact-isl
```

## Standalone Adapter Debugging

Python adapter:

```bash
source .venv/bin/activate
PYTHONPATH=. python dev/benchmarks/dataset_load_python.py \
  --format synthetic \
  --path '' \
  --options-json '{}' \
  --source-json '{"kind":"inline_synthetic","inline":{"marker":"__aiperf_synthetic","synthetic_config":{"entries":2,"turns":1,"prompts":{"input_tokens":12,"output_tokens":8}}}}' \
  --fixture-id synthetic-py \
  --seed 42 \
  --model test-model \
  --tokenizer HuggingFaceTB/SmolLM2-135M-Instruct \
  --apply-chat-template
```

Rust adapter:

```bash
source .venv/bin/activate
cd rust
cargo run -p aiperf-runtime --example dataset_load_bench --release -- \
  --format synthetic \
  --path '' \
  --options-json '{}' \
  --source-json '{"kind":"inline_synthetic","inline":{"marker":"__aiperf_synthetic","synthetic_config":{"entries":2,"turns":1,"prompts":{"input_tokens":12,"output_tokens":8}}}}' \
  --fixture-id synthetic-rust \
  --seed 42 \
  --model test-model \
  --tokenizer HuggingFaceTB/SmolLM2-135M-Instruct \
  --apply-chat-template
```

Standalone adapter output is always one JSON line matching the shared `Sample`
schema used by the harness.
