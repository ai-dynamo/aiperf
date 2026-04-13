# Design: `aiperf anonymize-trace`

Privacy-preserving trace anonymization for shareable LLM benchmarks.

## Problem

Inference providers and enterprises want to share realistic LLM traces for benchmarking but cannot expose actual prompt content. Real production logs contain sensitive user data, proprietary system prompts, and PII. Synthetic benchmarks lack the workload characteristics (prefix sharing, ISL/OSL distributions, request timing) that make production traces valuable.

AIPerf already supports replaying Mooncake traces with `hash_ids` for prefix-aware benchmarking, and has a `RollingHasher` that produces stable block hashes. But there is no CLI command to create these traces from raw conversation logs. Users must write custom scripts (like the block_parse.py proof-of-concept) to bridge the gap.

## Solution

Add an `aiperf anonymize-trace` CLI command that converts raw OpenAI-compatible chat logs into privacy-preserving Mooncake traces. The command tokenizes conversations using the target model's tokenizer and chat template, hashes token blocks via the existing `RollingHasher`, and emits JSONL that `aiperf profile` can directly replay.

## Input Format

JSONL where each line is a conversation record:

```jsonl
{"timestamp": 0, "messages": [{"role": "user", "content": "What is the capital of France?"}], "output": "The capital of France is Paris."}
{"timestamp": 100, "session_id": "sess_1", "messages": [{"role": "user", "content": "Explain ML"}], "output": "Machine learning is..."}
{"timestamp": 200, "session_id": "sess_1", "messages": [{"role": "user", "content": "Give an example"}], "output": "For instance..."}
```

### Required Fields

- `messages` — OpenAI-compatible message array. Each message must have `role` and `content` keys. At least one message required.
- `output` — Assistant response text. Used only for `output_length` token counting; stripped from output.

### Optional Fields

- `timestamp` — Milliseconds since trace start. Passed through to output for `--fixed-schedule` replay. If absent, a warning is emitted once after processing: "No timestamps found in input. The output trace will not support --fixed-schedule replay. Consider adding timestamps or using --request-rate during replay."
- `session_id` — String grouping lines into multi-turn conversations. If absent, each line is treated as independent (single-turn).

## Output Format

Standard Mooncake trace JSONL, directly consumable by `aiperf profile --custom-dataset-type mooncake_trace`:

```jsonl
{"timestamp": 0, "input_length": 12, "output_length": 30, "hash_ids": [1, 2, 3]}
{"timestamp": 100, "input_length": 14, "output_length": 46, "hash_ids": [4, 5, 6, 7], "session_id": "sess_1"}
{"timestamp": 200, "input_length": 21, "output_length": 12, "hash_ids": [4, 5, 8, 9, 10], "session_id": "sess_1"}
```

### Fields

- `timestamp` — Passed through from input. Omitted if not present in input.
- `input_length` — Token count after applying chat template and tokenizing the full message array.
- `output_length` — Token count of the output text.
- `hash_ids` — Block hash sequence from `RollingHasher`.
- `session_id` — Passed through from input if present.

### Privacy Guarantee

All actual text content is stripped. Only token counts and hash patterns survive. Hash IDs are consecutive integers assigned by insertion order — they cannot be reversed to recover token content.

**Shared:** request timestamps, token counts, hash ID sequences, session grouping, prefix cache hit patterns.

**Protected:** actual prompt text, token IDs, assistant responses, user information, proprietary system prompts.

## CLI Interface

```
aiperf anonymize-trace \
  --input-file raw_logs.jsonl \
  --output-file anonymized_trace.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --block-size 512
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-file` | Yes | — | Path to input JSONL with raw conversation logs |
| `--output-file` | No | `<input_stem>_anonymized.jsonl` | Path to output Mooncake trace JSONL |
| `--model` | Yes | — | HuggingFace model name for tokenizer and chat template |
| `--block-size` | No | 512 | Tokens per block for hashing (matches `analyze-trace` default) |

### `--model` Is the Target Model

The `--model` argument specifies the model you intend to benchmark against, not the model that generated the original logs. This is intentional:

- The chat template applied during anonymization must match the target inference server's template, since template tokens are part of what gets cached in the KV cache.
- Token counts (ISL/OSL) should reflect what the target model's tokenizer produces, since these determine benchmark load characteristics.
- A trace anonymized for Llama will have different token counts and block boundaries than one anonymized for Mistral, because their tokenizers differ.

Example: if you have Claude production logs and want to benchmark Llama 3.1 70B as a replacement, use `--model meta-llama/Llama-3.1-70B-Instruct`.

### Error Handling

- Non-zero exit on fatal errors (file not found, tokenizer load failure).
- Individual malformed lines (invalid JSON, missing `messages`, empty messages array) are skipped with a per-line warning.
- Summary includes count of skipped lines.

### Summary Output

After processing, print:
- Total requests processed
- Total requests skipped (with reason)
- Sessions detected (if multi-turn)
- Unique hash IDs generated
- No-timestamps warning (if applicable)
- Output file path

## Multi-Turn Handling

When `session_id` is present, lines sharing the same ID are treated as sequential turns in a conversation:

1. Group lines by `session_id`.
2. Sort turns within each session by `timestamp` (or input order if no timestamps).
3. For each turn, build the accumulated message history: all prior turns' messages (including prior assistant responses from the `output` field) plus the current turn's messages. This means turn N's input includes all user and assistant messages from turns 1 through N.
4. Apply chat template to the accumulated messages, tokenize, hash. After emitting the output record, append `{"role": "assistant", "content": turn.output}` to the accumulated history for the next turn.
5. Because conversation history accumulates, turn N's hash sequence naturally shares a prefix with turn N-1's — matching real KV cache behavior.

When `session_id` is absent, each line is independent. The `RollingHasher` instance is shared across all requests (both single-turn and multi-turn) so that prefix overlap between unrelated requests is also captured (e.g., shared system prompts).

## Code Structure

### New Files

| File | Purpose |
|------|---------|
| `src/aiperf/cli_commands/anonymize_trace.py` | Thin Cyclopts CLI wrapper, delegates to core logic |
| `src/aiperf/dataset/synthesis/anonymize.py` | Core logic: read, tokenize, hash, write |
| `tests/unit/dataset/synthesis/test_anonymize.py` | Unit tests for core logic |
| `docs/tutorials/anonymize-trace.md` | Tutorial: privacy-preserving trace sharing |

### Modified Files

| File | Change |
|------|--------|
| `src/aiperf/cli.py` | Register `anonymize-trace` command via lazy import string |
| `README.md` | Add anonymize-trace tutorial to tutorial index |
| `docs/api/synthesis.md` | Add anonymize-trace to synthesis API documentation |

### No New Dependencies

Uses existing: `RollingHasher`, AIPerf's `Tokenizer` wrapper, `orjson`, `rich` (for summary), `pydantic` (for input validation).

### Core Logic Flow (`anonymize.py`)

```
def anonymize_trace(input_file, output_file, model, block_size):
    tokenizer = load_tokenizer(model)
    hasher = RollingHasher(block_size=block_size)

    records = read and validate input JSONL
    sessions = group by session_id (or each line = own session)

    for each session (sorted by timestamp):
        accumulated_messages = []
        for each turn:
            accumulated_messages.extend(turn.messages)
            templated = tokenizer.apply_chat_template(accumulated_messages)
            input_ids = tokenizer.encode(templated)
            blocks = split input_ids into chunks of block_size
            hash_ids = hasher.hash_token_blocks(blocks)
            output_ids = tokenizer.encode(turn.output)

            emit {timestamp?, input_length, output_length, hash_ids, session_id?}

            # Add assistant response to history for next turn's prefix
            accumulated_messages.append({"role": "assistant", "content": turn.output})

    print summary
```

## Tutorial Outline (`docs/tutorials/anonymize-trace.md`)

1. **Introduction** — What trace anonymization is, why share traces, the privacy guarantee.
2. **Preparing your input** — Input JSONL format with single-turn and multi-turn examples.
3. **Choosing your target model** — Explains `--model` as target model. Walks through "migrating from Claude to self-hosted Llama" use case.
4. **Running the command** — Basic invocation, output file defaults, block-size tuning.
5. **Verifying the output** — Using `aiperf analyze-trace` to inspect the anonymized trace (ISL/OSL distributions, prefix hit rates).
6. **Replaying the trace** — Full `aiperf profile` command with `--custom-dataset-type mooncake_trace --fixed-schedule`.
7. **What gets shared vs. protected** — Summary table of shared vs. protected data.

## Testing

### Unit Tests

- Single-turn: N independent requests produce correct hash_ids and token counts.
- Multi-turn: turns within a session share prefix hash_ids; turn N extends turn N-1.
- Prefix overlap: two requests with identical system prompts share initial hash_ids.
- Missing timestamps: warning emitted, output omits timestamp field.
- Malformed lines: skipped with warning, valid lines still processed.
- Output format: each line validates as Mooncake trace schema.

### Verification

Output traces should be loadable by `aiperf profile --custom-dataset-type mooncake_trace` without errors (tested via integration or manual verification in the tutorial).
